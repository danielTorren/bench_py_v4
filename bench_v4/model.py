"""
BENCH v4 — vectorized Python ABM for household energy renovation.

Agent attributes are stored as numpy arrays; all random draws use
self._np_rng (numpy Generator).

Tick procedure order:
    _recall_memory      pre-2016 renovation status (first tick only)
    _update_dwelling    probabilistic dw_age update (from 2025)
    _knowledge          awareness → guilt → knowledge score
    _motivation         personal/social norm gates
    _consideration      PBC gates (sticky)
    _utility            U1 probit score for investment
    _action             renovation decision
    _update_income      CGE income scaling          (every year)
    _save_energy        gas savings                 (from 2017)
    _invest             investment cost tracking    (from 2017)
    _learn              social learning             (from 2017)
    _update_energy      dw_elab degradation        (from 2017)
    _update_memory      cooldown / invest1 reset
"""

import csv
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .params import (
    ES_GROUPS, NL_GROUPS, N_HOUSEHOLDS,
    CGE_FILES,
    GUILT_THRESH, MOTIVATION_THRESH,
    PBC_INVEST_THRESH, PBC_CONSERV_THRESH, PBC_SWITCH_THRESH,
    UTILITY_COEF, I1_COST, GAS_SAVE_FRACTION,
    COOLDOWN_BY_DWAGE,
    LEARNING_RATE, LEARNING_CAP, SLOW_NEIGHBOR_MIN,
    RECALL_PROB, DWAGE_UPDATE,
    START_YEAR, END_YEAR,
    GRID_HALF,
)

# Numpy LUT derived from COOLDOWN_BY_DWAGE; index 0 unused (dw_age is 1-3)
_COOLDOWN_LUT = np.array(
    [0] + [COOLDOWN_BY_DWAGE[k] for k in sorted(COOLDOWN_BY_DWAGE)],
    dtype=np.int32,
)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _load_cge(filepath: str) -> list[float]:
    values = []
    with open(filepath, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if row and row[0].strip():
                values.append(float(row[0].strip()))
    return values


def _max_mean_median_arr(arr: np.ndarray) -> float:
    """max(mean, median) — mirrors NetLogo: max list mean median.

    Pure-Python path for tiny arrays (0–8 elements) from patch-neighbour
    lookups.  Avoids the ~15 µs per-call overhead of np.median on small inputs.
    """
    n = len(arr)
    if n == 0:
        return 0.0
    vals = arr.tolist()
    mean = sum(vals) / n
    if n == 1:
        return mean
    vals.sort()
    mid = n >> 1
    median = vals[mid] if n & 1 else (vals[mid - 1] + vals[mid]) * 0.5
    return mean if mean >= median else median


def _rand_num_vec(rng, lo: float, hi: float, step: float, size: int) -> np.ndarray:
    """Vectorised equivalent of household._rand_num for `size` draws."""
    n_steps = math.floor((hi - lo) / step)
    return lo + step * rng.integers(0, n_steps + 1, size=size).astype(np.float64)


def _categorical_vec(rng, breaks: list, size: int) -> np.ndarray:
    """Vectorised equivalent of household._categorical."""
    uppers = [b[0] for b in breaks]
    values = np.array([b[1] for b in breaks], dtype=np.int8)
    rn = rng.uniform(0, 100, size)
    idx = np.searchsorted(uppers, rn, side='right')
    return values[np.clip(idx, 0, len(breaks) - 1)]


def _rand_er_vec(rng, spec, size: int) -> np.ndarray:
    """Vectorised equivalent of household._rand_er (None → fixed -0.01)."""
    if spec is None:
        return np.full(size, -0.01, dtype=np.float64)
    return _rand_num_vec(rng, spec[0], spec[1], 0.01, size)


# ---------------------------------------------------------------------------
# Results container
# ---------------------------------------------------------------------------

@dataclass
class AnnualStats:
    year: int
    n_renovated: int
    n_conservation: int
    n_switching: int
    n_invested: int
    total_gas_saved: float
    total_energy_conservation: float
    total_energy_switching: float
    total_investment: float
    total_invest_conservation: float
    total_invest_switching: float
    avg_aware: float
    avg_pn1: float
    avg_sn1: float
    high_guilt_pct: float
    high_m1_pct: float
    high_m2_pct: float
    high_m3_pct: float
    renov_by_dwage: dict[int, int] = field(default_factory=dict)
    total_by_dwage: dict[int, int] = field(default_factory=dict)
    renov_by_group: dict[int, int] = field(default_factory=dict)
    total_by_group: dict[int, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Main model class
# ---------------------------------------------------------------------------

class BENCHv4:
    """
    Vectorized BENCH v4 (renovation/insulation module).

    Parameters
    ----------
    case_study   : "ES" | "NL"
    seed         : random seed (None → 1)
    learning     : "Slow dynamics" | "Fast dynamics" | "Informative" | "No learning"
    memory       : apply recall of pre-2016 renovations in first tick
    investment   : whether Investment behaviour is enabled
    data_dir     : path to CGE CSV files
    n_households : synthetic population size (None → survey default)
    """

    def __init__(
        self,
        case_study: str = "NL",
        seed: int | None = None,
        learning: str = "Informative",
        memory: bool = True,
        investment: bool = True,
        data_dir: str | None = None,
        n_households: int | None = None,
    ):
        self.case_study   = case_study
        self.learning     = learning
        self.memory_on    = memory
        self.investment   = investment
        self.n_households = n_households or N_HOUSEHOLDS[case_study]

        if seed is None:
            seed = 1
        self.seed = seed
        self._np_rng = np.random.default_rng(seed)

        if data_dir is None:
            here = Path(__file__).parent.parent
            data_dir = str(here / "data")
        self.data_dir = data_dir

        self.year: int = START_YEAR
        self.n:    int = 0

        self.cge: list[float] = []
        self.history: list[AnnualStats] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def setup(self) -> None:
        self._load_data()
        self._create_arrays()        # Option A: numpy batch initialization
        self._place_on_grid()        # numpy grid placement
        self._build_neighbor_index() # precompute spatial index

    def go(self) -> bool:
        if self.year > END_YEAR:
            return False

        self._act1[:] = False
        self._recall_memory()
        self._update_dwelling()

        self._knowledge()
        self._motivation()
        self._consideration()
        self._utility()
        self._action()

        self._update_income()
        if self.year >= 2017:
            self._save_energy()
            self._invest()
            self._learn()
            self._update_energy()

        self._update_memory()
        self.history.append(self._collect_stats())
        self.year += 1
        self.n    += 1
        return True

    def run(self, verbose: bool = False) -> list[AnnualStats]:
        self.setup()
        while self.go():
            if verbose:
                last = self.history[-1]
                pct = 100 * last.n_renovated / self.n_households
                print(
                    f"  year={last.year}  renovated={last.n_renovated} "
                    f"({pct:.1f}%)  gas_saved={last.total_gas_saved:.0f}"
                )
        return self.history

    # ------------------------------------------------------------------
    # Initialisation — Option A: direct numpy batch draws
    # ------------------------------------------------------------------

    def _load_data(self) -> None:
        fname = CGE_FILES[self.case_study]
        fpath = os.path.join(self.data_dir, fname)
        self.cge = _load_cge(fpath)

    def _create_arrays(self) -> None:
        """Create all agent attribute arrays using numpy batch draws."""
        N   = self.n_households
        rng = self._np_rng
        groups_list = ES_GROUPS if self.case_study == "ES" else NL_GROUPS

        # --- Group assignment ---
        rn_group   = rng.uniform(0, 100, N)
        cum_uppers = [gp.cum_upper for gp in groups_list]
        gi_arr     = np.clip(
            np.searchsorted(cum_uppers, rn_group, side='right'),
            0, len(groups_list) - 1,
        )
        self._h_group = np.array([gp.group_id for gp in groups_list], dtype=np.int8)[gi_arr]

        # --- Pre-allocate attribute arrays ---
        self._income  = np.empty(N, dtype=np.float64)
        self._gas     = np.empty(N, dtype=np.float64)
        self._edu     = np.empty(N, dtype=np.int8)
        self._age     = np.empty(N, dtype=np.int8)
        self._dw_st   = np.empty(N, dtype=np.int8)
        self._dw_elab = np.empty(N, dtype=np.int8)
        self._dw_type = np.empty(N, dtype=np.int8)
        self._dw_age  = np.empty(N, dtype=np.int8)
        self._dw_size = np.empty(N, dtype=np.int8)
        self._know    = np.empty(N, dtype=np.float64)
        self._cee_aw  = np.empty(N, dtype=np.float64)
        self._ed_aw   = np.empty(N, dtype=np.float64)
        self._pn      = np.empty((N, 3), dtype=np.float64)
        self._sn      = np.empty((N, 3), dtype=np.float64)
        self._pbcI    = np.empty((N, 3), dtype=np.float64)
        self._pbcC    = np.empty((N, 3), dtype=np.float64)
        self._pbcS    = np.empty((N, 3), dtype=np.float64)
        self._ene_pat = np.empty((N, 3), dtype=np.float64)
        self._erI     = np.empty((N, 3), dtype=np.float64)

        # --- Fill each group in one batch ---
        for gi, gp in enumerate(groups_list):
            mask = gi_arr == gi
            n_g  = int(np.sum(mask))
            if n_g == 0:
                continue

            lo, hi, st = gp.income_range
            self._income[mask] = _rand_num_vec(rng, lo, hi, st, n_g)

            lo, hi, st = gp.gas_range
            self._gas[mask] = _rand_num_vec(rng, lo, hi, st, n_g)

            lo, hi = gp.know_range
            self._know[mask] = _rand_num_vec(rng, lo, hi, 0.05, n_g)

            lo, hi = gp.cee_aw_range
            self._cee_aw[mask] = _rand_num_vec(rng, lo, hi, 0.05, n_g)

            lo, hi = gp.ed_aw_range
            self._ed_aw[mask] = _rand_num_vec(rng, lo, hi, 0.05, n_g)

            pn_lo, pn_hi = gp.pn_range
            for j in range(3):
                self._pn[mask, j] = _rand_num_vec(rng, pn_lo, pn_hi, 0.05, n_g)

            sn_lo, sn_hi = gp.sn_range
            for j in range(3):
                self._sn[mask, j] = _rand_num_vec(rng, sn_lo, sn_hi, 0.05, n_g)

            pbcI_lo, pbcI_hi = gp.pbcI_range
            for j in range(3):
                self._pbcI[mask, j] = _rand_num_vec(rng, pbcI_lo, pbcI_hi, 0.05, n_g)

            pbcC_lo, pbcC_hi = gp.pbcC_range
            for j in range(3):
                self._pbcC[mask, j] = _rand_num_vec(rng, pbcC_lo, pbcC_hi, 0.05, n_g)

            pbcS_lo, pbcS_hi = gp.pbcS_range
            for j in range(3):
                self._pbcS[mask, j] = _rand_num_vec(rng, pbcS_lo, pbcS_hi, 0.05, n_g)

            ep_lo, ep_hi = gp.ene_pat_range
            for j in range(3):
                self._ene_pat[mask, j] = _rand_num_vec(rng, ep_lo, ep_hi, 0.05, n_g)

            for j in range(3):
                self._erI[mask, j] = _rand_er_vec(rng, gp.erI_ranges[j], n_g)

            self._edu[mask]     = _categorical_vec(rng, gp.edu_breaks,    n_g)
            self._age[mask]     = _categorical_vec(rng, gp.age_breaks,    n_g)
            self._dw_age[mask]  = _categorical_vec(rng, gp.dwage_breaks,  n_g)
            self._dw_size[mask] = _categorical_vec(rng, gp.dwsize_breaks, n_g)
            self._dw_elab[mask] = _categorical_vec(rng, gp.elab_breaks,   n_g)

            self._dw_st[mask]   = np.where(
                rng.uniform(0, 100, n_g) < gp.owner_thresh, np.int8(1), np.int8(2))
            self._dw_type[mask] = np.where(
                rng.uniform(0, 100, n_g) < gp.dtype_thresh, np.int8(1), np.int8(2))

        # --- Simulation state (all zero / False at start) ---
        self._aware  = np.zeros(N, dtype=np.float64)
        self._k      = np.zeros(N, dtype=np.float64)
        self._U1     = np.zeros(N, dtype=np.float64)

        self._guilt  = np.zeros(N,      dtype=bool)
        self._m_st   = np.zeros((N, 3), dtype=bool)
        self._cI_st  = np.zeros((N, 3), dtype=bool)
        self._cC_st  = np.zeros((N, 3), dtype=bool)
        self._cS_st  = np.zeros((N, 3), dtype=bool)

        self._act1      = np.zeros(N, dtype=bool)
        self._invest1   = np.zeros(N, dtype=bool)
        self._act1_year = np.zeros(N, dtype=np.int32)
        self._insulated = np.zeros(N, dtype=bool)

        self._save_a0 = np.zeros(N, dtype=np.float64)
        self._invs_a0 = np.zeros(N, dtype=np.float64)

    def _place_on_grid(self) -> None:
        """Draw grid positions; scale preserves ~0.1 agents/cell density."""
        scale = math.sqrt(self.n_households / N_HOUSEHOLDS[self.case_study])
        half  = max(1, round(GRID_HALF * scale))
        self._grid_min = -half
        self._grid_max =  half
        self._grid_x = self._np_rng.integers(-half, half + 1,
                                              self.n_households).astype(np.int32)
        self._grid_y = self._np_rng.integers(-half, half + 1,
                                              self.n_households).astype(np.int32)

    def _build_neighbor_index(self) -> None:
        """Precompute _nbr_idx[i] = array of agent indices on the 8 adjacent patches."""
        grid_dict: dict[tuple[int, int], list[int]] = {}
        for i in range(self.n_households):
            key = (int(self._grid_x[i]), int(self._grid_y[i]))
            if key not in grid_dict:
                grid_dict[key] = []
            grid_dict[key].append(i)

        self._nbr_idx: list[np.ndarray] = []
        for i in range(self.n_households):
            gx, gy = int(self._grid_x[i]), int(self._grid_y[i])
            nbrs: list[int] = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nbrs.extend(grid_dict.get((gx + dx, gy + dy), []))
            self._nbr_idx.append(np.array(nbrs, dtype=np.int32))

    # ------------------------------------------------------------------
    # go() sub-routines
    # ------------------------------------------------------------------

    def _recall_memory(self) -> None:
        """Assign pre-2016 renovation status (vectorised via numpy RNG)."""
        if self.year != START_YEAR or not self.memory_on:
            return
        recall_probs = RECALL_PROB[self.case_study]
        g_keys = np.where(self._h_group <= 4, self._h_group, np.int8(5))
        probs  = np.array([recall_probs.get(int(g), 0.0) for g in g_keys],
                          dtype=np.float64)
        recalled = self._np_rng.uniform(0, 100, self.n_households) <= probs
        self._act1[recalled]      = True
        self._invest1[recalled]   = True
        self._insulated[recalled] = True

    def _update_dwelling(self) -> None:
        """Probabilistically update dw_age from 2025 onwards."""
        dw_table = DWAGE_UPDATE.get(self.case_study, {})
        for (yr_lo, yr_hi), (p_new, p_mid) in dw_table.items():
            if yr_lo <= self.year < yr_hi:
                dag = self._np_rng.uniform(0, 100, self.n_households)
                self._dw_age[:] = np.where(dag < p_new, 1,
                                  np.where(dag < p_mid,  2, 3))
                break

    def _knowledge(self) -> None:
        thresh = GUILT_THRESH[self.case_study]
        self._aware = (self._know + self._cee_aw + self._ed_aw) / 3.0
        self._guilt = self._aware >= thresh
        self._k     = np.where(self._guilt, self._aware / 7.0, 0.0)

    def _motivation(self) -> None:
        """Update m_st only for guilty households (non-guilty retain prior value)."""
        thr = MOTIVATION_THRESH[self.case_study]
        pn1_thr, sn1_thr = thr["m1"]
        pn2_thr, sn2_thr = thr["m2"]
        pn3_thr, sn3_thr = thr["m3"]
        g = self._guilt
        self._m_st[g, 0] = (self._pn[g, 0] >= pn1_thr) & (self._sn[g, 0] >= sn1_thr)
        self._m_st[g, 1] = (self._pn[g, 1] >= pn2_thr) & (self._sn[g, 1] >= sn2_thr)
        self._m_st[g, 2] = (self._pn[g, 2] >= pn3_thr) & (self._sn[g, 2] >= sn3_thr)

    def _consideration(self) -> None:
        """Update cX_st only where motivation is H (sticky otherwise)."""
        pbc_inv = PBC_INVEST_THRESH[self.case_study]
        pbc_sw  = PBC_SWITCH_THRESH[self.case_study]
        owner   = self._dw_st == 1

        if self.investment:
            m = self._m_st[:, 0]
            for j in range(3):
                self._cI_st[m, j] = (self._pbcI[m, j] >= pbc_inv) & owner[m]

        m = self._m_st[:, 1]
        for j in range(3):
            self._cC_st[m, j] = (
                (self._pbcC[m, j] >= PBC_CONSERV_THRESH) &
                (self._ene_pat[m, j] != 3)
            )

        m = self._m_st[:, 2]
        for j in range(3):
            self._cS_st[m, j] = self._pbcS[m, j] >= pbc_sw

    def _utility(self) -> None:
        c = UTILITY_COEF
        self._U1[:] = 0.0
        mask = self._cI_st[:, 0]
        self._U1[mask] = (
              self._edu[mask].astype(np.float64)     * c["edu"]
            + self._age[mask].astype(np.float64)     * c["age"]
            + self._dw_elab[mask].astype(np.float64) * c["dw_elab"]
            + self._dw_type[mask].astype(np.float64) * c["dw_type"]
            + self._dw_age[mask].astype(np.float64)  * c["dw_age"]
            + self._dw_size[mask].astype(np.float64) * c["dw_size"]
            + self._gas[mask]    * c["gas"]
            + self._pn[mask, 0]  * c["pn1"]
            + self._erI[mask, 0]
        )

    def _action(self) -> None:
        eligible       = (self._U1 > 0) & ~self._invest1 & ~self._insulated & (self._dw_elab > 1)
        self._act1     = eligible
        self._invest1 |= eligible

    def _save_energy(self) -> None:
        self._save_a0[:] = 0.0
        m = self._act1
        self._save_a0[m] = self._gas[m] * GAS_SAVE_FRACTION
        self._gas[m]    -= self._save_a0[m]

    def _invest(self) -> None:
        self._invs_a0[:] = 0.0
        self._invs_a0[self._act1] = I1_COST

    def _learn(self) -> None:
        """
        Social learning.  Option B: collect unique candidate neighbours once,
        compute each neighbour's stats a single time (avoids redundant recomputation
        when j borders multiple active agents), then apply updates.
        """
        if self.learning == "No learning":
            return

        lc = LEARNING_CAP
        lr = LEARNING_RATE

        # Informative broadcast to all households
        if self.learning == "Informative":
            for arr in (self._know, self._cee_aw, self._ed_aw):
                m = arr <= lc
                arr[m] = np.minimum(arr[m] * (1.0 + lr), lc + lr)

        active_idx = np.where(self._act1 | self._invest1)[0]
        if len(active_idx) == 0:
            return

        slow = (self.learning == "Slow dynamics")

        # --- Option B: collect unique candidates, compute stats once each ---
        cand: set = set()
        for i in active_idx:
            for jv in self._nbr_idx[i]:
                cand.add(int(jv))

        nbr_know:  dict[int, float] = {}
        nbr_cee:   dict[int, float] = {}
        nbr_ed:    dict[int, float] = {}
        nbr_pn1:   dict[int, float] = {}
        nbr_sn1:   dict[int, float] = {}
        nbr_pbcI1: dict[int, float] = {}
        for j in cand:
            nn = self._nbr_idx[j]
            if len(nn) == 0:
                continue
            nbr_know[j]   = _max_mean_median_arr(self._know[nn])
            nbr_cee[j]    = _max_mean_median_arr(self._cee_aw[nn])
            nbr_ed[j]     = _max_mean_median_arr(self._ed_aw[nn])
            nbr_pn1[j]    = _max_mean_median_arr(self._pn[nn, 0])
            nbr_sn1[j]    = _max_mean_median_arr(self._sn[nn, 0])
            nbr_pbcI1[j]  = _max_mean_median_arr(self._pbcI[nn, 0])

        # --- Apply social learning from each active agent ---
        for i in active_idx:
            if self._pbcI[i, 0] < lc:
                self._pbcI[i, 0] = min(float(self._pbcI[i, 0]) * (1.0 + lr), lc)

            nbrs = self._nbr_idx[i]
            if len(nbrs) == 0:
                continue
            if slow and len(nbrs) <= SLOW_NEIGHBOR_MIN:
                continue

            for j in nbrs:
                if j not in nbr_know:
                    continue
                if self._know[j]   < nbr_know[j]   and self._know[j]   < lc:
                    self._know[j]   = min(float(self._know[j])   * (1.0 + lr), lc)
                if self._cee_aw[j] < nbr_cee[j]    and self._cee_aw[j] < lc:
                    self._cee_aw[j] = min(float(self._cee_aw[j]) * (1.0 + lr), lc)
                if self._ed_aw[j]  < nbr_ed[j]     and self._ed_aw[j]  < lc:
                    self._ed_aw[j]  = min(float(self._ed_aw[j])  * (1.0 + lr), lc)
                if self._pn[j, 0]  < nbr_pn1[j]    and self._pn[j, 0]  < lc:
                    self._pn[j, 0]  = min(float(self._pn[j, 0])  * (1.0 + lr), lc)
                if self._sn[j, 0]  < nbr_sn1[j]    and self._sn[j, 0]  < lc:
                    self._sn[j, 0]  = min(float(self._sn[j, 0])  * (1.0 + lr), lc)
                if self._pbcI[j,0] < nbr_pbcI1[j]  and self._pbcI[j,0] < lc:
                    self._pbcI[j,0] = min(float(self._pbcI[j,0]) * (1.0 + lr), lc)

    def _update_income(self) -> None:
        if self.n >= len(self.cge):
            return
        self._income *= self.cge[self.n]

    def _update_energy(self) -> None:
        m = self._act1
        self._dw_elab[m & (self._dw_elab >= 2)] -= 1

    def _update_memory(self) -> None:
        self._act1_year[self._invest1] += 1
        cooldowns = _COOLDOWN_LUT[self._dw_age]
        expired = self._act1_year >= cooldowns
        self._invest1[expired]   = False
        self._act1_year[expired] = 0

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def _collect_stats(self) -> AnnualStats:
        N = self.n_households

        renov_by_dwage: dict[int, int] = {}
        total_by_dwage: dict[int, int] = {}
        for cat in (1, 2, 3):
            age_mask = self._dw_age == cat
            total_by_dwage[cat] = int(np.sum(age_mask))
            renov_by_dwage[cat] = int(np.sum(self._act1 & age_mask))

        g_arr = np.where(self._h_group <= 4, self._h_group, 5)
        renov_by_group: dict[int, int] = {}
        total_by_group: dict[int, int] = {}
        for g in range(1, 6):
            grp_mask = g_arr == g
            total_by_group[g] = int(np.sum(grp_mask))
            renov_by_group[g] = int(np.sum(self._act1 & grp_mask))

        return AnnualStats(
            year                      = self.year,
            n_renovated               = int(np.sum(self._act1)),
            n_conservation            = 0,
            n_switching               = 0,
            n_invested                = int(np.sum(self._invest1)),
            total_gas_saved           = float(np.sum(self._save_a0)),
            total_energy_conservation = 0.0,
            total_energy_switching    = 0.0,
            total_investment          = float(np.sum(self._invs_a0)),
            total_invest_conservation = 0.0,
            total_invest_switching    = 0.0,
            avg_aware  = float(np.mean(self._aware)),
            avg_pn1    = float(np.mean(self._pn[:, 0])),
            avg_sn1    = float(np.mean(self._sn[:, 0])),
            high_guilt_pct = 100.0 * float(np.sum(self._guilt))      / N,
            high_m1_pct    = 100.0 * float(np.sum(self._m_st[:, 0])) / N,
            high_m2_pct    = 100.0 * float(np.sum(self._m_st[:, 1])) / N,
            high_m3_pct    = 100.0 * float(np.sum(self._m_st[:, 2])) / N,
            renov_by_dwage = renov_by_dwage,
            total_by_dwage = total_by_dwage,
            renov_by_group = renov_by_group,
            total_by_group = total_by_group,
        )

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    def renovation_rate_by_vintage(self) -> dict[int, list[float]]:
        result = {1: [], 2: [], 3: []}
        for s in self.history:
            for cat in (1, 2, 3):
                tot = s.total_by_dwage.get(cat, 0)
                ren = s.renov_by_dwage.get(cat, 0)
                result[cat].append(100.0 * ren / tot if tot > 0 else 0.0)
        return result

    def renovation_rate_by_income(self) -> dict[int, list[float]]:
        result = {g: [] for g in range(1, 6)}
        for s in self.history:
            for g in range(1, 6):
                tot = s.total_by_group.get(g, 0)
                ren = s.renov_by_group.get(g, 0)
                result[g].append(100.0 * ren / tot if tot > 0 else 0.0)
        return result

    def years(self) -> list[int]:
        return [s.year for s in self.history]

    def summary(self) -> str:
        lines = [
            f"BENCH v4 — {self.case_study}  Learning={self.learning}  seed={self.seed}",
            f"Years {START_YEAR}–{END_YEAR}  |  N={self.n_households} households",
            "",
            f"{'Year':>6}  {'Renov':>7}  {'%Renov':>7}  {'GasSaved(kWh)':>14}  {'Invest(EUR)':>12}",
        ]
        for s in self.history:
            pct = 100 * s.n_renovated / self.n_households
            lines.append(
                f"{s.year:>6}  {s.n_renovated:>7}  {pct:>7.2f}  "
                f"{s.total_gas_saved:>14,.0f}  {s.total_investment:>12,.0f}"
            )
        return "\n".join(lines)
