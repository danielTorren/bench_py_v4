"""
BENCH v4 — vectorized Python ABM for household energy renovation.

Agent attributes are stored as numpy arrays; all random draws use
self._np_rng (numpy Generator).

This is a literal port of the NetLogo model `BENCH_ v04_ B-NLD.ESP.nlogox`,
including its known defects.  Do not "fix" behaviour here.

Tick procedure order (same as the NetLogo `go`):
    _recall_memory      pre-2016 renovation status (first tick only)
    _update_dwelling    probabilistic dw_age update (from 2025)
    _knowledge          awareness → guilt → knowledge score
    _motivation         personal/social norm gates
    _consideration      PBC gates (sticky)
    _utility            U1 probit score for investment
    _action             renovation decision
    _save_energy        gas savings                 (from 2017)
    _invest             investment cost tracking    (from 2017)
    _learn              social learning             (from 2017)
    _update_income      CGE income scaling          (every year)
    _update_energy      dw_elab improvement         (from 2017)
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
    LEARNING_RATE, LEARNING_CAP, SLOW_NEIGHBOR_MIN, PBC_NEIGHBOR_CAP_SLOW_FAST,
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

    The population size is fixed by the case study (NetLogo `create-turtles`):
    793 households for ES, 759 for NL.
    """

    def __init__(
        self,
        case_study: str = "NL",
        seed: int | None = None,
        learning: str = "Informative",
        memory: bool = True,
        investment: bool = True,
        data_dir: str | None = None,
    ):
        self.case_study   = case_study
        self.learning     = learning
        self.memory_on    = memory
        self.investment   = investment
        self.n_households = N_HOUSEHOLDS[case_study]

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
        self._create_arrays()
        self._place_on_grid()
        self._build_neighbor_index()

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

        if self.year >= 2017:
            self._save_energy()
            self._invest()
            self._learn()
        self._update_income()
        if self.year >= 2017:
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
    # Initialisation: NetLogo `setup` group distributions
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
        """Draw grid positions on the fixed NetLogo world (-44..44 on both axes).

        NetLogo uses `setxy random-xcor random-ycor`; a turtle's patch is then
        uniform over the 89 x 89 patches, which is what is drawn here.
        """
        half = GRID_HALF
        self._grid_min = -half
        self._grid_max =  half
        self._grid_x = self._np_rng.integers(-half, half + 1,
                                              self.n_households).astype(np.int32)
        self._grid_y = self._np_rng.integers(-half, half + 1,
                                              self.n_households).astype(np.int32)

    def _build_neighbor_index(self) -> None:
        """Precompute _nbr_idx[i] = array of agent indices on the 8 adjacent patches."""
        n    = self.n_households
        half = int(self._grid_max)
        W    = 2 * half + 1

        # Shift to [0, W-1] coordinates so positions map to a linear patch id
        gx  = self._grid_x + half   # int32, shape (n,)
        gy  = self._grid_y + half
        pid = gx * np.int32(W) + gy  # linear patch id, shape (n,)

        # Sort agents by patch — enables a single bulk binary-search call
        order = np.argsort(pid, kind='stable')
        spid  = pid[order]

        # Compute all 8 neighbour patch ids for every agent simultaneously
        DX = np.array([-1,-1,-1, 0, 0, 1, 1, 1], dtype=np.int32)
        DY = np.array([-1, 0, 1,-1, 1,-1, 0, 1], dtype=np.int32)
        # The NetLogo world wraps on both axes (view has wrappingAllowedX/Y="true"),
        # so `neighbors` always reports 8 patches, including at the edges.  Take the
        # coordinates modulo W to match; clipping instead would give perimeter
        # agents artificially few neighbours.
        nx    = (gx[:, None] + DX) % W   # (n, 8)
        ny    = (gy[:, None] + DY) % W
        npid  = (nx * W + ny).astype(np.int32)

        # Flatten to (source-agent, patch-id) pairs.  Wrapping means every id is
        # in range, so there is nothing to discard here.
        src  = np.repeat(np.arange(n, dtype=np.int32), 8)
        pids = npid.ravel()

        # Single bulk searchsorted: locate every queried patch in the sorted list
        lo = np.searchsorted(spid, pids, side='left')
        hi = np.searchsorted(spid, pids, side='right')

        # Drop (agent, patch) pairs where the patch has no residents
        has = lo < hi
        src, lo, hi = src[has], lo[has], hi[has]
        cnts = hi - lo  # agents per non-empty lookup

        total = int(cnts.sum())
        if total == 0:
            self._nbr_idx = [np.empty(0, dtype=np.int32)] * n
            return

        # Expand ranges [lo[k], hi[k]) into a flat index array — fully vectorised
        # cum[k] = start position in the output for range k
        cum      = np.r_[np.int64(0), np.cumsum(cnts.astype(np.int64))]
        within   = np.arange(total, dtype=np.int64) - np.repeat(cum[:-1], cnts)
        nbr_flat = order[np.repeat(lo, cnts) + within].astype(np.int32)
        src_flat = np.repeat(src, cnts)

        # Sort by source agent and split into per-agent arrays
        idx        = np.argsort(src_flat, kind='stable')
        src_sorted = src_flat[idx]
        nbr_sorted = nbr_flat[idx]

        unique_src, starts = np.unique(src_sorted, return_index=True)
        ends = np.empty_like(starts)
        ends[:-1] = starts[1:]
        ends[-1]  = total

        nbr_idx: list[np.ndarray] = [np.empty(0, dtype=np.int32)] * n
        for k in range(len(unique_src)):
            nbr_idx[unique_src[k]] = nbr_sorted[starts[k]:ends[k]]
        self._nbr_idx = nbr_idx

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
        # NetLogo `action` has two sequential ifs.  The first tests
        # h.sta = "insulated", but the second does not and overrides it, so
        # the "insulated" status has no effect on the decision.
        eligible       = (self._U1 > 0) & ~self._invest1 & (self._dw_elab > 1)
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
        Social learning, ported line by line from the NetLogo `learn` procedure.

        NetLogo runs `ask turtles [...]` in a random order, and each active
        agent changes its neighbours at once.  So an agent later in the order
        reads values that earlier agents already changed in this tick.  This
        loop keeps that sequential behaviour.

        Neighbours: NetLogo does `create-links-to other turtles-on neighbors`
        and then uses `link-neighbors`.  Links are never removed, but agents
        never move and patch adjacency is symmetric, so `link-neighbors` of an
        active agent is always the same set as `turtles-on neighbors`.  The
        precomputed patch neighbourhood (`_nbr_idx`, torus) is therefore used.

        Gates are ported as written: growth is `x + x * 0.05` with no upper
        clamp, so values can pass 6.6.  The `pbcI1` neighbour gate is 6.5 in
        "Slow dynamics" and "Fast dynamics" and 6.6 in "Informative".
        """
        if self.learning not in ("Slow dynamics", "Fast dynamics", "Informative"):
            return

        lc = LEARNING_CAP
        lr = LEARNING_RATE
        informative = self.learning == "Informative"
        slow        = self.learning == "Slow dynamics"
        pbc_gate    = lc if informative else PBC_NEIGHBOR_CAP_SLOW_FAST

        know, cee, ed = self._know, self._cee_aw, self._ed_aw
        pn, sn, pbcI  = self._pn, self._sn, self._pbcI
        active        = self._act1 | self._invest1
        rng           = self._np_rng

        def stats(j):
            nn = self._nbr_idx[j]
            return (
                _max_mean_median_arr(know[nn]),
                _max_mean_median_arr(cee[nn]),
                _max_mean_median_arr(ed[nn]),
                _max_mean_median_arr(pn[nn, 0]),
                _max_mean_median_arr(sn[nn, 0]),
                _max_mean_median_arr(pbcI[nn, 0]),
            )

        def update(j, s):
            ngb_k, ngb_ca, ngb_ed, ngb_pn1, ngb_sn1, ngb_pbcI1 = s
            if know[j] < ngb_k and know[j] < lc:
                know[j] = know[j] + know[j] * lr
            if cee[j] < ngb_ca and cee[j] < lc:
                cee[j] = cee[j] + cee[j] * lr
            if ed[j] < ngb_ed and ed[j] < lc:
                ed[j] = ed[j] + ed[j] * lr
            if pn[j, 0] < ngb_pn1 and pn[j, 0] < lc:
                pn[j, 0] = pn[j, 0] + pn[j, 0] * lr
            if sn[j, 0] < ngb_sn1 and sn[j, 0] < lc:
                sn[j, 0] = sn[j, 0] + lr * sn[j, 0]
            if pbcI[j, 0] < pbc_gate and pbcI[j, 0] < ngb_pbcI1:
                pbcI[j, 0] = pbcI[j, 0] + pbcI[j, 0] * lr

        for i in rng.permutation(self.n_households):
            if informative:
                if know[i] <= lc:
                    know[i] = know[i] + know[i] * lr
                if cee[i] <= lc:
                    cee[i] = cee[i] + cee[i] * lr
                if ed[i] <= lc:
                    ed[i] = ed[i] + ed[i] * lr

            if not active[i]:
                continue

            if pbcI[i, 0] < lc:
                pbcI[i, 0] = pbcI[i, 0] + pbcI[i, 0] * lr

            nbrs = self._nbr_idx[i]
            if len(nbrs) == 0:
                continue
            nbrs = rng.permutation(nbrs)

            if slow:
                # All neighbour statistics are set first, then applied only
                # if the agent has more than four link-neighbours.
                nbr_stats = [(j, stats(j)) for j in nbrs]
                if len(nbrs) > SLOW_NEIGHBOR_MIN:
                    for j, s in nbr_stats:
                        update(j, s)
            else:
                for j in nbrs:
                    update(j, stats(j))

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

    # --- Paper-style multi-year aggregation (Niamir et al. 2024, Figs. 5-7) ---

    def renovation_rate_5yr_by_vintage(self, **kwargs):
        """
        Renovation rate per vintage cohort aggregated over 5-year windows.

        Returns ``(end_years, {1: rates, 2: rates, 3: rates})``.  See
        ``bench_v4.aggregate`` for the definition and the keyword arguments
        (``end_years``, ``window``, ``denominator``, ``min_year``).

        ``min_year`` defaults to ``aggregate.REPORT_MIN_YEAR`` (None), which
        keeps every year 2016-2050; pass ``min_year=2017`` to drop 2016.
        """
        from .aggregate import REPORT_MIN_YEAR, multi_year_rate

        kwargs.setdefault("min_year", REPORT_MIN_YEAR)
        years = self.years()
        out = {}
        end_years: list[int] = []
        for cat in (1, 2, 3):
            end_years, rates, _ = multi_year_rate(
                years,
                [s.renov_by_dwage.get(cat, 0) for s in self.history],
                [s.total_by_dwage.get(cat, 0) for s in self.history],
                **kwargs,
            )
            out[cat] = rates
        return end_years, out

    def renovation_rate_5yr_by_income(self, **kwargs):
        """
        Renovation rate per income group aggregated over 5-year windows.

        ``min_year`` defaults to ``aggregate.REPORT_MIN_YEAR`` (None), which
        keeps every year 2016-2050; pass ``min_year=2017`` to drop 2016.
        """
        from .aggregate import REPORT_MIN_YEAR, multi_year_rate

        kwargs.setdefault("min_year", REPORT_MIN_YEAR)
        years = self.years()
        out = {}
        end_years: list[int] = []
        for g in range(1, 6):
            end_years, rates, _ = multi_year_rate(
                years,
                [s.renov_by_group.get(g, 0) for s in self.history],
                [s.total_by_group.get(g, 0) for s in self.history],
                **kwargs,
            )
            out[g] = rates
        return end_years, out

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
