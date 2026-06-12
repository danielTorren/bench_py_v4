"""
BENCH v4 — Python translation of BENCH_v04_B-NLD.ESP.nlogox.
Vectorized: agent attributes stored as numpy arrays; per-agent loops replaced
with array operations.  Random draws that affect seed-reproducibility still
use Python's random module (same call sequence as the original model).

go() procedure order (unchanged from NetLogo):
    update.info     → _update_info
    recallmemory    → _recall_memory
    update.dwelling → _update_dwelling
    knowledge       → _knowledge
    motivation      → _motivation
    consideration   → _consideration
    utility         → _utility
    action          → _action
    save.energy     → _save_energy   (year >= 2017)
    invest          → _invest        (year >= 2017)
    learn           → _learn         (year >= 2017)
    update.income   → _update_income
    update.energy   → _update_energy (year >= 2017)
    update.memory   → _update_memory
"""

import csv
import math
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .household import Household
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
    GRID_MIN, GRID_MAX,
)

# Cooldown lookup indexed by dw_age value (1→15, 2→7, 3→2; index 0 unused)
_COOLDOWN_LUT = np.array([0, 15, 7, 2], dtype=np.int32)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_cge(filepath: str) -> List[float]:
    values = []
    with open(filepath, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if row and row[0].strip():
                values.append(float(row[0].strip()))
    return values


def _max_mean_median_arr(arr: np.ndarray) -> float:
    """max(mean, median) — mirrors NetLogo: max list mean median.

    Pure-Python path for the tiny arrays (0–8 elements) produced by patch
    neighbour lookups.  Avoids the ~15 µs per-call overhead of np.median on
    small inputs, which dominates runtime when called ~160k times per run.
    """
    n = len(arr)
    if n == 0:
        return 0.0
    vals = arr.tolist()          # one fast C-level copy out of numpy
    mean = sum(vals) / n
    if n == 1:
        return mean              # mean == median for a single element
    vals.sort()
    mid = n >> 1
    median = vals[mid] if n & 1 else (vals[mid - 1] + vals[mid]) * 0.5
    return mean if mean >= median else median


# ---------------------------------------------------------------------------
# Results container
# ---------------------------------------------------------------------------

@dataclass
class AnnualStats:
    year: int
    n_renovated: int          # count(act1=True) that year
    n_conservation: int       # count(act2=True) — always 0 in v4
    n_switching: int          # count(act3=True) — always 0 in v4
    n_invested:  int          # count(invest1=True) — cumulative in-process
    total_gas_saved: float    # kWh saved by renovation
    total_energy_conservation: float  # kWh saved by conservation — 0 in v4
    total_energy_switching: float     # kWh saved by switching    — 0 in v4
    total_investment: float           # EUR spent on renovation
    total_invest_conservation: float  # EUR spent on conservation — 0 in v4
    total_invest_switching: float     # EUR spent on switching    — 0 in v4
    avg_aware: float
    avg_pn1: float
    avg_sn1: float
    high_guilt_pct: float     # % households with guilt=="H"
    high_m1_pct: float        # % motivated for investment
    high_m2_pct: float        # % motivated for conservation
    high_m3_pct: float        # % motivated for switching
    # by dw_age category
    renov_by_dwage: Dict[int, int] = field(default_factory=dict)
    total_by_dwage: Dict[int, int] = field(default_factory=dict)
    # by income group (1-5, where 5 covers groups 5-7)
    renov_by_group: Dict[int, int] = field(default_factory=dict)
    total_by_group: Dict[int, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Main model class
# ---------------------------------------------------------------------------

class BENCHv4:
    """
    Vectorized BENCH v4 (renovation/insulation module).

    Parameters
    ----------
    case_study : "ES" (Spain-Navarre) | "NL" (Netherlands-Overijssel)
    seed       : random seed; None → 1
    learning   : "Slow dynamics" | "Fast dynamics" | "Informative" | "No learning"
    memory     : True/False — apply recall of pre-2016 renovations in first tick
    investment : True/False — whether Investment behaviour is enabled
    data_dir   : path to folder containing the CGE CSV files
    n_households : synthetic population size (None → survey default)
    """

    def __init__(
        self,
        case_study: str = "NL",
        seed: Optional[int] = None,
        learning: str = "Informative",
        memory: bool = True,
        investment: bool = True,
        data_dir: Optional[str] = None,
        n_households: Optional[int] = None,
    ):
        self.case_study  = case_study
        self.learning    = learning
        self.memory_on   = memory
        self.investment  = investment
        self.n_households = n_households or N_HOUSEHOLDS[case_study]

        if seed is None:
            seed = 1
        self.seed = seed
        random.seed(seed)
        # Separate numpy RNG for vectorised dwelling updates (keeps Python
        # random state clean for recall_memory reproducibility)
        self._np_rng = np.random.default_rng(seed)

        if data_dir is None:
            here = Path(__file__).parent.parent
            data_dir = str(here / "data")
        self.data_dir = data_dir

        self.year: int = START_YEAR
        self.n:    int = 0

        self.households: List[Household] = []  # populated during setup, then cleared
        self.cge: List[float] = []
        self.history: List[AnnualStats] = []
        self.a1_cum: int = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Initialise households, grid and load data (mirrors NetLogo setup)."""
        self._load_data()
        self._create_households()   # Household objects drawn via Python random
        self._place_on_grid()       # grid positions drawn via Python random
        self._init_arrays()         # extract to numpy, build spatial index, free objects

    def go(self) -> bool:
        """Execute one simulation tick (one calendar year).  Returns False when done."""
        if self.year > END_YEAR:
            return False

        self._update_info()
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

    def run(self, verbose: bool = False) -> List[AnnualStats]:
        """Run the full simulation from START_YEAR to END_YEAR."""
        self.setup()
        while self.go():
            if verbose:
                last = self.history[-1]
                pct_renov = 100 * last.n_renovated / self.n_households
                print(
                    f"  year={last.year}  renovated={last.n_renovated} "
                    f"({pct_renov:.1f}%)  gas_saved={last.total_gas_saved:.0f}"
                )
        return self.history

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _load_data(self) -> None:
        fname = CGE_FILES[self.case_study]
        fpath = os.path.join(self.data_dir, fname)
        self.cge = _load_cge(fpath)

    def _create_households(self) -> None:
        """Create n_households agents using empirical distributions."""
        groups = ES_GROUPS if self.case_study == "ES" else NL_GROUPS
        for hh_id in range(self.n_households):
            rn = random.uniform(0, 100)
            for cum_upper, group_id, params in groups:
                if rn < cum_upper:
                    self.households.append(
                        Household(hh_id, self.case_study, rn, params, group_id)
                    )
                    break
            else:
                _, group_id, params = groups[-1]
                self.households.append(
                    Household(hh_id, self.case_study, rn, params, group_id)
                )

    def _place_on_grid(self) -> None:
        """Place agents on a scaled grid that preserves the original agent density."""
        scale = math.sqrt(self.n_households / N_HOUSEHOLDS[self.case_study])
        half  = max(1, round((GRID_MAX - GRID_MIN) / 2 * scale))
        self._grid_min = -half
        self._grid_max =  half

        self._grid: Dict[Tuple[int, int], List[Household]] = {}
        for hh in self.households:
            hh.grid_x = random.randint(self._grid_min, self._grid_max)
            hh.grid_y = random.randint(self._grid_min, self._grid_max)
            key = (hh.grid_x, hh.grid_y)
            if key not in self._grid:
                self._grid[key] = []
            self._grid[key].append(hh)

    def _init_arrays(self) -> None:
        """Extract household attributes to numpy arrays, build spatial index, free objects."""
        hhs = self.households
        N   = self.n_households

        # --- Demographics ---
        self._h_group = np.array([h.h_group for h in hhs], dtype=np.int8)
        self._income  = np.array([h.income  for h in hhs], dtype=np.float64)
        self._gas     = np.array([h.gas     for h in hhs], dtype=np.float64)
        self._edu     = np.array([h.edu     for h in hhs], dtype=np.int8)
        self._age     = np.array([h.age     for h in hhs], dtype=np.int8)
        self._dw_st   = np.array([h.dw_st   for h in hhs], dtype=np.int8)
        self._dw_elab = np.array([h.dw_elab for h in hhs], dtype=np.int8)
        self._dw_type = np.array([h.dw_type for h in hhs], dtype=np.int8)
        self._dw_age  = np.array([h.dw_age  for h in hhs], dtype=np.int8)
        self._dw_size = np.array([h.dw_size for h in hhs], dtype=np.int8)

        # --- Awareness / knowledge ---
        self._know   = np.array([h.know   for h in hhs], dtype=np.float64)
        self._cee_aw = np.array([h.cee_aw for h in hhs], dtype=np.float64)
        self._ed_aw  = np.array([h.ed_aw  for h in hhs], dtype=np.float64)
        self._aware  = np.zeros(N, dtype=np.float64)
        self._k      = np.zeros(N, dtype=np.float64)

        # --- Behavioral norms / PBC (N, 3) ---
        self._pn    = np.array([[h.pn[j]    for j in range(3)] for h in hhs], dtype=np.float64)
        self._sn    = np.array([[h.sn[j]    for j in range(3)] for h in hhs], dtype=np.float64)
        self._pbcI  = np.array([[h.pbcI[j]  for j in range(3)] for h in hhs], dtype=np.float64)
        self._pbcC  = np.array([[h.pbcC[j]  for j in range(3)] for h in hhs], dtype=np.float64)
        self._pbcS  = np.array([[h.pbcS[j]  for j in range(3)] for h in hhs], dtype=np.float64)
        self._ene_pat = np.array([[h.ene_pat[j] for j in range(3)] for h in hhs], dtype=np.float64)
        self._erI   = np.array([[h.erI[j]   for j in range(3)] for h in hhs], dtype=np.float64)

        # --- Utility ---
        self._U1 = np.zeros(N, dtype=np.float64)

        # --- Status flags (bool; True ≡ "H") ---
        # Only updated when the preceding condition is met (sticky between ticks)
        self._guilt = np.zeros(N,      dtype=bool)
        self._m_st  = np.zeros((N, 3), dtype=bool)
        self._cI_st = np.zeros((N, 3), dtype=bool)
        self._cC_st = np.zeros((N, 3), dtype=bool)
        self._cS_st = np.zeros((N, 3), dtype=bool)

        # --- Actions ---
        self._act1      = np.zeros(N, dtype=bool)
        self._invest1   = np.zeros(N, dtype=bool)
        self._act1_year = np.zeros(N, dtype=np.int32)
        # h_sta flags — insulated is set in recall_memory and never reset
        self._insulated = np.zeros(N, dtype=bool)
        self._efficient = np.zeros(N, dtype=bool)

        # --- Energy / investment tallies (only channel 0 active in v4) ---
        self._save_a0 = np.zeros(N, dtype=np.float64)
        self._invs_a0 = np.zeros(N, dtype=np.float64)

        # --- Spatial ---
        self._grid_x = np.array([h.grid_x for h in hhs], dtype=np.int32)
        self._grid_y = np.array([h.grid_y for h in hhs], dtype=np.int32)

        self._build_neighbor_index()

        # Free Household objects and the temporary grid dict
        self.households = []
        self._grid = {}

    def _build_neighbor_index(self) -> None:
        """Precompute _nbr_idx[i] = array of agent indices on the 8 adjacent patches."""
        grid_dict: Dict[Tuple[int, int], List[int]] = {}
        for i in range(self.n_households):
            key = (int(self._grid_x[i]), int(self._grid_y[i]))
            if key not in grid_dict:
                grid_dict[key] = []
            grid_dict[key].append(i)

        self._nbr_idx: List[np.ndarray] = []
        for i in range(self.n_households):
            gx, gy = int(self._grid_x[i]), int(self._grid_y[i])
            nbrs: List[int] = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nbrs.extend(grid_dict.get((gx + dx, gy + dy), []))
            self._nbr_idx.append(np.array(nbrs, dtype=np.int32))

    # ------------------------------------------------------------------
    # go() sub-routines (in order of execution)
    # ------------------------------------------------------------------

    def _update_info(self) -> None:
        """Reset act1 each tick."""
        self._act1[:] = False

    def _recall_memory(self) -> None:
        """Assign historical renovation status to some households in the first year."""
        if self.year != START_YEAR or not self.memory_on:
            return
        recall_probs = RECALL_PROB[self.case_study]
        for i in range(self.n_households):
            g_key = min(int(self._h_group[i]), 5)
            p = recall_probs.get(g_key, 0.0)
            if random.uniform(0, 100) <= p:
                self._act1[i]      = True
                self._invest1[i]   = True
                self._insulated[i] = True

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
        """Compute awareness and guilt; mirrors knowledge procedure."""
        thresh = GUILT_THRESH[self.case_study]
        self._aware = (self._know + self._cee_aw + self._ed_aw) / 3.0
        self._guilt = self._aware >= thresh
        self._k     = np.where(self._guilt, self._aware / 7.0, 0.0)

    def _motivation(self) -> None:
        """Set m_st to H/L for guilty households; non-guilty households retain prior value."""
        thr = MOTIVATION_THRESH[self.case_study]
        pn1_thr, sn1_thr = thr["m1"]
        pn2_thr, sn2_thr = thr["m2"]
        pn3_thr, sn3_thr = thr["m3"]
        g = self._guilt
        self._m_st[g, 0] = (self._pn[g, 0] >= pn1_thr) & (self._sn[g, 0] >= sn1_thr)
        self._m_st[g, 1] = (self._pn[g, 1] >= pn2_thr) & (self._sn[g, 1] >= sn2_thr)
        self._m_st[g, 2] = (self._pn[g, 2] >= pn3_thr) & (self._sn[g, 2] >= sn3_thr)

    def _consideration(self) -> None:
        """
        Evaluate PBC constraints for each behaviour.
        Only updates households whose motivation status is H;
        others retain their prior consideration status (sticky).
        """
        pbc_inv = PBC_INVEST_THRESH[self.case_study]
        pbc_sw  = PBC_SWITCH_THRESH[self.case_study]
        owner   = self._dw_st == 1

        # Investment: update only where m1 is H
        if self.investment:
            m = self._m_st[:, 0]
            for j in range(3):
                self._cI_st[m, j] = (self._pbcI[m, j] >= pbc_inv) & owner[m]

        # Conservation: update only where m2 is H
        m = self._m_st[:, 1]
        for j in range(3):
            self._cC_st[m, j] = (
                (self._pbcC[m, j] >= PBC_CONSERV_THRESH) &
                (self._ene_pat[m, j] != 3)
            )

        # Switching: update only where m3 is H
        m = self._m_st[:, 2]
        for j in range(3):
            self._cS_st[m, j] = self._pbcS[m, j] >= pbc_sw

    def _utility(self) -> None:
        """Calculate renovation utility U1; zero for non-eligible households."""
        c = UTILITY_COEF
        self._U1[:] = 0.0
        mask = self._cI_st[:, 0]
        self._U1[mask] = (
              self._edu[mask].astype(np.float64)      * c["edu"]
            + self._age[mask].astype(np.float64)      * c["age"]
            + self._dw_elab[mask].astype(np.float64)  * c["dw_elab"]
            + self._dw_type[mask].astype(np.float64)  * c["dw_type"]
            + self._dw_age[mask].astype(np.float64)   * c["dw_age"]
            + self._dw_size[mask].astype(np.float64)  * c["dw_size"]
            + self._gas[mask]    * c["gas"]
            + self._pn[mask, 0]  * c["pn1"]
            + self._erI[mask, 0]
        )

    def _action(self) -> None:
        """Decide whether to renovate; mirrors action procedure."""
        eligible = (
            (self._U1 > 0)
            & ~self._invest1
            & ~self._insulated
            & (self._dw_elab > 1)
        )
        self._act1     = eligible
        self._invest1 |= eligible
        self.a1_cum   += int(np.sum(eligible))

    def _save_energy(self) -> None:
        """Calculate gas savings for renovating households."""
        self._save_a0[:] = 0.0
        m = self._act1
        self._save_a0[m] = self._gas[m] * GAS_SAVE_FRACTION
        self._gas[m]    -= self._save_a0[m]

    def _invest(self) -> None:
        """Record investment cost for renovating households."""
        self._invs_a0[:] = 0.0
        self._invs_a0[self._act1] = I1_COST

    def _learn(self) -> None:
        """
        Social learning and information diffusion.

        Informative: broadcast +5% knowledge boost to all households.
        Slow/Fast/Informative social: active households (act1 OR invest1)
            boost their spatial neighbours.  Slow requires >4 neighbours;
            Fast and Informative update regardless of count.
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

        # Social learning from active households to their neighbours
        active_idx = np.where(self._act1 | self._invest1)[0]
        slow = (self.learning == "Slow dynamics")

        for i in active_idx:
            # Self-boost pbcI[0]
            if self._pbcI[i, 0] < lc:
                self._pbcI[i, 0] = min(float(self._pbcI[i, 0]) * (1.0 + lr), lc)

            nbrs = self._nbr_idx[i]
            if len(nbrs) == 0:
                continue

            # Each neighbour computes stats from its OWN patch-neighbours
            ngb_k:     Dict[int, float] = {}
            ngb_ca:    Dict[int, float] = {}
            ngb_ed:    Dict[int, float] = {}
            ngb_pn1:   Dict[int, float] = {}
            ngb_sn1:   Dict[int, float] = {}
            ngb_pbcI1: Dict[int, float] = {}
            for j in nbrs:
                nn = self._nbr_idx[j]
                if len(nn) == 0:
                    continue
                ngb_k[j]     = _max_mean_median_arr(self._know[nn])
                ngb_ca[j]    = _max_mean_median_arr(self._cee_aw[nn])
                ngb_ed[j]    = _max_mean_median_arr(self._ed_aw[nn])
                ngb_pn1[j]   = _max_mean_median_arr(self._pn[nn, 0])
                ngb_sn1[j]   = _max_mean_median_arr(self._sn[nn, 0])
                ngb_pbcI1[j] = _max_mean_median_arr(self._pbcI[nn, 0])

            # Slow dynamics: skip if active hh does not have enough neighbours
            if slow and len(nbrs) <= SLOW_NEIGHBOR_MIN:
                continue

            for j in nbrs:
                if j not in ngb_k:
                    continue
                if self._know[j]   < ngb_k[j]     and self._know[j]   < lc:
                    self._know[j]   = min(float(self._know[j])   * (1.0 + lr), lc)
                if self._cee_aw[j] < ngb_ca[j]    and self._cee_aw[j] < lc:
                    self._cee_aw[j] = min(float(self._cee_aw[j]) * (1.0 + lr), lc)
                if self._ed_aw[j]  < ngb_ed[j]    and self._ed_aw[j]  < lc:
                    self._ed_aw[j]  = min(float(self._ed_aw[j])  * (1.0 + lr), lc)
                if self._pn[j, 0]  < ngb_pn1[j]   and self._pn[j, 0]  < lc:
                    self._pn[j, 0]  = min(float(self._pn[j, 0])  * (1.0 + lr), lc)
                if self._sn[j, 0]  < ngb_sn1[j]   and self._sn[j, 0]  < lc:
                    self._sn[j, 0]  = min(float(self._sn[j, 0])  * (1.0 + lr), lc)
                if self._pbcI[j,0] < ngb_pbcI1[j] and self._pbcI[j,0] < lc:
                    self._pbcI[j,0] = min(float(self._pbcI[j,0]) * (1.0 + lr), lc)

    def _update_income(self) -> None:
        """Multiply household income by CGE growth factor."""
        if self.n >= len(self.cge):
            return
        self._income *= self.cge[self.n]

    def _update_energy(self) -> None:
        """Improve energy label for renovating households."""
        m = self._act1
        self._dw_elab[m & (self._dw_elab >= 2)] -= 1
        self._efficient[m & (self._dw_elab == 1)] = True

    def _update_memory(self) -> None:
        """Tick cooldown counters and reset invest1 after the cooldown expires."""
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

        renov_by_dwage: Dict[int, int] = {}
        total_by_dwage: Dict[int, int] = {}
        for cat in (1, 2, 3):
            age_mask = self._dw_age == cat
            total_by_dwage[cat] = int(np.sum(age_mask))
            renov_by_dwage[cat] = int(np.sum(self._act1 & age_mask))

        # Groups 5-7 all map to bucket 5
        g_arr = np.where(self._h_group <= 4, self._h_group, 5)
        renov_by_group: Dict[int, int] = {}
        total_by_group: Dict[int, int] = {}
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

    def renovation_rate_by_vintage(self) -> Dict[int, List[float]]:
        result = {1: [], 2: [], 3: []}
        for s in self.history:
            for cat in (1, 2, 3):
                tot = s.total_by_dwage.get(cat, 0)
                ren = s.renov_by_dwage.get(cat, 0)
                result[cat].append(100.0 * ren / tot if tot > 0 else 0.0)
        return result

    def renovation_rate_by_income(self) -> Dict[int, List[float]]:
        result = {g: [] for g in range(1, 6)}
        for s in self.history:
            for g in range(1, 6):
                tot = s.total_by_group.get(g, 0)
                ren = s.renov_by_group.get(g, 0)
                result[g].append(100.0 * ren / tot if tot > 0 else 0.0)
        return result

    def years(self) -> List[int]:
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
