"""
BENCH v4 — Python translation of BENCH_v04_B-NLD.ESP.nlogox.

The go() method follows the NetLogo procedure bodies in order:
    tick
    recallmemory
    update.info      -> _update_info
    update.dwelling  -> _update_dwelling
    knowledge        -> _knowledge
    motivation       -> _motivation
    consideration    -> _consideration
    utility          -> _utility
    action           -> _action
    save.energy      -> _save_energy
    invest           -> _invest
    learn            -> _learn
    update.income    -> _update_income
    update.energy    -> _update_energy
    update.memory    -> _update_memory
    year += 1, n += 1

All thresholds and empirical parameters live in params.py.
"""

import csv
import math
import os
import random
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _load_cge(filepath: str) -> List[float]:
    """Load a single-column CGE CSV into a list of floats."""
    values = []
    with open(filepath, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if row and row[0].strip():
                values.append(float(row[0].strip()))
    return values


def _mean(vals):
    return sum(vals) / len(vals) if vals else 0.0


def _median(vals):
    if not vals:
        return 0.0
    return statistics.median(vals)


def _max_mean_median(vals):
    """max(mean, median) — mirrors NetLogo: max list mean median."""
    if not vals:
        return 0.0
    return max(_mean(vals), _median(vals))


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
    Python translation of BENCH NetLogo v4 (renovation/insulation module).

    Parameters
    ----------
    case_study : "ES" (Spain-Navarre) | "NL" (Netherlands-Overijssel)
    seed       : random seed; None = generate new seed
    learning   : "Slow dynamics" | "Fast dynamics" | "Informative" | "No learning"
    memory     : True/False — apply recall of pre-2016 renovations in first tick
    investment : True/False — whether Investment behaviour is enabled
    data_dir   : path to folder containing the CGE CSV files
                 (defaults to the netlogo/ subfolder next to this package)
    """

    def __init__(
        self,
        case_study: str = "NL",
        seed: Optional[int] = None,
        learning: str = "Informative",
        memory: bool = True,
        investment: bool = True,
        data_dir: Optional[str] = None,
    ):
        self.case_study = case_study
        self.learning   = learning
        self.memory_on  = memory
        self.investment = investment

        # random seed
        if seed is None:
            seed = random.randint(0, 2**31 - 1)
        self.seed = seed
        random.seed(seed)

        # data directory
        if data_dir is None:
            here = Path(__file__).parent.parent
            data_dir = str(here / "data")
        self.data_dir = data_dir

        # simulation state
        self.year: int = START_YEAR
        self.n:    int = 0          # CGE time index, mirrors NetLogo's n global

        # households and grid
        self.households: List[Household] = []
        self._grid: Dict[Tuple[int, int], List[Household]] = {}

        # CGE income growth series
        self.cge: List[float] = []

        # results
        self.history: List[AnnualStats] = []

        # cumulative action counters (mirroring a1.com etc.)
        self.a1_cum: int = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Initialise households, grid and load data (mirrors NetLogo setup)."""
        self._load_data()
        self._create_households()
        self._place_on_grid()

    def go(self) -> bool:
        """
        Execute one simulation tick (one calendar year).
        Returns False when the model should stop (year > END_YEAR).
        """
        if self.year > END_YEAR:
            return False

        # --- NetLogo go procedure body ---
        self._update_info()
        self._recall_memory()      # only acts in year == START_YEAR if memory_on
        self._update_dwelling()

        # behavioural pipeline (COM = True by default in v4)
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

        # collect stats before incrementing year
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
                pct_renov = 100 * last.n_renovated / len(self.households) if self.households else 0
                print(
                    f"  year={last.year}  renovated={last.n_renovated} "
                    f"({pct_renov:.1f}%)  gas_saved={last.total_gas_saved:.0f}"
                )
        return self.history

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _load_data(self) -> None:
        groups = ES_GROUPS if self.case_study == "ES" else NL_GROUPS
        fname  = CGE_FILES[self.case_study]
        fpath  = os.path.join(self.data_dir, fname)
        self.cge = _load_cge(fpath)

    def _create_households(self) -> None:
        """
        Create N_HOUSEHOLDS agents using empirical distributions.
        Mirrors the `create-turtles N [...]` block in NetLogo setup.
        """
        groups = ES_GROUPS if self.case_study == "ES" else NL_GROUPS
        n_hh   = N_HOUSEHOLDS[self.case_study]

        for hh_id in range(n_hh):
            rn = random.uniform(0, 100)
            # find which income group this rn maps to
            for cum_upper, group_id, params in groups:
                if rn < cum_upper:
                    hh = Household(hh_id, self.case_study, rn, params, group_id)
                    self.households.append(hh)
                    break
            else:
                # rn == 100 edge case: assign last group
                _, group_id, params = groups[-1]
                hh = Household(hh_id, self.case_study, rn, params, group_id)
                self.households.append(hh)

    def _place_on_grid(self) -> None:
        """
        Place agents on the NetLogo grid (GRID_MIN to GRID_MAX in both axes).
        Mirrors `setxy random-xcor random-ycor`.
        """
        self._grid = {}
        span = GRID_MAX - GRID_MIN + 1
        for hh in self.households:
            hh.grid_x = random.randint(GRID_MIN, GRID_MAX)
            hh.grid_y = random.randint(GRID_MIN, GRID_MAX)
            key = (hh.grid_x, hh.grid_y)
            if key not in self._grid:
                self._grid[key] = []
            self._grid[key].append(hh)

    def _get_patch_neighbors(self, hh: Household) -> List[Household]:
        """
        Return all turtles on the 8 adjacent patches (NetLogo neighbors).
        Wrapping is NOT used here (no wrapping in view settings that affect
        turtle placement — turtles stay within bounds).
        """
        result = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                key = (hh.grid_x + dx, hh.grid_y + dy)
                result.extend(self._grid.get(key, []))
        return result

    # ------------------------------------------------------------------
    # go procedure sub-routines (in order of execution)
    # ------------------------------------------------------------------

    def _update_info(self) -> None:
        """reset act1 to False each tick — mirrors update.info"""
        for hh in self.households:
            hh.act1 = False

    def _recall_memory(self) -> None:
        """
        Assign historical renovation status to some households in the first
        year.  Mirrors the recallmemory procedure.
        """
        if self.year != START_YEAR or not self.memory_on:
            return

        recall_probs = RECALL_PROB[self.case_study]

        for hh in self.households:
            # NetLogo maps groups 5-7 all to prob[5]
            g = hh.h_group if hh.h_group <= 4 else 5
            p = recall_probs.get(g, 0.0)
            aa = random.uniform(0, 100)
            if aa <= p:
                hh.act1    = True
                hh.invest1 = True
                hh.h_sta   = "insulated"

    def _update_dwelling(self) -> None:
        """
        Probabilistically update dw_age from MESSAGEix-Buildings inputs
        from 2025 onwards.  Mirrors update.dwelling.
        """
        dw_table = DWAGE_UPDATE.get(self.case_study, {})
        for (yr_lo, yr_hi), (p_new, p_mid) in dw_table.items():
            if yr_lo <= self.year < yr_hi:
                for hh in self.households:
                    dag = random.uniform(0, 100)
                    if dag < p_new:
                        hh.dw_age = 1
                    elif dag < p_mid:
                        hh.dw_age = 2
                    else:
                        hh.dw_age = 3
                break   # only one band applies per year

    def _knowledge(self) -> None:
        """
        Compute awareness and guilt.  Mirrors knowledge procedure.
            aware = (know + cee.aw + ed.aw) / 3
            guilt threshold: NL=4.6, ES=5.2
            k = aware / 7  if guilt == "H"
        """
        thresh = GUILT_THRESH[self.case_study]
        for hh in self.households:
            hh.aware = (hh.know + hh.cee_aw + hh.ed_aw) / 3.0
            hh.guilt = "H" if hh.aware >= thresh else "L"
            hh.k     = (hh.aware / 7.0) if hh.guilt == "H" else 0.0

    def _motivation(self) -> None:
        """
        Set m_st[0/1/2] to "H" or "L" based on pn and sn vs thresholds.
        Only households with guilt="H" proceed.  Mirrors motivation procedure.
        """
        thresholds = MOTIVATION_THRESH[self.case_study]
        pn1_thr, sn1_thr = thresholds["m1"]
        pn2_thr, sn2_thr = thresholds["m2"]
        pn3_thr, sn3_thr = thresholds["m3"]
        for hh in self.households:
            if hh.guilt != "H":
                continue
            hh.m_st[0] = "H" if (hh.pn[0] >= pn1_thr and hh.sn[0] >= sn1_thr) else "L"
            hh.m_st[1] = "H" if (hh.pn[1] >= pn2_thr and hh.sn[1] >= sn2_thr) else "L"
            hh.m_st[2] = "H" if (hh.pn[2] >= pn3_thr and hh.sn[2] >= sn3_thr) else "L"

    def _consideration(self) -> None:
        """
        Evaluate perceived-behavioural-control constraints.
        Mirrors consideration procedure.

        Investment (cI_st):
            NL: pbcI1 >= 1 AND dw_st==1 (owner)
            ES: pbcI1 >= 2.2 AND dw_st==1
        Conservation (cC_st):
            pbcC1 >= 1 AND ene_pat != 3 (not "almost always efficient")
        Switching (cS_st):
            NL: pbcS1 >= 1   ES: pbcS1 >= 3.5  (only if currently grey or brown)
        """
        pbc_inv = PBC_INVEST_THRESH[self.case_study]
        pbc_sw  = PBC_SWITCH_THRESH[self.case_study]

        for hh in self.households:
            # investment — requires motivation m_st[0]=="H"
            if self.investment and hh.m_st[0] == "H":
                owner = (hh.dw_st == 1)
                for i in range(3):
                    if hh.pbcI[i] >= pbc_inv and owner:
                        hh.cI_st[i] = "H"
                    else:
                        hh.cI_st[i] = "L"
            # conservation — m_st[1]=="H"
            if hh.m_st[1] == "H":
                for i in range(3):
                    if hh.pbcC[i] >= PBC_CONSERV_THRESH and hh.ene_pat[i] != 3:
                        hh.cC_st[i] = "H"
                    else:
                        hh.cC_st[i] = "L"
            # switching — m_st[2]=="H"
            if hh.m_st[2] == "H":
                for i in range(3):
                    if hh.pbcS[i] >= pbc_sw:
                        hh.cS_st[i] = "H"
                    else:
                        hh.cS_st[i] = "L"

    def _utility(self) -> None:
        """
        Calculate renovation utility U1 for eligible households.
        Mirrors utility procedure:
            if cI1.st == "L": U1 = 0
            if cI1.st == "H":
                U1 = edu*0.0563284 + age*0.0008106 + dw_elab*(-0.0769971)
                   + dw_type*0.4265 + dw_age*0.0883428 + dw_size*0.0857047
                   + gas*0.0000488 + pn1*0.052849 + erI1
        """
        c = UTILITY_COEF
        for hh in self.households:
            if hh.cI_st[0] != "H":
                hh.U1 = 0.0
            else:
                hh.U1 = (
                    hh.edu      * c["edu"]
                    + hh.age    * c["age"]
                    + hh.dw_elab * c["dw_elab"]
                    + hh.dw_type * c["dw_type"]
                    + hh.dw_age  * c["dw_age"]
                    + hh.dw_size * c["dw_size"]
                    + hh.gas     * c["gas"]
                    + hh.pn[0]   * c["pn1"]
                    + hh.erI[0]
                )

    def _action(self) -> None:
        """
        Decide whether to renovate.  Mirrors action procedure:
            if U1 <= 0 or invest1==True or h_sta=="insulated": act1 = False
            if U1 > 0 and invest1==False and dw_elab > 1:      act1 = True, invest1 = True
        """
        for hh in self.households:
            if hh.U1 <= 0 or hh.invest1 or hh.h_sta == "insulated":
                hh.act1 = False
            elif hh.U1 > 0 and not hh.invest1 and hh.dw_elab > 1:
                hh.act1    = True
                hh.invest1 = True

        # cumulative counter
        self.a1_cum += sum(1 for hh in self.households if hh.act1)

    def _save_energy(self) -> None:
        """
        Calculate and apply gas savings for renovating households.
        Mirrors save.energy procedure (only from year >= 2017):
            save.a1 = gas * 0.20
            gas = gas - save.a1
        """
        for hh in self.households:
            if hh.act1:
                hh.save_a[0] = hh.gas * GAS_SAVE_FRACTION
                hh.gas       = hh.gas - hh.save_a[0]
            else:
                hh.save_a[0] = 0.0

    def _invest(self) -> None:
        """
        Record investment cost for renovating households.
        Mirrors invest procedure (only from year >= 2017):
            invs.a1 = I1.cost  if act1 == True
        """
        for hh in self.households:
            hh.invs_a[0] = I1_COST if hh.act1 else 0.0

    def _learn(self) -> None:
        """
        Social learning and information diffusion.
        Mirrors learn procedure (only from year >= 2017).

        Slow dynamics:
            Active households (act1 OR invest1):
                1. Self-boost pbcI[0] by 5% (capped at 6.6)
                2. For each neighbor, let neighbor compute their own
                   neighborhood stats.
                3. If the active hh has > 4 patch-neighbors, each of
                   those neighbors updates know/cee_aw/ed_aw/pn1/sn1/pbcI1.

        Fast dynamics:
            Same as Slow but the update happens inside the neighbor loop
            regardless of the >4 count check.

        Informative:
            ALL households get +5% to know, cee_aw, ed_aw (capped).
            Active households additionally trigger neighbor social learning
            (same logic as Fast dynamics).
        """
        if self.learning == "No learning":
            return

        # Informative: broadcast knowledge boost to all households first
        if self.learning == "Informative":
            for hh in self.households:
                if hh.know   <= LEARNING_CAP:
                    hh.know   = min(hh.know   + hh.know   * LEARNING_RATE, LEARNING_CAP + LEARNING_RATE)
                if hh.cee_aw <= LEARNING_CAP:
                    hh.cee_aw = min(hh.cee_aw + hh.cee_aw * LEARNING_RATE, LEARNING_CAP + LEARNING_RATE)
                if hh.ed_aw  <= LEARNING_CAP:
                    hh.ed_aw  = min(hh.ed_aw  + hh.ed_aw  * LEARNING_RATE, LEARNING_CAP + LEARNING_RATE)

        # Social learning: active households influence neighbors
        for hh in self.households:
            if not (hh.act1 or hh.invest1):
                continue

            # Self: boost pbcI1 if below cap
            if hh.pbcI[0] < LEARNING_CAP:
                hh.pbcI[0] = min(hh.pbcI[0] * (1 + LEARNING_RATE), LEARNING_CAP)

            # find neighbors (link-neighbors in NetLogo)
            neighbors = self._get_patch_neighbors(hh)
            if not neighbors:
                continue

            # Each neighbor computes neighborhood stats from THEIR OWN patch-neighbors
            for ngb in neighbors:
                patch_ngbs = self._get_patch_neighbors(ngb)
                if not patch_ngbs:
                    continue
                ngb.ngb_k     = _max_mean_median([n.know   for n in patch_ngbs])
                ngb.ngb_ca    = _max_mean_median([n.cee_aw for n in patch_ngbs])
                ngb.ngb_ed    = _max_mean_median([n.ed_aw  for n in patch_ngbs])
                ngb.ngb_pn1   = _max_mean_median([n.pn[0]  for n in patch_ngbs])
                ngb.ngb_sn1   = _max_mean_median([n.sn[0]  for n in patch_ngbs])
                ngb.ngb_pbcI1 = _max_mean_median([n.pbcI[0] for n in patch_ngbs])

            # Slow dynamics: update only if active hh has > SLOW_NEIGHBOR_MIN neighbors
            # Fast / Informative: update regardless
            if self.learning == "Slow dynamics" and len(neighbors) <= SLOW_NEIGHBOR_MIN:
                continue

            for ngb in neighbors:
                if ngb.know   < ngb.ngb_k    and ngb.know   < LEARNING_CAP:
                    ngb.know   = min(ngb.know   * (1 + LEARNING_RATE), LEARNING_CAP)
                if ngb.cee_aw < ngb.ngb_ca   and ngb.cee_aw < LEARNING_CAP:
                    ngb.cee_aw = min(ngb.cee_aw * (1 + LEARNING_RATE), LEARNING_CAP)
                if ngb.ed_aw  < ngb.ngb_ed   and ngb.ed_aw  < LEARNING_CAP:
                    ngb.ed_aw  = min(ngb.ed_aw  * (1 + LEARNING_RATE), LEARNING_CAP)
                if ngb.pn[0]  < ngb.ngb_pn1  and ngb.pn[0]  < LEARNING_CAP:
                    ngb.pn[0]  = min(ngb.pn[0]  * (1 + LEARNING_RATE), LEARNING_CAP)
                if ngb.sn[0]  < ngb.ngb_sn1  and ngb.sn[0]  < LEARNING_CAP:
                    ngb.sn[0]  = min(ngb.sn[0]  * (1 + LEARNING_RATE), LEARNING_CAP)
                if ngb.pbcI[0] < ngb.ngb_pbcI1 and ngb.pbcI[0] < LEARNING_CAP:
                    ngb.pbcI[0] = min(ngb.pbcI[0] * (1 + LEARNING_RATE), LEARNING_CAP)

    def _update_income(self) -> None:
        """
        Multiply household income by CGE growth factor for current n.
        Mirrors update.income procedure.
        All income groups use column 0 of row n from the CGE CSV.
        """
        if self.n >= len(self.cge):
            return
        multiplier = self.cge[self.n]
        for hh in self.households:
            hh.income *= multiplier

    def _update_energy(self) -> None:
        """
        Improve energy label for renovating households.
        Mirrors update.energy procedure:
            if act1 and dw_elab >= 2: dw_elab -= 1
            if act1 and dw_elab == 1: h_sta = "Efficient"
        """
        for hh in self.households:
            if hh.act1:
                if hh.dw_elab >= 2:
                    hh.dw_elab -= 1
                elif hh.dw_elab == 1:
                    hh.h_sta = "Efficient"

    def _update_memory(self) -> None:
        """
        Update renovation cooldown counters and reset invest1 after the
        cooldown period expires.  Mirrors update.memory procedure:
            if invest1: act1_year += 1
            if act1_year >= cooldown(dw_age): invest1 = False, act1_year = 0
        """
        for hh in self.households:
            if hh.invest1:
                hh.act1_year += 1
            cooldown = COOLDOWN_BY_DWAGE.get(hh.dw_age, 7)
            if hh.act1_year >= cooldown:
                hh.invest1   = False
                hh.act1_year = 0

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def _collect_stats(self) -> AnnualStats:
        n_hh = len(self.households)
        renovated  = [hh for hh in self.households if hh.act1]
        in_process = [hh for hh in self.households if hh.invest1]

        renov_by_dwage: Dict[int, int] = {1: 0, 2: 0, 3: 0}
        total_by_dwage: Dict[int, int] = {1: 0, 2: 0, 3: 0}
        renov_by_group: Dict[int, int] = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        total_by_group: Dict[int, int] = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

        for hh in self.households:
            age_cat = hh.dw_age
            grp     = hh.h_group if hh.h_group <= 4 else 5
            total_by_dwage[age_cat] += 1
            total_by_group[grp]     += 1
            if hh.act1:
                renov_by_dwage[age_cat] += 1
                renov_by_group[grp]     += 1

        high_guilt = sum(1 for hh in self.households if hh.guilt == "H")
        high_m1    = sum(1 for hh in self.households if hh.m_st[0] == "H")
        high_m2    = sum(1 for hh in self.households if hh.m_st[1] == "H")
        high_m3    = sum(1 for hh in self.households if hh.m_st[2] == "H")

        return AnnualStats(
            year                     = self.year,
            n_renovated              = len(renovated),
            n_conservation           = sum(1 for hh in self.households if hh.act2),
            n_switching              = sum(1 for hh in self.households if hh.act3),
            n_invested               = len(in_process),
            total_gas_saved          = sum(hh.save_a[0] for hh in self.households),
            total_energy_conservation= sum(hh.save_a[1] for hh in self.households),
            total_energy_switching   = sum(hh.save_a[2] for hh in self.households),
            total_investment         = sum(hh.invs_a[0] for hh in self.households),
            total_invest_conservation= sum(hh.invs_a[1] for hh in self.households),
            total_invest_switching   = sum(hh.invs_a[2] for hh in self.households),
            avg_aware  = _mean([hh.aware for hh in self.households]),
            avg_pn1    = _mean([hh.pn[0] for hh in self.households]),
            avg_sn1    = _mean([hh.sn[0] for hh in self.households]),
            high_guilt_pct = 100.0 * high_guilt / n_hh if n_hh else 0.0,
            high_m1_pct    = 100.0 * high_m1    / n_hh if n_hh else 0.0,
            high_m2_pct    = 100.0 * high_m2    / n_hh if n_hh else 0.0,
            high_m3_pct    = 100.0 * high_m3    / n_hh if n_hh else 0.0,
            renov_by_dwage = renov_by_dwage,
            total_by_dwage = total_by_dwage,
            renov_by_group = renov_by_group,
            total_by_group = total_by_group,
        )

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    def renovation_rate_by_vintage(self) -> Dict[int, List[float]]:
        """
        Return renovation rates (%) by vintage category over time,
        matching Fig. 5 in the paper.

        Returns dict: {dw_age_category: [pct_year_1, pct_year_2, ...]}
        where pct is relative to the total in each cohort.
        """
        result = {1: [], 2: [], 3: []}
        for s in self.history:
            for cat in (1, 2, 3):
                tot = s.total_by_dwage.get(cat, 0)
                ren = s.renov_by_dwage.get(cat, 0)
                result[cat].append(100.0 * ren / tot if tot > 0 else 0.0)
        return result

    def renovation_rate_by_income(self) -> Dict[int, List[float]]:
        """
        Return renovation rates (%) by income group over time,
        matching Fig. 7 in the paper.
        """
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
        """Print a brief text summary."""
        lines = [
            f"BENCH v4 — {self.case_study}  Learning={self.learning}  seed={self.seed}",
            f"Years {START_YEAR}–{END_YEAR}  |  N={len(self.households)} households",
            "",
            f"{'Year':>6}  {'Renov':>7}  {'%Renov':>7}  {'GasSaved(kWh)':>14}  {'Invest(EUR)':>12}",
        ]
        n_hh = len(self.households)
        for s in self.history:
            pct = 100 * s.n_renovated / n_hh if n_hh else 0
            lines.append(
                f"{s.year:>6}  {s.n_renovated:>7}  {pct:>7.2f}  "
                f"{s.total_gas_saved:>14,.0f}  {s.total_investment:>12,.0f}"
            )
        return "\n".join(lines)
