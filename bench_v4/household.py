"""
Household agent for BENCH v4.

Each household maps 1-to-1 to a NetLogo turtle.  All attribute names follow
the NetLogo originals (with dots replaced by underscores) so that the model
code reads like a direct translation of the procedure bodies.
"""

import math
import random
from typing import Optional


def _rand_num(lo: float, hi: float, step: float) -> float:
    """
    Exact Python equivalent of NetLogo's randomNumber reporter:
        lower.value + step.value * random(1 + floor((upper - lower) / step))
    Produces a uniformly-distributed stepped value in [lo, hi].
    """
    n_steps = math.floor((hi - lo) / step)
    return lo + step * random.randint(0, n_steps)


def _categorical(rn: float, breaks: list) -> int:
    """
    Assign a categorical value given a random float rn in [0, 100) and a list
    of (cumulative_upper, value) tuples.  The last entry's upper should be 100.
    """
    for upper, val in breaks:
        if rn < upper:
            return val
    return breaks[-1][1]


def _rand_er(spec, step: float = 0.01) -> float:
    """
    Resolve an error-term spec from params.  None means fixed value of -0.01.
    """
    if spec is None:
        return -0.01
    return _rand_num(spec[0], spec[1], step)


class Household:
    """
    One household agent, matching the NetLogo turtle-own variables.

    Attribute name mapping (NetLogo -> Python):
        h.id        -> h_id
        h.group     -> h_group
        h.sta       -> h_sta
        dw.st       -> dw_st
        dw.elab     -> dw_elab
        dw.type     -> dw_type
        dw.age      -> dw_age
        dw.size     -> dw_size
        ene.prov    -> ene_prov
        ene.patN    -> ene_pat (list of 3)
        cee.aw      -> cee_aw
        ed.aw       -> ed_aw
        m1.st/m2.st/m3.st -> m_st (list of 3 strings)
        pnN/snN     -> pn/sn (lists of 3)
        pbcIN/pbcCN/pbcSN -> pbcI/pbcC/pbcS (lists of 3)
        erIN/erCN/erSN    -> erI/erC/erS (lists of 3)
        cI1.st etc. -> cI_st/cC_st/cS_st (lists of 3)
        act1.year   -> act1_year
        invs.aN     -> invs_a (list of 9)
        save.aN     -> save_a (list of 6)
        ngb.*       -> ngb_* (neighbor cache, computed each step)
    """

    __slots__ = [
        "h_id", "h_group", "h_sta", "sdgroup",
        "income", "gen", "edu", "ecom", "age",
        "dw_st", "dw_elab", "dw_type", "dw_age", "dw_size",
        "elec", "gas",
        "aware", "know", "cee_aw", "ed_aw", "guilt", "k",
        "m1", "m2", "m3", "m_st",           # m_st[0]=m1.st, [1]=m2.st, [2]=m3.st
        "pn", "sn",                           # lists of 3
        "pbcI", "pbcC", "pbcS",              # lists of 3
        "ene_pat",                            # list of 3
        "ene_prov",
        "erI", "erC", "erS",                 # lists of 3
        "cI_st", "cC_st", "cS_st",           # constraint statuses, lists of 3
        "U1", "act1", "act2", "act3",
        "act1_year", "invest1", "invest2",
        "save_a",                             # list of 6
        "invs_a",                             # list of 9
        "renov",
        # spatial
        "grid_x", "grid_y",
        # neighbor cache (filled during learning)
        "ngb_k", "ngb_ca", "ngb_ed",
        "ngb_pn1", "ngb_sn1", "ngb_pbcI1",
        # transient draw value used for recall / dwelling update
        "rn",
    ]

    def __init__(self, h_id: int, case_study: str, rn: float, group_params: dict,
                 h_group: int):
        p = group_params
        self.h_id    = h_id
        self.h_group = h_group
        self.h_sta   = ""
        self.sdgroup = 0
        self.rn      = rn          # kept for income update (used as group selector)

        # --- behavioral init ---
        self.aware = 0.0
        self.guilt = "Null"
        self.k     = 0.0
        self.m1 = 0.0
        self.m2 = 0.0
        self.m3 = 0.0
        self.m_st = ["Null", "Null", "Null"]

        # --- action init ---
        self.U1       = 0.0
        self.act1     = False
        self.act2     = False   # conservation (not active in v4)
        self.act3     = False   # switching    (not active in v4)
        self.act1_year = 0
        self.invest1  = False
        self.invest2  = False
        self.save_a   = [0.0] * 6
        self.invs_a   = [0.0] * 9
        self.renov    = False

        # --- constraint statuses ---
        self.cI_st = ["Null", "Null", "Null"]
        self.cC_st = ["Null", "Null", "Null"]
        self.cS_st = ["Null", "Null", "Null"]

        # --- neighbor cache ---
        self.ngb_k    = 0.0
        self.ngb_ca   = 0.0
        self.ngb_ed   = 0.0
        self.ngb_pn1  = 0.0
        self.ngb_sn1  = 0.0
        self.ngb_pbcI1 = 0.0

        # --- spatial (set later by model) ---
        self.grid_x = 0
        self.grid_y = 0

        # --- stochastic initialization from empirical distributions ---
        self.elec = _rand_num(1000, 5000, 100)

        lo, hi, st = p["income_range"]
        self.income = _rand_num(lo, hi, st)
        lo, hi, st = p["gas_range"]
        self.gas    = _rand_num(lo, hi, st)

        self.know   = _rand_num(*p["know_range"],   0.05)
        self.cee_aw = _rand_num(*p["cee_aw_range"], 0.05)
        self.ed_aw  = _rand_num(*p["ed_aw_range"],  0.05)

        pn_lo, pn_hi = p["pn_range"]
        self.pn = [_rand_num(pn_lo, pn_hi, 0.05) for _ in range(3)]
        sn_lo, sn_hi = p["sn_range"]
        self.sn = [_rand_num(sn_lo, sn_hi, 0.05) for _ in range(3)]

        pbcI_lo, pbcI_hi = p["pbcI_range"]
        self.pbcI = [_rand_num(pbcI_lo, pbcI_hi, 0.05) for _ in range(3)]
        pbcC_lo, pbcC_hi = p["pbcC_range"]
        self.pbcC = [_rand_num(pbcC_lo, pbcC_hi, 0.05) for _ in range(3)]
        pbcS_lo, pbcS_hi = p["pbcS_range"]
        self.pbcS = [_rand_num(pbcS_lo, pbcS_hi, 0.05) for _ in range(3)]

        ep_lo, ep_hi = p["ene_pat_range"]
        self.ene_pat = [_rand_num(ep_lo, ep_hi, 0.05) for _ in range(3)]

        self.erI = [_rand_er(s) for s in p["erI_ranges"]]
        self.erC = [_rand_er(s) for s in p["erC_ranges"]]
        self.erS = [_rand_er(s) for s in p["erS_ranges"]]

        # --- gender ---
        ge = random.uniform(0, 100)
        self.gen = 1 if ge < p["gender_thresh"] else 2

        # --- age ---
        sa = random.uniform(0, 100)
        self.age = _categorical(sa, [(ub, v) for ub, v in p["age_breaks"]])

        # --- economic comfort ---
        ec = random.uniform(0, 100)
        self.ecom = _categorical(ec, [(ub, v) for ub, v in p["ecom_breaks"]])

        # --- education ---
        ed_r = random.uniform(0, 100)
        self.edu = _categorical(ed_r, [(ub, v) for ub, v in p["edu_breaks"]])

        # --- dwelling status (owner=1, renter=2) ---
        ow = random.uniform(0, 100)
        self.dw_st = 1 if ow < p["owner_thresh"] else 2

        # --- dwelling type ---
        ty = random.uniform(0, 100)
        self.dw_type = 1 if ty < p["dtype_thresh"] else 2

        # --- dwelling age ---
        dag = random.uniform(0, 100)
        self.dw_age = _categorical(dag, [(ub, v) for ub, v in p["dwage_breaks"]])

        # --- dwelling size ---
        dsi = random.uniform(0, 100)
        self.dw_size = _categorical(dsi, [(ub, v) for ub, v in p["dwsize_breaks"]])

        # --- energy label ---
        en = random.uniform(0, 100)
        self.dw_elab = _categorical(en, [(ub, v) for ub, v in p["elab_breaks"]])

        # --- energy provider ---
        prov = random.uniform(0, 100)
        self.ene_prov = 2 if prov < p["prov_thresh"] else 1
