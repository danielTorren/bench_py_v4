"""
All empirical parameters and initialization distributions for BENCH v4.
Transcribed directly from NetLogo BENCH_v04_B-NLD.ESP.nlogox.

Encoding convention
-------------------
case_study : "ES" | "NL"
Income groups : 1-7  (groups 5-7 share h.group=5 in the go loop but keep their
                       distinct distributions here so stats can be tracked)

Behavioral Likert variables are on 1-7 scale.
dw.elab : 1-6  (A=1, F=6)  -- lower = more efficient
dw.age  : 1=new (<10 yr), 2=middle (11-35 yr), 3=old (>35 yr)
dw.type : 1=apartment, 2=house
dw.st   : 1=owner, 2=renter
ene.prov: 1=grey, 2=brown, 3=green
"""

# ---------------------------------------------------------------------------
# Utility function coefficients (insulation/renovation, from NetLogo utility)
# U1 = edu*a + age*b + dw_elab*c + dw_type*d + dw_age*e + dw_size*f + gas*g + pn1*h + erI1
# ---------------------------------------------------------------------------
UTILITY_COEF = {
    "edu":    0.0563284,
    "age":    0.0008106,
    "dw_elab": -0.0769971,
    "dw_type": 0.4265,
    "dw_age":  0.0883428,
    "dw_size": 0.0857047,
    "gas":     0.0000488,
    "pn1":     0.052849,
}

# ---------------------------------------------------------------------------
# Motivation thresholds (knowledge/guilt, personal norm, social norm)
# ---------------------------------------------------------------------------
GUILT_THRESH = {"NL": 4.6, "ES": 5.2}

# pn1 and sn1 thresholds for motivation state m1 (investment)
MOTIVATION_THRESH = {
    "NL": {"m1": (4.7, 3.5), "m2": (4.8, 3.6), "m3": (4.8, 3.7)},
    "ES": {"m1": (5.67, 4.77), "m2": (5.40, 4.45), "m3": (5.78, 5.05)},
}

# ---------------------------------------------------------------------------
# Consideration thresholds for pbcI1 (investment PBC)
# ---------------------------------------------------------------------------
PBC_INVEST_THRESH = {"NL": 1.0, "ES": 2.2}
PBC_CONSERV_THRESH = 1.0   # both countries
PBC_SWITCH_THRESH = {"NL": 1.0, "ES": 3.5}

# ---------------------------------------------------------------------------
# Learning parameters
# ---------------------------------------------------------------------------
LEARNING_RATE = 0.05      # 5% update per step
LEARNING_CAP  = 6.6       # upper cap for behavioral attributes
SLOW_NEIGHBOR_MIN = 4     # "count link-neighbors > 4" threshold in Slow dynamics

# ---------------------------------------------------------------------------
# Memory recall probabilities (% who had renovated before 2016)
# ---------------------------------------------------------------------------
RECALL_PROB = {
    "ES": {1: 2.3, 2: 1.7, 3: 2.9, 4: 3.0, 5: 2.5},   # group 5 covers 5-7
    "NL": {1: 1.8, 2: 1.4, 3: 1.5, 4: 3.6, 5: 1.2},
}

# ---------------------------------------------------------------------------
# Investment costs (EUR)
# ---------------------------------------------------------------------------
I1_COST = 3000.0    # insulation
I2_COST = 4000.0    # installation
I3_COST = 300.0     # appliances

# ---------------------------------------------------------------------------
# Renovation cooldown periods by dwelling age category
# (invest1 is reset to False after this many years)
# ---------------------------------------------------------------------------
COOLDOWN_BY_DWAGE = {1: 15, 2: 7, 3: 2}

# ---------------------------------------------------------------------------
# Gas savings from renovation
# ---------------------------------------------------------------------------
GAS_SAVE_FRACTION = 0.20   # 20% of current gas consumption

# ---------------------------------------------------------------------------
# NetLogo grid dimensions (for spatial neighbourhood)
# ---------------------------------------------------------------------------
GRID_MIN = -44
GRID_MAX =  44   # inclusive

# ---------------------------------------------------------------------------
# Dwelling age update distributions from MESSAGEix-Buildings
# (probabilistic reassignment from 2025 onwards, 5-yr windows)
# Format: { case_study: { year_range: (p_new, p_mid) } }
#   dw.age=1 if r < p_new
#   dw.age=2 if p_new <= r < p_new+p_mid
#   dw.age=3 otherwise
# ---------------------------------------------------------------------------
DWAGE_UPDATE = {
    "NL": {
        (2025, 2030): (51, 83),
        (2030, 2035): (51, 83),
        (2035, 2040): (53, 83),
        (2040, 2045): (57, 84),   # note: NL has duplicate block 2040-2045
        (2045, 2050): (59, 84),
    },
    "ES": {
        (2025, 2030): (49, 80),
        (2030, 2035): (49, 84),
        (2035, 2040): (49, 84),
        (2040, 2045): (53, 83),   # second duplicate block
        (2045, 2050): (57, 84),
    },
}

# ---------------------------------------------------------------------------
# Empirical initialization distributions
# Each entry: (cumulative_upper_bound, group_id, params_dict)
# rn is drawn in [0, 100); a household belongs to group where rn < cum_upper.
#
# params_dict keys:
#   income_range   : (low, high, step)
#   gas_range      : (low, high, step)
#   know_range     : (lo, hi)
#   cee_aw_range, ed_aw_range, pn_range, sn_range,
#   pbcI_range, pbcC_range, pbcS_range, ene_pat_range
#   erI_ranges, erC_ranges, erS_ranges  : lists of (lo, hi) per sub-action
#   gender_thresh  : P(gen=1)  (female)
#   age_breaks     : list of (cum_pct, age_val) — final bin has no upper limit
#   ecom_breaks    : list of (cum_pct, ecom_val)
#   edu_breaks     : list of (cum_pct, edu_val)
#   owner_thresh   : P(dw.st=1) owner
#   dtype_thresh   : P(dw.type=1) apartment
#   dwage_breaks   : list of (cum_pct, dw_age_val)
#   dwsize_breaks  : list of (cum_pct, dw_size_val)
#   elab_breaks    : list of (cum_pct, dw_elab_val)
#   prov_thresh    : P(ene.prov=2) brown  (else grey=1)
# ---------------------------------------------------------------------------

ES_GROUPS = [
    # (rn < 11.4) -> group 1
    (11.4, 1, {
        "income_range": (800, 10000, 100),
        "gas_range":    (0, 5000, 100),
        "know_range":   (3, 7),     "cee_aw_range": (3, 7),    "ed_aw_range":  (2.4, 7),
        "pn_range":     (2.87, 7),  "sn_range":     (1, 7),
        "pbcI_range":   (1, 7),     "pbcC_range":   (1, 7),    "pbcS_range":   (1, 7),
        "ene_pat_range":(1, 2.5),
        "erI_ranges":   [(-0.03, 0.01), (-0.01, 0.00), (-0.04, -0.02)],
        "erC_ranges":   [(-0.05, -0.05), (-0.05, -0.03), (-0.05, -0.03)],
        "erS_ranges":   [(-0.01, 0.00), None, (-0.02, -0.01)],   # None -> fixed -0.01
        "gender_thresh": 71,
        "age_breaks":   [(6, 1), (64, 2), (95, 3), (100, 4)],
        "ecom_breaks":  [(73, 1), (96, 2), (100, 3)],
        "edu_breaks":   [(36, 1), (58, 2), (100, 3)],
        "owner_thresh": 37,
        "dtype_thresh": 83,
        "dwage_breaks": [(22, 1), (69, 2), (100, 3)],
        "dwsize_breaks":[(71, 1), (93, 2), (100, 3)],
        "elab_breaks":  [(21, 1), (63.1, 2), (78.9, 3), (94.7, 4), (100, 5)],
        "prov_thresh":  3.49,
    }),
    # (rn >= 11.4 and < 58.2) -> group 2
    (58.2, 2, {
        "income_range": (10000, 30000, 100),
        "gas_range":    (0, 5000, 100),
        "know_range":   (1, 7),     "cee_aw_range": (2, 7),    "ed_aw_range":  (1, 7),
        "pn_range":     (2.25, 7),  "sn_range":     (1, 7),
        "pbcI_range":   (1, 7),     "pbcC_range":   (1, 7),    "pbcS_range":   (1, 7),
        "ene_pat_range":(1, 2.5),
        "erI_ranges":   [(-0.03, -0.01), (-0.01, 0.00), (-0.04, -0.02)],
        "erC_ranges":   [(-0.05, -0.03), (-0.05, -0.03), (-0.05, -0.03)],
        "erS_ranges":   [(-0.01, 0.00), None, (-0.02, -0.01)],
        "gender_thresh": 63,
        "age_breaks":   [(3, 1), (48, 2), (93, 3), (100, 4)],
        "ecom_breaks":  [(24, 1), (75, 2), (100, 3)],
        "edu_breaks":   [(19, 1), (46, 2), (100, 3)],
        "owner_thresh": 78,
        "dtype_thresh": 81,
        "dwage_breaks": [(30, 1), (76, 2), (100, 3)],
        "dwsize_breaks":[(72, 1), (91, 2), (100, 3)],
        "elab_breaks":  [(32.3, 1), (64.6, 2), (83.1, 3), (90.8, 4), (97, 5), (100, 6)],
        "prov_thresh":  4.25,
    }),
    # (rn >= 58.2 and < 86) -> group 3
    (86.0, 3, {
        "income_range": (30001, 50000, 100),
        "gas_range":    (0, 5000, 100),
        "know_range":   (1, 7),     "cee_aw_range": (1, 7),    "ed_aw_range":  (1, 7),
        "pn_range":     (2, 7),     "sn_range":     (1, 7),
        "pbcI_range":   (2.2, 7),   "pbcC_range":   (1, 7),    "pbcS_range":   (1, 7),
        "ene_pat_range":(1, 3),
        "erI_ranges":   [(-0.03, -0.01), (-0.01, 0.00), (-0.04, -0.02)],
        "erC_ranges":   [(-0.05, -0.03), (-0.05, -0.02), None],   # None -> fixed -0.01
        "erS_ranges":   [(-0.302, -0.01), (-0.02, -0.01), (-0.02, -0.01)],
        "gender_thresh": 46,
        "age_breaks":   [(2, 1), (46, 2), (91, 3), (100, 4)],
        "ecom_breaks":  [(28, 1), (75, 2), (100, 3)],
        "edu_breaks":   [(10, 1), (31, 2), (100, 3)],
        "owner_thresh": 85,
        "dtype_thresh": 76,
        "dwage_breaks": [(29, 1), (78, 2), (100, 3)],
        "dwsize_breaks":[(58, 1), (86, 2), (100, 3)],
        "elab_breaks":  [(32, 1), (57.5, 2), (76.6, 3), (83, 4), (93.6, 5), (100, 6)],
        "prov_thresh":  5.71,
    }),
    # (rn >= 86 and < 94.7) -> group 4
    (94.7, 4, {
        "income_range": (50001, 70000, 100),
        "gas_range":    (0, 5000, 100),
        "know_range":   (3.66, 6.33), "cee_aw_range": (2.92, 7),  "ed_aw_range": (2.33, 7),
        "pn_range":     (2, 7),       "sn_range":     (1, 7),
        "pbcI_range":   (2.6, 7),     "pbcC_range":   (1, 7),    "pbcS_range":  (1, 7),
        "ene_pat_range":(1, 2),
        "erI_ranges":   [(-0.03, -0.02), (-0.02, 0.00), (-0.04, -0.02)],
        "erC_ranges":   [(-0.05, -0.03), (-0.06, -0.02), (-0.05, -0.02)],
        "erS_ranges":   [None, None, (-0.02, -0.01)],   # None -> fixed -0.01
        "gender_thresh": 50,
        "age_breaks":   [(2, 1), (47, 2), (92, 3), (100, 4)],
        "ecom_breaks":  [(8, 1), (49, 2), (100, 3)],
        "edu_breaks":   [(5, 1), (18, 2), (100, 3)],
        "owner_thresh": 94,
        "dtype_thresh": 65,
        "dwage_breaks": [(33, 1), (80, 2), (100, 3)],
        "dwsize_breaks":[(59, 1), (76, 2), (100, 3)],
        "elab_breaks":  [(31.3, 1), (56.3, 2), (68.8, 3), (87.5, 4), (100, 5)],
        "prov_thresh":  3.03,
    }),
    # (rn >= 94.7 and < 97.8) -> group 5
    (97.8, 5, {
        "income_range": (70001, 90000, 100),
        "gas_range":    (0, 5000, 100),
        "know_range":   (3.83, 7),   "cee_aw_range": (3.28, 7),  "ed_aw_range": (2.8, 7),
        "pn_range":     (2.5, 7),    "sn_range":     (1, 7),
        "pbcI_range":   (1.6, 6.8),  "pbcC_range":   (1.5, 7),  "pbcS_range":  (1, 7),
        "ene_pat_range":(1, 2),
        "erI_ranges":   [(-0.03, 0.01), (-0.02, 0.00), (-0.04, -0.02)],
        "erC_ranges":   [(-0.05, -0.03), (-0.06, -0.02), (-0.05, -0.03)],
        "erS_ranges":   [None, None, (-0.02, -0.01)],
        "gender_thresh": 39,
        "age_breaks":   [(52, 2), (96, 3), (100, 4)],    # no age=1
        "ecom_breaks":  [(7, 1), (84, 2), (100, 3)],
        "edu_breaks":   [(10, 2), (100, 3)],              # no edu=1
        "owner_thresh": 96,
        "dtype_thresh": 74,
        "dwage_breaks": [(35, 1), (87, 2), (100, 3)],
        "dwsize_breaks":[(57, 1), (83, 2), (100, 3)],
        "elab_breaks":  [(33.3, 2), (100, 3)],            # no elab=1
        "prov_thresh":  13.4,
    }),
    # (rn >= 97.8 and < 98.7) -> group 6 (mapped to h.group=5 in go loop)
    (98.7, 6, {
        "income_range": (90001, 110000, 100),
        "gas_range":    (0, 5000, 100),
        "know_range":   (3.83, 7),   "cee_aw_range": (3.5, 6.5),  "ed_aw_range": (3.2, 7),
        "pn_range":     (2.25, 6.75),"sn_range":     (1, 6),
        "pbcI_range":   (1, 5.6),    "pbcC_range":   (1, 5.6),   "pbcS_range":  (1, 7),
        "ene_pat_range":(1, 1.75),
        "erI_ranges":   [(-0.03, 0.01), (-0.02, 0.00), (-0.04, -0.02)],
        "erC_ranges":   [(-0.05, -0.03), (-0.06, -0.02), (-0.05, -0.03)],
        "erS_ranges":   [None, None, (-0.02, -0.01)],
        "gender_thresh": 57,
        "age_breaks":   [(28, 2), (100, 3)],   # no age=1,4
        "ecom_breaks":  [(25, 1), (50, 2), (100, 3)],
        "edu_breaks":   [(100, 3)],             # all edu=3
        "owner_thresh": 86,
        "dtype_thresh": 86,
        "dwage_breaks": [(71, 2), (100, 3)],   # no dw.age=1
        "dwsize_breaks":[(14, 1), (71, 2), (100, 3)],
        "elab_breaks":  [(66.7, 2), (100, 6)],
        "prov_thresh":  0.0,   # ene.prov = 1 always
    }),
    # (rn >= 98.7 and <= 100) -> group 7 (mapped to h.group=5 in go loop)
    (100.0, 7, {
        "income_range": (110001, 150000, 100),
        "gas_range":    (0, 5000, 100),
        "know_range":   (4, 6.33),   "cee_aw_range": (4.57, 6.28),  "ed_aw_range": (3.5, 6.33),
        "pn_range":     (4.42, 7),   "sn_range":     (3.33, 6.2),
        "pbcI_range":   (3.6, 7),    "pbcC_range":   (1, 6.5),      "pbcS_range":  (3, 6),
        "ene_pat_range":(1, 3),
        "erI_ranges":   [(-0.03, 0.01), (-0.02, 0.00), (-0.04, -0.02)],
        "erC_ranges":   [(-0.05, -0.03), (-0.06, -0.02), (-0.05, -0.03)],
        "erS_ranges":   [None, None, (-0.02, -0.01)],
        "gender_thresh": 40,
        "age_breaks":   [(20, 2), (70, 3), (100, 4)],
        "ecom_breaks":  [(17, 1), (100, 3)],   # no ecom=2
        "edu_breaks":   [(20, 2), (100, 3)],
        "owner_thresh": 80,
        "dtype_thresh": 50,
        "dwage_breaks": [(30, 1), (80, 2), (100, 3)],
        "dwsize_breaks":[(30, 1), (80, 2), (100, 3)],
        "elab_breaks":  [(50, 2), (100, 6)],
        "prov_thresh":  0.0,   # ene.prov = 1 always
    }),
]

# ---------------------------------------------------------------------------
NL_GROUPS = [
    # (rn < 5.5) -> group 1
    (5.5, 1, {
        "income_range": (800, 10000, 100),
        "gas_range":    (0, 5000, 100),
        "know_range":   (1, 7),       "cee_aw_range": (1.2, 6.8),  "ed_aw_range": (1, 7),
        "pn_range":     (1, 6.5),     "sn_range":     (1, 6.5),
        "pbcI_range":   (1, 7),       "pbcC_range":   (1, 7),      "pbcS_range":  (1, 7),
        "ene_pat_range":(0.25, 2.22),
        "erI_ranges":   [(-0.03, -0.01), (-0.02, -0.01), (-0.04, -0.02)],
        "erC_ranges":   [(-0.04, -0.03), (-0.06, -0.03), (-0.04, -0.02)],
        "erS_ranges":   [(-0.01, 0.00), (-0.02, -0.01), None],  # None -> fixed -0.01
        "gender_thresh": 66.7,
        "age_breaks":   [(1.8, 1), (35.1, 2), (78.9, 3), (100, 4)],
        "ecom_breaks":  [(43.9, 1), (64.9, 2), (100, 3)],
        "edu_breaks":   [(57.9, 1), (82.5, 2), (100, 3)],
        "owner_thresh": 49,
        "dtype_thresh": 28.1,
        "dwage_breaks": [(14, 1), (63.2, 2), (100, 3)],
        "dwsize_breaks":[(47.1, 1), (80.7, 2), (100, 3)],
        "elab_breaks":  [(25, 1), (34.4, 2), (40.6, 3), (43.7, 4), (100, 5)],
        "prov_thresh":  24.57,
    }),
    # (rn >= 5.5 and <= 40.19) -> group 2
    (40.19, 2, {
        "income_range": (10000, 30000, 100),
        "gas_range":    (0, 5000, 100),
        "know_range":   (1, 7),       "cee_aw_range": (1, 6.8),    "ed_aw_range": (1.3, 7),
        "pn_range":     (2, 7),       "sn_range":     (1, 7),
        "pbcI_range":   (1.4, 7),     "pbcC_range":   (1, 7),      "pbcS_range":  (1, 7),
        "ene_pat_range":(0.25, 1.7),
        "erI_ranges":   [(-0.03, -0.01), (-0.02, -0.01), (-0.04, -0.02)],
        "erC_ranges":   [(-0.05, -0.02), (-0.06, -0.03), (-0.04, -0.01)],
        "erS_ranges":   [(-0.01, 0.00), (-0.02, -0.01), None],
        "gender_thresh": 54.6,
        "age_breaks":   [(18.4, 2), (57.4, 3), (100, 4)],   # no age=1
        "ecom_breaks":  [(18.9, 1), (41.5, 2), (100, 3)],
        "edu_breaks":   [(72.1, 1), (84.6, 2), (100, 3)],
        "owner_thresh": 52,
        "dtype_thresh": 21.2,
        "dwage_breaks": [(8.6, 1), (42.6, 2), (100, 3)],
        "dwsize_breaks":[(54.3, 1), (83.6, 2), (100, 3)],
        "elab_breaks":  [(8.5, 1), (22.6, 2), (35, 3), (39, 4), (100, 5)],
        "prov_thresh":  42.33,
    }),
    # (rn > 40.19 and <= 78.17) -> group 3
    (78.17, 3, {
        "income_range": (30001, 50000, 100),
        "gas_range":    (0, 5000, 100),
        "know_range":   (2.6, 7),     "cee_aw_range": (2.71, 7),  "ed_aw_range": (1.4, 7),
        "pn_range":     (1.5, 7),     "sn_range":     (1, 7),
        "pbcI_range":   (2.2, 7),     "pbcC_range":   (1, 7),     "pbcS_range":  (1, 7),
        "ene_pat_range":(0, 1.6),
        "erI_ranges":   [(-0.03, -0.01), (-0.02, 0.00), (-0.05, -0.02)],
        "erC_ranges":   [(-0.05, -0.02), (-0.06, -0.03), (-0.04, -0.01)],
        "erS_ranges":   [None, (-0.02, -0.01), None],
        "gender_thresh": 39.7,
        "age_breaks":   [(19.6, 2), (64.4, 3), (100, 4)],
        "ecom_breaks":  [(7.4, 1), (30.3, 2), (100, 3)],
        "edu_breaks":   [(49.9, 1), (74.8, 2), (100, 3)],
        "owner_thresh": 80,
        "dtype_thresh": 87.3,
        "dwage_breaks": [(11.2, 1), (52.9, 2), (100, 3)],
        "dwsize_breaks":[(36.9, 1), (77.6, 2), (100, 3)],
        "elab_breaks":  [(16.4, 1), (33.4, 2), (43.3, 3), (47.3, 4), (100, 5)],
        "prov_thresh":  36.13,
    }),
    # (rn > 78.17 and <= 91.69) -> group 4
    (91.69, 4, {
        "income_range": (50001, 70000, 100),
        "gas_range":    (0, 5000, 100),
        "know_range":   (2.8, 6.6),   "cee_aw_range": (3, 6.6),   "ed_aw_range": (1.8, 7),
        "pn_range":     (2.5, 7),     "sn_range":     (1.1, 7),
        "pbcI_range":   (2.2, 7),     "pbcC_range":   (1, 7),     "pbcS_range":  (1, 7),
        "ene_pat_range":(0.6, 1.6),
        "erI_ranges":   [(-0.03, -0.01), (-0.02, 0.00), (-0.05, -0.02)],
        "erC_ranges":   [(-0.05, -0.02), (-0.06, -0.03), (-0.04, -0.01)],
        "erS_ranges":   [None, (-0.02, -0.01), None],
        "gender_thresh": 37.1,
        "age_breaks":   [(26.4, 2), (79.3, 3), (100, 4)],
        "ecom_breaks":  [(2.9, 1), (44.3, 2), (100, 3)],
        "edu_breaks":   [(57.2, 1), (79.3, 2), (100, 3)],
        "owner_thresh": 92,
        "dtype_thresh": 5.1,
        "dwage_breaks": [(20, 1), (71.4, 2), (100, 3)],
        "dwsize_breaks":[(23.6, 1), (61.4, 2), (100, 3)],
        "elab_breaks":  [(22.9, 1), (42.9, 2), (60, 3), (67.1, 4), (100, 5)],
        "prov_thresh":  46.43,
    }),
    # (rn > 91.69 and <= 97.39) -> group 5
    (97.39, 5, {
        "income_range": (70001, 90000, 100),
        "gas_range":    (0, 5000, 100),
        "know_range":   (3, 5.66),    "cee_aw_range": (3.14, 6.42),  "ed_aw_range": (2, 7),
        "pn_range":     (3.5, 6.25),  "sn_range":     (1, 5.8),
        "pbcI_range":   (2.2, 7),     "pbcC_range":   (1, 6.5),      "pbcS_range":  (1, 7),
        "ene_pat_range":(0.70, 1.60),
        "erI_ranges":   [(-0.03, -0.01), (-0.02, 0.00), (-0.04, -0.02)],
        "erC_ranges":   [(-0.04, -0.03), (-0.06, -0.03), (-0.03, -0.01)],
        "erS_ranges":   [None, (-0.02, -0.01), None],
        "gender_thresh": 39,
        "age_breaks":   [(25.4, 2), (84.7, 3), (100, 4)],
        "ecom_breaks":  [(22.2, 2), (100, 3)],   # no ecom=1
        "edu_breaks":   [(57.2, 1), (79.3, 2), (100, 3)],
        "owner_thresh": 92,
        "dtype_thresh": 6.8,
        "dwage_breaks": [(10.2, 1), (96, 2), (100, 3)],
        "dwsize_breaks":[(15.3, 1), (54, 2), (100, 3)],
        "elab_breaks":  [(25.9, 1), (40.7, 2), (51.8, 3), (55.5, 4), (100, 5)],
        "prov_thresh":  44.07,
    }),
    # (rn > 97.39 and <= 98.36) -> group 6
    (98.36, 6, {
        "income_range": (90001, 110000, 100),
        "gas_range":    (0, 3000, 100),
        "know_range":   (3.1, 4.66),  "cee_aw_range": (2.72, 5.57),  "ed_aw_range": (3.66, 6.33),
        "pn_range":     (3.62, 5.75), "sn_range":     (2, 3.83),
        "pbcI_range":   (2.4, 6.4),   "pbcC_range":   (1, 5.6),      "pbcS_range":  (1.5, 5.5),
        "ene_pat_range":(0.78, 1.82),
        "erI_ranges":   [(-0.03, -0.01), (-0.02, 0.00), (-0.04, -0.02)],
        "erC_ranges":   [(-0.04, -0.03), (-0.06, -0.03), (-0.03, -0.01)],
        "erS_ranges":   [None, (-0.02, -0.01), None],
        "gender_thresh": 50,
        "age_breaks":   [(40, 2), (90, 3), (100, 4)],
        "ecom_breaks":  [(33.3, 2), (100, 3)],
        "edu_breaks":   [(10, 1), (20, 2), (100, 3)],
        "owner_thresh": 20,
        "dtype_thresh": 0.0,   # dw.type always 2
        "dwage_breaks": [(40, 1), (60, 2), (100, 3)],
        "dwsize_breaks":[(14, 1), (71, 2), (100, 3)],
        "elab_breaks":  [(100, 5)],   # all elab=5
        "prov_thresh":  20,
    }),
    # (rn > 98.36 and <= 100) -> group 7
    (100.0, 7, {
        "income_range": (110001, 150000, 100),
        "gas_range":    (0, 3000, 100),
        "know_range":   (3.33, 5.33),  "cee_aw_range": (3.85, 6.21),  "ed_aw_range": (2.66, 6),
        "pn_range":     (2.62, 7),     "sn_range":     (1, 5.66),
        "pbcI_range":   (1.6, 6.6),    "pbcC_range":   (1, 7),         "pbcS_range":  (3, 6),
        "ene_pat_range":(1, 3),
        "erI_ranges":   [(-0.03, -0.01), (-0.02, 0.00), (-0.04, -0.02)],
        "erC_ranges":   [(-0.04, -0.03), (-0.06, -0.03), (-0.03, -0.01)],
        "erS_ranges":   [None, (-0.02, -0.01), None],
        "gender_thresh": 37.1,
        "age_breaks":   [(5.9, 2), (70.6, 3), (100, 4)],
        "ecom_breaks":  [(2.9, 1), (44.3, 2), (100, 3)],
        "edu_breaks":   [(29.4, 1), (64.7, 2), (100, 3)],
        "owner_thresh": 80,
        "dtype_thresh": 5.7,
        "dwage_breaks": [(20, 1), (71.4, 2), (100, 3)],
        "dwsize_breaks":[(23.6, 1), (61.4, 2), (100, 3)],
        "elab_breaks":  [(20, 1), (50, 2), (70, 3), (90, 4), (100, 5)],
        "prov_thresh":  29.41,
    }),
]

# Number of households per case study
N_HOUSEHOLDS = {"ES": 793, "NL": 759}

# CGE data file paths (relative to netlogo/ folder)
CGE_FILES = {
    "ES": "cge-es-ssp2-h.csv",
    "NL": "cge-nl-ssp2-h.csv",
}

# Simulation year range
START_YEAR = 2016
END_YEAR   = 2050
