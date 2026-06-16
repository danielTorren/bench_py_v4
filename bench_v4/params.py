"""
All empirical parameters and initialization distributions for BENCH v4.
Transcribed directly from NetLogo BENCH_v04_B-NLD.ESP.nlogox.

Encoding convention
-------------------
case_study : "ES" | "NL"
Income groups : 1-7  (groups 5-7 share h.group=5 in the go loop but keep
                       their distinct distributions for stats)
Behavioral Likert variables : 1-7 scale
dw.elab : 1-6  (A=1, F=6) — lower is more efficient
dw.age  : 1=new (<10 yr), 2=middle (11-35 yr), 3=old (>35 yr)
dw.type : 1=apartment, 2=house
dw.st   : 1=owner, 2=renter
"""

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Agent group initialization distributions
# ---------------------------------------------------------------------------

@dataclass
class GroupParams:
    """Empirical initialization distributions for one income group."""
    cum_upper:     float                              # cumulative population share (0–100)
    group_id:      int                                # group label (1–7)
    income_range:  tuple[float, float, float]         # (lo, hi, step)
    gas_range:     tuple[float, float, float]
    know_range:    tuple[float, float]                # step = 0.05
    cee_aw_range:  tuple[float, float]
    ed_aw_range:   tuple[float, float]
    pn_range:      tuple[float, float]
    sn_range:      tuple[float, float]
    pbcI_range:    tuple[float, float]
    pbcC_range:    tuple[float, float]
    pbcS_range:    tuple[float, float]
    ene_pat_range: tuple[float, float]
    erI_ranges:    list[tuple[float, float] | None]   # per sub-action; None → fixed −0.01
    erC_ranges:    list[tuple[float, float] | None]
    erS_ranges:    list[tuple[float, float] | None]
    gender_thresh: float                              # P(female)
    age_breaks:    list[tuple[float, int]]            # [(cum_pct, value), ...]
    ecom_breaks:   list[tuple[float, int]]
    edu_breaks:    list[tuple[float, int]]
    owner_thresh:  float                              # P(owner)
    dtype_thresh:  float                              # P(apartment)
    dwage_breaks:  list[tuple[float, int]]
    dwsize_breaks: list[tuple[float, int]]
    elab_breaks:   list[tuple[float, int]]
    prov_thresh:   float                              # P(brown energy provider)


# ---------------------------------------------------------------------------
# Utility function coefficients
# U1 = edu*a + age*b + dw_elab*c + dw_type*d + dw_age*e + dw_size*f + gas*g + pn1*h + erI1
# ---------------------------------------------------------------------------
UTILITY_COEF: dict[str, float] = {#These coefficents are from the probit regression of the survey data, can be found in paper: Demand-side solutions for climate mitigation: Bottom-up drivers of household energy behavior change in the Netherlands and Spain (Energy Research & Social Science, 2020)
    "edu":     0.0563284,
    "age":     0.0008106,
    "dw_elab": -0.0769971,
    "dw_type": 0.4265,
    "dw_age":  0.0883428,
    "dw_size": 0.0857047,
    "gas":     0.0000488,
    "pn1":     0.052849,
}

# ---------------------------------------------------------------------------
# Behavioral thresholds
# ---------------------------------------------------------------------------
GUILT_THRESH = {"NL": 4.6, "ES": 5.2}

MOTIVATION_THRESH = {
    "NL": {"m1": (4.7, 3.5), "m2": (4.8, 3.6), "m3": (4.8, 3.7)},
    "ES": {"m1": (5.67, 4.77), "m2": (5.40, 4.45), "m3": (5.78, 5.05)},
}

PBC_INVEST_THRESH  = {"NL": 1.0, "ES": 2.2}
PBC_CONSERV_THRESH = 1.0
PBC_SWITCH_THRESH  = {"NL": 1.0, "ES": 3.5}

# ---------------------------------------------------------------------------
# Learning parameters
# ---------------------------------------------------------------------------
LEARNING_RATE     = 0.05
LEARNING_CAP      = 6.6
SLOW_NEIGHBOR_MIN = 4

# ---------------------------------------------------------------------------
# Memory recall probabilities (% who had renovated before 2016)
# ---------------------------------------------------------------------------
RECALL_PROB = {
    "ES": {1: 2.3, 2: 1.7, 3: 2.9, 4: 3.0, 5: 2.5},
    "NL": {1: 1.8, 2: 1.4, 3: 1.5, 4: 3.6, 5: 1.2},
}

# ---------------------------------------------------------------------------
# Investment costs (EUR)
# ---------------------------------------------------------------------------
I1_COST = 3000.0    # insulation
I2_COST = 4000.0    # installation  (future: conservation)
I3_COST = 300.0     # appliances    (future: switching)

# ---------------------------------------------------------------------------
# Renovation cooldown periods by dwelling age category
# ---------------------------------------------------------------------------
COOLDOWN_BY_DWAGE = {1: 15, 2: 7, 3: 2}

# ---------------------------------------------------------------------------
# Gas savings from renovation
# ---------------------------------------------------------------------------
GAS_SAVE_FRACTION = 0.20

# ---------------------------------------------------------------------------
# Grid half-width for spatial neighbourhood (original NetLogo: ±44 patches)
# ---------------------------------------------------------------------------
GRID_HALF = 44

# ---------------------------------------------------------------------------
# Dwelling age update distributions (MESSAGEix-Buildings, 2025 onwards)
# { year_range: (p_new, p_mid) }
#   dw.age=1 if r < p_new  |  dw.age=2 if r < p_new+p_mid  |  dw.age=3 otherwise
# ---------------------------------------------------------------------------
DWAGE_UPDATE = {
    "NL": {
        (2025, 2030): (51, 83),
        (2030, 2035): (51, 83),
        (2035, 2040): (53, 83),
        (2040, 2045): (57, 84),
        (2045, 2050): (59, 84),
    },
    "ES": {
        (2025, 2030): (49, 80),
        (2030, 2035): (49, 84),
        (2035, 2040): (49, 84),
        (2040, 2045): (53, 83),
        (2045, 2050): (57, 84),
    },
}

# ---------------------------------------------------------------------------
# ES income groups  (cumulative population share; last entry = 100.0)
# ---------------------------------------------------------------------------
ES_GROUPS: list[GroupParams] = [
    GroupParams(
        cum_upper=11.4, group_id=1,
        income_range=(800, 10000, 100),    gas_range=(0, 5000, 100),
        know_range=(3, 7),     cee_aw_range=(3, 7),     ed_aw_range=(2.4, 7),
        pn_range=(2.87, 7),    sn_range=(1, 7),
        pbcI_range=(1, 7),     pbcC_range=(1, 7),       pbcS_range=(1, 7),
        ene_pat_range=(1, 2.5),
        erI_ranges=[(-0.03, 0.01),  (-0.01, 0.00),  (-0.04, -0.02)],
        erC_ranges=[(-0.05, -0.05), (-0.05, -0.03), (-0.05, -0.03)],
        erS_ranges=[(-0.01, 0.00),  None,            (-0.02, -0.01)],
        gender_thresh=71,
        age_breaks=[(6, 1), (64, 2), (95, 3), (100, 4)],
        ecom_breaks=[(73, 1), (96, 2), (100, 3)],
        edu_breaks=[(36, 1), (58, 2), (100, 3)],
        owner_thresh=37,  dtype_thresh=83,
        dwage_breaks=[(22, 1), (69, 2), (100, 3)],
        dwsize_breaks=[(71, 1), (93, 2), (100, 3)],
        elab_breaks=[(21, 1), (63.1, 2), (78.9, 3), (94.7, 4), (100, 5)],
        prov_thresh=3.49,
    ),
    GroupParams(
        cum_upper=58.2, group_id=2,
        income_range=(10000, 30000, 100),  gas_range=(0, 5000, 100),
        know_range=(1, 7),     cee_aw_range=(2, 7),     ed_aw_range=(1, 7),
        pn_range=(2.25, 7),    sn_range=(1, 7),
        pbcI_range=(1, 7),     pbcC_range=(1, 7),       pbcS_range=(1, 7),
        ene_pat_range=(1, 2.5),
        erI_ranges=[(-0.03, -0.01), (-0.01, 0.00),  (-0.04, -0.02)],
        erC_ranges=[(-0.05, -0.03), (-0.05, -0.03), (-0.05, -0.03)],
        erS_ranges=[(-0.01, 0.00),  None,            (-0.02, -0.01)],
        gender_thresh=63,
        age_breaks=[(3, 1), (48, 2), (93, 3), (100, 4)],
        ecom_breaks=[(24, 1), (75, 2), (100, 3)],
        edu_breaks=[(19, 1), (46, 2), (100, 3)],
        owner_thresh=78,  dtype_thresh=81,
        dwage_breaks=[(30, 1), (76, 2), (100, 3)],
        dwsize_breaks=[(72, 1), (91, 2), (100, 3)],
        elab_breaks=[(32.3, 1), (64.6, 2), (83.1, 3), (90.8, 4), (97, 5), (100, 6)],
        prov_thresh=4.25,
    ),
    GroupParams(
        cum_upper=86.0, group_id=3,
        income_range=(30001, 50000, 100),  gas_range=(0, 5000, 100),
        know_range=(1, 7),     cee_aw_range=(1, 7),     ed_aw_range=(1, 7),
        pn_range=(2, 7),       sn_range=(1, 7),
        pbcI_range=(2.2, 7),   pbcC_range=(1, 7),       pbcS_range=(1, 7),
        ene_pat_range=(1, 3),
        erI_ranges=[(-0.03, -0.01), (-0.01, 0.00),  (-0.04, -0.02)],
        erC_ranges=[(-0.05, -0.03), (-0.05, -0.02), None],
        erS_ranges=[(-0.302, -0.01), (-0.02, -0.01), (-0.02, -0.01)],
        gender_thresh=46,
        age_breaks=[(2, 1), (46, 2), (91, 3), (100, 4)],
        ecom_breaks=[(28, 1), (75, 2), (100, 3)],
        edu_breaks=[(10, 1), (31, 2), (100, 3)],
        owner_thresh=85,  dtype_thresh=76,
        dwage_breaks=[(29, 1), (78, 2), (100, 3)],
        dwsize_breaks=[(58, 1), (86, 2), (100, 3)],
        elab_breaks=[(32, 1), (57.5, 2), (76.6, 3), (83, 4), (93.6, 5), (100, 6)],
        prov_thresh=5.71,
    ),
    GroupParams(
        cum_upper=94.7, group_id=4,
        income_range=(50001, 70000, 100),  gas_range=(0, 5000, 100),
        know_range=(3.66, 6.33), cee_aw_range=(2.92, 7),  ed_aw_range=(2.33, 7),
        pn_range=(2, 7),         sn_range=(1, 7),
        pbcI_range=(2.6, 7),     pbcC_range=(1, 7),       pbcS_range=(1, 7),
        ene_pat_range=(1, 2),
        erI_ranges=[(-0.03, -0.02), (-0.02, 0.00),  (-0.04, -0.02)],
        erC_ranges=[(-0.05, -0.03), (-0.06, -0.02), (-0.05, -0.02)],
        erS_ranges=[None, None, (-0.02, -0.01)],
        gender_thresh=50,
        age_breaks=[(2, 1), (47, 2), (92, 3), (100, 4)],
        ecom_breaks=[(8, 1), (49, 2), (100, 3)],
        edu_breaks=[(5, 1), (18, 2), (100, 3)],
        owner_thresh=94,  dtype_thresh=65,
        dwage_breaks=[(33, 1), (80, 2), (100, 3)],
        dwsize_breaks=[(59, 1), (76, 2), (100, 3)],
        elab_breaks=[(31.3, 1), (56.3, 2), (68.8, 3), (87.5, 4), (100, 5)],
        prov_thresh=3.03,
    ),
    GroupParams(
        cum_upper=97.8, group_id=5,
        income_range=(70001, 90000, 100),  gas_range=(0, 5000, 100),
        know_range=(3.83, 7),   cee_aw_range=(3.28, 7),  ed_aw_range=(2.8, 7),
        pn_range=(2.5, 7),      sn_range=(1, 7),
        pbcI_range=(1.6, 6.8),  pbcC_range=(1.5, 7),     pbcS_range=(1, 7),
        ene_pat_range=(1, 2),
        erI_ranges=[(-0.03, 0.01),  (-0.02, 0.00),  (-0.04, -0.02)],
        erC_ranges=[(-0.05, -0.03), (-0.06, -0.02), (-0.05, -0.03)],
        erS_ranges=[None, None, (-0.02, -0.01)],
        gender_thresh=39,
        age_breaks=[(52, 2), (96, 3), (100, 4)],
        ecom_breaks=[(7, 1), (84, 2), (100, 3)],
        edu_breaks=[(10, 2), (100, 3)],
        owner_thresh=96,  dtype_thresh=74,
        dwage_breaks=[(35, 1), (87, 2), (100, 3)],
        dwsize_breaks=[(57, 1), (83, 2), (100, 3)],
        elab_breaks=[(33.3, 2), (100, 3)],
        prov_thresh=13.4,
    ),
    GroupParams(
        cum_upper=98.7, group_id=6,
        income_range=(90001, 110000, 100), gas_range=(0, 5000, 100),
        know_range=(3.83, 7),   cee_aw_range=(3.5, 6.5),  ed_aw_range=(3.2, 7),
        pn_range=(2.25, 6.75),  sn_range=(1, 6),
        pbcI_range=(1, 5.6),    pbcC_range=(1, 5.6),      pbcS_range=(1, 7),
        ene_pat_range=(1, 1.75),
        erI_ranges=[(-0.03, 0.01),  (-0.02, 0.00),  (-0.04, -0.02)],
        erC_ranges=[(-0.05, -0.03), (-0.06, -0.02), (-0.05, -0.03)],
        erS_ranges=[None, None, (-0.02, -0.01)],
        gender_thresh=57,
        age_breaks=[(28, 2), (100, 3)],
        ecom_breaks=[(25, 1), (50, 2), (100, 3)],
        edu_breaks=[(100, 3)],
        owner_thresh=86,  dtype_thresh=86,
        dwage_breaks=[(71, 2), (100, 3)],
        dwsize_breaks=[(14, 1), (71, 2), (100, 3)],
        elab_breaks=[(66.7, 2), (100, 6)],
        prov_thresh=0.0,
    ),
    GroupParams(
        cum_upper=100.0, group_id=7,
        income_range=(110001, 150000, 100), gas_range=(0, 5000, 100),
        know_range=(4, 6.33),   cee_aw_range=(4.57, 6.28), ed_aw_range=(3.5, 6.33),
        pn_range=(4.42, 7),     sn_range=(3.33, 6.2),
        pbcI_range=(3.6, 7),    pbcC_range=(1, 6.5),       pbcS_range=(3, 6),
        ene_pat_range=(1, 3),
        erI_ranges=[(-0.03, 0.01),  (-0.02, 0.00),  (-0.04, -0.02)],
        erC_ranges=[(-0.05, -0.03), (-0.06, -0.02), (-0.05, -0.03)],
        erS_ranges=[None, None, (-0.02, -0.01)],
        gender_thresh=40,
        age_breaks=[(20, 2), (70, 3), (100, 4)],
        ecom_breaks=[(17, 1), (100, 3)],
        edu_breaks=[(20, 2), (100, 3)],
        owner_thresh=80,  dtype_thresh=50,
        dwage_breaks=[(30, 1), (80, 2), (100, 3)],
        dwsize_breaks=[(30, 1), (80, 2), (100, 3)],
        elab_breaks=[(50, 2), (100, 6)],
        prov_thresh=0.0,
    ),
]

# ---------------------------------------------------------------------------
# NL income groups  (cumulative population share; last entry = 100.0)
# ---------------------------------------------------------------------------
NL_GROUPS: list[GroupParams] = [
    GroupParams(
        cum_upper=5.5, group_id=1,
        income_range=(800, 10000, 100),    gas_range=(0, 5000, 100),
        know_range=(1, 7),      cee_aw_range=(1.2, 6.8),  ed_aw_range=(1, 7),
        pn_range=(1, 6.5),      sn_range=(1, 6.5),
        pbcI_range=(1, 7),      pbcC_range=(1, 7),        pbcS_range=(1, 7),
        ene_pat_range=(0.25, 2.22),
        erI_ranges=[(-0.03, -0.01), (-0.02, -0.01), (-0.04, -0.02)],
        erC_ranges=[(-0.04, -0.03), (-0.06, -0.03), (-0.04, -0.02)],
        erS_ranges=[(-0.01, 0.00),  (-0.02, -0.01), None],
        gender_thresh=66.7,
        age_breaks=[(1.8, 1), (35.1, 2), (78.9, 3), (100, 4)],
        ecom_breaks=[(43.9, 1), (64.9, 2), (100, 3)],
        edu_breaks=[(57.9, 1), (82.5, 2), (100, 3)],
        owner_thresh=49,  dtype_thresh=28.1,
        dwage_breaks=[(14, 1), (63.2, 2), (100, 3)],
        dwsize_breaks=[(47.1, 1), (80.7, 2), (100, 3)],
        elab_breaks=[(25, 1), (34.4, 2), (40.6, 3), (43.7, 4), (100, 5)],
        prov_thresh=24.57,
    ),
    GroupParams(
        cum_upper=40.19, group_id=2,
        income_range=(10000, 30000, 100),  gas_range=(0, 5000, 100),
        know_range=(1, 7),      cee_aw_range=(1, 6.8),    ed_aw_range=(1.3, 7),
        pn_range=(2, 7),        sn_range=(1, 7),
        pbcI_range=(1.4, 7),    pbcC_range=(1, 7),        pbcS_range=(1, 7),
        ene_pat_range=(0.25, 1.7),
        erI_ranges=[(-0.03, -0.01), (-0.02, -0.01), (-0.04, -0.02)],
        erC_ranges=[(-0.05, -0.02), (-0.06, -0.03), (-0.04, -0.01)],
        erS_ranges=[(-0.01, 0.00),  (-0.02, -0.01), None],
        gender_thresh=54.6,
        age_breaks=[(18.4, 2), (57.4, 3), (100, 4)],
        ecom_breaks=[(18.9, 1), (41.5, 2), (100, 3)],
        edu_breaks=[(72.1, 1), (84.6, 2), (100, 3)],
        owner_thresh=52,  dtype_thresh=21.2,
        dwage_breaks=[(8.6, 1), (42.6, 2), (100, 3)],
        dwsize_breaks=[(54.3, 1), (83.6, 2), (100, 3)],
        elab_breaks=[(8.5, 1), (22.6, 2), (35, 3), (39, 4), (100, 5)],
        prov_thresh=42.33,
    ),
    GroupParams(
        cum_upper=78.17, group_id=3,
        income_range=(30001, 50000, 100),  gas_range=(0, 5000, 100),
        know_range=(2.6, 7),    cee_aw_range=(2.71, 7),   ed_aw_range=(1.4, 7),
        pn_range=(1.5, 7),      sn_range=(1, 7),
        pbcI_range=(2.2, 7),    pbcC_range=(1, 7),        pbcS_range=(1, 7),
        ene_pat_range=(0, 1.6),
        erI_ranges=[(-0.03, -0.01), (-0.02, 0.00),  (-0.05, -0.02)],
        erC_ranges=[(-0.05, -0.02), (-0.06, -0.03), (-0.04, -0.01)],
        erS_ranges=[None, (-0.02, -0.01), None],
        gender_thresh=39.7,
        age_breaks=[(19.6, 2), (64.4, 3), (100, 4)],
        ecom_breaks=[(7.4, 1), (30.3, 2), (100, 3)],
        edu_breaks=[(49.9, 1), (74.8, 2), (100, 3)],
        owner_thresh=80,  dtype_thresh=87.3,
        dwage_breaks=[(11.2, 1), (52.9, 2), (100, 3)],
        dwsize_breaks=[(36.9, 1), (77.6, 2), (100, 3)],
        elab_breaks=[(16.4, 1), (33.4, 2), (43.3, 3), (47.3, 4), (100, 5)],
        prov_thresh=36.13,
    ),
    GroupParams(
        cum_upper=91.69, group_id=4,
        income_range=(50001, 70000, 100),  gas_range=(0, 5000, 100),
        know_range=(2.8, 6.6),  cee_aw_range=(3, 6.6),    ed_aw_range=(1.8, 7),
        pn_range=(2.5, 7),      sn_range=(1.1, 7),
        pbcI_range=(2.2, 7),    pbcC_range=(1, 7),        pbcS_range=(1, 7),
        ene_pat_range=(0.6, 1.6),
        erI_ranges=[(-0.03, -0.01), (-0.02, 0.00),  (-0.05, -0.02)],
        erC_ranges=[(-0.05, -0.02), (-0.06, -0.03), (-0.04, -0.01)],
        erS_ranges=[None, (-0.02, -0.01), None],
        gender_thresh=37.1,
        age_breaks=[(26.4, 2), (79.3, 3), (100, 4)],
        ecom_breaks=[(2.9, 1), (44.3, 2), (100, 3)],
        edu_breaks=[(57.2, 1), (79.3, 2), (100, 3)],
        owner_thresh=92,  dtype_thresh=5.1,
        dwage_breaks=[(20, 1), (71.4, 2), (100, 3)],
        dwsize_breaks=[(23.6, 1), (61.4, 2), (100, 3)],
        elab_breaks=[(22.9, 1), (42.9, 2), (60, 3), (67.1, 4), (100, 5)],
        prov_thresh=46.43,
    ),
    GroupParams(
        cum_upper=97.39, group_id=5,
        income_range=(70001, 90000, 100),  gas_range=(0, 5000, 100),
        know_range=(3, 5.66),   cee_aw_range=(3.14, 6.42), ed_aw_range=(2, 7),
        pn_range=(3.5, 6.25),   sn_range=(1, 5.8),
        pbcI_range=(2.2, 7),    pbcC_range=(1, 6.5),       pbcS_range=(1, 7),
        ene_pat_range=(0.70, 1.60),
        erI_ranges=[(-0.03, -0.01), (-0.02, 0.00),  (-0.04, -0.02)],
        erC_ranges=[(-0.04, -0.03), (-0.06, -0.03), (-0.03, -0.01)],
        erS_ranges=[None, (-0.02, -0.01), None],
        gender_thresh=39,
        age_breaks=[(25.4, 2), (84.7, 3), (100, 4)],
        ecom_breaks=[(22.2, 2), (100, 3)],
        edu_breaks=[(57.2, 1), (79.3, 2), (100, 3)],
        owner_thresh=92,  dtype_thresh=6.8,
        dwage_breaks=[(10.2, 1), (96, 2), (100, 3)],
        dwsize_breaks=[(15.3, 1), (54, 2), (100, 3)],
        elab_breaks=[(25.9, 1), (40.7, 2), (51.8, 3), (55.5, 4), (100, 5)],
        prov_thresh=44.07,
    ),
    GroupParams(
        cum_upper=98.36, group_id=6,
        income_range=(90001, 110000, 100), gas_range=(0, 3000, 100),
        know_range=(3.1, 4.66), cee_aw_range=(2.72, 5.57), ed_aw_range=(3.66, 6.33),
        pn_range=(3.62, 5.75),  sn_range=(2, 3.83),
        pbcI_range=(2.4, 6.4),  pbcC_range=(1, 5.6),       pbcS_range=(1.5, 5.5),
        ene_pat_range=(0.78, 1.82),
        erI_ranges=[(-0.03, -0.01), (-0.02, 0.00),  (-0.04, -0.02)],
        erC_ranges=[(-0.04, -0.03), (-0.06, -0.03), (-0.03, -0.01)],
        erS_ranges=[None, (-0.02, -0.01), None],
        gender_thresh=50,
        age_breaks=[(40, 2), (90, 3), (100, 4)],
        ecom_breaks=[(33.3, 2), (100, 3)],
        edu_breaks=[(10, 1), (20, 2), (100, 3)],
        owner_thresh=20,  dtype_thresh=0.0,
        dwage_breaks=[(40, 1), (60, 2), (100, 3)],
        dwsize_breaks=[(14, 1), (71, 2), (100, 3)],
        elab_breaks=[(100, 5)],
        prov_thresh=20,
    ),
    GroupParams(
        cum_upper=100.0, group_id=7,
        income_range=(110001, 150000, 100), gas_range=(0, 3000, 100),
        know_range=(3.33, 5.33), cee_aw_range=(3.85, 6.21), ed_aw_range=(2.66, 6),
        pn_range=(2.62, 7),      sn_range=(1, 5.66),
        pbcI_range=(1.6, 6.6),   pbcC_range=(1, 7),         pbcS_range=(3, 6),
        ene_pat_range=(1, 3),
        erI_ranges=[(-0.03, -0.01), (-0.02, 0.00),  (-0.04, -0.02)],
        erC_ranges=[(-0.04, -0.03), (-0.06, -0.03), (-0.03, -0.01)],
        erS_ranges=[None, (-0.02, -0.01), None],
        gender_thresh=37.1,
        age_breaks=[(5.9, 2), (70.6, 3), (100, 4)],
        ecom_breaks=[(2.9, 1), (44.3, 2), (100, 3)],
        edu_breaks=[(29.4, 1), (64.7, 2), (100, 3)],
        owner_thresh=80,  dtype_thresh=5.7,
        dwage_breaks=[(20, 1), (71.4, 2), (100, 3)],
        dwsize_breaks=[(23.6, 1), (61.4, 2), (100, 3)],
        elab_breaks=[(20, 1), (50, 2), (70, 3), (90, 4), (100, 5)],
        prov_thresh=29.41,
    ),
]

# ---------------------------------------------------------------------------
N_HOUSEHOLDS = {"ES": 793, "NL": 759}

CGE_FILES = {
    "ES": "cge-es-ssp2-h.csv",
    "NL": "cge-nl-ssp2-h.csv",
}

START_YEAR = 2016
END_YEAR   = 2050
