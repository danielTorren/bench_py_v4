"""
Multi-year aggregation of renovation counts, matching Niamir et al. (2024).

The paper reports renovation rates as (Fig. 5, 6 and 7 captions):

    "the percentage of households renovation within specific age cohorts
     (<10, 11-35, >35), relative to the total number of households in each
     cohort, observed over a 5-year period"

so the plotted value at year Y is

    100 * (renovations during [Y-4, Y]) / (households in the cohort)

NOT the single-year rate at Y.  This distinction matters a lot here: the
renovation cooldown (15 / 7 / 2 years by dwelling vintage) synchronises
households into cohort waves, so the annual series oscillates with a period of
2 to 7 years while the 5-year aggregate is smooth.

Cohort membership is not constant over a window, because ``_update_dwelling``
redraws ``dw_age`` every year from 2025 onwards.  The denominator is therefore
the mean cohort size over the window (``denominator="mean"``, the default);
``denominator="last"`` uses the cohort size in year Y instead.
"""

import warnings
from typing import Iterable, List, Sequence, Tuple

import numpy as np

# End years of the reporting windows used in the paper's figures.
REPORT_YEARS: List[int] = [2020, 2025, 2030, 2035, 2040, 2045, 2050]

WINDOW = 5

# First year admitted to a reporting window, project-wide.  None = include every
# model year, which is what reproduces the paper.
#
# The run is 2016-2050 = 35 years = exactly 7 windows of 5, and the paper's
# Figs. 5 to 7 show exactly 7 points (2020, 2025, ..., 2050).  The windows tile
# the run with no gap or overlap, each labelled by its END year:
#
#     2020 -> 2016-2020    2035 -> 2031-2035    2050 -> 2046-2050
#     2025 -> 2021-2025    2040 -> 2036-2040
#     2030 -> 2026-2030    2045 -> 2041-2045
#
# That is why the paper's BENCH figures start at 2020 rather than 2016: the
# years 2016-2019 are inside the first window, not missing from the chart.  So
# 2016 belongs in the aggregation and must not be trimmed.
#
# Set to 2017 to exclude the 2016 initialisation tick instead.  Be aware of what
# that does: on tick 1 every household past the behavioural gates renovates at
# once, then serves its cooldown, so for the 15-year and 7-year vintage cohorts
# 2016 IS the entire 2016-2020 window and excluding it sends them to zero.  See
# MODEL_AUDIT.md section 3.3 item 3.
REPORT_MIN_YEAR: int | None = None

_WARNED_FALLBACK = False


def window_bounds(end_year: int, window: int = WINDOW) -> Tuple[int, int]:
    """Inclusive (first_year, last_year) of the reporting window ending at end_year."""
    return end_year - window + 1, end_year


def multi_year_rate(
    years: Sequence[int],
    renovations: Sequence[float],
    cohort_totals: Sequence[float],
    end_years: Iterable[int] = REPORT_YEARS,
    window: int = WINDOW,
    denominator: str = "mean",
    min_year: int | None = None,
) -> Tuple[List[int], np.ndarray, List[bool]]:
    """
    Aggregate an annual renovation series into windowed rates.

    Parameters
    ----------
    years         : annual year labels, ascending.
    renovations   : renovation *counts* per year (not percentages).
    cohort_totals : cohort size per year (denominator source).
    end_years     : window end years to report.
    window        : window length in years (5 in the paper).
    denominator   : "mean" (mean cohort size over the window) or
                    "last" (cohort size in the end year).
    min_year      : drop years before this from every window.  Set to 2017 to
                    exclude the 2016 initialisation tick, which the paper's
                    "over 33 years (2017-2050)" framing implies.

    Returns
    -------
    (end_years_used, rates_pct, window_is_complete)
        ``window_is_complete[i]`` is False when the window was clipped by the
        start of the run or by ``min_year`` (the rate then covers fewer years
        and is not comparable with the others).
    """
    if denominator not in ("mean", "last"):
        raise ValueError(f"denominator must be 'mean' or 'last', got {denominator!r}")

    years = np.asarray(years)
    renov = np.asarray(renovations, dtype=float)
    total = np.asarray(cohort_totals, dtype=float)

    out_years: List[int] = []
    out_rates: List[float] = []
    out_full: List[bool] = []

    for end_year in end_years:
        lo, hi = window_bounds(end_year, window)
        if min_year is not None:
            lo = max(lo, min_year)
        sel = (years >= lo) & (years <= hi)
        if not sel.any():
            continue

        if denominator == "mean":
            denom = total[sel].mean()
        else:
            last = years == end_year
            denom = total[last][0] if last.any() else total[sel][-1]

        out_years.append(int(end_year))
        out_rates.append(100.0 * renov[sel].sum() / denom if denom > 0 else 0.0)
        out_full.append(int(sel.sum()) == window)

    return out_years, np.asarray(out_rates), out_full


def multi_year_rate_from_df(
    df,
    renov_col: str,
    total_col: str,
    pct_col: str | None = None,
    **kwargs,
) -> Tuple[List[int], np.ndarray, List[bool]]:
    """
    ``multi_year_rate`` for one ``annual_results.csv`` DataFrame.

    Falls back to summing the annual percentage column when the raw count
    columns are absent (CSVs written before the count columns were added).
    That fallback equals the exact value only while cohort size is constant,
    so it is approximate from 2025 onwards.
    """
    if renov_col in df.columns and total_col in df.columns:
        return multi_year_rate(
            df["year"].values, df[renov_col].values, df[total_col].values, **kwargs
        )

    if pct_col is None or pct_col not in df.columns:
        raise KeyError(
            f"{renov_col!r}/{total_col!r} missing and no usable fallback column; "
            f"re-run the model to regenerate annual_results.csv"
        )

    global _WARNED_FALLBACK
    if not _WARNED_FALLBACK:
        _WARNED_FALLBACK = True
        warnings.warn(
            f"annual_results.csv has no {renov_col!r} column, so the "
            f"{WINDOW}-year rates fall back to summing annual percentages. "
            "That is exact only while cohort sizes are constant, and cohort "
            "sizes change every year from 2025 (_update_dwelling). Re-run the "
            "model to regenerate the CSVs for exact values.",
            RuntimeWarning,
            stacklevel=2,
        )

    # Approximate: sum the per-year rates, i.e. denominator varies within window.
    years = df["year"].values
    pct = df[pct_col].values
    kwargs.pop("denominator", None)
    window = kwargs.get("window", WINDOW)
    min_year = kwargs.get("min_year")
    end_years = kwargs.get("end_years", REPORT_YEARS)

    out_years, out_rates, out_full = [], [], []
    for end_year in end_years:
        lo, hi = window_bounds(end_year, window)
        if min_year is not None:
            lo = max(lo, min_year)
        sel = (years >= lo) & (years <= hi)
        if not sel.any():
            continue
        out_years.append(int(end_year))
        out_rates.append(float(np.sum(pct[sel])))
        out_full.append(int(np.sum(sel)) == window)
    return out_years, np.asarray(out_rates), out_full


def stack_runs(dfs, renov_col, total_col, pct_col=None, **kwargs):
    """Apply ``multi_year_rate_from_df`` to every run; return (years, matrix, full)."""
    rows = []
    ref_years: List[int] = []
    full: List[bool] = []
    for df in dfs:
        yrs, rates, fl = multi_year_rate_from_df(
            df, renov_col, total_col, pct_col=pct_col, **kwargs
        )
        if not ref_years:
            ref_years, full = yrs, fl
        rows.append(rates)
    return ref_years, np.array(rows), full
