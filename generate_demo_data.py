"""
Pre-compute BENCH v4 scenarios for the interactive web demo.

Runs a one-factor-at-a-time (OFAT) sweep across four behavioral thresholds
(guilt, PBC-invest, personal norm m1, social norm m1) × N values each,
for every combination of case study (NL / ES) and learning mode (4 options).

Each scenario gets S independent seeds; results are aggregated to
mean ± 95% confidence interval of the mean per year.

Output: ../BENCH-x-ABM-model-store/docs/demo_scenarios.json

That is the file docs/demo.html fetches at runtime (see demo.html, the
`fetch('demo_scenarios.json')` call), so writing there updates the live demo
directly with no copy step.  The path is resolved relative to THIS script, not
to the working directory, so it works from anywhere.

Usage
-----
python generate_demo_data.py                                  # defaults
python generate_demo_data.py --seeds 50 --jobs 8
python generate_demo_data.py --n-params 8                     # 8 slider ticks
python generate_demo_data.py --n-households 2000              # larger population
python generate_demo_data.py --output path/to/docs/demo_scenarios.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

# ---------------------------------------------------------------------------
# Make bench_v4 importable when script is run from any working directory
# ---------------------------------------------------------------------------
_BENCH_PATH = str(Path(__file__).resolve().parent)
if _BENCH_PATH not in sys.path:
    sys.path.insert(0, _BENCH_PATH)

# ---------------------------------------------------------------------------
# Default output: the model-store repo's docs/ folder, which is what the live
# demo page fetches.  Resolved against this file so the default holds wherever
# the script is invoked from.
# ---------------------------------------------------------------------------
DEFAULT_OUTPUT = (
    Path(_BENCH_PATH).parent / "BENCH-x-ABM-model-store" / "docs" / "demo_scenarios.json"
)

# ---------------------------------------------------------------------------
# Scenario axes
# ---------------------------------------------------------------------------
CASES = ["NL", "ES"]
LEARNINGS = ["No learning", "Slow dynamics", "Fast dynamics", "Informative"]

PARAM_LABELS: dict[str, str] = {
    "guilt":      "Guilt threshold",
    "pbc_invest": "PBC-invest threshold",
    "pn_m1":      "Personal norm (invest)",
    "sn_m1":      "Social norm (invest)",
}

# Empirical defaults from params.py (shown as reference lines in the demo)
DEFAULTS: dict[str, dict] = {
    "NL": {"guilt": 4.6, "pbc_invest": 1.0,  "pn_m1": 4.7,  "sn_m1": 3.5},
    "ES": {"guilt": 5.2, "pbc_invest": 2.2,  "pn_m1": 5.67, "sn_m1": 4.77},
}

METRICS: dict[str, dict] = {
    "pct_renovated":       {"label": "Annual renovation rate",    "unit": "%"},
    "total_gas_saved_kwh": {"label": "Cumulative gas saved",      "unit": "kWh"},
    "avg_aware":           {"label": "Mean awareness score",      "unit": "(1–7)"},
    "high_m1_pct":         {"label": "High motivation to invest", "unit": "%"},
}

# Per-vintage renovation rate over 5-year windows, i.e. Fig. 5 of
# Niamir et al. (2024).  Separate from METRICS above because these series are
# indexed by `years_5yr` (7 window end-years), not by `years` (35 annual points).
VINTAGE_SERIES: dict[str, dict] = {
    "renov_5yr_dwage1": {"label": "<10 years",   "color": "#2E86AB"},
    "renov_5yr_dwage2": {"label": "11-35 years", "color": "#F18F01"},
    "renov_5yr_dwage3": {"label": ">35 years",   "color": "#A23B72"},
}

# ---------------------------------------------------------------------------
# Worker (runs in a separate process — patching is safe)
# ---------------------------------------------------------------------------

def _run_one(
    bench_path: str,
    case: str,
    learning: str,
    seed: int,
    overrides: dict,
    n_households: int | None = None,
) -> list[dict]:
    """
    Run a single BENCHv4 instance with optional threshold overrides.

    Patches module-level constants in bench_v4.model for this process only.
    Each joblib worker is a separate OS process (loky backend on Windows),
    so patches never bleed across concurrent runs.
    """
    import sys
    if bench_path not in sys.path:
        sys.path.insert(0, bench_path)

    import bench_v4.params as _p
    import bench_v4.model as _m

    # Always restore empirical defaults before applying overrides.
    # Without this, a worker that processed an OFAT job (which patches
    # _m.GUILT_THRESH etc.) and then picks up a default-scenario job
    # (overrides={}) would skip the if-block and run with stale patches.
    _m.GUILT_THRESH      = dict(_p.GUILT_THRESH)
    _m.PBC_INVEST_THRESH = dict(_p.PBC_INVEST_THRESH)
    _m.MOTIVATION_THRESH = {
        cs: {mk: tuple(tv) for mk, tv in inner.items()}
        for cs, inner in _p.MOTIVATION_THRESH.items()
    }

    if overrides:
        if "guilt" in overrides:
            gt = dict(_p.GUILT_THRESH)
            gt[case] = overrides["guilt"]
            _m.GUILT_THRESH = gt

        if "pbc_invest" in overrides:
            pt = dict(_p.PBC_INVEST_THRESH)
            pt[case] = overrides["pbc_invest"]
            _m.PBC_INVEST_THRESH = pt

        if "pn_m1" in overrides or "sn_m1" in overrides:
            # Deep-copy the nested motivation dict (avoid mutating the original)
            mt = {
                cs: {mk: list(tv) for mk, tv in inner.items()}
                for cs, inner in _p.MOTIVATION_THRESH.items()
            }
            pn, sn = mt[case]["m1"]
            if "pn_m1" in overrides:
                pn = overrides["pn_m1"]
            if "sn_m1" in overrides:
                sn = overrides["sn_m1"]
            mt[case]["m1"] = (pn, sn)
            # Restore all entries to tuples (model expects tuples)
            for cs in mt:
                for mk in mt[cs]:
                    mt[cs][mk] = tuple(mt[cs][mk])
            _m.MOTIVATION_THRESH = mt

    from bench_v4.model import BENCHv4  # noqa: PLC0415

    model = BENCHv4(case_study=case, seed=seed, learning=learning,
                    n_households=n_households)
    model.run()

    n_hh = model.n_households
    return [
        {
            "year":               s.year,
            "pct_renovated":      round(100.0 * s.n_renovated / n_hh, 4) if n_hh else 0.0,
            "total_gas_saved_kwh": round(s.total_gas_saved, 2),
            "avg_aware":          round(s.avg_aware, 4),
            "high_m1_pct":        round(s.high_m1_pct, 4),
            # Raw per-vintage counts, needed to build the paper's Fig. 5 rate.
            # They must come back per seed: the 5-year window has to be applied
            # BEFORE collapsing across seeds, because a seed's annual values are
            # strongly correlated (the cohort-wave effect), so a confidence
            # interval cannot be reconstructed from per-year intervals later.
            "rv": [s.renov_by_dwage.get(c, 0) for c in (1, 2, 3)],
            "tv": [s.total_by_dwage.get(c, 0) for c in (1, 2, 3)],
        }
        for s in model.history
    ]


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _mean_ci(mat: np.ndarray) -> dict:
    """mean and 95% CI of the mean across rows (seeds), rounded for JSON."""
    n    = mat.shape[0]
    mean = np.mean(mat, axis=0)
    ci   = 1.96 * np.std(mat, axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(mean)
    return {
        "mean": [round(v, 4) for v in mean.tolist()],
        "lo":   [round(v, 4) for v in (mean - ci).tolist()],
        "hi":   [round(v, 4) for v in (mean + ci).tolist()],
    }


def _aggregate(all_runs: list[list[dict]]) -> dict:
    """
    Reduce seed-runs to mean ± 95% CI of the mean.

    Annual metrics are collapsed year by year.  The per-vintage Fig. 5 rates are
    windowed per seed FIRST (via bench_v4.aggregate, the same definition the
    plots and BENCHv4.renovation_rate_5yr_* use), then collapsed, so their
    intervals account for within-seed correlation across years.
    """
    from bench_v4.aggregate import multi_year_rate  # noqa: PLC0415

    years = [r["year"] for r in all_runs[0]]
    out: dict = {"years": years}

    for metric in METRICS:
        out[metric] = _mean_ci(
            np.array([[r[metric] for r in run] for run in all_runs])
        )

    end_years: list[int] = []
    for ci_, cat in enumerate((1, 2, 3)):
        per_seed = []
        for run in all_runs:
            end_years, rates, _ = multi_year_rate(
                years,
                [r["rv"][ci_] for r in run],
                [r["tv"][ci_] for r in run],
            )
            per_seed.append(rates)
        out[f"renov_5yr_dwage{cat}"] = _mean_ci(np.array(per_seed))

    out["years_5yr"] = end_years
    return out


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate BENCH v4 demo data")
    parser.add_argument(
        "--output", default=str(DEFAULT_OUTPUT),
        help=f"Output JSON path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--seeds", type=int, default=100,
        help="Monte Carlo seeds per scenario (default: 100)",
    )
    parser.add_argument(
        "--jobs", type=int, default=-1,
        help="Parallel workers; -1 = all cores (default: -1)",
    )
    parser.add_argument(
        "--n-params", type=int, default=12,
        help="Number of evenly-spaced slider ticks per parameter (default: 12)",
    )
    parser.add_argument(
        "--n-households", type=int, default=None,
        help="Agents per run; None uses survey default (~759 NL / 793 ES)",
    )
    args = parser.parse_args()

    # Fail before the sweep, not after it.  mkdir(parents=True) below would
    # otherwise happily invent a whole directory tree if the target repo is not
    # checked out next to this one, and the ~3 min of compute would land there.
    out_path = Path(args.output)
    if not out_path.parent.is_dir():
        parser.error(
            f"output directory does not exist: {out_path.parent}\n"
            f"        Expected the model-store repo alongside this one. Either clone it, "
            f"or pass --output explicitly."
        )

    # Build parameter value grid from CLI arg
    param_values = [round(v, 4) for v in np.linspace(1.0, 7.0, args.n_params).tolist()]
    params: dict[str, dict] = {
        p: {"label": label, "values": param_values}
        for p, label in PARAM_LABELS.items()
    }

    # ------------------------------------------------------------------
    # Build flat job list:
    #   For each (case, learning): one default scenario + 4×N OFAT scenarios
    # ------------------------------------------------------------------
    jobs: list[tuple] = []       # (scenario_key, case, learning, seed, overrides)
    scenario_keys: list[str] = []

    for case in CASES:
        for learning in LEARNINGS:
            # Baseline — all thresholds at empirical defaults
            key = f"{case}|{learning}|default"
            if key not in scenario_keys:
                scenario_keys.append(key)
            for seed in range(1, args.seeds + 1):
                jobs.append((key, case, learning, seed, {}))

            # OFAT — vary one parameter at a time
            for param, pinfo in params.items():
                for vi, val in enumerate(pinfo["values"]):
                    key = f"{case}|{learning}|{param}|{vi}"
                    if key not in scenario_keys:
                        scenario_keys.append(key)
                    for seed in range(1, args.seeds + 1):
                        jobs.append((key, case, learning, seed, {param: val}))

    n_scenarios = len(scenario_keys)
    n_total     = len(jobs)
    print(f"Scenarios      : {n_scenarios}")
    print(f"Seeds/scenario : {args.seeds}")
    print(f"Slider ticks   : {args.n_params}")
    print(f"Households     : {args.n_households or 'survey default (~759 NL / 793 ES)'}")
    print(f"Total runs     : {n_total}")
    print(f"Workers        : {args.jobs}  (-1 = all cores)")
    print("Running …\n")

    # ------------------------------------------------------------------
    # Parallel execution
    # ------------------------------------------------------------------
    results = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(_run_one)(_BENCH_PATH, case, learning, seed, overrides,
                          args.n_households)
        for (_, case, learning, seed, overrides) in jobs
    )

    # ------------------------------------------------------------------
    # Group raw results by scenario key
    # ------------------------------------------------------------------
    grouped: dict[str, list] = defaultdict(list)
    for (key, *_), run_result in zip(jobs, results):
        grouped[key].append(run_result)

    # ------------------------------------------------------------------
    # Aggregate and serialise
    # ------------------------------------------------------------------
    from bench_v4.params import N_HOUSEHOLDS as _SURVEY_N  # noqa: PLC0415
    from bench_v4.aggregate import REPORT_YEARS, WINDOW     # noqa: PLC0415

    # Resolve actual household count used per case
    if args.n_households is not None:
        n_hh_used = {case: args.n_households for case in CASES}
    else:
        n_hh_used = {case: _SURVEY_N[case] for case in CASES}

    years = [r["year"] for r in results[0]]

    output = {
        "years":          years,
        "n_seeds":        args.seeds,
        "n_households":   n_hh_used,
        "cases":          CASES,
        "learning_modes": LEARNINGS,
        "params":         {p: {"label": v["label"], "values": v["values"]}
                           for p, v in params.items()},
        "defaults":       DEFAULTS,
        "metrics":        METRICS,
        "vintage_series": VINTAGE_SERIES,
        "years_5yr":      list(REPORT_YEARS),
        "window":         WINDOW,
        "scenarios":      {key: _aggregate(grouped[key])
                           for key in scenario_keys},
    }

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, separators=(",", ":"))

    size_kb = out_path.stat().st_size / 1024
    print(f"\nDone. Written to: {out_path.resolve()}")
    print(f"File size: {size_kb:.1f} KB")


if __name__ == "__main__":
    main()
