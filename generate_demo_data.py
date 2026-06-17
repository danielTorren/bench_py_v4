"""
Pre-compute BENCH v4 scenarios for the interactive web demo.

Runs a one-factor-at-a-time (OFAT) sweep across four behavioral thresholds
(guilt, PBC-invest, personal norm m1, social norm m1) × N values each,
for every combination of case study (NL / ES) and learning mode (4 options).

Each scenario gets S independent seeds; results are aggregated to
mean ± 95% confidence interval of the mean per year.

Output: demo_scenarios.json  — copy to bench-models-archive/docs/

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
        }
        for s in model.history
    ]


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _aggregate(all_runs: list[list[dict]]) -> dict:
    """Reduce seed-runs to mean ± 95% CI of the mean per year × metric."""
    years = [r["year"] for r in all_runs[0]]
    n = len(all_runs)
    out: dict = {"years": years}
    for metric in METRICS:
        mat  = np.array([[r[metric] for r in run] for run in all_runs])
        mean = np.mean(mat, axis=0)
        ci   = 1.96 * np.std(mat, axis=0, ddof=1) / np.sqrt(n)
        out[metric] = {
            "mean": [round(v, 4) for v in mean.tolist()],
            "lo":   [round(v, 4) for v in (mean - ci).tolist()],
            "hi":   [round(v, 4) for v in (mean + ci).tolist()],
        }
    return out


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate BENCH v4 demo data")
    parser.add_argument(
        "--output", default="demo_scenarios.json",
        help="Output JSON path (default: demo_scenarios.json)",
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
    years = [r["year"] for r in results[0]]

    output = {
        "years":          years,
        "cases":          CASES,
        "learning_modes": LEARNINGS,
        "params":         {p: {"label": v["label"], "values": v["values"]}
                           for p, v in params.items()},
        "defaults":       DEFAULTS,
        "metrics":        METRICS,
        "scenarios":      {key: _aggregate(grouped[key])
                           for key in scenario_keys},
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, separators=(",", ":"))

    size_kb = out_path.stat().st_size / 1024
    print(f"\nDone. Written to: {out_path.resolve()}")
    print(f"File size: {size_kb:.1f} KB")


if __name__ == "__main__":
    main()
