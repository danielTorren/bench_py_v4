"""
Compare BENCH v4 output against the published values in Niamir et al. (2024).

Runs the four BENCH scenarios from the paper (NL/ES x SD/ID), aggregates the
renovation rate the way the paper does (5-year windows, per dwelling-vintage
cohort, see bench_v4.aggregate), and tabulates model vs published values.

    python compare_to_paper.py                   # 100 seeds, ~15 s
    python compare_to_paper.py --seeds 30        # quicker
    python compare_to_paper.py --case NL         # one case study
    python compare_to_paper.py --plot cmp.png    # also save a figure
    python compare_to_paper.py --csv cmp.csv     # machine-readable output

Reference values
----------------
PAPER_FIG5 below is digitised by eye from Fig. 5 of the published PDF, which
has no data table and no supplementary data file. Treat the values as accurate
to roughly +/- 0.5 percentage points, and read differences smaller than that as
noise. Replace them if the underlying series is ever obtained from the authors
(see MODEL_AUDIT.md section 3.2, item 1).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

_BENCH_PATH = str(Path(__file__).resolve().parent)
if _BENCH_PATH not in sys.path:
    sys.path.insert(0, _BENCH_PATH)

from bench_v4.aggregate import REPORT_YEARS, WINDOW, multi_year_rate  # noqa: E402

# Paper scenario name -> the model's `learning` setting.
SCENARIOS = {
    "SD": "Slow dynamics",   # Table 1: "Slow Dynamic", the BENCH baseline
    "ID": "Informative",     # Table 1: "Informative Dynamic"
}

COHORTS = {1: "<10 yr", 2: "11-35 yr", 3: ">35 yr"}

# Read off Fig. 5. Index order matches REPORT_YEARS: 2020, 2025, ..., 2050.
PAPER_FIG5: dict[tuple[str, str], dict[int, list[float]]] = {
    ("NL", "SD"): {
        1: [6.0, 1.0, 2.0, 1.5, 1.5, 1.5, 0.5],
        2: [6.8, 6.8, 2.0, 1.5, 2.0, 2.5, 1.2],
        3: [10.5, 5.5, 2.0, 1.5, 2.0, 1.5, 0.5],
    },
    ("NL", "ID"): {
        1: [9.5, 2.5, 7.0, 5.0, 5.0, 0.5, 1.0],
        2: [6.0, 7.5, 5.0, 4.0, 5.0, 2.0, 1.0],
        3: [15.0, 27.0, 4.5, 4.0, 4.5, 3.0, 1.0],
    },
    ("ES", "SD"): {
        1: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        2: [1.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
        3: [1.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0],
    },
    ("ES", "ID"): {
        1: [2.5, 1.0, 2.0, 0.5, 1.5, 0.5, 0.5],
        2: [1.5, 2.5, 2.0, 1.5, 2.2, 0.5, 0.8],
        3: [7.0, 3.5, 2.0, 0.8, 1.0, 0.2, 0.2],
    },
}

DIGITISING_TOLERANCE = 0.5  # percentage points


def _run_seed(bench_path: str, case: str, learning: str, seed: int,
              n_households: int | None) -> dict:
    """One model run; returns the per-cohort windowed rates for that seed."""
    import sys
    if bench_path not in sys.path:
        sys.path.insert(0, bench_path)
    from bench_v4 import BENCHv4
    from bench_v4.aggregate import multi_year_rate

    m = BENCHv4(case_study=case, learning=learning, seed=seed,
                n_households=n_households)
    m.run()
    years = m.years()
    out = {}
    for cat in COHORTS:
        _, rates, _ = multi_year_rate(
            years,
            [s.renov_by_dwage.get(cat, 0) for s in m.history],
            [s.total_by_dwage.get(cat, 0) for s in m.history],
        )
        out[cat] = rates
    return out


def run_scenario(case: str, scen: str, seeds: int, jobs: int,
                 n_households: int | None) -> dict[int, np.ndarray]:
    """Returns {cohort: (seeds, n_windows) array of windowed rates}."""
    per_seed = Parallel(n_jobs=jobs)(
        delayed(_run_seed)(_BENCH_PATH, case, SCENARIOS[scen], s, n_households)
        for s in range(1, seeds + 1)
    )
    return {cat: np.array([r[cat] for r in per_seed]) for cat in COHORTS}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=100,
                    help="Monte Carlo seeds per scenario (default: 100, as the paper)")
    ap.add_argument("--jobs", type=int, default=-1, help="Parallel workers (-1 = all cores)")
    ap.add_argument("--case", choices=["NL", "ES"], help="Limit to one case study")
    ap.add_argument("--n-households", type=int, default=None,
                    help="Agents per run (default: survey size, 759 NL / 793 ES)")
    ap.add_argument("--plot", metavar="PNG", help="Also save a model-vs-paper figure")
    ap.add_argument("--csv", metavar="CSV", help="Also write the table as CSV")
    args = ap.parse_args()

    cases = [args.case] if args.case else ["NL", "ES"]
    print(f"BENCH v4 vs Niamir et al. (2024) Fig. 5")
    print(f"{args.seeds} seeds/scenario  |  {WINDOW}-year windows  |  "
          f"households: {args.n_households or 'survey default'}")
    print(f"Paper values digitised from Fig. 5, +/- ~{DIGITISING_TOLERANCE} pp\n")

    results: dict = {}
    rows: list = []
    all_err: list[float] = []

    for case in cases:
        for scen in SCENARIOS:
            key = (case, scen)
            if key not in PAPER_FIG5:
                continue
            results[key] = run_scenario(case, scen, args.seeds, args.jobs,
                                        args.n_households)

            hdr = "  ".join(f"{y:>6d}" for y in REPORT_YEARS)
            print(f"--- {case} {scen} " + "-" * 52)
            print(f"  {'cohort':9s} {'':6s}  {hdr}   MAE")
            for cat, label in COHORTS.items():
                mat   = results[key][cat]
                model = mat.mean(0)
                paper = np.array(PAPER_FIG5[key][cat])
                err   = np.abs(model - paper)
                all_err.extend(err.tolist())

                ci = 1.96 * mat.std(0, ddof=1) / np.sqrt(len(mat)) if len(mat) > 1 else np.zeros_like(model)
                print(f"  {label:9s} {'model':6s}  " +
                      "  ".join(f"{v:6.1f}" for v in model) + f"  {err.mean():5.2f}")
                print(f"  {'':9s} {'paper':6s}  " +
                      "  ".join(f"{v:6.1f}" for v in paper))
                print(f"  {'':9s} {'+/-CI':6s}  " +
                      "  ".join(f"{v:6.2f}" for v in ci))

                for i, yr in enumerate(REPORT_YEARS):
                    rows.append({
                        "case": case, "scenario": scen, "cohort": label,
                        "window_end": yr,
                        "window_start": yr - WINDOW + 1,
                        "model": round(float(model[i]), 3),
                        "ci95": round(float(ci[i]), 3),
                        "paper": float(paper[i]),
                        "abs_error": round(float(err[i]), 3),
                    })
            print()

    err_arr = np.array(all_err)
    within  = (err_arr <= DIGITISING_TOLERANCE).mean() * 100
    print("=" * 68)
    print(f"Overall MAE vs paper : {err_arr.mean():.2f} pp   (median {np.median(err_arr):.2f})")
    print(f"Points within +/-{DIGITISING_TOLERANCE} pp : {within:.0f} %  "
          f"({int((err_arr <= DIGITISING_TOLERANCE).sum())} of {len(err_arr)})")
    print(f"Worst single point   : {err_arr.max():.2f} pp")
    print()
    print("Known contributors to the residual, see MODEL_AUDIT.md:")
    print("  1.2 #1  'Slow dynamics' reaches ~1 agent/run, so SD is effectively")
    print("          No learning, not the social baseline the paper describes")
    print("  3.3 #3  every household starts with an empty renovation history, so the")
    print("          2016 tick fires as one synchronised burst")
    print("  2.3 #2  dwelling vintage shares are applied in reverse order from 2025")

    if args.csv:
        import csv
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nCSV written to {Path(args.csv).resolve()}")

    if args.plot:
        _plot(results, args.plot, args.seeds)


def _plot(results: dict, path: str, seeds: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = list(results)
    fig, axes = plt.subplots(len(keys), 3, figsize=(13, 3.1 * len(keys)),
                             sharex=True, squeeze=False)
    colors = {1: "#2E86AB", 2: "#F18F01", 3: "#A23B72"}

    for r, key in enumerate(keys):
        case, scen = key
        for c, (cat, label) in enumerate(COHORTS.items()):
            ax    = axes[r][c]
            mat   = results[key][cat]
            model = mat.mean(0)
            ci    = 1.96 * mat.std(0, ddof=1) / np.sqrt(len(mat)) if len(mat) > 1 else 0
            paper = PAPER_FIG5[key][cat]

            ax.plot(REPORT_YEARS, model, "-o", color=colors[cat], lw=2, ms=5, label="Model")
            ax.fill_between(REPORT_YEARS, model - ci, model + ci, color=colors[cat], alpha=0.18)
            ax.plot(REPORT_YEARS, paper, "--s", color="#555555", lw=1.5, ms=4, label="Paper Fig. 5")
            ax.grid(alpha=0.35, ls="--")
            if r == 0:
                ax.set_title(label, fontsize=11, fontweight="bold")
            if c == 0:
                ax.set_ylabel(f"{case} {scen}\n% of cohort per {WINDOW} yr", fontsize=9)
            if r == len(keys) - 1:
                ax.set_xlabel("Window end year", fontsize=9)
            if r == 0 and c == 0:
                ax.legend(fontsize=8, frameon=True)

    fig.suptitle(f"BENCH v4 vs Niamir et al. (2024) Fig. 5   "
                 f"(model = mean of {seeds} seeds, band = 95% CI)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure written to {Path(path).resolve()}")


if __name__ == "__main__":
    main()
