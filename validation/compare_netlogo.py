"""
Compare the Python port with the original NetLogo model, seed ensemble against
seed ensemble.

The two models use different random number generators, so single runs cannot
match.  The test is distributional: for each case, scenario, metric and year,
the Python ensemble mean is compared with the NetLogo ensemble mean by a
two-sample z statistic.  If the port is faithful, about 5 % of cells have
|z| > 1.96 by chance, and very few have |z| > 3.

Usage
-----
    python validation/run_netlogo.py --runs 100      # once, writes the NetLogo table
    python validation/compare_netlogo.py             # runs Python, writes the report

Year convention: in the BehaviorSpace table the row labelled `year = Y` holds
model year Y - 1, and model year 2050 is not recorded.  Both models are
compared on model years 2016-2049.
"""

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

METRICS = (
    ["a1"]
    + [f"group{g}.a1" for g in range(1, 6)]
    + [f"dwage{c}.a1" for c in range(1, 4)]
    + [f"number.dwage{c}" for c in range(1, 4)]
)


def load_netlogo(paths: list[Path]):
    """Returns {(case, learning): {seed: {metric: {model_year: value}}}}."""
    out: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for path in paths:
        _load_table(path, out)
    return out


def _load_table(path: Path, out: dict) -> None:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    hdr_i = next(i for i, r in enumerate(rows) if r and r[0] == "[run number]")
    hdr = rows[hdr_i]
    col = {name: k for k, name in enumerate(hdr)}
    for r in rows[hdr_i + 1:]:
        step = int(r[col["[step]"]])
        if step == 0:
            continue
        key = (r[col["case-study"]], r[col["Learning"]])
        seed = int(float(r[col["Seed-for-random"]]))
        year = int(float(r[col["year"]])) - 1
        for m in METRICS:
            out[key][seed][m][year] = float(r[col[m]])


def _python_run(case: str, learning: str, seed: int) -> dict:
    from bench_v4 import BENCHv4
    m = BENCHv4(case_study=case, learning=learning, seed=seed)
    m.run()
    res: dict = defaultdict(dict)
    for s in m.history:
        res["a1"][s.year] = s.n_renovated
        for g in range(1, 6):
            res[f"group{g}.a1"][s.year] = s.renov_by_group.get(g, 0)
        for c in range(1, 4):
            res[f"dwage{c}.a1"][s.year] = s.renov_by_dwage.get(c, 0)
            res[f"number.dwage{c}"][s.year] = s.total_by_dwage.get(c, 0)
    return dict(res)


def _matrix(runs: list[dict], metric: str, years: list[int]) -> np.ndarray:
    return np.array([[r[metric].get(y, np.nan) for y in years] for r in runs], dtype=float)


def _z(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    se = np.sqrt(a.var(0, ddof=1) / len(a) + b.var(0, ddof=1) / len(b))
    diff = a.mean(0) - b.mean(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(se > 0, diff / se, np.where(diff == 0, 0.0, np.inf))
    return z


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--table", type=Path, nargs="+",
                   default=[ROOT / "validation" / "netlogo_out" / "netlogo_table.csv"],
                   help="One or more BehaviorSpace tables (seeds are merged)")
    p.add_argument("--jobs", type=int, default=-1)
    p.add_argument("--report", type=Path, default=ROOT / "validation" / "comparison.md")
    args = p.parse_args()

    nl = load_netlogo(args.table)
    years = list(range(2016, 2050))
    lines = ["# Python port vs NetLogo: ensemble comparison", "",
             f"NetLogo tables: {', '.join(f'`{t.name}`' for t in args.table)}. Model years {years[0]}-{years[-1]}.",
             "z = (mean_python - mean_netlogo) / standard error. "
             "Expected by chance: about 5 % of cells with |z| > 1.96. "
             "Cells where both models are always zero count as z = 0.", ""]
    summary = ["| case | scenario | seeds | total renovations NetLogo | Python | z | "
               "cells |z|>1.96 | cells |z|>3 |",
               "|---|---|---|---|---|---|---|---|"]
    detail: list[str] = []
    all_z: list[float] = []

    for (case, learning), by_seed in sorted(nl.items()):
        seeds = sorted(by_seed)
        nl_runs = [by_seed[s] for s in seeds]
        py_runs = Parallel(n_jobs=args.jobs)(
            delayed(_python_run)(case, learning, s) for s in seeds)

        cell_z: list[float] = []
        detail += [f"## {case} / {learning} ({len(seeds)} seeds)", "",
                   "| metric | NetLogo mean (sum over years) | Python mean | max |z| | year of max |",
                   "|---|---|---|---|---|"]
        for m in METRICS:
            a = _matrix(py_runs, m, years)
            b = _matrix(nl_runs, m, years)
            z = _z(a, b)
            cell_z += np.abs(z).tolist()
            k = int(np.nanargmax(np.abs(z))) if np.isfinite(z).any() else 0
            detail.append(f"| {m} | {b.mean(0).sum():.1f} | {a.mean(0).sum():.1f} | "
                          f"{abs(z[k]):.2f} | {years[k]} |")
        detail.append("")

        tot_py = _matrix(py_runs, "a1", years).sum(1)
        tot_nl = _matrix(nl_runs, "a1", years).sum(1)
        zt = float(_z(tot_py[:, None], tot_nl[:, None])[0])
        cz = np.array(cell_z)
        all_z += cell_z
        summary.append(
            f"| {case} | {learning} | {len(seeds)} | {tot_nl.mean():.1f} | {tot_py.mean():.1f} | "
            f"{zt:+.2f} | {100 * (cz > 1.96).mean():.1f} % | {int((cz > 3).sum())} of {len(cz)} |")

    az = np.array(all_z)
    lines += summary + ["",
                        f"All cells: {100 * (az > 1.96).mean():.1f} % with |z| > 1.96, "
                        f"{int((az > 3).sum())} of {len(az)} with |z| > 3.", ""] + detail
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:len(summary) + 8]))
    print(f"\nReport written: {args.report}")


if __name__ == "__main__":
    main()
