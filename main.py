"""
Entry point for BENCH v4 Python model.

Runs every scenario in a YAML config file.  Run i of a scenario uses the
random seed `seed + i` (`seed` defaults to 1), so runs differ from each other
and each run can be repeated.

Folder structure
----------------
output/
  {CONFIG}_{YYYYMMDD_HHMMSS}/
    {CONFIG}.yaml                      <- copy of the config used
    {RUN_LABEL}/                       <- one folder per scenario
      runs/
        run_001_seed_1/
          annual_results.csv
          run_config.json
          summary.txt
        ...
      plots/

Usage
-----
    python main.py --config configs/paper_reproduction.yaml
    python main.py --config configs/paper_reproduction.yaml --jobs 4 --no-plot
"""

import argparse
import os
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).parent))

from bench_v4 import BENCHv4
from bench_v4.output import save_run


_LEARNING_SLUG = {
    "Slow dynamics": "Slow_dynamics",
    "Fast dynamics": "Fast_dynamics",
    "Informative":   "Informative",
    "No learning":   "No_learning",
}


def _make_config_dir(output_dir: str, label: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, f"{label}_{timestamp}")
    os.makedirs(path, exist_ok=True)
    return path


def _run_and_save(case, learning, seed, memory, run_label_i, runs_dir,
                  verbose=False):
    """Worker: run one model instance, save outputs, return the model."""
    model = BENCHv4(case_study=case, seed=seed, learning=learning, memory=memory)
    model.run(verbose=verbose)
    run_dir = os.path.join(runs_dir, f"{run_label_i}_seed_{model.seed}")
    save_run(model, run_dir)
    return model


def _run_and_save_tagged(si, case, learning, seed, memory, run_label_i, runs_dir):
    """Like _run_and_save but returns (scenario_index, model) for flat parallel dispatch."""
    model = _run_and_save(case, learning, seed, memory, run_label_i, runs_dir)
    return si, model


def _run_batch_tagged(batch_args: list) -> list:
    """Run a batch of tagged jobs sequentially; reduces dispatch overhead."""
    return [_run_and_save_tagged(*a) for a in batch_args]


def _print_mean_table(all_models, case: str, learning: str) -> None:
    """Print mean annual renovation rates across all seeds for one scenario."""
    if not all_models:
        return
    accum: dict = defaultdict(lambda: defaultdict(list))
    for model in all_models:
        n_hh = model.n_households
        for s in model.history:
            pct = 100 * s.n_renovated / n_hh if n_hh else 0.0
            accum["all"][s.year].append(pct)
            for cat in (1, 2, 3):
                t = s.total_by_dwage.get(cat, 0)
                p = 100 * s.renov_by_dwage.get(cat, 0) / t if t else 0.0
                accum[f"dwage{cat}"][s.year].append(p)

    def mean(vals: list) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    #n = len(all_models)
    #print(f"\n{case} / {learning}  ({n} run{'s' if n != 1 else ''})")
    #print(f"  {'Year':>4}  {'All%':>6}  {'New%':>6}  {'Mid%':>6}  {'Old%':>6}")
    #for yr in sorted(accum["all"]):
    #    print(f"  {yr:>4}  "
    #          f"{mean(accum['all'][yr]):>5.2f}%  "
    #          f"{mean(accum['dwage1'][yr]):>5.2f}%  "
    #          f"{mean(accum['dwage2'][yr]):>5.2f}%  "
    #          f"{mean(accum['dwage3'][yr]):>5.2f}%")




def _load_yaml(config_path: str):
    try:
        import yaml
    except ImportError:
        print("ERROR: PyYAML is required for --config. Install with: uv add pyyaml")
        sys.exit(1)
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Run BENCH v4 ABM")
    parser.add_argument("--config",    required=True,
                        help="YAML scenario file (runs all entries in sequence)")
    parser.add_argument("--jobs",      type=int, default=-1,
                        help="Parallel workers for ensemble runs  "
                             "(default: -1 = all available cores)")
    parser.add_argument("--output-dir", default="output",
                        help="Root folder for all outputs  (default: output/)")
    parser.add_argument("--no-plot",   action="store_true",
                        help="Skip plot generation")
    args = parser.parse_args()

    t0 = datetime.now()
    
    if args.config:
        scenarios   = _load_yaml(args.config)
        config_stem = Path(args.config).stem
        parent_dir  = _make_config_dir(args.output_dir, config_stem)
        total_runs  = sum(sc.get("runs", 1) for sc in scenarios)

        print(f"Config run folder : {parent_dir}")
        print(f"Scenarios         : {len(scenarios)}")
        print(f"Total runs        : {total_runs}")
        print(f"Jobs              : {args.jobs}  (-1 = all available cores)")

        # Save the config file used for this run (reproducibility)
        shutil.copy(args.config, os.path.join(parent_dir, Path(args.config).name))

        # Pre-create per-scenario directories and build a flat job list across
        # all scenarios × seeds so the entire batch runs in one parallel pool
        # (no nested parallelism).
        sc_dirs: list = []
        worker_args: list = []

        for si, sc in enumerate(scenarios):
            slug  = _LEARNING_SLUG.get(sc.get("learning", "Informative"),
                                        sc.get("learning", "Informative").replace(" ", "_"))
            label = sc.get("run_label") or f"{sc.get('case_study', 'NL')}_{slug}"
            sc_dir   = os.path.join(parent_dir, label)
            runs_dir = os.path.join(sc_dir, "runs")
            os.makedirs(runs_dir, exist_ok=True)
            sc_dirs.append((sc_dir, runs_dir))

            case      = sc.get("case_study", "NL")
            learning  = sc.get("learning", "Informative")
            memory_sc = sc.get("memory", True)
            runs      = sc.get("runs", 1)
            # Run i of a scenario uses seed base_seed + i, so every run is
            # different and every run can be repeated.  NetLogo equivalent:
            # Generate-seed? = false, Seed-for-random = base_seed + i.
            base_seed = sc.get("seed", 1)

            for i in range(runs):
                worker_args.append(
                    (si, case, learning, base_seed + i, memory_sc,
                     f"run_{i+1:03d}", runs_dir)
                )

        # Batch jobs 4-per-core to reduce dispatch overhead.
        eff_workers = os.cpu_count() or 8
        if args.jobs not in (-1, None):
            eff_workers = max(1, abs(args.jobs))
        n_batches  = max(1, 4 * eff_workers)
        batch_size = max(1, (len(worker_args) + n_batches - 1) // n_batches)
        batches    = [worker_args[i:i + batch_size]
                      for i in range(0, len(worker_args), batch_size)]
        raw_nested = Parallel(n_jobs=args.jobs, verbose=1)(
            delayed(_run_batch_tagged)(b) for b in batches
        ) or []
        raw = [item for sublist in raw_nested for item in sublist]

        # Group returned models by scenario index
        sc_models: dict = defaultdict(list)
        for si, model in raw:
            sc_models[si].append(model)

        # Per-scenario post-processing (sequential — just I/O and plotting)
        for si, sc in enumerate(scenarios):
            sc_dir, _ = sc_dirs[si]
            _print_mean_table(sc_models[si],
                              sc.get("case_study", "NL"),
                              sc.get("learning", "Informative"))
            if not args.no_plot:
                try:
                    from bench_v4.plotting import plot_all
                    plot_all(sc_dir)
                except ImportError as e:
                    print(f"Plotting skipped for scenario {si} (missing dependency: {e})")

        if not args.no_plot:
            try:
                from bench_v4.plotting import plot_multi_scenario
                print("\nGenerating multi-scenario comparison plots...")
                plot_multi_scenario(parent_dir)
            except ImportError as e:
                print(f"Multi-scenario plotting skipped (missing dependency: {e})")

        print(f"\nAll done.  Results in: {parent_dir}")


    elapsed = datetime.now() - t0
    print(f"\nCompleted in {str(elapsed).split('.')[0]}")


if __name__ == "__main__":
    main()
