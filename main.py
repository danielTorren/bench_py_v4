"""
Entry point for BENCH v4 Python model.

Folder structure created per run batch
---------------------------------------
output/
  {CASE}_{LEARNING}_{YYYYMMDD_HHMMSS}/        <- one folder per configuration
    runs/
      run_001_seed_12345/
        annual_results.csv
        run_config.json
        summary.txt
      run_002_seed_67890/
        ...
    plots/
      renovation_rate_overall.png
      renovation_by_vintage.png
      renovation_by_vintage_cumulative.png
      renovation_by_income.png
      behaviour_count_by_type.png
      investment_by_type.png
      energy_saved_by_type.png
      motivation_over_time.png
      gas_savings_annual.png
      avg_awareness.png

Usage examples
--------------
Single run (verbose output):
    python main.py --case NL --learning Informative --seed 42 --verbose

100-run ensemble (parallel, saves data + plots automatically):
    python main.py --case NL --learning Informative --runs 100

Run all scenarios from a YAML config file:
    python main.py --config configs/bench_v4_scenarios.yaml

Custom output directory:
    python main.py --case ES --runs 50 --output-dir results

Limit parallel workers:
    python main.py --case NL --runs 100 --jobs 4

Skip plotting (data only):
    python main.py --case NL --runs 20 --no-plot
"""

import argparse
import os
import sys
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
                  verbose=False, n_households=None):
    """Worker: run one model instance, save outputs, return the model."""
    model = BENCHv4(case_study=case, seed=seed, learning=learning, memory=memory,
                    n_households=n_households)
    model.run(verbose=verbose)
    run_dir = os.path.join(runs_dir, f"{run_label_i}_seed_{model.seed}")
    save_run(model, run_dir)
    return model


def _print_mean_table(all_models, case, learning):
    from collections import defaultdict
    accum = defaultdict(lambda: defaultdict(list))
    for model in all_models:
        n_hh = model.n_households
        for s in model.history:
            pct = 100 * s.n_renovated / n_hh if n_hh else 0
            accum["all"][s.year].append(pct)
            for cat in (1, 2, 3):
                t = s.total_by_dwage.get(cat, 0)
                p = 100 * s.renov_by_dwage.get(cat, 0) / t if t else 0
                accum[f"dwage{cat}"][s.year].append(p)

    n = len(all_models)
    for yr in sorted(accum["all"].keys()):
        def m(key):
            v = accum[key][yr]
            return sum(v) / len(v) if v else 0.0


def _run_scenario(case, learning, runs, memory, output_dir,
                  run_label, verbose, no_plot, n_jobs=-1, timestamped=True,
                  n_households=None):
    """Execute one scenario (possibly many seed runs) and save outputs."""
    slug  = _LEARNING_SLUG.get(learning, learning.replace(" ", "_"))
    label = run_label if run_label else f"{case}_{slug}"
    if timestamped:
        config_dir = _make_config_dir(output_dir, label)
    else:
        config_dir = os.path.join(output_dir, label)
        os.makedirs(config_dir, exist_ok=True)
    runs_dir = os.path.join(config_dir, "runs")
    os.makedirs(runs_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Scenario : {case} / {learning}")
    print(f"Runs     : {runs}")
    print(f"Output   : {config_dir}")
    print(f"{'='*60}")

    single_run = runs == 1

    if single_run:
        model = _run_and_save(case, learning, None, memory, "run_001", runs_dir,
                              verbose=verbose, n_households=n_households)
        print(model.summary())
        all_models = [model]
    else:
        worker_args = [
            (case, learning, None, memory, f"run_{i+1:03d}", runs_dir,
             False, n_households)
            for i in range(runs)
        ]
        all_models = Parallel(n_jobs=n_jobs)(
            delayed(_run_and_save)(*arg) for arg in worker_args
        )

    _print_mean_table(all_models, case, learning)

    if not no_plot:
        try:
            from bench_v4.plotting import plot_all
            plot_all(config_dir)
        except ImportError as e:
            print(f"Plotting skipped (missing dependency: {e})")

    return config_dir


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
    parser.add_argument("--config",    default=None,
                        help="YAML scenario file (runs all entries in sequence)")
    parser.add_argument("--case",      default="NL", choices=["ES", "NL"],
                        help="Case study  (default: NL)")
    parser.add_argument("--learning",  default="Informative",
                        choices=["Slow dynamics", "Fast dynamics",
                                 "Informative", "No learning"],
                        help="Learning type  (default: Informative)")
    parser.add_argument("--seed",      type=int, default=None,
                        help="Fixed random seed for a single run")
    parser.add_argument("--runs",      type=int, default=1,
                        help="Number of Monte Carlo runs  (default: 1)")
    parser.add_argument("--jobs",      type=int, default=-1,
                        help="Parallel workers for ensemble runs  "
                             "(default: -1 = all available cores)")
    parser.add_argument("--n-households", type=int, default=None,
                        help="Synthetic population size (default: survey size ~759 NL / 793 ES)")
    parser.add_argument("--no-memory", action="store_true",
                        help="Disable pre-2016 memory recall")
    parser.add_argument("--output-dir", default="output",
                        help="Root folder for all outputs  (default: output/)")
    parser.add_argument("--no-plot",   action="store_true",
                        help="Skip plot generation")
    parser.add_argument("--verbose",   action="store_true",
                        help="Print year-by-year progress")
    args = parser.parse_args()

    memory = not args.no_memory

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
        for sc in scenarios:
            _run_scenario(
                case         = sc.get("case_study", "NL"),
                learning     = sc.get("learning", "Informative"),
                runs         = sc.get("runs", 1),
                memory       = sc.get("memory", True),
                output_dir   = parent_dir,
                run_label    = sc.get("run_label", None),
                verbose      = args.verbose,
                no_plot      = args.no_plot,
                n_jobs       = args.jobs,
                timestamped  = False,
                n_households = sc.get("n_households", args.n_households),
            )

        if not args.no_plot:
            try:
                from bench_v4.plotting import plot_multi_scenario
                print("\nGenerating multi-scenario comparison plots...")
                plot_multi_scenario(parent_dir)
            except ImportError as e:
                print(f"Multi-scenario plotting skipped (missing dependency: {e})")

        print(f"\nAll done.  Results in: {parent_dir}")

    else:
        single = args.runs == 1
        slug   = _LEARNING_SLUG.get(args.learning, args.learning.replace(" ", "_"))
        label  = f"{args.case}_{slug}"

        config_dir = _make_config_dir(args.output_dir, label)
        runs_dir   = os.path.join(config_dir, "runs")
        os.makedirs(runs_dir, exist_ok=True)
        print(f"Output folder: {config_dir}")
        if not single:
            print(f"Jobs         : {args.jobs}  (-1 = all available cores)")

        if single:
            seed  = args.seed
            model = _run_and_save(args.case, args.learning, seed, memory,
                                  "run_001", runs_dir, verbose=args.verbose,
                                  n_households=args.n_households)
            print(model.summary())
            all_models = [model]
        else:
            worker_args = [
                (args.case, args.learning, None, memory, f"run_{i+1:03d}", runs_dir,
                 False, args.n_households)
                for i in range(args.runs)
            ]
            all_models = Parallel(n_jobs=args.jobs)(
                delayed(_run_and_save)(*arg) for arg in worker_args
            )

        _print_mean_table(all_models, args.case, args.learning)

        if not args.no_plot:
            try:
                from bench_v4.plotting import plot_all
                plot_all(config_dir)
            except ImportError as e:
                print(f"Plotting skipped (missing dependency: {e})")

        print(f"\nDone.  All results in: {config_dir}")
    
    elapsed = datetime.now() - t0
    print(f"\nCompleted in {str(elapsed).split('.')[0]}")


if __name__ == "__main__":
    main()
