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

100-run ensemble (saves data + plots automatically):
    python main.py --case NL --learning Informative --runs 100

Run all scenarios from a YAML config file:
    python main.py --config configs/bench_v4_scenarios.yaml

Custom output directory:
    python main.py --case ES --runs 50 --output-dir results

Skip plotting (data only):
    python main.py --case NL --runs 20 --no-plot
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

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


def _run_single(case, learning, seed, memory, verbose):
    model = BENCHv4(case_study=case, seed=seed, learning=learning, memory=memory)
    model.run(verbose=verbose)
    return model


def _print_mean_table(all_models, case, learning):
    from collections import defaultdict
    accum = defaultdict(lambda: defaultdict(list))
    for model in all_models:
        n_hh = len(model.households)
        for s in model.history:
            pct = 100 * s.n_renovated / n_hh if n_hh else 0
            accum["all"][s.year].append(pct)
            for cat in (1, 2, 3):
                t = s.total_by_dwage.get(cat, 0)
                p = 100 * s.renov_by_dwage.get(cat, 0) / t if t else 0
                accum[f"dwage{cat}"][s.year].append(p)

    n = len(all_models)
    #print(f"\nMean renovation rate  ({case} / {learning} / N={n} runs)")
    #print(f"{'Year':>6}  {'All%':>7}  {'New%':>7}  {'Mid%':>7}  {'Old%':>7}")
    for yr in sorted(accum["all"].keys()):
        def m(key):
            v = accum[key][yr]
            return sum(v) / len(v) if v else 0.0
        #print(f"{yr:>6}  {m('all'):>7.2f}  {m('dwage1'):>7.2f}  "
        #      f"{m('dwage2'):>7.2f}  {m('dwage3'):>7.2f}")


def _run_scenario(case, learning, runs, memory, output_dir,
                  run_label, verbose, no_plot, timestamped=True):
    """Execute one scenario (possibly many seed runs) and save outputs."""
    slug  = _LEARNING_SLUG.get(learning, learning.replace(" ", "_"))
    label = run_label if run_label else f"{case}_{slug}"
    if timestamped:
        config_dir = _make_config_dir(output_dir, label)
    else:
        config_dir = os.path.join(output_dir, label)
        os.makedirs(config_dir, exist_ok=True)
    runs_dir    = os.path.join(config_dir, "runs")
    os.makedirs(runs_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Scenario : {case} / {learning}")
    print(f"Runs     : {runs}")
    print(f"Output   : {config_dir}")
    print(f"{'='*60}")

    single_run  = runs == 1
    all_models  = []

    for i in range(runs):
        seed      = None   # always random for ensembles
        run_label_i = f"run_{i+1:03d}"

        model = _run_single(case, learning, seed, memory,
                            verbose=verbose and single_run)
        all_models.append(model)

        run_dir = os.path.join(runs_dir, f"{run_label_i}_seed_{model.seed}")
        save_run(model, run_dir)

        if single_run:
            print(model.summary())
        else:
            n_hh = len(model.households)
            pct  = 100 * sum(s.n_renovated for s in model.history) / (
                       n_hh * len(model.history)) if n_hh and model.history else 0
            #print(f"  {run_label_i}  seed={model.seed}  "
            #      f"mean_annual_renov={pct:.2f}%")

    if not single_run:
        _print_mean_table(all_models, case, learning)

    if not no_plot:
        try:
            from bench_v4.plotting import plot_all
            #print("\nGenerating plots...")
            plot_all(config_dir)
        except ImportError as e:
            print(f"Plotting skipped (missing dependency: {e})")

    #print(f"Done -> {config_dir}")
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

    if args.config:
        scenarios   = _load_yaml(args.config)
        config_stem = Path(args.config).stem
        parent_dir  = _make_config_dir(args.output_dir, config_stem)
        total_runs  = sum(sc.get("runs", 1) for sc in scenarios)
        print(f"Config run folder : {parent_dir}")
        print(f"Scenarios         : {len(scenarios)}")
        print(f"Total runs        : {total_runs}")
        for sc in scenarios:
            _run_scenario(
                case       = sc.get("case_study", "NL"),
                learning   = sc.get("learning", "Informative"),
                runs       = sc.get("runs", 1),
                memory     = sc.get("memory", True),
                output_dir = parent_dir,
                run_label  = sc.get("run_label", None),
                verbose    = args.verbose,
                no_plot    = args.no_plot,
                timestamped= False,
            )

        # multi-scenario comparison plots (all scenarios combined)
        if not args.no_plot:
            try:
                from bench_v4.plotting import plot_multi_scenario
                print("\nGenerating multi-scenario comparison plots...")
                plot_multi_scenario(parent_dir)
            except ImportError as e:
                print(f"Multi-scenario plotting skipped (missing dependency: {e})")

        print(f"\nAll done.  Results in: {parent_dir}")
    else:
        # honour --seed only for single runs (ignored for ensembles)
        single = args.runs == 1
        slug   = _LEARNING_SLUG.get(args.learning, args.learning.replace(" ", "_"))
        label  = f"{args.case}_{slug}"

        config_dir = _make_config_dir(args.output_dir, label)
        runs_dir   = os.path.join(config_dir, "runs")
        os.makedirs(runs_dir, exist_ok=True)
        print(f"Output folder: {config_dir}")

        all_models = []
        for i in range(args.runs):
            seed        = args.seed if (single and args.seed is not None) else None
            run_label_i = f"run_{i+1:03d}"

            if args.verbose or single:
                print(f"\n--- {run_label_i}/{args.runs}  case={args.case}  "
                      f"learning={args.learning} ---")

            model = _run_single(args.case, args.learning, seed, memory,
                                verbose=args.verbose)
            all_models.append(model)

            run_dir = os.path.join(runs_dir, f"{run_label_i}_seed_{model.seed}")
            save_run(model, run_dir)

            if single:
                print(model.summary())
            else:
                n_hh = len(model.households)
                pct  = 100 * sum(s.n_renovated for s in model.history) / (
                           n_hh * len(model.history)) if n_hh and model.history else 0
                #print(f"  {run_label_i}  seed={model.seed}  "
                #      f"mean_annual_renov={pct:.2f}%  -> {run_dir}")

        if not single:
            _print_mean_table(all_models, args.case, args.learning)

        if not args.no_plot:
            try:
                from bench_v4.plotting import plot_all
                #print("\nGenerating plots...")
                plot_all(config_dir)
            except ImportError as e:
                print(f"Plotting skipped (missing dependency: {e})")

        print(f"\nDone.  All results in: {config_dir}")


if __name__ == "__main__":
    main()
