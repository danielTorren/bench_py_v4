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
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).parent))

from bench_v4 import BENCHv4
from bench_v4.output import save_run
from bench_v4.params import N_HOUSEHOLDS


def _load_population(pop_type: str, n_households: int, case: str = "NL",
                     copula_seed: int = 1):
    """Build the initial population DataFrame for a single scenario.

    Returns a DataFrame (passed to BENCHv4 via population_df) or None
    (falls back to the original GroupParams sampling).

    pop_type choices:
      groupparams  – original hardcoded group distributions (returns None)
      copula       – Gaussian copula fitted from the real survey; n_households
                     synthetic agents generated on the fly
      survey755    – the real survey respondents used directly (n_households
                     is ignored; BENCHv4 receives all respondents)
    """
    if pop_type == "groupparams":
        return None

    _SURVEY_DIR = Path(__file__).parent / "survey_population"
    sys.path.insert(0, str(_SURVEY_DIR))
    try:
        from build_population import load_real_population, generate_from_copula  # type: ignore
    except ImportError as exc:
        print(f"  WARNING: survey_population modules not importable ({exc})."
              f"  Falling back to GroupParams.")
        return None

    try:
        df_real = load_real_population(case)
        print(f"  Survey data loaded: {len(df_real)} valid respondents ({case}).")
    except Exception as exc:
        print(f"  WARNING: Could not load survey data ({exc})."
              f"  Falling back to GroupParams.")
        return None

    if pop_type == "survey755":
        print(f"  Using the {len(df_real)} real survey respondents as agents.")
        return df_real

    # copula: generate n_households synthetic agents
    rng    = np.random.default_rng(copula_seed)
    pop_df = generate_from_copula(df_real, n_households, rng)
    print(f"  Copula population generated: {len(pop_df):,} agents (seed={copula_seed}).")
    return pop_df


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
                  verbose=False, n_households=None, population_df=None):
    """Worker: run one model instance, save outputs, return the model."""
    model = BENCHv4(case_study=case, seed=seed, learning=learning, memory=memory,
                    n_households=n_households, population_df=population_df)
    model.run(verbose=verbose)
    run_dir = os.path.join(runs_dir, f"{run_label_i}_seed_{model.seed}")
    save_run(model, run_dir)
    return model


def _run_and_save_tagged(si, case, learning, seed, memory, run_label_i, runs_dir,
                         n_households=None, population_df=None):
    """Like _run_and_save but returns (scenario_index, model) for flat parallel dispatch."""
    model = _run_and_save(case, learning, seed, memory, run_label_i, runs_dir,
                          n_households=n_households, population_df=population_df)
    return si, model


def _run_batch_tagged(batch_args: list) -> list:
    """Run a batch of tagged jobs sequentially; reduces serialization overhead."""
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
    parser.add_argument("--population", default="groupparams",
                        choices=["groupparams", "copula", "survey755"],
                        help="Initial population source: groupparams (default), "
                             "copula (Gaussian copula from real survey), or "
                             "survey755 (755 real respondents as agents)")
    parser.add_argument("--copula-seed", type=int, default=1,
                        help="RNG seed for copula sampling (default: 1)")
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

        # Save the config file used for this run (reproducibility)
        shutil.copy(args.config, os.path.join(parent_dir, Path(args.config).name))

        # Pre-create per-scenario directories and build a flat job list across
        # all scenarios × seeds so the entire batch runs in one parallel pool
        # (no nested parallelism).
        sc_dirs: list = []
        worker_args: list = []
        saved_pops: set = set()   # (case, pop_type) pairs already written to disk

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
            n_hh      = sc.get("n_households", args.n_households)
            runs      = sc.get("runs", 1)

            # Population: per-scenario key overrides CLI flag
            pop_type   = sc.get("population", args.population)
            cop_seed   = sc.get("copula_seed", args.copula_seed)
            eff_n      = n_hh or N_HOUSEHOLDS.get(case, 10_000)
            pop_df     = _load_population(pop_type, eff_n, case, cop_seed)

            # Save one population CSV per (case, pop_type) — same population is
            # reused for all seeds and scenarios with the same case study.
            if pop_df is not None and (case, pop_type) not in saved_pops:
                pop_path = os.path.join(parent_dir, f"population_{case}_{pop_type}.csv")
                pop_df.to_csv(pop_path, index=False)
                saved_pops.add((case, pop_type))
                print(f"  Population saved : {pop_path}  ({len(pop_df):,} agents)")

            for i in range(runs):
                worker_args.append(
                    (si, case, learning, None, memory_sc,
                     f"run_{i+1:03d}", runs_dir, n_hh, pop_df)
                )

        # Batch jobs 4-per-core: each worker gets a chunk of runs so population_df
        # is serialised once per batch rather than once per individual run.
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

    else:
            parser.error("--config is required. Use: python main.py --config <yaml_file>")
    
    elapsed = datetime.now() - t0
    print(f"\nCompleted in {str(elapsed).split('.')[0]}")


if __name__ == "__main__":
    main()
