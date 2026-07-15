"""
Sobol sensitivity analysis runner for BENCH v4.

Generates a Saltelli sample matrix, runs the model once per (parameter set,
seed) pair in parallel, averages across seeds per parameter set, then computes
first-order (S1) and total-order (ST) Sobol indices for three output metrics:
  - mean_annual_renov_pct   : mean annual renovation rate (%) over 2016-2050
  - total_energy_saved_MWh  : cumulative energy savings across all years (MWh)
  - cum_renovations         : total renovation events summed over all years

Parallelism
-----------
All n_eval * n_seeds model runs are submitted as individual jobs to
joblib.Parallel, so both evaluations AND seeds are parallelised simultaneously.
Worker count comes from n_jobs in the YAML config (default -1 = all cores) and
can be overridden at the command line with --workers.

How parameter injection works
------------------------------
Behavioural thresholds in bench_v4/params.py are imported by model.py at module
load time.  Dict-valued parameters (GUILT_THRESH, MOTIVATION_THRESH, etc.) are
shared mutable objects, so modifying them in-place is visible to all subsequent
model.run() calls without re-importing the module.  The scalar PBC_CONSERV_THRESH
is patched directly on the bench_v4.model module namespace, which is the global
scope that _consideration() looks up at call time.  Both changes are reversed
inside a context manager after each run.  This is safe in multiprocessing /
joblib because each worker process has its own copy of the module namespace.

Usage
-----
    python sensitivity/run_sa.py
    python sensitivity/run_sa.py --config sensitivity/sa_config.yaml
    python sensitivity/run_sa.py --workers 16   # override n_jobs from config
    python sensitivity/run_sa.py --dry-run      # print counts, skip model runs
"""

import argparse
import csv
import json
import os
import shutil
import sys
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).parent.parent))

import bench_v4.model as _m
from bench_v4 import BENCHv4


# ---------------------------------------------------------------------------
# Parameter patching helpers
# ---------------------------------------------------------------------------

def _save_originals(case_study):
    return {
        "guilt_thresh":  _m.GUILT_THRESH[case_study],
        "m1":            _m.MOTIVATION_THRESH[case_study]["m1"],
        "m2":            _m.MOTIVATION_THRESH[case_study]["m2"],
        "m3":            _m.MOTIVATION_THRESH[case_study]["m3"],
        "pbc_invest":    _m.PBC_INVEST_THRESH[case_study],
        "pbc_conserv":   _m.PBC_CONSERV_THRESH,
        "pbc_switch":    _m.PBC_SWITCH_THRESH[case_study],
        "utility_coef":  dict(_m.UTILITY_COEF),   # shallow copy — dict is mutable
        "learning_rate": _m.LEARNING_RATE,
    }


def _apply_params(case_study, pdict):
    if "guilt_thresh" in pdict:
        _m.GUILT_THRESH[case_study] = pdict["guilt_thresh"]
    for mkey, pn_key, sn_key in [
        ("m1", "motivation_m1_pn", "motivation_m1_sn"),
        ("m2", "motivation_m2_pn", "motivation_m2_sn"),
        ("m3", "motivation_m3_pn", "motivation_m3_sn"),
    ]:
        if pn_key in pdict or sn_key in pdict:
            orig_pn, orig_sn = _m.MOTIVATION_THRESH[case_study][mkey]
            new_pn = pdict.get(pn_key, orig_pn)
            new_sn = pdict.get(sn_key, orig_sn)
            _m.MOTIVATION_THRESH[case_study][mkey] = (new_pn, new_sn)
    if "pbc_invest" in pdict:
        _m.PBC_INVEST_THRESH[case_study] = pdict["pbc_invest"]
    if "pbc_conserv" in pdict:
        _m.PBC_CONSERV_THRESH = pdict["pbc_conserv"]
    if "pbc_switch" in pdict:
        _m.PBC_SWITCH_THRESH[case_study] = pdict["pbc_switch"]
    # Utility-function coefficients — patch in-place so model.UTILITY_COEF reflects change
    for coef_key in ("dw_type", "dw_age", "edu", "pn1", "gas"):
        param_name = f"coef_{coef_key}"
        if param_name in pdict:
            _m.UTILITY_COEF[coef_key] = pdict[param_name]
    if "learning_rate" in pdict:
        _m.LEARNING_RATE = pdict["learning_rate"]


def _restore_originals(case_study, orig):
    _m.GUILT_THRESH[case_study] = orig["guilt_thresh"]
    for mkey in ("m1", "m2", "m3"):
        _m.MOTIVATION_THRESH[case_study][mkey] = orig[mkey]
    _m.PBC_INVEST_THRESH[case_study] = orig["pbc_invest"]
    _m.PBC_CONSERV_THRESH            = orig["pbc_conserv"]
    _m.PBC_SWITCH_THRESH[case_study] = orig["pbc_switch"]
    _m.UTILITY_COEF.update(orig["utility_coef"])
    _m.LEARNING_RATE = orig["learning_rate"]


@contextmanager
def _patched(case_study, pdict):
    orig = _save_originals(case_study)
    _apply_params(case_study, pdict)
    try:
        yield
    finally:
        _restore_originals(case_study, orig)


# ---------------------------------------------------------------------------
# Model execution helpers
# ---------------------------------------------------------------------------

def _extract_outputs(model):
    n_hh  = model.n_households
    n_yrs = len(model.history)
    if n_hh == 0 or n_yrs == 0:
        return {"mean_annual_renov_pct": 0.0,
                "total_energy_saved_MWh": 0.0,
                "cum_renovations": 0.0}
    return {
        "mean_annual_renov_pct": (
            sum(s.n_renovated for s in model.history) / (n_hh * n_yrs) * 100.0
        ),
        # Sum all three energy-saving channels; conservation and switching are 0 in v4
        "total_energy_saved_MWh": (
            sum(s.total_gas_saved + s.total_energy_conservation + s.total_energy_switching
                for s in model.history) / 1000.0
        ),
        "cum_renovations": float(sum(s.n_renovated for s in model.history)),
    }


def _run_one(args):
    """Run a single (eval_idx, seed) model instance and return (eval_idx, outputs)."""
    case_study, learning, memory, eval_idx, seed, param_names, param_row, n_households, population_df = args
    pdict = dict(zip(param_names, param_row))
    with _patched(case_study, pdict):
        model = BENCHv4(case_study=case_study, learning=learning,
                        memory=memory, seed=seed, n_households=n_households,
                        population_df=population_df)
        model.run()
        return eval_idx, _extract_outputs(model)


def _run_batch(args_list: list) -> list:
    """Run a batch of _run_one jobs sequentially; reduces serialization overhead."""
    return [_run_one(a) for a in args_list]


# ---------------------------------------------------------------------------
# SALib wrappers (handle v1 and v2 APIs)
# ---------------------------------------------------------------------------

def _saltelli_sample(problem, n_samples, calc_second_order):
    try:
        from SALib.sample.sobol import sample
        return sample(problem, n_samples, calc_second_order=calc_second_order)
    except ImportError:
        try:
            from SALib.sample.saltelli import sample
            return sample(problem, n_samples, calc_second_order=calc_second_order)
        except ImportError:
            from SALib.sample import saltelli
            return saltelli.sample(problem, n_samples,
                                   calc_second_order=calc_second_order)


def _sobol_analyze(problem, Y, calc_second_order):
    try:
        from SALib.analyze.sobol import analyze
        return analyze(problem, Y, calc_second_order=calc_second_order,
                       print_to_console=False)
    except ImportError:
        from SALib.analyze import sobol
        return sobol.analyze(problem, Y, calc_second_order=calc_second_order,
                             print_to_console=False)


def _write_csv(path, headers, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Sobol SA for BENCH v4")
    parser.add_argument("--config", default="sensitivity/sa_config.yaml",
                        help="YAML config file (default: sensitivity/sa_config.yaml)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (overrides n_jobs in config; "
                             "-1 = all available cores)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print run counts and exit without running models")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    case_study   = cfg.get("case_study", "NL")
    learning     = cfg.get("learning", "Informative")
    memory       = cfg.get("memory", True)
    n_households = cfg.get("n_households", None)
    pop_type     = cfg.get("population", "groupparams")
    copula_seed  = cfg.get("copula_seed", 1)
    n_samples    = cfg.get("n_samples", 64)
    n_seeds      = cfg.get("n_seeds", 5)
    calc_s2      = cfg.get("calc_second_order", False)
    output_dir   = cfg.get("output_dir", "output")
    parameters   = cfg["parameters"]
    param_names  = [p["name"]   for p in parameters]
    param_bounds = [p["bounds"] for p in parameters]
    n_params     = len(parameters)

    n_jobs = args.workers if args.workers is not None else cfg.get("n_jobs", -1)

    problem = {"num_vars": n_params, "names": param_names, "bounds": param_bounds}

    try:
        param_matrix = _saltelli_sample(problem, n_samples, calc_s2)
    except Exception as exc:
        print(f"ERROR: could not import SALib ({exc})")
        print("Install with: uv add salib")
        sys.exit(1)

    n_eval = len(param_matrix)
    n_runs = n_eval * n_seeds

    print(f"Sobol SA    : {case_study} / {learning}")
    print(f"Households  : {n_households or 'survey default'}")
    print(f"Parameters  : {n_params}")
    print(f"Evals     : {n_eval}  (n_samples={n_samples}, calc_second_order={calc_s2})")
    print(f"Seeds/eval: {n_seeds}")
    print(f"Total runs: {n_runs}  (parallelised over evals x seeds)")
    print(f"Jobs      : {n_jobs}  (-1 = all available cores)")

    if args.dry_run:
        print("\n[dry-run] Parameter names and bounds:")
        for p in parameters:
            print(f"  {p['name']:30s}  {p['bounds']}")
        return

    config_stem = Path(args.config).stem
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir     = os.path.join(output_dir, f"{config_stem}_{case_study}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"Output dir: {run_dir}\n")

    # Save the config file used for this run (reproducibility)
    shutil.copy(args.config, os.path.join(run_dir, Path(args.config).name))

    # --- Build population (same for all parameter evaluations) ---
    population_df = None
    if pop_type != "groupparams":
        _survey_dir = Path(__file__).parent.parent / "survey_population"
        sys.path.insert(0, str(_survey_dir))
        try:
            from build_population import load_real_population, generate_from_copula  # type: ignore
            df_real = load_real_population(case_study)
            if pop_type == "survey755":
                population_df = df_real
                n_households  = n_households or len(df_real)
                print(f"Population  : survey755  ({len(population_df)} real respondents)")
            elif pop_type == "copula":
                eff_n = n_households or len(df_real)
                rng   = np.random.default_rng(copula_seed)
                population_df = generate_from_copula(df_real, eff_n, rng)
                n_households  = eff_n
                print(f"Population  : copula  ({len(population_df):,} synthetic agents, seed={copula_seed})")
            # Save one copy for reproducibility
            pop_path = os.path.join(run_dir, f"population_{case_study}_{pop_type}.csv")
            population_df.to_csv(pop_path, index=False)
            print(f"  Saved: {pop_path}")
        except Exception as exc:
            print(f"WARNING: could not build {pop_type!r} population ({exc}). "
                  f"Falling back to GroupParams.")
            population_df = None
    else:
        print(f"Population  : groupparams (hardcoded prior)")

    # --- Build flat worker arg list: one job per (eval_idx, seed) pair ---
    worker_args = [
        (case_study, learning, memory, i, i * n_seeds + s + 1,
         param_names, param_matrix[i].tolist(), n_households, population_df)
        for i in range(n_eval)
        for s in range(n_seeds)
    ]

    # Batch jobs 4-per-core: each worker gets a chunk of runs so population_df
    # is serialised once per batch rather than once per individual run.
    eff_workers = os.cpu_count() or 8
    if n_jobs not in (-1, None):
        eff_workers = max(1, abs(n_jobs))
    n_batches  = max(1, 4 * eff_workers)
    batch_size = max(1, (n_runs + n_batches - 1) // n_batches)
    batches    = [worker_args[i:i + batch_size] for i in range(0, n_runs, batch_size)]

    print(f"Running {n_runs} model runs in {len(batches)} batches "
          f"(~{batch_size} runs/batch, {eff_workers} workers) ...\n")
    t0 = datetime.now()
    raw_nested = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(_run_batch)(b) for b in batches
    )
    raw = [item for sublist in (raw_nested or []) for item in sublist]
    elapsed = datetime.now() - t0
    print(f"\nCompleted in {str(elapsed).split('.')[0]}")

    # --- Aggregate seeds per evaluation ---
    seed_results: dict[int, list] = defaultdict(list)
    for eval_idx, out in raw:
        seed_results[eval_idx].append(out)

    output_keys = ["mean_annual_renov_pct", "total_energy_saved_MWh", "cum_renovations"]
    all_outputs = {k: [] for k in output_keys}

    for i in range(n_eval):
        runs = seed_results[i]
        averaged = {k: sum(r[k] for r in runs) / len(runs) for k in output_keys}
        for k in output_keys:
            all_outputs[k].append(averaged[k])
        print(f"  [{i+1:4d}/{n_eval}]  "
              f"renov={averaged['mean_annual_renov_pct']:.2f}%  "
              f"energy={averaged['total_energy_saved_MWh']:.0f} MWh  "
              f"cum_renov={averaged['cum_renovations']:.0f}")

    # --- Save raw data ---
    sample_rows = [[f"{v:.6f}" for v in row] for row in param_matrix]
    _write_csv(os.path.join(run_dir, "samples.csv"), param_names, sample_rows)
    _write_csv(
        os.path.join(run_dir, "outputs.csv"),
        output_keys,
        [[f"{all_outputs[k][i]:.6f}" for k in output_keys] for i in range(n_eval)],
    )

    # --- Sobol analysis ---
    indices = {}
    for k in output_keys:
        Y  = np.array(all_outputs[k])
        Si = _sobol_analyze(problem, Y, calc_s2)
        indices[k] = {
            "S1":      Si["S1"].tolist(),
            "S1_conf": Si["S1_conf"].tolist(),
            "ST":      Si["ST"].tolist(),
            "ST_conf": Si["ST_conf"].tolist(),
        }
        if calc_s2:
            indices[k]["S2"]      = Si["S2"].tolist()
            indices[k]["S2_conf"] = Si["S2_conf"].tolist()

    result_doc = {
        "case_study":   case_study,
        "learning":     learning,
        "n_samples":    n_samples,
        "n_seeds":      n_seeds,
        "n_eval":       n_eval,
        "param_names":  param_names,
        "output_names": output_keys,
        "indices":      indices,
    }
    with open(os.path.join(run_dir, "sa_indices.json"), "w") as f:
        json.dump(result_doc, f, indent=2)

    print(f"\nSaved to {run_dir}:")
    print(f"  samples.csv      {n_eval} rows x {n_params} params")
    print(f"  outputs.csv      {n_eval} rows x {len(output_keys)} outputs")
    print(f"  sa_indices.json  S1 + ST for {len(output_keys)} outputs")
    
    
    print(f"\nPlot with:")
    print(f"  python sensitivity/plot_sa.py --results {run_dir}")


if __name__ == "__main__":
    main()
