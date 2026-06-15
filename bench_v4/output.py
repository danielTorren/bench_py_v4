"""Per-run output saving for BENCH v4."""

import csv
import json
import os


def save_run(model, run_dir: str) -> None:
    """Save all outputs for a single completed model run to run_dir."""
    os.makedirs(run_dir, exist_ok=True)
    _save_annual_results(model, run_dir)
    _save_run_config(model, run_dir)
    _save_summary(model, run_dir)


def _pct_by_dwage(s, cat: int) -> float:
    t = s.total_by_dwage.get(cat, 0)
    return 100.0 * s.renov_by_dwage.get(cat, 0) / t if t else 0.0


def _pct_by_group(s, g: int) -> float:
    t = s.total_by_group.get(g, 0)
    return 100.0 * s.renov_by_group.get(g, 0) / t if t else 0.0


def _save_annual_results(model, run_dir: str) -> None:
    path = os.path.join(run_dir, "annual_results.csv")
    n_hh = model.n_households
    cum = {1: 0.0, 2: 0.0, 3: 0.0}

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "year",
            "n_renovated", "n_conservation", "n_switching",
            "pct_renovated", "pct_conservation", "pct_switching",
            "renov_pct_dwage1", "renov_pct_dwage2", "renov_pct_dwage3",
            "renov_cum_pct_dwage1", "renov_cum_pct_dwage2", "renov_cum_pct_dwage3",
            "renov_pct_grp1", "renov_pct_grp2", "renov_pct_grp3",
            "renov_pct_grp4", "renov_pct_grp5",
            "total_gas_saved_kwh", "total_energy_conservation_kwh",
            "total_energy_switching_kwh",
            "total_investment_eur", "total_invest_conservation_eur",
            "total_invest_switching_eur",
            "avg_aware", "avg_pn1", "avg_sn1",
            "high_guilt_pct", "high_m1_pct", "high_m2_pct", "high_m3_pct",
        ])
        for s in model.history:
            pct     = 100.0 * s.n_renovated    / n_hh if n_hh else 0.0
            pct_con = 100.0 * s.n_conservation / n_hh if n_hh else 0.0
            pct_sw  = 100.0 * s.n_switching    / n_hh if n_hh else 0.0

            for cat in (1, 2, 3):
                cum[cat] += _pct_by_dwage(s, cat)

            writer.writerow([
                s.year,
                s.n_renovated, s.n_conservation, s.n_switching,
                round(pct, 4), round(pct_con, 4), round(pct_sw, 4),
                round(_pct_by_dwage(s, 1), 4), round(_pct_by_dwage(s, 2), 4), round(_pct_by_dwage(s, 3), 4),
                round(cum[1], 4), round(cum[2], 4), round(cum[3], 4),
                round(_pct_by_group(s, 1), 4), round(_pct_by_group(s, 2), 4), round(_pct_by_group(s, 3), 4),
                round(_pct_by_group(s, 4), 4), round(_pct_by_group(s, 5), 4),
                round(s.total_gas_saved, 2),
                round(s.total_energy_conservation, 2),
                round(s.total_energy_switching, 2),
                round(s.total_investment, 2),
                round(s.total_invest_conservation, 2),
                round(s.total_invest_switching, 2),
                round(s.avg_aware, 4), round(s.avg_pn1, 4), round(s.avg_sn1, 4),
                round(s.high_guilt_pct, 4), round(s.high_m1_pct, 4),
                round(s.high_m2_pct, 4), round(s.high_m3_pct, 4),
            ])


def _save_run_config(model, run_dir: str) -> None:
    path = os.path.join(run_dir, "run_config.json")
    config = {
        "case_study":    model.case_study,
        "learning":      model.learning,
        "seed":          model.seed,
        "n_households":  model.n_households,
        "memory":        model.memory_on,
        "start_year":    model.history[0].year  if model.history else None,
        "end_year":      model.history[-1].year if model.history else None,
    }
    with open(path, "w") as f:
        json.dump(config, f, indent=2)


def _save_summary(model, run_dir: str) -> None:
    path = os.path.join(run_dir, "summary.txt")
    with open(path, "w") as f:
        f.write(model.summary())
        f.write("\n")
