"""
Batch plotting for BENCH v4 — mean ± 95% CI across seed runs.

Reads per-run CSVs from:
    <config_dir>/runs/run_*/annual_results.csv

Saves plots to:
    <config_dir>/plots/
"""

import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ---------------------------------------------------------------------------
# Visual config
# ---------------------------------------------------------------------------

VINTAGE_LABELS = {1: "New (<10 yr)", 2: "Middle (11-35 yr)", 3: "Old (>35 yr)"}
VINTAGE_COLORS = {1: "#2E86AB", 2: "#F18F01", 3: "#A23B72"}
VINTAGE_MARKERS = {1: "o", 2: "s", 3: "^"}

GROUP_LABELS = {1: "G1 (lowest)", 2: "G2", 3: "G3", 4: "G4", 5: "G5+ (highest)"}
GROUP_COLORS  = {1: "#2E86AB", 2: "#F18F01", 3: "#A23B72", 4: "#27AE60", 5: "#8E44AD"}
GROUP_MARKERS = {1: "o", 2: "s", 3: "^", 4: "D", 5: "P"}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def plot_all(config_dir: str) -> List[str]:
    """
    Generate all summary plots from the completed runs in config_dir.

    Folder layout expected:
        config_dir/runs/run_NNN_seed_XXXX/annual_results.csv
    Plots saved to:
        config_dir/plots/

    Returns list of saved file paths.
    """
    runs_dir  = os.path.join(config_dir, "runs")
    plots_dir = os.path.join(config_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    dfs = _load_runs(runs_dir)
    if not dfs:
        print(f"No run data found in {runs_dir}")
        return []

    years = dfs[0]["year"].values
    n     = len(dfs)
    saved = []

    saved += _plot_renovation_overall(dfs, years, n, plots_dir)
    saved += _plot_renovation_by_vintage(dfs, years, n, plots_dir)
    saved += _plot_renovation_cumulative_by_vintage(dfs, years, n, plots_dir)
    saved += _plot_renovation_by_income(dfs, years, n, plots_dir)
    saved += _plot_behaviour_count_by_type(dfs, years, n, plots_dir)
    saved += _plot_investment_by_type(dfs, years, n, plots_dir)
    saved += _plot_energy_saved(dfs, years, n, plots_dir)
    saved += _plot_motivation_over_time(dfs, years, n, plots_dir)
    saved += _plot_gas_savings(dfs, years, n, plots_dir)
    saved += _plot_avg_awareness(dfs, years, n, plots_dir)

    print(f"  Saved {len(saved)} plots -> {plots_dir}")
    return saved


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_runs(runs_dir: str) -> List[pd.DataFrame]:
    dfs = []
    if not os.path.isdir(runs_dir):
        return dfs
    for entry in sorted(os.scandir(runs_dir), key=lambda e: e.name):
        if entry.is_dir():
            csv_path = os.path.join(entry.path, "annual_results.csv")
            if os.path.exists(csv_path):
                dfs.append(pd.read_csv(csv_path))
    return dfs


def _ci(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (mean, lower_95, upper_95) across rows (runs) of matrix."""
    n = matrix.shape[0]
    mean = np.mean(matrix, axis=0)
    if n < 2:
        return mean, mean, mean
    std    = np.std(matrix, axis=0, ddof=1)
    margin = 1.96 * std / np.sqrt(n)
    return mean, mean - margin, mean + margin


def _fig_save(plots_dir: str, filename: str) -> str:
    path = os.path.join(plots_dir, filename)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path


def _add_ci_band(ax, x, lo, hi, color, alpha=0.15):
    ax.fill_between(x, lo, hi, color=color, alpha=alpha)


def _style_ax(ax, title: str, xlabel: str, ylabel: str, n_runs: int) -> None:
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.45)
    if n_runs > 1:
        note = f"Shaded band = 95% CI  (N={n_runs} runs)"
        ax.annotate(note, xy=(0.01, 0.01), xycoords="axes fraction",
                    fontsize=8, color="#666666", va="bottom")
    ax.legend(loc="best", frameon=True, facecolor="white",
              edgecolor="#d0d0d0", fontsize=9)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(5))


# ---------------------------------------------------------------------------
# Individual plot functions
# ---------------------------------------------------------------------------

def _plot_renovation_overall(dfs, years, n, plots_dir):
    mat = np.array([df["pct_renovated"].values for df in dfs])
    mean, lo, hi = _ci(mat)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    line, = ax.plot(years, mean, color="#2E86AB", linewidth=2,
                    marker="o", markersize=4, label=f"Mean (N={n})")
    if n > 1:
        _add_ci_band(ax, years, lo, hi, line.get_color())
    _style_ax(ax, "Annual Renovation Rate", "Year",
              "% Households Renovating", n)
    plt.tight_layout()
    return [_fig_save(plots_dir, "renovation_rate_overall.png")]


def _plot_renovation_by_vintage(dfs, years, n, plots_dir):
    cols = {1: "renov_pct_dwage1", 2: "renov_pct_dwage2", 3: "renov_pct_dwage3"}

    fig, ax = plt.subplots(figsize=(11, 6))
    for cat, col in cols.items():
        mat = np.array([df[col].values for df in dfs])
        mean, lo, hi = _ci(mat)
        color  = VINTAGE_COLORS[cat]
        marker = VINTAGE_MARKERS[cat]
        ax.plot(years, mean, color=color, linewidth=2, marker=marker,
                markersize=4, label=VINTAGE_LABELS[cat])
        if n > 1:
            _add_ci_band(ax, years, lo, hi, color)

    _style_ax(ax, "Annual Renovation Rate by Dwelling Vintage",
              "Year", "% Households Renovating", n)
    plt.tight_layout()
    return [_fig_save(plots_dir, "renovation_by_vintage.png")]


def _plot_renovation_cumulative_by_vintage(dfs, years, n, plots_dir):
    cols = {1: "renov_cum_pct_dwage1", 2: "renov_cum_pct_dwage2",
            3: "renov_cum_pct_dwage3"}

    fig, ax = plt.subplots(figsize=(11, 6))
    for cat, col in cols.items():
        mat = np.array([df[col].values for df in dfs])
        mean, lo, hi = _ci(mat)
        color  = VINTAGE_COLORS[cat]
        marker = VINTAGE_MARKERS[cat]
        ax.plot(years, mean, color=color, linewidth=2, marker=marker,
                markersize=4, label=VINTAGE_LABELS[cat])
        if n > 1:
            _add_ci_band(ax, years, lo, hi, color)

    _style_ax(ax, "Cumulative Renovation Rate by Dwelling Vintage",
              "Year", "Cumulative % Renovating", n)
    plt.tight_layout()
    return [_fig_save(plots_dir, "renovation_by_vintage_cumulative.png")]


def _plot_renovation_by_income(dfs, years, n, plots_dir):
    cols = {
        1: "renov_pct_grp1", 2: "renov_pct_grp2", 3: "renov_pct_grp3",
        4: "renov_pct_grp4", 5: "renov_pct_grp5",
    }

    fig, ax = plt.subplots(figsize=(11, 6))
    for g, col in cols.items():
        mat = np.array([df[col].values for df in dfs])
        mean, lo, hi = _ci(mat)
        color  = GROUP_COLORS[g]
        marker = GROUP_MARKERS[g]
        ax.plot(years, mean, color=color, linewidth=2, marker=marker,
                markersize=4, label=GROUP_LABELS[g])
        if n > 1:
            _add_ci_band(ax, years, lo, hi, color)

    _style_ax(ax, "Annual Renovation Rate by Income Group",
              "Year", "% Households Renovating", n)
    plt.tight_layout()
    return [_fig_save(plots_dir, "renovation_by_income.png")]


def _plot_gas_savings(dfs, years, n, plots_dir):
    mat = np.array([df["total_gas_saved_kwh"].values for df in dfs])
    mean, lo, hi = _ci(mat)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    line, = ax.plot(years, mean, color="#27AE60", linewidth=2,
                    marker="s", markersize=4, label=f"Mean (N={n})")
    if n > 1:
        _add_ci_band(ax, years, lo, hi, line.get_color())
    _style_ax(ax, "Annual Gas Savings", "Year", "Total Gas Saved (kWh)", n)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:,.0f}")
    )
    plt.tight_layout()
    return [_fig_save(plots_dir, "gas_savings_annual.png")]


def _plot_avg_awareness(dfs, years, n, plots_dir):
    mat = np.array([df["avg_aware"].values for df in dfs])
    mean, lo, hi = _ci(mat)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    line, = ax.plot(years, mean, color="#8E44AD", linewidth=2,
                    marker="^", markersize=4, label=f"Mean (N={n})")
    if n > 1:
        _add_ci_band(ax, years, lo, hi, line.get_color())
    ax.set_ylim(bottom=0, top=7)
    _style_ax(ax, "Average Household Awareness over Time",
              "Year", "Average Awareness (0-7 scale)", n)
    plt.tight_layout()
    return [_fig_save(plots_dir, "avg_awareness.png")]


# ---------------------------------------------------------------------------
# New plots: behaviour count, investment by type, energy saved, motivation
# ---------------------------------------------------------------------------

def _plot_behaviour_count_by_type(dfs, years, n, plots_dir):
    """Count of households undertaking each behaviour type per year."""
    specs = [
        ("n_renovated",    "Renovation (insulation)",   "#2E86AB", "o"),
        ("n_conservation", "Conservation (behaviour)",  "#F18F01", "s"),
        ("n_switching",    "Switching (energy source)", "#A23B72", "^"),
    ]
    fig, ax = plt.subplots(figsize=(11, 6))
    for col, label, color, marker in specs:
        if col not in dfs[0].columns:
            continue
        mat = np.array([df[col].values for df in dfs])
        mean, lo, hi = _ci(mat)
        ax.plot(years, mean, color=color, linewidth=2, marker=marker,
                markersize=4, label=label)
        if n > 1:
            _add_ci_band(ax, years, lo, hi, color)

    _style_ax(ax, "Household Behaviour Change Count by Type (95% CI)",
              "Year", "Number of Households", n)
    plt.tight_layout()
    return [_fig_save(plots_dir, "behaviour_count_by_type.png")]


def _plot_investment_by_type(dfs, years, n, plots_dir):
    """Annual investment expenditure by behaviour type (EUR)."""
    specs = [
        ("total_investment_eur",          "Renovation",   "#2E86AB", "o"),
        ("total_invest_conservation_eur", "Conservation", "#F18F01", "s"),
        ("total_invest_switching_eur",    "Switching",    "#A23B72", "^"),
    ]
    fig, ax = plt.subplots(figsize=(11, 6))
    for col, label, color, marker in specs:
        if col not in dfs[0].columns:
            continue
        mat = np.array([df[col].values for df in dfs])
        mean, lo, hi = _ci(mat)
        ax.plot(years, mean, color=color, linewidth=2, marker=marker,
                markersize=4, label=label)
        if n > 1:
            _add_ci_band(ax, years, lo, hi, color)

    _style_ax(ax, "Annual Investment by Behaviour Type (95% CI)",
              "Year", "Total Investment (EUR)", n)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:,.0f}")
    )
    plt.tight_layout()
    return [_fig_save(plots_dir, "investment_by_type.png")]


def _plot_energy_saved(dfs, years, n, plots_dir):
    """Annual energy saved by behaviour type (kWh)."""
    specs = [
        ("total_gas_saved_kwh",           "Renovation (gas)",  "#2E86AB", "o"),
        ("total_energy_conservation_kwh", "Conservation",      "#F18F01", "s"),
        ("total_energy_switching_kwh",    "Switching",         "#A23B72", "^"),
    ]
    fig, ax = plt.subplots(figsize=(11, 6))
    for col, label, color, marker in specs:
        if col not in dfs[0].columns:
            continue
        mat = np.array([df[col].values for df in dfs])
        mean, lo, hi = _ci(mat)
        ax.plot(years, mean, color=color, linewidth=2, marker=marker,
                markersize=4, label=label)
        if n > 1:
            _add_ci_band(ax, years, lo, hi, color)

    _style_ax(ax, "Annual Energy Saved by Behaviour Type (95% CI)",
              "Year", "Energy Saved (kWh)", n)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:,.0f}")
    )
    plt.tight_layout()
    return [_fig_save(plots_dir, "energy_saved_by_type.png")]


def _plot_motivation_over_time(dfs, years, n, plots_dir):
    """Percentage of households that are motivated (each type) over time."""
    specs = [
        ("high_guilt_pct", "Awareness (guilt=H)",           "#555555", "o"),
        ("high_m1_pct",    "Motivated: Renovation (m1=H)",  "#2E86AB", "s"),
        ("high_m2_pct",    "Motivated: Conservation (m2=H)","#F18F01", "^"),
        ("high_m3_pct",    "Motivated: Switching (m3=H)",   "#A23B72", "D"),
    ]
    fig, ax = plt.subplots(figsize=(11, 6))
    for col, label, color, marker in specs:
        if col not in dfs[0].columns:
            continue
        mat = np.array([df[col].values for df in dfs])
        mean, lo, hi = _ci(mat)
        ax.plot(years, mean, color=color, linewidth=2, marker=marker,
                markersize=4, label=label)
        if n > 1:
            _add_ci_band(ax, years, lo, hi, color)

    ax.set_ylim(bottom=0, top=100)
    _style_ax(ax, "Household Motivation over Time (95% CI)",
              "Year", "% Households", n)
    plt.tight_layout()
    return [_fig_save(plots_dir, "motivation_over_time.png")]
