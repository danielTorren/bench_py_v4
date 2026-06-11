"""
Sobol sensitivity analysis plotter for BENCH v4.

Reads the sa_indices.json produced by run_sa.py and saves two types of plot
to a plots/ subfolder inside the results directory:

  sobol_{output_name}.png   Horizontal grouped bar chart for each output metric
                            showing S1 (first-order) and ST (total-order) indices
                            with 95% confidence interval error bars.

  sobol_summary.png         Combined grid figure: rows = output metrics,
                            columns = S1 / ST, one bar per parameter.

Usage:
    python sensitivity/plot_sa.py --results sensitivity/results/<run_folder>
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

_PARAM_LABELS = {
    "guilt_thresh":       "Guilt threshold",
    "motivation_m1_pn":   "Invest. motiv. — pn thresh",
    "motivation_m1_sn":   "Invest. motiv. — sn thresh",
    "motivation_m2_pn":   "Conserv. motiv. — pn thresh",
    "motivation_m2_sn":   "Conserv. motiv. — sn thresh",
    "motivation_m3_pn":   "Switch. motiv. — pn thresh",
    "motivation_m3_sn":   "Switch. motiv. — sn thresh",
    "pbc_invest":         "PBC invest threshold",
    "pbc_conserv":        "PBC conservation threshold",
    "pbc_switch":         "PBC switching threshold",
}

_OUTPUT_LABELS = {
    "mean_annual_renov_pct": "Mean annual renovation rate (%)",
    "total_energy_saved_MWh": "Total energy saved (MWh)",
    "cum_renovations":       "Cumulative renovations (count)",
}


def _plabel(name):
    return _PARAM_LABELS.get(name, name)

def _olabel(name):
    return _OUTPUT_LABELS.get(name, name)


# ---------------------------------------------------------------------------
# Individual output plots
# ---------------------------------------------------------------------------

def _plot_one(ax, param_names, indices_for_output, title):
    n = len(param_names)
    y = np.arange(n)
    bar_h = 0.35

    S1      = np.array(indices_for_output["S1"])
    ST      = np.array(indices_for_output["ST"])
    S1_conf = np.array(indices_for_output["S1_conf"])
    ST_conf = np.array(indices_for_output["ST_conf"])

    ax.barh(y + bar_h / 2, ST, bar_h, xerr=ST_conf, capsize=3,
            label="ST  (total order)", color="darkorange", alpha=0.85,
            error_kw={"elinewidth": 1.2, "ecolor": "black"})
    ax.barh(y - bar_h / 2, S1, bar_h, xerr=S1_conf, capsize=3,
            label="S1  (first order)", color="steelblue", alpha=0.90,
            error_kw={"elinewidth": 1.2, "ecolor": "black"})

    ax.set_yticks(y)
    ax.set_yticklabels([_plabel(p) for p in param_names], fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_xlabel("Sobol sensitivity index")
    ax.set_title(title, fontsize=10, pad=6)
    ax.legend(fontsize=8, loc="lower right")

    x_lo = min(-0.05, float(np.nanmin(np.concatenate([S1, ST]))) - 0.05)
    ax.set_xlim(left=x_lo)


def plot_individual(results_dir, doc, plots_dir):
    param_names  = doc["param_names"]
    output_names = doc["output_names"]
    indices      = doc["indices"]
    n_params     = len(param_names)

    for out_name in output_names:
        fig_h = max(4.0, 0.55 * n_params + 1.5)
        fig, ax = plt.subplots(figsize=(9, fig_h))

        meta  = f"{doc['case_study']} / {doc['learning']}  "
        meta += f"(N={doc['n_samples']}, seeds={doc['n_seeds']})"
        title = f"{_olabel(out_name)}\n{meta}"

        _plot_one(ax, param_names, indices[out_name], title)
        plt.tight_layout()

        fname = os.path.join(plots_dir, f"sobol_{out_name}.png")
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: sobol_{out_name}.png")


# ---------------------------------------------------------------------------
# Combined summary figure
# ---------------------------------------------------------------------------

def plot_summary(results_dir, doc, plots_dir):
    param_names  = doc["param_names"]
    output_names = doc["output_names"]
    indices      = doc["indices"]
    n_out        = len(output_names)
    n_params     = len(param_names)

    fig_h = max(5.0, 0.55 * n_params + 1.0) * n_out
    fig, axes = plt.subplots(n_out, 2, figsize=(14, fig_h),
                             squeeze=False)

    meta = (f"{doc['case_study']} / {doc['learning']}  "
            f"N={doc['n_samples']}  seeds={doc['n_seeds']}")
    fig.suptitle(f"Sobol sensitivity analysis — {meta}", fontsize=11, y=1.01)

    for row, out_name in enumerate(output_names):
        idx = indices[out_name]
        y       = np.arange(n_params)
        S1      = np.array(idx["S1"])
        ST      = np.array(idx["ST"])
        S1_conf = np.array(idx["S1_conf"])
        ST_conf = np.array(idx["ST_conf"])
        labels  = [_plabel(p) for p in param_names]

        # S1 subplot
        ax_s1 = axes[row][0]
        ax_s1.barh(y, S1, 0.6, xerr=S1_conf, capsize=3,
                   color="steelblue", alpha=0.90,
                   error_kw={"elinewidth": 1.2, "ecolor": "black"})
        ax_s1.set_yticks(y)
        ax_s1.set_yticklabels(labels, fontsize=8)
        ax_s1.axvline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.5)
        ax_s1.set_xlabel("S1 (first order)")
        ax_s1.set_title(_olabel(out_name), fontsize=9)

        # ST subplot
        ax_st = axes[row][1]
        ax_st.barh(y, ST, 0.6, xerr=ST_conf, capsize=3,
                   color="darkorange", alpha=0.85,
                   error_kw={"elinewidth": 1.2, "ecolor": "black"})
        ax_st.set_yticks(y)
        ax_st.set_yticklabels(labels, fontsize=8)
        ax_st.axvline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.5)
        ax_st.set_xlabel("ST (total order)")
        ax_st.set_title(_olabel(out_name), fontsize=9)

    plt.tight_layout()
    fname = os.path.join(plots_dir, "sobol_summary.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: sobol_summary.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot Sobol SA results for BENCH v4")
    parser.add_argument("--results", required=True,
                        help="Path to SA results folder (contains sa_indices.json)")
    args = parser.parse_args()

    indices_path = os.path.join(args.results, "sa_indices.json")
    if not os.path.exists(indices_path):
        print(f"ERROR: sa_indices.json not found in {args.results}")
        sys.exit(1)

    with open(indices_path) as f:
        doc = json.load(f)

    plots_dir = os.path.join(args.results, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    print(f"Results : {args.results}")
    print(f"Config  : {doc['case_study']} / {doc['learning']}  "
          f"n_samples={doc['n_samples']}  n_seeds={doc['n_seeds']}")
    print(f"Params  : {len(doc['param_names'])}")
    print(f"Outputs : {doc['output_names']}")
    print()

    plot_individual(args.results, doc, plots_dir)
    plot_summary(args.results, doc, plots_dir)

    print(f"\nAll plots saved to: {plots_dir}")


if __name__ == "__main__":
    main()
