#!/usr/bin/env python3
"""Turn results.csv (+ per-run alpha.csv) into presentation figures -> figures/*.png.

Five figures, one job each (no dual axes):
  1. reproduce_table1.png  — baseline left/right path split vs penalty p (the paper's Table 1)
  2. tradeoff.png          — zone-visit rate vs true return, per method (THE decision figure)
  3. alpha_trajectory.png  — learned α over training (shows the Lagrangian controller working)
  4. compute.png           — wall-clock train time per method (the "feel for Rechenzeit")
  5. signal_correctness.png— cost↔forecast-error correlation per method (the E2 decider)

    python plot_results.py --results results.csv --runs runs --out figures

Colors: Okabe–Ito (colorblind-safe), assigned to methods in a FIXED order so a
method keeps its color across every figure.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Okabe–Ito, fixed method -> hue (identity follows the entity, never its rank)
METHOD_COLOR = {
    "baseline": "#999999",  # grey = the paper reference
    "ens":      "#0072B2",  # blue
    "bnn":      "#56B4E9",  # light blue (sibling of ens — same axis)
    "lagr":     "#D55E00",  # vermillion
    "bnn+lagr": "#009E73",  # green (stretch combo)
}
METHOD_LABEL = {
    "baseline": "baseline (KDE + fixed p)",
    "ens":      "ensemble var",
    "bnn":      "BNN var",
    "lagr":     "Lagrangian α",
    "bnn+lagr": "BNN + Lagrangian",
}

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 150, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
})


def _order(methods):
    return [m for m in METHOD_COLOR if m in set(methods)]


def fig_table1(df, out):
    b = df[df.method == "baseline"]
    if b.empty:
        return None
    g = b.groupby("knob_value").agg(left=("left_path_pct", "mean"),
                                    right=("right_path_pct", "mean")).sort_index()
    ps = g.index.to_numpy()
    x = np.arange(len(ps))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - 0.19, g["left"], 0.36, label="left path %", color="#E69F00")
    ax.bar(x + 0.19, g["right"], 0.36, label="right path %", color="#0072B2")
    ax.set_xticks(x, [f"p={int(p)}" for p in ps])
    ax.set_ylabel("swing-ups through path (%)")
    ax.set_title("Reproducing paper Table 1: path split vs penalty")
    ax.legend(frameon=False)
    for xi, (l, r) in enumerate(zip(g["left"], g["right"])):
        ax.text(xi - 0.19, l + 1, f"{l:.0f}", ha="center", fontsize=9)
        ax.text(xi + 0.19, r + 1, f"{r:.0f}", ha="center", fontsize=9)
    fig.tight_layout(); fig.savefig(out / "reproduce_table1.png"); plt.close(fig)
    return "reproduce_table1.png"


def fig_tradeoff(df, out):
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    for m in _order(df.method.unique()):
        sub = df[df.method == m]
        # mean +/- std over seeds, per knob value
        g = sub.groupby("knob_value").agg(
            zx=("zone_step_frac", "mean"), zs=("zone_step_frac", "std"),
            ry=("true_return_mean", "mean"), rs=("true_return_mean", "std"),
            knob=("knob", "first"),
        ).reset_index()
        ax.errorbar(g["zx"], g["ry"], xerr=g["zs"].fillna(0), yerr=g["rs"].fillna(0),
                    fmt="o", ms=8, capsize=3, color=METHOD_COLOR[m],
                    label=METHOD_LABEL[m], lw=1.5)
        for _, row in g.iterrows():
            ax.annotate(f"{row['knob']}={row['knob_value']:g}",
                        (row["zx"], row["ry"]), textcoords="offset points",
                        xytext=(6, 4), fontsize=8, color=METHOD_COLOR[m])
    ax.set_xlabel("zone-visit rate  (← safer)")
    ax.set_ylabel("true return  (higher better →)")
    ax.set_title("Safety–performance trade-off (mean ± std over seeds)")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(out / "tradeoff.png"); plt.close(fig)
    return "tradeoff.png"


def fig_alpha(runs_dir, out):
    curves = []
    for a in sorted(runs_dir.glob("lagr*/alpha.csv")):
        vals = np.array([float(x) for x in a.read_text().split()])
        if len(vals) > 1:
            curves.append((a.parent.name, vals))
    if not curves:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, vals in curves:
        ax.plot(np.arange(len(vals)), vals, lw=2, color=METHOD_COLOR["lagr"], alpha=0.7)
    ax.set_xlabel("dual update step")
    ax.set_ylabel("learned multiplier α")
    ax.set_title("Lagrangian controller: α adapts to hold the constraint")
    fig.tight_layout(); fig.savefig(out / "alpha_trajectory.png"); plt.close(fig)
    return "alpha_trajectory.png"


def fig_compute(df, out):
    methods = _order(df.method.unique())
    g = df.groupby("method")["wall_clock_train_s"].mean()
    fig, ax = plt.subplots(figsize=(6, 3.6))
    y = np.arange(len(methods))
    ax.barh(y, [g[m] for m in methods], color=[METHOD_COLOR[m] for m in methods], height=0.6)
    ax.set_yticks(y, [METHOD_LABEL[m] for m in methods])
    ax.invert_yaxis()
    ax.set_xlabel("mean wall-clock training time (s)")
    ax.set_title("Compute cost per method")
    for yi, m in enumerate(methods):
        ax.text(g[m], yi, f" {g[m]:.0f}s", va="center", fontsize=9)
    fig.tight_layout(); fig.savefig(out / "compute.png"); plt.close(fig)
    return "compute.png"


def fig_signal(df, out):
    sub = df.dropna(subset=["cost_vs_forecast_err_corr"])
    if sub.empty:
        return None
    methods = _order(sub.method.unique())
    g = sub.groupby("method")["cost_vs_forecast_err_corr"].mean()
    fig, ax = plt.subplots(figsize=(6, 3.6))
    x = np.arange(len(methods))
    ax.bar(x, [g[m] for m in methods], color=[METHOD_COLOR[m] for m in methods], width=0.6)
    ax.axhline(0, color="#333333", lw=0.8)
    ax.set_xticks(x, [METHOD_LABEL[m] for m in methods], rotation=15, ha="right")
    ax.set_ylabel("Spearman(cost, forecast error)")
    ax.set_title("E2 decider: does the penalty fire where the model is wrong?")
    for xi, m in enumerate(methods):
        ax.text(xi, g[m], f"{g[m]:.2f}", ha="center",
                va="bottom" if g[m] >= 0 else "top", fontsize=9)
    fig.tight_layout(); fig.savefig(out / "signal_correctness.png"); plt.close(fig)
    return "signal_correctness.png"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results.csv")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()

    df = pd.read_csv(args.results)
    if "knob_value" not in df:  # older results.csv
        df["knob_value"] = df.get("hparam").map(lambda h: eval(h)["value"] if isinstance(h, str) else np.nan)
    out = Path(args.out); out.mkdir(exist_ok=True)

    made = [f for f in [
        fig_table1(df, out), fig_tradeoff(df, out), fig_alpha(Path(args.runs), out),
        fig_compute(df, out), fig_signal(df, out),
    ] if f]
    print(f"wrote {len(made)} figures to {out}/: " + ", ".join(made))


if __name__ == "__main__":
    main()
