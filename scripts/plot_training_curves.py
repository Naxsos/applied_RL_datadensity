#!/usr/bin/env python3
"""Plot training curves (return, zone_step_frac) across methods, averaged over seeds.

Usage:
    python scripts/plot_training_curves.py \
        --groups 'baseline p=2:runs/vis/baseline_E1_p2.0_seed*' \
                 'baseline p=10:runs/vis/baseline_E1_p10.0_seed*' \
                 'baseline p=30:runs/vis/baseline_E1_p30.0_seed*' \
                 'lagr ε=0.01:runs/vis/lagr_E1_epsilon0.01_seed*' \
        --out figures/training_curves.png
"""
from __future__ import annotations
import argparse
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METHOD_COLOR = {
    "baseline p=2": "#AAAAAA",
    "baseline p=10": "#666666",
    "baseline p=30": "#333333",
    "lagr ε=0.01": "#D55E00",
}


def load_group(dirs):
    dfs = []
    for d in dirs:
        csv = Path(d) / "train_metrics.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        dfs.append(df)
    return dfs


def plot_metric(ax, groups, metric, ylabel):
    for label, dfs in groups:
        if not dfs:
            continue
        # align on timestep, compute mean and std across seeds
        merged = pd.concat(dfs).groupby("timestep")[metric]
        mean = merged.mean()
        std = merged.std()
        color = METHOD_COLOR.get(label, None)
        ax.plot(mean.index, mean.values, label=label, lw=1.5, color=color)
        ax.fill_between(mean.index, (mean - std).values, (mean + std).values,
                        alpha=0.15, color=color)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.tick_params(labelsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="+", required=True,
                    help="'LABEL:GLOB_PATTERN' pairs")
    ap.add_argument("--out", default="figures/training_curves.png")
    args = ap.parse_args()

    groups = []
    for g in args.groups:
        label, pattern = g.split(":", 1)
        dirs = sorted(glob.glob(pattern))
        if not dirs:
            print(f"warning: {pattern} matched nothing")
            continue
        dfs = load_group(dirs)
        groups.append((label, dfs))
        print(f"  {label}: {len(dfs)} seeds")

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    plot_metric(axes[0], groups, "true_return_mean", "Return (clean)")
    axes[0].legend(fontsize=8)

    plot_metric(axes[1], groups, "zone_step_frac", "Zone step fraction")
    axes[1].set_xlabel("Timestep", fontsize=10)

    fig.suptitle("Training curves (mean ± std over seeds)", fontsize=11)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
