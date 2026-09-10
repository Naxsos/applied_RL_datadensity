#!/usr/bin/env python3
"""Combine saved traj_final.npy files into a multi-panel trajectory plot.

Usage:
    python scripts/plot_combined_traj.py \
        --groups 'baseline p=2:runs/vis/baseline_E1_p2.0_seed*' \
                 'baseline p=10:runs/vis/baseline_E1_p10.0_seed*' \
                 'baseline p=30:runs/vis/baseline_E1_p30.0_seed*' \
                 'lagr ε=0.01:runs/vis/lagr_E1_epsilon0.01_seed*' \
        --out figures/vis_trajectories.png
"""
from __future__ import annotations
import argparse
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import yaml
from scripts.plot_trajectories import plot_trajectory_snapshot, _draw_zone_arc, METHOD_COLOR

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="+", required=True,
                    help="'LABEL:GLOB_PATTERN' pairs, e.g. 'baseline p=10:runs/vis/baseline_E1_p10.0_seed*'")
    ap.add_argument("--step", default="final",
                    help="which traj to load: 'final' or a step number like '50000'")
    ap.add_argument("--out", default="figures/combined_traj.png")
    args = ap.parse_args()

    groups = []
    for g in args.groups:
        label, pattern = g.split(":", 1)
        dirs = sorted(glob.glob(pattern))
        if not dirs:
            ap.error(f"pattern matched nothing: {pattern}")
        groups.append((label, dirs))

    n = len(groups)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.5), squeeze=False)
    axes = axes[0]

    for ax, (label, dirs) in zip(axes, groups):
        # load and concatenate npy files
        all_obs = []
        for d in dirs:
            npy = Path(d) / "traj" / f"traj_{args.step}.npy"
            if not npy.exists():
                print(f"  warning: {npy} not found, skipping")
                continue
            all_obs.append(np.load(npy))
        if not all_obs:
            ax.set_title(f"{label}\n(no data)", fontsize=10)
            continue
        obs = np.concatenate(all_obs)

        # detect method from first run's config
        cfg = yaml.safe_load((Path(dirs[0]) / "config.yaml").read_text())
        method = cfg.get("method", "baseline")

        # plot
        xs, ys = obs[:, 1], obs[:, 0]  # sin θ, cos θ
        color = METHOD_COLOR.get(method, "#333333")
        cmap = mcolors.LinearSegmentedColormap.from_list("w2m", ["#ffffff", color])
        ax.hexbin(xs, ys, gridsize=40, cmap=cmap, linewidths=0.2,
                  mincnt=1, bins="log", zorder=1)

        # zone arc
        from denrl.env import as_zone, ZONE_LOW, ZONE_HIGH
        zone = as_zone(cfg["env"].get("excluded_zone", (ZONE_LOW, ZONE_HIGH)),
                       symmetric=cfg["env"].get("zone_symmetric", False))
        _draw_zone_arc(ax, zone)

        # unit circle
        th = np.linspace(0, 2 * np.pi, 300)
        ax.plot(np.sin(th), np.cos(th), "k-", lw=0.6, alpha=0.3, zorder=4)
        ax.set_xlim(1.15, -1.15)
        ax.set_ylim(-1.15, 1.15)
        ax.set_aspect("equal")
        ax.set_xlabel("sin θ", fontsize=9)
        ax.set_ylabel("cos θ", fontsize=9)
        ax.set_title(f"{label}  ({len(dirs)} seeds)", fontsize=10, color=color, fontweight="bold")
        ax.tick_params(labelsize=8)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
