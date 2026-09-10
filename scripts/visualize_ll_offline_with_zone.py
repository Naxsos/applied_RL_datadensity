#!/usr/bin/env python3
"""Visualize the new LL offline dataset with zone-termination "hole".

Shows the low-density region created by zone termination, overlaid with the
avoidable box zone to demonstrate the constraint structure.

    python scripts/visualize_ll_offline_with_zone.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/LL_offline.parquet")
    ap.add_argument("--out", default="figures/ll_offline_with_zone.png")
    ap.add_argument("--sample-points", type=int, default=25000,
                    help="number of points shown in scatter plots.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data_path = Path(args.data)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_parquet(data_path)
    obs_cols = sorted([c for c in df.columns if c.startswith("obs")], key=lambda c: int(c[3:]))
    obs = df[obs_cols].to_numpy(dtype=np.float64)

    # Sample for plotting
    rng = np.random.default_rng(args.seed)
    plot_idx = rng.choice(len(obs), size=min(args.sample_points, len(obs)), replace=False)
    o = obs[plot_idx]

    x, y = o[:, 0], o[:, 1]
    vx, vy = o[:, 2], o[:, 3]
    angle, ang_vel = o[:, 4], o[:, 5]
    speed = np.hypot(vx, vy)
    tilt_abs = np.abs(angle)

    # Zone definition comes from the canonical LL config file.
    repo_root = Path(__file__).resolve().parent.parent
    zone_cfg = yaml.safe_load((repo_root / "configs" / "ll_box_zone.yaml").read_text())
    zone_x = tuple(zone_cfg["x"])
    zone_y = tuple(zone_cfg["y"])
    zone_mask = (x >= zone_x[0]) & (x <= zone_x[1]) & (y >= zone_y[0]) & (y <= zone_y[1])

    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 150,
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
    })
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    # Styling
    normal_style = dict(s=4, c="#6e6e6e", alpha=0.3, linewidths=0, label="data points")
    zone_style = dict(s=6, c="#d55e00", alpha=0.8, linewidths=0, label="in zone (should be rare/zero)")

    # Plot 1: Position space with zone box
    ax = axes[0, 0]
    ax.scatter(x[~zone_mask], y[~zone_mask], **normal_style)
    ax.scatter(x[zone_mask], y[zone_mask], **zone_style)
    ax.add_patch(plt.Rectangle(
        (zone_x[0], zone_y[0]), zone_x[1] - zone_x[0], zone_y[1] - zone_y[0],
        fill=False, edgecolor='tab:blue', linewidth=2, label='zone box'
    ))
    ax.set_xlabel("x position")
    ax.set_ylabel("y altitude")
    ax.set_title("Position space: zone termination creates a hole")

    # Plot 2: Speed vs altitude
    ax = axes[0, 1]
    ax.scatter(speed[~zone_mask], y[~zone_mask], **normal_style)
    ax.scatter(speed[zone_mask], y[zone_mask], **zone_style)
    ax.set_xlabel("speed ||(vx, vy)||")
    ax.set_ylabel("y altitude")
    ax.set_title("Speed vs altitude")

    # Plot 3: Angle vs angular velocity
    ax = axes[1, 0]
    ax.scatter(angle[~zone_mask], ang_vel[~zone_mask], **normal_style)
    ax.scatter(angle[zone_mask], ang_vel[zone_mask], **zone_style)
    ax.set_xlabel("angle")
    ax.set_ylabel("angular velocity")
    ax.set_title("Attitude (angle, angular velocity)")

    # Plot 4: Tilt vs altitude
    ax = axes[1, 1]
    ax.scatter(tilt_abs[~zone_mask], y[~zone_mask], **normal_style)
    ax.scatter(tilt_abs[zone_mask], y[zone_mask], **zone_style)
    ax.set_xlabel("|angle|")
    ax.set_ylabel("y altitude")
    ax.set_title("Low-altitude tilted states")

    # Legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)

    zone_in_data = zone_mask.mean() * 100.0
    fig.suptitle(
        f"LunarLander offline data (zone-filtered): {len(df)} transitions\n"
        f"Zone box coverage: {zone_in_data:.2f}% (should be 0 or very low)",
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path)
    plt.close(fig)

    print(
        f"wrote {out_path} from {data_path} "
        f"(N={obs.shape[0]}, plotted={o.shape[0]}, "
        f"zone_in_data={zone_in_data:.2f}%)"
    )


if __name__ == "__main__":
    main()
