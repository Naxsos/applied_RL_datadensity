#!/usr/bin/env python3
"""Visualize LunarLander offline KDE density as a colorscale scatter plot.

Example:
    python scripts/visualize_ll_density_scatter.py \
        --data data/LL_offline.parquet \
        --out figures/ll_density_scatter.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import yaml
from sklearn.neighbors import KernelDensity


def _sample_idx(n: int, max_n: int, seed: int) -> np.ndarray:
    idx = np.arange(n)
    if max_n <= 0 or max_n >= n:
        return idx
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(idx, size=max_n, replace=False))


def _sample_box_points(zone_x: tuple[float, float], zone_y: tuple[float, float], n: int, seed: int) -> np.ndarray:
    if n <= 0:
        return np.empty((0, 2), dtype=np.float64)
    rng = np.random.default_rng(seed)
    x0, x1 = sorted(zone_x)
    y0, y1 = sorted(zone_y)
    xs = rng.uniform(x0, x1, size=n)
    ys = rng.uniform(y0, y1, size=n)
    return np.column_stack([xs, ys])


def _load_obs(dataset_path: Path) -> np.ndarray:
    df = pd.read_parquet(dataset_path)
    obs_cols = sorted(
        [c for c in df.columns if c.startswith("obs")],
        key=lambda c: int(c[3:]),
    )
    if len(obs_cols) < 2:
        raise ValueError("Expected observation columns obs0.. for LunarLander data.")
    return df[obs_cols].to_numpy(dtype=np.float64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/LL_offline.parquet")
    ap.add_argument("--out", default="figures/ll_density_scatter.png")
    ap.add_argument("--bandwidth", type=float, default=0.1)
    ap.add_argument("--sample-points", type=int, default=25000,
                    help="Number of observed points shown and scored in the scatter plot.")
    ap.add_argument("--box-grid-points", type=int, default=20000,
                    help="Number of synthetic points sampled inside the excluded box for visualization only; these are never included in KDE fitting.")
    ap.add_argument("--max-fit-points", type=int, default=5000,
                    help="Cap KDE fit size for speed.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data_path = Path(args.data)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    obs = _load_obs(data_path)
    fit_idx = _sample_idx(obs.shape[0], args.max_fit_points, args.seed)
    plot_idx = _sample_idx(obs.shape[0], args.sample_points, args.seed + 1)

    xy_fit = obs[fit_idx, :2]
    xy_plot = obs[plot_idx, :2]

    kde = KernelDensity(bandwidth=args.bandwidth).fit(xy_fit)
    density = np.exp(kde.score_samples(xy_plot))

    x, y = xy_plot[:, 0], xy_plot[:, 1]

    repo_root = Path(__file__).resolve().parent.parent
    zone_cfg = yaml.safe_load((repo_root / "configs" / "LL_box_zone.yaml").read_text())
    zone_x = tuple(zone_cfg["x"])
    zone_y = tuple(zone_cfg["y"])
    zone_mask = (x >= zone_x[0]) & (x <= zone_x[1]) & (y >= zone_y[0]) & (y <= zone_y[1])

    box_points = _sample_box_points(zone_x, zone_y, args.box_grid_points, args.seed + 2)
    box_density = np.exp(kde.score_samples(box_points)) if box_points.size else np.empty((0,), dtype=np.float64)

    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 150,
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.2,
    })

    fig, ax = plt.subplots(figsize=(9, 7))
    scatter_norm = LogNorm(
        vmin=max(float(np.min(np.concatenate([density, box_density]))), 1e-6),
        vmax=float(np.max(np.concatenate([density, box_density]))),
    )
    sc = ax.scatter(
        x,
        y,
        c=density,
        s=6,
        cmap="viridis",
        norm=scatter_norm,
        alpha=0.8,
        linewidths=0,
    )
    if box_points.size:
        ax.scatter(
            box_points[:, 0],
            box_points[:, 1],
            c=box_density,
            s=1.5,
            cmap="viridis",
            norm=scatter_norm,
            alpha=0.6,
            linewidths=0,
            label="synthetic box points (display only)",
        )
    ax.add_patch(Rectangle(
        (zone_x[0], zone_y[0]),
        zone_x[1] - zone_x[0],
        zone_y[1] - zone_y[0],
        fill=False,
        edgecolor="red",
        linewidth=2,
        label="LL box zone",
    ))
    ax.scatter(
        x[zone_mask],
        y[zone_mask],
        s=10,
        facecolors="none",
        edgecolors="white",
        linewidths=0.4,
        alpha=0.9,
        label="observed points inside box",
    )
    ax.set_xlabel("x position")
    ax.set_ylabel("y altitude")
    ax.set_title("LunarLander offline KDE density in position space")

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("KDE density")

    box_share = 100.0 * float(zone_mask.mean())
    observed_box_density = density[zone_mask]
    fig.suptitle(
        f"KDE bandwidth={args.bandwidth:g}, fit_points={len(fit_idx)}, plotted={len(plot_idx)}, box_grid={len(box_points)}\n"
        f"Box share={box_share:.2f}%, observed box density median="
        f"{(float(np.median(observed_box_density)) if observed_box_density.size else float('nan')):.3f}",
        y=0.98,
    )
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

    print(
        f"wrote {out_path} from {data_path} "
        f"(fit={len(fit_idx)}, plotted={len(plot_idx)}, "
        f"density_min={density.min():.4f}, density_max={density.max():.4f})"
    )


if __name__ == "__main__":
    main()
