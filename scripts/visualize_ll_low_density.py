#!/usr/bin/env python3
"""Visualize low-density pockets in LunarLander offline data.

Recreates figures/ll_low_density_projections.png from data/LL_offline.parquet.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity


def _load_obs(dataset_path: Path) -> np.ndarray:
    df = pd.read_parquet(dataset_path)
    obs_cols = sorted(
        [c for c in df.columns if c.startswith("obs")],
        key=lambda c: int(c[3:]),
    )
    if len(obs_cols) < 6:
        raise ValueError("Expected LunarLander observations with columns obs0..obs7.")
    return df[obs_cols].to_numpy(dtype=np.float64)


def _sample_idx(n: int, max_n: int, seed: int) -> np.ndarray:
    idx = np.arange(n)
    if max_n <= 0 or max_n >= n:
        return idx
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(idx, size=max_n, replace=False))


def _density_mask(
    obs: np.ndarray,
    bandwidth: float,
    low_pct: float,
    max_fit_points: int,
    max_eval_points: int,
    seed: int,
) -> tuple[np.ndarray, float, np.ndarray]:
    fit_idx = _sample_idx(obs.shape[0], max_fit_points, seed)
    eval_idx = _sample_idx(obs.shape[0], max_eval_points, seed + 1)
    kde = KernelDensity(bandwidth=bandwidth).fit(obs[fit_idx])
    dens = np.exp(kde.score_samples(obs[eval_idx]))
    threshold = float(np.percentile(dens, low_pct))
    return dens <= threshold, threshold, eval_idx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/LL_offline.parquet")
    ap.add_argument("--out", default="figures/ll_low_density_projections.png")
    ap.add_argument("--bandwidth", type=float, default=0.25)
    ap.add_argument("--low-percentile", type=float, default=5.0)
    ap.add_argument("--sample-points", type=int, default=25000,
                    help="number of points shown in the scatter projections.")
    ap.add_argument("--max-fit-points", type=int, default=5000,
                    help="cap KDE fit size for speed.")
    ap.add_argument("--max-eval-points", type=int, default=25000,
                    help="cap points scored by KDE before thresholding.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data_path = Path(args.data)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    obs = _load_obs(data_path)
    low_mask, threshold, eval_idx = _density_mask(
        obs,
        bandwidth=args.bandwidth,
        low_pct=args.low_percentile,
        max_fit_points=args.max_fit_points,
        max_eval_points=args.max_eval_points,
        seed=args.seed,
    )

    plot_idx = _sample_idx(low_mask.shape[0], args.sample_points, args.seed + 2)
    o = obs[eval_idx][plot_idx]
    m = low_mask[plot_idx]

    x, y = o[:, 0], o[:, 1]
    vx, vy = o[:, 2], o[:, 3]
    angle, ang_vel = o[:, 4], o[:, 5]
    speed = np.hypot(vx, vy)
    tilt_abs = np.abs(angle)

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

    dense_style = dict(s=4, c="#6e6e6e", alpha=0.22, linewidths=0, label="higher density")
    low_style = dict(s=6, c="#d55e00", alpha=0.55, linewidths=0, label="low density (bottom pct)")

    ax = axes[0, 0]
    ax.scatter(x[~m], y[~m], **dense_style)
    ax.scatter(x[m], y[m], **low_style)
    ax.set_xlabel("x position")
    ax.set_ylabel("y altitude")
    ax.set_title("Position space")

    ax = axes[0, 1]
    ax.scatter(speed[~m], y[~m], **dense_style)
    ax.scatter(speed[m], y[m], **low_style)
    ax.set_xlabel("speed ||(vx, vy)||")
    ax.set_ylabel("y altitude")
    ax.set_title("Near-ground high-speed pockets")

    ax = axes[1, 0]
    ax.scatter(angle[~m], ang_vel[~m], **dense_style)
    ax.scatter(angle[m], ang_vel[m], **low_style)
    ax.set_xlabel("angle")
    ax.set_ylabel("angular velocity")
    ax.set_title("Attitude instability")

    ax = axes[1, 1]
    ax.scatter(tilt_abs[~m], y[~m], **dense_style)
    ax.scatter(tilt_abs[m], y[m], **low_style)
    ax.set_xlabel("|angle|")
    ax.set_ylabel("y altitude")
    ax.set_title("Low-altitude tilted states")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    low_share = 100.0 * float(low_mask.mean())
    fig.suptitle(
        "LunarLander offline data: low-density projections\n"
        f"KDE bandwidth={args.bandwidth:g}, threshold={args.low_percentile:g}th pct "
        f"(density <= {threshold:.3e}), low-density share={low_share:.1f}%",
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path)
    plt.close(fig)

    print(
        f"wrote {out_path} from {data_path} "
        f"(N={obs.shape[0]}, scored={len(eval_idx)}, plotted={o.shape[0]}, "
        f"low-density share={low_share:.2f}%)"
    )


if __name__ == "__main__":
    main()
