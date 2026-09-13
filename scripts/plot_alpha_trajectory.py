#!/usr/bin/env python3
"""Plot the Lagrangian dual variable alpha over training, across seeds.

Matches methodology.typ's "Constraint behavior (Lagrangian only)" metric:
alpha trajectory over training, plus constraint value C vs. epsilon.

Usage:
    python scripts/plot_alpha_trajectory.py --runs 'runs/vis/lagr_E1_epsilon0.01_seed*' \
        --epsilon 0.01 --out figures/alpha_trajectory.png
    python scripts/plot_alpha_trajectory.py --runs 'runs/**/lagr_LL_*_seed*' \
        --epsilon 0.05 --out figures/ll_alpha_trajectory.png
"""
from __future__ import annotations
import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

_STEP_FMT = FuncFormatter(lambda x, _: f"{int(x / 1000)}k" if x else "0")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--runs",
        nargs="+",
        default=[
            "runs/**/lagr_*_seed*",
            "runs/**/lagr_*_seed*_matrix*",
        ],
        help="one or more glob patterns matching run dirs (for example: 'runs/vis/lagr_E1_epsilon0.01_seed*' or 'runs/**/lagr_LL_*_seed*')",
    )
    ap.add_argument("--epsilon", type=float, default=0.01)
    ap.add_argument("--out", default="figures/alpha_trajectory.png")
    args = ap.parse_args()

    matched = []
    for pattern in args.runs:
        matched.extend(glob.glob(pattern, recursive=True))
    run_dirs = sorted({Path(p) for p in matched if Path(p).is_dir()})
    if not run_dirs:
        raise SystemExit(f"no run dirs matched: {args.runs}")

    seeds = []
    for d in run_dirs:
        csv_path = d / "alpha.csv"
        if not csv_path.exists():
            print(f"  warning: {csv_path} not found, skipping")
            continue
        df = pd.read_csv(csv_path)
        required = {"step", "alpha", "constraint_C"}
        missing = required - set(df.columns)
        if missing:
            print(f"  warning: {csv_path} missing columns {sorted(missing)}, skipping")
            continue
        if seeds and len(df) != len(seeds[0]):
            print(
                f"  warning: {csv_path} has {len(df)} rows but expected {len(seeds[0])}; "
                "skipping to keep a common step grid across seeds"
            )
            continue
        seeds.append(df)

    if not seeds:
        raise SystemExit(f"no valid alpha.csv files found in: {args.runs}")

    # all selected seeds should share the same update_freq/total_steps; if a run set
    # mixes old and newer LL revisions, skip the mismatched ones instead of crashing.
    steps = seeds[0]["step"].to_numpy()
    alpha_mat = np.stack([s["alpha"].to_numpy() for s in seeds])
    C_mat = np.stack([s["constraint_C"].to_numpy() for s in seeds])

    fig, (ax_alpha, ax_c) = plt.subplots(1, 2, figsize=(10, 4))

    for a in alpha_mat:
        ax_alpha.plot(steps, a, color="#D55E00", alpha=0.25, lw=1)
    ax_alpha.plot(steps, alpha_mat.mean(axis=0), color="#D55E00", lw=2, label="mean")
    ax_alpha.fill_between(
        steps, alpha_mat.mean(axis=0) - alpha_mat.std(axis=0),
        alpha_mat.mean(axis=0) + alpha_mat.std(axis=0),
        color="#D55E00", alpha=0.15, label="±1 std",
    )
    ax_alpha.set_xlabel("training step")
    ax_alpha.set_ylabel(r"$\alpha$")
    ax_alpha.set_title(rf"dual variable $\alpha$ ({len(seeds)} seeds)")
    ax_alpha.xaxis.set_major_formatter(_STEP_FMT)
    ax_alpha.legend(fontsize=9)

    for c in C_mat:
        ax_c.plot(steps, c, color="#184f95", alpha=0.25, lw=1)
    ax_c.plot(steps, C_mat.mean(axis=0), color="#184f95", lw=2, label="mean $C$")
    ax_c.axhline(args.epsilon, color="black", ls="--", lw=1, label=rf"$\epsilon={args.epsilon}$")
    ax_c.set_xlabel("training step")
    ax_c.set_ylabel(r"constraint value $C$")
    ax_c.set_title(rf"constraint $C$ vs. $\epsilon$ ({len(seeds)} seeds)")
    ax_c.xaxis.set_major_formatter(_STEP_FMT)
    ax_c.legend(fontsize=9)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"saved -> {out} ({len(seeds)} seeds, {len(steps)} steps each)")


if __name__ == "__main__":
    main()
