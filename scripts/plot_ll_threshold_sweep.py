#!/usr/bin/env python3
"""Plot LL threshold-sweep metrics against the KDE threshold.

Example:
    python scripts/plot_ll_threshold_sweep.py \
        --runs runs \
        --method baseline \
        --seed 0 \
        --suffix _threshold_sweep \
        --out figures/ll_threshold_sweep_baseline_seed0.png
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RUN_RE = re.compile(
    r"^(?P<method>baseline|lagr)_LL_t(?P<threshold>[0-9.]+)_seed(?P<seed>\d+)_(?P<suffix>.+)$"
)


def _load_rows(runs_dir: Path, method: str, seed: int, suffix: str) -> list[dict]:
    rows = []
    for metrics_path in runs_dir.glob("*/metrics.json"):
        row = json.loads(metrics_path.read_text())
        run_id = row.get("run_id", "")
        match = RUN_RE.match(run_id)
        if not match:
            continue
        if match.group("method") != method:
            continue
        if int(match.group("seed")) != seed:
            continue
        if f"_{match.group('suffix')}" != suffix:
            continue
        row["_threshold"] = float(match.group("threshold"))
        rows.append(row)
    if not rows:
        raise SystemExit(
            f"no threshold-sweep runs found for method={method!r}, seed={seed}, suffix={suffix!r} under {runs_dir}/"
        )
    rows.sort(key=lambda r: r["_threshold"])
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs", help="directory containing run subdirectories")
    ap.add_argument("--method", choices=("baseline", "lagr"), default="baseline")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--suffix", default="_threshold_sweep")
    ap.add_argument("--out", default=None, help="output image path")
    args = ap.parse_args()

    runs_dir = Path(args.runs)
    rows = _load_rows(runs_dir, method=args.method, seed=args.seed, suffix=args.suffix)

    thresholds = np.array([r["_threshold"] for r in rows], dtype=float)
    zone_steps = np.array([r["train_zone_steps"] for r in rows], dtype=float)
    penalized_steps = np.array([r["train_penalized_steps"] for r in rows], dtype=float)
    strict_landing = np.array([r["strict_landing_rate"] for r in rows], dtype=float)
    landing = np.array([r["landing_success_rate"] for r in rows], dtype=float)

    keep = thresholds <= 0.3
    thresholds = thresholds[keep]
    zone_steps = zone_steps[keep]
    penalized_steps = penalized_steps[keep]
    strict_landing = strict_landing[keep]
    landing = landing[keep]

    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 150,
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
    })

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    line_zone, = ax1.plot(
        thresholds, zone_steps,
        marker="o", linewidth=2, color="#1f77b4", label="train zone steps",
    )
    line_penalized, = ax1.plot(
        thresholds, penalized_steps,
        marker="s", linewidth=2, color="#ff7f0e", label="train penalized steps",
    )
    line_strict, = ax2.plot(
        thresholds, strict_landing,
        marker="^", linewidth=2, color="#2ca02c", label="strict landing rate",
    )
    line_landing, = ax2.plot(
        thresholds, landing,
        marker="d", linewidth=2, color="#d62728", label="landing success rate",
    )

    ax1.set_xlabel("KDE threshold")
    ax1.set_ylabel("Step count")
    ax2.set_ylabel("Strict landing rate")
    ax2.set_ylim(0.0, 1.0)

    title_method = "baseline" if args.method == "baseline" else "lagrangian"
    ax1.set_title(f"LL threshold sweep ({title_method}, seed={args.seed})")

    ax1.set_xlim(float(thresholds.min()), float(thresholds.max()))

    lines = [line_zone, line_penalized, line_strict, line_landing]
    ax1.legend(
        lines,
        [line.get_label() for line in lines],
        loc="upper left",
        bbox_to_anchor=(0.75, 1.0),
        frameon=False,
    )

    out_path = Path(args.out) if args.out else Path("figures") / f"ll_threshold_sweep_{args.method}_seed{args.seed}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)

    print(f"wrote {out_path} from {len(rows)} runs")


if __name__ == "__main__":
    main()
