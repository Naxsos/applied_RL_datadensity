#!/usr/bin/env python3
"""Compare and visualize LunarLander runs, with multi-seed aggregation.

Usage:
    # Compare two specific runs
    python scripts/compare_ll_runs.py --runs baseline_LL_seed0 lagr_LL_seed0

    # Aggregate all seeds for each method
    python scripts/compare_ll_runs.py --runs baseline_LL_* lagr_LL_* --aggregate

    # Write to a custom path
    python scripts/compare_ll_runs.py --runs baseline_LL_* lagr_LL_* --aggregate --out figures/ll_compare.png
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_metrics(run_path: Path) -> dict | None:
    metrics_file = run_path / "metrics.json"
    if not metrics_file.exists():
        return None
    with open(metrics_file) as f:
        return json.load(f)


def group_by_method(run_paths: list[Path]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for run_path in run_paths:
        metrics = load_metrics(run_path)
        if metrics is None:
           continue
        method = metrics.get("method", "unknown")
        grouped[method].append(metrics)
    return dict(grouped)


def aggregate_metrics(metrics_list: list[dict]) -> dict:
    if not metrics_list:
        return {}

    keys_to_agg = [
        "zone_visit_rate",
        "true_return_mean",
        "true_return_std",
        "landing_success_rate",
        "strict_landing_rate",
        "crash_rate",
        "timeout_rate",
        "episode_len_mean",
        "wall_clock_train_s",
    ]

    result: dict[str, float | int] = {}
    for key in keys_to_agg:
        values = [float(m.get(key, np.nan)) for m in metrics_list]
        values = [v for v in values if not np.isnan(v)]
        if values:
           result[f"{key}_mean"] = float(np.mean(values))
           result[f"{key}_std"] = float(np.std(values))
    result["n_seeds"] = len(metrics_list)
    return result


def get_value(data: dict, key: str) -> float:
    if key in data:
        return float(data[key])
    if f"{key}_mean" in data:
        return float(data[f"{key}_mean"])
    return 0.0


def get_error(data: dict, key: str) -> float:
    candidates = [f"{key}_std", f"{key.replace('_mean', '')}_std"]
    for candidate in candidates:
        if candidate in data and data[candidate] is not None:
            return float(data[candidate])
    return 0.0


def compute_label_position(ax, value: float, err: float, pad_frac: float = 0.02) -> tuple[float, str]:
    ymin, ymax = ax.get_ylim()
    span = ymax - ymin if ymax > ymin else 1.0
    anchor = value + max(err, 0.0)
    pad = span * pad_frac
    y = min(anchor + pad, ymax - pad * 0.4)
    va = "bottom"
    if y <= anchor:
        y = max(anchor - pad, ymin + pad * 0.4)
        va = "top"
    return y, va


def annotate_bar(ax, x, value, err, fmt_value, fmt_err=None):
    text = fmt_value.format(value)
    if err > 0:
        if fmt_err is None:
            fmt_err = fmt_value
        text += f"\n± {fmt_err.format(err)}"
    y, va = compute_label_position(ax, value, err)
    ax.annotate(
        text,
        (x, y),
        ha="center",
        va=va,
        fontsize=9,
        fontweight="bold",
        clip_on=False,
    )


def compare_runs(runs_data: dict[str, dict], out_path: str = "figures/ll_run_comparison.png") -> None:
    methods = sorted(runs_data.keys())
    if not methods:
        raise ValueError("No run metrics available to compare")

    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(len(methods))]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # 1. Landing success rate
    ax = axes[0, 0]
    labels = ["Landing\nSuccess", "Strict\nLanding"]
    x = np.array([0.0, 0.75])
    width = 0.22
    for i, method in enumerate(methods):
        data = runs_data[method]
        vals = [get_value(data, "landing_success_rate"), get_value(data, "strict_landing_rate")]
        errs = [get_error(data, "landing_success_rate"), get_error(data, "strict_landing_rate")]
        pos = x + i * width - width * (len(methods) - 1) / 2
        ax.bar(pos, vals, width=width, color=colors[i], edgecolor="black", linewidth=0.8, alpha=0.9, label=method, yerr=errs, capsize=4)
        for xi, v, e in zip(pos, vals, errs):
            annotate_bar(ax, xi, v, e, "{:.1%}", "{:.1%}")
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.08)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.35, 1.10)
    ax.set_title("Landing success rates")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=max(1, len(methods)), frameon=False)

    # 2. Failure rates
    ax = axes[0, 1]
    labels = ["Timeout", "Crash"]
    x = np.array([0.0, 0.75])
    width = 0.22
    for i, method in enumerate(methods):
        data = runs_data[method]
        vals = [get_value(data, "timeout_rate"), get_value(data, "crash_rate")]
        errs = [get_error(data, "timeout_rate"), get_error(data, "crash_rate")]
        pos = x + i * width - width * (len(methods) - 1) / 2
        ax.bar(pos, vals, width=width, color=colors[i], edgecolor="black", linewidth=0.8, alpha=0.9, label=method, yerr=errs, capsize=4)
        for xi, v, e in zip(pos, vals, errs):
            annotate_bar(ax, xi, v, e, "{:.1%}", "{:.1%}")
    ax.set_ylabel("Rate")
    ax.set_ylim(0, max(0.6, max(get_value(runs_data[m], "timeout_rate") for m in methods) * 1.45))
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.35, 1.10)
    ax.set_title("Failure rates")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=max(1, len(methods)), frameon=False)

    # 3. Return
    ax = axes[0, 2]
    returns = [get_value(runs_data[m], "true_return_mean") for m in methods]
    errs = [get_error(runs_data[m], "true_return_mean") for m in methods]
    ax.bar(methods, returns, yerr=errs, capsize=5, color=colors, edgecolor="black", linewidth=0.8, alpha=0.8)
    ax.set_ylabel("True return")
    ax.set_title("Clean return comparison")
    ax.grid(axis="y", alpha=0.25)
    for i, (v, e) in enumerate(zip(returns, errs)):
        annotate_bar(ax, i, v, e, "{:.1f}", "{:.1f}")

    # 4. Episode length
    ax = axes[1, 0]
    lengths = [get_value(runs_data[m], "episode_len_mean") for m in methods]
    errs = [get_error(runs_data[m], "episode_len_mean") for m in methods]
    ax.bar(methods, lengths, yerr=errs, capsize=5, color=colors, edgecolor="black", linewidth=0.8, alpha=0.8)
    ax.axhline(500, color="#C0392B", linestyle="--", linewidth=1.5, label="Max steps")
    ax.set_ylabel("Steps")
    ax.set_title("Mean episode length")
    ax.grid(axis="y", alpha=0.25)
    for i, (v, e) in enumerate(zip(lengths, errs)):
        annotate_bar(ax, i, v, e, "{:.1f}", "{:.1f}")

    # 5. Zone visit rate
    ax = axes[1, 1]
    zones = [get_value(runs_data[m], "zone_visit_rate") for m in methods]
    errs = [get_error(runs_data[m], "zone_visit_rate") for m in methods]
    ax.bar(methods, zones, yerr=errs, capsize=5, color=colors, edgecolor="black", linewidth=0.8, alpha=0.8)
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.08)
    ax.set_title("Zone visit rate")
    ax.grid(axis="y", alpha=0.25)
    for i, (v, e) in enumerate(zip(zones, errs)):
        annotate_bar(ax, i, v, e, "{:.1%}", "{:.1%}")

    # 6. Summary panel
    ax = axes[1, 2]
    ax.axis("off")
    lines = []
    for method in methods:
        data = runs_data[method]
        n = int(data.get("n_seeds", 1))
        ret = get_value(data, "true_return_mean")
        ret_err = get_error(data, "true_return_mean")
        strict = get_value(data, "strict_landing_rate")
        timeout = get_value(data, "timeout_rate")
        zone = get_value(data, "zone_visit_rate")
        lines.append(f"{method}:")
        lines.append(f"  strict landing: {strict:.1%}")
        if ret_err > 0:
           lines.append(f"  return: {ret:.1f} ± {ret_err:.1f}")
        else:
           lines.append(f"  return: {ret:.1f}")
        lines.append(f"  timeout: {timeout:.1%}")
        lines.append(f"  zone visit: {zone:.1%}")
        if n > 1:
           lines.append(f"  seeds: {n}")
        lines.append("")
    summary_text = "\n".join(lines)
    ax.text(
        0.05,
        0.98,
        summary_text,
        fontsize=9,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#F7F7F7", edgecolor="#8C8C8C", linewidth=1),
    )

    fig.suptitle("LunarLander Run Comparison", fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="run directories to compare (e.g., baseline_LL_seed0 lagr_LL_seed0)")
    ap.add_argument("--aggregate", action="store_true", help="aggregate by method name across seeds")
    ap.add_argument("--out", default="figures/ll_run_comparison.png", help="output path for the visualization")
    args = ap.parse_args()

    runs_dir = Path("runs")
    run_paths: list[Path] = []
    for run_glob in args.runs:
        if "*" in run_glob:
           run_paths.extend(sorted(runs_dir.glob(run_glob)))
        else:
           run_paths.append(runs_dir / run_glob)

    if not run_paths:
        print(f"No runs found matching: {args.runs}")
        return

    print(f"Found {len(run_paths)} run(s)")

    if args.aggregate:
        runs_data = group_by_method(run_paths)
        for method, metrics_list in runs_data.items():
           runs_data[method] = aggregate_metrics(metrics_list)
           print(f"{method}: {len(metrics_list)} seed(s)")
    else:
        runs_data = {}
        for run_path in run_paths:
           metrics = load_metrics(run_path)
           if metrics:
               runs_data[run_path.name] = metrics
               print(f"  {run_path.name}")

    compare_runs(runs_data, out_path=args.out)


if __name__ == "__main__":
    main()
