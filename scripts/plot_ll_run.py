#!/usr/bin/env python3
"""Plot LunarLander run metrics as a visual summary."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    metrics_path = Path("runs/baseline_LL_seed0/metrics.json")
    with open(metrics_path) as f:
        m = json.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    ax = axes[0, 0]
    labels = ["Landing\nSuccess", "Strict\nLanding", "Crash", "Timeout"]
    values = [
        m["landing_success_rate"],
        m["strict_landing_rate"],
        m["crash_rate"],
        m["timeout_rate"],
    ]
    colors = ["#2ecc71", "#3498db", "#e74c3c", "#f39c12"]
    ax.bar(labels, values, color=colors)
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1)
    ax.set_title("Landing Outcome Rates")
    for i, v in enumerate(values):
        ax.text(i, v + 0.03, f"{v:.0%}", ha="center", fontweight="bold")

    ax = axes[0, 1]
    mean_ret = m["true_return_mean"]
    std_ret = m["true_return_std"]
    ax.bar(["Clean Return"], [mean_ret], yerr=std_ret, capsize=10, color="#9b59b6")
    ax.set_ylabel("Return")
    ax.set_title("True Return (mean ± std)")
    ax.text(0, mean_ret + std_ret + 10, f"{mean_ret:.1f}", ha="center", fontweight="bold")

    ax = axes[1, 0]
    ep_len = m["episode_len_mean"]
    ax.bar(["Mean Episode\nLength"], [ep_len], color="#1abc9c")
    ax.axhline(500, color="red", linestyle="--", label="Max steps (500)")
    ax.set_ylabel("Steps")
    ax.set_title("Episode Length")
    ax.legend()
    ax.text(0, ep_len + 10, f"{ep_len:.0f}", ha="center", fontweight="bold")

    ax = axes[1, 1]
    ax.axis("off")
    summary_text = (
        "Baseline LL Run (seed 0)\n\n"
        f"✓ Landing success: {m['landing_success_rate']:.0%}\n"
        f"✓ True return: {m['true_return_mean']:.1f}±{m['true_return_std']:.1f}\n"
        f"✓ Zone visit: {m['zone_visit_rate']:.0%}\n"
        f"✓ Crash rate: {m['crash_rate']:.0%}\n\n"
        f"✗ Strict landing: {m['strict_landing_rate']:.0%}\n"
        f"✗ Timeout rate: {m['timeout_rate']:.0%}\n\n"
        "Interpretation:\n"
        "Policy reaches near-pad safely (99%)\n"
        "but struggles to commit to final\n"
        "landing (only 36% terminal success).\n"
        "Timeouts dominate (64%) → agent\n"
        "hovers rather than descends.\n"
    )
    ax.text(
        0.1,
        0.5,
        summary_text,
        fontsize=10,
        family="monospace",
        verticalalignment="center",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )

    fig.suptitle("LunarLander Baseline (continuous SAC)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig("figures/ll_run_metrics.png", dpi=150)
    print("wrote figures/ll_run_metrics.png")


if __name__ == "__main__":
    main()
