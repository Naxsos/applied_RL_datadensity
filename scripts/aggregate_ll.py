#!/usr/bin/env python3
"""Aggregate LL runs into LL-specific tables keyed by method and sweep value."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

LL_WEIGHTS = {
    "strict_landing_rate": (0.30, "max"),
    "landing_success_rate": (0.20, "max"),
    "zone_visit_rate": (0.20, "min"),
    "true_return_mean": (0.15, "max"),
    "crash_rate": (0.10, "min"),
    "timeout_rate": (0.05, "min"),
}

LL_METRICS = [
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


def load_ll_runs(runs_dir: Path) -> pd.DataFrame:
    rows = []
    for metrics_json in runs_dir.glob("*/metrics.json"):
        row = json.loads(metrics_json.read_text())
        if row.get("env") != "LL":
            continue
        row["knob"] = (row.get("hparam") or {}).get("knob")
        row["knob_value"] = (row.get("hparam") or {}).get("value")
        rows.append(row)
    if not rows:
        raise SystemExit(f"no LL metrics.json under {runs_dir}/")
    return pd.DataFrame(rows)


def normalize(series: pd.Series, direction: str) -> pd.Series:
    x = series.astype(float)
    if x.max() == x.min():
        return pd.Series(0.5, index=x.index)
    z = (x - x.min()) / (x.max() - x.min())
    return z if direction == "max" else 1 - z


def aggregate_ll(df: pd.DataFrame) -> pd.DataFrame:
    agg_spec: dict[str, tuple[str, str]] = {"n_runs": ("run_id", "count")}
    for metric in LL_METRICS:
        agg_spec[f"{metric}_mean"] = (metric, "mean")
        agg_spec[f"{metric}_std"] = (metric, "std")

    agg = (
        df.groupby(["method", "knob", "knob_value"], dropna=False)
        .agg(**agg_spec)
        .reset_index()
    )

    total = pd.Series(0.0, index=agg.index)
    for metric, (weight, direction) in LL_WEIGHTS.items():
        col = f"{metric}_mean"
        values = agg[col].fillna(agg[col].mean() if agg[col].notna().any() else 0.0)
        total = total + weight * normalize(values, direction)
    agg["weighted_total"] = total
    return agg.sort_values(["weighted_total", "method"], ascending=[False, True]).reset_index(drop=True)


def to_markdown(agg: pd.DataFrame) -> str:
    lines = [
        "# LL Decision Table\n",
        "| method | knob | value | n_runs | strict_landing | landing_success | zone_visit | return | crash | timeout | episode_len | wall_clock_s | weighted_total |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in agg.iterrows():
        lines.append(
            "| {method} | {knob} | {value:.3f} | {n_runs:d} | {strict:.3f} | {landing:.3f} | "
            "{zone:.3f} | {ret:.1f} | {crash:.3f} | {timeout:.3f} | {ep_len:.1f} | {wall:.1f} | {score:.3f} |".format(
                method=row["method"],
                knob=row["knob"] or "-",
                value=float(row["knob_value"]),
                n_runs=int(row["n_runs"]),
                strict=float(row["strict_landing_rate_mean"]),
                landing=float(row["landing_success_rate_mean"]),
                zone=float(row["zone_visit_rate_mean"]),
                ret=float(row["true_return_mean_mean"]),
                crash=float(row["crash_rate_mean"]),
                timeout=float(row["timeout_rate_mean"]),
                ep_len=float(row["episode_len_mean_mean"]),
                wall=float(row["wall_clock_train_s_mean"]),
                score=float(row["weighted_total"]),
            )
        )

    best = agg.iloc[0]
    lines.append(
        "\n**Best LL setting by weighted_total:** "
        f"`{best['method']}` with `{best['knob']}={float(best['knob_value']):.3f}`."
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--out", default=".")
    args = parser.parse_args()

    df = load_ll_runs(Path(args.runs))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "ll_results.csv", index=False)

    agg = aggregate_ll(df)
    agg.to_csv(out / "ll_decision_table.csv", index=False)
    (out / "ll_decision_table.md").write_text(to_markdown(agg))
    print(f"wrote {out}/ll_results.csv ({len(df)} runs), {out}/ll_decision_table.csv, and {out}/ll_decision_table.md")
    print(agg[["method", "knob", "knob_value", "weighted_total"]].to_string(index=False))


if __name__ == "__main__":
    main()
