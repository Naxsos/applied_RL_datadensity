#!/usr/bin/env python3
"""Aggregate runs/*/metrics.json -> results.csv + decision_table.md (§7).

- results.csv     : one row per run (all §5 fields).
- decision_table.md: means ± std per method, with the weighted score from the
                     spec's rubric. robustness_score is derived from each
                     method's own knob sweep here (not per-run).

Usage:  python aggregate.py --runs runs --out .
"""
from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# §7 rubric weights
WEIGHTS = {
    "zone_step_frac": (0.30, "min"),        # lower better
    "true_return_mean": (0.25, "max"),       # higher better
    "robustness_score": (0.20, "max"),
    "fits_local": (0.15, "max"),
    "cost_vs_forecast_err_corr": (0.10, "max"),
}


def load_runs(runs_dir: Path) -> pd.DataFrame:
    rows = []
    for mj in runs_dir.glob("*/metrics.json"):
        r = json.loads(mj.read_text())
        r["knob"] = (r.get("hparam") or {}).get("knob")
        r["knob_value"] = (r.get("hparam") or {}).get("value")
        rows.append(r)
    if not rows:
        raise SystemExit(f"no metrics.json under {runs_dir}/")
    return pd.DataFrame(rows)


def add_robustness(df: pd.DataFrame) -> dict:
    """robustness_score per method: 1 - normalized spread across its knob sweep."""
    from denrl.metrics import robustness_score
    scores = {}
    for method, g in df.groupby("method"):
        # mean over seeds per knob value, then spread across knob values
        per_knob = (g.groupby("knob_value")[["true_return_mean", "zone_step_frac"]]
                    .mean().reset_index().to_dict("records"))
        scores[method] = robustness_score(per_knob)
    return scores


def normalize(series, direction):
    x = series.astype(float)
    if x.max() == x.min():
        return pd.Series(0.5, index=x.index)
    z = (x - x.min()) / (x.max() - x.min())
    return z if direction == "max" else 1 - z


def build_decision_table(df: pd.DataFrame) -> pd.DataFrame:
    agg = df.groupby("method").agg(
        zone_step_frac=("zone_step_frac", "mean"),
        true_return_mean=("true_return_mean", "mean"),
        fits_local=("fits_local", "mean"),
        cost_vs_forecast_err_corr=("cost_vs_forecast_err_corr", "mean"),
        wall_clock_train_s=("wall_clock_train_s", "mean"),
    )
    rob = add_robustness(df)
    agg["robustness_score"] = agg.index.map(rob)

    total = pd.Series(0.0, index=agg.index)
    for col, (w, direction) in WEIGHTS.items():
        col_vals = agg[col].fillna(agg[col].mean() if agg[col].notna().any() else 0.0)
        total = total + w * normalize(col_vals, direction)
    agg["weighted_total"] = total
    return agg.sort_values("weighted_total", ascending=False)


def to_markdown(agg: pd.DataFrame) -> str:
    cols = ["zone_step_frac", "true_return_mean", "robustness_score",
            "fits_local", "cost_vs_forecast_err_corr", "wall_clock_train_s", "weighted_total"]
    lines = ["# Decision Table\n", "| method | " + " | ".join(cols) + " |",
             "|" + "---|" * (len(cols) + 1)]
    for method, row in agg.iterrows():
        vals = " | ".join(f"{row[c]:.3f}" if isinstance(row[c], float) else str(row[c]) for c in cols)
        lines.append(f"| {method} | {vals} |")
    winner = agg.index[0]
    lines.append(f"\n**Winner (highest weighted_total): `{winner}`** — carry into the full "
                 f"benchmark vs the paper baseline.")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default=".")
    args = ap.parse_args()

    df = load_runs(Path(args.runs))
    out = Path(args.out)
    df.to_csv(out / "results.csv", index=False)

    agg = build_decision_table(df)
    (out / "decision_table.md").write_text(to_markdown(agg))
    print(f"wrote {out}/results.csv ({len(df)} runs) and {out}/decision_table.md")
    print(agg[["weighted_total"]].to_string())


if __name__ == "__main__":
    main()
