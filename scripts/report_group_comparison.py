#!/usr/bin/env python3
"""Comparison table -> a tidy CSV the paper's Typst table reads directly
(report/results.typ).

By default, baseline is broken out by its own p sweep (p=2 / p=10 / p=30)
rather than pooled, since pooling mixes regimes that behave very differently;
lagr gets one column per epsilon it was run at. Pass --pool to collapse each
method (baseline, lagr) into a single column instead -- answers "how much
does knob choice matter" rather than "does lagr beat the best baseline."

One row per metric: mean/std/n for each group, plus whether lower is better,
so the Typst table can decide which value to bold without hardcoding that
per metric.

Usage:
    python scripts/report_group_comparison.py --runs runs/vis --out report/data/group_comparison.csv
    python scripts/report_group_comparison.py --runs runs/vis --pool --out report/data/group_comparison.csv
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

import numpy as np

METRICS = [
    ("zone_depth_mean", "Zone depth-weighted step fraction", True),
    ("true_return_mean", "True return", False),
    ("upright_success_rate", "Upright success rate", False),
    ("time_to_upright_mean", "Time to upright (steps)", True),
    ("wall_clock_train_s", "Wall-clock training time (s)", True),
]


def load_runs(runs_dir: Path) -> list[dict]:
    rows = []
    for mj in runs_dir.glob("*/metrics.json"):
        rows.append(json.loads(mj.read_text()))
    if not rows:
        raise SystemExit(f"no metrics.json under {runs_dir}/")
    return rows


def fmt_value(v: float) -> str:
    """2.0 -> '2', 0.01 -> '0.01' -- matches run_id-style knob formatting."""
    return f"{v:g}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/vis")
    ap.add_argument("--out", default="report/data/group_comparison.csv")
    ap.add_argument("--pool", action="store_true",
                     help="collapse each method's knob sweep into a single column")
    args = ap.parse_args()

    rows = load_runs(Path(args.runs))
    by_group: dict[str, list[dict]] = {}
    group_sort_key: dict[str, tuple] = {}
    for r in rows:
        method = r["method"]
        hparam = r.get("hparam") or {}
        knob, value = hparam.get("knob", ""), hparam.get("value", 0.0)
        gid = method if args.pool else f"{method}_{knob}{fmt_value(value)}"
        by_group.setdefault(gid, []).append(r)
        # baseline groups sort by knob value ascending, then lagr groups after
        method_rank = 0 if method == "baseline" else 1
        group_sort_key[gid] = (method_rank, float(value))

    groups = sorted(by_group, key=lambda g: group_sort_key[g])
    out_rows = []
    for key, label, lower_better in METRICS:
        row = {"metric": label, "lower_is_better": int(lower_better)}
        for g in groups:
            vals = np.array([r[key] for r in by_group[g] if r.get(key) is not None], dtype=float)
            # fixed-point (not round()+str, which flips tiny values like 1e-05
            # into scientific notation Typst's float() may not expect)
            row[f"{g}_mean"] = f"{vals.mean():.6f}" if len(vals) else ""
            row[f"{g}_std"] = f"{vals.std():.6f}" if len(vals) else ""
            row[f"{g}_n"] = len(vals)
        out_rows.append(row)

    fieldnames = ["metric", "lower_is_better"] + [f"{g}_{s}" for g in groups for s in ("mean", "std", "n")]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)
    print(f"wrote {out_path} ({len(out_rows)} metrics x {len(groups)} groups: {groups})")


if __name__ == "__main__":
    main()
