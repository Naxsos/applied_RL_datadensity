#!/usr/bin/env python3
"""LL comparison table -> CSVs the report reads directly.

Usage:
    python scripts/report_ll_comparison.py --runs runs/LL --out report/ll_group_comparison.csv
    python scripts/report_ll_comparison.py --runs runs/LL --pool --out report/ll_group_comparison.csv
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
    ("landing_success_rate", "Landing success rate", False),
    ("strict_landing_rate", "Strict landing rate", False),
    ("crash_rate", "Crash rate", True),
    ("timeout_rate", "Timeout rate", True),
    ("wall_clock_train_s", "Wall-clock training time (s)", True),
]


def load_runs(runs_dir: Path) -> list[dict]:
    rows = []
    for metrics_json in runs_dir.glob("*/metrics.json"):
        rows.append(json.loads(metrics_json.read_text()))
    if not rows:
        raise SystemExit(f"no metrics.json under {runs_dir}/")
    return rows


def fmt_value(v: float) -> str:
    return f"{v:g}"


def write_csv(by_group: dict[str, list[dict]], groups: list[str], out_path: Path) -> None:
    out_rows = []
    for key, label, lower_better in METRICS:
        row = {"metric": label, "lower_is_better": int(lower_better)}
        for g in groups:
            vals = np.array([r[key] for r in by_group[g] if r.get(key) is not None], dtype=float)
            row[f"{g}_mean"] = f"{vals.mean():.6f}" if len(vals) else ""
            row[f"{g}_std"] = f"{vals.std():.6f}" if len(vals) else ""
            row[f"{g}_n"] = len(vals)
        out_rows.append(row)

    fieldnames = ["metric", "lower_is_better"] + [f"{g}_{s}" for g in groups for s in ("mean", "std", "n")]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)
    print(f"wrote {out_path} ({len(out_rows)} metrics x {len(groups)} groups: {groups})")


def build_groups(rows: list[dict], pooled: bool) -> tuple[list[str], dict[str, list[dict]], dict[str, tuple]]:
    by_group: dict[str, list[dict]] = {}
    group_sort_key: dict[str, tuple] = {}

    for r in rows:
        method = r["method"]
        hparam = r.get("hparam") or {}
        knob, value = hparam.get("knob", ""), hparam.get("value", 0.0)
        gid = method if pooled else f"{method}_{knob}{fmt_value(value)}"
        by_group.setdefault(gid, []).append(r)
        method_rank = 0 if method == "baseline" else 1
        if value == "":
            group_sort_key[gid] = (method_rank, 0.0)
        else:
            group_sort_key[gid] = (method_rank, float(value))

    groups = sorted(by_group, key=lambda g: group_sort_key[g])
    return groups, by_group, group_sort_key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/LL")
    ap.add_argument("--out", default="report/data/ll_group_comparison.csv")
    ap.add_argument("--pool", action="store_true", help="pooled baseline vs lagr summary")
    ap.add_argument("--by-p", action="store_true", help="split baseline by individual p values and keep lagr fixed at epsilon=0.01")
    args = ap.parse_args()

    if args.pool and args.by_p:
        ap.error("choose either --pool or --by-p, not both")

    rows = load_runs(Path(args.runs))
    pooled = bool(args.pool)
    if not args.pool and not args.by_p:
        pooled = True

    out_path = Path(args.out)
    if args.by_p:
        out_path = out_path.with_name(f"{out_path.stem}_by_p{out_path.suffix}")

    groups, by_group, _ = build_groups(rows, pooled=pooled)
    write_csv(by_group, groups, out_path)


if __name__ == "__main__":
    main()
