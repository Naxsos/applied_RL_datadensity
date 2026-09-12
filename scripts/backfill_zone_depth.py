#!/usr/bin/env python3
"""Backfill `zone_depth_mean` into existing runs/*/metrics.json.

Re-runs evaluate_policy() from each run's saved policy.zip (same protocol/seed
as the original final eval) purely to compute the new depth-weighted safety
metric -- it does not retrain, and every other field in metrics.json is left
untouched; only "zone_depth_mean" is added/overwritten.

Also re-checks zone_step_frac against the stored value as a fidelity check:
if re-evaluation doesn't reproduce the original number, something about the
protocol (seed, episode count) has drifted and the new field shouldn't be
trusted blindly.

Usage:
    python scripts/backfill_zone_depth.py --runs runs/vis
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import yaml

from denrl import metrics
from denrl.env import make_sim_env, resolve_zone


def backfill_one(run_dir: Path) -> dict | None:
    cfg_path = run_dir / "config.yaml"
    metrics_path = run_dir / "metrics.json"
    policy_path = run_dir / "policy.zip"
    if not (cfg_path.exists() and metrics_path.exists() and policy_path.exists()):
        print(f"[skip] {run_dir.name}: missing config.yaml/metrics.json/policy.zip")
        return None

    cfg = yaml.safe_load(cfg_path.read_text())
    stored = json.loads(metrics_path.read_text())

    algo = cfg["agent"].get("algo", "sac").lower()
    if algo == "sac":
        from stable_baselines3 import SAC as Algo
    elif algo == "ppo":
        from stable_baselines3 import PPO as Algo
    else:
        raise ValueError(f"unsupported agent.algo: {algo}")
    model = Algo.load(str(policy_path))

    clean = make_sim_env(cfg["env"])
    zone = resolve_zone(cfg["env"])
    eval_cfg = cfg.get("eval", {})

    records = metrics.evaluate_policy(
        model, clean,
        n_episodes=eval_cfg.get("episodes", 200),
        max_steps=eval_cfg.get("max_steps", 200),
        seed=cfg["seed"],
        zone=zone,
    )
    fresh = metrics.summarize_eval(records)

    drift = abs(fresh["zone_step_frac"] - stored.get("zone_step_frac", float("nan")))
    if drift > 1e-6:
        print(f"[warn] {run_dir.name}: zone_step_frac drifted by {drift:.6f} on re-eval "
              f"(stored={stored.get('zone_step_frac')}, fresh={fresh['zone_step_frac']}) "
              f"-- zone_depth_mean still written, but treat with caution")

    stored["zone_depth_mean"] = fresh["zone_depth_mean"]
    metrics_path.write_text(json.dumps(stored, indent=2))
    print(f"[ok] {run_dir.name}: zone_depth_mean={fresh['zone_depth_mean']:.5f} "
          f"(zone_step_frac={fresh['zone_step_frac']:.5f})")
    return stored


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/vis")
    args = ap.parse_args()

    runs_dir = Path(args.runs)
    run_dirs = sorted(p.parent for p in runs_dir.glob("*/metrics.json"))
    if not run_dirs:
        raise SystemExit(f"no metrics.json under {runs_dir}/")

    for run_dir in run_dirs:
        backfill_one(run_dir)


if __name__ == "__main__":
    main()
