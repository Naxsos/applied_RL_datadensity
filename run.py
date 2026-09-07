#!/usr/bin/env python3
"""Single-run entrypoint: one (method, env, seed) -> runs/<run_id>/metrics.json.

Usage:
    python run.py configs/baseline_E1.yaml --seed 0
    python run.py configs/ens_E2.yaml --seed 3 --steps 150000

Every run reuses the same wrapper/eval/logger; only the plugged-in cost_signal
and weight differ. See EXPERIMENT_SPEC.md §5 for the metrics schema.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml


def load_offline(env_cfg: dict) -> dict:
    """Load the frozen offline dataset the cost signals are fit on."""
    import pandas as pd
    path = env_cfg.get("offline_dataset", f"data/{env_cfg['id']}_offline.parquet")
    df = pd.read_parquet(path)
    obs = df[[c for c in df.columns if c.startswith("obs")]].to_numpy()
    act = df[[c for c in df.columns if c.startswith("act")]].to_numpy()
    nxt = df[[c for c in df.columns if c.startswith("next")]].to_numpy()
    return {"obs": obs, "act": act, "next_obs": nxt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--eval-episodes", type=int, default=None, help="override eval.episodes (fast smoke tests)")
    ap.add_argument("--out", default="runs")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    cfg["seed"] = args.seed
    total_steps = args.steps or cfg["agent"].get("total_steps", 150_000)

    run_id = cfg.get("run_id") or f"{cfg['method']}_{cfg['env']['id']}_seed{args.seed}"
    out = Path(args.out) / run_id
    out.mkdir(parents=True, exist_ok=True)
    # write the resolved config up front: a full run is ~30 min, so an interrupted
    # one otherwise leaves a bare directory with no record of what it was
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    np.random.seed(args.seed)
    try:
        import torch
        torch.manual_seed(args.seed)
    except Exception:
        pass

    # --- build the four-method-agnostic pipeline ---
    from denrl.registry import build_env
    from denrl.weights import WeightUpdateCallback, LagrangianWeight
    from denrl import metrics
    from stable_baselines3 import SAC, PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    penalized, clean, cost_signal, weight, transition_model, zone = build_env(cfg)

    # fit cost-signal normalizer ONCE on offline data, then freeze
    cost_signal.fit(load_offline(cfg["env"]))

    # n_envs = 1 so the single weight instance sees every step (Lagrangian needs this)
    vec = DummyVecEnv([lambda: penalized])

    callbacks = []
    if isinstance(weight, LagrangianWeight):
        callbacks.append(WeightUpdateCallback(weight, update_freq=cfg["weight"].get("update_freq", 1000)))

    algo = cfg["agent"].get("algo", "sac").lower()
    lr = cfg["agent"].get("learning_rate")
    if algo == "sac":
        kwargs = {"seed": args.seed, "verbose": 0}
        if lr is not None:
            kwargs["learning_rate"] = lr
        model = SAC("MlpPolicy", vec, **kwargs)
    elif algo == "ppo":
        kwargs = {"seed": args.seed, "verbose": 0}
        if lr is not None:
            kwargs["learning_rate"] = lr
        model = PPO("MlpPolicy", vec, **kwargs)
    else:
        raise ValueError(f"unsupported agent.algo: {algo}")

    t0 = time.perf_counter()
    model.learn(total_timesteps=total_steps, callback=callbacks or None)
    wall = time.perf_counter() - t0

    # --- evaluate on CLEAN reward ---
    eval_cfg = cfg.get("eval", {})
    records = metrics.evaluate_policy(
        model, clean,
        n_episodes=args.eval_episodes or eval_cfg.get("episodes", 2000),
        max_steps=eval_cfg.get("max_steps", 200),
        seed=args.seed,
        zone=zone,
    )
    result = metrics.summarize_eval(records)

    # signal-correctness (E2 decider) — needs a held-out grid with ground truth
    grid = load_offline(cfg["env"])
    result.update(metrics.signal_correctness(cost_signal, transition_model, grid, None))

    # compute + constraint + provenance
    n_models = getattr(transition_model, "n_models", 1)
    peak_mem = _peak_mem_mb()
    result.update({
        "run_id": run_id,
        "method": cfg["method"],
        "env": cfg["env"]["id"],
        "seed": args.seed,
        "wall_clock_train_s": round(wall, 1),
        "peak_mem_mb": peak_mem,
        "n_models": n_models,
        "fits_local": (peak_mem is None) or (peak_mem < cfg.get("compute", {}).get("local_mem_budget_mb", 18000)),
        "hparam": _knob(cfg),
    })
    result.update(weight.log_state())
    if isinstance(weight, LagrangianWeight):
        # log C alongside α: a flat α with an unsatisfied C means the dual step is
        # too slow, which is invisible from alpha.csv alone
        rows = ["update,alpha,constraint_C"]
        rows += [f"{i},{a},{c}" for i, (a, c) in
                 enumerate(zip(weight.trajectory[1:], weight.constraint_trajectory))]
        (out / "alpha.csv").write_text("\n".join(rows))
        result["alpha_trajectory_len"] = len(weight.trajectory)
        result["alpha_init"] = weight.trajectory[0]

    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    model.save(out / "policy")
    print(f"[{run_id}] zone={result['zone_visit_rate']:.3f} "
          f"return={result['true_return_mean']:.1f} wall={wall:.0f}s -> {out}/metrics.json")


def _knob(cfg: dict) -> dict:
    w = cfg["weight"]
    if w["type"] == "fixed":
        return {"knob": "p", "value": w["p"]}
    return {"knob": "epsilon", "value": w["epsilon"]}


def _peak_mem_mb():
    try:
        import resource
        import sys
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(rss / (1024 if sys.platform == "darwin" else 1024) / 1024, 1) if sys.platform == "darwin" else round(rss / 1024, 1)
    except Exception:
        return None


if __name__ == "__main__":
    main()
