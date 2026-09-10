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


def _auto_run_id(cfg: dict, seed: int) -> str:
    w = cfg["weight"]
    if w["type"] == "fixed":
        knob = f"p{w['p']}"
    elif w["type"] == "lagrangian":
        knob = f"epsilon{w['epsilon']}"
    else:
        knob = w["type"]
    return f"{cfg['method']}_{cfg['env']['id']}_{knob}_seed{seed}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--eval-episodes", type=int, default=None, help="override eval.episodes (fast smoke tests)")
    ap.add_argument("--device", default=None, help="torch device for supported models: auto, cpu, mps, cuda")
    ap.add_argument("--out", default="runs")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    cfg["seed"] = args.seed
    total_steps = args.steps or cfg["agent"].get("total_steps", 150_000)

    if cfg.get("run_id"):
        run_id = cfg["run_id"]
    else:
        knob = _knob(cfg)
        run_id = f"{cfg['method']}_{cfg['env']['id']}_{knob['knob']}{knob['value']}_seed{args.seed}"    out = Path(args.out) / run_id
    if out.exists():
        v = 2
        while (Path(args.out) / f"{run_id}_v{v}").exists():
            v += 1
        run_id = f"{run_id}_v{v}"
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
        torch = None

    # --- build the four-method-agnostic pipeline ---
    from denrl.registry import build_env
    from denrl.weights import WeightUpdateCallback, EvalCallback, LagrangianWeight
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
        callbacks.append(WeightUpdateCallback(weight, update_freq=cfg["weight"].get("update_freq", 1000),
                                                 out_dir=str(out)))

    eval_cfg = cfg.get("eval", {})
    callbacks.append(EvalCallback(
        clean_env=clean, zone=zone, out_dir=out,
        eval_freq=eval_cfg.get("eval_freq", 10_000),
        n_episodes=eval_cfg.get("eval_episodes_during_training", 50),
        max_steps=eval_cfg.get("max_steps", 200),
    ))

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

    # save policy FIRST so it's never lost if final eval is slow/hangs
    model.save(out / "policy")

    # --- evaluate on CLEAN reward ---
    eval_cfg = cfg.get("eval", {})
    records, eval_obs = metrics.evaluate_policy(
        model, clean,
        n_episodes=args.eval_episodes or eval_cfg.get("episodes", 200),
        max_steps=eval_cfg.get("max_steps", 200),
        seed=args.seed,
        zone=zone,
        collect_obs=True,
    )
    result = metrics.summarize_eval(records)

    from scripts.plot_trajectories import plot_trajectory_snapshot
    plot_trajectory_snapshot(
        eval_obs, zone, out / "traj" / "traj_final.png",
        title=f"final  zone={result['zone_step_frac']:.3f}  return={result['true_return_mean']:.0f}",
        method=cfg["method"],
    )

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
        "device": device,
    })
    result.update(weight.log_state())
    if isinstance(weight, LagrangianWeight):
        result["alpha_trajectory_len"] = len(weight.trajectory)
        result["alpha_init"] = weight.trajectory[0]

    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    print(f"[{run_id}] zone_frac={result['zone_step_frac']:.3f} "
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


def _resolve_device(torch_mod, requested: str | None) -> str:
    if requested is None or requested == "auto":
        if torch_mod is None:
            return "cpu"
        if hasattr(torch_mod.backends, "mps") and torch_mod.backends.mps.is_available():
            return "mps"
        if torch_mod.cuda.is_available():
            return "cuda"
        return "cpu"

    device = requested.lower()
    if device == "mps":
        if torch_mod is None or not hasattr(torch_mod.backends, "mps") or not torch_mod.backends.mps.is_available():
            raise RuntimeError("device=mps requested but torch MPS is not available")
        return device
    if device == "cuda":
        if torch_mod is None or not torch_mod.cuda.is_available():
            raise RuntimeError("device=cuda requested but torch CUDA is not available")
        return device
    if device == "cpu":
        return device
    raise ValueError(f"unsupported device: {requested}")


if __name__ == "__main__":
    main()
