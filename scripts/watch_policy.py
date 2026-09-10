#!/usr/bin/env python3
"""Watch a trained policy run live in a Gymnasium render window.

Usage:
    python scripts/watch_policy.py runs/lagr_E1_epsilon0.02_seed0
    python scripts/watch_policy.py runs/baseline_E1_p10_seed0 --episodes 5
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml
import gymnasium as gym
from stable_baselines3 import SAC

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="path to run directory containing policy.zip and config.yaml")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--stochastic", action="store_true", help="sample from policy instead of taking mean")
    ap.add_argument("--delay", type=float, default=0.1, help="seconds between steps (default 0.1)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    model = SAC.load(str(run_dir / "policy"), env=None)

    from denrl.env import make_sim_env
    env = make_sim_env(cfg["env"], render_mode="human")

    from denrl.env import obs_to_theta
    for ep in range(args.episodes):
        obs, _ = env.reset()
        ep_ret = 0.0
        thetas = []
        step = 0
        while True:
            env.render()
            action, _ = model.predict(obs, deterministic=not args.stochastic)
            obs, reward, term, trunc, _ = env.step(action)
            ep_ret += float(reward)
            th = obs_to_theta(obs)
            thetas.append(th)
            from denrl.env import in_zone, PAPER_ZONE
            iz = PAPER_ZONE.contains(obs)
            print(f"  step {step:3d}  θ={th:+.3f}  cos={obs[0]:+.3f}  sin={obs[1]:+.3f}  {'IN ZONE!' if iz else ''}", end="\r")
            step += 1
            time.sleep(args.delay)
            if term or trunc:
                break
        travel = float(np.sum(np.diff(np.unwrap(thetas)))) if len(thetas) > 1 else 0.0
        path = "right" if travel > 0 else "left"
        print(f"\nepisode {ep + 1}: return={ep_ret:.1f}  travel={travel:.2f}  path={path}  "
              f"θ_start={thetas[0]:.2f}  θ_end={thetas[-1]:.2f}")

    env.close()


if __name__ == "__main__":
    main()
