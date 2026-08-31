#!/usr/bin/env python3
"""Evaluate a hand-built reference policy on the clean Pendulum task.

The automatic controller has two deliberately simple phases:

1. replay a bang-bang torque schedule that swings up on the side opposite the
   one-sided excluded zone;
2. switch to a clipped PD controller near the upright position.

No RL model, transition model, density estimate, or reward penalty is used.
This makes the result a useful feasibility/reference row for the decision table.

Examples (from the repository root):
    python scripts/manual_policy.py
    python scripts/manual_policy.py --episodes 20 --start-noise 0.02
    python scripts/manual_policy.py --keyboard
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from denrl.env import FixedStart, Zone, obs_to_theta  # noqa: E402


class ManualReferencePolicy:
    """Open-loop bang-bang swing-up followed by feedback balancing."""

    def __init__(self, torque_file: Path, kp: float = 6.0, kd: float = 1.0):
        self.swing_actions = np.asarray(np.load(torque_file), dtype=np.float32)
        if self.swing_actions.ndim != 1 or self.swing_actions.size == 0:
            raise ValueError(f"expected a non-empty 1-D torque sequence in {torque_file}")
        self.kp, self.kd = float(kp), float(kd)
        self.step_index = 0

    def reset(self) -> None:
        self.step_index = 0

    def action(self, obs) -> np.ndarray:
        if self.step_index < len(self.swing_actions):
            torque = float(self.swing_actions[self.step_index])
        else:
            theta = obs_to_theta(obs)
            theta_dot = float(obs[2])
            torque = float(np.clip(-self.kp * theta - self.kd * theta_dot, -2.0, 2.0))
        self.step_index += 1
        return np.array([torque], dtype=np.float32)


def keyboard_action() -> tuple[np.ndarray, bool]:
    """Read keys from Gymnasium's pygame window: arrows apply torque, Q quits."""
    import pygame

    pygame.event.pump()
    keys = pygame.key.get_pressed()
    torque = 2.0 * float(keys[pygame.K_RIGHT]) - 2.0 * float(keys[pygame.K_LEFT])
    return np.array([torque], dtype=np.float32), bool(keys[pygame.K_q] or keys[pygame.K_ESCAPE])


def evaluate(args) -> dict:
    import gymnasium as gym

    zone = Zone(np.radians(args.zone[0]), np.radians(args.zone[1]))
    render_mode = "human" if args.render or args.keyboard else None
    env = FixedStart(
        gym.make("Pendulum-v1", render_mode=render_mode),
        theta=np.pi,
        theta_dot=0.0,
        noise=args.start_noise,
        seed=args.seed,
    )
    policy = ManualReferencePolicy(args.torque_file, args.kp, args.kd)
    rng = np.random.default_rng(args.seed)
    records = []

    try:
        for _ in range(args.episodes):
            obs, _ = env.reset(seed=int(rng.integers(1 << 31)))
            policy.reset()
            episode_return = 0.0
            entered_zone = False
            upright_at = None
            quit_requested = False

            for step in range(args.max_steps):
                if args.keyboard:
                    action, quit_requested = keyboard_action()
                    if quit_requested:
                        break
                else:
                    action = policy.action(obs)

                obs, reward, terminated, truncated, _ = env.step(action)
                theta = obs_to_theta(obs)
                episode_return += float(reward)
                entered_zone |= zone.contains(obs)
                if upright_at is None and abs(theta) < 0.2:
                    upright_at = step
                if terminated or truncated:
                    break

            records.append({
                "return": episode_return,
                "entered_zone": entered_zone,
                "upright": upright_at is not None,
                "steps_to_upright": upright_at if upright_at is not None else args.max_steps,
            })
            if quit_requested:
                break
    finally:
        env.close()

    returns = np.asarray([record["return"] for record in records], dtype=float)
    result = {
        "method": "manual_keyboard" if args.keyboard else "manual_reference",
        "episodes": len(records),
        "zone_degrees": [float(args.zone[0]), float(args.zone[1])],
        "zone_visit_rate": float(np.mean([r["entered_zone"] for r in records])),
        "upright_success_rate": float(np.mean([r["upright"] for r in records])),
        "time_to_upright_mean": float(np.mean([r["steps_to_upright"] for r in records])),
        "true_return_mean": float(returns.mean()),
        "true_return_std": float(returns.std()),
    }
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start-noise", type=float, default=0.0,
                        help="uniform reset noise for both angle and angular velocity")
    parser.add_argument("--zone", nargs=2, type=float, default=(72.0, 108.0),
                        metavar=("LOW_DEG", "HIGH_DEG"))
    parser.add_argument("--torque-file", type=Path,
                        default=ROOT / "data" / "reference_safe_swingup_E1.npy")
    parser.add_argument("--kp", type=float, default=6.0)
    parser.add_argument("--kd", type=float, default=1.0)
    parser.add_argument("--render", action="store_true", help="show the automatic controller")
    parser.add_argument("--keyboard", action="store_true",
                        help="control torque with left/right arrows; Q or Escape quits")
    parser.add_argument("--json", type=Path, help="also write the result to this JSON file")
    args = parser.parse_args()
    if args.episodes < 1 or args.max_steps < 1:
        parser.error("--episodes and --max-steps must be positive")
    if args.start_noise < 0:
        parser.error("--start-noise cannot be negative")
    return args


def main() -> None:
    args = parse_args()
    result = evaluate(args)
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
