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

# Mirrors gymnasium's PendulumEnv.render() geometry (screen_dim=500, bound=2.2,
# rod anchored at screen center, rotated by theta + pi/2, final surf flipped
# vertically) so the overlay lines up with the rod pixel-for-pixel.
_RENDER_BOUND = 2.2
_RENDER_FPS = 30


def _theta_to_point(theta, radius, screen_dim, offset):
    import pygame

    v = pygame.math.Vector2(radius, 0).rotate_rad(theta + np.pi / 2)
    x = v.x + offset
    y = screen_dim - (v.y + offset)  # undo PendulumEnv's final vertical flip
    return int(x), int(y)


def draw_forbidden_zone(screen, screen_dim, zone):
    """Draw the excluded angle band as a red wedge onto an already-blitted frame."""
    if not hasattr(zone, "lo"):
        return  # not a pendulum-style angle zone (e.g. lunar lander)
    from pygame import gfxdraw

    scale = screen_dim / (_RENDER_BOUND * 2)
    offset = screen_dim // 2
    radius = scale  # matches rod_length = 1 * scale

    def draw_band(lo, hi):
        n = max(2, int(abs(hi - lo) / 0.05) + 1)
        center = (offset, screen_dim - offset)
        arc = [_theta_to_point(t, radius, screen_dim, offset) for t in np.linspace(lo, hi, n)]
        wedge = [center] + arc
        gfxdraw.filled_polygon(screen, wedge, (220, 40, 40, 90))
        gfxdraw.aapolygon(screen, wedge, (150, 20, 20, 200))

    draw_band(zone.lo, zone.hi)
    if zone.symmetric:
        draw_band(-zone.hi, -zone.lo)


def render_with_zone(env, zone, screen, clock):
    """Render one frame ourselves (rgb_array -> blit -> overlay -> single flip)
    instead of using render_mode="human", whose own internal flip would show a
    bare pendulum frame before our overlay lands, causing visible flicker."""
    import pygame

    frame = env.render()  # (H, W, 3) uint8
    surf = pygame.surfarray.make_surface(frame.transpose(1, 0, 2))
    screen.blit(surf, (0, 0))
    draw_forbidden_zone(screen, env.unwrapped.screen_dim, zone)
    pygame.event.pump()
    pygame.display.flip()
    clock.tick(_RENDER_FPS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="path to run directory containing policy.zip and config.yaml")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--stochastic", action="store_true", help="sample from policy instead of taking mean")
    ap.add_argument("--delay", type=float, default=0, help="seconds between steps (default 0.1)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    model = SAC.load(str(run_dir / "policy"), env=None)

    from denrl.env import make_sim_env, resolve_zone
    env = make_sim_env(cfg["env"], render_mode="rgb_array")
    zone = resolve_zone(cfg["env"])

    import pygame
    pygame.init()
    pygame.display.init()
    screen = pygame.display.set_mode((env.unwrapped.screen_dim, env.unwrapped.screen_dim))
    clock = pygame.time.Clock()

    from denrl.env import obs_to_theta
    for ep in range(args.episodes):
        obs, _ = env.reset()
        ep_ret = 0.0
        thetas = []
        step = 0
        while True:
            render_with_zone(env, zone, screen, clock)
            action, _ = model.predict(obs, deterministic=not args.stochastic)
            obs, reward, term, trunc, _ = env.step(action)
            ep_ret += float(reward)
            th = obs_to_theta(obs)
            thetas.append(th)
            iz = zone.contains(obs)
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
