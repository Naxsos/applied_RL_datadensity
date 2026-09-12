"""Pendulum-specific env helpers and wrappers."""
from __future__ import annotations

import gymnasium as gym
import numpy as np

# Paper's excluded low-density zone: θ ∈ [2π/5, 3π/5] -- on ONE side of the swing.
ZONE_LOW, ZONE_HIGH = 2 * np.pi / 5, 3 * np.pi / 5
# E3 = E1 physics but a wider/shifted danger zone.
E3_ZONE = (np.pi / 4, 3 * np.pi / 4)


def obs_to_theta(obs) -> float:
    """Pendulum obs = [cosθ, sinθ, θ̇] -> θ in (-π, π]."""
    return float(np.arctan2(obs[1], obs[0]))


class Zone:
    """Pendulum excluded angle band."""

    def __init__(self, lo, hi, symmetric: bool = False):
        self.lo, self.hi = float(lo), float(hi)
        self.symmetric = bool(symmetric)

    def contains(self, obs) -> bool:
        theta = obs_to_theta(obs)
        if self.symmetric:
            theta = abs(theta)
        return self.lo <= theta <= self.hi

    def depth(self, obs) -> float:
        """Normalized penetration depth: 0 outside the zone or right at either edge,
        rising linearly to 1 at the zone center. Combines frequency and severity when
        averaged over steps -- a policy that only clips the edge scores near 0 per
        step even on every entry, one that crosses the center scores near 1."""
        theta = obs_to_theta(obs)
        if self.symmetric:
            theta = abs(theta)
        if not (self.lo <= theta <= self.hi):
            return 0.0
        half_width = (self.hi - self.lo) / 2.0
        if half_width <= 0:
            return 1.0
        center = (self.lo + self.hi) / 2.0
        return 1.0 - abs(theta - center) / half_width

    def __iter__(self):
        return iter((self.lo, self.hi))

    def __eq__(self, other):
        o = as_zone(other) if not isinstance(other, Zone) else other
        return (self.lo, self.hi, self.symmetric) == (o.lo, o.hi, o.symmetric)

    def __repr__(self):
        return (f"Zone(lo={self.lo:.4f}, hi={self.hi:.4f}, "
                f"{'symmetric' if self.symmetric else 'one-sided'})")


def as_zone(zone, symmetric: bool = False) -> Zone:
    if isinstance(zone, Zone):
        return zone
    lo, hi = zone
    return Zone(lo, hi, symmetric=symmetric)


PAPER_ZONE = Zone(ZONE_LOW, ZONE_HIGH)


def in_zone(obs, zone=PAPER_ZONE) -> bool:
    return as_zone(zone).contains(obs)


class ChaoticBandPendulum(gym.Wrapper):
    """E2 testbed — makes uncertainty ≠ density."""

    def __init__(self, env, band=(-0.6, 0.6), noise=0.6, seed=0):
        super().__init__(env)
        self.band = band
        self.noise = noise
        self._rng = np.random.default_rng(seed)

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        th = obs_to_theta(obs)
        if self.band[0] <= th <= self.band[1]:
            obs = np.asarray(obs, dtype=np.float32).copy()
            obs[2] += self.noise * self._rng.standard_normal()
            info["chaotic_band"] = True
        return obs, reward, term, trunc, info


class FixedStart(gym.Wrapper):
    """Spec §3: episode starts from the resting state with optional noise."""

    def __init__(self, env, theta: float = np.pi, theta_dot: float = 0.0,
                 noise: float = 0.05, seed: int = 0):
        super().__init__(env)
        self.theta, self.theta_dot, self.noise = float(theta), float(theta_dot), float(noise)
        self._rng = np.random.default_rng(seed)

    def reset(self, **kwargs):
        seed = kwargs.get("seed")
        rng = np.random.default_rng(seed) if seed is not None else self._rng
        obs, info = self.env.reset(**kwargs)
        th = self.theta + rng.uniform(-self.noise, self.noise)
        thd = self.theta_dot + rng.uniform(-self.noise, self.noise)
        self.env.unwrapped.state = np.array([th, thd], dtype=np.float64)
        return np.asarray(self.env.unwrapped._get_obs(), dtype=np.float32), info


def make_pendulum_env(env_cfg: dict) -> gym.Env:
    env_id = env_cfg.get("id", "E1")
    if env_id == "E2":
        env = ChaoticBandPendulum(
            gym.make("Pendulum-v1"),
            band=tuple(env_cfg.get("chaotic_band", (-0.6, 0.6))),
            noise=env_cfg.get("chaotic_noise", 0.6),
            seed=env_cfg.get("seed", 0),
        )
    else:
        env = gym.make("Pendulum-v1")

    start = env_cfg.get("start_state")
    if start:
        env = FixedStart(env, theta=start.get("theta", np.pi),
                         theta_dot=start.get("theta_dot", 0.0),
                         noise=start.get("noise", 0.05),
                         seed=env_cfg.get("seed", 0))
    return env


def resolve_pendulum_zone(env_cfg: dict) -> Zone:
    env_id = env_cfg.get("id", "E1")
    if env_id == "E3":
        zone_tuple = env_cfg.get("excluded_zone", E3_ZONE)
    else:
        zone_tuple = env_cfg.get("excluded_zone", (ZONE_LOW, ZONE_HIGH))
    return as_zone(zone_tuple, symmetric=env_cfg.get("zone_symmetric", False))
