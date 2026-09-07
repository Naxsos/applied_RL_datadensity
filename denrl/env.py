"""Simulated env + shared penalized reward wrapper (§2, §3)."""
from __future__ import annotations

import numpy as np
import gymnasium as gym

from .env_lunar_lander import as_lunar_zone, make_lunar_lander_env
from .env_pendulum import (
    PAPER_ZONE,
    ZONE_HIGH,
    ZONE_LOW,
    ChaoticBandPendulum,
    FixedStart,
    Zone,
    as_zone,
    in_zone as pendulum_in_zone,
    make_pendulum_env,
    obs_to_theta,
    resolve_pendulum_zone,
)

__all__ = [
    "ZONE_LOW",
    "ZONE_HIGH",
    "PAPER_ZONE",
    "Zone",
    "as_zone",
    "obs_to_theta",
    "in_zone",
    "ChaoticBandPendulum",
    "FixedStart",
    "resolve_zone",
    "make_sim_env",
    "PenalizedEnv",
]


class _ModelDynamics(gym.Wrapper):
    """Replaces the base env's transition with a learned model's prediction."""

    def __init__(self, env, model):
        super().__init__(env)
        self.model = model

    def step(self, action):
        obs = self.env.unwrapped._get_obs() if hasattr(self.env.unwrapped, "_get_obs") else None
        # advance the real env for reward/termination bookkeeping...
        _obs, reward, term, trunc, info = self.env.step(action)
        # ...but overwrite the *state* with the model's prediction (model-based RL)
        if obs is not None:
            pred = self.model.predict(obs, action)
            info["model_obs"] = pred
            return np.asarray(pred, dtype=np.float32), reward, term, trunc, info
        return _obs, reward, term, trunc, info


def resolve_zone(env_cfg: dict):
    env_id = env_cfg.get("id", "E1")
    if env_id in ("E1", "E2", "E3"):
        return resolve_pendulum_zone(env_cfg)
    if env_id == "LL":
        return as_lunar_zone(env_cfg.get("excluded_zone", None))
    raise ValueError(f"unknown env id {env_id}")


def make_sim_env(env_cfg: dict, transition_model=None) -> gym.Env:
    """Build the env. Dynamics stay real physics unless env.use_model_dynamics is set
    (holding dynamics fixed across methods isolates the cost-signal effect)."""
    env_id = env_cfg.get("id", "E1")
    if env_id in ("E1", "E2", "E3"):
        env = make_pendulum_env(env_cfg)
    elif env_id == "LL":
        env = make_lunar_lander_env(env_cfg)
    else:
        raise ValueError(f"unknown env id {env_id}")

    # model-based rollout is OPT-IN; default keeps identical real dynamics for all methods
    if transition_model is not None and env_cfg.get("use_model_dynamics", False):
        env = _ModelDynamics(env, transition_model)
    return env


def in_zone(obs, zone=PAPER_ZONE) -> bool:
    if hasattr(zone, "contains"):
        return bool(zone.contains(obs))
    return pendulum_in_zone(obs, zone)


class PenalizedEnv(gym.Wrapper):
    """The single shared reward wrapper (the whole comparison hinges on reuse here)."""

    def __init__(self, env, cost_signal, weight, zone=PAPER_ZONE):
        super().__init__(env)
        self.cost_signal = cost_signal
        self.weight = weight
        self.zone = zone if hasattr(zone, "contains") else as_zone(zone)

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        c = self.cost_signal.cost(obs, action)
        w = self.weight.value(c, info)
        info.update({
            "clean_reward": float(reward),
            "cost": float(c),
            "raw_cost": float(self.cost_signal.raw(obs, action)),
            "weight": float(w),
            "in_zone": self.zone.contains(obs),
        })
        self.weight.observe(c, info)   # after info is filled: Lagrangian reads in_zone
        return obs, float(reward) - w * c, term, trunc, info
