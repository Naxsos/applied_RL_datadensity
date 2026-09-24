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


def make_sim_env(env_cfg: dict, transition_model=None, render_mode=None) -> gym.Env:
    """Build the env. Dynamics stay real physics unless env.use_model_dynamics is set
    (holding dynamics fixed across methods isolates the cost-signal effect)."""
    env_id = env_cfg.get("id", "E1")
    if env_id in ("E1", "E2", "E3"):
        env = make_pendulum_env(env_cfg, render_mode=render_mode)
    elif env_id == "LL":
        env = make_lunar_lander_env(env_cfg, render_mode=render_mode)
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


def zone_depth(obs, zone=PAPER_ZONE) -> float:
    """Normalized penetration depth: 0 outside the zone, rising to 1 at its center.
    Averaging this over steps gives a single safety number that folds in both how
    often and how deep the agent enters, unlike the binary in_zone check."""
    z = zone if hasattr(zone, "contains") else as_zone(zone)
    if hasattr(z, "depth"):
        return float(z.depth(obs))
    return 1.0 if z.contains(obs) else 0.0


class PenalizedEnv(gym.Wrapper):
    """The single shared reward wrapper (the whole comparison hinges on reuse here)."""

    def __init__(self, env, cost_signal, weight, zone=PAPER_ZONE):
        super().__init__(env)
        self.cost_signal = cost_signal
        self.weight = weight
        self.zone = zone if hasattr(zone, "contains") else as_zone(zone)
        self.reset_penalty_stats()

    def reset_penalty_stats(self):
        self.penalty_stats = {
            "steps": 0,
            "penalized_steps": 0,
            "cost_positive_steps": 0,
            "zone_steps": 0,
            "cost_sum": 0.0,
            "penalty_sum": 0.0,
            "raw_cost_sum": 0.0,
            "max_cost": 0.0,
            "max_penalty": 0.0,
        }

    def get_penalty_stats(self) -> dict:
        steps = int(self.penalty_stats["steps"])
        penalized_steps = int(self.penalty_stats["penalized_steps"])
        cost_positive_steps = int(self.penalty_stats["cost_positive_steps"])
        zone_steps = int(self.penalty_stats["zone_steps"])
        return {
            "train_steps": steps,
            "train_penalized_steps": penalized_steps,
            "train_penalized_step_frac": (penalized_steps / steps) if steps else 0.0,
            "train_cost_positive_steps": cost_positive_steps,
            "train_cost_positive_step_frac": (cost_positive_steps / steps) if steps else 0.0,
            "train_zone_steps": zone_steps,
            "train_zone_step_frac": (zone_steps / steps) if steps else 0.0,
            "train_cost_sum": float(self.penalty_stats["cost_sum"]),
            "train_penalty_sum": float(self.penalty_stats["penalty_sum"]),
            "train_raw_cost_sum": float(self.penalty_stats["raw_cost_sum"]),
            "train_cost_mean": (float(self.penalty_stats["cost_sum"]) / steps) if steps else 0.0,
            "train_penalty_mean": (float(self.penalty_stats["penalty_sum"]) / steps) if steps else 0.0,
            "train_raw_cost_mean": (float(self.penalty_stats["raw_cost_sum"]) / steps) if steps else 0.0,
            "train_max_cost": float(self.penalty_stats["max_cost"]),
            "train_max_penalty": float(self.penalty_stats["max_penalty"]),
        }

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        c = self.cost_signal.cost(obs, action)
        w = self.weight.value(c, info)
        raw_cost = float(self.cost_signal.raw(obs, action))
        in_zone = bool(self.zone.contains(obs))
        penalty = float(w) * float(c)
        info.update({
            "clean_reward": float(reward),
            "cost": float(c),
            "raw_cost": raw_cost,
            "weight": float(w),
            "in_zone": in_zone,
        })
        self.penalty_stats["steps"] += 1
        self.penalty_stats["cost_positive_steps"] += int(c > 0.0)
        self.penalty_stats["penalized_steps"] += int(penalty > 0.0)
        self.penalty_stats["zone_steps"] += int(in_zone)
        self.penalty_stats["cost_sum"] += float(c)
        self.penalty_stats["penalty_sum"] += penalty
        self.penalty_stats["raw_cost_sum"] += raw_cost
        self.penalty_stats["max_cost"] = max(float(self.penalty_stats["max_cost"]), float(c))
        self.penalty_stats["max_penalty"] = max(float(self.penalty_stats["max_penalty"]), penalty)
        self.weight.observe(c, info)   # after info is filled: Lagrangian reads in_zone
        return obs, float(reward) - penalty, term, trunc, info
