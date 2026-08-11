"""Simulated env + the single penalized-reward wrapper shared by ALL methods (§2, §3).

`make_sim_env` builds the model-based env (paper: transition model inside step()).
If no transition model is supplied it falls back to real Gymnasium physics so the
pipeline runs before the LSTM is trained.

`PenalizedEnv` is the ONE place the reward is modified:
    effective = clean − weight.value(cost) · cost
Every method plugs a (cost_signal, weight) pair into this identical wrapper.
"""
from __future__ import annotations
import numpy as np
import gymnasium as gym

# Paper's excluded low-density zone: θ ∈ [2π/5, 3π/5]
ZONE_LOW, ZONE_HIGH = 2 * np.pi / 5, 3 * np.pi / 5


def obs_to_theta(obs) -> float:
    """Pendulum obs = [cosθ, sinθ, θ̇] -> θ in (-π, π]."""
    return float(np.arctan2(obs[1], obs[0]))


def in_zone(obs, zone=(ZONE_LOW, ZONE_HIGH)) -> bool:
    lo, hi = zone
    return lo <= abs(obs_to_theta(obs)) <= hi


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


class ChaoticBandPendulum(gym.Wrapper):
    """E2 testbed — makes uncertainty ≠ density.

    Injects stochastic dynamics in a well-sampled 'chaotic' angle band (Region B):
    dense data BUT high model uncertainty. The paper's excluded zone stays
    sparse-but-simple (Region A): low density BUT trivially predictable.

    A density model (KDE) will penalize Region A (wrong); an uncertainty model
    (ensemble / BNN) should penalize Region B (right). That contrast is the whole
    point of E2 — see `cost_vs_forecast_err_corr` in metrics.

    NOTE: this injects *aleatoric* noise, which ensemble variance captures well.
    For a purely *epistemic* contrast, instead make Region B deterministic-but-
    more-nonlinear and undersample it; the wrapper below is the simplest version
    that already separates the methods.
    """

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
            obs[2] += self.noise * self._rng.standard_normal()  # perturb θ̇ in the band
            info["chaotic_band"] = True
        return obs, reward, term, trunc, info


def make_sim_env(env_cfg: dict, transition_model=None) -> gym.Env:
    """Build the env. Dynamics stay real physics unless env.use_model_dynamics is set
    (holding dynamics fixed across methods isolates the cost-signal effect)."""
    env_id = env_cfg.get("id", "E1")
    if env_id in ("E1", "E3"):
        env = gym.make("Pendulum-v1")
    elif env_id == "E2":
        env = ChaoticBandPendulum(
            gym.make("Pendulum-v1"),
            band=tuple(env_cfg.get("chaotic_band", (-0.6, 0.6))),
            noise=env_cfg.get("chaotic_noise", 0.6),
            seed=env_cfg.get("seed", 0),
        )
    else:
        raise ValueError(f"unknown env id {env_id}")

    # model-based rollout is OPT-IN; default keeps identical real dynamics for all methods
    if transition_model is not None and env_cfg.get("use_model_dynamics", False):
        env = _ModelDynamics(env, transition_model)
    return env


class PenalizedEnv(gym.Wrapper):
    """The single shared reward wrapper (the whole comparison hinges on reuse here)."""

    def __init__(self, env, cost_signal, weight, zone=(ZONE_LOW, ZONE_HIGH)):
        super().__init__(env)
        self.cost_signal = cost_signal
        self.weight = weight
        self.zone = zone

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        c = self.cost_signal.cost(obs, action)
        w = self.weight.value(c, info)
        info.update({
            "clean_reward": float(reward),
            "cost": float(c),
            "raw_cost": float(self.cost_signal.raw(obs, action)),
            "weight": float(w),
            "in_zone": in_zone(obs, self.zone),
        })
        self.weight.observe(c, info)   # after info is filled: Lagrangian reads in_zone
        return obs, float(reward) - w * c, term, trunc, info
