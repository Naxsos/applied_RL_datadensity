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

# Paper's excluded low-density zone: θ ∈ [2π/5, 3π/5] -- on ONE side of the swing.
ZONE_LOW, ZONE_HIGH = 2 * np.pi / 5, 3 * np.pi / 5


def obs_to_theta(obs) -> float:
    """Pendulum obs = [cosθ, sinθ, θ̇] -> θ in (-π, π]."""
    return float(np.arctan2(obs[1], obs[0]))


class Zone:
    """The excluded angle band, and which side(s) of the swing it blocks.

    ``symmetric=False`` (paper, default): the band is θ ∈ [lo, hi], one side
    only. A swing-up from the resting state (θ=π) to upright (θ=0) can route
    around it via the other side -- that detour is exactly what Table 1's
    left/right path split measures, and it is what makes `zone_visit_rate` a
    measure of *avoidance*.

    ``symmetric=True``: the band is mirrored onto |θ|, blocking BOTH routes.
    Every swing-up must then cross it, so `zone_visit_rate` degenerates into
    "did the agent swing up at all" (measured: baseline p=30 scored 0.49 by
    only reaching upright in 6% of episodes, while a policy that solves the
    task scores ~0.9). Retained only because E2's frozen offline dataset and
    transition models were generated under it; do not use for new experiments.
    """

    def __init__(self, lo, hi, symmetric: bool = False):
        self.lo, self.hi = float(lo), float(hi)
        self.symmetric = bool(symmetric)

    def contains(self, obs) -> bool:
        theta = obs_to_theta(obs)
        if self.symmetric:
            theta = abs(theta)
        return self.lo <= theta <= self.hi

    def __iter__(self):          # keeps `lo, hi = zone` / tuple(zone) working
        return iter((self.lo, self.hi))

    def __eq__(self, other):
        o = as_zone(other) if not isinstance(other, Zone) else other
        return (self.lo, self.hi, self.symmetric) == (o.lo, o.hi, o.symmetric)

    def __repr__(self):
        return (f"Zone(lo={self.lo:.4f}, hi={self.hi:.4f}, "
                f"{'symmetric' if self.symmetric else 'one-sided'})")


def as_zone(zone, symmetric: bool = False) -> Zone:
    """Accept a Zone, or a bare (lo, hi) pair as it comes out of a config."""
    if isinstance(zone, Zone):
        return zone
    lo, hi = zone
    return Zone(lo, hi, symmetric=symmetric)


PAPER_ZONE = Zone(ZONE_LOW, ZONE_HIGH)


def in_zone(obs, zone=PAPER_ZONE) -> bool:
    return as_zone(zone).contains(obs)


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


class FixedStart(gym.Wrapper):
    """Spec §3: every episode starts at the resting position θ=π, θ̇=0.

    Gymnasium's Pendulum resets θ uniformly over the whole circle, which makes
    `zone_visit_rate` mostly a statement about where the episode happened to
    start: a run measured 17/200 episodes starting INSIDE the zone and 42 behind
    it, and even p=2 vs p=10 then land on the same 0.80 rate. From θ=π the two
    routes to upright are mirror images, so whether the agent goes around the
    zone is its own choice -- which is the thing the metric is supposed to
    measure, and what makes the left/right path split meaningful.

    `noise` keeps a little spread: with a deterministic policy and a perfectly
    deterministic start, all N eval episodes would be identical copies and the
    rate could only ever be 0 or 1.
    """

    def __init__(self, env, theta: float = np.pi, theta_dot: float = 0.0,
                 noise: float = 0.05, seed: int = 0):
        super().__init__(env)
        self.theta, self.theta_dot, self.noise = float(theta), float(theta_dot), float(noise)
        self._rng = np.random.default_rng(seed)

    def reset(self, **kwargs):
        # honour an explicit per-episode seed so eval stays reproducible
        seed = kwargs.get("seed")
        rng = np.random.default_rng(seed) if seed is not None else self._rng
        obs, info = self.env.reset(**kwargs)
        th = self.theta + rng.uniform(-self.noise, self.noise)
        thd = self.theta_dot + rng.uniform(-self.noise, self.noise)
        self.env.unwrapped.state = np.array([th, thd], dtype=np.float64)
        return np.asarray(self.env.unwrapped._get_obs(), dtype=np.float32), info


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

    # start state is OPT-IN per config: E2's existing runs were done under gym's
    # random reset, so turning it on globally would make them incomparable.
    start = env_cfg.get("start_state")
    if start:
        env = FixedStart(env, theta=start.get("theta", np.pi),
                         theta_dot=start.get("theta_dot", 0.0),
                         noise=start.get("noise", 0.05),
                         seed=env_cfg.get("seed", 0))
    return env


class PenalizedEnv(gym.Wrapper):
    """The single shared reward wrapper (the whole comparison hinges on reuse here)."""

    def __init__(self, env, cost_signal, weight, zone=PAPER_ZONE):
        super().__init__(env)
        self.cost_signal = cost_signal
        self.weight = weight
        self.zone = as_zone(zone)

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
