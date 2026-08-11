"""AXIS 1 — WHAT to penalize (§2 of the spec).

Every CostSignal returns a scalar in [0, 1]:  0 = safe/known, 1 = fully penalized.
`raw()` returns the native unit (density / variance / predictive-var) for the
E2 decider metric `cost_vs_forecast_err_corr`.

Normalization is fit ONCE on the offline dataset and frozen, so that a given
weight value means the same thing across all three signals.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class CostSignal(ABC):
    """Pluggable cost signal. Fit once on offline data, then frozen."""

    @abstractmethod
    def fit(self, dataset: dict) -> "CostSignal":
        """dataset keys: 'obs' (N,obs_dim), 'act' (N,act_dim), 'next_obs' (N,obs_dim)."""

    @abstractmethod
    def raw(self, obs, action) -> float:
        """Native-unit value (unnormalized), for logging + correlation metric."""

    @abstractmethod
    def cost(self, obs, action) -> float:
        """Normalized penalty in [0, 1]."""

    # --- shared p95 normalization helper ---
    def _fit_p95(self, raw_values) -> float:
        raw = np.asarray(list(raw_values), dtype=float)
        return float(max(np.percentile(raw, 95), 1e-8))


class KDEDensityCost(CostSignal):
    """Paper baseline: KDE density -> penalize LOW density (backward-looking).

    cost = 1 - clip(density / density_ref, 0, 1)
    """

    def __init__(self, bandwidth: float = 0.1, ref_percentile: float = 5.0):
        self.bandwidth = bandwidth
        self.ref_percentile = ref_percentile
        self._kde = None
        self._ref = 1.0

    @staticmethod
    def _encode(obs):
        # positional encoding of the pendulum state (cosθ, sinθ, θ̇ already in obs)
        return np.atleast_2d(np.asarray(obs, dtype=float))

    def fit(self, dataset):
        from sklearn.neighbors import KernelDensity
        X = self._encode(dataset["obs"])
        self._kde = KernelDensity(bandwidth=self.bandwidth).fit(X)
        # scoring the reference percentile only needs a representative sample, not
        # all N points -- score_samples(X) over the full offline set is O(N^2)-ish
        # and dominates wall-clock once N gets into the tens of thousands (measured:
        # 18s @ N=23.7k vs 185s @ N=72k). Matches the subsampling _VarianceCost uses.
        idx = np.random.default_rng(0).choice(len(X), size=min(2000, len(X)), replace=False)
        dens = np.exp(self._kde.score_samples(X[idx]))
        self._ref = float(max(np.percentile(dens, self.ref_percentile), 1e-8))
        return self

    def raw(self, obs, action) -> float:
        if self._kde is None:
            return 0.0
        return float(np.exp(self._kde.score_samples(self._encode(obs))[0]))

    def cost(self, obs, action) -> float:
        return float(np.clip(1.0 - self.raw(obs, action) / self._ref, 0.0, 1.0))


class _VarianceCost(CostSignal):
    """Shared logic: penalize HIGH predictive variance (forward-looking)."""

    def __init__(self, model, agg: str = "mean"):
        self.model = model   # Ensemble or BNN from transition.py
        self.agg = agg
        self._scale = 1.0

    def _var(self, obs, action) -> float:
        _, var = self.model.predict_dist(obs, action)
        return float(np.mean(var) if self.agg == "mean" else np.max(var))

    def fit(self, dataset):
        obs, act = dataset["obs"], dataset["act"]
        idx = np.random.default_rng(0).choice(len(obs), size=min(2000, len(obs)), replace=False)
        self._scale = self._fit_p95(self._var(obs[i], act[i]) for i in idx)
        return self

    def raw(self, obs, action) -> float:
        return self._var(obs, action)

    def cost(self, obs, action) -> float:
        return float(np.clip(self.raw(obs, action) / self._scale, 0.0, 1.0))


class EnsembleVarianceCost(_VarianceCost):
    """Run `ens` — variance across N independent models (Gerrit/Jonas 1a)."""


class BNNUncertaintyCost(_VarianceCost):
    """Run `bnn` — MC-dropout predictive variance from one model (Gerrit/Jonas 1b)."""
