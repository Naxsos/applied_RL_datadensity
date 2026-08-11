"""State-transition models (§3, §6 of the spec).

The paper uses an LSTM (50 units, 4-step window). For the scaffold we expose a
minimal delta-dynamics network that satisfies the interface the rest of the
harness needs:

    predict(obs, action)      -> next_obs            (point estimate)
    predict_dist(obs, action) -> (mean, var)         (BNN / MC-dropout only)

An ``Ensemble`` is just N independently-initialised models. A ``BNN`` is one
model with dropout kept ON at inference (MC-dropout) so it yields a predictive
variance at ~1x the cost of the ensemble — that trade-off is what run `bnn`
exists to measure.

Real LSTM + sliding-window training lives in scripts/train_transition.py (stub).
Until a checkpoint exists, cost signals still run end-to-end (numbers are just
not meaningful yet) and the simulated env falls back to real physics.
"""
from __future__ import annotations
from typing import Sequence
import numpy as np

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except Exception:  # torch optional until you actually train
    _HAS_TORCH = False


if _HAS_TORCH:

    class DeltaDynamics(nn.Module):
        """Predicts Δobs from (obs, action). Placeholder for the paper's LSTM."""

        def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64, dropout: float = 0.0):
            super().__init__()
            self.dropout_p = dropout
            self.net = nn.Sequential(
                nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden, obs_dim),
            )

        def forward(self, x):
            return self.net(x)

        @torch.no_grad()
        def predict(self, obs, action) -> np.ndarray:
            x = torch.as_tensor(np.concatenate([np.ravel(obs), np.ravel(action)]),
                                dtype=torch.float32)[None]
            was_training = self.training
            self.eval()
            delta = self(x)[0].cpu().numpy()
            if was_training:
                self.train()
            return np.ravel(obs) + delta


class Ensemble:
    """N independently-initialised models -> epistemic variance (run `ens`)."""

    def __init__(self, models: Sequence):
        assert len(models) >= 2, "ensemble needs >= 2 models"
        self.models = list(models)

    @property
    def n_models(self) -> int:
        return len(self.models)

    def predict(self, obs, action) -> np.ndarray:
        return np.mean([m.predict(obs, action) for m in self.models], axis=0)

    def predict_dist(self, obs, action):
        preds = np.stack([m.predict(obs, action) for m in self.models], axis=0)
        return preds.mean(0), preds.var(0)


class BNN:
    """One dropout model, kept in train() at inference -> MC-dropout variance (run `bnn`)."""

    def __init__(self, model, mc_samples: int = 10):
        self.model = model
        self.mc_samples = mc_samples

    @property
    def n_models(self) -> int:
        return 1

    def predict(self, obs, action) -> np.ndarray:
        return self.predict_dist(obs, action)[0]

    def predict_dist(self, obs, action):
        if not _HAS_TORCH:
            raise RuntimeError("BNN requires torch")
        self.model.train()  # keep dropout active
        preds = np.stack([self.model.predict(obs, action) for _ in range(self.mc_samples)], axis=0)
        return preds.mean(0), preds.var(0)


class PhysicsFallback:
    """Used by the simulated env when no transition checkpoint is provided.

    Steps the *real* Gymnasium dynamics so the whole pipeline runs before the
    LSTM is trained. Not an uncertainty source — cost signals should use a real
    model, this only keeps the sim env alive.
    """

    def __init__(self, env):
        self._env = env

    def predict(self, obs, action) -> np.ndarray:
        raise NotImplementedError("PhysicsFallback is handled inside make_sim_env")


def load_transition(cfg: dict):
    """Build/load the transition model described by a config block.

    cfg example:
        {type: lstm, frozen_checkpoint: models/lstm_E2.pt, obs_dim: 3, act_dim: 1}
    Returns a single model, an Ensemble, or a BNN — or None to signal the sim
    env should use PhysicsFallback.
    """
    ckpt = cfg.get("frozen_checkpoint")
    if ckpt is None:
        return None  # -> physics fallback
    if not _HAS_TORCH:
        raise RuntimeError("torch required to load a transition checkpoint")
    obs_dim, act_dim = cfg.get("obs_dim", 3), cfg.get("act_dim", 1)
    hidden, dropout = cfg.get("hidden", 64), cfg.get("dropout", 0.0)

    def _one(seed_offset=0, drop=dropout):
        m = DeltaDynamics(obs_dim, act_dim, hidden, drop)
        state = torch.load(ckpt, map_location="cpu")
        # checkpoint may be a dict of {idx: state_dict} for ensembles
        if isinstance(state, dict) and seed_offset in state:
            m.load_state_dict(state[seed_offset])
        elif isinstance(state, dict) and "state_dict" in state:
            m.load_state_dict(state["state_dict"])
        else:
            m.load_state_dict(state)
        return m

    kind = cfg.get("type", "lstm")
    if kind == "ensemble":
        n = cfg.get("n_models", 5)
        return Ensemble([_one(i) for i in range(n)])
    if kind == "bnn":
        return BNN(_one(drop=max(dropout, 0.1)), mc_samples=cfg.get("mc_samples", 10))
    return _one()
