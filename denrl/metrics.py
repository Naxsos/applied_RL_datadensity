"""Metric computation matching the §5 schema exactly.

Everything that goes into a table cell is computed here so field names stay
identical across runs and `aggregate.py` can be mechanical.
"""
from __future__ import annotations
from typing import Optional
import numpy as np

from .env import obs_to_theta, in_zone, PAPER_ZONE


def evaluate_policy(model, clean_env, n_episodes: int = 2000, max_steps: int = 200, seed: int = 0,
                     zone=PAPER_ZONE):
    """Roll the policy on the CLEAN (unpenalized) env. Returns per-episode records.

    Each record: {return, entered_zone, path, upright, steps_to_upright}.
    """
    rng = np.random.default_rng(seed)
    records = []
    for ep in range(n_episodes):
        obs, _ = clean_env.reset(seed=int(rng.integers(1 << 31)))
        ep_ret, entered, thetas = 0.0, False, []
        upright_at: Optional[int] = None
        for t in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = clean_env.step(action)
            ep_ret += float(reward)
            th = obs_to_theta(obs)
            thetas.append(th)
            entered = entered or in_zone(obs, zone)
            if upright_at is None and abs(th) < 0.2:
                upright_at = t
            if term or trunc:
                break
        records.append({
            "return": ep_ret,
            "entered_zone": entered,
            "path": _classify_path(thetas),
            "upright": upright_at is not None,
            "steps_to_upright": upright_at if upright_at is not None else max_steps,
        })
    return records


def _classify_path(thetas) -> str:
    """Which way did the swing-up go? Sign of cumulative angle travel (crude but consistent)."""
    if len(thetas) < 2:
        return "none"
    travel = np.sum(np.diff(np.unwrap(thetas)))
    return "left" if travel > 0 else "right"


def summarize_eval(records) -> dict:
    rets = np.array([r["return"] for r in records], dtype=float)
    n = len(records)
    n_left = sum(r["path"] == "left" for r in records)
    n_right = sum(r["path"] == "right" for r in records)
    return {
        "zone_visit_rate": float(np.mean([r["entered_zone"] for r in records])),
        "left_path_pct": round(100 * n_left / max(n, 1)),
        "right_path_pct": round(100 * n_right / max(n, 1)),
        "true_return_mean": float(rets.mean()),
        "true_return_std": float(rets.std()),
        "upright_success_rate": float(np.mean([r["upright"] for r in records])),
        "time_to_upright_mean": float(np.mean([r["steps_to_upright"] for r in records])),
    }


def signal_correctness(cost_signal, transition_model, grid_obs, grid_act) -> dict:
    """The E2 decider: does the cost track TRUE forecast error?

    Spearman correlation between cost(s,a) and the transition model's forecast
    MSE on a held-out grid. High = penalty fires where the model is actually wrong.
    Requires a transition_model with .predict and ground-truth next states.
    """
    from scipy.stats import spearmanr
    costs, errs = [], []
    for obs, act, next_true in zip(grid_obs["obs"], grid_obs["act"], grid_obs["next_obs"]):
        costs.append(cost_signal.cost(obs, act))
        if transition_model is not None:
            pred = transition_model.predict(obs, act)
            errs.append(float(np.mean((pred - next_true) ** 2)))
        else:
            errs.append(np.nan)
    costs, errs = np.array(costs), np.array(errs)
    if np.all(np.isnan(errs)):
        return {"cost_vs_forecast_err_corr": None, "penalizes_region": None}
    corr = float(spearmanr(costs, errs, nan_policy="omit").correlation)
    return {"cost_vs_forecast_err_corr": corr, "penalizes_region": None}  # region tag: set per-E2 dataset


def robustness_score(sweep_rows) -> float:
    """1 - normalized spread of (return, zone_rate) across a method's own knob sweep.

    sweep_rows: list of dicts with 'true_return_mean' and 'zone_visit_rate'.
    Flat trade-off across knob values -> high score -> 'easy to set'.
    """
    if len(sweep_rows) < 2:
        return float("nan")
    rets = np.array([r["true_return_mean"] for r in sweep_rows], dtype=float)
    zones = np.array([r["zone_visit_rate"] for r in sweep_rows], dtype=float)

    def _norm_spread(x):
        rng = x.max() - x.min()
        scale = abs(np.mean(x)) + 1e-8
        return rng / scale

    spread = 0.5 * (_norm_spread(rets) + _norm_spread(zones))
    return float(max(0.0, 1.0 - spread))
