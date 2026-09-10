"""Metric computation matching the §5 schema exactly.

Everything that goes into a table cell is computed here so field names stay
identical across runs and `aggregate.py` can be mechanical.
"""
from __future__ import annotations
from typing import Optional
import numpy as np

from .env import obs_to_theta, in_zone, PAPER_ZONE


def _looks_like_lunar_lander(obs) -> bool:
    return np.shape(obs)[0] == 8


def _ll_touchdown_like(obs) -> bool:
    arr = np.asarray(obs, dtype=float).reshape(-1)
    if arr.shape[0] < 8:
        return False
    x, y, vx, vy, angle, ang_vel, leg_l, leg_r = [float(v) for v in arr[:8]]
    stable_pose = (abs(x) < 0.25 and y < 0.50 and
                   abs(vx) < 0.20 and abs(vy) < 0.20 and
                   abs(angle) < 0.25 and abs(ang_vel) < 0.25)
    has_contact = (leg_l > 0.5) or (leg_r > 0.5)
    return bool(stable_pose or has_contact)


def evaluate_policy(model, clean_env, n_episodes: int = 2000, max_steps: int = 200, seed: int = 0,
                     zone=PAPER_ZONE, collect_obs: bool = False):
    """Roll the policy on the CLEAN (unpenalized) env. Returns per-episode records.

    Each record: {return, entered_zone, path, upright, steps_to_upright}.
    For LL, landing is reported both as strict terminal landings and
    touchdown-like stable near-pad outcomes.
    """
    rng = np.random.default_rng(seed)
    records = []
    all_obs = []
    env_is_ll = getattr(clean_env.unwrapped, "spec", None) and getattr(clean_env.unwrapped.spec, "id", "") == "LunarLander-v3"
    for ep in range(n_episodes):
        obs, _ = clean_env.reset(seed=int(rng.integers(1 << 31)))
        ep_ret, entered, thetas = 0.0, False, []
        zone_steps = 0
        landed_at: Optional[int] = None
        strict_landed = False
        crashed = False
        timeout = False
        upright_at: Optional[int] = None
        last_reward = 0.0
        last_obs = np.asarray(obs)
        for t in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = clean_env.step(action)
            last_obs = np.asarray(obs)
            if collect_obs:
                all_obs.append(last_obs)
            last_reward = float(reward)
            ep_ret += last_reward
            if not _looks_like_lunar_lander(obs) and np.shape(obs)[0] >= 3:
                th = obs_to_theta(obs)
                thetas.append(th)
                if upright_at is None and abs(th) < 0.2:
                    upright_at = t
            in_z = in_zone(obs, zone)
            entered = entered or in_z
            zone_steps += int(in_z)
            if env_is_ll:
                if term and landed_at is None and not crashed:
                    # LunarLander-v3 does not expose landed/crashed info keys.
                    # Final positive terminal reward is a robust proxy for successful landing.
                    if last_reward > 0.0:
                        strict_landed = True
                        landed_at = t
                    else:
                        crashed = True
                timeout = timeout or bool(trunc)
            if term or trunc:
                break
        if env_is_ll and (not strict_landed) and _ll_touchdown_like(last_obs):
            landed_at = t if landed_at is None else landed_at
        if env_is_ll and not (term or trunc):
            timeout = True
        records.append({
            "return": ep_ret,
            "entered_zone": entered,
            "zone_steps": zone_steps,
            "zone_step_frac": zone_steps / max(t + 1, 1),
            "path": None if env_is_ll else _classify_path(thetas),
            "upright": None if env_is_ll else (upright_at is not None),
            "steps_to_upright": None if env_is_ll else (upright_at if upright_at is not None else max_steps),
            "landed": (landed_at is not None) if env_is_ll else None,
            "strict_landed": strict_landed if env_is_ll else None,
            "crashed": crashed if env_is_ll else None,
            "timed_out": timeout if env_is_ll else None,
            "episode_len": t + 1,
            "env_kind": "LL" if env_is_ll else "pendulum",
        })
    if collect_obs:
        return records, np.array(all_obs, dtype=np.float32)
    return records


def _classify_path(thetas) -> str:
    """Which way did the swing-up go? Sign of cumulative angle travel (crude but consistent)."""
    if len(thetas) < 2:
        return "none"
    travel = np.sum(np.diff(np.unwrap(thetas)))
    return "right" if travel > 0 else "left"


def summarize_eval(records) -> dict:
    rets = np.array([r["return"] for r in records], dtype=float)
    n = len(records)
    ll_mode = any(r.get("env_kind") == "LL" for r in records)
    n_left = sum(r.get("path") == "left" for r in records)
    n_right = sum(r.get("path") == "right" for r in records)
    out = {
        "zone_visit_rate": float(np.mean([r["entered_zone"] for r in records])),
        "zone_step_frac": float(np.mean([r["zone_step_frac"] for r in records])),
        "zone_steps_mean": float(np.mean([r["zone_steps"] for r in records])),
        "left_path_pct": None if ll_mode else round(100 * n_left / max(n, 1)),
        "right_path_pct": None if ll_mode else round(100 * n_right / max(n, 1)),
        "true_return_mean": float(rets.mean()),
        "true_return_std": float(rets.std()),
        "upright_success_rate": None if ll_mode else float(np.mean([r["upright"] for r in records])),
        "time_to_upright_mean": None if ll_mode else float(np.mean([r["steps_to_upright"] for r in records])),
        "episode_len_mean": float(np.mean([r.get("episode_len", 0) for r in records])),
    }
    if ll_mode:
        out["landing_success_rate"] = float(np.mean([r.get("landed", False) for r in records]))
        out["strict_landing_rate"] = float(np.mean([r.get("strict_landed", False) for r in records]))
        out["crash_rate"] = float(np.mean([r.get("crashed", False) for r in records]))
        out["timeout_rate"] = float(np.mean([r.get("timed_out", False) for r in records]))
    return out


def signal_correctness(cost_signal, transition_model, grid_obs, grid_act) -> dict:
    """The E2 decider: does the cost track TRUE forecast error?

    Spearman correlation between cost(s,a) and the transition model's forecast
    MSE on a held-out grid. High = penalty fires where the model is actually wrong.
    Requires a transition_model with .predict and ground-truth next states.
    """
    from scipy.stats import spearmanr
    n = len(grid_obs["obs"])
    # subsample to keep wall-clock reasonable (KDE query is O(n_fit) per point)
    max_points = 2000
    if n > max_points:
        idx = np.random.default_rng(0).choice(n, size=max_points, replace=False)
    else:
        idx = np.arange(n)
    costs, errs = [], []
    for i in idx:
        obs, act, next_true = grid_obs["obs"][i], grid_obs["act"][i], grid_obs["next_obs"][i]
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

    sweep_rows: list of dicts with 'true_return_mean' and 'zone_step_frac'.
    Flat trade-off across knob values -> high score -> 'easy to set'.
    """
    if len(sweep_rows) < 2:
        return float("nan")
    rets = np.array([r["true_return_mean"] for r in sweep_rows], dtype=float)
    zones = np.array([r["zone_step_frac"] for r in sweep_rows], dtype=float)

    def _norm_spread(x):
        rng = x.max() - x.min()
        scale = abs(np.mean(x)) + 1e-8
        return rng / scale

    spread = 0.5 * (_norm_spread(rets) + _norm_spread(zones))
    return float(max(0.0, 1.0 - spread))
