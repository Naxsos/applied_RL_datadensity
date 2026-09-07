"""Build components from a config dict (§8). Keeps run.py declarative."""
from __future__ import annotations

from .costs import KDEDensityCost, EnsembleVarianceCost, BNNUncertaintyCost
from .weights import FixedWeight, LagrangianWeight
from .env import make_sim_env, PenalizedEnv, resolve_zone
from .transition import load_transition


def build_cost_signal(cfg: dict, transition_model=None):
    t = cfg["type"]
    if t == "kde_density":
        return KDEDensityCost(
            bandwidth=cfg.get("bandwidth", 0.1),
            ref_percentile=cfg.get("ref_percentile", 5.0),
            max_fit_points=cfg.get("max_fit_points", 25_000),
        )
    if t == "ensemble_variance":
        if transition_model is None:
            raise ValueError("ensemble_variance needs an Ensemble transition model")
        return EnsembleVarianceCost(transition_model, agg=cfg.get("agg", "mean"))
    if t == "bnn_uncertainty":
        if transition_model is None:
            raise ValueError("bnn_uncertainty needs a BNN transition model")
        return BNNUncertaintyCost(transition_model, agg=cfg.get("agg", "mean"))
    raise ValueError(f"unknown cost_signal.type {t}")


def build_weight(cfg: dict):
    t = cfg["type"]
    if t == "fixed":
        return FixedWeight(p=cfg["p"])
    if t == "lagrangian":
        return LagrangianWeight(
            epsilon=cfg["epsilon"],
            eta_alpha=cfg.get("eta_alpha", 0.1),
            alpha_init=cfg.get("alpha_init", 1.0),
            alpha_max=cfg.get("alpha_max"),
            constraint=cfg.get("constraint", "zone"),
        )
    raise ValueError(f"unknown weight.type {t}")


def build_env(cfg: dict):
    """Returns (penalized_env, clean_env, cost_signal, weight, transition_model, zone).

    `zone` is the env-specific excluded low-density region for this run.
    Pendulum accepts an angle band (`env.excluded_zone` + `env.zone_symmetric`);
    LunarLander accepts a threshold dictionary under `env.excluded_zone`.
    """
    transition_model = load_transition(cfg["transition_model"])
    cost_signal = build_cost_signal(cfg["cost_signal"], transition_model)
    weight = build_weight(cfg["weight"])
    zone = resolve_zone(cfg["env"])

    penalized = PenalizedEnv(make_sim_env(cfg["env"], transition_model), cost_signal, weight, zone=zone)
    clean = make_sim_env(cfg["env"], transition_model)  # unpenalized, for eval
    return penalized, clean, cost_signal, weight, transition_model, zone
