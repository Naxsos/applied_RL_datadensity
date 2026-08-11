"""denrl — density/uncertainty-guided RL comparison harness.

Implements EXPERIMENT_SPEC.md: two orthogonal, pluggable axes
(CostSignal = WHAT to penalize, PenaltyWeight = HOW MUCH) that every method
reuses so table cells are comparable.
"""
from .costs import CostSignal, KDEDensityCost, EnsembleVarianceCost, BNNUncertaintyCost
from .weights import PenaltyWeight, FixedWeight, LagrangianWeight, WeightUpdateCallback
from .env import PenalizedEnv, make_sim_env
from .registry import build_cost_signal, build_weight, build_env

__all__ = [
    "CostSignal", "KDEDensityCost", "EnsembleVarianceCost", "BNNUncertaintyCost",
    "PenaltyWeight", "FixedWeight", "LagrangianWeight", "WeightUpdateCallback",
    "PenalizedEnv", "make_sim_env",
    "build_cost_signal", "build_weight", "build_env",
]
