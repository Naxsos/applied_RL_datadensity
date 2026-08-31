"""AXIS 2 — HOW MUCH to penalize (§2 of the spec).

FixedWeight     : the paper's constant p.
LagrangianWeight: learned multiplier α via dual ascent (Maram/Theresa 3).
                  α ← max(0, α + η·(C_mean − ε))

The env calls `observe(cost)` every step; a callback calls `update()` every
`update_freq` steps. Keep n_envs = 1 so a single shared weight instance sees all
steps (see run.py).
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List
import numpy as np

try:
    from stable_baselines3.common.callbacks import BaseCallback
except Exception:  # SB3 optional at import time
    BaseCallback = object


class PenaltyWeight(ABC):
    @abstractmethod
    def value(self, cost, info) -> float:
        """Current weight applied to the cost this step."""

    def observe(self, cost: float, info: dict | None = None) -> None:
        """Accumulate step costs (no-op for fixed). `info` carries the step's
        constraint indicators (e.g. `in_zone`) so a weight can pick its own."""

    def update(self) -> None:
        """Dual step (no-op for fixed)."""

    def log_state(self) -> dict:
        return {}


class FixedWeight(PenaltyWeight):
    """Paper: constant p."""

    def __init__(self, p: float):
        self.p = float(p)

    def value(self, cost, info) -> float:
        return self.p

    def log_state(self) -> dict:
        return {"p": self.p}


class LagrangianWeight(PenaltyWeight):
    """Constrained-MDP dual variable. Only knob is ε (allowed violation rate).

    ``constraint`` selects WHAT ε bounds — the two are on different scales, so
    mixing them silently makes the constraint unsatisfiable:

    - ``"zone"``   C = fraction of TRAINING STEPS inside the excluded zone. ε is
      then an allowed *violation rate* (ε=0.02 -> "in the zone at most 2% of
      steps") — the standard CMDP expected-cost-per-step form. Default.
      NB this is not the same number as the reported `zone_visit_rate`, which is
      per-EPISODE (did the episode ever enter). Per-episode is always the larger
      of the two; don't read ε as a bound on it.
    - ``"cost"``   C = mean normalized cost signal (spec §2's literal reading).
      Note KDE cost is nonzero almost everywhere, so a meaningful ε here is
      O(0.1-1), NOT the 0.02 that makes sense as a violation rate.

    Both quantities are always accumulated and logged, so a run is diagnosable
    regardless of which one drives the dual step.
    """

    CONSTRAINTS = ("zone", "cost")

    def __init__(self, epsilon: float, eta_alpha: float = 0.1,
                 alpha_init: float = 1.0, alpha_max: float | None = None,
                 constraint: str = "zone"):
        if constraint not in self.CONSTRAINTS:
            raise ValueError(f"constraint must be one of {self.CONSTRAINTS}, got {constraint!r}")
        self.epsilon = float(epsilon)
        self.eta = float(eta_alpha)
        self.alpha = float(alpha_init)
        self.alpha_max = alpha_max
        self.constraint = constraint
        self._costs: List[float] = []
        self._zone: List[float] = []
        self.trajectory: List[float] = [self.alpha]
        self.constraint_trajectory: List[float] = []
        self.last_constraint: float | None = None
        self.last_cost_mean: float | None = None
        self.last_zone_rate: float | None = None

    def value(self, cost, info) -> float:
        return self.alpha

    def observe(self, cost: float, info: dict | None = None) -> None:
        self._costs.append(float(cost))
        self._zone.append(float(bool((info or {}).get("in_zone", False))))

    def update(self) -> None:
        if not self._costs:
            return
        self.last_cost_mean = float(np.mean(self._costs))
        self.last_zone_rate = float(np.mean(self._zone))
        C_mean = self.last_zone_rate if self.constraint == "zone" else self.last_cost_mean

        self.alpha = max(0.0, self.alpha + self.eta * (C_mean - self.epsilon))
        if self.alpha_max is not None:
            self.alpha = min(self.alpha, self.alpha_max)
        self.last_constraint = C_mean
        self.trajectory.append(self.alpha)
        self.constraint_trajectory.append(C_mean)
        self._costs.clear()
        self._zone.clear()

    def log_state(self) -> dict:
        return {
            "alpha": self.alpha,
            "epsilon": self.epsilon,
            "constraint_signal": self.constraint,
            "constraint_C": self.last_constraint,
            "constraint_C_zone_rate": self.last_zone_rate,   # both logged: the two
            "constraint_C_cost_mean": self.last_cost_mean,   # scales are not comparable
            "constraint_satisfied": (self.last_constraint is not None
                                     and self.last_constraint <= self.epsilon),
        }


class WeightUpdateCallback(BaseCallback):
    """Triggers weight.update() every `update_freq` env steps and logs alpha.csv."""

    def __init__(self, weight: PenaltyWeight, update_freq: int = 1000,
                 out_dir: str | None = None, verbose: int = 0):
        super().__init__(verbose)
        self.weight = weight
        self.update_freq = update_freq
        self.out_dir = out_dir
        self._csv_path = None

    def _on_training_start(self):
        if self.out_dir is not None and isinstance(self.weight, LagrangianWeight):
            from pathlib import Path
            self._csv_path = Path(self.out_dir) / "alpha.csv"
            self._csv_path.write_text("step,alpha,constraint_C\n")

    def _on_step(self) -> bool:
        if self.n_calls % self.update_freq == 0:
            self.weight.update()
            if self._csv_path is not None:
                w = self.weight
                with open(self._csv_path, "a") as f:
                    f.write(f"{self.num_timesteps},{w.alpha:.4f},{w.last_constraint or 0:.4f}\n")
        return True


class EvalCallback(BaseCallback):
    """Periodically evaluate on the clean env and log to a CSV."""

    def __init__(self, clean_env, zone, out_dir, eval_freq: int = 10_000,
                 n_episodes: int = 50, max_steps: int = 200, verbose: int = 0):
        super().__init__(verbose)
        self.clean_env = clean_env
        self.zone = zone
        self.out_dir = out_dir
        self.eval_freq = eval_freq
        self.n_episodes = n_episodes
        self.max_steps = max_steps
        self._csv_path = None

    def _on_training_start(self):
        from pathlib import Path
        self._csv_path = Path(self.out_dir) / "train_metrics.csv"
        self._csv_path.write_text(
            "timestep,true_return_mean,true_return_std,zone_step_frac,zone_steps_mean,"
            "upright_success_rate,time_to_upright_mean,left_path_pct,right_path_pct\n"
        )

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            self._evaluate()
        return True

    def _evaluate(self):
        from . import metrics
        records, obs = metrics.evaluate_policy(
            self.model, self.clean_env,
            n_episodes=self.n_episodes,
            max_steps=self.max_steps,
            seed=self.n_calls,
            zone=self.zone,
            collect_obs=True,
        )
        summary = metrics.summarize_eval(records)
        row = (f"{self.num_timesteps},{summary['true_return_mean']:.2f},"
               f"{summary['true_return_std']:.2f},"
               f"{summary['zone_step_frac']:.4f},{summary['zone_steps_mean']:.1f},"
               f"{summary['upright_success_rate']:.4f},{summary['time_to_upright_mean']:.1f},"
               f"{summary['left_path_pct']},{summary['right_path_pct']}\n")
        with open(self._csv_path, "a") as f:
            f.write(row)
        print(f"  [eval @{self.num_timesteps}] return={summary['true_return_mean']:.1f} "
              f"zone={summary['zone_step_frac']:.3f} ({summary['zone_steps_mean']:.1f} steps) "
              f"upright={summary['upright_success_rate']:.2f}")
        self._plot_trajectories(obs, summary)

    def _plot_trajectories(self, obs, summary):
        from pathlib import Path
        from scripts.plot_trajectories import plot_trajectory_snapshot
        title = (f"step {self.num_timesteps}  zone={summary['zone_step_frac']:.3f}  "
                 f"return={summary['true_return_mean']:.0f}")
        plot_trajectory_snapshot(obs, self.zone,
                                 Path(self.out_dir) / "traj" / f"traj_{self.num_timesteps}.png",
                                 title=title, method=self._method)

    @property
    def _method(self):
        """Best-effort method name from the run config."""
        from pathlib import Path
        cfg_path = Path(self.out_dir) / "config.yaml"
        if cfg_path.exists():
            import yaml
            return yaml.safe_load(cfg_path.read_text()).get("method", "baseline")
        return "baseline"
