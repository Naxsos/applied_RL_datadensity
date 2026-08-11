# Experiment-Runner Spec — Density/Uncertainty-Guided RL

**Goal:** Compare four methods on one identical footing so the decision table is trustworthy,
then double down on the winner and benchmark it against the paper (ES2025-194).

The trick to comparable numbers: the paper's penalized reward has **two independent knobs**.
We implement each as a pluggable interface so *all four methods reuse the same env wrapper,
the same SAC config, the same evaluation loop, and the same logger*. Only the plugged-in
component changes between runs.

```
effective_reward = task_reward(s, a)  −  weight.value(cost, info) · cost_signal.cost(s, a)
                   └── clean, unpenalized ──┘   └── AXIS 2: HOW MUCH ──┘   └─ AXIS 1: WHAT ─┘
```

---

## 1. The four methods = combinations of two axes

| Run label   | Cost signal (AXIS 1)        | Weight (AXIS 2)         | Origin            |
|-------------|-----------------------------|-------------------------|-------------------|
| `baseline`  | `KDEDensityCost`            | `FixedWeight(p)`        | Paper             |
| `ens`       | `EnsembleVarianceCost`      | `FixedWeight(λ)`        | Gerrit/Jonas 1a   |
| `bnn`       | `BNNUncertaintyCost`        | `FixedWeight(λ)`        | Gerrit/Jonas 1b   |
| `lagr`      | `KDEDensityCost`            | `LagrangianWeight(ε)`   | Maram/Theresa 3   |
| `bnn+lagr`* | `BNNUncertaintyCost`        | `LagrangianWeight(ε)`   | stretch combo     |

\* Optional stretch cell — only run if time allows (keep it simple per professor's note).

Because AXIS 1 and AXIS 2 are orthogonal, the table answers **two questions independently**:
"best cost signal?" (baseline vs ens vs bnn) and "fixed-p vs learned-α?" (baseline vs lagr).

---

## 2. Pluggable interfaces (the whole point)

```python
class CostSignal(Protocol):
    """AXIS 1 — WHAT to penalize. Must return a scalar in [0, 1]."""
    def cost(self, s, a) -> float: ...          # 0 = safe/known, 1 = penalized
    def fit(self, dataset) -> None: ...         # trained ONCE on the offline data, frozen after
    @property
    def raw(self, s, a) -> float: ...           # unnormalized value (density / variance / NLL) for logging

class PenaltyWeight(Protocol):
    """AXIS 2 — HOW MUCH. Maps a cost to a scalar penalty weight."""
    def value(self, cost, info) -> float: ...   # FixedWeight: constant; Lagrangian: current α
    def update(self, batch_cost_mean) -> None:  # no-op for FixedWeight; dual step for Lagrangian
        ...
    def log_state(self) -> dict: ...            # {} for fixed; {"alpha":…, "constraint":…} for Lagrangian
```

### Cost-signal normalization (critical for comparability)
All three cost signals must land on the **same [0,1] scale** or the weights aren't comparable.
Fit the normalizer on the offline dataset's cost distribution, freeze it:

- `KDEDensityCost.cost = 1 − clip(density(s) / density_ref, 0, 1)` (ref = paper threshold 0.025 region)
- `EnsembleVarianceCost.cost = clip(σ²_ensemble(s,a) / σ²_p95, 0, 1)` (p95 from offline data)
- `BNNUncertaintyCost.cost = clip(predictive_var(s,a) / var_p95, 0, 1)`

Log **both** `cost` (normalized, used in reward) and `raw` (native units, for the correlation metric).

### Weight implementations
```python
FixedWeight(p):            value = p                    # p ∈ sweep, e.g. {2,10,30} scaled to [0,1] cost
LagrangianWeight(ε, η):    value = α  (current)
    update(C_mean):        α ← max(0, α + η·(C_mean − ε))   # dual ascent; C_mean = mean cost over batch
```

---

## 3. Shared, frozen components (identical across ALL runs)

- **Transition model:** the paper's LSTM (50 units + dense, 4-step sliding window), trained once,
  reused by every run. *Exception:* `ens` needs N of them (see §6 compute note).
- **Agent:** SAC (Stable-Baselines3), default hyperparameters, **150 000** training steps.
- **Seeds:** `{0, 1, 2, 3, 4}` — every cell is the mean ± std over these 5 seeds. No single-run cells.
- **Start state:** fixed resting position θ=π, θ̇=0 (paper setup).
- **Eval:** 2000 episodes, ≤200 steps each, actions greedy, on the **clean unpenalized reward**.

---

## 4. Environments

| Env id | What                                                      | Isolates                        |
|--------|-----------------------------------------------------------|---------------------------------|
| `E1`   | Pendulum, excluded zone θ∈[2π/5, 3π/5] (paper-identical)  | reproduce Table 1; lagr vs p    |
| `E2`   | uncertainty ≠ density testbed (below)                     | ensemble/bnn cost-signal claim  |
| `E3`   | E1 with unknown/shifted optimal-p                         | lagr auto-tuning claim          |

**E2 construction** (the decisive one for the ensemble): produce an offline dataset with two
contrasting regions —
- **Region A:** sparse data, *simple* dynamics (low true forecast error).
- **Region B:** dense data, *chaotic* dynamics (high true forecast error).

Use Acrobot / double-pendulum swing-up, or Pendulum with an injected chaotic well-sampled band.
KDE penalizes A (wrong); a good uncertainty signal penalizes B (right).

---

## 5. Metrics — exact schema every run must log

One `metrics.json` per (method, env, seed). Field names are fixed so aggregation is trivial.

```jsonc
{
  "run_id": "ens_E2_seed3",
  "method": "ens", "env": "E2", "seed": 3,

  // --- SAFETY (headline, ↓ better) ---
  "zone_visit_rate": 0.15,        // frac. of eval episodes passing through penalized zone
  "left_path_pct": 15, "right_path_pct": 85,   // reproduces Table 1

  // --- TASK PERFORMANCE (↑ better), on CLEAN reward ---
  "true_return_mean": -142.3, "true_return_std": 12.1,
  "upright_success_rate": 0.91,  // stabilized upright within episode
  "time_to_upright_mean": 63.0,  // steps

  // --- SIGNAL CORRECTNESS (the E2 decider) ---
  "cost_vs_forecast_err_corr": 0.78,   // Spearman(cost(s,a), true transition MSE) on held-out grid
  "penalizes_region": "B",             // which region got the mass ("A"=wrong, "B"=right for E2)

  // --- CONSTRAINT (lagr only, else null) ---
  "alpha_final": 4.2, "alpha_trajectory": [...],
  "constraint_C": 0.019, "epsilon": 0.02, "constraint_satisfied": true,

  // --- COMPUTE (real decision criterion per transcript) ---
  "wall_clock_train_s": 1830, "peak_mem_mb": 5400,
  "n_models": 5, "total_params": 210000,
  "fits_local": true,           // peak_mem_mb < LOCAL_MEM_BUDGET (set to your MacBook's)

  "hparam": {"knob": "lambda", "value": 0.6}   // the swept knob for this run
}
```

**HP-robustness** is *derived at aggregation time*, not per run: for each method, take the sweep
over its own knob and compute the spread of `(true_return_mean, zone_visit_rate)` across knob
values. Flat = robust = easy to set (the Lagrangian's selling point). Report as
`robustness_score = 1 − normalized_spread`.

---

## 6. Run matrix

```
methods × envs × seeds × knob-sweep

baseline : E1,E3   × 5 seeds × p ∈ {2,10,30}
ens      : E1,E2   × 5 seeds × λ ∈ {0.2,0.6,1.0}   (+ N ∈ {3,5} as secondary)
bnn      : E1,E2   × 5 seeds × λ ∈ {0.2,0.6,1.0}
lagr     : E1,E3   × 5 seeds × ε ∈ {0.02,0.05}
```

**Compute note (ens):** N LSTMs ≈ N× transition-model train + inference cost. `bnn` exists
precisely to get the same uncertainty signal at ~1× cost — the `wall_clock_train_s` /
`fits_local` columns are where that trade-off shows up. Log honestly; do not silently cap N.

**Staging (professor's "kleine Experimente zuerst"):**
1. Week-1 local pass: `baseline` on E1 (reproduce Table 1) + one default seed of each method on E1. Eyeball the 4 core metrics.
2. Decisive pass: full matrix, one A100 batch on Vast.ai (E2 + E3 + seeds + sweeps).
3. Aggregate → fill decision table → pick winner(s).

---

## 7. Output layout & aggregation

```
runs/
  <run_id>/
    metrics.json
    config.yaml            # exact resolved config for reproducibility
    alpha.csv              # lagr only
    trajectories.parquet   # for the histogram/path plots (Fig 3 style)
aggregate.py  →  results.csv (one row per run) + decision_table.md (means ± std)
```

`aggregate.py` produces the final decision matrix directly from `results.csv`:

| Criterion (weight)          | baseline | ens | bnn | lagr |
|-----------------------------|----------|-----|-----|------|
| zone_visit_rate ↓ (0.30)    |          |     |     |      |
| true_return ↑ (0.25)        |          |     |     |      |
| robustness_score ↑ (0.20)   |          |     |     |      |
| fits_local / compute (0.15) |          |     |     |      |
| cost_vs_forecast_corr (0.10)|          |     |     |      |
| **weighted total**          |          |     |     |      |

---

## 8. Config skeleton (`config.yaml`)

```yaml
run_id: ens_E2_seed3
seed: 3
env:
  id: E2                       # E1 | E2 | E3
  excluded_zone: [1.2566, 1.8849]   # [2π/5, 3π/5]
  start_state: {theta: 3.14159, theta_dot: 0.0}
transition_model:
  type: lstm
  units: 50
  window: 4
  frozen_checkpoint: models/lstm_E2.pt
cost_signal:
  type: ensemble_variance      # kde_density | ensemble_variance | bnn_uncertainty
  n_models: 5                  # ens only
  normalizer: {method: p95, ref_dataset: data/E2_offline.parquet}
weight:
  type: fixed                  # fixed | lagrangian
  p: 0.6                       # fixed only  (the swept knob)
  # --- lagrangian only ---
  # epsilon: 0.02
  # eta_alpha: 0.01
  # alpha_init: 1.0
agent:
  algo: sac
  total_steps: 150000
eval:
  episodes: 2000
  max_steps: 200
  reward: clean                # ALWAYS evaluate on unpenalized reward
compute:
  local_mem_budget_mb: 18000   # set to your MacBook; drives fits_local
```

---

## 9. Non-negotiables (or the table lies)

1. **Evaluate on the clean reward.** Methods with different penalties are only comparable on the
   unpenalized task objective.
2. **≥5 seeds per cell.** SAC variance will otherwise flip the ranking.
3. **Same frozen transition model** for baseline/bnn/lagr; N copies for ens. Nothing else differs.
4. **Both `cost` and `raw` logged.** The E2 decider (`cost_vs_forecast_err_corr`) needs native units.
5. **Log dropped work.** If N is capped or a sweep point is skipped, write it to the run — silent
   truncation reads as full coverage.
```
