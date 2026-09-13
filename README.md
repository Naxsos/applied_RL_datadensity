# Density/Uncertainty-Guided RL

Comparison harness for penalized-reward RL methods that avoid low-density (or
high-uncertainty) regions of the state space, evaluated on Pendulum (E1/E2/E3) and
LunarLander.

## Installation

Requires **Python 3.10–3.13** and pip. Python 3.14 is not yet supported: `gymnasium`'s `box2d`/`pygame` dependencies (needed for LunarLander) don't ship prebuilt wheels for it yet

```bash
git clone <repo-url>
cd applied_RL_datadensity

python3.10 -m venv .venv         
source .venv/bin/activate         # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

`torch` picks a compute device automatically at run time (CUDA > MPS > CPU); pass
`--device cpu|mps|cuda` to `run.py` to override.

To verify the install works end-to-end, run `scripts/smoke_test_pendulum.sh` (see
below).

## Repository layout

| Path | Contents |
|---|---|
| `run.py` | Entrypoint for a single (method, env, seed) run -> `runs/<run_id>/metrics.json`. |
| `aggregate.py` | Collects `runs/*/metrics.json` -> `results.csv` + `decision_table.md`. |
| `denrl/` | Core library: cost signals (`costs.py`), penalty weights (`weights.py`), the shared penalized-reward env wrapper (`env.py`, `env_pendulum.py`, `env_lunar_lander.py`), transition models (`transition.py`), eval metrics (`metrics.py`), and env/method wiring (`registry.py`). |
| `configs/` | One YAML per (method, env) combination — e.g. `baseline_E1.yaml`, `lagr_E1.yaml`, `bnn_E2.yaml`, `baseline_LL_box.yaml` — consumed by `run.py`. |
| `scripts/` | Offline data generation (`generate_offline_data.py`), transition-model training (`train_transition.py`), run-matrix/comparison drivers (`run_matrix*.sh`, `run_comparison_pendulum.sh`, `smoke_test_pendulum.sh`), and plotting/analysis helpers. |
| `data/` | Generated offline datasets (`<env>_offline.parquet`) — not tracked in git, recreate with `generate_offline_data.py`. |
| `runs/` | Output of each training run (config, policy checkpoint, metrics) — not tracked in git. |
| `models/` | Frozen trained transition models (`E2_ensemble.pt`, `E2_bnn.pt`) used by the uncertainty methods. |
| `figures/` | Generated plots referenced by the writeup. |

## Baseline vs Lagrangian comparison (E1, local pass)
```bash
python scripts/generate_offline_data.py --env E1   # if data/E1_offline.parquet doesn't exist yet
bash scripts/run_comparison_pendulum.sh             # -> runs/vis/, 24 full runs (~30 min each)
```
`run_comparison_pendulum.sh` sweeps `baseline_E1` over `p in {2, 10, 30}` and `lagr_E1`
(epsilon=0.01), 6 seeds each -> 24 runs, all full-length (`total_steps: 150000`). 


Before committing to that, sanity-check the pipeline end-to-end in under a minute:
```bash
bash scripts/smoke_test_pendulum.sh    # -> runs/smoke/, 2 runs, --steps 2000 --eval-episodes 5
```
This generates `data/E1_offline.parquet` if missing, then runs one `baseline_E1` config and
the `lagr_E1` config with tiny step/eval-episode counts so `run.py`'s full path (offline fit,
training, eval, `metrics.json`) is exercised without waiting on real training. It won't
produce meaningful `zone_visit_rate`/`return` numbers (2000 steps isn't enough to swing up) —
it only checks that the run completes and metrics.json is well-formed.

## LunarLander comparison (LL)

```bash
python scripts/generate_offline_data.py --env LL
bash scripts/run_matrix_LL.sh
```

`run_matrix_LL.sh` sweeps the LL baseline over `p in {2, 10, 30}` and the LL Lagrangian
variant over `epsilon = 0.01`, using 6 seeds per setting. The default run length is
`total_steps: 300000` and the default evaluation horizon is `eval_episodes: 200` with
`max_steps: 500`.