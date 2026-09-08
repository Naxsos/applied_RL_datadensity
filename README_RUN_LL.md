# Running LunarLander experiments

This guide is the LL-specific runbook for the continuous LunarLander setup in this repo.

## Setup
```bash
pip install -r requirements.txt
```

## Main configs
- `configs/baseline_LL.yaml` — fixed penalty baseline with `weight.p: 0.0`
- `configs/lagr_LL.yaml` — learned Lagrangian multiplier `alpha`

Both use:
- `env.id: LL`
- `env.continuous: true`
- `env.offline_dataset: data/LL_offline.parquet`
- SAC for training

## 1. Generate the offline dataset
```bash
python scripts/generate_offline_data.py --env LL
```

This writes:
- `data/LL_offline.parquet`

Unlike the Pendulum-style environments, LL offline data generation does **not** stop episodes on excluded-zone entry. That avoids creating an artificial blind spot close to the ground.

## 2. Visualize low-density pockets in the offline data
```bash
python scripts/visualize_ll_low_density.py --data data/LL_offline.parquet --out figures/ll_low_density_projections.png
```

This writes:
- `figures/ll_low_density_projections.png`

Useful options:
```bash
python scripts/visualize_ll_low_density.py \
  --data data/LL_offline.parquet \
  --bandwidth 0.25 \
  --low-percentile 5.0 \
  --sample-points 25000 \
  --out figures/ll_low_density_projections.png
```

## 3. Run the baseline
```bash
python run.py configs/baseline_LL.yaml --seed 0
```

This writes a run directory like:
- `runs/baseline_LL_seed0/`

Key outputs:
- `runs/baseline_LL_seed0/config.yaml`
- `runs/baseline_LL_seed0/metrics.json`
- `runs/baseline_LL_seed0/policy.zip`

## 4. Run the Lagrangian variant
```bash
python run.py configs/lagr_LL.yaml --seed 0
```

This writes:
- `runs/lagr_LL_seed0/`

Additional output for the Lagrangian method:
- `runs/lagr_LL_seed0/alpha.csv`

## 5. Plot a single LL run summary
```bash
python scripts/plot_ll_run.py
```

This reads:
- `runs/baseline_LL_seed0/metrics.json`

and writes:
- `figures/ll_run_metrics.png`

If you want to plot a different run, update the `metrics_path` in `scripts/plot_ll_run.py`.

## 6. Compare LL runs
Compare one baseline run against one Lagrangian run:

```bash
python scripts/compare_ll_runs.py --runs baseline_LL_seed0 lagr_LL_seed0
```

Aggregate all matching seeds per method:

```bash
python scripts/compare_ll_runs.py --runs baseline_LL_* lagr_LL_* --aggregate
```

This writes:
- `figures/ll_run_comparison.png`

The comparison figure includes:
- landing success and strict landing
- timeout and crash rates
- clean return with deviation
- episode length with deviation
- zone visit rate with deviation

## 7. Useful multi-seed runs
Run a small LL comparison:

```bash
python run.py configs/baseline_LL.yaml --seed 0
python run.py configs/baseline_LL.yaml --seed 1
python run.py configs/lagr_LL.yaml --seed 0
python run.py configs/lagr_LL.yaml --seed 1
python scripts/compare_ll_runs.py --runs baseline_LL_* lagr_LL_* --aggregate
```

## Notes
- LL uses the same overall penalty pipeline as the other environments, but with an LL-specific excluded low-density region.
- The current baseline config is intentionally conservative with `p: 0.0`.
- Evaluation uses clean reward with `500` episodes and `500` max steps per episode in the shipped LL configs.
