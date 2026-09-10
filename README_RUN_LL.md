# Running LunarLander experiments

This guide is the LL-specific runbook for the continuous LunarLander setup in this repo.

The LL setup uses two related but distinct ideas:
- the **offline dataset** defines which observed states are dense or sparse for the KDE cost signal;
- the **excluded zone** is a hand-specified near-ground risk region used for evaluation and, for the Lagrangian variant, the zone-rate constraint.

That means the LL zone is **not** automatically inferred from the KDE. It is an LL-specific safety prior motivated by risky, weakly covered states near the ground, while the density penalty itself is still learned from the frozen offline dataset.

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
The resulting parquet file therefore remains the reference distribution for KDE density estimation, instead of baking the manual LL risk zone into data collection itself.

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
- `runs/baseline_LL_p2_seed0/`

Key outputs:
- `runs/baseline_LL_p2_seed0/config.yaml`
- `runs/baseline_LL_p2_seed0/metrics.json`
- `runs/baseline_LL_p2_seed0/policy.zip`

## 4. Run the Lagrangian variant
```bash
python run.py configs/lagr_LL.yaml --seed 0
```

This writes:
- `runs/lagr_LL_epsilon0.05_seed0/`

Additional output for the Lagrangian method:
- `runs/lagr_LL_epsilon0.05_seed0/alpha.csv`

## 5. Plot a single LL run summary
```bash
python scripts/plot_ll_run.py
```

This reads:
- `runs/baseline_LL_p2_seed0/metrics.json`

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

For matched-run statistical testing, use:

```bash
python scripts/analyze_ll_stats.py --runs baseline_LL_* lagr_LL_*
```

This writes:
- `figures/ll_stats_analysis.json`
- `figures/ll_stats_analysis.md`

The stats script also reports whether the selected LL runs satisfy the repository experiment spec (matched config, clean-reward eval, and sufficient seeds).

To quantify whether the manual LL zone matches the KDE low-density pocket, use:

```bash
python scripts/validate_ll_zone.py
```

This writes:
- `figures/ll_zone_validation.json`
- `figures/ll_zone_validation.md`

The report includes overlap, precision/recall, and threshold suggestions derived from low-density states in the offline dataset.

## 7. LL run matrix
Run the default LL matrix:

```bash
./scripts/run_matrix_LL.sh
```

This launches a resumable sweep over:
- `baseline_LL` with `p in {0.0, 2.0, 10.0, 30.0}`
- `lagr_LL` with `epsilon in {0.02, 0.05, 0.1}`
- seeds `0 1 2`

Each run is isolated in its own process and the matrix uses a resumable, parallel launcher with a live progress indicator while forcing `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, and `OPENBLAS_NUM_THREADS=1` inside each job. That usually improves total wall-clock time on multi-core machines by avoiding thread oversubscription while still letting you see how far the sweep has progressed.

Useful overrides:
```bash
SEEDS="0 1" PARALLEL=2 ./scripts/run_matrix_LL.sh
STEPS=300000 EVAL_EPISODES=300 SUFFIX=pilot ./scripts/run_matrix_LL.sh
PY=.venv/bin/python PARALLEL=3 ./scripts/run_matrix_LL.sh
```

Run directories are named like:
- `runs/baseline_LL_p0.0_seed0_matrix/`
- `runs/lagr_LL_epsilon0.05_seed2_matrix/`

The script skips any run that already has `metrics.json`, so re-running it resumes incomplete sweeps instead of repeating finished jobs.

Aggregate LL-only results with:

```bash
python scripts/aggregate_ll.py --runs runs --out .
```

This writes:
- `ll_results.csv`
- `ll_decision_table.csv`
- `ll_decision_table.md`

## 8. Useful multi-seed runs
Run a small LL comparison manually:

```bash
python run.py configs/baseline_LL.yaml --seed 0
python run.py configs/baseline_LL.yaml --seed 1
python run.py configs/lagr_LL.yaml --seed 0
python run.py configs/lagr_LL.yaml --seed 1
python scripts/compare_ll_runs.py --runs baseline_LL_* lagr_LL_* --aggregate
```

## Notes
- LL uses the same overall penalty pipeline as the other environments, but with an LL-specific near-ground risk zone.
- The current default LL thresholds are data-driven from the offline KDE validation pass: `altitude_max=0.21`, `speed_min=0.72`, `descent_speed_min=0.32`, `tilt_min=0.47`, `angular_speed_min=0.25`.
- KDE density is still fit from the frozen offline dataset; the LL risk zone is now a simple thresholded approximation to the low-density pocket rather than a purely manual contour.
- The current baseline config is intentionally conservative with `p: 0.0`.
- Evaluation uses clean reward with `500` episodes and `500` max steps per episode in the shipped LL configs.
