# Running LunarLander experiments

This guide is the LL-specific runbook for the continuous LunarLander setup in this repo.

The LL setup uses one idea:
- the **offline dataset** defines which observed states are dense or sparse for the KDE cost signal. In-zone transitions are filtered out, creating a "hole" in the observed distribution that the KDE learns to penalize.
- the **excluded zone** (a central box-shaped region in position space) is used both during data collection (filtering) and at evaluation (constraint).

This keeps valid landing trajectories around the box while still creating a clear low-density pocket in the data distribution, giving the agent a reason to avoid it.

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
- `data/LL_legacy_offline.parquet`

Like the Pendulum-style environments, LL offline data generation now terminates on excluded-zone entry. This creates a "hole" in the state distribution where the KDE low-density cost signal is learned to penalize the zone, giving the agent a reason to route around it.

## 2. Visualize the offline dataset with the excluded zone
```bash
python scripts/visualize_ll_offline_with_zone.py
```

This writes:
- `figures/ll_offline_with_zone.png`

Shows the dataset distribution with the excluded zone box overlaid. The zone box should contain 0% (or very low %) of the data, confirming that in-zone transitions were filtered out and created a natural "hole" in the observed state distribution that KDE learns to penalize as low-density.

Optional: Visualize low-density pockets detected by KDE:
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
- `runs/baseline_LL_p0.0_seed0/`
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
- LL uses the same zone-aware data collection and penalty pipeline as the other environments.
- The default LL zone is a compact avoidable box in position space (`x in [-0.5, 0.5]`, `y in [0.65, 0.85]`).
- Data generation filters out in-zone transitions, creating a natural low-density "hole" in the observed states that KDE learns to penalize while leaving valid trajectories around the box.
- The current baseline config is intentionally conservative with `p: 0.0`.
- Evaluation uses clean reward with `500` episodes and `500` max steps per episode in the shipped LL configs.
