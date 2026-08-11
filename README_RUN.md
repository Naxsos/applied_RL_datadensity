# Running the comparison harness

Implements `EXPERIMENT_SPEC.md`. Two pluggable axes (`CostSignal`, `PenaltyWeight`)
that every method reuses -> comparable table cells.

## Setup
```bash
pip install -r requirements.txt
```

## Week-1 local pass (one seed each, physics fallback — no LSTM needed yet)
```bash
python scripts/generate_offline_data.py --env E1        # -> data/E1_offline.parquet
python run.py configs/baseline_E1.yaml --seed 0         # reproduce Table 1
python run.py configs/lagr_E1.yaml     --seed 0         # learned-α vs fixed-p
python aggregate.py                                     # -> results.csv + decision_table.md
```
`baseline`/`lagr` run immediately: `frozen_checkpoint: null` makes the sim env
use real physics, so you get end-to-end numbers before any model is trained.

## Enabling the uncertainty methods (E2)
```bash
python scripts/generate_offline_data.py --env E2                       # TODO: inject chaotic band
python scripts/train_transition.py --env E2 --kind ensemble --n 5      # -> models/E2_ensemble.pt
python scripts/train_transition.py --env E2 --kind bnn --dropout 0.1   # -> models/E2_bnn.pt
python run.py configs/ens_E2.yaml --seed 0
python run.py configs/bnn_E2.yaml --seed 0
```

## Full matrix (A100 / Vast.ai pass)
```bash
bash scripts/run_matrix.sh    # all methods x seeds x knob sweeps, then aggregate
```

## What each piece maps to
| File | Role | Spec |
|---|---|---|
| `denrl/costs.py` | AXIS 1 — KDE / ensemble-var / BNN-var, all normalized to [0,1] | §2 |
| `denrl/weights.py` | AXIS 2 — FixedWeight(p) / LagrangianWeight(ε) + dual-update callback | §2 |
| `denrl/env.py` | the ONE shared penalized-reward wrapper | §2–3 |
| `denrl/metrics.py` | §5 schema: zone rate, clean return, signal↔error corr, robustness | §5 |
| `run.py` | one (method, env, seed) -> `runs/<id>/metrics.json` | §5,§7 |
| `aggregate.py` | -> `results.csv` + weighted `decision_table.md` | §7 |

## Before trusting the table (spec §9)
1. eval is on the **clean** reward (already wired).
2. **≥5 seeds** per cell — `run_matrix.sh` does this.
3. same frozen transition model for baseline/bnn/lagr; N copies for `ens`.
4. both `cost` and `raw` are logged.
5. TODO markers (E2 env, paper's real LSTM) are the two things to finish past the scaffold.
