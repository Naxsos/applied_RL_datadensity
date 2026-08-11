#!/usr/bin/env bash
# Intermediate results for the professor meeting — reduced but HONEST scale.
# Matches "machen Sie kleine Experimente ... ein Gefühl für Rechenzeit und Güte".
#
# Defaults are tuned to finish on a MacBook overnight. Override via env vars:
#   STEPS=150000 EVAL=2000 SEEDS="0 1 2 3 4" bash scripts/run_intermediate.sh
#
# NOTE: STEPS < 150000 is NOT directly comparable to the paper — state the
# reduced scale on the slide. Bump to 150000 for the paper-faithful pass.
set -euo pipefail
cd "$(dirname "$0")/.."

STEPS=${STEPS:-50000}
EVAL=${EVAL:-500}
SEEDS=${SEEDS:-"0 1 2"}

echo ">> intermediate run: STEPS=$STEPS EVAL=$EVAL SEEDS=[$SEEDS]"

# make sure data exists (idempotent)
[ -f data/E1_offline.parquet ] || python3 scripts/generate_offline_data.py --env E1
[ -f data/E2_offline.parquet ] || python3 scripts/generate_offline_data.py --env E2
[ -f models/E2_ensemble.pt ]   || python3 scripts/train_transition.py --env E2 --kind ensemble --n 5
[ -f models/E2_bnn.pt ]        || python3 scripts/train_transition.py --env E2 --kind bnn --dropout 0.1

sweep () {  # sweep <config> <weight-key> <val...>
  local cfg=$1 key=$2; shift 2
  for v in "$@"; do
    for s in $SEEDS; do
      python3 - "$cfg" "$key" "$v" "$s" "$STEPS" "$EVAL" <<'PY'
import sys, yaml, tempfile, subprocess, pathlib
cfg, key, val, seed, steps, ev = sys.argv[1:]
d = yaml.safe_load(pathlib.Path(cfg).read_text())
d["weight"][key] = float(val)
d["run_id"] = f"{d['method']}_{d['env']['id']}_{key}{val}_seed{seed}"
tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
yaml.safe_dump(d, tmp); tmp.close()
subprocess.run([sys.executable, "run.py", tmp.name, "--seed", seed,
                "--steps", steps, "--eval-episodes", ev], check=True)
PY
    done
  done
}

sweep configs/baseline_E1.yaml p       2 10 30     # reproduces Table 1
sweep configs/lagr_E1.yaml     epsilon 0.02 0.05
sweep configs/ens_E2.yaml      p       0.6
sweep configs/bnn_E2.yaml      p       0.6

python3 aggregate.py
python3 plot_results.py
echo ">> done. See decision_table.md and figures/*.png"
