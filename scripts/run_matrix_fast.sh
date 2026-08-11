#!/usr/bin/env bash
# REDUCED-SCOPE Stage A pass for a same-day decision signal.
#
# NOT the spec's full-fidelity matrix (that's run_matrix.sh: 5 seeds, 150k
# steps, 2000-episode eval). This trades statistical rigor for wall-clock:
# 2 seeds, 50k steps (1/3 of full), 200 eval episodes, and 4 runs at once
# across cores. Treat results as a rough triage signal, not the final
# decision-table numbers -- rerun run_matrix.sh at full scale before
# actually picking a winner to double down on.
#
# run_id gets a "_fast" suffix so these don't collide with full-fidelity runs
# of the same config/seed/knob later.
set -euo pipefail
cd "$(dirname "$0")/.."

SEEDS=(0 1)
STEPS=50000
EVAL_EPISODES=200
PARALLEL=4

JOBFILE=$(mktemp)
trap 'rm -f "$JOBFILE"' EXIT

gen_sweep () {  # gen_sweep <config> <knob-yaml-key> <val1> <val2> ...
  local cfg=$1 key=$2; shift 2
  for v in "$@"; do
    for s in "${SEEDS[@]}"; do
      python - "$cfg" "$key" "$v" "$s" "$STEPS" "$EVAL_EPISODES" <<'PY'
import sys, yaml, tempfile, pathlib
cfg, key, val, seed, steps, eval_ep = sys.argv[1:]
d = yaml.safe_load(pathlib.Path(cfg).read_text())
d["weight"][key] = float(val)
d["run_id"] = f"{d['method']}_{d['env']['id']}_{key}{val}_seed{seed}_fast"
tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, dir="/tmp")
yaml.safe_dump(d, tmp); tmp.close()
print(f"OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 {sys.executable} run.py {tmp.name} "
      f"--seed {seed} --steps {steps} --eval-episodes {eval_ep}")
PY
    done
  done
}

{
  gen_sweep configs/baseline_E1.yaml p       2 10 30
  gen_sweep configs/ens_E2.yaml      p       0.2 0.6 1.0
  gen_sweep configs/bnn_E2.yaml      p       0.2 0.6 1.0
  gen_sweep configs/lagr_E1.yaml     epsilon 0.02 0.05
  gen_sweep configs/baseline_E3.yaml p       2 10 30
  gen_sweep configs/lagr_E3.yaml     epsilon 0.02 0.05
} > "$JOBFILE"

n=$(wc -l < "$JOBFILE" | tr -d ' ')
echo "$n jobs queued, running $PARALLEL at a time..."
xargs -P "$PARALLEL" -I{} bash -c '{}' < "$JOBFILE"

python aggregate.py --runs runs --out .
