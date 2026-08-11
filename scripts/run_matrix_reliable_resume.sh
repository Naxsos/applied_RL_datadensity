#!/usr/bin/env bash
# Resume run_matrix_reliable.sh after it was killed partway through --
# skips any (config, knob, seed) combo that already has runs/<run_id>_v2/metrics.json.
set -euo pipefail
cd "$(dirname "$0")/.."

SEEDS=(0 1)
STEPS=150000
EVAL_EPISODES=500
PARALLEL=4
SUFFIX="v2"

JOBFILE=$(mktemp)
trap 'rm -f "$JOBFILE"' EXIT

gen_sweep () {  # gen_sweep <config> <knob-yaml-key> <val1> <val2> ...
  local cfg=$1 key=$2; shift 2
  for v in "$@"; do
    for s in "${SEEDS[@]}"; do
      python - "$cfg" "$key" "$v" "$s" "$STEPS" "$EVAL_EPISODES" "$SUFFIX" <<'PY'
import sys, yaml, tempfile, pathlib
cfg, key, val, seed, steps, eval_ep, suffix = sys.argv[1:]
d = yaml.safe_load(pathlib.Path(cfg).read_text())
d["weight"][key] = float(val)
run_id = f"{d['method']}_{d['env']['id']}_{key}{val}_seed{seed}_{suffix}"
if (pathlib.Path("runs") / run_id / "metrics.json").exists():
    sys.exit(0)  # already done, skip
d["run_id"] = run_id
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
echo "$n jobs remaining, running $PARALLEL at a time..."
xargs -P "$PARALLEL" -I{} bash -c '{}' < "$JOBFILE"

python aggregate.py --runs runs --out .
