#!/usr/bin/env bash
# Rerun of the E1/E3 matrix under the ONE-SIDED excluded zone (denrl/env.py::Zone).
#
# Why a rerun: until now in_zone() mirrored the zone onto |theta|, so it blocked
# both routes from the resting state to upright. Zone avoidance and swing-up were
# mutually exclusive, which is what made every method score 0.5-0.9 zone_visit_rate
# (baseline p=30 got its "safe" 0.49 by only reaching upright in 6% of episodes).
# With the paper's one-sided zone the detour exists again, so zone_visit_rate
# measures avoidance instead of failure-to-solve.
#
# E2 is NOT rerun: its offline dataset and transition-model checkpoints were built
# under the symmetric zone, and its claim (density vs uncertainty) does not hinge on
# zone geometry. Its configs pin zone_symmetric: true so the runs stay consistent.
#
# Skips any combo that already has runs/<run_id>/metrics.json, so it is resumable.
set -euo pipefail
cd "$(dirname "$0")/.."

# Overridable for a quick triage pass, e.g.
#   STEPS=50000 SEEDS="0 1" EVAL_EPISODES=200 SUFFIX=v3fast ./scripts/run_matrix_zonefix.sh
read -r -a SEEDS <<< "${SEEDS:-0 1 2}"
STEPS=${STEPS:-150000}
EVAL_EPISODES=${EVAL_EPISODES:-500}
PARALLEL=${PARALLEL:-4}
SUFFIX=${SUFFIX:-v3}
PY=${PY:-.venv/bin/python}

echo "seeds=${SEEDS[*]} steps=$STEPS eval_episodes=$EVAL_EPISODES suffix=$SUFFIX"

JOBFILE=$(mktemp)
trap 'rm -f "$JOBFILE"' EXIT

gen_sweep () {  # gen_sweep <config> <knob-yaml-key> <val1> <val2> ...
  local cfg=$1 key=$2; shift 2
  for v in "$@"; do
    for s in "${SEEDS[@]}"; do
      "$PY" - "$cfg" "$key" "$v" "$s" "$STEPS" "$EVAL_EPISODES" "$SUFFIX" <<'PY'
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
  gen_sweep configs/lagr_E1.yaml     epsilon 0.02 0.05
  gen_sweep configs/baseline_E3.yaml p       2 10 30
  gen_sweep configs/lagr_E3.yaml     epsilon 0.02 0.05
} > "$JOBFILE"

n=$(wc -l < "$JOBFILE" | tr -d ' ')
echo "$n jobs remaining, running $PARALLEL at a time..."
xargs -P "$PARALLEL" -I{} bash -c '{}' < "$JOBFILE"

if [ "${AGGREGATE:-1}" = "1" ]; then
  "$PY" aggregate.py --runs runs --out .
fi
