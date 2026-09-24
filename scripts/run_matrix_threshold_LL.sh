#!/usr/bin/env bash
# Sweep the KDE threshold for the LL box configs.
#
# Example:
#   THRESHOLDS="0.025 0.05 0.1 0.2 0.3 0.4" SEEDS="0 1" PARALLEL=2 ./scripts/run_matrix_threshold_LL.sh
#
# This keeps the env geometry fixed and only varies the KDE threshold to see how
# the cost signal changes while the box remains the same artificial task.
set -euo pipefail
cd "$(dirname "$0")/.."

read -r -a SEEDS <<< "${SEEDS:-0}"
read -r -a THRESHOLDS <<< "${THRESHOLDS:-0.001 0.002 0.005 0.01}"

STEPS=${STEPS:-300000}
EVAL_EPISODES=${EVAL_EPISODES:-200}
PARALLEL=${PARALLEL:-4}
SUFFIX=${SUFFIX:-threshold_sweep3}
PY=${PY:-python}

JOBFILE=$(mktemp)
trap 'rm -f "$JOBFILE"' EXIT

sweep() {  # sweep <config> <val1> <val2> ...
  local cfg=$1; shift
  for threshold in "$@"; do
    for s in "${SEEDS[@]}"; do
      "$PY" - "$cfg" "$threshold" "$s" "$STEPS" "$EVAL_EPISODES" "$SUFFIX" <<'PY'
import pathlib
import sys
import tempfile

import yaml

cfg, val, seed, steps, eval_ep, suffix = sys.argv[1:]
d = yaml.safe_load(pathlib.Path(cfg).read_text())
d["cost_signal"]["threshold"] = float(val)
run_id = f"{d['method']}_{d['env']['id']}_t{val}_seed{seed}_{suffix}"
if (pathlib.Path("runs") / run_id / "metrics.json").exists():
    sys.exit(0)
d["run_id"] = run_id
tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, dir="/tmp")
yaml.safe_dump(d, tmp, sort_keys=False)
tmp.close()
print(
    "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
    f"{sys.executable} run.py {tmp.name} --seed {seed} --steps {steps} --eval-episodes {eval_ep}"
)
PY
    done
  done
}

{
  #sweep configs/baseline_LL_box.yaml "${THRESHOLDS[@]}"
  sweep configs/lagr_LL_box.yaml "${THRESHOLDS[@]}"
} > "$JOBFILE"

declare -a jobs=()
while IFS= read -r job; do
  [[ -n "$job" ]] && jobs+=("$job")
done < "$JOBFILE"

n=${#jobs[@]}
if (( n == 0 )); then
  echo "No threshold-sweep jobs to run."
  exit 0
fi

completed=0
failed=0
next_idx=0
declare -a running_pids=()

print_progress() {
  local pct=0
  local bar=''
  if (( n > 0 )); then
    pct=$(( completed * 100 / n ))
  fi
  local filled=$(( completed * 20 / n ))
  local i
  for ((i=0; i<filled; i++)); do bar+='#'; done
  for ((i=filled; i<20; i++)); do bar+='-'; done
  printf '\r[%3d%%] %d/%d done | %d running | [%s] ' "$pct" "$completed" "$n" "${#running_pids[@]}" "$bar"
}

start_job() {
  local cmd=$1
  bash -c "$cmd" &
  running_pids+=("$!")
}

check_finished() {
  local i pid status
  for i in "${!running_pids[@]}"; do
    pid=${running_pids[$i]}
    if ! kill -0 "$pid" 2>/dev/null; then
      status=0
      if wait "$pid"; then
        status=0
      else
        status=$?
      fi
      unset 'running_pids[$i]'
      ((completed++))
      if (( status != 0 )); then
        failed=1
      fi
      return 0
    fi
  done
  return 1
}

echo "seeds=${SEEDS[*]} thresholds=${THRESHOLDS[*]} steps=$STEPS eval_episodes=$EVAL_EPISODES suffix=$SUFFIX"
echo "$n jobs remaining, running $PARALLEL at a time..."

while (( next_idx < n || ${#running_pids[@]} > 0 )); do
  while (( next_idx < n && ${#running_pids[@]} < PARALLEL )); do
    start_job "${jobs[next_idx]}"
    ((next_idx++))
  done

  if (( ${#running_pids[@]} == 0 )); then
    break
  fi

  if check_finished; then
    print_progress
  else
    print_progress
    sleep 1
  fi
done

printf '\n'
if (( failed != 0 )); then
  echo "Threshold sweep finished with one or more failed jobs." >&2
  exit 1
fi

echo "Threshold sweep complete: $completed/$n jobs finished successfully."
