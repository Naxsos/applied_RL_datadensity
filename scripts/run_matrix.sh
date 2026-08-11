#!/usr/bin/env bash
# Full run matrix (§6). Batch this on the A100/Vast.ai pass; keep the Week-1
# local pass to a single seed of each method on E1.
set -euo pipefail
cd "$(dirname "$0")/.."

SEEDS=(0 1 2 3 4)

sweep () {  # sweep <config> <knob-yaml-key> <val1> <val2> ...
  local cfg=$1 key=$2; shift 2
  for v in "$@"; do
    for s in "${SEEDS[@]}"; do
      # override the knob via a tmp config; simplest is a small python patch
      python - "$cfg" "$key" "$v" "$s" <<'PY'
import sys, yaml, tempfile, subprocess, pathlib
cfg, key, val, seed = sys.argv[1:]
d = yaml.safe_load(pathlib.Path(cfg).read_text())
d["weight"][key] = float(val)
d["run_id"] = f"{d['method']}_{d['env']['id']}_{key}{val}_seed{seed}"
tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
yaml.safe_dump(d, tmp); tmp.close()
subprocess.run([sys.executable, "run.py", tmp.name, "--seed", seed], check=True)
PY
    done
  done
}

sweep configs/baseline_E1.yaml p       2 10 30
sweep configs/ens_E2.yaml      p       0.2 0.6 1.0
sweep configs/bnn_E2.yaml      p       0.2 0.6 1.0
sweep configs/lagr_E1.yaml     epsilon 0.02 0.05
sweep configs/baseline_E3.yaml p       2 10 30
sweep configs/lagr_E3.yaml     epsilon 0.02 0.05

python aggregate.py --runs runs --out .
