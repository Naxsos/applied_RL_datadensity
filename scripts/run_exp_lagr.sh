#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=runs/exp_lagr

# # Baseline p=10, 3 seeds
# for s in 0 1 2; do
#   echo "=== baseline p=10 seed=$s ==="
#   python run.py configs/baseline_E1.yaml --seed $s --out $OUT
# done

# Lagr epsilon=0.01, 3 seeds
for s in 0 1 2; do
  echo "=== lagr epsilon=0.01 seed=$s ==="
  python - configs/lagr_E1.yaml 0.01 $s $OUT <<'PY'
import sys, yaml, tempfile, subprocess, pathlib
cfg_path, eps, seed, out = sys.argv[1:]
d = yaml.safe_load(pathlib.Path(cfg_path).read_text())
d["weight"]["epsilon"] = float(eps)
tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
yaml.safe_dump(d, tmp, sort_keys=False); tmp.close()
subprocess.run([sys.executable, "run.py", tmp.name, "--seed", seed, "--out", out], check=True)
PY
done

# Lagr epsilon=0.005, 3 seeds
for s in 0 1 2; do
  echo "=== lagr epsilon=0.005 seed=$s ==="
  python - configs/lagr_E1.yaml 0.005 $s $OUT <<'PY'
import sys, yaml, tempfile, subprocess, pathlib
cfg_path, eps, seed, out = sys.argv[1:]
d = yaml.safe_load(pathlib.Path(cfg_path).read_text())
d["weight"]["epsilon"] = float(eps)
tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
yaml.safe_dump(d, tmp, sort_keys=False); tmp.close()
subprocess.run([sys.executable, "run.py", tmp.name, "--seed", seed, "--out", out], check=True)
PY
done

echo "=== ALL 9 RUNS DONE ==="
