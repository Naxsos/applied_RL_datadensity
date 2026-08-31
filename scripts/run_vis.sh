#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=runs/vis
SEEDS=(0 1 2 3 4 5)

# Baseline p=2, p=10, p=30
for p in 2 10 30; do
  for s in "${SEEDS[@]}"; do
    echo "=== baseline p=$p seed=$s ==="
    python - configs/baseline_E1.yaml $p $s $OUT <<'PY'
import sys, yaml, tempfile, subprocess, pathlib
cfg_path, p, seed, out = sys.argv[1:]
d = yaml.safe_load(pathlib.Path(cfg_path).read_text())
d["weight"]["p"] = float(p)
tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
yaml.safe_dump(d, tmp, sort_keys=False); tmp.close()
subprocess.run([sys.executable, "run.py", tmp.name, "--seed", seed, "--out", out], check=True)
PY
  done
done

# Lagr epsilon=0.01
for s in "${SEEDS[@]}"; do
  echo "=== lagr epsilon=0.01 seed=$s ==="
  python run.py configs/lagr_E1.yaml --seed $s --out $OUT
done

echo "=== ALL 24 RUNS DONE ==="
