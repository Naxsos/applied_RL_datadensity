#!/usr/bin/env bash
# Fast end-to-end sanity check for the E1 pipeline: generates the offline
# dataset (if missing) and runs a couple of tiny-step configs through
# run.py, mirroring run_comparison_pendulum.sh's structure but with
# --steps/--eval-episodes cut way down so it finishes in ~1 minute instead
# of ~12 hours. Use this before kicking off the real sweep on a clean branch.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=runs/smoke
STEPS=2000
EVAL_EPISODES=5

if [ ! -f data/E1_offline.parquet ]; then
  echo "=== generating E1 offline data ==="
  python scripts/generate_offline_data.py --env E1
fi

echo "=== smoke: baseline p=10 seed=0 ==="
python - configs/baseline_E1.yaml 10 0 "$OUT" "$STEPS" "$EVAL_EPISODES" <<'PY'
import sys, yaml, tempfile, subprocess, pathlib
cfg_path, p, seed, out, steps, eval_episodes = sys.argv[1:]
d = yaml.safe_load(pathlib.Path(cfg_path).read_text())
d["weight"]["p"] = float(p)
tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
yaml.safe_dump(d, tmp, sort_keys=False); tmp.close()
subprocess.run([sys.executable, "run.py", tmp.name, "--seed", seed, "--out", out,
                 "--steps", steps, "--eval-episodes", eval_episodes], check=True)
PY

echo "=== smoke: lagr epsilon=0.01 seed=0 ==="
python run.py configs/lagr_E1.yaml --seed 0 --out "$OUT" --steps "$STEPS" --eval-episodes "$EVAL_EPISODES"

echo "=== SMOKE TEST OK -> $OUT ==="
