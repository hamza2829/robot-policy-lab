#!/usr/bin/env bash
set -euo pipefail
mkdir -p artifacts
DATASET=$(robot-policy-lab generate-data --output data --episodes 128)
robot-policy-lab validate-data "$DATASET"
cp "$DATASET/manifest.json" artifacts/manifest.json
robot-policy-lab train "$DATASET" --output artifacts/one_step.pt --chunk-size 1 --epochs 50
robot-policy-lab train "$DATASET" --output artifacts/chunked.pt --chunk-size 4 --epochs 50
robot-policy-lab evaluate --one-step artifacts/one_step.pt --chunked artifacts/chunked.pt \
  --episodes 128 --start-seed 10000 --output artifacts/evaluation.json
python -m pip freeze > artifacts/environment.txt
python - <<'PY'
import json
from pathlib import Path

results = json.loads(Path("artifacts/evaluation.json").read_text())
assert results["policies"]["expert"]["success_rate"] >= 0.95, "expert regression"
for name in ("one_step", "chunked"):
    assert results["checkpoints"][name]["final_loss"] < results["checkpoints"][name]["initial_loss"]
PY
