#!/usr/bin/env bash
# Run only after uploading the updated communication package and checking HCU 0.
set -euo pipefail
cd /home/cjm/OpenCOOD-main/cjmnet
export PYTHONPATH=/home/cjm/OpenCOOD-main/cjmnet:/home/cjm/OpenCOOD-main
export ROCR_VISIBLE_DEVICES=0
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
export PYTHONHASHSEED=20260908
COMM_PYTHON=/home/cjm/miniconda3/envs/opencood/bin/python
PREVIOUS_RUN=/data/cjm/datasets/logs/gspr_comm_a1b1_seed20260908_20260908_160747
FRONTEND_CONFIG="$PREVIOUS_RUN/frontend_config.yaml"
FRONTEND_CHECKPOINT=/data/cjm/datasets/logs/gspr_joint_full_v1_seed20260907_20260907_161155/net_best_validation.pth
COMM_RUN="/data/cjm/datasets/logs/gspr_comm_residual_seed20260908_$(date +%Y%m%d_%H%M%S)"
CONFIG="${COMM_RUN}.yaml"
test -f "$FRONTEND_CONFIG"
test -f "$FRONTEND_CHECKPOINT"
# Copy the previous experiment protocol; change only the new model settings.
"$COMM_PYTHON" - "$PREVIOUS_RUN/experiment.yaml" "$CONFIG" <<'PY'
import sys
import yaml
with open(sys.argv[1]) as f:
    config = yaml.safe_load(f)
config['communication'].update(variant='residual', correction_hidden=16, max_adjustment=0.1)
with open(sys.argv[2], 'x') as f:
    yaml.safe_dump(config, f, sort_keys=False)
PY
echo "Output: $COMM_RUN"
"$COMM_PYTHON" -m unittest gspr_communication.test_codec gspr_communication.test_tensors -v
"$COMM_PYTHON" -m gspr_communication.verify --variant residual --config "$CONFIG" \
  --frontend-config "$FRONTEND_CONFIG" --frontend-checkpoint "$FRONTEND_CHECKPOINT"
"$COMM_PYTHON" -u -m gspr_communication.train --config "$CONFIG" \
  --frontend-config "$FRONTEND_CONFIG" --frontend-checkpoint "$FRONTEND_CHECKPOINT" \
  --run-dir "$COMM_RUN" 2>&1 | tee "${COMM_RUN}_train.log"
"$COMM_PYTHON" -u -m gspr_communication.evaluate --config "$COMM_RUN/experiment.yaml" \
  --frontend-config "$COMM_RUN/frontend_config.yaml" --frontend-checkpoint "$FRONTEND_CHECKPOINT" \
  --selectors "$COMM_RUN/selectors_best.pth" --compare-ego \
  --output-dir "${COMM_RUN}_validation" 2>&1 | tee "${COMM_RUN}_evaluation.log"
"$COMM_PYTHON" - "${COMM_RUN}_validation" <<'PY'
import json
import sys
from pathlib import Path
import yaml
p = Path(sys.argv[1])
ap = yaml.safe_load((p/'eval.yaml').read_text())
print('Results:', p)
print({k: ap[k] for k in ('ap30', 'ap_50', 'ap_70')})
print((p/'communication.json').read_text())
rows = [json.loads(s) for s in (p/'frames.jsonl').read_text().splitlines()]
for key in ('replacement_fraction', 'mean_abs_score_adjustment'):
    print(key, sum(r[key] for r in rows)/len(rows))
PY
