#!/usr/bin/env bash
set -euo pipefail
cd /home/cjm/OpenCOOD-main/cjmnet
export PYTHONPATH=/home/cjm/OpenCOOD-main/cjmnet:/home/cjm/OpenCOOD-main
export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0}"
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
export PYTHONHASHSEED=20260913
PY=/home/cjm/miniconda3/envs/opencood/bin/python
FRONTEND_CONFIG="${FRONTEND_CONFIG:-/data/cjm/datasets/logs/gspr_joint_full_v1_seed20260907_20260907_161155/config.yaml}"
FRONTEND_CHECKPOINT="${FRONTEND_CHECKPOINT:-/data/cjm/datasets/logs/gspr_joint_full_v1_seed20260907_20260907_161155/net_best_validation.pth}"
CONFIG="${CONFIG:-gspr_evidence/experiment.yaml}"
STAGE="${STAGE:-both}"
case "$STAGE" in filters|single|both) ;; *) echo 'STAGE must be filters, single or both'; exit 1 ;; esac
OUT="/data/cjm/datasets/logs/gspr_harm_validation_${STAGE}_$(date +%Y%m%d_%H%M%S)"
mkdir "$OUT"
exec > >(tee -a "$OUT/console.log") 2>&1
echo "DEVELOPMENT VALIDATION ONLY: $OUT"
"$PY" -m unittest gspr_evidence.test_harm_diagnostic -v
for weather in clean fog rain snow; do
  "$PY" -m gspr_evidence.harm_diagnostic --config "$CONFIG" \
    --frontend-config "$FRONTEND_CONFIG" --frontend-checkpoint "$FRONTEND_CHECKPOINT" \
    --weather "$weather" --stage "$STAGE" \
    --output-dir "$OUT/$weather"
done
echo "Send back: $OUT/{clean,fog,rain,snow}/diagnosis.md"
