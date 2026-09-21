#!/usr/bin/env bash
set -euo pipefail
cd /home/cjm/OpenCOOD-main/cjmnet
export PYTHONPATH=/home/cjm/OpenCOOD-main/cjmnet:/home/cjm/OpenCOOD-main
export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0}"
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
export PYTHONHASHSEED=20260913
PY=/home/cjm/miniconda3/envs/opencood/bin/python
: "${FRONTEND_CONFIG:?Set FRONTEND_CONFIG to the frozen detector yaml}"
: "${FRONTEND_CHECKPOINT:?Set FRONTEND_CHECKPOINT to the frozen GSPR/AttFuse checkpoint}"
: "${RUN:?Set RUN to the completed scheme-1 directory}"
OUT="${OUT:-${RUN}_opv2vw_test_$(date +%Y%m%d_%H%M%S)}"
"$PY" -m local_fusion_utility.evaluate --frontend-config "$FRONTEND_CONFIG" \
  --frontend-checkpoint "$FRONTEND_CHECKPOINT" --utility-checkpoint "$RUN/utility/best.pth" \
  --loss-gain-checkpoint "$RUN/loss_gain/best.pth" --calibration "$RUN/calibration.json" \
  --phase benchmark --output-dir "$OUT"

