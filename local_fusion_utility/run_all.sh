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
: "${RUN:?Set RUN to a new path under /data/cjm/datasets/logs}"

"$PY" -m unittest local_fusion_utility.test_core -v
"$PY" -m local_fusion_utility.prepare --frontend-config "$FRONTEND_CONFIG" \
  --frontend-checkpoint "$FRONTEND_CHECKPOINT" --split train --output-dir "$RUN/train_cache"
"$PY" -m local_fusion_utility.prepare --frontend-config "$FRONTEND_CONFIG" \
  --frontend-checkpoint "$FRONTEND_CHECKPOINT" --split validation --output-dir "$RUN/validation_cache"
"$PY" -m local_fusion_utility.train --train-cache "$RUN/train_cache" \
  --validation-cache "$RUN/validation_cache" --variant utility --output-dir "$RUN/utility"
"$PY" -m local_fusion_utility.train --train-cache "$RUN/train_cache" \
  --validation-cache "$RUN/validation_cache" --variant loss_gain --output-dir "$RUN/loss_gain"
"$PY" -m local_fusion_utility.calibrate --validation-cache "$RUN/validation_cache" \
  --utility-checkpoint "$RUN/utility/best.pth" --loss-gain-checkpoint "$RUN/loss_gain/best.pth" \
  --output-file "$RUN/calibration.json"
"$PY" -m local_fusion_utility.evaluate --frontend-config "$FRONTEND_CONFIG" \
  --frontend-checkpoint "$FRONTEND_CHECKPOINT" --utility-checkpoint "$RUN/utility/best.pth" \
  --loss-gain-checkpoint "$RUN/loss_gain/best.pth" --calibration "$RUN/calibration.json" \
  --phase development --scale-ablation --no-keep-ablation --output-dir "${RUN}_development"

