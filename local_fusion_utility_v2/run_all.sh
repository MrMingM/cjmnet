#!/usr/bin/env bash
set -euo pipefail
cd /home/cjm/OpenCOOD-main/cjmnet
export PYTHONPATH=/home/cjm/OpenCOOD-main/cjmnet:/home/cjm/OpenCOOD-main
export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0}"
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
export PYTHONHASHSEED=20260913
PY="${PY:-/home/cjm/miniconda3/envs/opencood/bin/python}"
FRONTEND_ROOT="${FRONTEND_ROOT:-/data/cjm/datasets/logs/gspr_joint_full_v1_seed20260907_20260907_161155}"
FRONTEND_CONFIG="${FRONTEND_CONFIG:-$FRONTEND_ROOT/config.yaml}"
FRONTEND_CHECKPOINT="${FRONTEND_CHECKPOINT:-$FRONTEND_ROOT/net_best_validation.pth}"
: "${RUN:?Set RUN to a new directory under /data/cjm/datasets/logs}"
test -f "$FRONTEND_CONFIG"
test -f "$FRONTEND_CHECKPOINT"
exec "$PY" -u -m local_fusion_utility_v2.pipeline \
  --config "${CONFIG:-local_fusion_utility_v2/experiment.yaml}" \
  --frontend-config "$FRONTEND_CONFIG" --frontend-checkpoint "$FRONTEND_CHECKPOINT" \
  --run "$RUN"
