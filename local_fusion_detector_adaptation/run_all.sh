#!/usr/bin/env bash
set -euo pipefail
cd /home/cjm/OpenCOOD-main/cjmnet
export PYTHONPATH=/home/cjm/OpenCOOD-main/cjmnet:/home/cjm/OpenCOOD-main
export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0}"
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
export PYTHONHASHSEED=20260927
PY="${PY:-/home/cjm/miniconda3/envs/opencood/bin/python}"
FRONTEND_ROOT="${FRONTEND_ROOT:-/data/cjm/datasets/logs/gspr_joint_full_v1_seed20260907_20260907_161155}"
FRONTEND_CONFIG="${FRONTEND_CONFIG:-$FRONTEND_ROOT/config.yaml}"
FRONTEND_CHECKPOINT="${FRONTEND_CHECKPOINT:-$FRONTEND_ROOT/net_best_validation.pth}"
V3_RUN="${V3_RUN:-/data/cjm/datasets/logs/local_fusion_v3_20260922_182832}"
: "${RUN:?Set RUN to a new directory under /data/cjm/datasets/logs}"
test -f "$FRONTEND_CONFIG"
test -f "$FRONTEND_CHECKPOINT"
test -f "$V3_RUN/residual/best.pth"
"$PY" -m unittest local_fusion_detector_adaptation.test_core -v
exec "$PY" -u -m local_fusion_detector_adaptation.pipeline \
  --pilot-config "${PILOT_CONFIG:-local_fusion_detector_adaptation/experiment.yaml}" \
  --v3-config "${V3_CONFIG:-local_fusion_v3/experiment.yaml}" \
  --frontend-config "$FRONTEND_CONFIG" --frontend-checkpoint "$FRONTEND_CHECKPOINT" \
  --v3-checkpoint "$V3_RUN/residual/best.pth" --run "$RUN"
