#!/bin/sh
set -eu

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
B0_RUN="${B0_RUN:-/data/cjm/datasets/logs/task_split_pilot_20260927_120637}"
RUN="${RUN:-/data/cjm/datasets/logs/task_split_v2_$(date +%Y%m%d_%H%M%S)}"
export RUN

test -f "$FRONTEND_CONFIG"
test -f "$FRONTEND_CHECKPOINT"
test -f "$V3_RUN/residual/best.pth"
test -f "$B0_RUN/protocol.json"
test -f "$B0_RUN/decision_results.json"
test -f "$B0_RUN/Split.pth"
test -f local_fusion_task_split_v2/experiment.yaml

"$PY" -u -m local_fusion_task_split_v2.test_core

exec "$PY" -u -m local_fusion_task_split_v2.pipeline \
  --config local_fusion_task_split_v2/experiment.yaml \
  --v3-config local_fusion_v3/experiment.yaml \
  --frontend-config "$FRONTEND_CONFIG" \
  --frontend-checkpoint "$FRONTEND_CHECKPOINT" \
  --v3-checkpoint "$V3_RUN/residual/best.pth" \
  --b0-run "$B0_RUN" \
  --run "$RUN"
