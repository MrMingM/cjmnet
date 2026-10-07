#!/bin/sh
set -eu

cd /home/cjm/OpenCOOD-main/cjmnet
export PYTHONPATH=/home/cjm/OpenCOOD-main/cjmnet:/home/cjm/OpenCOOD-main
export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-2}"
unset HIP_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES
export PYTHONHASHSEED=20261007

PY="${PY:-/home/cjm/miniconda3/envs/opencood/bin/python}"
FRONTEND_ROOT="${FRONTEND_ROOT:-/data/cjm/datasets/logs/gspr_joint_full_v1_seed20260907_20260907_161155}"
FRONTEND_CONFIG="${FRONTEND_CONFIG:-$FRONTEND_ROOT/config.yaml}"
FRONTEND_CHECKPOINT="${FRONTEND_CHECKPOINT:-$FRONTEND_ROOT/net_best_validation.pth}"
V3_RUN="${V3_RUN:-/data/cjm/datasets/logs/local_fusion_v3_20260922_182832}"
V3_CHECKPOINT="${V3_CHECKPOINT:-$V3_RUN/residual/best.pth}"
B0_RUN="${B0_RUN:-/data/cjm/datasets/logs/task_split_pilot_20260927_120637}"
CONFIG="${CONFIG:-local_fusion_action_utility_audit/experiment.yaml}"
V3_CONFIG="${V3_CONFIG:-local_fusion_v3/experiment.yaml}"
RUN="${RUN:-/data/cjm/datasets/logs/action_utility_audit_$(date +%Y%m%d_%H%M%S)}"
export RUN

# Python validates the resolved RUN, all paths and checkpoints, records failures,
# and serializes concurrent drivers. Direct invocation also writes driver.log.
exec "$PY" -u -m local_fusion_action_utility_audit.driver \
  --run "$RUN" --config "$CONFIG" --v3-config "$V3_CONFIG" \
  --frontend-config "$FRONTEND_CONFIG" --frontend-checkpoint "$FRONTEND_CHECKPOINT" \
  --v3-checkpoint "$V3_CHECKPOINT" --b0-run "$B0_RUN" "$@"
