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
RUN="${RUN:-/data/cjm/datasets/logs/task_split_pilot_20260927_120637}"
OUT="${OUT:-$RUN/task_gap_audit}"

test -f "$RUN/protocol.json"
test -f "$RUN/decision_results.json"
test -f "$RUN/Shared.pth"
test -f "$RUN/Split.pth"
test -f "$FRONTEND_CONFIG"
test -f "$FRONTEND_CHECKPOINT"
test -f "$V3_RUN/residual/best.pth"
test ! -e "$OUT"

"$PY" -u -m local_fusion_task_split_pilot.test_task_gap_audit

exec "$PY" -u -m local_fusion_task_split_pilot.audit_task_gap \
  --run "$RUN" \
  --v3-config local_fusion_v3/experiment.yaml \
  --frontend-config "$FRONTEND_CONFIG" \
  --frontend-checkpoint "$FRONTEND_CHECKPOINT" \
  --v3-checkpoint "$V3_RUN/residual/best.pth" \
  --output "$OUT"
