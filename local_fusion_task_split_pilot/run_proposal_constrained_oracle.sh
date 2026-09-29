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
ORIGINAL_ORACLE_RUN="${ORIGINAL_ORACLE_RUN:-/data/cjm/datasets/logs/task_source_oracle_v2_20260928_113748}"
OUT="${OUT:-/data/cjm/datasets/logs/proposal_task_oracle_$(date +%Y%m%d_%H%M%S)}"

test -f "$B0_RUN/protocol.json"
test -f "$B0_RUN/decision_results.json"
test -f "$B0_RUN/Shared.pth"
test -f "$ORIGINAL_ORACLE_RUN/protocol.json"
test -f "$ORIGINAL_ORACLE_RUN/death_test_results.json"
test -f "$FRONTEND_CONFIG"
test -f "$FRONTEND_CHECKPOINT"
test -f "$V3_RUN/residual/best.pth"
test ! -e "$OUT"

"$PY" -m unittest local_fusion_task_source_oracle.test_core

"$PY" -u -m local_fusion_task_split_pilot.audit_oracle_actions \
  --run-dir "$ORIGINAL_ORACLE_RUN" \
  --output "$OUT.action_audit.json"

exec "$PY" -u -m local_fusion_task_split_pilot.proposal_constrained_oracle \
  --frontend-config "$FRONTEND_CONFIG" \
  --frontend-checkpoint "$FRONTEND_CHECKPOINT" \
  --v3-checkpoint "$V3_RUN/residual/best.pth" \
  --b0-run "$B0_RUN" \
  --original-oracle-run "$ORIGINAL_ORACLE_RUN" \
  --output "$OUT"
