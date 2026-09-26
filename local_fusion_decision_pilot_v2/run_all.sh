#!/usr/bin/env bash
set -euo pipefail
cd /home/cjm/OpenCOOD-main/cjmnet
export PYTHONPATH=/home/cjm/OpenCOOD-main/cjmnet:/home/cjm/OpenCOOD-main
export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0}"
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
export PYTHONHASHSEED=20260926
PY="${PY:-/home/cjm/miniconda3/envs/opencood/bin/python}"
FRONTEND_ROOT="${FRONTEND_ROOT:-/data/cjm/datasets/logs/gspr_joint_full_v1_seed20260907_20260907_161155}"
FRONTEND_CONFIG="${FRONTEND_CONFIG:-$FRONTEND_ROOT/config.yaml}"
FRONTEND_CHECKPOINT="${FRONTEND_CHECKPOINT:-$FRONTEND_ROOT/net_best_validation.pth}"
: "${V3_RUN:?Set V3_RUN to the completed v3 run directory}"
: "${RUN:?Set RUN to a new follow-up output directory}"
test -f "$V3_RUN/residual/best.pth"
test -f "$FRONTEND_CONFIG"
test -f "$FRONTEND_CHECKPOINT"
"$PY" -m unittest local_fusion_decision_pilot_v2.test_core -v
args=(
  --pilot-config "${PILOT_CONFIG:-local_fusion_decision_pilot_v2/experiment.yaml}"
  --v3-config "${V3_CONFIG:-$V3_RUN/resolved_experiment.yaml}"
  --v3-checkpoint "$V3_RUN/residual/best.pth"
  --frontend-config "$FRONTEND_CONFIG"
  --frontend-checkpoint "$FRONTEND_CHECKPOINT"
  --run "$RUN"
)
if [[ "${RESUME:-0}" == "1" ]]; then args+=(--resume); fi
exec "$PY" -u -m local_fusion_decision_pilot_v2.pipeline "${args[@]}"
