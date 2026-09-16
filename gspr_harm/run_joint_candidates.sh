#!/usr/bin/env bash
set -euo pipefail
cd /home/cjm/OpenCOOD-main/cjmnet
export PYTHONPATH=/home/cjm/OpenCOOD-main/cjmnet:/home/cjm/OpenCOOD-main
export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0}"
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
export PYTHONHASHSEED=20260909
PYTHON=/home/cjm/miniconda3/envs/opencood/bin/python
FRONTEND_CONFIG="${FRONTEND_CONFIG:-/data/cjm/datasets/logs/gspr_joint_full_v1_seed20260907_20260907_161155/config.yaml}"
FRONTEND_CHECKPOINT="${FRONTEND_CHECKPOINT:-/data/cjm/datasets/logs/gspr_joint_full_v1_seed20260907_20260907_161155/net_best_validation.pth}"
SOURCE_RUN="${SOURCE_RUN:-/data/cjm/datasets/logs/gspr_harm_full_validation_20260912_164658_2847677}"
test -f "$FRONTEND_CONFIG"
test -f "$FRONTEND_CHECKPOINT"
test -f "$SOURCE_RUN/summary.json"
"$PYTHON" -m unittest gspr_harm.test_joint_candidates -v
RUN="/data/cjm/datasets/logs/gspr_harm_joint_candidates_$(date +%Y%m%d_%H%M%S)_$$"
"$PYTHON" -u -m gspr_harm.joint_candidates \
  --source-run "$SOURCE_RUN" \
  --frontend-config "$FRONTEND_CONFIG" \
  --frontend-checkpoint "$FRONTEND_CHECKPOINT" \
  --weather fog rain snow \
  --output-dir "$RUN"
printf '\n查看结果：cat "%s/summary.json"\n' "$RUN"
