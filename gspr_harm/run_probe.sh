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
test -f "$FRONTEND_CONFIG"
test -f "$FRONTEND_CHECKPOINT"
"$PYTHON" -m unittest gspr_harm.test_core gspr_communication.test_codec gspr_communication.test_tensors -v
RUN="/data/cjm/datasets/logs/gspr_harm_a0b0_$(date +%Y%m%d_%H%M%S)_$$"
"$PYTHON" -u -m gspr_harm.scan \
  --config gspr_review/experiment.yaml \
  --frontend-config "$FRONTEND_CONFIG" \
  --frontend-checkpoint "$FRONTEND_CHECKPOINT" \
  --frames 8 --calibration-frames 2 --scenes 0 1 --stride 5 \
  --weather fog rain snow --region-blocks 1 --random-repeats 3 \
  --output-dir "$RUN"
printf '\n诊断结束，查看结果：\ncat "%s/summary.json"\n' "$RUN"
