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
CALIBRATION_RUN="${CALIBRATION_RUN:-/data/cjm/datasets/logs/gspr_harm_a0b0_20260912_155547_2669287}"
test -f "$FRONTEND_CONFIG"
test -f "$FRONTEND_CHECKPOINT"
test -f "$CALIBRATION_RUN/summary.json"
"$PYTHON" -m unittest gspr_harm.test_core gspr_harm.test_validation gspr_harm.test_analyze -v
RUN="/data/cjm/datasets/logs/gspr_harm_full_validation_$(date +%Y%m%d_%H%M%S)_$$"
printf '完整验证输出目录：%s\n' "$RUN"
"$PYTHON" -u -m gspr_harm.scan \
  --config gspr_review/experiment.yaml \
  --frontend-config "$FRONTEND_CONFIG" \
  --frontend-checkpoint "$FRONTEND_CHECKPOINT" \
  --full-validation --calibration-run "$CALIBRATION_RUN" \
  --weather fog rain snow --region-blocks 1 --random-repeats 3 \
  --output-dir "$RUN"
"$PYTHON" -m gspr_harm.analyze --run "$RUN"
printf '\n汇总结果：%s/summary.json\n分场景结果：各天气子目录/scenes.json\n' "$RUN"
