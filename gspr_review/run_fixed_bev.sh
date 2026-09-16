#!/usr/bin/env bash
set -euo pipefail
cd /home/cjm/OpenCOOD-main/cjmnet
export PYTHONPATH=/home/cjm/OpenCOOD-main/cjmnet:/home/cjm/OpenCOOD-main
export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0}"
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
export PYTHONHASHSEED=20260909
FIXED_PYTHON=/home/cjm/miniconda3/envs/opencood/bin/python
"$FIXED_PYTHON" -m unittest gspr_review.test_bev_location gspr_review.test_fixed_bev gspr_review.test_packet_replay -v
"$FIXED_PYTHON" -u -m gspr_review.fixed_bev_experiment "$@"
