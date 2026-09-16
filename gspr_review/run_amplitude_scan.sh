#!/usr/bin/env bash
set -euo pipefail
cd /home/cjm/OpenCOOD-main/cjmnet
export PYTHONPATH=/home/cjm/OpenCOOD-main/cjmnet:/home/cjm/OpenCOOD-main
export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0}"
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
export PYTHONHASHSEED=20260909
SCAN_PYTHON=/home/cjm/miniconda3/envs/opencood/bin/python
"$SCAN_PYTHON" -m unittest gspr_review.test_amplitude -v
"$SCAN_PYTHON" -u -m gspr_review.scan_amplitude "$@"
