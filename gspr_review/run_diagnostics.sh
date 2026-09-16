#!/usr/bin/env bash
# Read-only evaluation: no training and no modification of existing checkpoints.
set -euo pipefail
cd /home/cjm/OpenCOOD-main/cjmnet
export PYTHONPATH=/home/cjm/OpenCOOD-main/cjmnet:/home/cjm/OpenCOOD-main
export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0}"
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
export PYTHONHASHSEED=20260909
REVIEW_PYTHON=/home/cjm/miniconda3/envs/opencood/bin/python
"$REVIEW_PYTHON" -m unittest gspr_review.test_diagnostics -v
"$REVIEW_PYTHON" -u -m gspr_review.diagnose "$@"
