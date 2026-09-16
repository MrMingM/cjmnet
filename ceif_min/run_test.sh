#!/usr/bin/env bash
set -euo pipefail
cd /home/cjm/OpenCOOD-main/cjmnet
export PYTHONPATH=/home/cjm/OpenCOOD-main/cjmnet:/home/cjm/OpenCOOD-main
export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0}"
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
export PYTHONHASHSEED=20260916
PY=/home/cjm/miniconda3/envs/opencood/bin/python
: "${RUN:?Set RUN to the completed training directory}"
OUT="${OUT:-${RUN}_opv2vw_test_$(date +%Y%m%d_%H%M%S)}"
"$PY" -m ceif_min.evaluate --run "$RUN" --phase benchmark --output-dir "$OUT"
