#!/bin/sh
set -eu
cd "$(dirname "$0")"
: "${ROCR_VISIBLE_DEVICES:?Set one idle physical HCU index}"
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES A1_SMOKE A1_MODE
export ROCR_VISIBLE_DEVICES PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec "${PYTHON:-/home/cjm/miniconda3/envs/opencood/bin/python}" -u -B driver.py
