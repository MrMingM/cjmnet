#!/bin/sh
set -eu
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
PYTHON=${PYTHON:-/home/cjm/miniconda3/envs/opencood/bin/python}
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-/data/cjm/datasets/logs/gspr-repe-rocm-20261025-v4}
PROJECT_ROOT=${PROJECT_ROOT:-/home/cjm/OpenCOOD-main/cjmnet}
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset PYTHONPATH PYTHONHOME
exec "$PYTHON" -B "$SCRIPT_DIR/diagnose.py" --root "$EXPERIMENT_ROOT" --project "$PROJECT_ROOT"
