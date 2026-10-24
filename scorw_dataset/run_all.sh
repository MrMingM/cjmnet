#!/bin/sh
set -eu
sh -n "$0"
REPO=/home/cjm/OpenCOOD-main/cjmnet
PYTHON=${PYTHON:-/home/cjm/miniconda3/envs/opencood/bin/python}
MODE=${MODE:-preflight}
RESUME=${RESUME:-0}
WORKERS=${WORKERS:-4}
GPU=${GPU:-0}
case "$MODE" in
    preflight|PREFLIGHT) MODE=preflight ;;
    smoke|SMOKE) MODE=smoke ;;
    full|FULL) MODE=full ;;
    *) echo "MODE must be preflight, smoke or full" >&2; exit 2 ;;
esac
case "$RESUME" in 0|1) ;; *) echo "RESUME must be 0 or 1" >&2; exit 2 ;; esac
case "$WORKERS" in ''|*[!0-9]*|0) echo "WORKERS must be positive" >&2; exit 2 ;; esac
case "$GPU" in ''|*[!0-9]*) echo "GPU must be nonnegative" >&2; exit 2 ;; esac
TMPDIR=${TMPDIR:-/data/cjm/datasets/tmp/scorw_fog}
export TMPDIR
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export ROCR_VISIBLE_DEVICES=
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
mkdir -p "$TMPDIR" /data/cjm/datasets/logs/scorw-fog-v1-coveragefix
cd "$REPO"
exec "$PYTHON" -u -B -m scorw_dataset.run --mode "$MODE" --resume "$RESUME" --workers "$WORKERS" --gpu "$GPU"
