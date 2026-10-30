#!/bin/sh
set -eu
MODE=${MODE:-full}
RESUME=${RESUME:-0}
case "$MODE" in preflight|smoke|full) ;; *) echo "MODE must be preflight, smoke or full" >&2; exit 2 ;; esac
case "$RESUME" in 0|1) ;; *) echo "RESUME must be 0 or 1" >&2; exit 2 ;; esac
PROJECT=/home/cjm/OpenCOOD-main/cjmnet
PYTHON=/home/cjm/miniconda3/envs/opencood/bin/python
cd "$PROJECT"
export TMPDIR=/data/cjm/datasets/tmp/scorw_diagnostics
mkdir -p "$TMPDIR"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export ROCR_VISIBLE_DEVICES=
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
if [ "$RESUME" = 1 ]; then
    exec "$PYTHON" -u -B -m scorw_diagnostics.run --mode "$MODE" --resume
fi
exec "$PYTHON" -u -B -m scorw_diagnostics.run --mode "$MODE"
