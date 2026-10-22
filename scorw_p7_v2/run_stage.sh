#!/bin/sh
set -eu
sh -n "$0"
REPO=/home/cjm/OpenCOOD-main/cjmnet
PYTHON=${PYTHON:-/home/cjm/miniconda3/envs/opencood/bin/python}
CONFIG=${CONFIG:-$REPO/scorw_p7_v2/config/protocol.json}
MODE=${MODE:-checks}
RESUME=${RESUME:-0}
case "$MODE" in checks|smoke|full) ;; *) echo "MODE must be checks, smoke or full" >&2; exit 2 ;; esac
case "$RESUME" in 0|1) ;; *) echo "RESUME must be 0 or 1" >&2; exit 2 ;; esac
TMPDIR=${TMPDIR:-/data/cjm/datasets/tmp/scorw_p7_v2}
export TMPDIR
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
ROCR_VISIBLE_DEVICES=
export ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
mkdir -p "$TMPDIR" /data/cjm/datasets/logs/scorw-p7-v2
cd "$REPO"
exec "$PYTHON" -u -B -m scorw_p7_v2.run --config "$CONFIG" --mode "$MODE" --resume "$RESUME"
