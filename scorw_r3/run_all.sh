#!/bin/sh
set -eu
sh -n "$0"
REPO=/home/cjm/OpenCOOD-main/cjmnet
PYTHON=${PYTHON:-/home/cjm/miniconda3/envs/opencood/bin/python}
MODE=${MODE:-preflight}
RESUME=${RESUME:-0}
case "$MODE" in
    preflight|PREFLIGHT) MODE=preflight ;;
    smoke|SMOKE) MODE=smoke ;;
    full|FULL) MODE=full ;;
    *) echo "MODE must be preflight, smoke or full" >&2; exit 2 ;;
esac
case "$RESUME" in 0|1) ;; *) echo "RESUME must be 0 or 1" >&2; exit 2 ;; esac
TMPDIR=${TMPDIR:-/data/cjm/datasets/tmp/scorw_r3}
export TMPDIR
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
mkdir -p "$TMPDIR" /data/cjm/datasets/logs/scorw-r3-v1-argsfix
cd "$REPO"
exec "$PYTHON" -u -B -m scorw_r3.run --mode "$MODE" --resume "$RESUME"
