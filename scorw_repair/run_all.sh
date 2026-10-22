#!/bin/sh
set -eu
sh -n "$0"
REPO=/home/cjm/OpenCOOD-main/cjmnet
PYTHON=${PYTHON:-/home/cjm/miniconda3/envs/opencood/bin/python}
MODE=${MODE:-preflight}
DOWNLOAD=${DOWNLOAD:-0}
RESUME=${RESUME:-0}
case "$MODE" in
    preflight|PREFLIGHT) MODE=preflight ;;
    smoke|SMOKE) MODE=smoke ;;
    full|FULL) MODE=full ;;
    *) echo "MODE must be preflight, smoke or full" >&2; exit 2 ;;
esac
case "$RESUME" in 0|1) ;; *) echo "RESUME must be 0 or 1" >&2; exit 2 ;; esac
if [ "$DOWNLOAD" != 0 ]; then
    echo "DOWNLOAD is disabled; set DOWNLOAD=0. No URL files are read." >&2
    exit 2
fi
TMPDIR=${TMPDIR:-/data/cjm/datasets/tmp/scorw_repair_v2_budgetfix}
export TMPDIR
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
mkdir -p "$TMPDIR" /data/cjm/datasets/logs/scorw-repair-v2-budgetfix
cd "$REPO"
exec "$PYTHON" -u -B -m scorw_repair.run --mode "$MODE" --download 0 --resume "$RESUME"
