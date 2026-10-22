#!/bin/sh
set -eu
sh -n "$0"
REPO=/home/cjm/OpenCOOD-main/cjmnet
PYTHON=${PYTHON:-/home/cjm/miniconda3/envs/opencood/bin/python}
MODE=${MODE:-checks}
SMOKE=${SMOKE:-0}
RESUME=${RESUME:-0}
CONFIG=${CONFIG:-$REPO/shared_weather_benchmark_p6/config/protocol.json}
TMPDIR=${TMPDIR:-/data/cjm/datasets/tmp/scorw_p6_v1}
export TMPDIR
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
ROCR_VISIBLE_DEVICES=
export ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
mkdir -p "$TMPDIR" /data/cjm/datasets/logs/scorw-p6-v1
cd "$REPO"
set -- --config "$CONFIG" --mode "$MODE"
case "$SMOKE" in
  1) set -- "$@" --smoke ;;
  0) ;;
  *) echo "SMOKE must be 0 or 1" >&2; exit 2 ;;
esac
case "$RESUME" in
  1) set -- "$@" --resume ;;
  0) ;;
  *) echo "RESUME must be 0 or 1" >&2; exit 2 ;;
esac
exec "$PYTHON" -u -B -m shared_weather_benchmark_p6.run "$@"
