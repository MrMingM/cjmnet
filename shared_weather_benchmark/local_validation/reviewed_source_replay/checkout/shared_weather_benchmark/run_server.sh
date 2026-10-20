#!/bin/sh
set -eu
sh -n "$0"
REPO=/home/cjm/OpenCOOD-main/cjmnet
PYTHON=${PYTHON:-/home/cjm/miniconda3/envs/opencood/bin/python}
MODE=${MODE:-checks}
SMOKE=${SMOKE:-1}
RESUME=${RESUME:-0}
FROM_STAGE=${FROM_STAGE:-P0}
CONFIG=${CONFIG:-$REPO/shared_weather_benchmark/config/prototype.json}
TMPDIR=${TMPDIR:-/data/cjm/datasets/tmp/shared_weather_benchmark}
export TMPDIR
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
# No phase imports Torch or uses GPU. Hide accelerator devices explicitly.
ROCR_VISIBLE_DEVICES=
export ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
mkdir -p "$TMPDIR" /data/cjm/datasets/logs/shared-weather-benchmark-v1
cd "$REPO"
case "$SMOKE" in
  1) DEFAULT_OUTPUT=/data/cjm/datasets/shared-weather-benchmark-v1-smoke ;;
  0) DEFAULT_OUTPUT=/data/cjm/datasets/shared-weather-benchmark-v1 ;;
  *) echo "SMOKE must be 0 or 1" >&2; exit 2 ;;
esac
OUTPUT_ROOT=${OUTPUT_ROOT:-$DEFAULT_OUTPUT}
case "$MODE" in
  checks) THROUGH=P2 ;;
  sample) THROUGH=P5 ;;
  phase) THROUGH=${THROUGH:-P3} ;;
  *) echo "MODE must be checks, sample or phase" >&2; exit 2 ;;
esac
set -- --config "$CONFIG" --output "$OUTPUT_ROOT" --through "$THROUGH" --from-stage "$FROM_STAGE"
case "$SMOKE" in 1) set -- "$@" --smoke ;; esac
case "$RESUME" in
  1) set -- "$@" --resume ;;
  0) ;;
  *) echo "RESUME must be 0 or 1" >&2; exit 2 ;;
esac
exec "$PYTHON" -u -B -m shared_weather_benchmark.run "$@"
