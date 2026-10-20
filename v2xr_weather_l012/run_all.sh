#!/bin/sh
set -eu
sh -n "$0"
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
GPU=${GPU:-0}
MODE=${MODE:-all}
RESUME=${RESUME:-0}
OUTPUT_ROOT=${OUTPUT_ROOT:-/data/cjm/datasets/logs/v2xr-weather-l012-v1}
TRAIN_ROOT=${TRAIN_ROOT:-/data/cjm/datasets/v2xr-official-train-v2/train}
PYTHON=${PYTHON:-/home/cjm/miniconda3/envs/opencood/bin/python}
case "$MODE" in all|preflight|self-test|smoke) ;; *) echo "Invalid MODE: $MODE" >&2; exit 2 ;; esac
case "$RESUME" in 0|1) ;; *) echo 'RESUME must be 0 or 1' >&2; exit 2 ;; esac
case "$GPU" in ''|*[!0-9]*) echo 'GPU must be one physical GPU index' >&2; exit 2 ;; esac
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
ROCR_VISIBLE_DEVICES=$GPU
PYTHONHASHSEED=20261016
TMPDIR=${TMPDIR:-/data/cjm/datasets/tmp/v2xr_weather_l012}
PYTHONDONTWRITEBYTECODE=1
PYTHONUNBUFFERED=1
export ROCR_VISIBLE_DEVICES PYTHONHASHSEED TMPDIR PYTHONDONTWRITEBYTECODE PYTHONUNBUFFERED
export OUTPUT_ROOT TRAIN_ROOT
mkdir -p "$TMPDIR"
"$PYTHON" -B "$SCRIPT_DIR/check_static.py"
set -- "$PYTHON" -B "$SCRIPT_DIR/orchestrator.py" --mode "$MODE" --output-root "$OUTPUT_ROOT" --train-root "$TRAIN_ROOT"
if [ "$RESUME" = 1 ]; then set -- "$@" --resume; fi
exec "$@"
