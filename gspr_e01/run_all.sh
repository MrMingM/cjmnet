#!/bin/sh
set -eu
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PYTHON=${PYTHON:-/home/cjm/miniconda3/envs/opencood/bin/python}
RUN_MODE=${RUN_MODE:-serial}
MODE=${MODE:-all}
RESUME=${RESUME:-0}
OUTPUT_ROOT=${OUTPUT_ROOT:-/data/cjm/datasets/logs/gspr-e01-v1}
ORIGINAL_ROOT=${ORIGINAL_ROOT:-/data/cjm/datasets/logs/v2xr-weather-l012-v1}
TRAIN_ROOT=${TRAIN_ROOT:-/data/cjm/datasets/v2xr-official-train-v2}
export PYTHONHASHSEED=20261016
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR=${TMPDIR:-/data/cjm/datasets/tmp/gspr_e01}
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES
mkdir -p "$TMPDIR"
case "$RUN_MODE" in
    serial) GPU=${GPU:-2}; GPU_E0=$GPU; GPU_E1=$GPU ;;
    parallel) : "${GPU_E0:?Set GPU_E0}"; : "${GPU_E1:?Set GPU_E1}" ;;
    *) echo 'RUN_MODE must be serial or parallel' >&2; exit 2 ;;
esac
"$PYTHON" -B "$HERE/check_static.py"
set -- "$PYTHON" -B "$HERE/orchestrator.py" --run-mode "$RUN_MODE" --mode "$MODE" \
    --gpu-e0 "$GPU_E0" --gpu-e1 "$GPU_E1" --resume "$RESUME" \
    --output-root "$OUTPUT_ROOT" --original-root "$ORIGINAL_ROOT" --train-root "$TRAIN_ROOT"
if [ -n "${E0_EXISTING_ROOT:-}" ]; then
    set -- "$@" --e0-existing-root "$E0_EXISTING_ROOT"
fi
exec "$@"
