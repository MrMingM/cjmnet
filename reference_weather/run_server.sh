#!/bin/sh
set -eu

TOOL_DIR=$(CDPATH= cd "$(dirname "$0")" && pwd)
REPO=$(CDPATH= cd "$TOOL_DIR/.." && pwd)
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONHASHSEED=20261013
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES

PY=${PY:-/home/cjm/miniconda3/envs/opencood/bin/python}
STAGE=${STAGE:-all}
if [ "$STAGE" = all ] || [ "$STAGE" = evaluate ]; then
  : "${GPU:?Set GPU to an available physical HIP GPU index, e.g. GPU=0}"
  export ROCR_VISIBLE_DEVICES="$GPU"
fi

set -- "$PY" -u -m reference_weather.run --stage "$STAGE" \
  --config "${CONFIG:-$TOOL_DIR/config.yaml}" --limit-frames "${LIMIT_FRAMES:-0}"
if [ -n "${OUTPUT_ROOT:-}" ]; then set -- "$@" --output-root "$OUTPUT_ROOT"; fi
if [ -n "${ASSETS_ROOT:-}" ]; then set -- "$@" --assets-root "$ASSETS_ROOT"; fi
if [ -n "${ATT_CHECKPOINT:-}" ]; then set -- "$@" --checkpoint "$ATT_CHECKPOINT"; fi
if [ -n "${ATT_CONFIG:-}" ]; then set -- "$@" --model-config "$ATT_CONFIG"; fi
if [ -n "${WORKERS:-}" ]; then set -- "$@" --workers "$WORKERS"; fi
if [ "${RESUME:-0}" = 1 ]; then set -- "$@" --resume; fi
exec "$@"
