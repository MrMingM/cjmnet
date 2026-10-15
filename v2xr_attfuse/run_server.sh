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
export PYTHONHASHSEED=20261015
export TMPDIR=${TMPDIR:-/data/cjm/datasets/tmp/v2xr_attfuse}
mkdir -p "$TMPDIR"
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
STAGE=${STAGE:-evaluate}
if [ "$STAGE" = evaluate ]; then
  : "${GPU:?Set GPU to an available physical HIP GPU index}"
  export ROCR_VISIBLE_DEVICES="$GPU"
fi
PY=${PY:-/home/cjm/miniconda3/envs/opencood/bin/python}
set -- "$PY" -u -m v2xr_attfuse.run --stage "$STAGE" --config "${CONFIG:-$TOOL_DIR/config.yaml}"
if [ -n "${OUTPUT_ROOT:-}" ]; then set -- "$@" --output-root "$OUTPUT_ROOT"; fi
if [ -n "${FRAMES_PER_SCENE:-}" ]; then set -- "$@" --frames-per-scene "$FRAMES_PER_SCENE"; fi
if [ "${RESUME:-0}" = 1 ]; then set -- "$@" --resume; fi
exec "$@"
