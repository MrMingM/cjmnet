#!/bin/sh
set -eu
TOOL_DIR=$(CDPATH= cd "$(dirname "$0")" && pwd)
REPO=$(CDPATH= cd "$TOOL_DIR/.." && pwd)
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export LC_ALL=C
export TMPDIR=${TMPDIR:-/data/cjm/datasets/tmp/v2xr_subset}
mkdir -p "$TMPDIR"
PY=${PY:-/home/cjm/miniconda3/envs/opencood/bin/python}
set -- "$PY" -u -m v2xr_subset.prepare --stage "${STAGE:-plan}"
if [ -n "${SOURCE_ROOT:-}" ]; then set -- "$@" --source "$SOURCE_ROOT"; fi
if [ -n "${OUTPUT_ROOT:-}" ]; then set -- "$@" --output "$OUTPUT_ROOT"; fi
if [ -n "${SPLIT:-}" ]; then set -- "$@" --split "$SPLIT"; fi
if [ -n "${FRAMES_PER_SCENE:-}" ]; then set -- "$@" --frames-per-scene "$FRAMES_PER_SCENE"; fi
if [ -n "${FOG_VARIANT:-}" ]; then set -- "$@" --fog-variant "$FOG_VARIANT"; fi
if [ -n "${SNOW_VARIANT:-}" ]; then set -- "$@" --snow-variant "$SNOW_VARIANT"; fi
if [ "${RESUME:-0}" = 1 ]; then set -- "$@" --resume; fi
exec "$@"
