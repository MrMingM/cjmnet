#!/bin/sh
set -eu

: "${FRONTEND_CONFIG:?set FRONTEND_CONFIG to the frozen F frontend YAML}"

TOOL_DIR=$(CDPATH= cd "$(dirname "$0")" && pwd)
REPO=$(CDPATH= cd "$TOOL_DIR/../.." && pwd)
if [ ! -f "$TOOL_DIR/materialize_physics_weather.py" ]; then
  printf 'Missing generator: %s\nCopy materialize_physics_weather.py to the server before running.\n' \
    "$TOOL_DIR/materialize_physics_weather.py" >&2
  exit 1
fi
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

PY=${PY:-/home/cjm/miniconda3/envs/opencood/bin/python}
EXPERIMENT_CONFIG=${EXPERIMENT_CONFIG:-local_fusion_v3/experiment.yaml}
WEATHER_DATASET_ROOT=${WEATHER_DATASET_ROOT:-/data/cjm/datasets/opv2v-physics-fixed-v1}

"$PY" -m unittest opencood.tools.test_materialize_physics_weather

set -- "$PY" -u -m opencood.tools.materialize_physics_weather \
  --experiment-config "$EXPERIMENT_CONFIG" \
  --frontend-config "$FRONTEND_CONFIG" \
  --output "$WEATHER_DATASET_ROOT"

if [ "${RESUME:-0}" = 1 ]; then
  set -- "$@" --resume
fi

"$@"
