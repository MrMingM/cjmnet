#!/bin/sh
set -eu

: "${RUN:?set RUN to the completed F/F+D adaptation run directory}"
: "${FRONTEND_CONFIG:?set FRONTEND_CONFIG}"
: "${FRONTEND_CHECKPOINT:?set FRONTEND_CHECKPOINT}"
: "${V3_CHECKPOINT:?set V3_CHECKPOINT}"
: "${SAQC_CHECKPOINT:?set SAQC_CHECKPOINT to saqc_quality.pth}"

V3_CONFIG=${V3_CONFIG:-local_fusion_v3/experiment.yaml}
PY=${PY:-/home/cjm/miniconda3/envs/opencood/bin/python}
PHASE=${PHASE:-development}
OUT=${OUT:-/data/cjm/datasets/logs/saqc_${PHASE}_$(date +%Y%m%d_%H%M%S)}
SMOKE=${SMOKE:-0}
CALIBRATION=${CALIBRATION:-}
WEATHER_DATASET_ROOT=${WEATHER_DATASET_ROOT:-/data/cjm/datasets/opv2v-physics-fixed-v1}

set -- "$PY" -u -m SAQC.evaluate \
  --run "$RUN" \
  --v3-config "$V3_CONFIG" \
  --frontend-config "$FRONTEND_CONFIG" \
  --frontend-checkpoint "$FRONTEND_CHECKPOINT" \
  --v3-checkpoint "$V3_CHECKPOINT" \
  --checkpoint "$SAQC_CHECKPOINT" \
  --output "$OUT" \
  --phase "$PHASE" \
  --smoke "$SMOKE" \
  --weather-dataset-root "$WEATHER_DATASET_ROOT"

if [ -n "$CALIBRATION" ]; then
  set -- "$@" --calibration "$CALIBRATION"
fi

"$@"
