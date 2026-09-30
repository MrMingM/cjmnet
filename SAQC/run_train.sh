#!/bin/sh
set -eu

: "${RUN:?set RUN to the completed F/F+D adaptation run directory}"
: "${FRONTEND_CONFIG:?set FRONTEND_CONFIG}"
: "${FRONTEND_CHECKPOINT:?set FRONTEND_CHECKPOINT}"
: "${V3_CHECKPOINT:?set V3_CHECKPOINT}"

V3_CONFIG=${V3_CONFIG:-local_fusion_v3/experiment.yaml}
PY=${PY:-/home/cjm/miniconda3/envs/opencood/bin/python}
OUT=${OUT:-/data/cjm/datasets/logs/saqc_train_$(date +%Y%m%d_%H%M%S)}
EPOCHS=${EPOCHS:-3}
SMOKE=${SMOKE:-0}
WEATHER_DATASET_ROOT=${WEATHER_DATASET_ROOT:-/data/cjm/datasets/opv2v-physics-fixed-v1}

"$PY" -u -m SAQC.train \
  --run "$RUN" \
  --v3-config "$V3_CONFIG" \
  --frontend-config "$FRONTEND_CONFIG" \
  --frontend-checkpoint "$FRONTEND_CHECKPOINT" \
  --v3-checkpoint "$V3_CHECKPOINT" \
  --output "$OUT" \
  --weathers clean,fog,rain,snow \
  --epochs "$EPOCHS" \
  --smoke "$SMOKE" \
  --weather-dataset-root "$WEATHER_DATASET_ROOT"
