#!/bin/sh
set -eu

: "${RUN:?set RUN to the completed F/F+D adaptation run directory}"
: "${FRONTEND_CONFIG:?set FRONTEND_CONFIG}"
: "${FRONTEND_CHECKPOINT:?set FRONTEND_CHECKPOINT}"
: "${V3_CHECKPOINT:?set V3_CHECKPOINT}"
: "${SAQC_CHECKPOINT:?set SAQC_CHECKPOINT to saqc_quality.pth}"

V3_CONFIG=${V3_CONFIG:-local_fusion_v3/experiment.yaml}
OUT=${OUT:-/data/cjm/datasets/logs/saqc_calibration_$(date +%Y%m%d_%H%M%S).json}
WEATHER=${WEATHER:-clean}
SMOKE=${SMOKE:-0}

python -m SAQC.calibrate \
  --run "$RUN" \
  --v3-config "$V3_CONFIG" \
  --frontend-config "$FRONTEND_CONFIG" \
  --frontend-checkpoint "$FRONTEND_CHECKPOINT" \
  --v3-checkpoint "$V3_CHECKPOINT" \
  --checkpoint "$SAQC_CHECKPOINT" \
  --output "$OUT" \
  --weather "$WEATHER" \
  --smoke "$SMOKE"
