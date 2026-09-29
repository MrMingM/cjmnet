#!/bin/sh
set -eu

REPO="${REPO:-/home/cjm/OpenCOOD-main/cjmnet}"
cd "$REPO"
export PYTHONPATH="$REPO:/home/cjm/OpenCOOD-main${PYTHONPATH:+:$PYTHONPATH}"
export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0}"
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
export PYTHONHASHSEED=20260929

PY="${PY:-/home/cjm/miniconda3/envs/opencood/bin/python}"
RUN="${RUN:-/data/cjm/datasets/logs/fusion_detector_adaptation_20260927_103227}"
EVAL_ROOT="${EVAL_ROOT:-$RUN/top256_score_geometry_20260928_155556/extraction}"
OUT="${OUT:-$RUN/ciassd_iou_eval_$(date +%Y%m%d_%H%M%S)}"
CHECKPOINT="${CHECKPOINT:-}"
FRONTEND_ROOT="${FRONTEND_ROOT:-/data/cjm/datasets/logs/gspr_joint_full_v1_seed20260907_20260907_161155}"
FRONTEND_CONFIG="${FRONTEND_CONFIG:-$FRONTEND_ROOT/config.yaml}"
FRONTEND_CHECKPOINT="${FRONTEND_CHECKPOINT:-$FRONTEND_ROOT/net_best_validation.pth}"
V3_RUN="${V3_RUN:-/data/cjm/datasets/logs/local_fusion_v3_20260922_182832}"

if [ ! -x "$PY" ] || [ ! -f "$EVAL_ROOT/candidate_audit.json" ] || [ ! -f "$CHECKPOINT" ]; then
  printf 'Missing Python, development cache, or CHECKPOINT\n' >&2
  exit 1
fi
if [ -e "$OUT" ]; then
  printf 'Output already exists: %s\n' "$OUT" >&2
  exit 1
fi
if [ -z "${DIRECT_IOU_CHECKPOINT:-}" ]; then
  found=0
  for candidate in "$RUN"/candidate_s0_s6_*/checkpoints/iou_regression_seed20260929.pt; do
    if [ -f "$candidate" ]; then
      found=$((found + 1))
      DIRECT_IOU_CHECKPOINT="$candidate"
    fi
  done
  if [ "$found" -ne 1 ]; then
    printf 'Set DIRECT_IOU_CHECKPOINT to the existing S0 direct-IoU checkpoint (found %s).\n' "$found" >&2
    exit 1
  fi
fi
if [ ! -f "$DIRECT_IOU_CHECKPOINT" ]; then
  printf 'Missing S0 direct-IoU checkpoint: %s\n' "$DIRECT_IOU_CHECKPOINT" >&2
  exit 1
fi
"$PY" -c 'import torch; import local_fusion_detector_adaptation.sota_reproduction.ciassd_iou.evaluate'
if [ "${CHECK_ONLY:-0}" = 1 ]; then
  printf 'Preflight passed.\n'
  exit 0
fi

"$PY" -u -m local_fusion_detector_adaptation.sota_reproduction.ciassd_iou.evaluate \
  --method-disabled --eval-root "$EVAL_ROOT" --output "$OUT/regression"

PATCH_ROOT="${PATCH_ROOT:-$OUT/validation_patches}"
if [ ! -f "$PATCH_ROOT/manifest.json" ]; then
  if [ -e "$PATCH_ROOT" ]; then
    printf 'Incomplete validation patch cache: %s\n' "$PATCH_ROOT" >&2
    exit 1
  fi
  "$PY" -u -m local_fusion_detector_adaptation.sota_reproduction.ciassd_iou.extract \
    --split validation --candidate-root "$EVAL_ROOT" --run "$RUN" \
    --v3-config "${V3_CONFIG:-local_fusion_v3/experiment.yaml}" \
    --frontend-config "$FRONTEND_CONFIG" \
    --frontend-checkpoint "$FRONTEND_CHECKPOINT" \
    --v3-checkpoint "$V3_RUN/residual/best.pth" \
    --output "$PATCH_ROOT"
fi

set -- "$PY" -u -m local_fusion_detector_adaptation.sota_reproduction.ciassd_iou.evaluate \
  --eval-root "$EVAL_ROOT" --patch-root "$PATCH_ROOT" \
  --checkpoint "$CHECKPOINT" --direct-iou-checkpoint "$DIRECT_IOU_CHECKPOINT" \
  --output "$OUT/results"
exec "$@"
