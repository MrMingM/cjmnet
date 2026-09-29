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
TRAIN_ROOT="${TRAIN_ROOT:-$RUN/candidate_ranker_train_20260929}"
OUT="${OUT:-$RUN/ciassd_iou_$(date +%Y%m%d_%H%M%S)}"
FRONTEND_ROOT="${FRONTEND_ROOT:-/data/cjm/datasets/logs/gspr_joint_full_v1_seed20260907_20260907_161155}"
FRONTEND_CONFIG="${FRONTEND_CONFIG:-$FRONTEND_ROOT/config.yaml}"
FRONTEND_CHECKPOINT="${FRONTEND_CHECKPOINT:-$FRONTEND_ROOT/net_best_validation.pth}"
V3_RUN="${V3_RUN:-/data/cjm/datasets/logs/local_fusion_v3_20260922_182832}"

if [ ! -x "$PY" ] || [ ! -f "$EVAL_ROOT/candidate_audit.json" ] || [ ! -f "$RUN/F.pth" ]; then
  printf 'Missing Python, validation cache, or frozen F checkpoint\n' >&2
  exit 1
fi
if [ -e "$OUT" ]; then
  printf 'Output already exists: %s\n' "$OUT" >&2
  exit 1
fi
"$PY" -c 'import torch; import local_fusion_detector_adaptation.sota_reproduction.ciassd_iou.evaluate'
if [ "${CHECK_ONLY:-0}" = 1 ]; then
  printf 'Preflight passed.\n'
  exit 0
fi

# Hard gate: disabled method must replay the original and fixed-budget top256 AP.
"$PY" -u -m local_fusion_detector_adaptation.sota_reproduction.ciassd_iou.evaluate \
  --method-disabled --eval-root "$EVAL_ROOT" --output "$OUT/regression"

if [ ! -f "$TRAIN_ROOT/candidate_audit.json" ]; then
  if [ -e "$TRAIN_ROOT" ]; then
    printf 'Incomplete official-train candidate cache: %s\n' "$TRAIN_ROOT" >&2
    exit 1
  fi
  "$PY" -u -m local_fusion_detector_adaptation.candidate_ranker_extract \
    --run "$RUN" --v3-config "${V3_CONFIG:-local_fusion_v3/experiment.yaml}" \
    --frontend-config "$FRONTEND_CONFIG" \
    --frontend-checkpoint "$FRONTEND_CHECKPOINT" \
    --v3-checkpoint "$V3_RUN/residual/best.pth" \
    --train-scenes "${TRAIN_SCENES:-32}" \
    --frames-per-scene "${FRAMES_PER_SCENE:-10}" \
    --seed "${EXPORT_SEED:-20260929}" --output "$TRAIN_ROOT"
fi

"$PY" -u -m local_fusion_detector_adaptation.sota_reproduction.ciassd_iou.extract \
  --split train --candidate-root "$TRAIN_ROOT" --run "$RUN" \
  --v3-config "${V3_CONFIG:-local_fusion_v3/experiment.yaml}" \
  --frontend-config "$FRONTEND_CONFIG" \
  --frontend-checkpoint "$FRONTEND_CHECKPOINT" \
  --v3-checkpoint "$V3_RUN/residual/best.pth" \
  --output "$OUT/train_patches"

exec "$PY" -u -m local_fusion_detector_adaptation.sota_reproduction.ciassd_iou.train \
  --train-root "$TRAIN_ROOT" --patch-root "$OUT/train_patches" \
  --epochs "${EPOCHS:-20}" --batch-size "${BATCH_SIZE:-1024}" \
  --learning-rate "${LEARNING_RATE:-0.001}" --output "$OUT/head"
