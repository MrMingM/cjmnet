#!/bin/sh
set -eu

REPO="${REPO:-/home/cjm/OpenCOOD-main/cjmnet}"
cd "$REPO"
export PYTHONPATH="$REPO:/home/cjm/OpenCOOD-main${PYTHONPATH:+:$PYTHONPATH}"
export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-3}"
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
export PYTHONHASHSEED=20260929
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-12}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-12}"

PY="${PY:-/home/cjm/miniconda3/envs/opencood/bin/python}"
RUN="${RUN:-/data/cjm/datasets/logs/fusion_detector_adaptation_20260927_103227}"
OUT="${OUT:-/data/cjm/datasets/logs/lmd_$(date +%Y%m%d_%H%M%S)}"
TRAIN_CACHE="${TRAIN_CACHE:-$OUT/train_cache}"
FRONTEND_ROOT="${FRONTEND_ROOT:-/data/cjm/datasets/logs/gspr_joint_full_v1_seed20260907_20260907_161155}"
FRONTEND_CONFIG="${FRONTEND_CONFIG:-$FRONTEND_ROOT/config.yaml}"
FRONTEND_CHECKPOINT="${FRONTEND_CHECKPOINT:-$FRONTEND_ROOT/net_best_validation.pth}"
V3_RUN="${V3_RUN:-/data/cjm/datasets/logs/local_fusion_v3_20260922_182832}"
V3_CHECKPOINT="${V3_CHECKPOINT:-$V3_RUN/residual/best.pth}"

if [ ! -x "$PY" ] || [ ! -f "$RUN/F.pth" ] || [ ! -f "$FRONTEND_CHECKPOINT" ] || [ ! -f "$V3_CHECKPOINT" ]; then
  printf 'Missing server Python or frozen F/frontend/v3 checkpoint\n' >&2
  exit 1
fi
if [ -e "$OUT" ] && [ ! -f "$TRAIN_CACHE/candidate_audit.json" ]; then
  printf 'Output exists but has no reusable complete train cache: %s\n' "$OUT" >&2
  exit 1
fi
"$PY" -m unittest local_fusion_detector_adaptation.sota_reproduction.lmd.test_lmd
if [ "${CHECK_ONLY:-0}" = 1 ]; then
  printf 'LMD preflight passed on physical HCU %s (program-visible cuda:0).\n' "$ROCR_VISIBLE_DEVICES"
  exit 0
fi

if [ ! -f "$TRAIN_CACHE/candidate_audit.json" ]; then
  if [ -e "$TRAIN_CACHE" ]; then
    printf 'Incomplete LMD train cache already exists: %s\n' "$TRAIN_CACHE" >&2
    exit 1
  fi
  set -- "$PY" -u -m local_fusion_detector_adaptation.sota_reproduction.lmd.extract \
    --run "$RUN" --v3-config "${V3_CONFIG:-local_fusion_v3/experiment.yaml}" \
    --frontend-config "$FRONTEND_CONFIG" --frontend-checkpoint "$FRONTEND_CHECKPOINT" \
    --v3-checkpoint "$V3_CHECKPOINT" --split train --output "$TRAIN_CACHE"
  if [ "${SMOKE:-0}" -gt 0 ]; then
    set -- "$@" --smoke "$SMOKE"
  fi
  "$@"
fi

MODEL_OUT="${MODEL_OUT:-$OUT/model}"
if [ -e "$MODEL_OUT" ]; then
  printf 'LMD model output already exists: %s\n' "$MODEL_OUT" >&2
  exit 1
fi
set -- "$PY" -u -m local_fusion_detector_adaptation.sota_reproduction.lmd.train \
  --train-root "$TRAIN_CACHE" --output "$MODEL_OUT" \
  --proposal-iou-threshold "${PROPOSAL_IOU_THRESHOLD:-0.2}" \
  --classification-iou-threshold "${CLASSIFICATION_IOU_THRESHOLD:-0.5}" \
  --max-rows-per-weather "${MAX_ROWS_PER_WEATHER:-50000}" \
  --jobs "${JOBS:-12}" --seed "${SEED:-20260929}"
if [ "${TRAIN_CLASSIFICATION:-0}" != 1 ]; then
  set -- "$@" --skip-classification
fi
exec "$@"
