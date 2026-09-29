#!/bin/sh
set -eu

REPO="${REPO:-/home/cjm/OpenCOOD-main/cjmnet}"
cd "$REPO"
export PYTHONPATH="$REPO:/home/cjm/OpenCOOD-main${PYTHONPATH:+:$PYTHONPATH}"
export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-3}"
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
export PYTHONHASHSEED=20260929

PY="${PY:-/home/cjm/miniconda3/envs/opencood/bin/python}"
RUN="${RUN:-/data/cjm/datasets/logs/fusion_detector_adaptation_20260927_103227}"
MODEL="${MODEL:-}"
MODE="${MODE:-validation}"
OUT="${OUT:-/data/cjm/datasets/logs/lmd_${MODE}_$(date +%Y%m%d_%H%M%S)}"
EVAL_CACHE="${EVAL_CACHE:-$OUT/cache}"
FRONTEND_ROOT="${FRONTEND_ROOT:-/data/cjm/datasets/logs/gspr_joint_full_v1_seed20260907_20260907_161155}"
FRONTEND_CONFIG="${FRONTEND_CONFIG:-$FRONTEND_ROOT/config.yaml}"
FRONTEND_CHECKPOINT="${FRONTEND_CHECKPOINT:-$FRONTEND_ROOT/net_best_validation.pth}"
V3_RUN="${V3_RUN:-/data/cjm/datasets/logs/local_fusion_v3_20260922_182832}"
V3_CHECKPOINT="${V3_CHECKPOINT:-$V3_RUN/residual/best.pth}"

if [ "$MODE" != validation ] && [ "$MODE" != benchmark ]; then
  printf 'MODE must be validation or benchmark\n' >&2
  exit 1
fi
if [ ! -x "$PY" ] || [ ! -f "$MODEL" ] || [ ! -f "$RUN/F.pth" ]; then
  printf 'Missing server Python, MODEL, or frozen F checkpoint\n' >&2
  exit 1
fi
"$PY" -m unittest local_fusion_detector_adaptation.sota_reproduction.lmd.test_lmd
if [ "${CHECK_ONLY:-0}" = 1 ]; then
  printf 'LMD evaluation preflight passed.\n'
  exit 0
fi
if [ -e "$OUT" ] && [ ! -f "$EVAL_CACHE/candidate_audit.json" ]; then
  printf 'Output exists but cache is incomplete: %s\n' "$OUT" >&2
  exit 1
fi
if [ ! -f "$EVAL_CACHE/candidate_audit.json" ]; then
  if [ -e "$EVAL_CACHE" ]; then
    printf 'Incomplete evaluation cache: %s\n' "$EVAL_CACHE" >&2
    exit 1
  fi
  set -- "$PY" -u -m local_fusion_detector_adaptation.sota_reproduction.lmd.extract \
    --run "$RUN" --v3-config "${V3_CONFIG:-local_fusion_v3/experiment.yaml}" \
    --frontend-config "$FRONTEND_CONFIG" --frontend-checkpoint "$FRONTEND_CHECKPOINT" \
    --v3-checkpoint "$V3_CHECKPOINT" --split "$MODE" --output "$EVAL_CACHE"
  if [ "${SMOKE:-0}" -gt 0 ]; then
    set -- "$@" --smoke "$SMOKE"
  fi
  "$@"
fi
RESULTS="${RESULTS:-$OUT/results}"
if [ -e "$RESULTS" ]; then
  printf 'Result directory already exists: %s\n' "$RESULTS" >&2
  exit 1
fi
exec "$PY" -u -m local_fusion_detector_adaptation.sota_reproduction.lmd.evaluate \
  --eval-root "$EVAL_CACHE" --model "$MODEL" --output "$RESULTS"
