#!/bin/sh
set -eu

cd /home/cjm/OpenCOOD-main/cjmnet
export PYTHONPATH="/home/cjm/OpenCOOD-main/cjmnet:/home/cjm/OpenCOOD-main${PYTHONPATH:+:$PYTHONPATH}"
export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0}"
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
export PYTHONHASHSEED=20260929

PY="${PY:-/home/cjm/miniconda3/envs/opencood/bin/python}"
RUN="${RUN:-/data/cjm/datasets/logs/fusion_detector_adaptation_20260927_103227}"
TRAIN_ROOT="${TRAIN_ROOT:-$RUN/candidate_ranker_train_20260929}"
EVAL_ROOT="${EVAL_ROOT:-$RUN/top256_score_geometry_20260928_155556/extraction}"
TRAIN_OVERLAY="${TRAIN_OVERLAY:-$RUN/candidate_r1_r2_train_overlay_v2}"
EVAL_OVERLAY="${EVAL_OVERLAY:-$RUN/candidate_r1_r2_eval_overlay_v2}"
MODE="${MODE:-audit-only}"
OUT="${OUT:-$RUN/candidate_r1_r2_${MODE}_$(date +%Y%m%d_%H%M%S)}"
FRONTEND_ROOT="${FRONTEND_ROOT:-/data/cjm/datasets/logs/gspr_joint_full_v1_seed20260907_20260907_161155}"
FRONTEND_CONFIG="${FRONTEND_CONFIG:-$FRONTEND_ROOT/config.yaml}"
FRONTEND_CHECKPOINT="${FRONTEND_CHECKPOINT:-$FRONTEND_ROOT/net_best_validation.pth}"
V3_RUN="${V3_RUN:-/data/cjm/datasets/logs/local_fusion_v3_20260922_182832}"

if [ "$MODE" != all ] && [ "$MODE" != export-only ] && \
   [ "$MODE" != gate-only ] && [ "$MODE" != audit-only ]; then
  printf 'MODE must be audit-only, all, export-only or gate-only\n' >&2
  exit 1
fi
if [ -f "$EVAL_ROOT/candidate_audit.json" ]; then
  DATA_ROOT="$EVAL_ROOT"
else
  DATA_ROOT="${EVAL_ROOT%/}/extraction"
fi
for path in "$RUN/protocol.json" "$RUN/F.pth" "$RUN/F+D.pth" \
            "$DATA_ROOT/candidate_audit.json" \
            "$FRONTEND_CONFIG" "$FRONTEND_CHECKPOINT" "$V3_RUN/residual/best.pth"; do
  if [ ! -f "$path" ]; then
    printf 'Missing frozen input: %s\n' "$path" >&2
    exit 1
  fi
done
if [ "$MODE" != audit-only ] && [ ! -f "$TRAIN_ROOT/candidate_audit.json" ]; then
  printf 'Missing training candidate audit: %s\n' "$TRAIN_ROOT" >&2
  exit 1
fi
if [ ! -x "$PY" ]; then
  printf 'Missing server Python: %s\n' "$PY" >&2
  exit 1
fi
"$PY" -c 'import local_fusion_detector_adaptation.candidate_r1_r2_extract; import local_fusion_detector_adaptation.candidate_r1_r2_gate; import torch, sklearn, scipy'
if [ "${CHECK_ONLY:-0}" = 1 ]; then
  printf 'R1/R2 preflight passed; no extraction or analysis ran.\n'
  exit 0
fi

if [ "$MODE" != gate-only ]; then
  if [ "$MODE" = audit-only ]; then
    set -- "validation:$DATA_ROOT:$EVAL_OVERLAY"
  else
    set -- "train:$TRAIN_ROOT:$TRAIN_OVERLAY" "validation:$DATA_ROOT:$EVAL_OVERLAY"
  fi
  for spec do
    split=${spec%%:*}
    rest=${spec#*:}
    base=${rest%%:*}
    destination=${rest#*:}
    if [ -f "$destination/overlay.json" ]; then
      printf 'Reusing complete %s overlay: %s\n' "$split" "$destination"
      continue
    fi
    if [ -e "$destination" ]; then
      printf 'Partial overlay exists: %s\n' "$destination" >&2
      exit 1
    fi
    "$PY" -u -m local_fusion_detector_adaptation.candidate_r1_r2_extract \
      --base-root "$base" --run "$RUN" \
      --v3-config "${V3_CONFIG:-local_fusion_v3/experiment.yaml}" \
      --frontend-config "$FRONTEND_CONFIG" \
      --frontend-checkpoint "$FRONTEND_CHECKPOINT" \
      --v3-checkpoint "$V3_RUN/residual/best.pth" \
      --source-topk "${SOURCE_TOPK:-256}" \
      --source-preselect "${SOURCE_PRESELECT:-2048}" \
      --output "$destination"
  done
fi
if [ "$MODE" = export-only ]; then
  printf 'R1/R2 overlay export complete.\n'
  exit 0
fi
if [ "$MODE" = audit-only ]; then
  set -- "$EVAL_OVERLAY/overlay.json"
else
  set -- "$TRAIN_OVERLAY/overlay.json" "$EVAL_OVERLAY/overlay.json"
fi
for path do
  if [ ! -f "$path" ]; then
    printf 'Missing overlay: %s\n' "$path" >&2
    exit 1
  fi
done
if [ -e "$OUT" ]; then
  printf 'Output already exists: %s\n' "$OUT" >&2
  exit 1
fi
if [ "$MODE" = audit-only ]; then
  exec "$PY" -u -m local_fusion_detector_adaptation.candidate_r1_r2_gate \
    --train-root "$TRAIN_ROOT" --eval-root "$DATA_ROOT" \
    --train-overlay "$TRAIN_OVERLAY" --eval-overlay "$EVAL_OVERLAY" \
    --output "$OUT" --audit-only
fi
exec "$PY" -u -m local_fusion_detector_adaptation.candidate_r1_r2_gate \
  --train-root "$TRAIN_ROOT" --eval-root "$DATA_ROOT" \
  --train-overlay "$TRAIN_OVERLAY" --eval-overlay "$EVAL_OVERLAY" \
  --seeds "${SEEDS:-20260929,20260930,20260931}" \
  --max-per-class-weather "${MAX_PER_CLASS_WEATHER:-5000}" \
  --output "$OUT"
