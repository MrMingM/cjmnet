#!/bin/sh
set -eu

cd /home/cjm/OpenCOOD-main/cjmnet
export PYTHONPATH="/home/cjm/OpenCOOD-main/cjmnet:/home/cjm/OpenCOOD-main${PYTHONPATH:+:$PYTHONPATH}"
export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0}"
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
export PYTHONHASHSEED=20260929

PY="${PY:-/home/cjm/miniconda3/envs/opencood/bin/python}"
RUN="${RUN:-/data/cjm/datasets/logs/fusion_detector_adaptation_20260927_103227}"
EVAL_ROOT="${EVAL_ROOT:-$RUN/top256_score_geometry_20260928_155556/extraction}"
TRAIN_ROOT="${TRAIN_ROOT:-$RUN/candidate_ranker_train_20260929}"
OUT="${OUT:-$RUN/candidate_ranker_pilot_$(date +%Y%m%d_%H%M%S)}"
MODE="${MODE:-train-eval}"
FRONTEND_ROOT="${FRONTEND_ROOT:-/data/cjm/datasets/logs/gspr_joint_full_v1_seed20260907_20260907_161155}"
FRONTEND_CONFIG="${FRONTEND_CONFIG:-$FRONTEND_ROOT/config.yaml}"
FRONTEND_CHECKPOINT="${FRONTEND_CHECKPOINT:-$FRONTEND_ROOT/net_best_validation.pth}"
V3_RUN="${V3_RUN:-/data/cjm/datasets/logs/local_fusion_v3_20260922_182832}"

if [ "$MODE" != train-eval ] && [ "$MODE" != evaluate-only ]; then
  printf 'MODE must be train-eval or evaluate-only\n' >&2
  exit 1
fi
if [ -f "$EVAL_ROOT/candidate_audit.json" ]; then
  DATA_ROOT="$EVAL_ROOT"
else
  DATA_ROOT="${EVAL_ROOT%/}/extraction"
fi
if [ ! -x "$PY" ] || [ ! -f "$DATA_ROOT/candidate_audit.json" ]; then
  printf 'Missing server Python or evaluation candidate_audit.json\n' >&2
  exit 1
fi
for weather in clean fog rain snow; do
  for name in candidate_rows.jsonl frame_targets.jsonl; do
    if [ ! -f "$DATA_ROOT/$weather/$name" ]; then
      printf 'Missing evaluation file: %s\n' "$DATA_ROOT/$weather/$name" >&2
      exit 1
    fi
  done
done
if [ "$MODE" = train-eval ]; then
  for path in "$RUN/protocol.json" "$RUN/F.pth" "$FRONTEND_CONFIG" \
              "$FRONTEND_CHECKPOINT" "$V3_RUN/residual/best.pth"; do
    if [ ! -f "$path" ]; then
      printf 'Missing frozen-model input: %s\n' "$path" >&2
      exit 1
    fi
  done
  if [ -e "$TRAIN_ROOT" ] && [ ! -f "$TRAIN_ROOT/candidate_audit.json" ]; then
    printf 'Partial training export exists: %s\n' "$TRAIN_ROOT" >&2
    exit 1
  fi
else
  if [ ! -f "${CHECKPOINT_DIR:-}/manifest.json" ]; then
    printf 'CHECKPOINT_DIR must contain manifest.json\n' >&2
    exit 1
  fi
fi
"$PY" -c 'import local_fusion_detector_adaptation.candidate_ranker_extract; import local_fusion_detector_adaptation.candidate_ranker_pilot; import torch; from opencood.utils import box_utils, eval_utils'
if [ "${CHECK_ONLY:-0}" = 1 ]; then
  printf 'Preflight passed; no extraction, training or evaluation ran.\n'
  exit 0
fi
if [ -e "$OUT" ]; then
  printf 'Output already exists: %s\n' "$OUT" >&2
  exit 1
fi

if [ "$MODE" = train-eval ]; then
  if [ ! -f "$TRAIN_ROOT/candidate_audit.json" ]; then
    "$PY" -u -m local_fusion_detector_adaptation.candidate_ranker_extract \
      --run "$RUN" \
      --v3-config "${V3_CONFIG:-local_fusion_v3/experiment.yaml}" \
      --frontend-config "$FRONTEND_CONFIG" \
      --frontend-checkpoint "$FRONTEND_CHECKPOINT" \
      --v3-checkpoint "$V3_RUN/residual/best.pth" \
      --train-scenes "${TRAIN_SCENES:-32}" \
      --frames-per-scene "${FRAMES_PER_SCENE:-10}" \
      --seed "${EXPORT_SEED:-20260929}" \
      --output "$TRAIN_ROOT"
  fi
  exec "$PY" -u -m local_fusion_detector_adaptation.candidate_ranker_pilot \
    --mode train-eval --train-root "$TRAIN_ROOT" --eval-root "$DATA_ROOT" \
    --output-dir "$OUT" --seeds "${SEEDS:-20260929,20260930,20260931}"
fi

exec "$PY" -u -m local_fusion_detector_adaptation.candidate_ranker_pilot \
  --mode evaluate-only --checkpoint-dir "$CHECKPOINT_DIR" \
  --eval-root "$DATA_ROOT" --output-dir "$OUT"
