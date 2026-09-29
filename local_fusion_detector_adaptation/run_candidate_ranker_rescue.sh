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
OUT="${OUT:-$RUN/candidate_ranker_rescue_$(date +%Y%m%d_%H%M%S)}"
MODE="${MODE:-train-eval}"

if [ -f "$EVAL_ROOT/candidate_audit.json" ]; then
  DATA_ROOT="$EVAL_ROOT"
else
  DATA_ROOT="${EVAL_ROOT%/}/extraction"
fi
if [ "$MODE" != train-eval ] && [ "$MODE" != evaluate-only ]; then
  printf 'MODE must be train-eval or evaluate-only\n' >&2
  exit 1
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
  if [ ! -f "$TRAIN_ROOT/candidate_audit.json" ]; then
    printf 'Missing completed training export: %s\n' "$TRAIN_ROOT" >&2
    exit 1
  fi
else
  if [ ! -f "${CHECKPOINT_DIR:-}/manifest.json" ]; then
    printf 'CHECKPOINT_DIR must contain manifest.json\n' >&2
    exit 1
  fi
fi

"$PY" -c 'import local_fusion_detector_adaptation.candidate_ranker_rescue; import torch; from opencood.utils import common_utils, eval_utils'
if [ "${CHECK_ONLY:-0}" = 1 ]; then
  printf 'Preflight passed; no research experiment ran.\n'
  exit 0
fi
if [ -e "$OUT" ]; then
  printf 'Output already exists: %s\n' "$OUT" >&2
  exit 1
fi
if [ "$MODE" = train-eval ]; then
  exec "$PY" -u -m local_fusion_detector_adaptation.candidate_ranker_rescue \
    --mode train-eval --train-root "$TRAIN_ROOT" --eval-root "$DATA_ROOT" \
    --output-dir "$OUT" --seeds "${SEEDS:-20260929,20260930,20260931}"
fi
exec "$PY" -u -m local_fusion_detector_adaptation.candidate_ranker_rescue \
  --mode evaluate-only --checkpoint-dir "$CHECKPOINT_DIR" \
  --eval-root "$DATA_ROOT" --output-dir "$OUT"
