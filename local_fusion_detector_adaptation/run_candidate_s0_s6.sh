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
OUT="${OUT:-$RUN/candidate_s0_s6_$(date +%Y%m%d_%H%M%S)}"

if [ ! -x "$PY" ] || [ ! -f "$RUN/F.pth" ]; then
  printf 'Missing server Python or frozen F checkpoint\n' >&2
  exit 1
fi
for root in "$TRAIN_ROOT" "$EVAL_ROOT"; do
  if [ ! -f "$root/candidate_audit.json" ]; then
    printf 'Missing top256 candidate audit: %s\n' "$root" >&2
    exit 1
  fi
  for weather in clean fog rain snow; do
    for name in candidate_rows.jsonl frame_targets.jsonl; do
      if [ ! -f "$root/$weather/$name" ]; then
        printf 'Missing candidate input: %s/%s/%s\n' "$root" "$weather" "$name" >&2
        exit 1
      fi
    done
  done
done
if [ -e "$OUT" ]; then
  printf 'Output already exists: %s\n' "$OUT" >&2
  exit 1
fi

"$PY" -c 'import local_fusion_detector_adaptation.candidate_s0_s6; import torch; from opencood.utils import common_utils'
if [ "${CHECK_ONLY:-0}" = 1 ]; then
  printf 'Preflight passed. No training or evaluation ran.\n'
  exit 0
fi

exec "$PY" -u -m local_fusion_detector_adaptation.candidate_s0_s6 \
  --train-root "$TRAIN_ROOT" --eval-root "$EVAL_ROOT" \
  --f-checkpoint "$RUN/F.pth" --output-dir "$OUT" \
  --seeds "${SEEDS:-20260929,20260930,20260931}"
