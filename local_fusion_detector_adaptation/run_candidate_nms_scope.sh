#!/bin/sh
set -eu

cd /home/cjm/OpenCOOD-main/cjmnet
export PYTHONPATH="/home/cjm/OpenCOOD-main/cjmnet:/home/cjm/OpenCOOD-main${PYTHONPATH:+:$PYTHONPATH}"

INPUT_ROOT="${INPUT_ROOT:-/data/cjm/datasets/logs/fusion_detector_adaptation_20260927_103227/top256_score_geometry_20260928_155556}"
PY="${PY:-/home/cjm/miniconda3/envs/opencood/bin/python}"
ARM="${ARM:-F}"
OUT="${OUT:-${INPUT_ROOT%/}/nms_scope_$(date +%Y%m%d_%H%M%S)}"

if [ "$ARM" != F ] && [ "$ARM" != 'F+D' ]; then
  printf 'ARM must be F or F+D\n' >&2
  exit 1
fi
if [ ! -x "$PY" ]; then
  printf 'Python is unavailable: %s\n' "$PY" >&2
  exit 1
fi
if [ -f "$INPUT_ROOT/candidate_audit.json" ]; then
  DATA_ROOT="$INPUT_ROOT"
else
  DATA_ROOT="${INPUT_ROOT%/}/extraction"
fi
if [ ! -f "$DATA_ROOT/candidate_audit.json" ]; then
  printf 'Missing candidate_audit.json under %s\n' "$DATA_ROOT" >&2
  exit 1
fi
for weather in clean fog rain snow; do
  for name in candidate_rows.jsonl frame_targets.jsonl; do
    if [ ! -f "$DATA_ROOT/$weather/$name" ]; then
      printf 'Missing input: %s\n' "$DATA_ROOT/$weather/$name" >&2
      exit 1
    fi
  done
done

if [ "${CHECK_ONLY:-0}" = 1 ]; then
  "$PY" -c 'import local_fusion_detector_adaptation.candidate_nms_scope; import gspr_evidence.stage3_trace; from opencood.utils import box_utils, eval_utils'
  printf 'Preflight passed; no experiment was run. Input: %s\n' "$DATA_ROOT"
  exit 0
fi
if [ -e "$OUT" ]; then
  printf 'Output already exists: %s\n' "$OUT" >&2
  exit 1
fi

exec "$PY" -u -m local_fusion_detector_adaptation.candidate_nms_scope \
  --input-root "$DATA_ROOT" --output-dir "$OUT" --arm "$ARM"
