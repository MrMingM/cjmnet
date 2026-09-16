#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 EVAL_ROOT" >&2
  exit 2
fi

EVAL_ROOT="$1"
GSPR_DIR="${GSPR_DIR:-/home/cjm/OpenCOOD-main/cjmnet}"
PYTHON="${PYTHON:-/home/cjm/miniconda3/envs/opencood/bin/python}"
DEVICE="${ROCR_VISIBLE_DEVICES:-0}"
OPENCOOD_ROOT="${OPENCOOD_ROOT:-/home/cjm/OpenCOOD-main}"

case "$EVAL_ROOT" in
  /data/cjm/datasets/*) ;;
  *)
    echo "EVAL_ROOT must be inside /data/cjm/datasets" >&2
    exit 2
    ;;
esac

for CONDITION in clean fog rain snow; do
  MODEL_DIR="$EVAL_ROOT/$CONDITION"
  if [[ ! -f "$MODEL_DIR/config.yaml" ]]; then
    echo "Missing $MODEL_DIR/config.yaml" >&2
    exit 1
  fi
  echo "========== $CONDITION =========="
  ROCR_VISIBLE_DEVICES="$DEVICE" \
  PYTHONPATH="$OPENCOOD_ROOT:$GSPR_DIR" \
  "$PYTHON" "$GSPR_DIR/analyze_gspr_reliability.py" \
    --model-dir "$MODEL_DIR" \
    --output "$MODEL_DIR/gspr_reliability_stats.json" \
    --num-workers 8 \
    > "$MODEL_DIR/reliability_audit.log" 2>&1
done

"$PYTHON" "$GSPR_DIR/summarize_gspr_reliability.py" \
  "$EVAL_ROOT/clean/gspr_reliability_stats.json" \
  "$EVAL_ROOT/fog/gspr_reliability_stats.json" \
  "$EVAL_ROOT/rain/gspr_reliability_stats.json" \
  "$EVAL_ROOT/snow/gspr_reliability_stats.json" \
  > "$EVAL_ROOT/gspr_reliability_summary.tsv"

echo "Wrote $EVAL_ROOT/gspr_reliability_summary.tsv"
