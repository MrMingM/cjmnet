#!/usr/bin/env bash
set -euo pipefail

cd /home/cjm/OpenCOOD-main/cjmnet
export PYTHONPATH=/home/cjm/OpenCOOD-main/cjmnet:/home/cjm/OpenCOOD-main
export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0}"
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
export PYTHONHASHSEED="${PYTHONHASHSEED:-20260916}"

PY="${PY:-/home/cjm/miniconda3/envs/opencood/bin/python}"
CONFIG="${CONFIG:-qa_observation_diagnostic/experiment.yaml}"
FRONTEND_CONFIG="${FRONTEND_CONFIG:-/data/cjm/datasets/logs/gspr_joint_full_v1_seed20260907_20260907_161155/config.yaml}"
FRONTEND_CHECKPOINT="${FRONTEND_CHECKPOINT:-/data/cjm/datasets/logs/gspr_joint_full_v1_seed20260907_20260907_161155/net_best_validation.pth}"
SMOKE="${SMOKE:-0}"
RUN_NAME="${RUN_NAME:-qa_observation_$(date +%Y%m%d_%H%M%S)}"
OUT="${OUT:-/data/cjm/datasets/logs/$RUN_NAME}"

case "$SMOKE" in
  ''|*[!0-9]*) echo "SMOKE must be a non-negative integer" >&2; exit 2 ;;
esac

mkdir -p "$OUT"
exec > >(tee -a "$OUT/console.log") 2>&1

echo "============================================================"
echo "Q-A observation diagnostic"
echo "DEVELOPMENT VALIDATION ONLY — no OPV2V-W test data"
echo "GPU physical index: $ROCR_VISIBLE_DEVICES"
echo "Output: $OUT"
echo "Smoke frames: $SMOKE (0 = full validation)"
echo "============================================================"

"$PY" -m unittest qa_observation_diagnostic.test_metrics qa_observation_diagnostic.test_geometry -v

for weather in clean fog rain snow; do
  echo "----- collecting $weather -----"
  "$PY" -m qa_observation_diagnostic.collect \
    --config "$CONFIG" \
    --frontend-config "$FRONTEND_CONFIG" \
    --frontend-checkpoint "$FRONTEND_CHECKPOINT" \
    --weather "$weather" \
    --smoke "$SMOKE" \
    --output-dir "$OUT/$weather"
done

echo "----- generating unified hypothesis report -----"
"$PY" -m qa_observation_diagnostic.report --root "$OUT"

echo
echo "DONE"
echo "Main report: $OUT/hypothesis_report.md"
echo "Machine-readable results: $OUT/hypothesis_results.json"
echo "Per-target evidence: $OUT/{clean,fog,rain,snow}/targets.jsonl"
echo "Per-frame visibility summaries: $OUT/{clean,fog,rain,snow}/frames.jsonl"
echo
echo "Please send back hypothesis_report.md and hypothesis_results.json."
