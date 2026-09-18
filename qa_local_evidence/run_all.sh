#!/bin/sh
set -eu

cd /home/cjm/OpenCOOD-main/cjmnet
export PYTHONPATH=/home/cjm/OpenCOOD-main/cjmnet:/home/cjm/OpenCOOD-main
export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-5}"
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
export PYTHONHASHSEED="${PYTHONHASHSEED:-20260917}"

PY="${PY:-/home/cjm/miniconda3/envs/opencood/bin/python}"
CONFIG="${CONFIG:-qa_observation_diagnostic/experiment.yaml}"
FRONTEND_CONFIG="${FRONTEND_CONFIG:-/data/cjm/datasets/logs/gspr_joint_full_v1_seed20260907_20260907_161155/config.yaml}"
FRONTEND_CHECKPOINT="${FRONTEND_CHECKPOINT:-/data/cjm/datasets/logs/gspr_joint_full_v1_seed20260907_20260907_161155/net_best_validation.pth}"
STAGE1_ROOT="${STAGE1_ROOT:-/data/cjm/datasets/logs/qa_observation_20260916_194240}"
SMOKE="${SMOKE:-0}"
RUN_NAME="${RUN_NAME:-qa_local_evidence_$(date +%Y%m%d_%H%M%S)}"
OUT="${OUT:-/data/cjm/datasets/logs/$RUN_NAME}"

if [ ! -d "$STAGE1_ROOT" ]; then
  echo "Stage-1 root not found: $STAGE1_ROOT" >&2
  exit 2
fi
if [ "$ROCR_VISIBLE_DEVICES" = "7" ]; then
  echo "HCU 7 is reserved; choose a free physical device 0-6." >&2
  exit 2
fi

mkdir -p "$OUT"

echo "============================================================"
echo "H-A5 local evidence audit"
echo "DEVELOPMENT VALIDATION ONLY — no OPV2V-W test data"
echo "Stage-1 root: $STAGE1_ROOT"
echo "GPU physical index: $ROCR_VISIBLE_DEVICES"
echo "Smoke candidate frames per weather: $SMOKE"
echo "Output: $OUT"
echo "============================================================"

"$PY" -m unittest qa_local_evidence.test_analysis -v

for weather in fog rain snow; do
  echo "----- H-A5 paired local audit: $weather -----"
  "$PY" -m qa_local_evidence.audit \
    --stage1-root "$STAGE1_ROOT" \
    --config "$CONFIG" \
    --frontend-config "$FRONTEND_CONFIG" \
    --frontend-checkpoint "$FRONTEND_CHECKPOINT" \
    --weather "$weather" \
    --smoke "$SMOKE" \
    --output-dir "$OUT/$weather"
done

"$PY" -m qa_local_evidence.report \
  --audit-root "$OUT" \
  --output-dir "$OUT"

echo
echo "DONE"
echo "Main report: $OUT/a5_report.md"
echo "Machine-readable results: $OUT/a5_results.json"
echo "Per-target rows: $OUT/{fog,rain,snow}/targets.jsonl"
echo
echo "Please send back a5_report.md and a5_results.json."
