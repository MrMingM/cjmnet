#!/usr/bin/env bash
set -euo pipefail
cd /home/cjm/OpenCOOD-main/cjmnet
export PYTHONPATH=/home/cjm/OpenCOOD-main/cjmnet:/home/cjm/OpenCOOD-main
export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0}"
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
export PYTHONHASHSEED=20260913
PY=/home/cjm/miniconda3/envs/opencood/bin/python
FRONTEND_CONFIG="${FRONTEND_CONFIG:-/data/cjm/datasets/logs/gspr_joint_full_v1_seed20260907_20260907_161155/config.yaml}"
FRONTEND_CHECKPOINT="${FRONTEND_CHECKPOINT:-/data/cjm/datasets/logs/gspr_joint_full_v1_seed20260907_20260907_161155/net_best_validation.pth}"
PHASE="${PHASE:-development}"
BASELINE="${BASELINE:-full}"
SMOKE="${SMOKE:-0}"
CONFIG="${CONFIG:-gspr_evidence/experiment.yaml}"
extra=()
if [[ "$BASELINE" == a0b0 ]]; then
  : "${BUDGET_BYTES:?A0B0 requires the explicitly chosen budget in bytes}"
  extra+=(--budget-bytes "$BUDGET_BYTES")
fi
if [[ -n "${SETTINGS:-}" ]]; then extra+=(--settings "$SETTINGS"); fi
OUT="${OUT:-/data/cjm/datasets/logs/ceif_audit_${PHASE}_${BASELINE}_$(date +%Y%m%d_%H%M%S)}"
mkdir "$OUT"
exec > >(tee -a "$OUT/console.log") 2>&1
echo "RESULT_DIR=$OUT"
echo "PHASE=$PHASE BASELINE=$BASELINE SMOKE=$SMOKE"
"$PY" -m unittest ceif_audit.test_audit ceif_audit.test_runner -v
for weather in clean fog rain snow; do
  "$PY" -m ceif_audit.run --phase "$PHASE" --weather "$weather" --baseline "$BASELINE" \
    --config "$CONFIG" --frontend-config "$FRONTEND_CONFIG" --frontend-checkpoint "$FRONTEND_CHECKPOINT" \
    --smoke "$SMOKE" "${extra[@]}" --output-dir "$OUT/$weather"
done
"$PY" -m ceif_audit.report --run "$OUT"
echo "Send back: $OUT/feasibility.md and clean/fog/rain/snow/audit.md"
