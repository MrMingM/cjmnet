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
extra=()
case "$PHASE" in
  development) CONFIG="${CONFIG:-gspr_evidence/experiment.yaml}" ;;
  benchmark)
    : "${DEV:?请把 DEV 设为已完成的开发预算扫描目录}"
    CONFIG="$DEV/experiment.yaml"
    extra=(--selection "$DEV/selection.json")
    test -f "$CONFIG"
    test -f "$DEV/selection.json"
    ;;
  *) echo 'PHASE 只能是 development 或 benchmark' >&2; exit 1 ;;
esac
OUT="/data/cjm/datasets/logs/gspr_a0b0_budget_${PHASE}_$(date +%Y%m%d_%H%M%S)"
mkdir "$OUT"
exec > >(tee -a "$OUT/console.log") 2>&1
echo "PHASE=$PHASE"
echo "RESULT_DIR=$OUT"
"$PY" -m unittest gspr_evidence.test_budget_scan -v
for weather in clean fog rain snow; do
  "$PY" -m gspr_evidence.budget_scan --phase "$PHASE" --weather "$weather" \
    --config "$CONFIG" --frontend-config "$FRONTEND_CONFIG" \
    --frontend-checkpoint "$FRONTEND_CHECKPOINT" "${extra[@]}" \
    --output-dir "$OUT/$weather"
done
"$PY" -m gspr_evidence.budget_report --run "$OUT"
echo "Send back: $OUT/budget_results.md"
