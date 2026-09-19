#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$PWD:/home/cjm/OpenCOOD-main"
export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-3}"
[[ "$ROCR_VISIBLE_DEVICES" =~ ^[0-6]$ ]] || { echo 'Use one physical HCU index 0..6'; exit 2; }
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
export PYTHONHASHSEED=20260917
PY="${PY:-/home/cjm/miniconda3/envs/opencood/bin/python}"
CONFIG="${CONFIG:-qa_observation_diagnostic/experiment.yaml}"
FRONTEND_CONFIG="${FRONTEND_CONFIG:-/data/cjm/datasets/logs/gspr_joint_full_v1_seed20260907_20260907_161155/config.yaml}"
FRONTEND_CHECKPOINT="${FRONTEND_CHECKPOINT:-/data/cjm/datasets/logs/gspr_joint_full_v1_seed20260907_20260907_161155/net_best_validation.pth}"
STAGE2_ROOT="${STAGE2_ROOT:-/data/cjm/datasets/logs/qa_evidence_validity_20260917_112301}"
STAGE3B_ROOT="${STAGE3B_ROOT:-/data/cjm/datasets/logs/qa_stage3b_diagnostic_20260918_110054}"
RUN_NAME="${RUN_NAME:-qa_stage3c_local_$(date +%Y%m%d_%H%M%S)}"
SMOKE="${SMOKE:-0}"
[[ "$RUN_NAME" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid RUN_NAME'; exit 2; }
[[ "$SMOKE" =~ ^[0-9]+$ ]] || { echo 'SMOKE must be a nonnegative integer'; exit 2; }
OUT="/data/cjm/datasets/logs/$RUN_NAME"
"$PY" -c 'import sys; from gspr_evidence.stage3_runtime import safe_output; safe_output(sys.argv[1], create=True)' "$OUT"
exec > >(tee "$OUT/console.log") 2>&1
echo "Stage-3C local diagnostic; smoke=$SMOKE; output=$OUT"
"$PY" -c 'import torch, shapely; assert torch.cuda.is_available(), "GPU/HCU unavailable"'
"$PY" -m unittest qa_local_intervention.test_local -v
"$PY" -m qa_local_intervention.plan --stage3b-root "$STAGE3B_ROOT" \
  --stage2-root "$STAGE2_ROOT" --output-dir "$OUT/plan"
run_weather() {
  "$PY" -m qa_local_intervention.run --plan "$OUT/plan/sampling_plan.json" \
    --stage2-root "$STAGE2_ROOT" --config "$CONFIG" \
    --frontend-config "$FRONTEND_CONFIG" --frontend-checkpoint "$FRONTEND_CHECKPOINT" \
    --weather "$1" --output-dir "$2" --smoke "$3"
}
# Always preserve the original weather queue, even for a small inference sample.
# Any failed test, replay guard or report validation aborts before the full run.
if [[ "$SMOKE" == 0 ]]; then
  for weather in fog rain snow; do
    run_weather "$weather" "$OUT/preflight/$weather" 2
  done
  "$PY" -m qa_local_intervention.report --output-dir "$OUT/preflight"
fi
for weather in fog rain snow; do
  run_weather "$weather" "$OUT/$weather" "$SMOKE"
done
"$PY" -m qa_local_intervention.report --output-dir "$OUT"
echo "DONE: $OUT/stage3c_report.md"
