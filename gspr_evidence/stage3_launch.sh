#!/usr/bin/env bash
# Shared launcher. Sourced only by an explicit A or B entrypoint.
set -euo pipefail
: "${STAGE:?Use run_stage3a.sh or run_stage3b.sh}"
[[ "$STAGE" == 3A || "$STAGE" == 3B ]] || exit 2
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
RUN_NAME="${RUN_NAME:-qa_stage${STAGE,,}_$(date +%Y%m%d_%H%M%S)}"
[[ "$RUN_NAME" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid RUN_NAME'; exit 2; }
SMOKE="${SMOKE:-0}"
[[ "$SMOKE" =~ ^[0-9]+$ ]] || { echo 'SMOKE must be a nonnegative integer'; exit 2; }
OUT="/data/cjm/datasets/logs/$RUN_NAME"
"$PY" -c 'import sys; from gspr_evidence.stage3_runtime import safe_output; safe_output(sys.argv[1], create=True)' "$OUT"
exec > >(tee "$OUT/console.log") 2>&1
echo "Stage-$STAGE development validation only; smoke=$SMOKE; output=$OUT"
"$PY" -m unittest gspr_evidence.test_stage3 -v
"$PY" -m gspr_evidence.stage3_report --mode offline --stage2-root "$STAGE2_ROOT" --output-dir "$OUT/stage2_5"
module="gspr_evidence.stage${STAGE,,}"
extra=()
if [[ "$STAGE" == 3B ]]; then
  extra=(--stage3a-root "$STAGE3A_ROOT")
fi
run_weather() {
  "$PY" -m "$module" --stage2-root "$STAGE2_ROOT" --config "$CONFIG" \
    --frontend-config "$FRONTEND_CONFIG" --frontend-checkpoint "$FRONTEND_CHECKPOINT" \
    --weather "$1" --output-dir "$2" --smoke "$3" "${extra[@]}"
}
# A full run always passes an independent two-candidate-frame smoke per weather first.
# It replays the original loader from its beginning; no subsampled weather generation.
if [[ "$SMOKE" == 0 ]]; then
  for weather in fog rain snow; do
    run_weather "$weather" "$OUT/preflight/$weather" 2
  done
fi
for weather in fog rain snow; do
  run_weather "$weather" "$OUT/$weather" "$SMOKE"
done
"$PY" -m gspr_evidence.stage3_report --mode "$STAGE" --output-dir "$OUT"
echo "DONE: $OUT/stage${STAGE,,}_report.md"
if [[ "$STAGE" == 3A ]]; then
  echo 'Read this report before deciding whether to launch Stage-3B manually.'
fi
