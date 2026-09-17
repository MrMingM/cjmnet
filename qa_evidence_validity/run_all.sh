#!/usr/bin/env bash
set -euo pipefail

cd /home/cjm/OpenCOOD-main/cjmnet
export PYTHONPATH=/home/cjm/OpenCOOD-main/cjmnet:/home/cjm/OpenCOOD-main
export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0}"
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
export PYTHONHASHSEED="${PYTHONHASHSEED:-20260917}"

PY="${PY:-/home/cjm/miniconda3/envs/opencood/bin/python}"
CONFIG="${CONFIG:-qa_observation_diagnostic/experiment.yaml}"
FRONTEND_CONFIG="${FRONTEND_CONFIG:-/data/cjm/datasets/logs/gspr_joint_full_v1_seed20260907_20260907_161155/config.yaml}"
FRONTEND_CHECKPOINT="${FRONTEND_CHECKPOINT:-/data/cjm/datasets/logs/gspr_joint_full_v1_seed20260907_20260907_161155/net_best_validation.pth}"
STAGE1_ROOT="${STAGE1_ROOT:-/data/cjm/datasets/logs/qa_observation_20260916_194240}"
RUN_NAME="${RUN_NAME:-qa_evidence_validity_$(date +%Y%m%d_%H%M%S)}"
OUT="${OUT:-/data/cjm/datasets/logs/$RUN_NAME}"

if [[ ! -d "$STAGE1_ROOT" ]]; then
  echo "Stage-1 root not found: $STAGE1_ROOT" >&2
  exit 2
fi

mkdir -p "$OUT/offline" "$OUT/peer"
exec > >(tee -a "$OUT/console.log") 2>&1

echo "============================================================"
echo "Stage-2 evidence validity audit"
echo "DEVELOPMENT VALIDATION ONLY — no OPV2V-W test data"
echo "Stage-1 root: $STAGE1_ROOT"
echo "GPU physical index: $ROCR_VISIBLE_DEVICES"
echo "Output: $OUT"
echo "============================================================"

"$PY" -m unittest qa_evidence_validity.test_offline -v

# 1) CPU/offline audit of the completed Stage-1 target logs.
"$PY" -m qa_evidence_validity.offline \
  --stage1-root "$STAGE1_ROOT" \
  --output-dir "$OUT/offline"

# 2) GPU audit: every weather-induced ego miss is checked with each source alone.
#    The collector still iterates the original deterministic frame queue so the
#    online weather realization matches Stage-1; GPU inference is skipped for
#    frames with no Stage-1 candidate.
for weather in fog rain snow; do
  echo "----- peer-alone audit: $weather -----"
  "$PY" -m qa_evidence_validity.peer_audit \
    --stage1-root "$STAGE1_ROOT" \
    --config "$CONFIG" \
    --frontend-config "$FRONTEND_CONFIG" \
    --frontend-checkpoint "$FRONTEND_CHECKPOINT" \
    --weather "$weather" \
    --output-dir "$OUT/peer/$weather"
done

# 3) Unified evidence-validity report.
"$PY" -m qa_evidence_validity.report \
  --stage1-root "$STAGE1_ROOT" \
  --offline-results "$OUT/offline/offline_results.json" \
  --peer-root "$OUT/peer" \
  --output-dir "$OUT"

echo
echo "DONE"
echo "Main report: $OUT/evidence_validity_report.md"
echo "Machine-readable results: $OUT/evidence_validity_results.json"
echo "Offline sensitivity audit: $OUT/offline/offline_results.json"
echo "Per-target peer-alone results: $OUT/peer/{fog,rain,snow}/peer_targets.jsonl"
echo
echo "Please send back evidence_validity_report.md and evidence_validity_results.json."
