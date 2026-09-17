#!/usr/bin/env bash
set -euo pipefail

cd /home/cjm/OpenCOOD-main/cjmnet
export PYTHONPATH=/home/cjm/OpenCOOD-main/cjmnet:/home/cjm/OpenCOOD-main
GPU="${GPU:-0}"
export ROCR_VISIBLE_DEVICES="${GPU}"
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
export PYTHONHASHSEED="${PYTHONHASHSEED:-20260917}"

PY="${PY:-/home/cjm/miniconda3/envs/opencood/bin/python}"
STAGE2_ROOT="${STAGE2_ROOT:-/data/cjm/datasets/logs/qa_evidence_validity_20260917_112301}"
FRONTEND_CONFIG="${FRONTEND_CONFIG:-/data/cjm/datasets/logs/gspr_joint_full_v1_seed20260907_20260907_161155/config.yaml}"
FRONTEND_CHECKPOINT="${FRONTEND_CHECKPOINT:-/data/cjm/datasets/logs/gspr_joint_full_v1_seed20260907_20260907_161155/net_best_validation.pth}"
CONFIG="${CONFIG:-qa_observation_diagnostic/experiment.yaml}"
MAX_CANDIDATE_FRAMES="${MAX_CANDIDATE_FRAMES:-0}"
RUN_NAME="${RUN_NAME:-qa_stage3a_$(date +%Y%m%d_%H%M%S)}"
OUT="${OUT:-/data/cjm/datasets/logs/${RUN_NAME}}"

if [[ ! -f "${STAGE2_ROOT}/peer/fog/protocol.json" ]]; then
  echo "Stage-2 protocol not found: ${STAGE2_ROOT}/peer/fog/protocol.json" >&2
  exit 2
fi

if [[ -z "${STAGE1_ROOT:-}" ]]; then
  STAGE1_ROOT="$("$PY" - "${STAGE2_ROOT}/peer/fog/protocol.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["stage1_root"])
PY
)"
fi

mkdir -p "${OUT}/stratification" "${OUT}/fog" "${OUT}/rain" "${OUT}/snow"
exec > >(tee -a "${OUT}/console.log") 2>&1

echo "============================================================"
echo "Stage-3A H-A6 disappearance audit"
echo "DEVELOPMENT VALIDATION ONLY — no OPV2V-W test"
echo "GPU/HCU physical index: ${GPU}"
echo "Stage-1: ${STAGE1_ROOT}"
echo "Stage-2: ${STAGE2_ROOT}"
echo "Output:  ${OUT}"
echo "MAX_CANDIDATE_FRAMES=${MAX_CANDIDATE_FRAMES} (0 = full)"
echo "============================================================"

"$PY" -m unittest gspr_evidence.test_stage3 -v

"$PY" -m gspr_evidence.stage3_stratify \
  --stage1-root "${STAGE1_ROOT}" \
  --stage2-root "${STAGE2_ROOT}" \
  --output-dir "${OUT}/stratification"

for weather in fog rain snow; do
  echo "----- Stage-3A: ${weather} -----"
  "$PY" -m gspr_evidence.stage3a \
    --stage1-root "${STAGE1_ROOT}" \
    --stage2-root "${STAGE2_ROOT}" \
    --config "${CONFIG}" \
    --frontend-config "${FRONTEND_CONFIG}" \
    --frontend-checkpoint "${FRONTEND_CHECKPOINT}" \
    --weather "${weather}" \
    --output-dir "${OUT}/${weather}" \
    --max-candidate-frames "${MAX_CANDIDATE_FRAMES}"
done

"$PY" -m gspr_evidence.stage3_report --phase a --root "${OUT}"

echo
echo "DONE: Stage-3A"
echo "Main report: ${OUT}/stage3a_report.md"
echo "Machine report: ${OUT}/stage3a_report.json"
echo "Stratification: ${OUT}/stratification/stage2_5_stratification.md"
echo "Per-target traces: ${OUT}/{fog,rain,snow}/targets.jsonl"
echo
echo "Do NOT launch Stage-3B automatically. Inspect Stage-3A first."
