#!/usr/bin/env bash
set -euo pipefail

cd /home/cjm/OpenCOOD-main/cjmnet
export PYTHONPATH=/home/cjm/OpenCOOD-main/cjmnet:/home/cjm/OpenCOOD-main
GPU="${GPU:-0}"
export ROCR_VISIBLE_DEVICES="${GPU}"
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
export PYTHONHASHSEED="${PYTHONHASHSEED:-20260917}"

PY="${PY:-/home/cjm/miniconda3/envs/opencood/bin/python}"
: "${STAGE3A_ROOT:?Set STAGE3A_ROOT to a completed Stage-3A output directory first}"
if [[ ! -f "${STAGE3A_ROOT}/stage3a_report.md" ]]; then
  echo "Completed Stage-3A report not found: ${STAGE3A_ROOT}/stage3a_report.md" >&2
  exit 2
fi

if [[ -z "${STAGE2_ROOT:-}" ]]; then
  STAGE2_ROOT="$("$PY" - "${STAGE3A_ROOT}/fog/protocol.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["stage2_root"])
PY
)"
fi
if [[ -z "${STAGE1_ROOT:-}" ]]; then
  STAGE1_ROOT="$("$PY" - "${STAGE3A_ROOT}/fog/protocol.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["stage1_root"])
PY
)"
fi

FRONTEND_CONFIG="${FRONTEND_CONFIG:-/data/cjm/datasets/logs/gspr_joint_full_v1_seed20260907_20260907_161155/config.yaml}"
FRONTEND_CHECKPOINT="${FRONTEND_CHECKPOINT:-/data/cjm/datasets/logs/gspr_joint_full_v1_seed20260907_20260907_161155/net_best_validation.pth}"
CONFIG="${CONFIG:-qa_observation_diagnostic/experiment.yaml}"
MAX_CANDIDATE_FRAMES="${MAX_CANDIDATE_FRAMES:-0}"
MAX_PEERS="${MAX_PEERS:-4}"
RUN_NAME="${RUN_NAME:-qa_stage3b_$(date +%Y%m%d_%H%M%S)}"
OUT="${OUT:-/data/cjm/datasets/logs/${RUN_NAME}}"

# Only create the run root here. stage3b.py creates each weather directory via
# rt.new_output(exist_ok=False), preventing accidental overwrite.
mkdir -p "${OUT}"
exec > >(tee -a "${OUT}/console.log") 2>&1

echo "============================================================"
echo "Stage-3B limited whole-agent source-subset Oracle"
echo "RUN ONLY AFTER REVIEWING STAGE-3A"
echo "DEVELOPMENT CANDIDATE FRAMES ONLY — no OPV2V-W test"
echo "GPU/HCU physical index: ${GPU}"
echo "Stage-3A: ${STAGE3A_ROOT}"
echo "Stage-2:  ${STAGE2_ROOT}"
echo "Stage-1:  ${STAGE1_ROOT}"
echo "Output:   ${OUT}"
echo "MAX_CANDIDATE_FRAMES=${MAX_CANDIDATE_FRAMES} (0 = full)"
echo "MAX_PEERS=${MAX_PEERS}"
echo "============================================================"

"$PY" -m unittest gspr_evidence.test_stage3 -v

for weather in fog rain snow; do
  echo "----- Stage-3B: ${weather} -----"
  "$PY" -m gspr_evidence.stage3b \
    --stage1-root "${STAGE1_ROOT}" \
    --stage2-root "${STAGE2_ROOT}" \
    --config "${CONFIG}" \
    --frontend-config "${FRONTEND_CONFIG}" \
    --frontend-checkpoint "${FRONTEND_CHECKPOINT}" \
    --weather "${weather}" \
    --output-dir "${OUT}/${weather}" \
    --max-candidate-frames "${MAX_CANDIDATE_FRAMES}" \
    --max-peers "${MAX_PEERS}"
done

"$PY" -m gspr_evidence.stage3_report --phase b --root "${OUT}"

echo
echo "DONE: Stage-3B"
echo "Main report: ${OUT}/stage3b_report.md"
echo "Machine report: ${OUT}/stage3b_report.json"
echo "Per-frame subset results: ${OUT}/{fog,rain,snow}/frames.jsonl"
