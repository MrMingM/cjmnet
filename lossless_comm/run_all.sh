#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/cjm/OpenCOOD-main}"
CJ="${CJ:-${ROOT}/cjmnet}"
PY="${PY:-/home/cjm/miniconda3/envs/opencood/bin/python}"
PHASE="${PHASE:-development}"
GPU="${GPU:-0}"
BACKEND="${BACKEND:-auto}"
LEVEL="${LEVEL:-3}"

FRONTEND_RUN="${FRONTEND_RUN:-/data/cjm/datasets/logs/gspr_joint_full_v1_seed20260907_20260907_161155}"
FRONTEND_CONFIG="${FRONTEND_CONFIG:-${FRONTEND_RUN}/config.yaml}"
FRONTEND_CKPT="${FRONTEND_CKPT:-${FRONTEND_RUN}/net_best_validation.pth}"
EXPERIMENT="${EXPERIMENT:-${CJ}/gspr_evidence/experiment.yaml}"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${OUT:-/data/cjm/datasets/logs/lossless_comm_${PHASE}_${STAMP}}"
mkdir -p "${OUT}"

echo "phase=${PHASE}"
echo "output=${OUT}"
echo "GPU/HCU physical index=${GPU}"
echo "backend=${BACKEND}, level=${LEVEL}"

for WEATHER in clean fog rain snow; do
  mkdir -p "${OUT}/${WEATHER}"
  echo "===== ${WEATHER} ====="
  ROCR_VISIBLE_DEVICES="${GPU}" \
  PYTHONPATH="${ROOT}:${CJ}" \
  "${PY}" -m lossless_comm.benchmark \
    --config "${EXPERIMENT}" \
    --frontend-config "${FRONTEND_CONFIG}" \
    --frontend-checkpoint "${FRONTEND_CKPT}" \
    --phase "${PHASE}" \
    --weather "${WEATHER}" \
    --backend "${BACKEND}" \
    --compression-level "${LEVEL}" \
    --output-dir "${OUT}/${WEATHER}" \
    2>&1 | tee "${OUT}/${WEATHER}/run.log"
done

PYTHONPATH="${ROOT}:${CJ}" "${PY}" -m lossless_comm.summarize --root "${OUT}"
echo "Done: ${OUT}/results.md"
