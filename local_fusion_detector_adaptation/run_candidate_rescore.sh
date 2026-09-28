#!/bin/sh
set -eu

cd /home/cjm/OpenCOOD-main/cjmnet
export PYTHONPATH=/home/cjm/OpenCOOD-main/cjmnet:/home/cjm/OpenCOOD-main
export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0}"
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
export PYTHONHASHSEED=20260927

PY="${PY:-/home/cjm/miniconda3/envs/opencood/bin/python}"
FRONTEND_ROOT="${FRONTEND_ROOT:-/data/cjm/datasets/logs/gspr_joint_full_v1_seed20260907_20260907_161155}"
FRONTEND_CONFIG="${FRONTEND_CONFIG:-$FRONTEND_ROOT/config.yaml}"
FRONTEND_CHECKPOINT="${FRONTEND_CHECKPOINT:-$FRONTEND_ROOT/net_best_validation.pth}"
V3_RUN="${V3_RUN:-/data/cjm/datasets/logs/local_fusion_v3_20260922_182832}"
RUN="${RUN:-/data/cjm/datasets/logs/fusion_detector_adaptation_20260927_103227}"
OUT="${OUT:-$RUN/candidate_rescore_$(date +%Y%m%d_%H%M%S)}"

require_file() {
  if [ ! -f "$1" ]; then
    printf 'Missing required file: %s\n' "$1" >&2
    exit 1
  fi
}
require_file "$RUN/protocol.json"
require_file "$RUN/decision_results.json"
require_file "$RUN/F.pth"
require_file "$RUN/F+D.pth"
require_file "$FRONTEND_CONFIG"
require_file "$FRONTEND_CHECKPOINT"
require_file "$V3_RUN/residual/best.pth"
if [ ! -x "$PY" ]; then
  printf 'Python executable missing or not executable: %s\n' "$PY" >&2
  exit 1
fi
if [ -e "$OUT" ]; then
  printf 'Output path already exists: %s\n' "$OUT" >&2
  exit 1
fi
printf 'Inputs verified. RUN=%s OUT=%s\n' "$RUN" "$OUT"
if [ "${CHECK_ONLY:-0}" = 1 ]; then
  exit 0
fi

"$PY" -m unittest local_fusion_detector_adaptation.test_candidate_rescore -v
exec "$PY" -u -m local_fusion_detector_adaptation.candidate_rescore \
  --run "$RUN" \
  --v3-config "${V3_CONFIG:-local_fusion_v3/experiment.yaml}" \
  --frontend-config "$FRONTEND_CONFIG" \
  --frontend-checkpoint "$FRONTEND_CHECKPOINT" \
  --v3-checkpoint "$V3_RUN/residual/best.pth" \
  --output "$OUT"
