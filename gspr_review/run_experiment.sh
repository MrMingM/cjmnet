#!/usr/bin/env bash
set -euo pipefail
cd /home/cjm/OpenCOOD-main/cjmnet
export PYTHONPATH=/home/cjm/OpenCOOD-main/cjmnet:/home/cjm/OpenCOOD-main
export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0}"
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
export PYTHONHASHSEED=20260909
REVIEW_PYTHON=/home/cjm/miniconda3/envs/opencood/bin/python
REVIEW_FRONTEND_CONFIG="${FRONTEND_CONFIG:-/data/cjm/datasets/logs/gspr_joint_full_v1_seed20260907_20260907_161155/config.yaml}"
REVIEW_FRONTEND_CHECKPOINT="${FRONTEND_CHECKPOINT:-/data/cjm/datasets/logs/gspr_joint_full_v1_seed20260907_20260907_161155/net_best_validation.pth}"
REVIEW_RUN="${REVIEW_RUN:-/data/cjm/datasets/logs/gspr_review_seed20260909_$(date +%Y%m%d_%H%M%S)}"
test -f "$REVIEW_FRONTEND_CONFIG"
test -f "$REVIEW_FRONTEND_CHECKPOINT"
test ! -e "$REVIEW_RUN"
echo "Review experiment: $REVIEW_RUN"
"$REVIEW_PYTHON" -m unittest gspr_communication.test_codec gspr_communication.test_tensors gspr_review.test_codec gspr_review.test_tensors gspr_review.test_resources -v
"$REVIEW_PYTHON" -m gspr_review.verify --frontend-config "$REVIEW_FRONTEND_CONFIG" --frontend-checkpoint "$REVIEW_FRONTEND_CHECKPOINT"
"$REVIEW_PYTHON" -u -m gspr_review.train --frontend-config "$REVIEW_FRONTEND_CONFIG" \
  --frontend-checkpoint "$REVIEW_FRONTEND_CHECKPOINT" --run-dir "$REVIEW_RUN" 2>&1 | tee "${REVIEW_RUN}_train.log"
for REVIEW_BRANCH in processed_lidar processed_lidar_weather; do
  for REVIEW_MODE in a0b0 protocol review shuffled; do
    REVIEW_EXTRA=()
    if [[ "$REVIEW_MODE" == review || "$REVIEW_MODE" == shuffled ]]; then
      REVIEW_EXTRA=(--reviewer "$REVIEW_RUN/reviewer_best.pth")
    fi
    "$REVIEW_PYTHON" -u -m gspr_review.evaluate --config "$REVIEW_RUN/experiment.yaml" \
      --frontend-config "$REVIEW_RUN/frontend_config.yaml" --frontend-checkpoint "$REVIEW_FRONTEND_CHECKPOINT" \
      --mode "$REVIEW_MODE" --lidar-key "$REVIEW_BRANCH" "${REVIEW_EXTRA[@]}" --compare-ego \
      --output-dir "${REVIEW_RUN}_${REVIEW_BRANCH}_${REVIEW_MODE}" \
      2>&1 | tee "${REVIEW_RUN}_${REVIEW_BRANCH}_${REVIEW_MODE}.log"
  done
done
"$REVIEW_PYTHON" -m gspr_review.summarize --run-dir "$REVIEW_RUN"
