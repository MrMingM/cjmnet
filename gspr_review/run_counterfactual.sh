#!/usr/bin/env bash
# Fresh, matched legacy/contrast/contrast+gain runs. Never retrains old checkpoints.
set -euo pipefail
cd /home/cjm/OpenCOOD-main/cjmnet
export PYTHONPATH=/home/cjm/OpenCOOD-main/cjmnet:/home/cjm/OpenCOOD-main
export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0}"
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
export PYTHONHASHSEED=20260909
CF_PYTHON=/home/cjm/miniconda3/envs/opencood/bin/python
CF_FRONTEND_CONFIG="${FRONTEND_CONFIG:-/data/cjm/datasets/logs/gspr_joint_full_v1_seed20260907_20260907_161155/config.yaml}"
CF_FRONTEND_CHECKPOINT="${FRONTEND_CHECKPOINT:-/data/cjm/datasets/logs/gspr_joint_full_v1_seed20260907_20260907_161155/net_best_validation.pth}"
CF_OUTPUT="${CF_OUTPUT:-/data/cjm/datasets/logs/gspr_cf_seed20260909_$(date +%Y%m%d_%H%M%S)}"
test -f "$CF_FRONTEND_CONFIG"
test -f "$CF_FRONTEND_CHECKPOINT"
"$CF_PYTHON" -m unittest gspr_communication.test_codec gspr_communication.test_tensors \
  gspr_review.test_codec gspr_review.test_tensors gspr_review.test_resources \
  gspr_review.test_diagnostics gspr_review.test_counterfactual -v
"$CF_PYTHON" -m gspr_review.prepare_counterfactual --output-dir "$CF_OUTPUT"
# Verify every architecture before any expensive training.
for CF_ARCH in legacy contrast contrast_gain; do
  "$CF_PYTHON" -m gspr_review.verify --config "$CF_OUTPUT/$CF_ARCH.yaml" \
    --frontend-config "$CF_FRONTEND_CONFIG" --frontend-checkpoint "$CF_FRONTEND_CHECKPOINT"
done
for CF_ARCH in legacy contrast contrast_gain; do
  CF_RUN="$CF_OUTPUT/$CF_ARCH"
  "$CF_PYTHON" -u -m gspr_review.train --config "$CF_OUTPUT/$CF_ARCH.yaml" \
    --frontend-config "$CF_FRONTEND_CONFIG" --frontend-checkpoint "$CF_FRONTEND_CHECKPOINT" \
    --run-dir "$CF_RUN" 2>&1 | tee "$CF_OUTPUT/${CF_ARCH}_train.log"
  for CF_BRANCH in processed_lidar processed_lidar_weather; do
    for CF_CONTROL in normal neutral permuted protocol a0b0; do
      CF_MODE=review
      CF_EXTRA=(--reviewer "$CF_RUN/reviewer_best.pth" --evidence-intervention "$CF_CONTROL")
      if [[ "$CF_CONTROL" == protocol || "$CF_CONTROL" == a0b0 ]]; then
        CF_MODE="$CF_CONTROL"
        CF_EXTRA=()
      fi
      "$CF_PYTHON" -u -m gspr_review.evaluate --config "$CF_RUN/experiment.yaml" \
        --frontend-config "$CF_RUN/frontend_config.yaml" --frontend-checkpoint "$CF_FRONTEND_CHECKPOINT" \
        --mode "$CF_MODE" --lidar-key "$CF_BRANCH" "${CF_EXTRA[@]}" --compare-ego \
        --output-dir "${CF_RUN}_${CF_BRANCH}_${CF_CONTROL}" \
        2>&1 | tee "${CF_RUN}_${CF_BRANCH}_${CF_CONTROL}.log"
    done
  done
  "$CF_PYTHON" -u -m gspr_review.diagnose --run-dir "$CF_RUN" --output-dir "${CF_RUN}_diagnostics"
done
"$CF_PYTHON" -m gspr_review.summarize_counterfactual --output-dir "$CF_OUTPUT"
echo "Send back: $CF_OUTPUT/comparison.json and *_diagnostics/summary.json"
