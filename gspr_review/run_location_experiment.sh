#!/usr/bin/env bash
set -euo pipefail
cd /home/cjm/OpenCOOD-main/cjmnet
export PYTHONPATH=/home/cjm/OpenCOOD-main/cjmnet:/home/cjm/OpenCOOD-main
export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0}"
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
export PYTHONHASHSEED=20260909
LOCATION_PYTHON=/home/cjm/miniconda3/envs/opencood/bin/python
"$LOCATION_PYTHON" -m unittest gspr_review.test_bev_location gspr_review.test_counterfactual \
  gspr_review.test_tensors gspr_review.test_diagnostics gspr_review.test_amplitude \
  gspr_communication.test_tensors gspr_review.test_codec gspr_review.test_packet_replay -v
"$LOCATION_PYTHON" -u -m gspr_review.location_experiment "$@"
