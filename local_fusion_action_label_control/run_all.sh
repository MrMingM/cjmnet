#!/bin/sh
set -eu

cd /home/cjm/OpenCOOD-main/cjmnet
export PYTHONPATH=/home/cjm/OpenCOOD-main/cjmnet:/home/cjm/OpenCOOD-main
export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-2}"
unset HIP_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES
export PYTHONHASHSEED=20261007
PY="${PY:-/home/cjm/miniconda3/envs/opencood/bin/python}"
: "${SOURCE_RUN:?Set SOURCE_RUN explicitly to the original action utility run}"
RUN="${RUN:-/data/cjm/datasets/logs/action_label_control_$(date +%Y%m%d_%H%M%S)}"
MODE="${MODE:-offline}"
export RUN
exec "$PY" -u -B -m local_fusion_action_label_control.pipeline \
  --source-run "$SOURCE_RUN" --run "$RUN" --mode "$MODE" "$@"
