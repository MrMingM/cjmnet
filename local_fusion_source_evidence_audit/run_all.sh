#!/bin/sh
set -eu

cd /home/cjm/OpenCOOD-main/cjmnet
export PYTHONPATH=/home/cjm/OpenCOOD-main/cjmnet:/home/cjm/OpenCOOD-main
export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-2}"
unset HIP_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES
export PYTHONHASHSEED=20261007
PY="${PY:-/home/cjm/miniconda3/envs/opencood/bin/python}"
RUN="${RUN:-/data/cjm/datasets/logs/source_evidence_audit_$(date +%Y%m%d_%H%M%S)}"
BASE_RUN="${BASE_RUN:-/data/cjm/datasets/logs/action_utility_audit_20261007_210243}"
export RUN BASE_RUN
"$PY" -c 'import sys; from local_fusion_source_evidence_audit.common import external_run; external_run(sys.argv[1], True)' "$RUN"
exec "$PY" -u -m local_fusion_source_evidence_audit.pipeline --run "$RUN" --base-run "$BASE_RUN" "$@" >> "$RUN/driver.log" 2>&1
