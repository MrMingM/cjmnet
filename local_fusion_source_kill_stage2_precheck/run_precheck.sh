#!/bin/sh
set -eu
cd /home/cjm/OpenCOOD-main/cjmnet
export PYTHONPATH=/home/cjm/OpenCOOD-main/cjmnet:/home/cjm/OpenCOOD-main
export ROCR_VISIBLE_DEVICES=""
export HIP_VISIBLE_DEVICES=""
export CUDA_VISIBLE_DEVICES=""
PY="${PY:-/home/cjm/miniconda3/envs/opencood/bin/python}"
RUN="${RUN:-/data/cjm/datasets/logs/source_kill_stage2_precheck_$(date +%Y%m%d_%H%M%S)}"
export PY RUN
"$PY" -m local_fusion_source_kill_stage2_precheck.driver --static
"$PY" -m unittest local_fusion_source_kill_stage2_precheck.test_core
"$PY" -m local_fusion_source_kill_stage2_precheck.test_core --smoke
exec "$PY" -u -m local_fusion_source_kill_stage2_precheck.driver --output "$RUN"
