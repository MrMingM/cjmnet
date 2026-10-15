#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT"
PY="${PY:-/home/cjm/miniconda3/envs/opencood/bin/python}"
export PYTHONPATH="$ROOT:$(dirname -- "$ROOT")"
export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-2}"
unset HIP_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES
sh local_fusion_source_kill_stage2/run_preflight.sh
exec "$PY" -m local_fusion_source_kill_stage2.driver --run "${RUN:?set RUN}" --component both
