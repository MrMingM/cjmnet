#!/bin/sh
set -eu
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
sh -n "$HERE/run_server.sh"
PY=${PYTHON:-/home/cjm/miniconda3/envs/opencood/bin/python}
export TMPDIR=${TMPDIR:-/data/cjm/datasets/tmp/v2xr_official_gt_audit}
mkdir -p "$TMPDIR"
unset HIP_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES
unset ROCR_VISIBLE_DEVICES
export PYTHONHASHSEED=20261016
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
# Native CPU ROI and official CPU voxelization; no detector or GPU allocation.
"$PY" "$HERE/package_check.py"
"$PY" "$HERE/test_reports.py"
set -- "$PY" -u "$HERE/main.py" \
  --runtime-root "${RUNTIME_ROOT:-/data/cjm/datasets/v2xr-official-runtime-v1}" \
  --model-dir "${MODEL_DIR:-/data/cjm/datasets/logs/v2xr-official-attfuse-model-v1}" \
  --view-root "${VIEW_ROOT:-/data/cjm/datasets/v2xr-official-validate-v1}" \
  --paired-root "${PAIRED_ROOT:-/data/cjm/datasets/v2xr-lidar-weather-validate-v1}" \
  --d1a-root "${D1A_ROOT:-/data/cjm/datasets/logs/v2xr-official-attfuse-clean-repro-v1}" \
  --output-root "${OUTPUT_ROOT:-/data/cjm/datasets/logs/v2xr-official-gt-audit-v1}" \
  --limit-frames "${LIMIT_FRAMES:-0}"
if [ "${RESUME:-0}" = 1 ]; then set -- "$@" --resume; fi
exec "$@"
