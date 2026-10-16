#!/bin/sh
set -eu

HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
sh -n "$HERE/run_server.sh"
PY=${PYTHON:-/home/cjm/miniconda3/envs/opencood/bin/python}
export TMPDIR=${TMPDIR:-/data/cjm/datasets/tmp/v2xr_official_repro}
mkdir -p "$TMPDIR"
unset HIP_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES
case ${STAGE:-evaluate} in
  evaluate|build)
    : "${GPU:?Set GPU to one available physical HIP card}"
    case "$GPU" in ''|*[!0-9]*) echo 'GPU must be one physical card number' >&2; exit 2;; esac
    export ROCR_VISIBLE_DEVICES="$GPU"
    ;;
  prepare) unset ROCR_VISIBLE_DEVICES ;;
  *) echo 'STAGE must be prepare, build, or evaluate' >&2; exit 2;;
esac
export PYTHONHASHSEED=20261016
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export PATH="/data/cjm/tools/7zip-26.04/bin:$PATH"
export PYTHONDONTWRITEBYTECODE=1

"$PY" "$HERE/test_contract.py"

set -- "$PY" -u "$HERE/run.py" --stage "${STAGE:-evaluate}" \
  --data-root "${DATA_ROOT:-/data/cjm/datasets/V2X-R}" \
  --paired-root "${PAIRED_ROOT:-/data/cjm/datasets/v2xr-lidar-weather-validate-v1}" \
  --view-root "${VIEW_ROOT:-/data/cjm/datasets/v2xr-official-validate-v1}" \
  --runtime-root "${RUNTIME_ROOT:-/data/cjm/datasets/v2xr-official-runtime-v1}" \
  --model-dir "${MODEL_DIR:-/data/cjm/datasets/logs/v2xr-official-attfuse-model-v1}" \
  --output-root "${OUTPUT_ROOT:-/data/cjm/datasets/logs/v2xr-official-attfuse-clean-repro-v1}" \
  --limit-frames "${LIMIT_FRAMES:-0}" --loader-workers "${LOADER_WORKERS:-16}"
if [ -n "${CHECKPOINT:-}" ]; then set -- "$@" --checkpoint "$CHECKPOINT"; fi
if [ -n "${YAML:-}" ]; then set -- "$@" --yaml "$YAML"; fi
if [ -n "${SEVEN_ZIP:-}" ]; then set -- "$@" --seven-zip "$SEVEN_ZIP"; fi
if [ "${RESUME:-0}" = 1 ]; then set -- "$@" --resume; fi
exec "$@"
