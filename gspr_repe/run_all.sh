#!/bin/sh
set -eu
sh -n "$0"
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
BACKEND=${BACKEND:-cuda}
MODE=${MODE:-all}
RESUME=${RESUME:-1}
if [ -n "${POINTCLOUD_ONLY_ROOT:-}" ]; then
  if [ "$BACKEND" != cuda ]; then
    echo 'POINTCLOUD_ONLY_ROOT currently supports the independent CUDA entry only.' >&2
    exit 2
  fi
  TRAIN_ROOT=${TRAIN_ROOT:-$POINTCLOUD_ONLY_ROOT/v2xr-official-train-v2/train}
  VIEW_ROOT=${VIEW_ROOT:-$POINTCLOUD_ONLY_ROOT/v2xr-official-validate-v1}
  PAIRED_ROOT=${PAIRED_ROOT:-$POINTCLOUD_ONLY_ROOT/v2xr-lidar-weather-validate-v1}
  CUDA_INPUT_ROOT=${CUDA_INPUT_ROOT:-/data/cjm/datasets/gspr-repe-cu124-pointcloud-inputs-v1}
  OUTPUT_ROOT=${OUTPUT_ROOT:-/data/cjm/datasets/logs/gspr-repe-cu124-pointcloud-v1}
  export POINTCLOUD_ONLY_ROOT VIEW_ROOT PAIRED_ROOT
fi
TRAIN_ROOT=${TRAIN_ROOT:-/data/cjm/datasets/v2xr-official-train-v2/train}
case "$BACKEND" in
  cuda)
    CUDA_ENV_ROOT=${CUDA_ENV_ROOT:-/workspace/cjmnet_cuda_env}
    PYTHON=${PYTHON:-$CUDA_ENV_ROOT/envs/cjmnet-cu124/bin/python}
    OUTPUT_ROOT=${OUTPUT_ROOT:-/data/cjm/datasets/logs/gspr-repe-cu124-v1}
    CUDA_INPUT_ROOT=${CUDA_INPUT_ROOT:-/data/cjm/datasets/gspr-repe-cu124-inputs-v1}
    RUNTIME_ROOT=${RUNTIME_ROOT:-$CUDA_ENV_ROOT/official-runtime-cu124}
    unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
    # Preserve scheduler/container isolation; select a device only on explicit request.
    if [ "${GPU+x}" = x ]; then
      if [ "${CUDA_VISIBLE_DEVICES+x}" = x ]; then
        echo 'CUDA_VISIBLE_DEVICES is already set by this environment; omit GPU to preserve allocation.' >&2
        exit 2
      fi
      case "$GPU" in ''|*[!0-9]*) echo 'GPU must be one device index' >&2; exit 2 ;; esac
      CUDA_VISIBLE_DEVICES=$GPU
      export CUDA_VISIBLE_DEVICES
    fi
    unset PYTHONPATH PYTHONHOME
    PYTHONNOUSERSITE=1
    LD_LIBRARY_PATH="$CUDA_ENV_ROOT/envs/cjmnet-cu124/lib:$CUDA_ENV_ROOT/cuda-12.4/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export CUDA_INPUT_ROOT RUNTIME_ROOT PYTHONNOUSERSITE LD_LIBRARY_PATH
    ;;
  rocm)
    GPU=${GPU:-0}
    PYTHON=${PYTHON:-/home/cjm/miniconda3/envs/opencood/bin/python}
    OUTPUT_ROOT=${OUTPUT_ROOT:-/data/cjm/datasets/logs/gspr-repe-v1}
    case "$GPU" in ''|*[!0-9]*) echo 'GPU must be one physical GPU index' >&2; exit 2 ;; esac
    unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
    ROCR_VISIBLE_DEVICES=$GPU
    export ROCR_VISIBLE_DEVICES
    ;;
  *) echo 'BACKEND must be cuda or rocm' >&2; exit 2 ;;
esac
case "$MODE" in all|preflight|smoke) ;; *) echo "Invalid MODE: $MODE" >&2; exit 2 ;; esac
case "$RESUME" in 0|1) ;; *) echo 'RESUME must be 0 or 1' >&2; exit 2 ;; esac
PYTHONHASHSEED=20261016
TMPDIR=${TMPDIR:-/data/cjm/datasets/tmp/gspr_repe_$BACKEND}
PYTHONDONTWRITEBYTECODE=1
PYTHONUNBUFFERED=1
OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export BACKEND PYTHONHASHSEED TMPDIR PYTHONDONTWRITEBYTECODE PYTHONUNBUFFERED
export OUTPUT_ROOT TRAIN_ROOT OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS
mkdir -p "$TMPDIR"
"$PYTHON" -B "$SCRIPT_DIR/check_static.py"
set -- "$PYTHON" -B "$SCRIPT_DIR/orchestrator.py" --mode "$MODE" --output-root "$OUTPUT_ROOT" --train-root "$TRAIN_ROOT"
if [ "$RESUME" = 1 ]; then set -- "$@" --resume; fi
exec "$@"
