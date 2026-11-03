#!/bin/sh
set -eu
sh -n "$0"
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PYTHON=${PYTHON:-/home/cjm/miniconda3/envs/opencood/bin/python}
BACKEND=${BACKEND:-rocm}
MODE=${MODE:-smoke}
RESUME=${RESUME:-1}
GPUS=${GPUS:-2,4}
OUTPUT_ROOT=${OUTPUT_ROOT:-/home/cjm/datasets/logs/gspr-e1e2-20261101-v5}
TMPDIR=${TMPDIR:-/home/cjm/datasets/tmp/gspr_e1e2}
case "$BACKEND" in rocm) ;; *) echo 'Matched E0 experiment requires ROCm' >&2; exit 2 ;; esac
case "$MODE" in preflight|smoke|all) ;; *) echo 'Invalid MODE' >&2; exit 2 ;; esac
case "$RESUME" in 0|1) ;; *) echo 'Invalid RESUME' >&2; exit 2 ;; esac
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES PYTHONPATH PYTHONHOME
PYTHONHASHSEED=20261016
PYTHONDONTWRITEBYTECODE=1
PYTHONNOUSERSITE=1
PYTHONUNBUFFERED=1
OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1
export BACKEND MODE RESUME GPUS OUTPUT_ROOT TMPDIR PYTHONHASHSEED
export PYTHONDONTWRITEBYTECODE PYTHONNOUSERSITE PYTHONUNBUFFERED OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS
mkdir -p "$TMPDIR"
"$PYTHON" -B "$SCRIPT_DIR/check_static.py"
set -- "$PYTHON" -B "$SCRIPT_DIR/orchestrator.py" --mode "$MODE" --output-root "$OUTPUT_ROOT" --gpus "$GPUS"
if [ "$RESUME" = 1 ]; then set -- "$@" --resume; fi
exec "$@"
