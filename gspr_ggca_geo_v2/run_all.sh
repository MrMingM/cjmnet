#!/bin/sh
set -eu
sh -n "$0"
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
BACKEND=${BACKEND:-rocm}
MODE=${MODE:-all}
RESUME=${RESUME:-1}
# Same interpreter/device convention as gspr_repe/run_all.sh, not a new environment.
PYTHON=${PYTHON:-/home/cjm/miniconda3/envs/opencood/bin/python}
GPU=${GPU:-2}
GGCA_TEST_GPU=${GGCA_TEST_GPU:-1}
A0_ROOT=${A0_ROOT:-/data/cjm/datasets/logs/gspr-repe-rocm-20261025-v4/B0}
OUTPUT_ROOT=${OUTPUT_ROOT:-/home/cjm/datasets/logs/gspr-ggca-geometry-v2-20261028-r2}
TMPDIR=${TMPDIR:-/home/cjm/datasets/tmp/gspr_ggca}
case "$BACKEND" in rocm) ;; *) echo 'GGCA requires BACKEND=rocm to match the designated A0' >&2; exit 2 ;; esac
case "$MODE" in preflight|smoke|all) ;; *) echo 'Invalid MODE' >&2; exit 2 ;; esac
case "$RESUME" in 0|1) ;; *) echo 'Invalid RESUME' >&2; exit 2 ;; esac
case "$GPU" in ''|*[!0-9]*) echo 'GPU must be one physical device index' >&2; exit 2 ;; esac
ROCR_VISIBLE_DEVICES=$GPU
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES POINTCLOUD_ONLY_ROOT PYTHONPATH PYTHONHOME
PYTHONHASHSEED=20261016
PYTHONDONTWRITEBYTECODE=1
PYTHONNOUSERSITE=1
PYTHONUNBUFFERED=1
OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export BACKEND MODE RESUME PYTHON A0_ROOT OUTPUT_ROOT TMPDIR ROCR_VISIBLE_DEVICES
export GGCA_TEST_GPU
export PYTHONHASHSEED PYTHONDONTWRITEBYTECODE PYTHONNOUSERSITE PYTHONUNBUFFERED
export OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS
mkdir -p "$TMPDIR"
"$PYTHON" -B "$SCRIPT_DIR/check_static.py"
"$PYTHON" -B "$SCRIPT_DIR/test_static.py"
"$PYTHON" -B "$SCRIPT_DIR/test_baseline.py"
"$PYTHON" -B "$SCRIPT_DIR/test_coordination_server.py"
set -- "$PYTHON" -B "$SCRIPT_DIR/coordinator.py" --mode "$MODE" --output-root "$OUTPUT_ROOT" --a0-root "$A0_ROOT"
if [ "$RESUME" = 1 ]; then set -- "$@" --resume; fi
exec "$@"
