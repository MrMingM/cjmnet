#!/bin/sh
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
: "${GPU:?Set GPU to an explicitly verified idle physical GPU (not 2 or 7)}"
: "${RUN:?Set an independent observability_audit RUN}"
export ROCR_VISIBLE_DEVICES="$GPU"
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
export PYTHONHASHSEED=20261012
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
PYTHON="${PYTHON:-/home/cjm/miniconda3/envs/opencood/bin/python}"
"$PYTHON" -m local_observability_audit.test_core --require-server
exec "$PYTHON" -u -m local_observability_audit.pipeline --run "$RUN" --gpu "$GPU" "$@"
