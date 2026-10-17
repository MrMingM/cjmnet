#!/bin/sh
set -eu
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
sh -n "$HERE/run_server.sh"
PY=${PYTHON:-/home/cjm/miniconda3/envs/opencood/bin/python}
export TMPDIR=${TMPDIR:-/data/cjm/datasets/tmp/v2xr_protocol_audit}
mkdir -p "$TMPDIR"
unset HIP_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES
case ${STAGE:-all} in
  all)
    : "${GPU:?Set GPU to one available physical HIP GPU}"
    ;;
  evaluate) ;;
  *) echo 'STAGE must be all or evaluate' >&2; exit 2 ;;
esac
if [ -n "${GPU:-}" ]; then
  case "$GPU" in *[!0-9]*) echo 'GPU must be one physical card number' >&2; exit 2 ;; esac
  export ROCR_VISIBLE_DEVICES="$GPU"
else
  unset ROCR_VISIBLE_DEVICES
fi
export PYTHONHASHSEED=20261016
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
set -- "$PY" -u "$HERE/main.py" --stage "${STAGE:-all}" \
  --output-root "${OUTPUT_ROOT:-/data/cjm/datasets/logs/v2xr-dminus1c-protocol-v1}" \
  --runtime-root "${RUNTIME_ROOT:-/data/cjm/datasets/v2xr-official-runtime-v1}" \
  --model-dir "${MODEL_DIR:-/data/cjm/datasets/logs/v2xr-official-attfuse-model-v1}" \
  --d1a-root "${D1A_ROOT:-/data/cjm/datasets/logs/v2xr-official-attfuse-clean-repro-v1}" \
  --gt-audit-root "${GT_AUDIT_ROOT:-/data/cjm/datasets/logs/v2xr-official-gt-audit-v1}" \
  --view-root "${VIEW_ROOT:-/data/cjm/datasets/v2xr-official-validate-v1}" \
  --limit-frames "${LIMIT_FRAMES:-0}"
if [ "${RESUME:-0}" = 1 ]; then set -- "$@" --resume; fi
exec "$@"
