#!/bin/sh
set -eu
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
sh -n "$HERE/run_server.sh"
MODE=${MODE:-train}
if [ "$MODE" = all ]; then
  if [ "${SMOKE:-0}" != 0 ] || [ "${LIMIT_FRAMES:-0}" != 0 ]; then
    echo 'MODE=all requires full training and full validation' >&2
    exit 2
  fi
  env MODE=train sh "$HERE/run_server.sh"
  env MODE=final sh "$HERE/run_server.sh"
  exit 0
fi
PY=${PYTHON:-/home/cjm/miniconda3/envs/opencood/bin/python}
export TMPDIR=${TMPDIR:-/data/cjm/datasets/tmp/v2xr_gspr_clean}
mkdir -p "$TMPDIR"
unset HIP_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES
case "$MODE" in
  select) unset ROCR_VISIBLE_DEVICES ;;
  self-test|train|clean-ap|final)
    : "${GPU:?Set GPU to one available physical HIP card number}"
    case "$GPU" in ''|*[!0-9]*) echo 'GPU must be one physical card number' >&2; exit 2 ;; esac
    export ROCR_VISIBLE_DEVICES="$GPU"
    ;;
  *) echo 'MODE must be self-test/train/clean-ap/select/final/all' >&2; exit 2 ;;
esac
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=20261016
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
set -- "$PY" -B -u "$HERE/main.py" "$MODE" \
  --data-root "${DATA_ROOT:-/data/cjm/datasets/V2X-R/train}" \
  --train-root "${TRAIN_ROOT:-/data/cjm/datasets/v2xr-official-train-v2}" \
  --output-root "${OUTPUT_ROOT:-/data/cjm/datasets/logs/v2xr-gspr-clean-train-v2}" \
  --baseline-root "${BASELINE_ROOT:-/data/cjm/datasets/logs/v2xr-clean-attfuse-train-v1}" \
  --runtime-root "${RUNTIME_ROOT:-/data/cjm/datasets/v2xr-official-runtime-v1}" \
  --model-dir "${MODEL_DIR:-/data/cjm/datasets/logs/v2xr-official-attfuse-model-v1}" \
  --d1a-root "${D1A_ROOT:-/data/cjm/datasets/logs/v2xr-official-attfuse-clean-repro-v1}" \
  --gt-audit-root "${GT_AUDIT_ROOT:-/data/cjm/datasets/logs/v2xr-official-gt-audit-v1}" \
  --d1c-root "${D1C_ROOT:-/data/cjm/datasets/logs/v2xr-dminus1c-protocol-v1}" \
  --view-root "${VIEW_ROOT:-/data/cjm/datasets/v2xr-official-validate-v1}" \
  --seven-zip "${SEVEN_ZIP:-/data/cjm/tools/7zip-26.04/bin/7zz}" \
  --limit-frames "${LIMIT_FRAMES:-0}"
if [ "${RESUME:-0}" = 1 ]; then set -- "$@" --resume; fi
if [ "${SMOKE:-0}" = 1 ]; then set -- "$@" --smoke; fi
if [ "$MODE" = clean-ap ]; then
  : "${CHECKPOINT:?Set CHECKPOINT}"
  : "${EVALUATION_ROOT:?Set EVALUATION_ROOT}"
  set -- "$@" --checkpoint "$CHECKPOINT" --evaluation-root "$EVALUATION_ROOT"
fi
exec "$@"
