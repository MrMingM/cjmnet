#!/bin/sh
set -eu
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
sh -n "$HERE/run_server.sh"
PY=${PYTHON:-/home/cjm/miniconda3/envs/opencood/bin/python}
export TMPDIR=${TMPDIR:-/data/cjm/datasets/tmp/v2xr_official_selected_eval}
mkdir -p "$TMPDIR"
unset HIP_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=20261016
ACTION=${ACTION:-export}
case "$ACTION" in
  export) unset ROCR_VISIBLE_DEVICES ;;
  rerun)
    : "${GPU:?Set GPU to one available physical HIP card number}"
    case "$GPU" in ''|*[!0-9]*) echo 'GPU must be one physical card number' >&2; exit 2 ;; esac
    export ROCR_VISIBLE_DEVICES="$GPU"
    ;;
  *) echo 'ACTION must be export or rerun' >&2; exit 2 ;;
esac
set -- "$PY" -B -u "$HERE/run.py" --action "$ACTION" \
  --training-root "${TRAINING_ROOT:-/data/cjm/datasets/logs/v2xr-clean-attfuse-train-v1}" \
  --output-root "${REPORT_ROOT:-/data/cjm/datasets/logs/v2xr-attfuse-official-protocol-v1}" \
  --limit-frames "${LIMIT_FRAMES:-0}"
if [ -n "${EVALUATION_ROOT:-}" ]; then set -- "$@" --evaluation-root "$EVALUATION_ROOT"; fi
if [ "${RESUME:-0}" = 1 ]; then set -- "$@" --resume; fi
exec "$@"
