#!/bin/sh
# Server only: prepare the slim copy, then compare sampled official CPU batches.
set -eu
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PYTHON=${PYTHON:-/home/cjm/miniconda3/envs/opencood/bin/python}
DATA_BASE=${DATA_BASE:-/data/cjm/datasets}
SLIM_ROOT=${SLIM_ROOT:-$DATA_BASE/v2xr-pointcloud-only-v1}
LOG_ROOT=${LOG_ROOT:-$DATA_BASE/logs/v2xr-pointcloud-only-v1}
TMPDIR=${TMPDIR:-$DATA_BASE/tmp/v2xr_pointcloud_only_v1}
RUNTIME_ROOT=${RUNTIME_ROOT:-$DATA_BASE/v2xr-official-runtime-v1}
PYTHONHASHSEED=20261016
PYTHONDONTWRITEBYTECODE=1
export TMPDIR PYTHONHASHSEED PYTHONDONTWRITEBYTECODE
if [ -e "$SLIM_ROOT" ]; then
  echo "Output exists; retain it and inspect before choosing a new path: $SLIM_ROOT" >&2
  exit 2
fi
mkdir -p "$LOG_ROOT" "$TMPDIR"
echo "$$" > "$LOG_ROOT/prepare.pid"
trap 'result=$?; echo "EXIT=$result"; printf "%s\n" "$result" > "$LOG_ROOT/exit_code"' 0
echo "START prepare -> $LOG_ROOT/prepare.log"
"$PYTHON" -u "$SCRIPT_DIR/slim_data.py" prepare --base "$DATA_BASE" --output "$SLIM_ROOT" \
  > "$LOG_ROOT/prepare.log" 2>&1
echo "START equivalence -> $LOG_ROOT/equivalence.log"
"$PYTHON" -u "$SCRIPT_DIR/check_equivalence.py" --slim-root "$SLIM_ROOT" \
  --runtime "$RUNTIME_ROOT" --frames 3 > "$LOG_ROOT/equivalence.log" 2>&1
echo "SLIM_PIPELINE_COMPLETE: $SLIM_ROOT"
