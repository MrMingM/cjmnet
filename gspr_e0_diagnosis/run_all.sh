#!/bin/sh
# Single serial E0 launcher. POSIX sh; never trains or reads formal test.
set -eu
HERE=$(CDPATH= cd -P "$(dirname "$0")" && pwd)
sh -n "$HERE/run_all.sh"
PYTHON=${PYTHON:-/home/cjm/miniconda3/envs/opencood/bin/python}
OUTPUT_ROOT=${OUTPUT_ROOT:-/data/cjm/datasets/logs/gspr-e0-diagnosis-v1}
ORIGINAL_ROOT=${ORIGINAL_ROOT:-/data/cjm/datasets/logs/v2xr-weather-l012-v1}
: "${PHYSICAL_GPU:?Set PHYSICAL_GPU to an available physical ROCm GPU number}"
export ROCR_VISIBLE_DEVICES="$PHYSICAL_GPU"
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
export PYTHONHASHSEED=20261016
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR=${TMPDIR:-/data/cjm/datasets/tmp/gspr_e0_diagnosis}
mkdir -p "$TMPDIR"
"$PYTHON" -B "$HERE/check_static.py"
"$PYTHON" -B "$HERE/e0_main.py" paths-check --output "$OUTPUT_ROOT" --original "$ORIGINAL_ROOT"
mkdir -p "$OUTPUT_ROOT/logs"
exec 9>"$OUTPUT_ROOT/.controller.lock"
if ! flock -n 9; then
    echo 'Another E0 controller or surviving worker holds this output lock.' >&2
    exit 2
fi
RUN=$(date -u +%Y%m%dT%H%M%SZ)_$$
CHILD=
cleanup() {
    if [ -n "$CHILD" ]; then
        kill -TERM "$CHILD" 2>/dev/null || :
        wait "$CHILD" 2>/dev/null || :
    fi
    exit 130
}
trap cleanup INT TERM HUP
for STAGE in preflight self-test smoke-paired smoke-R0 smoke-R1 smoke-R2 smoke-R3 smoke-analysis baseline paired reliability intervention-R1 intervention-R2 intervention-R3 error-analysis final
do
    LOG="$OUTPUT_ROOT/logs/${RUN}_${STAGE}.log"
    printf 'START %s log=%s\n' "$STAGE" "$LOG"
    "$PYTHON" -B "$HERE/e0_main.py" "$STAGE" --output "$OUTPUT_ROOT" --original "$ORIGINAL_ROOT" >"$LOG" 2>&1 &
    CHILD=$!
    if wait "$CHILD"; then
        CHILD=
        printf 'COMPLETE %s\n' "$STAGE"
    else
        CODE=$?
        CHILD=
        printf 'FAILED %s exit=%s log=%s\n' "$STAGE" "$CODE" "$LOG" >&2
        tail -n 80 "$LOG" >&2
        exit "$CODE"
    fi
done
printf 'E0 COMPLETE: %s/stages/final/E0_CONCLUSIONS.md\n' "$OUTPUT_ROOT"
