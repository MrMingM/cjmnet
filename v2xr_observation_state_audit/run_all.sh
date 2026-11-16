#!/bin/sh
set -eu
cd "$(dirname "$0")"
: "${OBS_OUT:=/data/new_disk/cjm_logs/v2xr-observation-state-audit-v1}"
: "${GPU:=4}"
: "${PYTHON:=/home/cjm/miniconda3/envs/opencood/bin/python}"
export OBS_OUT GPU ROCR_VISIBLE_DEVICES="$GPU" LR0_EMPTY_POLICY=zero PYTHONHASHSEED=20261017
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
mkdir -p "$OBS_OUT"
exec 9>"$OBS_OUT/.lock"
flock -n 9
stage=smoke
trap 'code=$?; printf "%s %s\n" "$stage" "$code" > "$OBS_OUT/status.txt"' 0
case "${1:-full}" in
    smoke|full) ;;
    *) exit 2 ;;
esac
"$PYTHON" -uB extract.py smoke
"$PYTHON" -uB analyze.py smoke
if [ "${1:-full}" = full ]; then
    sleep 8
    stage=full
    printf '%s\n' 'full running' > "$OBS_OUT/status.txt"
    "$PYTHON" -uB extract.py full
    stage=analysis
    "$PYTHON" -uB analyze.py full
fi
stage=complete
