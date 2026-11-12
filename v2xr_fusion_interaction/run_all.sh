#!/bin/sh
set -eu
cd "$(dirname "$0")"
sh -n run_all.sh
root=/data/new_disk/cjm_logs/v2xr-fusion-interaction-v1
python=/home/cjm/miniconda3/envs/opencood/bin/python
if ! test -d "$root" || ! test -w "$root"; then
    printf 'A writable experiment directory is required: %s\n' "$root" >&2
    exit 1
fi
mkdir -p "$root/F2" "$root/F3" "$root/F4" "$root/BASELINE" "$root/tmp"
exec 9>"$root/.lock"
flock -n 9
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
export PYTHONHASHSEED=20261017 LR0_EMPTY_POLICY=zero
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
TMPDIR="$root/tmp"
export TMPDIR
"$python" -uB run_group.py audit >> "$root/audit.log" 2>&1

launch() {
    group=$1
    gpu=$2
    stage=$3
    ROCR_VISIBLE_DEVICES="$gpu" "$python" -uB run_group.py "$group" "$stage" >> "$root/$group/$stage.log" 2>&1
}

parallel_stage() {
    phase=$1
    launch F2 2 "$phase" & p2=$!
    launch F3 3 "$phase" & p3=$!
    launch F4 4 "$phase" & p4=$!
    s2=0; s3=0; s4=0
    wait "$p2" || s2=$?
    wait "$p3" || s3=$?
    wait "$p4" || s4=$?
    printf '%s F2=%s F3=%s F4=%s\n' "$phase" "$s2" "$s3" "$s4" >> "$root/launch_status.log"
    test "$s2" -eq 0 && test "$s3" -eq 0 && test "$s4" -eq 0
}
parallel_stage smoke
"$python" -uB run_group.py gate >> "$root/audit.log" 2>&1
parallel_stage full
"$python" -uB collect_results.py >> "$root/collect.log" 2>&1
