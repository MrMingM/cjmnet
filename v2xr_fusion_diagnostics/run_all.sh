#!/bin/sh
set -eu
cd "$(dirname "$0")"
sh -n run_all.sh
root=/data/cjm/datasets/logs/v2xr-fusion-diagnostics-v1
python=/home/cjm/miniconda3/envs/opencood/bin/python
mkdir -p "$root/D1" "$root/D2" "$root/D3" "$root/tmp"
exec 9>"$root/.lock"
flock -n 9
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
export PYTHONHASHSEED=20261017 LR0_EMPTY_POLICY=zero
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
TMPDIR="$root/tmp"
export TMPDIR
"$python" -uB worker.py audit >> "$root/audit.log" 2>&1

launch() {
    group=$1
    card=$2
    stage=$3
    ROCR_VISIBLE_DEVICES="$card" "$python" -uB worker.py "$group" "$stage" >> "$root/$group/$stage.log" 2>&1
}

parallel_stage() {
    phase=$1
    launch D1 0 "$phase" & p1=$!
    launch D2 2 "$phase" & p2=$!
    launch D3 4 "$phase" & p3=$!
    s1=0; s2=0; s3=0
    wait "$p1" || s1=$?
    wait "$p2" || s2=$?
    wait "$p3" || s3=$?
    printf '%s D1=%s D2=%s D3=%s\n' "$phase" "$s1" "$s2" "$s3" >> "$root/launch_status.log"
    test "$s1" -eq 0 && test "$s2" -eq 0 && test "$s3" -eq 0
}
parallel_stage smoke
parallel_stage full
"$python" -uB collect_results.py >> "$root/collect.log" 2>&1
