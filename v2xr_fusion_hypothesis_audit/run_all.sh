#!/bin/sh
set -eu
cd "$(dirname "$0")"
AUDIT_OUT=${AUDIT_OUT:-/data/cjm/datasets/logs/v2xr-fusion-hypothesis-audit-v1}
PY=${PYTHON:-/home/cjm/miniconda3/envs/opencood/bin/python}
export AUDIT_OUT
export PYTHONHASHSEED=20261017 LR0_EMPTY_POLICY=zero OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
mkdir -p "$AUDIT_OUT"
exec 9>"$AUDIT_OUT/.run.lock"
flock -n 9
stage=preflight
finish() {
    result=$?
    if [ "$result" -ne 0 ]; then
        printf '{"status":"failed","stage":"%s","exit_code":%s}\n' "$stage" "$result" > "$AUDIT_OUT/FAILURE.json"
        cp "$AUDIT_OUT/FAILURE.json" "$AUDIT_OUT/STATUS.json"
    fi
}
trap finish 0
trap 'exit 130' INT
trap 'exit 143' TERM
step() {
    stage=$1
    printf '{"status":"running","stage":"%s"}\n' "$stage" > "$AUDIT_OUT/STATUS.json"
    printf 'STAGE %s\n' "$stage"
}
parallel() {
    suffix=$1
    shift
    ROCR_VISIBLE_DEVICES=1 "$PY" -uB main.py V1 "$@" >> "$AUDIT_OUT/V1$suffix.log" 2>&1 & p1=$!
    ROCR_VISIBLE_DEVICES=2 "$PY" -uB main.py V2 "$@" >> "$AUDIT_OUT/V2$suffix.log" 2>&1 & p2=$!
    ROCR_VISIBLE_DEVICES=3 "$PY" -uB main.py V3 "$@" >> "$AUDIT_OUT/V3$suffix.log" 2>&1 & p3=$!
    s1=0; s2=0; s3=0
    wait "$p1" || s1=$?
    wait "$p2" || s2=$?
    wait "$p3" || s3=$?
    printf 'V1=%s V2=%s V3=%s\n' "$s1" "$s2" "$s3"
    [ "$s1$s2$s3" = 000 ]
}
step preflight
"$PY" -uB main.py contract
"$PY" -B -m unittest discover -s tests
step shared_smoke
ROCR_VISIBLE_DEVICES=2 "$PY" -uB main.py extract --smoke
sleep 8
step diagnostic_smoke
parallel _smoke --smoke
sleep 8
if [ -n "${AUDIT_PROBE_FRAME:-}" ]; then
    step targeted_replay
    ROCR_VISIBLE_DEVICES=2 "$PY" -uB main.py extract --probe "$AUDIT_PROBE_FRAME"
    sleep 8
fi
step shared_extract
ROCR_VISIBLE_DEVICES=2 "$PY" -uB main.py extract
sleep 8
step parallel_diagnostics
parallel ''
step report
"$PY" -uB main.py summary
printf '{"status":"complete","stage":"report"}\n' > "$AUDIT_OUT/STATUS.json"
printf '%s\n' 'AUDIT COMPLETE'
