#!/bin/sh
set -eu
cd "$(dirname "$0")"
: "${NEG_OUT:=/data/new_disk/cjm_logs/v2xr-negative-collaboration-v1}"
: "${PYTHON:=/home/cjm/miniconda3/envs/opencood/bin/python}"
export NEG_OUT LR0_EMPTY_POLICY=zero PYTHONHASHSEED=20261017 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
mkdir -p "$NEG_OUT"
exec 9>"$NEG_OUT/.lock"
flock -n 9
stage=preflight
trap 'code=$?; printf "%s %s\n" "$stage" "$code" > "$NEG_OUT/status.txt"' 0
if [ "${1:-}" = "--with-lrg-training" ]; then
    : "${LRG_RUN_DIR:=$NEG_OUT/lrg-training/LRG}"
    export LRG_RUN_DIR
    export TMPDIR="$NEG_OUT/tmp"
    mkdir -p "$TMPDIR" "$NEG_OUT/E0" "$NEG_OUT/E1" "$NEG_OUT/E2"
    "$PYTHON" -uB audit.py preflight --with-lrg-training
    stage=training_and_evaluation
    printf '%s running\n' "$stage" > "$NEG_OUT/status.txt"
    evaluate_group() {
        group=$1
        if [ -f "$NEG_OUT/$group/RESULTS.json" ]; then
            printf '%s\n' "Reusing completed $group"
            return
        fi
        "$PYTHON" -uB audit.py "$group" smoke >> "$NEG_OUT/$group/smoke.log" 2>&1 || return
        sleep 8
        "$PYTHON" -uB audit.py "$group" full --with-lrg-training >> "$NEG_OUT/$group/full.log" 2>&1
    }
    (
        export ROCR_VISIBLE_DEVICES=4
        evaluate_group E0
    ) & p0=$!
    (
        export ROCR_VISIBLE_DEVICES=3
        if [ ! -f "$LRG_RUN_DIR/FINAL_RESULTS.json" ]; then
            "$PYTHON" -uB train_lrg.py >> "$NEG_OUT/E1/train.log" 2>&1 || exit $?
            sleep 8
        fi
        "$PYTHON" -uB audit.py register-lrg >> "$NEG_OUT/E1/train.log" 2>&1 || exit $?
        evaluate_group E1
    ) & p1=$!
    (
        export ROCR_VISIBLE_DEVICES=5
        evaluate_group E2
    ) & p2=$!
    a=0; b=0; c=0
    wait "$p0" || a=$?
    wait "$p1" || b=$?
    wait "$p2" || c=$?
    printf 'E0=%s E1=%s E2=%s\n' "$a" "$b" "$c" | tee "$NEG_OUT/full.exit"
    [ "$a$b$c" = 000 ]
    stage=report
    "$PYTHON" -uB audit.py report
    stage=complete
    exit 0
fi
"$PYTHON" -uB audit.py preflight
parallel() {
    label=$1
    for group in E0 E1 E2; do mkdir -p "$NEG_OUT/$group"; done
    ROCR_VISIBLE_DEVICES=4 "$PYTHON" -uB audit.py E0 "$label" >> "$NEG_OUT/E0/$label.log" 2>&1 & p0=$!
    ROCR_VISIBLE_DEVICES=3 "$PYTHON" -uB audit.py E1 "$label" >> "$NEG_OUT/E1/$label.log" 2>&1 & p1=$!
    ROCR_VISIBLE_DEVICES=5 "$PYTHON" -uB audit.py E2 "$label" >> "$NEG_OUT/E2/$label.log" 2>&1 & p2=$!
    a=0; b=0; c=0
    wait "$p0" || a=$?
    wait "$p1" || b=$?
    wait "$p2" || c=$?
    printf 'E0=%s E1=%s E2=%s\n' "$a" "$b" "$c" | tee "$NEG_OUT/$label.exit"
    [ "$a$b$c" = 000 ]
}
stage=smoke
parallel smoke
sleep 8
stage=full
parallel full
stage=report
"$PYTHON" -uB audit.py report
stage=complete
