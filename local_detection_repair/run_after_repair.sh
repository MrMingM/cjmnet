#!/bin/sh
set -u

# Sequential pipeline after phase-1 repair training.
# POSIX sh only, using portable syntax and LF line endings.
#
# Stages:
#   preflight -> selector -> validation(clean/fog/rain/snow)
#   -> optional fixed-plan benchmark(clean/fog/rain/snow)
#
# On failure:
#   - exits immediately
#   - writes FAILED_STAGE
#   - keeps a dedicated stage log
# Re-run with the same RUN_ROOT to skip completed stages and continue.

ROOT=/home/cjm/OpenCOOD-main/cjmnet
OPENCOOD=/home/cjm/OpenCOOD-main
PY=${PY:-/home/cjm/miniconda3/envs/opencood/bin/python}
CONFIG=${CONFIG:-$ROOT/local_detection_repair/experiment.yaml}
FRONTEND_CONFIG=${FRONTEND_CONFIG:-/data/cjm/datasets/logs/gspr_joint_full_v1_seed20260907_20260907_161155/config.yaml}
FRONTEND_CHECKPOINT=${FRONTEND_CHECKPOINT:-/data/cjm/datasets/logs/gspr_joint_full_v1_seed20260907_20260907_161155/net_best_validation.pth}
GPU=${GPU:-0}
ABLATION=${ABLATION:-joint}
PRESET=${PRESET:-joint_selector}
RUN_FINAL_TEST=${RUN_FINAL_TEST:-0}

if [ -z "${REPAIR_CHECKPOINT:-}" ]; then
    echo "ERROR: set REPAIR_CHECKPOINT to a completed phase-1 repair_best.pth"
    exit 2
fi

if [ ! -f "$REPAIR_CHECKPOINT" ]; then
    echo "ERROR: repair checkpoint not found: $REPAIR_CHECKPOINT"
    exit 2
fi

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: config not found: $CONFIG"
    exit 2
fi

if [ ! -f "$FRONTEND_CONFIG" ]; then
    echo "ERROR: frontend config not found: $FRONTEND_CONFIG"
    exit 2
fi

if [ ! -f "$FRONTEND_CHECKPOINT" ]; then
    echo "ERROR: frontend checkpoint not found: $FRONTEND_CHECKPOINT"
    exit 2
fi

case "$RUN_FINAL_TEST" in
    0|1) ;;
    *)
        echo "ERROR: RUN_FINAL_TEST must be 0 or 1"
        exit 2
        ;;
esac

cd "$ROOT" || exit 2
unset HIP_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES
export PYTHONHASHSEED=20260920
PYTHONPATH_VALUE="$ROOT:$OPENCOOD"

if [ -z "${RUN_ROOT:-}" ]; then
    RUN_ROOT="/data/cjm/datasets/logs/local_repair_after_phase1_$(date +%Y%m%d_%H%M%S)"
fi

if [ ! -d "$RUN_ROOT" ]; then
    mkdir -p "$RUN_ROOT" || exit 2
fi

STATUS_FILE="$RUN_ROOT/pipeline_status.tsv"
FAILED_FILE="$RUN_ROOT/FAILED_STAGE"
LOCK_FILE="$RUN_ROOT/locked_plan.txt"
DONE_DIR="$RUN_ROOT/done"
mkdir -p "$DONE_DIR" || exit 2

timestamp()
{
    date '+%Y-%m-%d %H:%M:%S'
}

record()
{
    stage=$1
    status=$2
    detail=$3
    printf '%s	%s	%s	%s
' "$(timestamp)" "$stage" "$status" "$detail" >> "$STATUS_FILE"
}

file_sha()
{
    sha256sum "$1" | awk '{print $1}'
}

# Lock the intended evaluation plan before validation is observed.
# This makes RUN_FINAL_TEST=1 a pre-declared fixed-plan run rather than
# validation-driven manual tuning followed by test.
if [ ! -f "$LOCK_FILE" ]; then
    {
        echo "created=$(timestamp)"
        echo "git_sha=$(git rev-parse HEAD 2>/dev/null || echo unavailable)"
        echo "config=$CONFIG"
        echo "config_sha256=$(file_sha "$CONFIG")"
        echo "frontend_config=$FRONTEND_CONFIG"
        echo "frontend_config_sha256=$(file_sha "$FRONTEND_CONFIG")"
        echo "frontend_checkpoint=$FRONTEND_CHECKPOINT"
        echo "frontend_checkpoint_sha256=$(file_sha "$FRONTEND_CHECKPOINT")"
        echo "repair_checkpoint=$REPAIR_CHECKPOINT"
        echo "repair_checkpoint_sha256=$(file_sha "$REPAIR_CHECKPOINT")"
        echo "ablation=$ABLATION"
        echo "preset=$PRESET"
        echo "selector_checkpoint_rule=selector_best.pth selected by validation loss"
        echo "selector_threshold=from locked experiment.yaml"
        echo "candidate_rules=from locked experiment.yaml"
        echo "run_final_test=$RUN_FINAL_TEST"
        echo "final_test_rule=no parameter/checkpoint/threshold/preset changes after validation"
    } > "$LOCK_FILE"
fi

run_stage()
{
    stage=$1
    log=$2
    shift 2

    marker="$DONE_DIR/$stage.done"
    if [ -f "$marker" ]; then
        echo "SKIP completed stage: $stage"
        record "$stage" "SKIPPED" "already complete"
        return 0
    fi

    rm -f "$FAILED_FILE"
    echo "START $stage"
    echo "LOG   $log"
    record "$stage" "STARTED" "$log"

    "$@" > "$log" 2>&1
    rc=$?

    if [ "$rc" -ne 0 ]; then
        printf '%s
' "$stage" > "$FAILED_FILE"
        record "$stage" "FAILED" "exit=$rc log=$log"
        echo ""
        echo "FAILED stage: $stage"
        echo "exit code: $rc"
        echo "log: $log"
        echo "failure marker: $FAILED_FILE"
        echo ""
        echo "After fixing the problem, re-run with:"
        echo "RUN_ROOT=$RUN_ROOT REPAIR_CHECKPOINT=$REPAIR_CHECKPOINT GPU=$GPU RUN_FINAL_TEST=$RUN_FINAL_TEST sh local_detection_repair/run_after_repair.sh"
        return "$rc"
    fi

    printf '%s
' "$(timestamp)" > "$marker"
    record "$stage" "DONE" "$log"
    echo "DONE  $stage"
    return 0
}

# 0. Cheap checks before allocating a long selector job.
run_stage "preflight" "$RUN_ROOT/preflight.log"     env ROCR_VISIBLE_DEVICES="$GPU" PYTHONPATH="$PYTHONPATH_VALUE"     "$PY" -m unittest local_detection_repair.test_repair -v || exit $?

# 1. Phase-2 selector.
SELECTOR_DIR="$RUN_ROOT/selector"
SELECTOR_LOG="$RUN_ROOT/selector.log"
SELECTOR_RESUME=
if [ -d "$SELECTOR_DIR" ]; then
    SELECTOR_RESUME=--resume
fi

run_stage "selector" "$SELECTOR_LOG"     env ROCR_VISIBLE_DEVICES="$GPU" PYTHONPATH="$PYTHONPATH_VALUE"     "$PY" -u -m local_detection_repair.train_selector     --config "$CONFIG"     --frontend-config "$FRONTEND_CONFIG"     --frontend-checkpoint "$FRONTEND_CHECKPOINT"     --repair-checkpoint "$REPAIR_CHECKPOINT"     --ablation "$ABLATION"     --output-dir "$SELECTOR_DIR" $SELECTOR_RESUME || exit $?

SELECTOR_CHECKPOINT="$SELECTOR_DIR/selector_best.pth"
if [ ! -f "$SELECTOR_CHECKPOINT" ]; then
    printf '%s
' "selector_checkpoint_missing" > "$FAILED_FILE"
    record "selector_checkpoint" "FAILED" "missing $SELECTOR_CHECKPOINT"
    echo "ERROR: selector finished but selector_best.pth is missing"
    exit 3
fi

if ! grep -q '^selector_checkpoint=' "$LOCK_FILE"; then
    {
        echo "selector_checkpoint=$SELECTOR_CHECKPOINT"
        echo "selector_checkpoint_sha256=$(file_sha "$SELECTOR_CHECKPOINT")"
    } >> "$LOCK_FILE"
fi

# Evaluation stages use a fresh output directory on every failed retry because
# evaluate.py intentionally refuses to overwrite a partial result directory.
eval_stage()
{
    protocol=$1
    weather=$2
    stage=$3
    base="$RUN_ROOT/$stage"
    marker="$DONE_DIR/$stage.done"

    if [ -f "$marker" ]; then
        echo "SKIP completed stage: $stage"
        record "$stage" "SKIPPED" "already complete"
        return 0
    fi

    attempt=1
    while [ -e "${base}_attempt_$(printf '%03d' "$attempt")" ]; do
        attempt=$((attempt + 1))
    done
    out="${base}_attempt_$(printf '%03d' "$attempt")"
    log="${out}.log"

    run_stage "$stage" "$log"         env ROCR_VISIBLE_DEVICES="$GPU" PYTHONPATH="$PYTHONPATH_VALUE"         "$PY" -u -m local_detection_repair.evaluate         --config "$CONFIG"         --frontend-config "$FRONTEND_CONFIG"         --frontend-checkpoint "$FRONTEND_CHECKPOINT"         --repair-checkpoint "$REPAIR_CHECKPOINT"         --selector-checkpoint "$SELECTOR_CHECKPOINT"         --protocol "$protocol"         --weather "$weather"         --preset "$PRESET"         --output-dir "$out"
}

# 2. Development validation: same fixed scheme, four conditions.
for weather in clean fog rain snow
do
    eval_stage development "$weather" "validation_$weather" || exit $?
done

if [ "$RUN_FINAL_TEST" = "0" ]; then
    record "pipeline" "WAITING_FOR_FINAL_TEST" "validation complete; final benchmark not requested"
    echo ""
    echo "VALIDATION PIPELINE COMPLETE"
    echo "Final benchmark was NOT run because RUN_FINAL_TEST=0."
    echo "To run the already locked plan later:"
    echo "RUN_ROOT=$RUN_ROOT REPAIR_CHECKPOINT=$REPAIR_CHECKPOINT GPU=$GPU RUN_FINAL_TEST=1 sh local_detection_repair/run_after_repair.sh"
    exit 0
fi

# 3. Formal fixed benchmark. No tuning occurs between validation and these runs.
for weather in clean fog rain snow
do
    eval_stage benchmark "$weather" "test_$weather" || exit $?
done

record "pipeline" "DONE" "selector + validation + fixed benchmark complete"
rm -f "$FAILED_FILE"
printf '%s
' "$(timestamp)" > "$RUN_ROOT/PIPELINE_DONE"

echo ""
echo "PIPELINE COMPLETE"
echo "RUN_ROOT=$RUN_ROOT"
echo "STATUS=$STATUS_FILE"
echo "LOCKED_PLAN=$LOCK_FILE"
