#!/bin/sh
set -u

# Full local-repair pipeline:
# smoke -> phase1 repair -> phase2 selector -> validation -> optional benchmark.
# Portable POSIX sh, LF line endings.
#
# Re-run with the same RUN_ROOT after a failure. Completed stages are skipped.
# If phase1 has last.pth, it resumes. If it failed before the first checkpoint,
# a new phase1 attempt directory is created and the failed attempt is preserved.

ROOT=/home/cjm/OpenCOOD-main/cjmnet
OPENCOOD=/home/cjm/OpenCOOD-main
PY=${PY:-/home/cjm/miniconda3/envs/opencood/bin/python}
CONFIG=${CONFIG:-$ROOT/local_detection_repair/experiment.yaml}
FRONTEND_CONFIG=${FRONTEND_CONFIG:-/data/cjm/datasets/logs/gspr_joint_full_v1_seed20260907_20260907_161155/config.yaml}
FRONTEND_CHECKPOINT=${FRONTEND_CHECKPOINT:-/data/cjm/datasets/logs/gspr_joint_full_v1_seed20260907_20260907_161155/net_best_validation.pth}
GPU=${GPU:-0}
RUN_FINAL_TEST=${RUN_FINAL_TEST:-0}

cd "$ROOT" || exit 2
unset HIP_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES
export PYTHONHASHSEED=20260920
PYTHONPATH_VALUE="$ROOT:$OPENCOOD"

case "$RUN_FINAL_TEST" in
    0|1) ;;
    *)
        echo "ERROR: RUN_FINAL_TEST must be 0 or 1"
        exit 2
        ;;
esac

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

if [ -z "${RUN_ROOT:-}" ]; then
    RUN_ROOT="/data/cjm/datasets/logs/local_repair_full_$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "$RUN_ROOT" || exit 2

STATUS_FILE="$RUN_ROOT/pipeline_status.tsv"
FAILED_FILE="$RUN_ROOT/FAILED_STAGE"
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

fail_stage()
{
    stage=$1
    rc=$2
    log=$3
    printf '%s
' "$stage" > "$FAILED_FILE"
    record "$stage" "FAILED" "exit=$rc log=$log"
    echo ""
    echo "FAILED stage: $stage"
    echo "exit code: $rc"
    echo "log: $log"
    echo "failure marker: $FAILED_FILE"
    echo ""
    echo "After fixing the problem, continue with the SAME RUN_ROOT:"
    echo "RUN_ROOT=$RUN_ROOT GPU=$GPU RUN_FINAL_TEST=$RUN_FINAL_TEST sh local_detection_repair/run_full_pipeline.sh"
    exit "$rc"
}

run_simple_stage()
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
        fail_stage "$stage" "$rc" "$log"
    fi

    printf '%s
' "$(timestamp)" > "$marker"
    record "$stage" "DONE" "$log"
    echo "DONE  $stage"
    return 0
}

# 0. Real smoke: includes train/clean plus validation clean/fog/rain/snow.
run_simple_stage "smoke" "$RUN_ROOT/smoke.log"     env ROCR_VISIBLE_DEVICES="$GPU" PYTHONPATH="$PYTHONPATH_VALUE"     "$PY" -u -m local_detection_repair.smoke     --config "$CONFIG"     --frontend-config "$FRONTEND_CONFIG"     --frontend-checkpoint "$FRONTEND_CHECKPOINT"

# Also keep the focused pure unit tests.
run_simple_stage "unit_tests" "$RUN_ROOT/unit_tests.log"     env ROCR_VISIBLE_DEVICES="$GPU" PYTHONPATH="$PYTHONPATH_VALUE"     "$PY" -m unittest local_detection_repair.test_repair -v

# 1. Locate a completed or resumable phase1 attempt.
PHASE1_CHECKPOINT=
PHASE1_DIR=
latest_attempt=$(ls -1dt "$RUN_ROOT"/phase1_attempt_* 2>/dev/null | head -n 1)

if [ -n "$latest_attempt" ] && [ -f "$latest_attempt/repair_best.pth" ]; then
    PHASE1_DIR=$latest_attempt
    PHASE1_CHECKPOINT="$PHASE1_DIR/repair_best.pth"
fi

if [ -z "$PHASE1_CHECKPOINT" ]; then
    if [ -n "$latest_attempt" ] && [ -f "$latest_attempt/last.pth" ]; then
        PHASE1_DIR=$latest_attempt
        resume_arg=--resume
    else
        attempt=1
        while [ -e "$RUN_ROOT/phase1_attempt_$(printf '%03d' "$attempt")" ]; do
            attempt=$((attempt + 1))
        done
        PHASE1_DIR="$RUN_ROOT/phase1_attempt_$(printf '%03d' "$attempt")"
        resume_arg=
    fi

    PHASE1_LOG="$PHASE1_DIR.log"
    echo "START phase1_repair"
    echo "OUT   $PHASE1_DIR"
    echo "LOG   $PHASE1_LOG"
    record "phase1_repair" "STARTED" "$PHASE1_LOG"

    env ROCR_VISIBLE_DEVICES="$GPU" PYTHONPATH="$PYTHONPATH_VALUE"         "$PY" -u -m local_detection_repair.train_repair         --config "$CONFIG"         --frontend-config "$FRONTEND_CONFIG"         --frontend-checkpoint "$FRONTEND_CHECKPOINT"         --output-dir "$PHASE1_DIR" $resume_arg > "$PHASE1_LOG" 2>&1
    rc=$?
    if [ "$rc" -ne 0 ]; then
        fail_stage "phase1_repair" "$rc" "$PHASE1_LOG"
    fi

    PHASE1_CHECKPOINT="$PHASE1_DIR/repair_best.pth"
    if [ ! -f "$PHASE1_CHECKPOINT" ]; then
        fail_stage "phase1_repair_checkpoint" 3 "$PHASE1_LOG"
    fi

    printf '%s
' "$(timestamp)" > "$DONE_DIR/phase1_repair.done"
    record "phase1_repair" "DONE" "$PHASE1_CHECKPOINT"
else
    echo "SKIP completed phase1: $PHASE1_CHECKPOINT"
    record "phase1_repair" "SKIPPED" "$PHASE1_CHECKPOINT"
fi

printf '%s
' "$PHASE1_CHECKPOINT" > "$RUN_ROOT/PHASE1_CHECKPOINT"
echo "Phase-1 checkpoint:"
echo "$PHASE1_CHECKPOINT"

# 2+. Delegate the remaining resumable stages to the post-phase1 pipeline.
# It shares RUN_ROOT, pipeline_status.tsv, FAILED_STAGE and done markers.
REPAIR_CHECKPOINT="$PHASE1_CHECKPOINT" RUN_ROOT="$RUN_ROOT" GPU="$GPU" RUN_FINAL_TEST="$RUN_FINAL_TEST" PY="$PY" CONFIG="$CONFIG" FRONTEND_CONFIG="$FRONTEND_CONFIG" FRONTEND_CHECKPOINT="$FRONTEND_CHECKPOINT" sh local_detection_repair/run_after_repair.sh
rc=$?

if [ "$rc" -ne 0 ]; then
    exit "$rc"
fi

rm -f "$FAILED_FILE"
printf '%s
' "$(timestamp)" > "$RUN_ROOT/FULL_PIPELINE_DONE"
record "full_pipeline" "DONE" "all requested stages complete"

echo ""
echo "FULL PIPELINE COMPLETE"
echo "RUN_ROOT=$RUN_ROOT"
echo "STATUS=$STATUS_FILE"
