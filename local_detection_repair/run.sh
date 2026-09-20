#!/bin/sh
set -eu

ROOT=/home/cjm/OpenCOOD-main/cjmnet
OPENCOOD=/home/cjm/OpenCOOD-main
PY=${PY:-/home/cjm/miniconda3/envs/opencood/bin/python}
CONFIG=${CONFIG:-$ROOT/local_detection_repair/experiment.yaml}
FRONTEND_CONFIG=${FRONTEND_CONFIG:-/data/cjm/datasets/logs/gspr_joint_full_v1_seed20260907_20260907_161155/config.yaml}
FRONTEND_CHECKPOINT=${FRONTEND_CHECKPOINT:-/data/cjm/datasets/logs/gspr_joint_full_v1_seed20260907_20260907_161155/net_best_validation.pth}
GPU=${GPU:-0}
WEATHER=${WEATHER:-clean}
ABLATION=${ABLATION:-joint}
PRESET=${PRESET:-joint_selector}
ACTION=${1:-}

cd "$ROOT"
unset HIP_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES
PYTHONPATH_VALUE="$ROOT:$OPENCOOD"
export PYTHONHASHSEED=20260920

case "$ACTION" in
  smoke)
    env ROCR_VISIBLE_DEVICES="$GPU" PYTHONPATH="$PYTHONPATH_VALUE"       "$PY" -m unittest local_detection_repair.test_repair -v
    env ROCR_VISIBLE_DEVICES="$GPU" PYTHONPATH="$PYTHONPATH_VALUE"       "$PY" -u -m local_detection_repair.smoke       --config "$CONFIG"       --frontend-config "$FRONTEND_CONFIG"       --frontend-checkpoint "$FRONTEND_CHECKPOINT"
    ;;

  repair)
    RUN_NAME=${RUN_NAME:-local_repair_phase1_$(date +%Y%m%d_%H%M%S)}
    OUT="/data/cjm/datasets/logs/$RUN_NAME"
    LOG="/data/cjm/datasets/logs/${RUN_NAME}_launcher.log"
    RESUME_ARG=
    if [ "${RESUME:-0}" = "1" ]; then
      RESUME_ARG=--resume
    fi
    nohup env ROCR_VISIBLE_DEVICES="$GPU" PYTHONPATH="$PYTHONPATH_VALUE"       "$PY" -u -m local_detection_repair.train_repair       --config "$CONFIG"       --frontend-config "$FRONTEND_CONFIG"       --frontend-checkpoint "$FRONTEND_CHECKPOINT"       --output-dir "$OUT" $RESUME_ARG >"$LOG" 2>&1 &
    echo "PID=$! RUN_NAME=$RUN_NAME"
    echo "LOG=$LOG"
    ;;

  selector)
    : "${REPAIR_CHECKPOINT:?set REPAIR_CHECKPOINT to repair_best.pth}"
    RUN_NAME=${RUN_NAME:-local_repair_phase2_${ABLATION}_$(date +%Y%m%d_%H%M%S)}
    OUT="/data/cjm/datasets/logs/$RUN_NAME"
    LOG="/data/cjm/datasets/logs/${RUN_NAME}_launcher.log"
    RESUME_ARG=
    if [ "${RESUME:-0}" = "1" ]; then
      RESUME_ARG=--resume
    fi
    nohup env ROCR_VISIBLE_DEVICES="$GPU" PYTHONPATH="$PYTHONPATH_VALUE"       "$PY" -u -m local_detection_repair.train_selector       --config "$CONFIG"       --frontend-config "$FRONTEND_CONFIG"       --frontend-checkpoint "$FRONTEND_CHECKPOINT"       --repair-checkpoint "$REPAIR_CHECKPOINT"       --ablation "$ABLATION"       --output-dir "$OUT" $RESUME_ARG >"$LOG" 2>&1 &
    echo "PID=$! RUN_NAME=$RUN_NAME"
    echo "LOG=$LOG"
    ;;

  validate)
    : "${REPAIR_CHECKPOINT:?set REPAIR_CHECKPOINT to repair_best.pth}"
    : "${SELECTOR_CHECKPOINT:?set SELECTOR_CHECKPOINT to selector_best.pth}"
    RUN_NAME=${RUN_NAME:-local_repair_val_${WEATHER}_${PRESET}_$(date +%Y%m%d_%H%M%S)}
    OUT="/data/cjm/datasets/logs/$RUN_NAME"
    LOG="/data/cjm/datasets/logs/${RUN_NAME}_launcher.log"
    nohup env ROCR_VISIBLE_DEVICES="$GPU" PYTHONPATH="$PYTHONPATH_VALUE"       "$PY" -u -m local_detection_repair.evaluate       --config "$CONFIG"       --frontend-config "$FRONTEND_CONFIG"       --frontend-checkpoint "$FRONTEND_CHECKPOINT"       --repair-checkpoint "$REPAIR_CHECKPOINT"       --selector-checkpoint "$SELECTOR_CHECKPOINT"       --protocol development       --weather "$WEATHER"       --preset "$PRESET"       --output-dir "$OUT" >"$LOG" 2>&1 &
    echo "PID=$! RUN_NAME=$RUN_NAME"
    echo "LOG=$LOG"
    ;;

  test)
    : "${REPAIR_CHECKPOINT:?set REPAIR_CHECKPOINT to repair_best.pth}"
    : "${SELECTOR_CHECKPOINT:?set SELECTOR_CHECKPOINT to selector_best.pth}"
    RUN_NAME=${RUN_NAME:-local_repair_test_${WEATHER}_${PRESET}_$(date +%Y%m%d_%H%M%S)}
    OUT="/data/cjm/datasets/logs/$RUN_NAME"
    LOG="/data/cjm/datasets/logs/${RUN_NAME}_launcher.log"
    nohup env ROCR_VISIBLE_DEVICES="$GPU" PYTHONPATH="$PYTHONPATH_VALUE"       "$PY" -u -m local_detection_repair.evaluate       --config "$CONFIG"       --frontend-config "$FRONTEND_CONFIG"       --frontend-checkpoint "$FRONTEND_CHECKPOINT"       --repair-checkpoint "$REPAIR_CHECKPOINT"       --selector-checkpoint "$SELECTOR_CHECKPOINT"       --protocol benchmark       --weather "$WEATHER"       --preset "$PRESET"       --output-dir "$OUT" >"$LOG" 2>&1 &
    echo "PID=$! RUN_NAME=$RUN_NAME"
    echo "LOG=$LOG"
    ;;

  *)
    echo "Usage: sh local_detection_repair/run.sh {smoke|repair|selector|validate|test}"
    echo "Set GPU=0..6. selector needs REPAIR_CHECKPOINT."
    echo "validate/test need REPAIR_CHECKPOINT and SELECTOR_CHECKPOINT."
    exit 2
    ;;
esac
