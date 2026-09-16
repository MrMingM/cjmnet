#!/usr/bin/env bash
set -euo pipefail

GSPR_DIR=${GSPR_DIR:-/home/cjm/OpenCOOD-main/cjmnet}
PYTHON=${PYTHON:-/home/cjm/miniconda3/envs/opencood/bin/python}
FULL_ROOT=${FULL_ROOT:-/data/cjm/datasets/gspr_opv2v_weather_full_v1}
DETECTION_ONLY=${DETECTION_ONLY:-/data/cjm/datasets/logs/gspr_attfuse_opv2v_20260904_154404/net_epoch11.pth}
LOG_ROOT=${LOG_ROOT:-/data/cjm/datasets/logs}

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 SEED [SEED ...]" >&2
  exit 2
fi
for required in \
  "$GSPR_DIR/pretrain_gspr.py" \
  "$GSPR_DIR/train_gspr_multitask.py" \
  "$GSPR_DIR/gspr_attfuse_config.yaml" \
  "$FULL_ROOT/train_manifest.jsonl" \
  "$FULL_ROOT/val_manifest.jsonl" \
  "$DETECTION_ONLY"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing required file: $required" >&2
    exit 1
  fi
done

index_file="$LOG_ROOT/gspr_full_replicates_$(date +%Y%m%d_%H%M%S).tsv"
printf 'seed\tpoint_run\tjoint_run\n' > "$index_file"
printf '%s\n' "$index_file" > "$LOG_ROOT/latest_gspr_replicate_index.txt"
echo "Replicate index: $index_file"

for seed in "$@"; do
  stamp=$(date +%Y%m%d_%H%M%S)
  point_run="$LOG_ROOT/gspr_point_full_v1_seed${seed}_${stamp}"
  joint_run="$LOG_ROOT/gspr_joint_full_v1_seed${seed}_${stamp}"
  mkdir -p "$point_run" "$joint_run"
  printf '%s\t%s\t%s\n' "$seed" "$point_run" "$joint_run" >> "$index_file"

  echo "[$(date -Is)] seed=$seed point pretraining: $point_run"
  "$PYTHON" "$GSPR_DIR/pretrain_gspr.py" \
    --train-manifest "$FULL_ROOT/train_manifest.jsonl" \
    --val-manifest "$FULL_ROOT/val_manifest.jsonl" \
    --output-dir "$point_run" \
    --init-attfuse-checkpoint "$DETECTION_ONLY" \
    --epochs 5 \
    --batch-size 1 \
    --num-workers 4 \
    --lr 0.0002 \
    --max-voxels 32000 \
    --sampling stratified \
    --noise-bearing-fraction 0.8 \
    --samples-per-epoch 0 \
    --seed "$seed" \
    > "$point_run/train.log" 2>&1

  if [[ ! -f "$point_run/gspr_best.pth" ]]; then
    echo "Point pretraining produced no gspr_best.pth for seed $seed" >&2
    exit 1
  fi

  echo "[$(date -Is)] seed=$seed joint training: $joint_run"
  "$PYTHON" "$GSPR_DIR/train_gspr_multitask.py" \
    --config "$GSPR_DIR/gspr_attfuse_config.yaml" \
    --run-dir "$joint_run" \
    --baseline-checkpoint "$DETECTION_ONLY" \
    --gspr-checkpoint "$point_run/gspr_best.pth" \
    --train-manifest "$FULL_ROOT/train_manifest.jsonl" \
    --val-manifest "$FULL_ROOT/val_manifest.jsonl" \
    --epochs 5 \
    --detection-batch-size 2 \
    --point-batch-size 1 \
    --max-train-steps 0 \
    --num-workers 4 \
    --lr 0.00002 \
    --lambda-point 1.0 \
    --point-sampling stratified \
    --noise-bearing-fraction 0.8 \
    --point-samples-per-epoch 0 \
    --freeze-detector-epochs 1 \
    --min-point-macro-auroc 0.995 \
    --seed "$seed" \
    > "$joint_run/train.log" 2>&1

  if [[ ! -f "$joint_run/net_best_validation.pth" ]]; then
    echo "Joint training produced no eligible best checkpoint for seed $seed" >&2
    exit 1
  fi
  echo "[$(date -Is)] seed=$seed complete"
done

echo "All requested replicates completed: $index_file"
