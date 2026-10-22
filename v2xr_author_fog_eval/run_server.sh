#!/bin/sh
# POSIX sh. This entry point contains evaluation only.
set -eu
SCRIPT_DIR=$(CDPATH= cd "$(dirname "$0")" && pwd)
REPO_DIR=$(dirname "$SCRIPT_DIR")
MODE=${MODE:-all}
RESUME=${RESUME:-0}
GPU=${GPU:-2}
OUTPUT_ROOT=${OUTPUT_ROOT:-/data/cjm/datasets/logs/v2xr-author-fog-attfuse-val-v1}
CHECKPOINT=${CHECKPOINT:-/home/cjm/OpenCOOD-main/cjmnet/logs/AttFuse_LiDAR_train_w_fog.pth}
PYTHON=${PYTHON:-/home/cjm/miniconda3/envs/opencood/bin/python}
case "$MODE" in self-test|smoke|all) ;; *) printf '%s\n' 'MODE must be self-test, smoke or all' >&2; exit 2 ;; esac
case "$RESUME" in 0|1) ;; *) printf '%s\n' 'RESUME must be 0 or 1' >&2; exit 2 ;; esac
case "$GPU" in ''|*[!0-9]*) printf '%s\n' 'GPU must be one physical GPU number' >&2; exit 2 ;; esac
unset CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES RANK WORLD_SIZE LOCAL_RANK SLURM_PROCID
ROCR_VISIBLE_DEVICES=$GPU
PYTHONHASHSEED=20261016
PYTHONDONTWRITEBYTECODE=1
PYTHONUNBUFFERED=1
export ROCR_VISIBLE_DEVICES PYTHONHASHSEED PYTHONDONTWRITEBYTECODE PYTHONUNBUFFERED
case "$OUTPUT_ROOT" in /data/*) ;; *) printf '%s\n' 'OUTPUT_ROOT must be on /data' >&2; exit 2 ;; esac
TMPDIR=${TMPDIR:-$OUTPUT_ROOT/tmp}
case "$TMPDIR" in /data/*) ;; *) printf '%s\n' 'TMPDIR must be on /data' >&2; exit 2 ;; esac
export TMPDIR
cd "$REPO_DIR"
exec "$PYTHON" -B -u -m v2xr_author_fog_eval.main --mode "$MODE" --resume "$RESUME" \
  --gpu "$GPU" --output-root "$OUTPUT_ROOT" --checkpoint "$CHECKPOINT" \
  --runtime-root "${RUNTIME_ROOT:-/data/cjm/datasets/v2xr-official-runtime-v1}" \
  --model-dir "${MODEL_DIR:-/data/cjm/datasets/logs/v2xr-official-attfuse-model-v1}" \
  --d1a-root "${D1A_ROOT:-/data/cjm/datasets/logs/v2xr-official-attfuse-clean-repro-v1}" \
  --gt-audit-root "${GT_AUDIT_ROOT:-/data/cjm/datasets/logs/v2xr-official-gt-audit-v1}" \
  --d1c-root "${D1C_ROOT:-/data/cjm/datasets/logs/v2xr-dminus1c-protocol-v1}" \
  --view-root "${VIEW_ROOT:-/data/cjm/datasets/v2xr-official-validate-v1}" \
  --cd-root "${CD_ROOT:-/data/cjm/datasets/logs/v2xr-weather-l012-v1}" \
  --cd-results "${CD_RESULTS:-$REPO_DIR/logs/train/FINAL_RESULTS.json}"
