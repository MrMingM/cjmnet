#!/bin/sh
set -eu

PY=${PY:-/home/cjm/miniconda3/envs/opencood/bin/python}
V3_CONFIG=${V3_CONFIG:-local_fusion_v3/experiment.yaml}
WEATHER_DATASET_ROOT=${WEATHER_DATASET_ROOT:-/data/cjm/datasets/opv2v-physics-fixed-v1}
SMOKE=${SMOKE:-0}
EPOCHS=${EPOCHS:-10}
SEED=${SEED:-20260930}
RESUME=${RESUME:-0}

if [ "$RESUME" != "0" ] && [ "$RESUME" != "1" ]; then
    echo "RESUME must be 0 or 1" >&2
    exit 1
fi

if [ "$RESUME" = "1" ]; then
    : "${OUT:?Set OUT to the existing Learned3DNMS experiment directory}"
    if [ ! -d "$OUT" ]; then
        echo "Resume directory does not exist: $OUT" >&2
        exit 1
    fi
    if [ ! -f "$OUT/cache/manifest.json" ]; then
        echo "Resume cache manifest missing: $OUT/cache/manifest.json" >&2
        exit 1
    fi
    printf 'RESUME Learned 3D NMS: %s\n' "$OUT"
else
    : "${RUN:?Set RUN to the completed adaptation run containing F.pth}"
    : "${FRONTEND_CONFIG:?Set FRONTEND_CONFIG}"
    : "${FRONTEND_CHECKPOINT:?Set FRONTEND_CHECKPOINT}"
    : "${V3_CHECKPOINT:?Set V3_CHECKPOINT}"

    OUT=${OUT:-/data/cjm/datasets/logs/learned_3d_nms_$(date +%Y%m%d_%H%M%S)}
    if [ -e "$OUT" ]; then
        echo "Refusing to overwrite existing experiment: $OUT" >&2
        exit 1
    fi
    if [ ! -f "$WEATHER_DATASET_ROOT/manifest.json" ]; then
        echo "Fixed weather dataset manifest missing: $WEATHER_DATASET_ROOT/manifest.json" >&2
        exit 1
    fi
    mkdir -p "$OUT"
    printf 'Learned 3D NMS output: %s\n' "$OUT"
    printf 'Data: %s (pre-generated PCD, no online augmentation)\n' "$WEATHER_DATASET_ROOT"
fi

# Fail-fast checks. This now also verifies that cached-IoU TP/FP accumulation
# is exactly consistent with OpenCOOD on a synthetic frame.
"$PY" -m unittest Learned3DNMS.test_learned_nms -v

if [ "$RESUME" = "0" ]; then
    # Frozen F is run once per split/condition and written to this run's cache.
    "$PY" -u -m Learned3DNMS.extract \
        --run "$RUN" \
        --v3-config "$V3_CONFIG" \
        --frontend-config "$FRONTEND_CONFIG" \
        --frontend-checkpoint "$FRONTEND_CHECKPOINT" \
        --v3-checkpoint "$V3_CHECKPOINT" \
        --weather-dataset-root "$WEATHER_DATASET_ROOT" \
        --smoke "$SMOKE" \
        --seed "$SEED" \
        --output "$OUT/cache"
else
    echo "Reuse completed extraction: $OUT/cache"
fi

# Train and test both paper variants on the SAME extracted candidates.
# In RESUME mode, an existing checkpoint is reused; partial evaluation output
# is never overwritten. The optimized rerun goes to validation_<variant>_fast.
for variant in d2d gossip
do
    echo "===== MODEL: $variant ====="
    TRAIN_DIR="$OUT/train_$variant"
    CHECKPOINT="$TRAIN_DIR/last.pt"

    if [ "$RESUME" = "1" ] && [ -f "$CHECKPOINT" ]; then
        echo "Reuse checkpoint: $CHECKPOINT"
    else
        if [ -e "$TRAIN_DIR" ]; then
            echo "Training directory exists but checkpoint is missing: $TRAIN_DIR" >&2
            echo "Do not delete it until you have inspected the partial run." >&2
            exit 1
        fi
        "$PY" -u -m Learned3DNMS.train \
            --cache "$OUT/cache" \
            --output "$TRAIN_DIR" \
            --variant "$variant" \
            --epochs "$EPOCHS" \
            --seed "$SEED"
    fi

    if [ "$RESUME" = "1" ]; then
        EVAL_DIR="$OUT/validation_${variant}_fast"
    else
        EVAL_DIR="$OUT/validation_$variant"
    fi

    if [ -f "$EVAL_DIR/results.md" ]; then
        echo "Reuse completed evaluation: $EVAL_DIR/results.md"
    else
        if [ -e "$EVAL_DIR" ]; then
            echo "Evaluation directory exists but is incomplete: $EVAL_DIR" >&2
            exit 1
        fi
        "$PY" -u -m Learned3DNMS.evaluate \
            --cache "$OUT/cache" \
            --checkpoint "$CHECKPOINT" \
            --output "$EVAL_DIR"
    fi
    echo "COMPLETED: $EVAL_DIR/results.md"
done

printf 'BOTH VARIANTS TRAIN + TEST COMPLETE\n'
if [ "$RESUME" = "1" ]; then
    printf 'D2D:    %s/validation_d2d_fast/results.md\n' "$OUT"
    printf 'Gossip: %s/validation_gossip_fast/results.md\n' "$OUT"
else
    printf 'D2D:    %s/validation_d2d/results.md\n' "$OUT"
    printf 'Gossip: %s/validation_gossip/results.md\n' "$OUT"
fi
