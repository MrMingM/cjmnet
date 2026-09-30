#!/bin/sh
set -eu

: "${RUN:?Set RUN to the completed adaptation run containing F.pth}"
: "${FRONTEND_CONFIG:?Set FRONTEND_CONFIG}"
: "${FRONTEND_CHECKPOINT:?Set FRONTEND_CHECKPOINT}"
: "${V3_CHECKPOINT:?Set V3_CHECKPOINT}"

PY=${PY:-/home/cjm/miniconda3/envs/opencood/bin/python}
V3_CONFIG=${V3_CONFIG:-local_fusion_v3/experiment.yaml}
WEATHER_DATASET_ROOT=${WEATHER_DATASET_ROOT:-/data/cjm/datasets/opv2v-physics-fixed-v1}
SMOKE=${SMOKE:-0}
EPOCHS=${EPOCHS:-10}
SEED=${SEED:-20260930}
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

# Fail-fast checks; never train if tensor invariants do not hold.
"$PY" -m unittest Learned3DNMS.test_learned_nms -v

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

# Train and test both paper variants on the SAME extracted candidates.
# The next variant starts only if the previous train + test succeeds.
for variant in d2d gossip
do
    echo "===== MODEL: $variant ====="
    "$PY" -u -m Learned3DNMS.train \
        --cache "$OUT/cache" \
        --output "$OUT/train_$variant" \
        --variant "$variant" \
        --epochs "$EPOCHS" \
        --seed "$SEED"

    # Automatic evaluation after this variant's successful training.
    "$PY" -u -m Learned3DNMS.evaluate \
        --cache "$OUT/cache" \
        --checkpoint "$OUT/train_$variant/last.pt" \
        --output "$OUT/validation_$variant"
    echo "COMPLETED: $OUT/validation_$variant/results.md"
done

printf 'BOTH VARIANTS TRAIN + TEST COMPLETE\n'
printf 'D2D:    %s/validation_d2d/results.md\n' "$OUT"
printf 'Gossip: %s/validation_gossip/results.md\n' "$OUT"
