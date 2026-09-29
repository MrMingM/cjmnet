# SAQC benchmark for cjmnet

This directory implements **Spatially-Aware Quality Calibration (SAQC)** as an
external reliability benchmark on the project's frozen F detector.

The purpose is not to claim SAQC as a new contribution. The question is:

> How much of the existing GT-IoU candidate-quality Oracle can an established
> spatial quality method recover on adverse-weather cooperative PointPillar
> detection?

## What is frozen

The GSPR frontend, PointPillar backbone, Where2comm/F fusion arm,
classification head, regression head and decoded candidate geometry remain
frozen. SAQC learns only a localization-quality head on local fused BEV
patches.

## Architecture adaptation

The SAQC paper is center-based and explicitly states that anchor-based
detectors need a new detection-to-feature association rule. This implementation
maps each decoded candidate `(x,y)` center to the nearest cell center of the
frozen anchor BEV grid and crops the local patch there.

See `METHOD_MAPPING.md` before interpreting results.

## Main files

- `model.py`: Local Spatial Quality Head (LSQH).
- `adapter.py`: frozen F feature extraction, decoded-center/grid mapping and patch extraction.
- `train.py`: full official-train weather-mixed LSQH training.
- `calibrate.py`: optional train-only soft-IoU Platt calibration.
- `evaluate.py`: development and historical OPV2V/OPV2V-W benchmark evaluation.
- `test_saqc.py`: CPU unit tests.

## Unit test

```sh
python -m unittest SAQC.test_saqc
```

## Smoke training

Set the same frozen F stack used by
`local_fusion_detector_adaptation`:

```sh
RUN=/data/.../fusion_detector_adaptation_... \
FRONTEND_CONFIG=... \
FRONTEND_CHECKPOINT=... \
V3_CHECKPOINT=... \
SMOKE=2 EPOCHS=1 \
sh SAQC/run_train.sh
```

`SMOKE=0` means the complete official train split. The main protocol trains
one shared quality model over Clean/Fog/Rain/Snow, not four weather-specific
models.

## Development evaluation

```sh
RUN=/data/... \
FRONTEND_CONFIG=... \
FRONTEND_CHECKPOINT=... \
V3_CHECKPOINT=... \
SAQC_CHECKPOINT=/data/.../saqc_quality.pth \
PHASE=development SMOKE=2 \
sh SAQC/run_eval.sh
```

After the smoke run passes, use `SMOKE=0` for the full development split.

## Optional train-only calibration

Calibration is not used to claim AP gains. SAQC ranking AP uses the
pre-calibration fused score. Calibration is included for reliability analysis.

```sh
RUN=/data/... \
FRONTEND_CONFIG=... \
FRONTEND_CHECKPOINT=... \
V3_CHECKPOINT=... \
SAQC_CHECKPOINT=/data/.../saqc_quality.pth \
WEATHER=clean \
sh SAQC/run_calibrate.sh
```

Pass the resulting JSON through `CALIBRATION=...` to `run_eval.sh` if
calibrated Q-ECE-style metrics are desired.

## Frozen historical benchmark

Only after train/development choices are frozen:

```sh
RUN=/data/... \
FRONTEND_CONFIG=... \
FRONTEND_CHECKPOINT=... \
V3_CHECKPOINT=... \
SAQC_CHECKPOINT=/data/.../saqc_quality.pth \
PHASE=benchmark SMOKE=0 \
sh SAQC/run_eval.sh
```

Benchmark mode reuses the project's fixed paths for OPV2V clean test and
OPV2V-W fog/rain/snow test and disables online weather augmentation.

## Main comparison

Evaluation reports:

- original F;
- SAQC on the original scorepass pool;
- top256 with original F score;
- top256 with SAQC score;
- top256 GT-IoU Oracle.

The top256 branch uses the same original rotated NMS, range filter and
per-frame final-output budget as original F.

The global-sort AP70 Oracle recovery ratio is:

`(top256_SAQC - original_F) / (top256_GT-IoU_Oracle - original_F)`.

A high Oracle is only an upper bound; it does not imply that SAQC is learnable
or will recover that gap.
