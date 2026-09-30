# Fixed physics weather dataset

This one-time server job saves the project's existing physics Fog/Rain/Snow
simulation as OPV2V-compatible PCD files. Later experiments read those files
directly. It uses **official clean train and validate only**. The existing
OPV2V-W weather test sets remain the formal test data.

## Generate once on the server

Run from `/home/cjm/OpenCOOD-main/cjmnet`, using the frontend YAML for the
frozen F checkpoint. Keep the generator log on the data disk.
Synchronize `materialize_physics_weather.py`,
`test_materialize_physics_weather.py`, and
`run_materialize_physics_weather.sh` together. The launcher checks that the
generator is present before running the tests or writing data.

```sh
mkdir -p /data/cjm/datasets/logs
nohup env FRONTEND_CONFIG=/path/to/the/frozen_F_frontend.yaml \
  sh opencood/tools/run_materialize_physics_weather.sh \
  > /data/cjm/datasets/logs/materialize_physics_weather.log 2>&1 &
```

If interrupted, run the same command with `RESUME=1`. The generator refuses
to overwrite an unrelated directory or a dataset generated with a different
config/source version. Output PCDs are written atomically; `manifest.json`
is marked `complete` only after both splits finish. Every 100th PCD per
weather is read back through the project's Open3D reader by default. Server
unit checks:

```sh
/home/cjm/miniconda3/envs/opencood/bin/python -m unittest \
  opencood.tools.test_materialize_physics_weather
```

The September 2026 initial generator could stop on official frame YAML that
contains NumPy scalar tags. After copying the corrected generator and test,
resume with `RESUME=1`. The manifest accepts this one documented generator
hash migration only while its status is `in_progress`; it still checks all
configuration, source, and lookup-table hashes and reuses completed PCDs.

The dataset root is `/data/cjm/datasets/opv2v-physics-fixed-v1`:

```text
fog/{train,validate}/<scene>/<cav>/<timestamp>.pcd
rain/{train,validate}/...
snow/{train,validate}/...
mixed/{train,validate}/...     # fixed per-frame Fog/Rain/Snow selection
manifest.json
```

All annotations and camera paths are symlinks to the official clean source.
The `mixed` PCDs are symlinks to one of the three generated conditions. These
links require the official source and the three weather directories to remain
in place. The source data is never modified.

## Use in later experiments

For a weather-specific run, set the model's `root_dir` and `validate_dir` to
the chosen condition's `train` and `validate` directories, and remove the
`weather_augmentation` setting. For example:

```yaml
root_dir: /data/cjm/datasets/opv2v-physics-fixed-v1/fog/train
validate_dir: /data/cjm/datasets/opv2v-physics-fixed-v1/fog/validate
```

For one fixed mixed-weather training set, use `mixed/train`. Clean uses the
official OPV2V roots. The SAQC train/calibrate/development scripts select these
paths automatically through `WEATHER_DATASET_ROOT` (default: the root above)
and reject an incomplete or mismatched manifest.

This freezes the epoch-0 random realization. It is reproducible and removes
per-model weather simulation, but it also removes the online pipeline's fresh
weather draws in later epochs. It uses the same local-sensor augmentors,
random-key convention, and clean/weather empty-voxel fallback. PCD RGB encoding
quantizes intensity to 8 bits. When read as a normal OPV2V split, the loader
also shuffles point order before voxelization. Thus voxel truncation may differ
from the old online weather branch even though the saved point set is fixed.
Keep this as a distinct training protocol in result tables. Do not describe its
validation as OPV2V-W test.
