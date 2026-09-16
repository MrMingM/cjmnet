# Lossless communication baselines

This package is intentionally isolated from frozen GSPR-v1 files.

## What it evaluates

1. `raw_full`
   - Existing project full communication.
   - All three AttFuse backbone scales are serialized in the old format.

2. `lossless_full`
   - Transmits every value of all three scales.
   - Exact float32 bit patterns are restored.
   - ReLU +0.0 may be represented by a bitmap.
   - Remaining float bytes are shuffled and compressed with Zstandard when
     available, otherwise zlib.
   - No feature selection, no quantization, no retraining.

3. `level0_recompute`
   - Transmits the complete first backbone scale only.
   - Receiver reruns the frozen `backbone.blocks[1]` and `[2]` for that peer.
   - Then the original scale-wise AttFuse path is used.
   - Recomputed scales are checked against sender-side scales.

4. `original_full`
   - Direct frozen GSPR-AttFuse forward for numerical path verification.

## Why level0-only is meaningful here

The current frozen backbone is sequential:

`level0 = block0(canvas)`
`level1 = block1(level0)`
`level2 = block2(level1)`

The block payload per shared coarse grid cell is:

- level0: `64 * 4 * 4 = 1024` float32 values
- level1: `128 * 2 * 2 = 512`
- level2: `256 * 1 * 1 = 256`
- total: `1792`

So level0-only removes `768 / 1792 = 42.86%` of the *raw feature values*
before lossless entropy coding. This is not a promise about final wire bytes:
packet metadata and actual feature entropy still matter.

## Dependency

No new dependency is required: `zlib` is built into Python.

For the preferred Zstandard experiment:

```bash
/home/cjm/miniconda3/envs/opencood/bin/pip install zstandard
```

This does not modify model code or checkpoints. If changing the environment is
undesirable, use `BACKEND=zlib`.

## First run: codec test

```bash
cd /home/cjm/OpenCOOD-main/cjmnet
PYTHONPATH=/home/cjm/OpenCOOD-main:/home/cjm/OpenCOOD-main/cjmnet \
/home/cjm/miniconda3/envs/opencood/bin/python -m unittest lossless_comm.test_codec -v
```

## Development evaluation

Full OPV2V validation, online physical weather for fog/rain/snow,
non-global AP:

```bash
cd /home/cjm/OpenCOOD-main/cjmnet

nohup env \
  PHASE=development \
  GPU=0 \
  BACKEND=auto \
  bash lossless_comm/run_all.sh \
  > /data/cjm/datasets/logs/lossless_comm_development_launcher.log 2>&1 &

tail -f /data/cjm/datasets/logs/lossless_comm_development_launcher.log
```

Do not call this OPV2V-W formal testing.

## Formal benchmark

Only after the code path is validated:

```bash
cd /home/cjm/OpenCOOD-main/cjmnet

nohup env \
  PHASE=benchmark \
  GPU=0 \
  BACKEND=auto \
  bash lossless_comm/run_all.sh \
  > /data/cjm/datasets/logs/lossless_comm_benchmark_launcher.log 2>&1 &

tail -f /data/cjm/datasets/logs/lossless_comm_benchmark_launcher.log
```

Formal mode follows the existing project benchmark loader:
OPV2V clean test + existing OPV2V-W fog/rain/snow test, no online weather,
all frames, non-global AP.

## Acceptance logic

For `lossless_full`, expected behavior is essentially identical logits/AP to
`raw_full`; if not, stop and debug the transport path.

For `level0_recompute`, first inspect:

- `max_level0_error`: must be 0.
- `max_level1_error`, `max_level2_error`: should be numerical-noise scale.
- AP delta vs `raw_full`.
- real `mean_bytes`.
- encode/decode/recompute milliseconds.

Only after these checks should communication savings be used in the paper.
