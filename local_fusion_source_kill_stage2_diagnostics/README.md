# K2 single-frame reproduction troubleshooting

Only sync this NEW directory to the server. Do not change Stage 2 code,
source_snapshot, PROTOCOL, caches, manifests, or historical results while K1
is running. This directory is outside the frozen Stage 2 source hash set.

The traceback alone does not tell whether the difference is a floating value,
selected action, shortlist, discrete key, or metadata. This tool reports exact
field paths and values without introducing any tolerance or acceptance rule.

Default: select the first pending primary K2 frame in the original worker
iteration order. Replay the original full ordered loader up to that frame,
preserving online weather RNG; evaluate only this frame on GPU.

1. Execute the original Stage 1 A_ONLY adapter, then compare to saved Stage 1.
2. Repeat the original adapter on identical inputs to check determinism.
3. Execute Stage 2 hook adapter without perturbations; compare both adapters.
4. Only when these match, instrument the existing K2 evaluate_frame to capture
   its records and check whether interleaving perturbation callbacks changes
   the reproduced baseline.

Also compare Shared prediction hash with the original S0 exact snapshot hash,
after verifying the one-frame cache hash. Record the GPU/software environment.
This is provenance information, not an automatic accept/reject criterion.

All output goes to `RUN/k2_diagnostics/<timestamp_pid>/`. No scientific verdicts,
training, nulls, K3, formal test, or changes to original K1/K2 result files.
Respect the original resource-stop deadline and existing input/code hashes.

Server background command (GPU 3 must be available):

```sh
cd /home/cjm/OpenCOOD-main/cjmnet
export PYTHONPATH=/home/cjm/OpenCOOD-main/cjmnet:/home/cjm/OpenCOOD-main
export ROCR_VISIBLE_DEVICES=3
unset HIP_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES
RUN=/data/cjm/datasets/logs/source_kill_stage2_20261014_143004
nohup /home/cjm/miniconda3/envs/opencood/bin/python -u \
  -m local_fusion_source_kill_stage2_diagnostics.replay --run "$RUN" \
  >"$RUN/k2-replay-diagnostic.log" 2>&1 </dev/null &
echo "PID=$!"
tail -f "$RUN/k2-replay-diagnostic.log"
```

For a known failing frame, add `--weather fog --mode same --frame INDEX`.
Use `--baseline-only` to skip the original perturbation callback probe.
Output log shows report path, comparisons, and the first exact differing field.
Read DIAGNOSIS.md/JSON before deciding any actual repair. Small floating
differences remain mismatches; this tool does not relax the frozen gate.

After a replay diagnostic, inspect ALL saved field differences using CPU only:

```sh
python -m local_fusion_source_kill_stage2_diagnostics.inspect \
  --diagnosis /data/cjm/datasets/logs/source_kill_stage2_20261014_143004/k2_diagnostics/20261014_070743_547924_3927511
```

This reads existing JSON, reports maximum continuous drift and exact changes
to action identities/discrete keys/metadata, and checks pairwise full9 order
including ties. A_ONLY legal pairs are reported separately from ineligible
B/E actions. No tolerance is applied, no GPU inference is performed, and no
K1/K2 results or code hashes are changed.

Local checks are pure synthetic:

```sh
python -m unittest local_fusion_source_kill_stage2_diagnostics.test_core
```
