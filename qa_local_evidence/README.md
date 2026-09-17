# H-A5 local evidence audit

This directory is an isolated, development-only diagnostic for H-A5. It does **not** modify the frozen GSPR frontend, AttFuse, detector, Stage-3 code, or postprocessor.

## Question

For targets that are detected by clean ego but missed by weather ego, Stage-1 sometimes labels the weather ego observation as `strong` using N_eff/coverage. H-A5 has two competing explanations:

- **H-A5a:** the sensing-side ruler overestimates target evidence;
- **H-A5b:** target-usable local evidence survives, but the local detection pipeline fails to preserve/use it.

The audit replays paired clean/weather ego-only branches from the same validation sample and traces the exact frozen detector path. It never uses OPV2V-W test data.

## What it can conclude

A weather ego miss is direct evidence for a late A5b-type local downstream loss when an IoU>=0.7 decoded proposal already exists before final filtering. For cases with no correct decoded box, the exact anchor that matched the clean target is inspected in weather: if its classification score still passes the normal threshold while localization fails, this supports a regression/localization bottleneck.

If a metric-strong case has neither signal, it is **A5a-compatible/upstream-unresolved**, not proof of A5a. A deeper PillarVFE/backbone feature intervention is required for that subset.

## Outputs

- `a5_report.md`: human-readable summary;
- `a5_results.json`: machine-readable aggregate results;
- `{fog,rain,snow}/targets.jsonl`: per-target paired clean/weather traces;
- `{fog,rain,snow}/protocol.json`: lineage and interpretation boundaries.

## Run

Use a physical HCU in 0-6 that is not occupied by Stage-3. HCU 7 is reserved.

```bash
ROCR_VISIBLE_DEVICES=5 bash qa_local_evidence/run_all.sh
```

Smoke first with the first 5 candidate frames per weather:

```bash
ROCR_VISIBLE_DEVICES=5 SMOKE=5 bash qa_local_evidence/run_all.sh
```

For background execution, wrap the script with `nohup` and redirect the launcher log to `/data/cjm/datasets/logs/`.
