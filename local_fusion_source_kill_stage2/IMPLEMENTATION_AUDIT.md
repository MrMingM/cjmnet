# Stage 2 implementation audit — 2026-10-14

**STAGE2 IMPLEMENTATION READY** means the implementation is ready for server
verification. It does not mean K1/K2, source reproduction, or scientific
hypotheses have passed. No local scientific run was performed.

## Local verification

- Python AST/import checks and pyflakes: passed.
- Unit/schema/fixture tests: 66 cases, 63 passed, 3 intentionally skipped on
  Windows (POSIX symlink guard and two synthetic Torch representation tests).
  The three run on the server through run_preflight.sh.
- Pure synthetic smoke: passed original Oracle proxy/shortlist/greedy,
  fixture postprocess boundary, atomic save/resume, and report state machine.
- POSIX sh -n: all 5 scripts passed; shell files use LF.
- Git diff --check: passed. Main checkout retained; no branch/worktree created.
- Stage 1 original six reports: hashes unchanged.
- All original Stage 1/P0 producer files listed in formal P0 manifest: hashes unchanged.
- Formal P0 report hashes: unchanged. Frozen GSPR source check passed.

## Section 42 implementation mapping

| Frozen requirement | Implementation |
|---|---|
| Four-weather Source AP30/50/70 reproduction <=1e-6 before any formal null | k1_source_specificity + original reproduction gate |
| Original Same/Task, target order, proxy, shortlist, compose, NMS, full key | source_adapter executes original run_oracle code object |
| Same <=6, Task three top-k union | original _shortlist_actions, runtime count checks |
| N1 entire decoded output vector, train only | nulls + calibration |
| N2 exact valid CAV count / 2N candidates / same proxy | nulls + source_adapter runtime assertions |
| Train-only alpha selection frozen before validation | calibration; immutable NULL_CONFIG / K1 protocol extension |
| Score-only geometry exactly preserved | original Shared rm tensor |
| Three adverse seeds; Clean first seed sanity | protocol + K1 driver |
| Paired scene bootstrap, 5000, seed20261014 | bootstrap; recompute AP from TP/FP/GT |
| Same/Task independent K1 verdict, unchanged fail rules | verdicts |
| Same 259/9/55 max and Task64/8/19 max baseline provenance | preflight + sample_gate; human-authorized Same description correction |
| Eight small perturbations; actual current-state evaluation | k2_label_stability + evaluate_target |
| Physical ROI .97 despite legacy helper's >=1 guard | Stage 2 footprint adapter, actual NumPy mask test, legacy helper untouched |
| Equivalent discrete-best source/pair sets; exact ID reported separately | stability |
| Family macro metrics + worst variant; unchanged thresholds | stability + verdicts |
| Scene-level correlation and raw Task stable/flip counts | bootstrap + K2 report/raw JSONL |
| Test realpath/alias/nested-path refusal before data iteration | preflight guards + loader guards |
| Fixed resource stop from formal P0, no new clock | common + preflight + every worker/condition entry |
| Isolated manifests/locks/cache; immutable config/input/source hashes | common + driver |
| K3 frozen; gates recorded only | protocol + final state machine; no trainer |

## Server checks still required

Actual dataset/checkpoint preflight, the three POSIX/Torch fixtures, two real
Source-frame smoke checks, and four-weather Source reproduction run only on
the server. Failure exits nonzero; no tolerance is relaxed. Scientific results
are written only by server workers and remain unknown at delivery.

The root protocol contains every frozen rule and the predeclared alpha
selection procedure. Actual alpha and train delta pool hashes are frozen in
an immutable K1 protocol extension before validation. The root protocol is
never edited after results begin.
