# V2X-R Clean-trained GSPR-AttFuse: official O

Random initialization; V2X-R Clean only; no OPV2V weights or point/noise labels.
Main metric: O / OFFICIAL_EXACT. Model adapter changes point features, never GT or AP.
Frames per condition: 829/829; FULL VALIDATION

| Condition | O AP30 | O AP50 | O AP70 | Official GT | Counted GT | None frames |
|---|---:|---:|---:|---:|---:|---:|
| clean | 0.85186399 | 0.82524928 | 0.66119491 | 6224 | 6216 | 43 |
| fog | 0.55213470 | 0.53726272 | 0.41628316 | 3673 | 3599 | 149 |
| snow | 0.43636937 | 0.41014700 | 0.27924909 | 5854 | 5732 | 99 |

O*/F audit details follow. These audits do not affect O selection or official matching.

# D-1D validation protocol results

Frames per condition: 829/829. FULL VALIDATION

Associated D-1D Clean training inventory: 8083 actual frames; published reference 8084; deviation -1 explicitly accepted before training.
This train-inventory statement applies to the D-1D run, not to OPV2V checkpoint pretraining.

O: condition-specific official GT; None skips frame. O*: same GT, None counts as zero predictions.
F: frozen D-1C Official Clean-selected paired fixed GT; None counts as zero predictions.
All three reuse one prediction cache and the unchanged official BEV/VOC2010 evaluator, global_sort=False.

| Condition | Official GT | Fixed GT | Removed | None frames | Official GT in None frames |
|---|---:|---:|---:|---:|---:|
| clean | 6224 | 6224 | 0 | 43 | 8 |
| fog | 3673 | 6224 | 2551 | 149 | 74 |
| snow | 5854 | 6224 | 370 | 99 | 122 |

| Condition | Protocol | AP30 | AP50 | AP70 |
|---|---|---:|---:|---:|
| clean | O | 0.85186399 | 0.82524928 | 0.66119491 |
| clean | O* | 0.85076905 | 0.82418855 | 0.66034505 |
| clean | F | 0.85076905 | 0.82418855 | 0.66034505 |
| fog | O | 0.55213470 | 0.53726272 | 0.41628316 |
| fog | O* | 0.54101083 | 0.52643847 | 0.40789630 |
| fog | F | 0.31926941 | 0.31066975 | 0.24071386 |
| snow | O | 0.43636937 | 0.41014700 | 0.27924909 |
| snow | O* | 0.42727523 | 0.40159935 | 0.27342941 |
| snow | F | 0.40187487 | 0.37772535 | 0.25717477 |

| Protocol | Clean AP70 | Fog AP70 | Snow AP70 | Clean-Fog pp | Clean-Snow pp |
|---|---:|---:|---:|---:|---:|
| O | 0.66119491 | 0.41628316 | 0.27924909 | 24.4912 | 38.1946 |
| O* | 0.66034505 | 0.40789630 | 0.27342941 | 25.2449 | 38.6916 |
| F | 0.66034505 | 0.24071386 | 0.25717477 | 41.9631 | 40.3170 |

| Weather | AP70 NoneSkipEffect pp | AP70 GTFilterEffect pp | AP70 GapMasking pp |
|---|---:|---:|---:|
| fog | 0.8387 | 16.7182 | 17.4719 |
| snow | 0.5820 | 1.6255 | 2.1224 |

These are descriptive protocol differences. O supports benchmark comparison; F is the robustness main result; O* audits None frames.
A large positive GTFilterEffect/GapMasking indicates that official scoring hides degradation on fixed Clean targets.
Near-zero effects mean small AP impact for these predictions, even when many GT were removed. Removed-target recall alone cannot establish small AP impact.
Single-seed validation differences do not establish statistical significance.

Checkpoint SHA256: `87941d2d31e10e707c94055c3bebd1ce301e847c58f3e0c62981821aebf13fd5`.
Frozen D-1C GT identity: `d5244c4a8fa78dc1d26df1a7ee1f4592c1dd24f91a3f9c575313a081bbeab776`.
