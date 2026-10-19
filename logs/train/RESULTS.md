# D-1D validation protocol results

Frames per condition: 829/829. FULL VALIDATION

Associated D-1D Clean training inventory: 8083 actual frames; published reference 8084; deviation -1 explicitly accepted before training.
This train-inventory statement applies to the D-1D run, not to OPV2V checkpoint pretraining.

O: condition-specific official GT; None skips frame. O*: same GT, None counts as zero predictions.
F: frozen D-1C Official Clean-selected paired fixed GT; None counts as zero predictions.
All three reuse one prediction cache and the unchanged official BEV/VOC2010 evaluator, global_sort=False.

| Condition | Official GT | Fixed GT | Removed | None frames | Official GT in None frames |
|---|---:|---:|---:|---:|---:|
| clean | 6224 | 6224 | 0 | 43 | 3 |
| fog | 3673 | 6224 | 2551 | 164 | 57 |
| snow | 5854 | 6224 | 370 | 114 | 115 |

| Condition | Protocol | AP30 | AP50 | AP70 |
|---|---|---:|---:|---:|
| clean | O | 0.87655091 | 0.84707141 | 0.70539766 |
| clean | O* | 0.87612841 | 0.84666312 | 0.70505766 |
| clean | F | 0.87612841 | 0.84666312 | 0.70505766 |
| fog | O | 0.56354597 | 0.55077779 | 0.44212603 |
| fog | O* | 0.55480050 | 0.54223046 | 0.43526483 |
| fog | F | 0.32740717 | 0.31998915 | 0.25686499 |
| snow | O | 0.46624621 | 0.43523409 | 0.28892218 |
| snow | O* | 0.45708695 | 0.42668405 | 0.28324639 |
| snow | F | 0.42991436 | 0.40131884 | 0.26640816 |

| Protocol | Clean AP70 | Fog AP70 | Snow AP70 | Clean-Fog pp | Clean-Snow pp |
|---|---:|---:|---:|---:|---:|
| O | 0.70539766 | 0.44212603 | 0.28892218 | 26.3272 | 41.6475 |
| O* | 0.70505766 | 0.43526483 | 0.28324639 | 26.9793 | 42.1811 |
| F | 0.70505766 | 0.25686499 | 0.26640816 | 44.8193 | 43.8649 |

| Weather | AP70 NoneSkipEffect pp | AP70 GTFilterEffect pp | AP70 GapMasking pp |
|---|---:|---:|---:|
| fog | 0.6861 | 17.8400 | 18.4921 |
| snow | 0.5676 | 1.6838 | 2.2174 |

These are descriptive protocol differences. O supports benchmark comparison; F is the robustness main result; O* audits None frames.
A large positive GTFilterEffect/GapMasking indicates that official scoring hides degradation on fixed Clean targets.
Near-zero effects mean small AP impact for these predictions, even when many GT were removed. Removed-target recall alone cannot establish small AP impact.
Single-seed validation differences do not establish statistical significance.

Checkpoint SHA256: `ccf0211d04f732246d53b70faf8c3f6eb494c19963177d9cedf3d2b106cf01d3`.
Frozen D-1C GT identity: `d5244c4a8fa78dc1d26df1a7ee1f4592c1dd24f91a3f9c575313a081bbeab776`.
