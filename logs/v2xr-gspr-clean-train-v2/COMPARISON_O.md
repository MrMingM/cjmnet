# GSPR vs plain AttFuse: same V2X-R Clean training and official O

Both random-init, seed20261017, 20 epochs, Clean-only selection; full829-frame validation.
Baseline epoch20 is highest under both its original F selection and O.
GSPR adds soft point weighting plus fixed keep/evidence regularization; no point labels.
Single seed; validation-selected results do not establish statistical significance or test generalization.

| Condition | Metric | Plain AttFuse | GSPR-AttFuse | GSPR - plain / pp |
|---|---|---:|---:|---:|
| clean | ap30 | 0.87655091 | 0.85186399 | -2.4687 |
| clean | ap50 | 0.84707141 | 0.82524928 | -2.1822 |
| clean | ap70 | 0.70539766 | 0.66119491 | -4.4203 |
| fog | ap30 | 0.56354597 | 0.55213470 | -1.1411 |
| fog | ap50 | 0.55077779 | 0.53726272 | -1.3515 |
| fog | ap70 | 0.44212603 | 0.41628316 | -2.5843 |
| snow | ap30 | 0.46624621 | 0.43636937 | -2.9877 |
| snow | ap50 | 0.43523409 | 0.41014700 | -2.5087 |
| snow | ap70 | 0.28892218 | 0.27924909 | -0.9673 |
