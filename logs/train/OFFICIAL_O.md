# AttFuse: official O protocol

FULL VALIDATION
Validation frames per condition: 829/829; selected epoch: 20.

Official weather GT and None-frame skipping; original dataset/model/postprocess/AP.
This reports the selected V2X-R Clean-trained model. No training is performed.

| Condition | AP30 | AP50 | AP70 | Official GT | Counted GT | Skipped None frames |
|---|---:|---:|---:|---:|---:|---:|
| clean | 0.87655091 | 0.84707141 | 0.70539766 | 6224 | 6221 | 43 |
| fog | 0.56354597 | 0.55077779 | 0.44212603 | 3673 | 3616 | 164 |
| snow | 0.46624621 | 0.43523409 | 0.28892218 | 5854 | 5739 | 114 |

Source summary: /data/cjm/datasets/logs/v2xr-attfuse-official-protocol-v1/inference_full/summary.json
Checkpoint SHA256: `ccf0211d04f732246d53b70faf8c3f6eb494c19963177d9cedf3d2b106cf01d3`.
Export verifies existing artifacts; rerun executes the frozen inference entry point.
The underlying evaluator also stores O*/F audit outputs; this report selects O only.
