# CIA-SSD-style IoU quality: frozen F, fixed top256

Development validation only. The 9 scenes have been reused in earlier work.
All methods use the same top256 pool, original rotated NMS, range filter,
and the original scorepass output count as each frame's budget.

| Weather | Method | AP70 frame | AP70 global | Oracle recovery frame | Oracle recovery global | Spearman | IoU70 AUC | PR-AUC | Recovered GT | Lost GT | New FP |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| clean | original_f | 0.8223 | 0.8811 | 0.0000 | 0.0000 | 0.5884 | 0.8291 | 0.7964 | 0 | 0 | 0 |
| clean | top256_fused | 0.8223 | 0.8811 | 0.0000 | 0.0000 | 0.5884 | 0.8291 | 0.7964 | 0 | 0 | 0 |
| clean | direct_iou_regression | 0.8151 | 0.8893 | -0.0632 | 0.1060 | 0.8733 | 0.9669 | 0.9496 | 19 | 24 | 195 |
| clean | ciassd_score | 0.8275 | 0.8908 | 0.0463 | 0.1256 | 0.7126 | 0.8945 | 0.8593 | 12 | 6 | 78 |
| clean | predicted_iou_only | 0.7670 | 0.8279 | -0.4885 | -0.6938 | 0.7940 | 0.9330 | 0.8974 | 28 | 82 | 305 |
| clean | gt_iou_oracle | 0.9354 | 0.9578 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 99 | 4 | 94 |
| fog | original_f | 0.7858 | 0.8447 | 0.0000 | 0.0000 | 0.6025 | 0.8410 | 0.8004 | 0 | 0 | 0 |
| fog | top256_fused | 0.7858 | 0.8447 | 0.0000 | 0.0000 | 0.6025 | 0.8410 | 0.8004 | 0 | 0 | 0 |
| fog | direct_iou_regression | 0.7865 | 0.8540 | 0.0054 | 0.1096 | 0.8735 | 0.9642 | 0.9433 | 22 | 23 | 173 |
| fog | ciassd_score | 0.7876 | 0.8519 | 0.0149 | 0.0851 | 0.7229 | 0.9022 | 0.8619 | 4 | 3 | 66 |
| fog | predicted_iou_only | 0.7195 | 0.7928 | -0.5446 | -0.6145 | 0.7877 | 0.9313 | 0.8925 | 23 | 86 | 295 |
| fog | gt_iou_oracle | 0.9075 | 0.9293 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 112 | 5 | 85 |
| rain | original_f | 0.8127 | 0.8735 | 0.0000 | 0.0000 | 0.5945 | 0.8321 | 0.7962 | 0 | 0 | 0 |
| rain | top256_fused | 0.8127 | 0.8735 | 0.0000 | 0.0000 | 0.5945 | 0.8321 | 0.7962 | 0 | 0 | 0 |
| rain | direct_iou_regression | 0.7966 | 0.8784 | -0.1353 | 0.0589 | 0.8756 | 0.9665 | 0.9487 | 14 | 27 | 204 |
| rain | ciassd_score | 0.8128 | 0.8821 | 0.0013 | 0.1039 | 0.7173 | 0.8963 | 0.8592 | 10 | 6 | 68 |
| rain | predicted_iou_only | 0.7464 | 0.8246 | -0.5576 | -0.5879 | 0.7944 | 0.9331 | 0.8968 | 23 | 79 | 321 |
| rain | gt_iou_oracle | 0.9315 | 0.9566 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 107 | 3 | 93 |
| snow | original_f | 0.6012 | 0.6844 | 0.0000 | 0.0000 | 0.5500 | 0.8300 | 0.7338 | 0 | 0 | 0 |
| snow | top256_fused | 0.6012 | 0.6844 | 0.0000 | 0.0000 | 0.5500 | 0.8300 | 0.7338 | 0 | 0 | 0 |
| snow | direct_iou_regression | 0.6087 | 0.7003 | 0.0366 | 0.1053 | 0.8362 | 0.9478 | 0.8941 | 52 | 46 | 215 |
| snow | ciassd_score | 0.6116 | 0.6965 | 0.0506 | 0.0803 | 0.6500 | 0.8822 | 0.7933 | 24 | 12 | 75 |
| snow | predicted_iou_only | 0.5179 | 0.6168 | -0.4057 | -0.4486 | 0.6818 | 0.8966 | 0.8098 | 55 | 140 | 379 |
| snow | gt_iou_oracle | 0.8067 | 0.8351 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 224 | 27 | 90 |

AP30/AP50/AP70 for both ordering rules and all candidate diagnostics are in `results.json`.
A negative recovery ratio means AP70 fell below original F. The denominator is the same fixed-budget GT-IoU Oracle AP70 minus original F AP70.
The IoU-only score is diagnostic; `ciassd_score` is the main CIA-SSD confidence function.
The oracle uses GT only after freezing the candidate pool and is not deployable.
