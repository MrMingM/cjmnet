# Learned 3D NMS (d2d) fixed physics validation

AP70 values use identical frozen F/top256 candidate sets.
Frame-order and global-sort AP must never be compared directly.

| Weather | Method | Frame AP70 | Global AP70 | Global Oracle Recovery |
|---|---|---:|---:|---:|
| clean | original_f | 0.755802 | 0.879868 | 0.0000 |
| clean | top256_original_nms | 0.755802 | 0.879868 | 0.0000 |
| clean | d2d_topk | 0.745781 | 0.872190 | -0.0904 |
| clean | d2d_original_nms | 0.751024 | 0.875290 | -0.0539 |
| clean | gt_iou_oracle | 0.919738 | 0.964781 | 1.0000 |
| fog | original_f | 0.726541 | 0.836885 | 0.0000 |
| fog | top256_original_nms | 0.726541 | 0.836885 | 0.0000 |
| fog | d2d_topk | 0.714677 | 0.829128 | -0.0787 |
| fog | d2d_original_nms | 0.721620 | 0.833258 | -0.0368 |
| fog | gt_iou_oracle | 0.890617 | 0.935389 | 1.0000 |
| rain | original_f | 0.747099 | 0.872398 | 0.0000 |
| rain | top256_original_nms | 0.747099 | 0.872398 | 0.0000 |
| rain | d2d_topk | 0.735991 | 0.864022 | -0.0940 |
| rain | d2d_original_nms | 0.741817 | 0.867279 | -0.0575 |
| rain | gt_iou_oracle | 0.915209 | 0.961499 | 1.0000 |
| snow | original_f | 0.624367 | 0.756147 | 0.0000 |
| snow | top256_original_nms | 0.624367 | 0.756147 | 0.0000 |
| snow | d2d_topk | 0.623327 | 0.755405 | -0.0055 |
| snow | d2d_original_nms | 0.630299 | 0.759544 | 0.0254 |
| snow | gt_iou_oracle | 0.831605 | 0.889899 | 1.0000 |
