# Learned 3D NMS (gossip) fixed physics validation

AP70 values use identical frozen F/top256 candidate sets.
Frame-order and global-sort AP must never be compared directly.

| Weather | Method | Frame AP70 | Global AP70 | Global Oracle Recovery |
|---|---|---:|---:|---:|
| clean | original_f | 0.755802 | 0.879868 | 0.0000 |
| clean | top256_original_nms | 0.755802 | 0.879868 | 0.0000 |
| clean | d2d_topk | 0.757162 | 0.884232 | 0.0514 |
| clean | d2d_original_nms | 0.756897 | 0.883839 | 0.0468 |
| clean | gt_iou_oracle | 0.919738 | 0.964781 | 1.0000 |
| fog | original_f | 0.726541 | 0.836885 | 0.0000 |
| fog | top256_original_nms | 0.726541 | 0.836885 | 0.0000 |
| fog | d2d_topk | 0.725808 | 0.839133 | 0.0228 |
| fog | d2d_original_nms | 0.725943 | 0.839018 | 0.0217 |
| fog | gt_iou_oracle | 0.890617 | 0.935389 | 1.0000 |
| rain | original_f | 0.747099 | 0.872398 | 0.0000 |
| rain | top256_original_nms | 0.747099 | 0.872398 | 0.0000 |
| rain | d2d_topk | 0.747810 | 0.876747 | 0.0488 |
| rain | d2d_original_nms | 0.748264 | 0.876582 | 0.0470 |
| rain | gt_iou_oracle | 0.915209 | 0.961499 | 1.0000 |
| snow | original_f | 0.624367 | 0.756147 | 0.0000 |
| snow | top256_original_nms | 0.624367 | 0.756147 | 0.0000 |
| snow | d2d_topk | 0.641470 | 0.773714 | 0.1313 |
| snow | d2d_original_nms | 0.643101 | 0.774394 | 0.1364 |
| snow | gt_iou_oracle | 0.831605 | 0.889899 | 1.0000 |
