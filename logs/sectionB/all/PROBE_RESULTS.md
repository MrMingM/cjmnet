# 动作标签离线对照

两组使用相同原特征、标准化和场景；OriginalLabel 直接复用原主权重。下表统一采用检测后果标准，严格检测非平局 pair 排除 KEEP 简化偏好。原标签标准另表报告，两个标准的 regret 不直接相减。

| Task | Weather | Label | Policy | Stratum | Pairwise | 最佳后果Top1 | Regret | 修改比例 | 检测同效仍改 | 真有益精度 | recovered/lost/newFP | TP缺口/额外lost |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| cls | clean | OriginalLabel | Greedy | all | 0.7456 | 0.8946 | 0.0971 | 24.39% | 23.03% | 0.0142 | 7/13/44 | 83/13 |
| cls | clean | OriginalLabel | Greedy | associated | 0.8709 | 0.9403 | 0.0564 | 31.59% | 30.50% | 0.0164 | 3/5/11 | 36/5 |
| cls | clean | OriginalLabel | Greedy | background | 0.6590 | 0.8670 | 0.1217 | 20.03% | 18.51% | 0.0121 | 4/8/33 | 47/8 |
| cls | clean | OriginalLabel | CommonConservative | all | 0.7456 | 0.9005 | 0.0906 | 0.00% | 0.00% | NA | 0/0/0 | 77/0 |
| cls | clean | OriginalLabel | CommonConservative | associated | 0.8709 | 0.9408 | 0.0550 | 0.00% | 0.00% | NA | 0/0/0 | 34/0 |
| cls | clean | OriginalLabel | CommonConservative | background | 0.6590 | 0.8760 | 0.1120 | 0.00% | 0.00% | NA | 0/0/0 | 43/0 |
| cls | clean | DetectionOutcomeLabel | Greedy | all | 0.8248 | 0.9005 | 0.0906 | 0.00% | 0.00% | NA | 0/0/0 | 77/0 |
| cls | clean | DetectionOutcomeLabel | Greedy | associated | 0.8992 | 0.9408 | 0.0550 | 0.00% | 0.00% | NA | 0/0/0 | 34/0 |
| cls | clean | DetectionOutcomeLabel | Greedy | background | 0.7734 | 0.8760 | 0.1120 | 0.00% | 0.00% | NA | 0/0/0 | 43/0 |
| cls | clean | DetectionOutcomeLabel | CommonConservative | all | 0.8248 | 0.9005 | 0.0906 | 0.00% | 0.00% | NA | 0/0/0 | 77/0 |
| cls | clean | DetectionOutcomeLabel | CommonConservative | associated | 0.8992 | 0.9408 | 0.0550 | 0.00% | 0.00% | NA | 0/0/0 | 34/0 |
| cls | clean | DetectionOutcomeLabel | CommonConservative | background | 0.7734 | 0.8760 | 0.1120 | 0.00% | 0.00% | NA | 0/0/0 | 43/0 |
| cls | fog | OriginalLabel | Greedy | all | 0.7425 | 0.8815 | 0.1087 | 27.39% | 26.34% | 0.0120 | 5/8/33 | 83/8 |
| cls | fog | OriginalLabel | Greedy | associated | 0.8634 | 0.9329 | 0.0636 | 36.64% | 35.93% | 0.0113 | 2/3/5 | 30/3 |
| cls | fog | OriginalLabel | Greedy | background | 0.6586 | 0.8514 | 0.1351 | 21.98% | 20.73% | 0.0126 | 3/5/28 | 53/5 |
| cls | fog | OriginalLabel | CommonConservative | all | 0.7425 | 0.8850 | 0.1047 | 0.00% | 0.00% | NA | 0/0/0 | 80/0 |
| cls | fog | OriginalLabel | CommonConservative | associated | 0.8634 | 0.9317 | 0.0642 | 0.00% | 0.00% | NA | 0/0/0 | 29/0 |
| cls | fog | OriginalLabel | CommonConservative | background | 0.6586 | 0.8576 | 0.1284 | 0.00% | 0.00% | NA | 0/0/0 | 51/0 |
| cls | fog | DetectionOutcomeLabel | Greedy | all | 0.8128 | 0.8850 | 0.1047 | 0.00% | 0.00% | NA | 0/0/0 | 80/0 |
| cls | fog | DetectionOutcomeLabel | Greedy | associated | 0.8882 | 0.9317 | 0.0642 | 0.00% | 0.00% | NA | 0/0/0 | 29/0 |
| cls | fog | DetectionOutcomeLabel | Greedy | background | 0.7605 | 0.8576 | 0.1284 | 0.00% | 0.00% | NA | 0/0/0 | 51/0 |
| cls | fog | DetectionOutcomeLabel | CommonConservative | all | 0.8128 | 0.8850 | 0.1047 | 0.00% | 0.00% | NA | 0/0/0 | 80/0 |
| cls | fog | DetectionOutcomeLabel | CommonConservative | associated | 0.8882 | 0.9317 | 0.0642 | 0.00% | 0.00% | NA | 0/0/0 | 29/0 |
| cls | fog | DetectionOutcomeLabel | CommonConservative | background | 0.7605 | 0.8576 | 0.1284 | 0.00% | 0.00% | NA | 0/0/0 | 51/0 |
| cls | rain | OriginalLabel | Greedy | all | 0.7361 | 0.8854 | 0.1037 | 25.30% | 23.63% | 0.0257 | 15/13/44 | 77/13 |
| cls | rain | OriginalLabel | Greedy | associated | 0.8617 | 0.9390 | 0.0570 | 34.02% | 32.99% | 0.0169 | 4/3/8 | 30/3 |
| cls | rain | OriginalLabel | Greedy | background | 0.6527 | 0.8531 | 0.1319 | 20.03% | 17.99% | 0.0347 | 11/10/36 | 47/10 |
| cls | rain | OriginalLabel | CommonConservative | all | 0.7361 | 0.8874 | 0.1007 | 0.00% | 0.00% | NA | 0/0/0 | 79/0 |
| cls | rain | OriginalLabel | CommonConservative | associated | 0.8617 | 0.9361 | 0.0586 | 0.00% | 0.00% | NA | 0/0/0 | 31/0 |
| cls | rain | OriginalLabel | CommonConservative | background | 0.6527 | 0.8580 | 0.1260 | 0.00% | 0.00% | NA | 0/0/0 | 48/0 |
| cls | rain | DetectionOutcomeLabel | Greedy | all | 0.8119 | 0.8874 | 0.1007 | 0.00% | 0.00% | NA | 0/0/0 | 79/0 |
| cls | rain | DetectionOutcomeLabel | Greedy | associated | 0.8880 | 0.9361 | 0.0586 | 0.00% | 0.00% | NA | 0/0/0 | 31/0 |
| cls | rain | DetectionOutcomeLabel | Greedy | background | 0.7614 | 0.8580 | 0.1260 | 0.00% | 0.00% | NA | 0/0/0 | 48/0 |
| cls | rain | DetectionOutcomeLabel | CommonConservative | all | 0.8119 | 0.8874 | 0.1007 | 0.00% | 0.00% | NA | 0/0/0 | 79/0 |
| cls | rain | DetectionOutcomeLabel | CommonConservative | associated | 0.8880 | 0.9361 | 0.0586 | 0.00% | 0.00% | NA | 0/0/0 | 31/0 |
| cls | rain | DetectionOutcomeLabel | CommonConservative | background | 0.7614 | 0.8580 | 0.1260 | 0.00% | 0.00% | NA | 0/0/0 | 48/0 |
| cls | snow | OriginalLabel | Greedy | all | 0.7450 | 0.8497 | 0.1316 | 15.29% | 14.10% | 0.0303 | 11/5/29 | 125/5 |
| cls | snow | OriginalLabel | Greedy | associated | 0.8406 | 0.8825 | 0.1061 | 19.69% | 18.53% | 0.0400 | 8/0/6 | 57/0 |
| cls | snow | OriginalLabel | Greedy | background | 0.6624 | 0.8309 | 0.1463 | 12.78% | 11.56% | 0.0217 | 3/5/23 | 68/5 |
| cls | snow | OriginalLabel | CommonConservative | all | 0.7450 | 0.8506 | 0.1299 | 0.00% | 0.00% | NA | 0/0/0 | 131/0 |
| cls | snow | OriginalLabel | CommonConservative | associated | 0.8406 | 0.8770 | 0.1110 | 0.00% | 0.00% | NA | 0/0/0 | 65/0 |
| cls | snow | OriginalLabel | CommonConservative | background | 0.6624 | 0.8354 | 0.1407 | 0.00% | 0.00% | NA | 0/0/0 | 66/0 |
| cls | snow | DetectionOutcomeLabel | Greedy | all | 0.7976 | 0.8506 | 0.1299 | 0.00% | 0.00% | NA | 0/0/0 | 131/0 |
| cls | snow | DetectionOutcomeLabel | Greedy | associated | 0.8570 | 0.8770 | 0.1110 | 0.00% | 0.00% | NA | 0/0/0 | 65/0 |
| cls | snow | DetectionOutcomeLabel | Greedy | background | 0.7462 | 0.8354 | 0.1407 | 0.00% | 0.00% | NA | 0/0/0 | 66/0 |
| cls | snow | DetectionOutcomeLabel | CommonConservative | all | 0.7976 | 0.8506 | 0.1299 | 0.00% | 0.00% | NA | 0/0/0 | 131/0 |
| cls | snow | DetectionOutcomeLabel | CommonConservative | associated | 0.8570 | 0.8770 | 0.1110 | 0.00% | 0.00% | NA | 0/0/0 | 65/0 |
| cls | snow | DetectionOutcomeLabel | CommonConservative | background | 0.7462 | 0.8354 | 0.1407 | 0.00% | 0.00% | NA | 0/0/0 | 66/0 |
| reg | clean | OriginalLabel | Greedy | all | 0.9358 | 0.9872 | 0.0097 | 0.00% | 0.00% | NA | 0/0/0 | 58/0 |
| reg | clean | OriginalLabel | Greedy | associated | 0.9480 | 0.9862 | 0.0103 | 0.00% | 0.00% | NA | 0/0/0 | 24/0 |
| reg | clean | OriginalLabel | Greedy | background | 0.9259 | 0.9878 | 0.0094 | 0.00% | 0.00% | NA | 0/0/0 | 34/0 |
| reg | clean | OriginalLabel | CommonConservative | all | 0.9358 | 0.9872 | 0.0097 | 0.00% | 0.00% | NA | 0/0/0 | 58/0 |
| reg | clean | OriginalLabel | CommonConservative | associated | 0.9480 | 0.9862 | 0.0103 | 0.00% | 0.00% | NA | 0/0/0 | 24/0 |
| reg | clean | OriginalLabel | CommonConservative | background | 0.9259 | 0.9878 | 0.0094 | 0.00% | 0.00% | NA | 0/0/0 | 34/0 |
| reg | clean | DetectionOutcomeLabel | Greedy | all | 0.9377 | 0.9872 | 0.0097 | 0.00% | 0.00% | NA | 0/0/0 | 58/0 |
| reg | clean | DetectionOutcomeLabel | Greedy | associated | 0.9461 | 0.9862 | 0.0103 | 0.00% | 0.00% | NA | 0/0/0 | 24/0 |
| reg | clean | DetectionOutcomeLabel | Greedy | background | 0.9310 | 0.9878 | 0.0094 | 0.00% | 0.00% | NA | 0/0/0 | 34/0 |
| reg | clean | DetectionOutcomeLabel | CommonConservative | all | 0.9377 | 0.9872 | 0.0097 | 0.00% | 0.00% | NA | 0/0/0 | 58/0 |
| reg | clean | DetectionOutcomeLabel | CommonConservative | associated | 0.9461 | 0.9862 | 0.0103 | 0.00% | 0.00% | NA | 0/0/0 | 24/0 |
| reg | clean | DetectionOutcomeLabel | CommonConservative | background | 0.9310 | 0.9878 | 0.0094 | 0.00% | 0.00% | NA | 0/0/0 | 34/0 |
| reg | fog | OriginalLabel | Greedy | all | 0.9256 | 0.9840 | 0.0115 | 0.00% | 0.00% | NA | 0/0/0 | 68/0 |
| reg | fog | OriginalLabel | Greedy | associated | 0.9399 | 0.9828 | 0.0128 | 0.00% | 0.00% | NA | 0/0/0 | 28/0 |
| reg | fog | OriginalLabel | Greedy | background | 0.9138 | 0.9847 | 0.0108 | 0.00% | 0.00% | NA | 0/0/0 | 40/0 |
| reg | fog | OriginalLabel | CommonConservative | all | 0.9256 | 0.9840 | 0.0115 | 0.00% | 0.00% | NA | 0/0/0 | 68/0 |
| reg | fog | OriginalLabel | CommonConservative | associated | 0.9399 | 0.9828 | 0.0128 | 0.00% | 0.00% | NA | 0/0/0 | 28/0 |
| reg | fog | OriginalLabel | CommonConservative | background | 0.9138 | 0.9847 | 0.0108 | 0.00% | 0.00% | NA | 0/0/0 | 40/0 |
| reg | fog | DetectionOutcomeLabel | Greedy | all | 0.9267 | 0.9840 | 0.0115 | 0.00% | 0.00% | NA | 0/0/0 | 68/0 |
| reg | fog | DetectionOutcomeLabel | Greedy | associated | 0.9372 | 0.9828 | 0.0128 | 0.00% | 0.00% | NA | 0/0/0 | 28/0 |
| reg | fog | DetectionOutcomeLabel | Greedy | background | 0.9179 | 0.9847 | 0.0108 | 0.00% | 0.00% | NA | 0/0/0 | 40/0 |
| reg | fog | DetectionOutcomeLabel | CommonConservative | all | 0.9267 | 0.9840 | 0.0115 | 0.00% | 0.00% | NA | 0/0/0 | 68/0 |
| reg | fog | DetectionOutcomeLabel | CommonConservative | associated | 0.9372 | 0.9828 | 0.0128 | 0.00% | 0.00% | NA | 0/0/0 | 28/0 |
| reg | fog | DetectionOutcomeLabel | CommonConservative | background | 0.9179 | 0.9847 | 0.0108 | 0.00% | 0.00% | NA | 0/0/0 | 40/0 |
| reg | rain | OriginalLabel | Greedy | all | 0.9382 | 0.9855 | 0.0122 | 0.00% | 0.00% | NA | 0/0/0 | 67/0 |
| reg | rain | OriginalLabel | Greedy | associated | 0.9432 | 0.9833 | 0.0141 | 0.00% | 0.00% | NA | 0/0/0 | 29/0 |
| reg | rain | OriginalLabel | Greedy | background | 0.9341 | 0.9868 | 0.0111 | 0.00% | 0.00% | NA | 0/0/0 | 38/0 |
| reg | rain | OriginalLabel | CommonConservative | all | 0.9382 | 0.9855 | 0.0122 | 0.00% | 0.00% | NA | 0/0/0 | 67/0 |
| reg | rain | OriginalLabel | CommonConservative | associated | 0.9432 | 0.9833 | 0.0141 | 0.00% | 0.00% | NA | 0/0/0 | 29/0 |
| reg | rain | OriginalLabel | CommonConservative | background | 0.9341 | 0.9868 | 0.0111 | 0.00% | 0.00% | NA | 0/0/0 | 38/0 |
| reg | rain | DetectionOutcomeLabel | Greedy | all | 0.9402 | 0.9855 | 0.0122 | 0.00% | 0.00% | NA | 0/0/0 | 67/0 |
| reg | rain | DetectionOutcomeLabel | Greedy | associated | 0.9433 | 0.9833 | 0.0141 | 0.00% | 0.00% | NA | 0/0/0 | 29/0 |
| reg | rain | DetectionOutcomeLabel | Greedy | background | 0.9377 | 0.9868 | 0.0111 | 0.00% | 0.00% | NA | 0/0/0 | 38/0 |
| reg | rain | DetectionOutcomeLabel | CommonConservative | all | 0.9402 | 0.9855 | 0.0122 | 0.00% | 0.00% | NA | 0/0/0 | 67/0 |
| reg | rain | DetectionOutcomeLabel | CommonConservative | associated | 0.9433 | 0.9833 | 0.0141 | 0.00% | 0.00% | NA | 0/0/0 | 29/0 |
| reg | rain | DetectionOutcomeLabel | CommonConservative | background | 0.9377 | 0.9868 | 0.0111 | 0.00% | 0.00% | NA | 0/0/0 | 38/0 |
| reg | snow | OriginalLabel | Greedy | all | 0.9132 | 0.9722 | 0.0198 | 0.00% | 0.00% | NA | 0/0/0 | 123/0 |
| reg | snow | OriginalLabel | Greedy | associated | 0.9229 | 0.9655 | 0.0248 | 0.00% | 0.00% | NA | 0/0/0 | 57/0 |
| reg | snow | OriginalLabel | Greedy | background | 0.9032 | 0.9760 | 0.0170 | 0.00% | 0.00% | NA | 0/0/0 | 66/0 |
| reg | snow | OriginalLabel | CommonConservative | all | 0.9132 | 0.9722 | 0.0198 | 0.00% | 0.00% | NA | 0/0/0 | 123/0 |
| reg | snow | OriginalLabel | CommonConservative | associated | 0.9229 | 0.9655 | 0.0248 | 0.00% | 0.00% | NA | 0/0/0 | 57/0 |
| reg | snow | OriginalLabel | CommonConservative | background | 0.9032 | 0.9760 | 0.0170 | 0.00% | 0.00% | NA | 0/0/0 | 66/0 |
| reg | snow | DetectionOutcomeLabel | Greedy | all | 0.9138 | 0.9722 | 0.0198 | 0.00% | 0.00% | NA | 0/0/0 | 123/0 |
| reg | snow | DetectionOutcomeLabel | Greedy | associated | 0.9196 | 0.9655 | 0.0248 | 0.00% | 0.00% | NA | 0/0/0 | 57/0 |
| reg | snow | DetectionOutcomeLabel | Greedy | background | 0.9079 | 0.9760 | 0.0170 | 0.00% | 0.00% | NA | 0/0/0 | 66/0 |
| reg | snow | DetectionOutcomeLabel | CommonConservative | all | 0.9138 | 0.9722 | 0.0198 | 0.00% | 0.00% | NA | 0/0/0 | 123/0 |
| reg | snow | DetectionOutcomeLabel | CommonConservative | associated | 0.9196 | 0.9655 | 0.0248 | 0.00% | 0.00% | NA | 0/0/0 | 57/0 |
| reg | snow | DetectionOutcomeLabel | CommonConservative | background | 0.9079 | 0.9760 | 0.0170 | 0.00% | 0.00% | NA | 0/0/0 | 66/0 |

原标签评价标准（全体标签组）：

| Task | Weather | Label | Policy | 原Pairwise | 原Top1 | 原Regret |
|---|---|---|---|---:|---:|---:|
| cls | clean | OriginalLabel | Greedy | 0.8531 | 0.5570 | 0.1687 |
| cls | clean | OriginalLabel | CommonConservative | 0.8531 | 0.5678 | 0.2306 |
| cls | clean | DetectionOutcomeLabel | Greedy | 0.6884 | 0.5678 | 0.2306 |
| cls | clean | DetectionOutcomeLabel | CommonConservative | 0.6884 | 0.5678 | 0.2306 |
| cls | fog | OriginalLabel | Greedy | 0.8453 | 0.5576 | 0.1700 |
| cls | fog | OriginalLabel | CommonConservative | 0.8453 | 0.5592 | 0.2465 |
| cls | fog | DetectionOutcomeLabel | Greedy | 0.6968 | 0.5592 | 0.2465 |
| cls | fog | DetectionOutcomeLabel | CommonConservative | 0.6968 | 0.5592 | 0.2465 |
| cls | rain | OriginalLabel | Greedy | 0.8489 | 0.5586 | 0.1737 |
| cls | rain | OriginalLabel | CommonConservative | 0.8489 | 0.5538 | 0.2445 |
| cls | rain | DetectionOutcomeLabel | Greedy | 0.6859 | 0.5538 | 0.2445 |
| cls | rain | DetectionOutcomeLabel | CommonConservative | 0.6859 | 0.5538 | 0.2445 |
| cls | snow | OriginalLabel | Greedy | 0.8367 | 0.5800 | 0.1820 |
| cls | snow | OriginalLabel | CommonConservative | 0.8367 | 0.5868 | 0.2192 |
| cls | snow | DetectionOutcomeLabel | Greedy | 0.7394 | 0.5868 | 0.2192 |
| cls | snow | DetectionOutcomeLabel | CommonConservative | 0.7394 | 0.5868 | 0.2192 |
| reg | clean | OriginalLabel | Greedy | 0.8158 | 0.6678 | 0.1314 |
| reg | clean | OriginalLabel | CommonConservative | 0.8158 | 0.6678 | 0.1314 |
| reg | clean | DetectionOutcomeLabel | Greedy | 0.8127 | 0.6678 | 0.1314 |
| reg | clean | DetectionOutcomeLabel | CommonConservative | 0.8127 | 0.6678 | 0.1314 |
| reg | fog | OriginalLabel | Greedy | 0.8158 | 0.6788 | 0.1294 |
| reg | fog | OriginalLabel | CommonConservative | 0.8158 | 0.6788 | 0.1294 |
| reg | fog | DetectionOutcomeLabel | Greedy | 0.8111 | 0.6788 | 0.1294 |
| reg | fog | DetectionOutcomeLabel | CommonConservative | 0.8111 | 0.6788 | 0.1294 |
| reg | rain | OriginalLabel | Greedy | 0.8127 | 0.6630 | 0.1355 |
| reg | rain | OriginalLabel | CommonConservative | 0.8127 | 0.6630 | 0.1355 |
| reg | rain | DetectionOutcomeLabel | Greedy | 0.8105 | 0.6630 | 0.1355 |
| reg | rain | DetectionOutcomeLabel | CommonConservative | 0.8105 | 0.6630 | 0.1355 |
| reg | snow | OriginalLabel | Greedy | 0.8317 | 0.7087 | 0.1170 |
| reg | snow | OriginalLabel | CommonConservative | 0.8317 | 0.7087 | 0.1170 |
| reg | snow | DetectionOutcomeLabel | Greedy | 0.8280 | 0.7087 | 0.1170 |
| reg | snow | DetectionOutcomeLabel | CommonConservative | 0.8280 | 0.7087 | 0.1170 |

训练预算（原标签更新次数由原分组、顺序和批规则复原）：

| Label | Task | Epoch | pairs | optimizer updates | loss |
|---|---|---:|---:|---:|---:|
| OriginalLabel | cls | 1 | 499638 | 122 | 0.411847 |
| OriginalLabel | cls | 2 | 499638 | 122 | 0.356049 |
| OriginalLabel | cls | 3 | 499638 | 122 | 0.350940 |
| OriginalLabel | cls | 4 | 499638 | 122 | 0.349669 |
| OriginalLabel | cls | 5 | 499638 | 122 | 0.349188 |
| OriginalLabel | cls | 6 | 499638 | 122 | 0.348682 |
| OriginalLabel | cls | 7 | 499638 | 122 | 0.348622 |
| OriginalLabel | cls | 8 | 499638 | 122 | 0.348329 |
| OriginalLabel | cls | 9 | 499638 | 122 | 0.347687 |
| OriginalLabel | cls | 10 | 499638 | 122 | 0.347454 |
| OriginalLabel | cls | 11 | 499638 | 122 | 0.347055 |
| OriginalLabel | cls | 12 | 499638 | 122 | 0.346684 |
| OriginalLabel | cls | 13 | 499638 | 122 | 0.346488 |
| OriginalLabel | cls | 14 | 499638 | 122 | 0.346368 |
| OriginalLabel | cls | 15 | 499638 | 122 | 0.346193 |
| OriginalLabel | cls | 16 | 499638 | 122 | 0.346042 |
| OriginalLabel | cls | 17 | 499638 | 122 | 0.345928 |
| OriginalLabel | cls | 18 | 499638 | 122 | 0.345949 |
| OriginalLabel | cls | 19 | 499638 | 122 | 0.345505 |
| OriginalLabel | cls | 20 | 499638 | 122 | 0.345544 |
| OriginalLabel | cls | 21 | 499638 | 122 | 0.345172 |
| OriginalLabel | cls | 22 | 499638 | 122 | 0.345089 |
| OriginalLabel | cls | 23 | 499638 | 122 | 0.345057 |
| OriginalLabel | cls | 24 | 499638 | 122 | 0.344882 |
| OriginalLabel | cls | 25 | 499638 | 122 | 0.344998 |
| OriginalLabel | cls | 26 | 499638 | 122 | 0.344752 |
| OriginalLabel | cls | 27 | 499638 | 122 | 0.344819 |
| OriginalLabel | cls | 28 | 499638 | 122 | 0.344512 |
| OriginalLabel | cls | 29 | 499638 | 122 | 0.344456 |
| OriginalLabel | cls | 30 | 499638 | 122 | 0.344570 |
| OriginalLabel | reg | 1 | 478636 | 117 | 0.402144 |
| OriginalLabel | reg | 2 | 478636 | 117 | 0.374500 |
| OriginalLabel | reg | 3 | 478636 | 117 | 0.372617 |
| OriginalLabel | reg | 4 | 478636 | 117 | 0.371521 |
| OriginalLabel | reg | 5 | 478636 | 117 | 0.370829 |
| OriginalLabel | reg | 6 | 478636 | 117 | 0.370324 |
| OriginalLabel | reg | 7 | 478636 | 117 | 0.370133 |
| OriginalLabel | reg | 8 | 478636 | 117 | 0.369899 |
| OriginalLabel | reg | 9 | 478636 | 117 | 0.369899 |
| OriginalLabel | reg | 10 | 478636 | 117 | 0.369838 |
| OriginalLabel | reg | 11 | 478636 | 117 | 0.369828 |
| OriginalLabel | reg | 12 | 478636 | 117 | 0.369799 |
| OriginalLabel | reg | 13 | 478636 | 117 | 0.369808 |
| OriginalLabel | reg | 14 | 478636 | 117 | 0.369813 |
| OriginalLabel | reg | 15 | 478636 | 117 | 0.369884 |
| OriginalLabel | reg | 16 | 478636 | 117 | 0.369900 |
| OriginalLabel | reg | 17 | 478636 | 117 | 0.369904 |
| OriginalLabel | reg | 18 | 478636 | 117 | 0.369881 |
| OriginalLabel | reg | 19 | 478636 | 117 | 0.369801 |
| OriginalLabel | reg | 20 | 478636 | 117 | 0.369933 |
| OriginalLabel | reg | 21 | 478636 | 117 | 0.369814 |
| OriginalLabel | reg | 22 | 478636 | 117 | 0.369908 |
| OriginalLabel | reg | 23 | 478636 | 117 | 0.369854 |
| OriginalLabel | reg | 24 | 478636 | 117 | 0.369970 |
| OriginalLabel | reg | 25 | 478636 | 117 | 0.369916 |
| OriginalLabel | reg | 26 | 478636 | 117 | 0.370045 |
| OriginalLabel | reg | 27 | 478636 | 117 | 0.369847 |
| OriginalLabel | reg | 28 | 478636 | 117 | 0.369916 |
| OriginalLabel | reg | 29 | 478636 | 117 | 0.369881 |
| OriginalLabel | reg | 30 | 478636 | 117 | 0.369855 |
| DetectionOutcomeLabel | cls | 1 | 362650 | 89 | 0.303926 |
| DetectionOutcomeLabel | cls | 2 | 362650 | 89 | 0.204977 |
| DetectionOutcomeLabel | cls | 3 | 362650 | 89 | 0.194192 |
| DetectionOutcomeLabel | cls | 4 | 362650 | 89 | 0.191268 |
| DetectionOutcomeLabel | cls | 5 | 362650 | 89 | 0.190446 |
| DetectionOutcomeLabel | cls | 6 | 362650 | 89 | 0.190328 |
| DetectionOutcomeLabel | cls | 7 | 362650 | 89 | 0.190264 |
| DetectionOutcomeLabel | cls | 8 | 362650 | 89 | 0.190133 |
| DetectionOutcomeLabel | cls | 9 | 362650 | 89 | 0.189942 |
| DetectionOutcomeLabel | cls | 10 | 362650 | 89 | 0.189808 |
| DetectionOutcomeLabel | cls | 11 | 362650 | 89 | 0.189908 |
| DetectionOutcomeLabel | cls | 12 | 362650 | 89 | 0.189755 |
| DetectionOutcomeLabel | cls | 13 | 362650 | 89 | 0.189617 |
| DetectionOutcomeLabel | cls | 14 | 362650 | 89 | 0.189670 |
| DetectionOutcomeLabel | cls | 15 | 362650 | 89 | 0.189578 |
| DetectionOutcomeLabel | cls | 16 | 362650 | 89 | 0.189762 |
| DetectionOutcomeLabel | cls | 17 | 362650 | 89 | 0.189527 |
| DetectionOutcomeLabel | cls | 18 | 362650 | 89 | 0.189687 |
| DetectionOutcomeLabel | cls | 19 | 362650 | 89 | 0.189306 |
| DetectionOutcomeLabel | cls | 20 | 362650 | 89 | 0.189638 |
| DetectionOutcomeLabel | cls | 21 | 362650 | 89 | 0.189314 |
| DetectionOutcomeLabel | cls | 22 | 362650 | 89 | 0.189425 |
| DetectionOutcomeLabel | cls | 23 | 362650 | 89 | 0.189360 |
| DetectionOutcomeLabel | cls | 24 | 362650 | 89 | 0.189262 |
| DetectionOutcomeLabel | cls | 25 | 362650 | 89 | 0.189203 |
| DetectionOutcomeLabel | cls | 26 | 362650 | 89 | 0.189187 |
| DetectionOutcomeLabel | cls | 27 | 362650 | 89 | 0.189124 |
| DetectionOutcomeLabel | cls | 28 | 362650 | 89 | 0.189129 |
| DetectionOutcomeLabel | cls | 29 | 362650 | 89 | 0.189225 |
| DetectionOutcomeLabel | cls | 30 | 362650 | 89 | 0.189262 |
| DetectionOutcomeLabel | reg | 1 | 326397 | 80 | 0.199145 |
| DetectionOutcomeLabel | reg | 2 | 326397 | 80 | 0.094247 |
| DetectionOutcomeLabel | reg | 3 | 326397 | 80 | 0.083380 |
| DetectionOutcomeLabel | reg | 4 | 326397 | 80 | 0.078117 |
| DetectionOutcomeLabel | reg | 5 | 326397 | 80 | 0.075108 |
| DetectionOutcomeLabel | reg | 6 | 326397 | 80 | 0.073230 |
| DetectionOutcomeLabel | reg | 7 | 326397 | 80 | 0.072244 |
| DetectionOutcomeLabel | reg | 8 | 326397 | 80 | 0.071519 |
| DetectionOutcomeLabel | reg | 9 | 326397 | 80 | 0.071157 |
| DetectionOutcomeLabel | reg | 10 | 326397 | 80 | 0.070978 |
| DetectionOutcomeLabel | reg | 11 | 326397 | 80 | 0.070858 |
| DetectionOutcomeLabel | reg | 12 | 326397 | 80 | 0.070747 |
| DetectionOutcomeLabel | reg | 13 | 326397 | 80 | 0.070734 |
| DetectionOutcomeLabel | reg | 14 | 326397 | 80 | 0.070708 |
| DetectionOutcomeLabel | reg | 15 | 326397 | 80 | 0.070754 |
| DetectionOutcomeLabel | reg | 16 | 326397 | 80 | 0.070702 |
| DetectionOutcomeLabel | reg | 17 | 326397 | 80 | 0.070709 |
| DetectionOutcomeLabel | reg | 18 | 326397 | 80 | 0.070760 |
| DetectionOutcomeLabel | reg | 19 | 326397 | 80 | 0.070654 |
| DetectionOutcomeLabel | reg | 20 | 326397 | 80 | 0.070693 |
| DetectionOutcomeLabel | reg | 21 | 326397 | 80 | 0.070694 |
| DetectionOutcomeLabel | reg | 22 | 326397 | 80 | 0.070761 |
| DetectionOutcomeLabel | reg | 23 | 326397 | 80 | 0.070692 |
| DetectionOutcomeLabel | reg | 24 | 326397 | 80 | 0.070732 |
| DetectionOutcomeLabel | reg | 25 | 326397 | 80 | 0.070716 |
| DetectionOutcomeLabel | reg | 26 | 326397 | 80 | 0.070753 |
| DetectionOutcomeLabel | reg | 27 | 326397 | 80 | 0.070670 |
| DetectionOutcomeLabel | reg | 28 | 326397 | 80 | 0.070667 |
| DetectionOutcomeLabel | reg | 29 | 326397 | 80 | 0.070663 |
| DetectionOutcomeLabel | reg | 30 | 326397 | 80 | 0.070710 |

独立动作的 recovered/lost/newFP 可能对同一 GT 重复计数，不能解释为整帧收益。两组 pair 数和更新次数可能不同，这是标签处理的一部分。没有套用原 S1/S2 成功门槛。
