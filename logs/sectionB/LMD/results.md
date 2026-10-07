# Adapted LMD-core benchmark

主分数是 LMD meta-regression 预测的 GT BEV IoU；固定 decoded geometry、top256、原 NMS 和原输出预算。

| 条件 | Original AP70 | LMD AP70 | Oracle AP70 | Oracle recovery | LMD global AP70 |
|---|---:|---:|---:|---:|---:|
| clean | 0.7558 | 0.7493 | 0.9197 | -0.040 | 0.8824 |
| fog | 0.7266 | 0.7168 | 0.8904 | -0.060 | 0.8398 |
| rain | 0.7457 | 0.7384 | 0.9153 | -0.043 | 0.8749 |
| snow | 0.6300 | 0.6351 | 0.8352 | 0.025 | 0.7732 |

完整 AP30/AP50/AP70、质量指标、TP/FP 变化及 selection-only 诊断见 results.json。
