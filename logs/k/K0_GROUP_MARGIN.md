# K0′ Group-level Label Margin Audit

K0′ 的恶劣天气主范围主要表现为连续分数/IoU 排序主导。

## K0′ 主表：validation associated

| Weather | Task | Groups | Discrete margin | Continuous-only | Exact tie | Beneficial-set |
|---|---|---:|---:|---:|---:|---:|
| clean | cls | 1741 | 5.28% | 91.96% | 2.76% | 5.92% |
| clean | reg | 1741 | 0.80% | 96.38% | 2.81% | 1.38% |
| fog | cls | 1684 | 4.93% | 91.51% | 3.56% | 6.83% |
| fog | reg | 1684 | 1.07% | 96.26% | 2.67% | 1.72% |
| rain | cls | 1737 | 4.84% | 92.17% | 2.99% | 6.39% |
| rain | reg | 1737 | 0.86% | 96.49% | 2.65% | 1.67% |
| snow | cls | 1651 | 8.30% | 78.38% | 13.33% | 12.30% |
| snow | reg | 1651 | 2.00% | 85.40% | 12.60% | 3.45% |
| adverse mean | cls | — | 6.02% | 87.35% | 6.63% | 8.50% |
| adverse mean | reg | — | 1.31% | 92.72% | 5.97% | 2.28% |

## ACTION-LEVEL CONTEXT：action-level net TP count change

仅统计 non-KEEP action；TP increase 是能直接带来净 TP 增长的 action 比例。

| Weather | Task | TP increase actions | TP decrease actions | TP unchanged | Ratio decrease:increase |
|---|---|---:|---:|---:|---:|
| clean | cls | 40/10432 (0.38%) | 1049 (10.06%) | 89.56% | 26.23 |
| clean | reg | 36/10432 (0.35%) | 934 (8.95%) | 90.70% | 25.94 |
| fog | cls | 36/10158 (0.35%) | 1161 (11.43%) | 88.22% | 32.25 |
| fog | reg | 47/10158 (0.46%) | 1094 (10.77%) | 88.77% | 23.28 |
| rain | cls | 44/10414 (0.42%) | 1000 (9.60%) | 89.98% | 22.73 |
| rain | reg | 47/10414 (0.45%) | 934 (8.97%) | 90.58% | 19.87 |
| snow | cls | 86/9990 (0.86%) | 1901 (19.03%) | 80.11% | 22.10 |
| snow | reg | 98/9990 (0.98%) | 1456 (14.57%) | 84.44% | 14.86 |

## 全 split/weather/task/stratum 指标

adverse mean 为三天气等权平均；train 只检查趋势，background 单列。

| Scope | Groups | Discrete | Continuous | Tie | Multi-top | Mean/median tie | Beneficial set | Continuous benefit | Harmful exists | KEEP-only |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| train/clean/cls/all | 15533 | 2.70% | 31.69% | 65.61% | 65.61% | 3.914/5.0 | 5.02% | 30.26% | 31.22% | 0 |
| train/clean/cls/associated | 5293 | 1.61% | 92.99% | 5.40% | 5.40% | 1.120/1.0 | 2.82% | 88.80% | 36.46% | 0 |
| train/clean/cls/background | 10240 | 3.27% | 0.00% | 96.73% | 96.73% | 5.358/5.0 | 6.15% | 0.00% | 28.52% | 0 |
| train/clean/reg/all | 15533 | 0.47% | 32.38% | 67.15% | 67.15% | 4.083/5.0 | 0.91% | 26.93% | 26.02% | 0 |
| train/clean/reg/associated | 5293 | 0.38% | 95.01% | 4.61% | 4.61% | 1.117/1.0 | 0.74% | 79.03% | 29.83% | 0 |
| train/clean/reg/background | 10240 | 0.52% | 0.00% | 99.48% | 99.48% | 5.616/5.0 | 1.00% | 0.00% | 24.05% | 0 |
| train/fog/cls/all | 15376 | 2.86% | 30.76% | 66.39% | 66.39% | 3.908/4.0 | 5.32% | 29.35% | 35.53% | 0 |
| train/fog/cls/associated | 5136 | 1.97% | 92.08% | 5.96% | 5.96% | 1.139/1.0 | 3.23% | 87.87% | 40.97% | 0 |
| train/fog/cls/background | 10240 | 3.30% | 0.00% | 96.70% | 96.70% | 5.296/5.0 | 6.37% | 0.00% | 32.80% | 0 |
| train/fog/reg/all | 15376 | 0.76% | 31.44% | 67.80% | 67.80% | 4.087/5.0 | 0.97% | 25.51% | 28.88% | 0 |
| train/fog/reg/associated | 5136 | 0.66% | 94.12% | 5.22% | 5.22% | 1.139/1.0 | 0.86% | 76.38% | 33.27% | 0 |
| train/fog/reg/background | 10240 | 0.81% | 0.00% | 99.19% | 99.19% | 5.565/5.0 | 1.03% | 0.00% | 26.68% | 0 |
| train/rain/cls/all | 15456 | 2.83% | 31.41% | 65.75% | 65.75% | 3.921/5.0 | 4.90% | 30.01% | 31.09% | 0 |
| train/rain/cls/associated | 5216 | 1.67% | 93.08% | 5.25% | 5.25% | 1.118/1.0 | 2.65% | 88.94% | 34.41% | 0 |
| train/rain/cls/background | 10240 | 3.43% | 0.00% | 96.57% | 96.57% | 5.348/5.0 | 6.04% | 0.00% | 29.39% | 0 |
| train/rain/reg/all | 15456 | 0.52% | 32.03% | 67.45% | 67.45% | 4.083/5.0 | 0.94% | 26.65% | 26.60% | 0 |
| train/rain/reg/associated | 5216 | 0.40% | 94.90% | 4.70% | 4.70% | 1.116/1.0 | 0.73% | 78.97% | 30.29% | 0 |
| train/rain/reg/background | 10240 | 0.59% | 0.00% | 99.41% | 99.41% | 5.594/5.0 | 1.04% | 0.00% | 24.73% | 0 |
| train/snow/cls/all | 15333 | 5.74% | 27.06% | 67.20% | 67.20% | 3.721/4.0 | 10.45% | 24.76% | 45.35% | 0 |
| train/snow/cls/associated | 5093 | 5.75% | 81.46% | 12.78% | 12.78% | 1.413/1.0 | 8.95% | 74.53% | 55.76% | 0 |
| train/snow/cls/background | 10240 | 5.73% | 0.00% | 94.27% | 94.27% | 4.869/5.0 | 11.20% | 0.00% | 40.18% | 0 |
| train/snow/reg/all | 15333 | 1.60% | 28.13% | 70.27% | 70.27% | 4.125/5.0 | 2.37% | 22.82% | 35.26% | 0 |
| train/snow/reg/associated | 5093 | 1.85% | 84.68% | 13.47% | 13.47% | 1.558/1.0 | 2.75% | 68.70% | 46.61% | 0 |
| train/snow/reg/background | 10240 | 1.47% | 0.00% | 98.53% | 98.53% | 5.402/5.0 | 2.18% | 0.00% | 29.61% | 0 |
| validation/clean/cls/all | 4621 | 7.21% | 34.65% | 58.15% | 58.15% | 3.701/4.0 | 9.95% | 33.26% | 41.53% | 0 |
| validation/clean/cls/associated | 1741 | 5.28% | 91.96% | 2.76% | 2.76% | 1.102/1.0 | 5.92% | 88.28% | 49.11% | 0 |
| validation/clean/cls/background | 2880 | 8.37% | 0.00% | 91.63% | 91.63% | 5.273/5.0 | 12.40% | 0.00% | 36.94% | 0 |
| validation/clean/reg/all | 4621 | 0.87% | 36.31% | 62.82% | 62.82% | 4.010/5.0 | 1.28% | 31.94% | 37.46% | 0 |
| validation/clean/reg/associated | 1741 | 0.80% | 96.38% | 2.81% | 2.81% | 1.141/1.0 | 1.38% | 84.78% | 43.02% | 0 |
| validation/clean/reg/background | 2880 | 0.90% | 0.00% | 99.10% | 99.10% | 5.744/5.0 | 1.22% | 0.00% | 34.10% | 0 |
| validation/fog/cls/all | 4564 | 7.58% | 33.76% | 58.65% | 58.65% | 3.658/4.0 | 11.50% | 32.58% | 43.76% | 0 |
| validation/fog/cls/associated | 1684 | 4.93% | 91.51% | 3.56% | 3.56% | 1.122/1.0 | 6.83% | 88.30% | 51.72% | 0 |
| validation/fog/cls/background | 2880 | 9.13% | 0.00% | 90.87% | 90.87% | 5.142/5.0 | 14.24% | 0.00% | 39.10% | 0 |
| validation/fog/reg/all | 4564 | 1.01% | 35.52% | 63.48% | 63.48% | 3.995/5.0 | 1.60% | 30.52% | 41.74% | 0 |
| validation/fog/reg/associated | 1684 | 1.07% | 96.26% | 2.67% | 2.67% | 1.132/1.0 | 1.72% | 82.72% | 48.22% | 0 |
| validation/fog/reg/background | 2880 | 0.97% | 0.00% | 99.03% | 99.03% | 5.669/5.0 | 1.53% | 0.00% | 37.95% | 0 |
| validation/rain/cls/all | 4617 | 7.19% | 34.68% | 58.13% | 58.13% | 3.683/4.0 | 11.26% | 33.35% | 41.04% | 0 |
| validation/rain/cls/associated | 1737 | 4.84% | 92.17% | 2.99% | 2.99% | 1.110/1.0 | 6.39% | 88.66% | 46.80% | 0 |
| validation/rain/cls/background | 2880 | 8.61% | 0.00% | 91.39% | 91.39% | 5.234/5.0 | 14.20% | 0.00% | 37.57% | 0 |
| validation/rain/reg/all | 4617 | 0.74% | 36.30% | 62.96% | 62.96% | 4.028/5.0 | 1.45% | 32.25% | 37.69% | 0 |
| validation/rain/reg/associated | 1737 | 0.86% | 96.49% | 2.65% | 2.65% | 1.140/1.0 | 1.67% | 85.72% | 43.29% | 0 |
| validation/rain/reg/background | 2880 | 0.66% | 0.00% | 99.34% | 99.34% | 5.770/5.0 | 1.32% | 0.00% | 34.31% | 0 |
| validation/snow/cls/all | 4531 | 9.49% | 28.56% | 61.95% | 61.95% | 3.668/4.0 | 14.94% | 26.37% | 51.16% | 0 |
| validation/snow/cls/associated | 1651 | 8.30% | 78.38% | 13.33% | 13.33% | 1.525/1.0 | 12.30% | 72.38% | 64.26% | 0 |
| validation/snow/cls/background | 2880 | 10.17% | 0.00% | 89.83% | 89.83% | 4.897/5.0 | 16.46% | 0.00% | 43.65% | 0 |
| validation/snow/reg/all | 4531 | 1.88% | 31.12% | 67.01% | 67.01% | 4.165/5.0 | 2.78% | 26.35% | 41.91% | 0 |
| validation/snow/reg/associated | 1651 | 2.00% | 85.40% | 12.60% | 12.60% | 1.635/1.0 | 3.45% | 72.32% | 55.12% | 0 |
| validation/snow/reg/background | 2880 | 1.81% | 0.00% | 98.19% | 98.19% | 5.615/5.0 | 2.40% | 0.00% | 34.34% | 0 |

## Scene 覆盖

| Scope | All scenes/max share | Discrete scenes/max share | Beneficial scenes/max share |
|---|---|---|---|
| train/clean/cls/all | 32/4.76% | 27/11.67% | 30/14.38% |
| train/clean/cls/associated | 32/7.92% | 21/25.88% | 23/35.57% |
| train/clean/cls/background | 32/3.12% | 26/12.24% | 30/9.37% |
| train/clean/reg/all | 32/4.76% | 15/12.33% | 19/10.64% |
| train/clean/reg/associated | 32/7.92% | 13/20.00% | 17/17.95% |
| train/clean/reg/background | 32/3.12% | 14/13.21% | 18/9.80% |
| train/fog/cls/all | 32/4.53% | 27/10.93% | 32/12.10% |
| train/fog/cls/associated | 32/7.32% | 24/19.80% | 26/24.10% |
| train/fog/cls/background | 32/3.12% | 27/10.06% | 32/9.05% |
| train/fog/reg/all | 32/4.53% | 16/13.68% | 17/11.41% |
| train/fog/reg/associated | 32/7.32% | 14/14.71% | 15/13.64% |
| train/fog/reg/background | 32/3.12% | 16/14.46% | 17/12.38% |
| train/rain/cls/all | 32/4.62% | 29/13.01% | 30/14.66% |
| train/rain/cls/associated | 32/7.55% | 24/24.14% | 26/30.43% |
| train/rain/cls/background | 32/3.12% | 29/12.25% | 30/11.15% |
| train/rain/reg/all | 32/4.62% | 17/13.58% | 23/13.79% |
| train/rain/reg/associated | 32/7.55% | 14/14.29% | 22/15.79% |
| train/rain/reg/background | 32/3.12% | 17/16.67% | 23/13.08% |
| train/snow/cls/all | 32/4.66% | 32/7.16% | 32/8.80% |
| train/snow/cls/associated | 32/7.74% | 31/10.24% | 32/14.47% |
| train/snow/cls/background | 32/3.12% | 32/7.16% | 32/6.54% |
| train/snow/reg/all | 32/4.66% | 30/12.24% | 31/7.71% |
| train/snow/reg/associated | 32/7.74% | 30/11.70% | 30/8.57% |
| train/snow/reg/background | 32/3.12% | 29/12.58% | 31/8.07% |
| validation/clean/cls/all | 9/15.41% | 9/42.64% | 9/33.26% |
| validation/clean/cls/associated | 9/22.52% | 8/48.91% | 8/44.66% |
| validation/clean/cls/background | 9/11.11% | 9/40.25% | 9/29.97% |
| validation/clean/reg/all | 9/15.41% | 6/35.00% | 6/32.20% |
| validation/clean/reg/associated | 9/22.52% | 6/28.57% | 6/37.50% |
| validation/clean/reg/background | 9/11.11% | 5/38.46% | 6/28.57% |
| validation/fog/cls/all | 9/15.43% | 9/43.64% | 9/29.52% |
| validation/fog/cls/associated | 9/22.80% | 7/56.63% | 7/41.74% |
| validation/fog/cls/background | 9/11.11% | 9/39.54% | 9/26.10% |
| validation/fog/reg/all | 9/15.43% | 6/36.96% | 7/27.40% |
| validation/fog/reg/associated | 9/22.80% | 6/27.78% | 7/24.14% |
| validation/fog/reg/background | 9/11.11% | 5/42.86% | 7/31.82% |
| validation/rain/cls/all | 9/15.42% | 9/49.10% | 9/34.62% |
| validation/rain/cls/associated | 9/22.57% | 7/58.33% | 7/46.85% |
| validation/rain/cls/background | 9/11.11% | 9/45.97% | 9/31.30% |
| validation/rain/reg/all | 9/15.42% | 5/35.29% | 6/37.31% |
| validation/rain/reg/associated | 9/22.57% | 5/33.33% | 6/41.38% |
| validation/rain/reg/background | 9/11.11% | 5/42.11% | 6/34.21% |
| validation/snow/cls/all | 9/15.56% | 9/34.65% | 9/23.78% |
| validation/snow/cls/associated | 9/23.32% | 9/40.15% | 9/29.06% |
| validation/snow/cls/background | 9/11.11% | 9/32.08% | 9/21.52% |
| validation/snow/reg/all | 9/15.56% | 9/29.41% | 8/23.81% |
| validation/snow/reg/associated | 9/23.32% | 8/24.24% | 8/21.05% |
| validation/snow/reg/background | 9/11.11% | 8/32.69% | 8/27.54% |

描述性信号：K0_SIGNAL_WEAK；不构成最终 STOP。KEEP-only 按无竞争 outcome tie 计入，同时单列；并列来源完整集合见 k0_group_records.jsonl。旧 action-level 风险表已逐项复现。本审计不检验重复采样稳定性；K2 尚待人工授权。K3 继续冻结。
