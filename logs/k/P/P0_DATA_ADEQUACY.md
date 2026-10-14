# P0-B Data Adequacy Audit

## 一句话结论

adverse pooled 单任务 H1 正例：train cls/reg=760/222，validation=429/115；validation Task−Same recovered 净差=58。同状态 Task 真实正例已从原记录核对，train joint Task 仍未生成。

## H1：原 S0 单任务 beneficial discrete groups

associated 为主结果，background 在各自行单列；cls/reg 不相加。scene ID 带 split 前缀。

| Scope | Total groups | Positives | Fraction | Unique-source | Multi-source | Scenes | Max scene share |
|---|---:|---:|---:|---:|---:|---:|---:|
| train+validation/adverse equal-weight/cls/associated | 6839 | 396.33 | 5.82% | 288.33 | 108 | 41 | 13.37% |
| train+validation/adverse equal-weight/cls/background | 13120 | 1237 | 9.43% | 834.67 | 402.33 | 41 | 9.08% |
| train+validation/adverse equal-weight/reg/associated | 6839 | 112.33 | 1.65% | 87 | 25.33 | 38 | 7.12% |
| train+validation/adverse equal-weight/reg/background | 13120 | 195.33 | 1.49% | 156 | 39.33 | 39 | 5.80% |
| train+validation/adverse pooled/cls/associated | 20517 | 1189 | 5.80% | 865 | 324 | 41 | 13.37% |
| train+validation/adverse pooled/cls/background | 39360 | 3711 | 9.43% | 2504 | 1207 | 41 | 9.08% |
| train+validation/adverse pooled/reg/associated | 20517 | 337 | 1.64% | 261 | 76 | 38 | 7.12% |
| train+validation/adverse pooled/reg/background | 39360 | 586 | 1.49% | 468 | 118 | 39 | 5.80% |
| train+validation/clean/cls/associated | 7034 | 252 | 3.58% | 185 | 67 | 31 | 21.03% |
| train+validation/clean/cls/background | 13120 | 987 | 7.52% | 680 | 307 | 39 | 10.84% |
| train+validation/clean/reg/associated | 7034 | 63 | 0.90% | 52 | 11 | 23 | 14.29% |
| train+validation/clean/reg/background | 13120 | 137 | 1.04% | 119 | 18 | 24 | 7.30% |
| train+validation/fog/cls/associated | 6820 | 281 | 4.12% | 201 | 80 | 33 | 17.08% |
| train+validation/fog/cls/background | 13120 | 1062 | 8.09% | 707 | 355 | 41 | 10.08% |
| train+validation/fog/reg/associated | 6820 | 73 | 1.07% | 59 | 14 | 22 | 9.59% |
| train+validation/fog/reg/background | 13120 | 149 | 1.14% | 125 | 24 | 24 | 9.40% |
| train+validation/rain/cls/associated | 6953 | 249 | 3.58% | 185 | 64 | 33 | 20.88% |
| train+validation/rain/cls/background | 13120 | 1028 | 7.84% | 727 | 301 | 39 | 12.45% |
| train+validation/rain/reg/associated | 6953 | 67 | 0.96% | 55 | 12 | 28 | 17.91% |
| train+validation/rain/reg/background | 13120 | 145 | 1.11% | 112 | 33 | 29 | 9.66% |
| train+validation/snow/cls/associated | 6744 | 659 | 9.77% | 479 | 180 | 41 | 10.02% |
| train+validation/snow/cls/background | 13120 | 1621 | 12.36% | 1070 | 551 | 41 | 6.29% |
| train+validation/snow/reg/associated | 6744 | 197 | 2.92% | 147 | 50 | 38 | 6.09% |
| train+validation/snow/reg/background | 13120 | 292 | 2.23% | 231 | 61 | 39 | 6.51% |
| train/adverse equal-weight/cls/associated | 5148.33 | 253.33 | 4.94% | 174.33 | 79 | 32 | 19.47% |
| train/adverse equal-weight/cls/background | 10240 | 806 | 7.87% | 519 | 287 | 32 | 8.40% |
| train/adverse equal-weight/reg/associated | 5148.33 | 74 | 1.44% | 58.67 | 15.33 | 30 | 10.81% |
| train/adverse equal-weight/reg/background | 10240 | 145 | 1.42% | 115.33 | 29.67 | 31 | 7.59% |
| train/adverse pooled/cls/associated | 15445 | 760 | 4.92% | 523 | 237 | 32 | 19.47% |
| train/adverse pooled/cls/background | 30720 | 2418 | 7.87% | 1557 | 861 | 32 | 8.40% |
| train/adverse pooled/reg/associated | 15445 | 222 | 1.44% | 176 | 46 | 30 | 10.81% |
| train/adverse pooled/reg/background | 30720 | 435 | 1.42% | 346 | 89 | 31 | 7.59% |
| train/clean/cls/associated | 5293 | 149 | 2.82% | 91 | 58 | 23 | 35.57% |
| train/clean/cls/background | 10240 | 630 | 6.15% | 409 | 221 | 30 | 9.37% |
| train/clean/reg/associated | 5293 | 39 | 0.74% | 30 | 9 | 17 | 17.95% |
| train/clean/reg/background | 10240 | 102 | 1.00% | 85 | 17 | 18 | 9.80% |
| train/fog/cls/associated | 5136 | 166 | 3.23% | 110 | 56 | 26 | 24.10% |
| train/fog/cls/background | 10240 | 652 | 6.37% | 402 | 250 | 32 | 9.05% |
| train/fog/reg/associated | 5136 | 44 | 0.86% | 37 | 7 | 15 | 13.64% |
| train/fog/reg/background | 10240 | 105 | 1.03% | 91 | 14 | 17 | 12.38% |
| train/rain/cls/associated | 5216 | 138 | 2.65% | 90 | 48 | 26 | 30.43% |
| train/rain/cls/background | 10240 | 619 | 6.04% | 426 | 193 | 30 | 11.15% |
| train/rain/reg/associated | 5216 | 38 | 0.73% | 29 | 9 | 22 | 15.79% |
| train/rain/reg/background | 10240 | 107 | 1.04% | 77 | 30 | 23 | 13.08% |
| train/snow/cls/associated | 5093 | 456 | 8.95% | 323 | 133 | 32 | 14.47% |
| train/snow/cls/background | 10240 | 1147 | 11.20% | 729 | 418 | 32 | 6.54% |
| train/snow/reg/associated | 5093 | 140 | 2.75% | 110 | 30 | 30 | 8.57% |
| train/snow/reg/background | 10240 | 223 | 2.18% | 178 | 45 | 31 | 8.07% |
| validation/adverse equal-weight/cls/associated | 1690.67 | 143 | 8.50% | 114 | 29 | 9 | 37.06% |
| validation/adverse equal-weight/cls/background | 2880 | 431 | 14.97% | 315.67 | 115.33 | 9 | 26.06% |
| validation/adverse equal-weight/reg/associated | 1690.67 | 38.33 | 2.28% | 28.33 | 10 | 8 | 20.87% |
| validation/adverse equal-weight/reg/background | 2880 | 50.33 | 1.75% | 40.67 | 9.67 | 8 | 22.52% |
| validation/adverse pooled/cls/associated | 5072 | 429 | 8.46% | 342 | 87 | 9 | 37.06% |
| validation/adverse pooled/cls/background | 8640 | 1293 | 14.97% | 947 | 346 | 9 | 26.06% |
| validation/adverse pooled/reg/associated | 5072 | 115 | 2.27% | 85 | 30 | 8 | 20.87% |
| validation/adverse pooled/reg/background | 8640 | 151 | 1.75% | 122 | 29 | 8 | 22.52% |
| validation/clean/cls/associated | 1741 | 103 | 5.92% | 94 | 9 | 8 | 44.66% |
| validation/clean/cls/background | 2880 | 357 | 12.40% | 271 | 86 | 9 | 29.97% |
| validation/clean/reg/associated | 1741 | 24 | 1.38% | 22 | 2 | 6 | 37.50% |
| validation/clean/reg/background | 2880 | 35 | 1.22% | 34 | 1 | 6 | 28.57% |
| validation/fog/cls/associated | 1684 | 115 | 6.83% | 91 | 24 | 7 | 41.74% |
| validation/fog/cls/background | 2880 | 410 | 14.24% | 305 | 105 | 9 | 26.10% |
| validation/fog/reg/associated | 1684 | 29 | 1.72% | 22 | 7 | 7 | 24.14% |
| validation/fog/reg/background | 2880 | 44 | 1.53% | 34 | 10 | 7 | 31.82% |
| validation/rain/cls/associated | 1737 | 111 | 6.39% | 95 | 16 | 7 | 46.85% |
| validation/rain/cls/background | 2880 | 409 | 14.20% | 301 | 108 | 9 | 31.30% |
| validation/rain/reg/associated | 1737 | 29 | 1.67% | 26 | 3 | 6 | 41.38% |
| validation/rain/reg/background | 2880 | 38 | 1.32% | 35 | 3 | 6 | 34.21% |
| validation/snow/cls/associated | 1651 | 203 | 12.30% | 156 | 47 | 9 | 29.06% |
| validation/snow/cls/background | 2880 | 474 | 16.46% | 341 | 133 | 9 | 21.52% |
| validation/snow/reg/associated | 1651 | 57 | 3.45% | 37 | 20 | 8 | 21.05% |
| validation/snow/reg/background | 2880 | 69 | 2.40% | 53 | 16 | 8 | 27.54% |

N/A 表示缺少原 cache/record、train Oracle 未生成或无正例导致比例不可定义，具体原因见 JSON。equal-weight 行的 n 是逐天气均值，scene 列给 pooled 覆盖作为背景；不当作新增样本。完整每 scene 计数保存在 JSON。跨天气重复候选的去重数只有原 cache 可提供。

## H2：真实 Oracle 统计与可用性

Task−Same recovered 为两条独立路径的净数量差，无法提供 Task-only GT 身份或同状态正例数。“selected separated”包含同 CAV 不同 family 和 KEEP/来源分开，不能直接当成跨 CAV 正例。

| Variant/scope | Same recovered | Task recovered | Extra recovered net | Selected separated | Same selected A | Task selected A | Same-state true task-positive | Joint Same positive targets | Scenes/max share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| FULL/train+validation/adverse equal-weight | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 待 record / train 未生成 |
| FULL/train+validation/adverse pooled | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 待 record / train 未生成 |
| FULL/train+validation/clean | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 待 record / train 未生成 |
| FULL/train+validation/fog | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 待 record / train 未生成 |
| FULL/train+validation/rain | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 待 record / train 未生成 |
| FULL/train+validation/snow | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 待 record / train 未生成 |
| FULL/train/adverse equal-weight | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 待 record / train 未生成 |
| FULL/train/adverse pooled | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 待 record / train 未生成 |
| FULL/train/clean | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 待 record / train 未生成 |
| FULL/train/fog | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 待 record / train 未生成 |
| FULL/train/rain | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 待 record / train 未生成 |
| FULL/train/snow | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 待 record / train 未生成 |
| FULL/validation/adverse equal-weight | 62.33 | 81.67 | 19.33 | 1131 | 86.33 | 102 | 21.33 | 86.33 | 8/29.69% |
| FULL/validation/adverse pooled | 187 | 245 | 58 | 3393 | 259 | 306 | 64 | 259 | 8/29.69% |
| FULL/validation/clean | 39 | 52 | 13 | 1265 | 52 | 65 | 14 | 52 | 5/42.86% |
| FULL/validation/fog | 39 | 50 | 11 | 1194 | 58 | 69 | 13 | 58 | 6/30.77% |
| FULL/validation/rain | 43 | 55 | 12 | 1226 | 59 | 69 | 14 | 59 | 5/50.00% |
| FULL/validation/snow | 105 | 140 | 35 | 973 | 142 | 168 | 37 | 142 | 8/27.03% |
| A_ONLY/train+validation/adverse equal-weight | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 待 record / train 未生成 |
| A_ONLY/train+validation/adverse pooled | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 待 record / train 未生成 |
| A_ONLY/train+validation/clean | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 待 record / train 未生成 |
| A_ONLY/train+validation/fog | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 待 record / train 未生成 |
| A_ONLY/train+validation/rain | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 待 record / train 未生成 |
| A_ONLY/train+validation/snow | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 待 record / train 未生成 |
| A_ONLY/train/adverse equal-weight | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 待 record / train 未生成 |
| A_ONLY/train/adverse pooled | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 待 record / train 未生成 |
| A_ONLY/train/clean | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 待 record / train 未生成 |
| A_ONLY/train/fog | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 待 record / train 未生成 |
| A_ONLY/train/rain | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 待 record / train 未生成 |
| A_ONLY/train/snow | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 待 record / train 未生成 |
| A_ONLY/validation/adverse equal-weight | 62.33 | 81.67 | 19.33 | 59 | 86.33 | 102 | 21.33 | 86.33 | 8/29.69% |
| A_ONLY/validation/adverse pooled | 187 | 245 | 58 | 177 | 259 | 306 | 64 | 259 | 8/29.69% |
| A_ONLY/validation/clean | 39 | 52 | 13 | 41 | 52 | 65 | 14 | 52 | 5/42.86% |
| A_ONLY/validation/fog | 39 | 50 | 11 | 39 | 58 | 69 | 13 | 58 | 6/30.77% |
| A_ONLY/validation/rain | 43 | 55 | 12 | 46 | 59 | 69 | 14 | 59 | 5/50.00% |
| A_ONLY/validation/snow | 105 | 140 | 35 | 92 | 142 | 168 | 37 | 142 | 8/27.03% |

Same-state true task-positive 由保存 Task current-state shortlist 的最佳 Task discrete6 严格优于 KEEP+Same 候选最佳 discrete6 定义；仅在原记录可用时计算。这是 shortlist 条件下的真实 outcome 比较，不能冒充穷举最优。联合 Same 表的统计单位是 GT-target ROI/current-state，不能冒充 S0 Shared candidate 的联合反事实。原 S0 只有 cls-only/reg-only outcome，joint Same candidate-level 标签未生成，本轮不运行 detector 来补造。

## Task-separation proxy（辅助，不进入真实正例）

两任务都分别离散优于 KEEP，且最佳集合不相交；分别按 action name 和 CAV ID 计数。这只描述单任务最优集合冲突，没有实际执行联合动作。

| Scope | Both individually beneficial | Disjoint best actions | Disjoint best CAVs |
|---|---:|---:|---:|
| train+validation/adverse equal-weight | 95.33 | 49.33 | 33 |
| train+validation/adverse pooled | 286 | 148 | 99 |
| train+validation/clean | 52 | 25 | 17 |
| train+validation/fog | 58 | 27 | 17 |
| train+validation/rain | 54 | 29 | 18 |
| train+validation/snow | 174 | 92 | 64 |
| train/adverse equal-weight | 62 | 31.67 | 21 |
| train/adverse pooled | 186 | 95 | 63 |
| train/clean | 33 | 15 | 12 |
| train/fog | 33 | 11 | 7 |
| train/rain | 31 | 18 | 13 |
| train/snow | 122 | 66 | 43 |
| validation/adverse equal-weight | 33.33 | 17.67 | 12 |
| validation/adverse pooled | 100 | 53 | 36 |
| validation/clean | 19 | 10 | 5 |
| validation/fog | 25 | 16 | 10 |
| validation/rain | 23 | 11 | 5 |
| validation/snow | 52 | 26 | 21 |

## 能否据此进入 K2/K3

当前仅盘点样本；没有最终 adequacy PASS/FAIL。Same 与 Task 应分别讨论 minimum-n 和 scene 约束。train Task 真标签缺失时，Task 学习样本量保持未知；train+validation true Task pooled 也保持未知。单列 observed_validation_* 只表示已知验证部分。建议候选门槛见 DATA_ADEQUACY_RECOMMENDATION.md，须人工冻结后才可实现下一阶段。
