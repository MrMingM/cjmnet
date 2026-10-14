# Source Utility Kill Protocol — Stage 1

## 一句话结论

K0′ 的恶劣天气主范围主要表现为连续分数/IoU 排序主导。 AOnly 保留 Full 大部分收益，BOnly 较弱：Oracle headroom 主要来自改变离散检测后果的动作。

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

## K1-0 主表：AP70（0–1）

| Weather | Shared | Full-Same | Full-Task | AOnly-Same | AOnly-Task | BOnly-Same | BOnly-Task |
|---|---:|---:|---:|---:|---:|---:|---:|
| clean | 0.822254 | 0.860024 | 0.871309 | 0.859931 | 0.871223 | 0.822269 | 0.822298 |
| fog | 0.785814 | 0.830193 | 0.840669 | 0.830171 | 0.840632 | 0.785799 | 0.785801 |
| rain | 0.812677 | 0.851623 | 0.859157 | 0.851572 | 0.859124 | 0.812709 | 0.812664 |
| snow | 0.601223 | 0.708617 | 0.731479 | 0.708514 | 0.731349 | 0.601315 | 0.601224 |
| adverse mean | 0.733238 | 0.796811 | 0.810435 | 0.796753 | 0.810368 | 0.733275 | 0.733230 |

## K1-0 收益（pp）

| Weather | Full Task−Shared | A Task−Shared | B Task−Shared | Full Task−Same | A Task−Same | B Task−Same |
|---|---:|---:|---:|---:|---:|---:|
| clean | +4.9055 | +4.8969 | +0.0044 | +1.1285 | +1.1292 | +0.0029 |
| fog | +5.4855 | +5.4818 | -0.0013 | +1.0476 | +1.0461 | +0.0002 |
| rain | +4.6480 | +4.6447 | -0.0013 | +0.7534 | +0.7552 | -0.0046 |
| snow | +13.0255 | +13.0125 | +0.0001 | +2.2861 | +2.2835 | -0.0091 |
| adverse mean | +7.7197 | +7.7130 | -0.0009 | +1.3624 | +1.3616 | -0.0045 |

decomposition 是 restricted-policy attribution，不是线性可加分解。 AOnly/BOnly 的收益不相加解释为 Full。

## 目前能得出什么结论

K0′ 的恶劣天气主范围主要表现为连续分数/IoU 排序主导。 AOnly 保留 Full 大部分收益，BOnly 较弱：Oracle headroom 主要来自改变离散检测后果的动作。
K0 描述性信号：K0_SIGNAL_WEAK。细分 split、并列数量、scene 集中度见 K0_GROUP_MARGIN.md/json；完整 AP30/50/70 与动作统计见 K1_0_ORACLE_DECOMPOSITION.md/json。

## 还不能得出什么结论

尚不能确认 source-specificity、跨采样 label stability 或实际学习能力。A/B/E 只对应原 outcome key，不等价于逐框身份/分数完全一致。GT hindsight Oracle 是 development 诊断，不能外推部署或正式 test。三条 greedy 路径相互独立，收益没有可加要求。K3 保持冻结。

## 下一阶段 K1/K2 应重点验证什么

下一阶段 K1 尤其需要 N2 random source mixing，检查收益是否依赖正确来源。
K1 检验来源特异性，K2 检验标签稳定性；二者 PASS 后才允许恢复 K3。当前阶段到此结束，等待人工判断，不自动启动任何下一阶段。14 天上限属于 resource stop，不构成 hypothesis rejection。
