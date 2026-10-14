# Data Adequacy Recommendation

## 一句话结论

建议把 Same 与 Task 分开设样本量规则；本轮仅列候选值，所有门槛待人工冻结，K3 继续冻结。

| 用途 | pooled 正例 n 候选 | scene 候选 | 单 scene 最大占比候选 |
|---|---|---|---|
| K2-Same | 50 / 100 / 200；优先讨论 100 | validation ≥5 / ≥7 | ≤30% / ≤40% |
| K2-Task | 20 / 30 / 50；须先取得同状态真实正例数 | validation ≥5 / ≥7 | ≤30% / ≤40% |
| K3-Same | train 100 / 200 / 500；优先讨论 200 | train ≥10 / ≥15 | ≤30% / ≤40% |
| K3-Task | train 50 / 100 / 200；当前没有真实 train Task labels | train ≥5 / ≥10 | ≤30% / ≤40% |

这些数字用于人工比较资源与可解释程度，不是已冻结规则，也不是统计功效保证。K2-Same 的 cls/reg 应分开计数；K3 还需单独检查 fit/calibration 分配。

## 现有正例与集中度

| Split | Task | adverse positive appearances | 去重物理 group | scenes | max scene share |
|---|---|---:|---:|---:|---:|
| train | cls | 760 | 待 cache | 32 | 19.47% |
| train | reg | 222 | 待 cache | 30 | 10.81% |
| validation | cls | 429 | 待 cache | 9 | 37.06% |
| validation | reg | 115 | 待 cache | 8 | 20.87% |
| train+validation | cls | 1189 | 待 cache | 41 | 13.37% |
| train+validation | reg | 337 | 待 cache | 38 | 7.12% |

## 为什么不沿用每天气 100

K0 的 beneficial group、Same Oracle 的 selected A、Task Oracle 的 selected A、以及同状态 Task 相对 Same 的真实额外正例是不同统计单位。不能用其中一个数量替代另一个。

- FULL fog：Same selected A=58，Task selected A=69，Task−Same recovered 净差=11。
- FULL rain：Same selected A=59，Task selected A=69，Task−Same recovered 净差=12。
- FULL snow：Same selected A=142，Task selected A=168，Task−Same recovered 净差=35。
- A_ONLY fog：Same selected A=58，Task selected A=69，Task−Same recovered 净差=11。
- A_ONLY rain：Same selected A=59，Task selected A=69，Task−Same recovered 净差=12。
- A_ONLY snow：Same selected A=142，Task selected A=168，Task−Same recovered 净差=35。

建议优先讨论 adverse pooled 的 stability 样本数，同时保留 train/validation 和逐天气表。pooled 不能冒充独立泛化证据，跨天气同一 frame/target 须另报去重数。旧 100/weather 可能筛掉现有关键动作，但本轮不因此降低或冻结任何最终门槛。

## 仍待人工决定

1. K2-Same 的对象选单任务 S0 candidate，还是联合 Same 的 GT-target/current-state；两者不能混计。
2. pooled n 使用天气出现次数还是去重物理 group；是否允许 train+validation 合并作稳定性诊断。
3. K2-Task 的 minimum-n、scene 覆盖与集中度上限，待真实同状态正例统计后冻结。
4. K3-Same/Task 各自的 train/fit/calibration 正例要求；当前没有 train joint Task 真标签，Task 学习 adequacy 仍未知。
5. 样本不足如何处理，以及稳定性成功率/效应量/置信度要求；本轮没有预设最终 kill rule。
