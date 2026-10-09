# S0 / S6 候选可靠性验证

开发集结果；9 个 validation 场景已经多次参与研究，不构成独立测试。
主 AP 沿用原融合分数，学习值只改变原 NMS 抑制组内顺序。
GT 仅用于训练标签与事后评价。S6 是原缓存特征的确定性变换。

| 天气 | 原流程 AP70 | 二分类 | IoU 回归 | S0 有序 | +cls 稳定性 | +几何稳定性 | S6 两者 | 打乱特征 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| clean | 0.8223 | 0.8223 | 0.8169 | 0.8205 | 0.8237 | 0.8185 | 0.8234 | 0.8194 |
| fog | 0.7858 | 0.7818 | 0.7802 | 0.7788 | 0.7767 | 0.7811 | 0.7779 | 0.7802 |
| rain | 0.8127 | 0.8009 | 0.8040 | 0.8005 | 0.7998 | 0.8025 | 0.8019 | 0.8016 |
| snow | 0.6012 | 0.5988 | 0.5942 | 0.5935 | 0.5954 | 0.5922 | 0.5947 | 0.5914 |

## 预定继续门槛

S0：未通过；S6：未通过。
At least two adverse weathers: mean original-frame-order AP70 gain >=0.005 versus original detector and strongest control, net new GT >0, and positive worst leave-one-scene margin. Clean decrease <=0.001. Passing is only an investment screen on repeatedly used validation scenes.


JSON 含每个种子、天气、方法的 AP30/50/70、TP/FP、GT 新增/丢失、直接抑制边及逐场景剔除结果。先比较 S0 与二分类/IoU 回归，再比较 S6 两者与 S0、单头及打乱特征对照。
旧 validation 上的任何提升仍需新场景复验。
