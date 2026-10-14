# Source Utility Stage-2 Precheck

## 一句话结论

adverse pooled 单任务 H1 正例：train cls/reg=760/222，validation=429/115；validation Task−Same recovered 净差=58。Task 同状态正例数与 scene 分布仍待原逐目标记录，train joint Task 未生成。

状态：partial_requires_server_cache。K1-0 = A-dominant descriptive decomposition；本轮没有 K1-0 PASS/FAIL 判定。K3 与所有新模型开发继续冻结。

P0-A：现有汇总无法判断最佳离散动作是否跨 source 等价；须读取原 S0 cache。

P0-B：adverse pooled 单任务 H1 正例：train cls/reg=760/222，validation=429/115；validation Task−Same recovered 净差=58。Task 同状态正例数与 scene 分布仍待原逐目标记录，train joint Task 未生成。

## 当前能回答与仍缺失的证据

- 本地结果是 partial summary：P0-A、去重物理 group、fit/calibration 与 H2 同状态/scene 仍需服务器 cache/record。
- K1-0 仅记 A-dominant descriptive decomposition，§40 没有其 PASS/FAIL 门槛。
- H1 cls/reg 的独立单任务正例不相加冒充 joint Same 正例。
- 独立 greedy 路径的恢复数量差不等价于 Task-only GT 数量，也不等价于同状态 Task 正例。
- 原 S0 没有 joint Same candidate-level outcomes；现有联合 Same 统计单位为 Oracle GT-target/current-state。
- train joint Task Oracle 没有生成；train 及 train+validation true Task n 保持 unavailable/null，proxy 单列。
- 同一 frame 在多天气重复出现，pooled 仅用于样本盘点，不能当作独立泛化证据。
- source 唯一性、扰动稳定性和可学习性未由本轮验证；建议门槛须人工冻结。

## 下一步需人工决定

请根据 redundancy、真实同状态 Task 正例、scene 集中度选择 K2-Same/Task minimum-n，并分别讨论未来 K3-Same/Task 训练正例门槛。候选值见 DATA_ADEQUACY_RECOMMENDATION.md。本轮到此停止，没有训练、GPU 推理、K1、K2 perturbation 或 K3。
