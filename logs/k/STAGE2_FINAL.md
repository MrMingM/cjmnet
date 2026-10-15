# Source Utility Kill Protocol — Stage 2

## 一句话结论

SOURCE UTILITY MAINLINE STOP。

- K1_SAME = INCONCLUSIVE
- K1_TASK = INCONCLUSIVE
- K2_SAME = INCONCLUSIVE
- K2_TASK = INCONCLUSIVE
- H1_SOURCE_SPECIFIC = STOP
- H2_TASK_SPECIFIC = NOT_APPLICABLE

## 复现修复与来源

K1 引用原 RUN 的完整结果，保留原 identity；K2 来自独立修复 RUN。所有继承的科学配置、源码和输入哈希均核验一致。
逐帧连续值采用原 S0 数值复现约定，离散后果、合法动作和最终选择严格核验；全部禁用且 outcome8 等于 KEEP 的 E 候选允许保留数量的排列变化；原 Source AP 门禁仍为 1e-6。

## 触发门槛

逐 family 的冻结阈值、置信区间和失败原因见 K1/K2 报告；缺失结果按 INCONCLUSIVE 处理。

## 结论边界

仅限当前 development/validation，不证明可学习、可部署或正式 test 泛化。K3 继续冻结。

## E 候选排列工程修复

仅对全部为 E、全部 allowed=false、各自 outcome8 等于 KEEP、且最终均选 KEEP 的目标，允许相同 source-pair 多重集合改变排列。推理动作、ROI、标签及 §42 科学门槛不改。Source AP 引用前一修复 RUN 的八项完整复现凭证，全部原始哈希与历史 AP 重新核验；未重新前向。逐帧原始差异和获准例外的证明保存在 reproduction.json。
