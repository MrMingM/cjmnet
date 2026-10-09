# Direction B Action Utility Audit

## 一句话结论

当前推理可见的分类输出特征不足以可靠预测分类来源效用。 当前预测几何信息不足以可靠判断回归来源效用。

S0：恶劣天气关联候选的最佳分类/回归来源不同率为 **69.79%**，达到预注册的明显差异线 10%。

S1：S1_WEAK，检验分类来源效用能否从推理输出学会。

S2：S2_WEAK，检验回归来源效用能否从预测几何学会。

S3：S3_FAIL，Conservative 三天气平均 AP70 改变 **+0.0000 pp**；Assisted-Task 为 **+0.1233 pp**，Proposal-Task-Greedy 为 **-0.8838 pp**。

| Method | Clean AP70 | Fog AP70 | Rain AP70 | Snow AP70 | adverse mean ΔAP70 |
|---|---:|---:|---:|---:|---:|
| Shared | 0.822254 | 0.785814 | 0.812677 | 0.601223 | +0.0000 pp |
| Assisted-Cls | 0.820579 | 0.786624 | 0.811931 | 0.604857 | +0.1233 pp |
| Assisted-Reg | 0.822254 | 0.785814 | 0.812677 | 0.601223 | +0.0000 pp |
| Assisted-Task | 0.820579 | 0.786624 | 0.811931 | 0.604857 | +0.1233 pp |
| Proposal-Cls-Greedy | 0.810538 | 0.780065 | 0.799685 | 0.593451 | -0.8838 pp |
| Proposal-Reg-Greedy | 0.822254 | 0.785814 | 0.812677 | 0.601223 | +0.0000 pp |
| Proposal-Task-Greedy | 0.810538 | 0.780065 | 0.799685 | 0.593451 | -0.8838 pp |
| Proposal-Task-Conservative | 0.822254 | 0.785814 | 0.812677 | 0.601223 | +0.0000 pp |

以下后果按 Clean/Fog/Rain/Snow 四条件累计；S3 recovered>lost 门槛单独使用三种恶劣天气。

| Method | recovered | lost | new FP | changed |
|---|---:|---:|---:|---:|
| Shared | 0 | 0 | 0 | 0 |
| Assisted-Cls | 17 | 11 | 30 | 2083 |
| Assisted-Reg | 0 | 0 | 0 | 0 |
| Assisted-Task | 17 | 11 | 30 | 2083 |
| Proposal-Cls-Greedy | 23 | 49 | 158 | 11437 |
| Proposal-Reg-Greedy | 0 | 0 | 0 | 1 |
| Proposal-Task-Greedy | 23 | 49 | 158 | 11438 |
| Proposal-Task-Conservative | 0 | 0 | 0 | 0 |

S1 verdict：**S1_WEAK**

S2 verdict：**S2_WEAK**

S3 verdict：**S3_FAIL**

竞争特征消融（有竞争特征减无竞争特征）：

| Task | Weather | pairwise accuracy Δ | regret Δ |
|---|---|---:|---:|
| classification | clean | +0.0039 | -0.0081 |
| classification | fog | +0.0047 | -0.0094 |
| classification | rain | +0.0044 | -0.0077 |
| classification | snow | +0.0073 | -0.0065 |
| regression | clean | +0.0037 | +0.0000 |
| regression | fog | +0.0025 | +0.0000 |
| regression | rain | +0.0024 | +0.0000 |
| regression | snow | +0.0052 | +0.0000 |

## 目前能得出什么结论

情况 C/D 比较 Assisted-Task 与 Proposal-Task-Greedy，使用相同学习动作和 Greedy 门槛；Conservative 单独用于 S3_PASS 判定，避免把保守阈值造成的少修改归因给候选竞争。

- 情况 A：当前推理可见的分类输出特征不足以可靠预测分类来源效用。
- 情况 B：当前预测几何信息不足以可靠判断回归来源效用。

S1/S2 所有标准化、权重和 margin 门槛均来自固定 train 场景。S3-B 推理决策中的 GT 字段为 0；没有访问正式 test。原 Shared 完整 validation 索引的 AP30/50/70 均已通过 1e-6 复现检查。

Validation 的 S0 和 S3 复用同一份 FP32 推理输出快照，逐帧核对输入和缓存哈希；快照生成时核对原 baseline 的 TP/FP 序列及完整 AP。

本次运行经过浮点漂移修复迁移，保留原版本生成并通过哈希检查的训练标签。原版本身份、诊断证据及迁移记录保存在 protocol.json 的 repairs 字段和 repair_shared_drift/。

## 还不能得出什么结论

这是多次用于开发的 B0 小样本 validation，不能外推为独立测试性能或最终论文结论。最佳来源不同率含同效来源的选择影响，不能当作任务分离 AP 增量。单动作独立标签、整帧多动作、联合分类/回归和帧顺序 AP 之间仍有差异；Assisted 与 Proposal 的差不能单独证明 NMS 是唯一根因。线性探针弱只限定当前特征与线性模型；不能证明所有推理信息都没有可学习信号。

## 下一步应该验证什么

- 优先验证更丰富的 source semantic evidence（来源对目标语义的证据）。
- 优先验证 source-specific geometry evidence / uncertainty（每个来源的几何证据及不确定程度）。

保持本次预注册门槛；不要在看到这些 validation 结果后自动调阈值或重跑挑结果。完整 AP30/AP50、背景指标和重叠冲突计数见 S0–S3_RESULTS.md/json。
