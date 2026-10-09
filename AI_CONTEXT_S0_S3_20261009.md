## 38. 2026-10-09：方向 B S0–S3 Action Utility 可学习性审计完成

### 38.1 实验目的与协议

本轮结果本地归档：`logs/sectionB/all/FINAL_RESULTS.md`；实现目录：`local_fusion_action_utility_audit/`。

本轮不再设计新的深层网络，而是直接回答：第33/36节已经证明存在很大 Oracle 空间后，Oracle 的 source/task action utility 是否能仅靠推理阶段可见信息学出来。

主要协议：

- 继续复用 B0 保存的 train/validation scenes 与 indices；没有访问 OPV2V / OPV2V-W 正式 test。
- train scenes 按 scene 固定拆分 probe-fit / probe-calibration；validation 只用于最终 development 评价，不参与阈值、标准化或模型选择。
- Shared baseline 在完整 validation indices 上的 AP30/AP50/AP70 先按 1e-6 门槛复现；validation 的 S0/S3 共用同一份 FP32 推理输出快照，并核对输入与缓存哈希。
- Shared 每帧产生 top256 proposal；固定 proposal ROI expansion=1.5；候选来源仍为 `KEEP_SHARED`、`single:i`、`query:i`。
- 每一个 counterfactual action 都重新从同一个原始 Shared prediction 出发，不使用逐 GT greedy Oracle 的路径状态作为标签。
- GT 只用于 proposal association、action outcome label 和最终评价；S3-B 的 proposal/action 决策中 GT 字段为 0。
- S1/S2 只使用 `torch.nn.Linear` pairwise ranker，不使用 MLP/Transformer/GNN。目的不是追求最高 AP，而是诊断当前 inference-visible signal 中是否存在可学习的 source utility。

### 38.2 S0：任务专属来源差异很强，但不能把不同率直接当 AP 增益

S0 在恶劣天气关联候选中统计到：最佳 classification source 与最佳 regression source 不同的比例为 **69.79%**，远高于预注册的 10% 明显差异线。

这继续支持第33/36节的结构性结论：同一个目标的分类与定位经常偏好不同协同来源，task-specific source selection 不是由旧 B0 Router 人为制造的小现象。

但需要保留边界：最佳来源不同率包含多个来源产生同效 outcome 后由固定排序规则选出不同 source 的情况，因此 69.79% 不能解释为“69.79% 的目标必须 task-separated”，也不能直接当作 Task-Same 的 AP 增量。

### 38.3 S1 / S2：当前 output-level 特征无法可靠学习 source utility

最终 verdict：

- `S1_WEAK`：当前推理可见的 classification output 特征不足以可靠预测 classification source utility。
- `S2_WEAK`：当前预测 box / geometry 一致性特征不足以可靠预测 regression source utility。

加入 candidate competition feature 后，pairwise accuracy 只小幅变化：

- classification：Clean/Fog/Rain/Snow 分别约 +0.0039/+0.0047/+0.0044/+0.0073；
- regression：分别约 +0.0037/+0.0025/+0.0024/+0.0052；regret 几乎没有额外改善。

尤其 regression selector 在 S3 中几乎完全退化为 KEEP：`Assisted-Reg` 四条件累计 changed=0；`Proposal-Reg-Greedy` 也只有 1 次修改。这说明当前仅根据 source 最终 box 与 Shared/其他 source 的几何差异、共识/离群程度，基本无法判断“哪一个来源真正拥有更好的定位证据”。

本轮只能否定“当前 output-level 手工特征 + 线性 ranker 能可靠学习 source utility”这一版本；不能证明 source utility 本身不可学习，也不能证明更底层的 inference-visible evidence 没有信息。

### 38.4 S3：弱 selector 大规模执行会放大误伤，但 competition 不是当前第一瓶颈

AP70：

| Method | Clean | Fog | Rain | Snow | adverse mean ΔAP70 |
|---|---:|---:|---:|---:|---:|
| Shared | 0.822254 | 0.785814 | 0.812677 | 0.601223 | +0.0000 pp |
| Assisted-Cls | 0.820579 | 0.786624 | 0.811931 | 0.604857 | +0.1233 pp |
| Assisted-Reg | 0.822254 | 0.785814 | 0.812677 | 0.601223 | +0.0000 pp |
| Assisted-Task | 0.820579 | 0.786624 | 0.811931 | 0.604857 | +0.1233 pp |
| Proposal-Cls-Greedy | 0.810538 | 0.780065 | 0.799685 | 0.593451 | -0.8838 pp |
| Proposal-Reg-Greedy | 0.822254 | 0.785814 | 0.812677 | 0.601223 | +0.0000 pp |
| Proposal-Task-Greedy | 0.810538 | 0.780065 | 0.799685 | 0.593451 | -0.8838 pp |
| Proposal-Task-Conservative | 0.822254 | 0.785814 | 0.812677 | 0.601223 | +0.0000 pp |

四条件累计后果：

- Assisted-Cls / Assisted-Task：17 recovered / 11 lost / 30 newFP / 2083 changed；
- Assisted-Reg：0/0/0/0；
- Proposal-Cls-Greedy：23 recovered / 49 lost / 158 newFP / 11437 changed；
- Proposal-Task-Greedy：23/49/158/11438；
- Proposal-Task-Conservative：0/0/0/0。

Assisted-Task 与 Assisted-Cls 完全一致，进一步说明 regression source choice 没有被当前 ranker 学出来。GT 只帮助限定关联 proposal 时仍有极弱的 classification 正信号（恶劣天气平均 +0.1233 pp），但一旦把相同 selector 推到所有 top256 proposal，误伤迅速超过恢复，三天气平均 AP70 变为 -0.8838 pp。

Conservative 版本在 train-calibration 的预注册安全条件下找不到足够可靠的 margin，最终选择全部 KEEP，因此 AP 与 Shared 完全一致。这说明当前 ranker 不只是“阈值没有调好”，而是没有形成一个足够可靠的高置信动作区域。

因此当前更准确的瓶颈顺序是：

1. **第一瓶颈：source utility 本身没有从当前 output-level feature 中被可靠识别；**
2. **第二瓶颈：把弱 action 扩散到大量 candidate 后，candidate competition / collateral harm 进一步放大错误。**

不能由 Assisted 与 Proposal 的差直接声称 NMS 是唯一根因，也不应在 source evidence 尚未解决时直接把复杂 GNN/Transformer competition model 作为下一优先级。

### 38.5 与第33/36/37节合并后的最新研究判断

当前完整证据链：

1. 第33节 Target-wise Task/Source Oracle：Fog/Rain/Snow `Task-Shared` 平均 **+7.72 pp AP70**；`Same-Shared` 平均 +6.36 pp；`Task-Same` 仍有 **+1.36 pp**，方向 B 显著存活。
2. 第36节把 ROI 限制到 Shared top256 proposal 后，`Task-Same` 三天气平均仍约 **+1.379 pp**，说明精确 GT 目标区域不是主要 Oracle 来源。
3. 第37节 CIA-SSD/LMD/SAQC/Learned 3D NMS 表明：候选质量可以预测得很准，但通用 quality/rescore/NMS 方案仍难稳定兑现 Oracle，尤其 LMD 是“质量预测准确但 AP 反而下降”的强反例。
4. 本节 S0 再次说明 cls/reg source 偏好差异是真实现象；S1/S2 则说明当前最终 output 中没有足够容易利用的 source utility 信号；S3 说明弱 selector 大范围执行会造成 lost TP/new FP。

因此方向 B 不再表述为“继续做一个更好的 quality head / action selector”，而重构为：

> **Evidence → Utility → Action → Competition**

也就是先回答每个 source 对当前 candidate 到底持有什么任务证据，再预测其 classification / regression utility，最后决定是否执行，并在必要时处理候选竞争。

当前优先级：

1. regression geometry evidence；
2. classification semantic evidence；
3. candidate competition。

### 38.6 下一步：S4 Source Evidence Audit

下一步不直接训练正式主网络，先做 S4：验证加入更底层的 source evidence 后，是否能把 S1/S2 从 `WEAK` 推到 `LEARNABLE`。

优先检查现有冻结流水线已经能提供的 inference-visible 信息：

- per-source 三尺度 BEV levels（`[source,C,H,W]`）；
- single/query 输出及与 Shared 的变化；
- candidate ROI 内的 source local feature norm/variance/consensus；
- source leave-one-out 后 Shared 分类/回归响应变化，作为 marginal contribution；
- GSPR 已冻结输出：`point_reliability`、`point_uncertainty`、`point_evidence`、`pillar_reliability`、`pillar_uncertainty`；
- 在坐标语义能够严格核对的前提下，候选预测框区域内的点/voxel/pillar 支持、覆盖、距离、观察角度等几何证据。

S4 继续使用与 S1/S2 完全相同的 counterfactual labels、scene split、candidate pool 和线性 pairwise ranker，优先只改变 evidence feature。只有新 evidence 在相同协议下明显提升 source utility learnability，才值得进入正式 source-evidence module；若仍失败，再讨论非线性 interaction 或 action family 是否不足，不能直接跳到复杂 candidate GNN。

### 38.7 当前结论边界

**目前能得出：**target/task/source 异质性是真实而且 Oracle 空间足够；当前 output-level feature 不能可靠逼近 Oracle source utility；大范围执行弱动作会产生明显 collateral harm。

**还不能得出：**不能证明 source utility 无法学习；不能证明 candidate competition 是唯一根因；不能把本节 development validation 结果当独立 test 或最终论文性能；不能把 69.79% 来源不同率当作 AP 增量。

**下一步应该验证：**S4 是否能用更底层 source semantic / geometry / GSPR evidence 显著提升 classification 和 regression source utility 的可学习性，优先观察 regression。
