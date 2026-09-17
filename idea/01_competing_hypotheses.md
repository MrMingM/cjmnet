# Q-A：恶劣天气下“观测不足”竞争假设研究地图

> 更新日期：2026-09-17  
> 本文件服务于 `idea/00_research_workflow.md` 的前两阶段：**现象 → 竞争假设 → 可证伪预测 → 最低成本验证 → 证据有效性审查**。  
> 当前仍只研究 Q-A，不设计新网络，不把诊断现象直接写成机制结论。

---

## 0. 总研究方向与当前问题

本项目总方向始终是：

> **恶劣天气下协同感知鲁棒性。**

当前研究问题 Q-A：

> **在没有 GT、没有 clean frame 的实际推理阶段，自车如何判断一个局部区域是“观测不足 / 未充分观测”，还是“已经充分观测，并且有理由相信该区域为空”？**

但第一轮诊断已经表明，最终 `weather miss` 不能直接等同于“自车没看到”。因此当前 Q-A 必须同时保留以下竞争解释：

1. sensing evidence 的量真的不足；
2. 数量相近，但目标相关结构不同；
3. “未观测”和“可信为空”需要 visibility/free-space 信息；
4. 遮挡与天气存在交互；
5. 当前 evidence-strength 指标可能把错误/无效证据误判为 strong；
6. 真正 task-usable 的 source evidence 可能存在，但 local downstream 或 fusion 没有正确利用。

当前目标不是证明某一条，而是继续排除解释。

---

# 1. 第一轮 Stage-1 已完成实验

代码：`qa_observation_diagnostic/`  
运行：OPV2V validation + online clean/fog/rain/snow，未使用 OPV2V-W test 做假设筛选。  
已完成运行：`/data/cjm/datasets/logs/qa_observation_20260916_194240`

主要输出：

- `hypothesis_report.md`
- `hypothesis_results.json`
- `{clean,fog,rain,snow}/targets.jsonl`
- `{clean,fog,rain,snow}/frames.jsonl`

这些结果属于 development diagnostic，不是论文最终结论。

---

# 2. Stage-1 已确认事实

## F1：Quantity proxies 高度冗余，但“quantity dominant”尚未被证明

逐 target 统计得到：

| Weather | Pearson(region N_eff, reliable count) | Spearman | Partial corr | Pearson(region N_eff, box N_eff) |
|---|---:|---:|---:|---:|
| Clean | 0.8695 | 0.9409 | 0.8067 | 0.8698 |
| Fog | 0.8709 | 0.9484 | 0.8083 | 0.8711 |
| Rain | 0.8697 | 0.9414 | 0.8068 | 0.8699 |
| Snow | 0.8676 | 0.9379 | 0.7837 | 0.8683 |

因此：

> `region N_eff`、reliable count、box N_eff 应视为同一个 **quantity family**，不应包装成多个独立创新。

但这只能证明 proxy 冗余，**不能单凭相关性证明 quantity 是唯一或主要机制，也不能据此否定 quantity 的重要性。**

---

## F2：Structure 在控制 quantity 后仍留下增量信号，但尚不能声称“结构比数量更重要”

Stage-1：

| Weather | Quantity AUC | Coverage AUC | Coverage partial corr | Quantity-matched positive-bin fraction |
|---|---:|---:|---:|---:|
| Clean | 0.9366 | 0.9379 | 0.2634 | 0.9000 |
| Fog | 0.9390 | 0.9350 | 0.2429 | 0.8966 |
| Rain | 0.9352 | 0.9350 | 0.2653 | 0.8148 |
| Snow | 0.8128 | 0.7931 | 0.1523 | 0.7333 |

正确解释是：

> quantity 本身仍然很强，但在控制 quantity、distance、box size、occlusion 后，coverage/structure 仍有一定条件增量。

不能写成：

> “structure 单独优于 quantity”。

当前 H-A2 只通过了第一轮 Cheap Gate，仍需要更严格验证。

---

## F3：Stage-1 的 `strong / weak evidence` 只是经验标签，不是真值

Stage-1 使用 clean-detected targets 的 distance-bin Q25 阈值，将 `box N_eff + coverage4x4` 用作 strong/weak 判据。

在 weather-induced ego misses 中，得到大量 `ego strong` 或 `peer strong` case，尤其 Snow：

- Fog：ego-strong weather-miss fraction ≈ 0.359
- Rain：≈ 0.363
- Snow：≈ 0.560

但当前**不能**据此声称：

> “这些目标已经保留足够正确的信息，只是网络没用好。”

仍有一个同等重要的竞争解释：

> **我们的 evidence-strength ruler 本身误判了 task-usable evidence。**

例如 Snow 下，散射/错误回波可能抬高点数、coverage、feature magnitude，而不增加正确车辆语义。

因此当前最重要的新问题是：

> **Snow 中这些被判为 strong 的目标，是真的还有 task-usable evidence，还是我们的尺子不准？**

---

## F4：Weather miss 不能直接等同于 sensing insufficiency

Stage-1 的 weather-induced ego miss partition：

### Fog（3377 targets）

- ego weak + peer strong + full recovers：1527（45.22%）
- ego weak + peer strong + full still misses：166（4.92%）
- ego strong + full miss：362（10.72%）
- ego strong + full recovers：849（25.14%）
- all-source weak：473（14.01%）

### Rain（1318 targets）

- ego weak + peer strong + full recovers：673（51.06%）
- ego weak + peer strong + full still misses：32（2.43%）
- ego strong + full miss：111（8.42%）
- ego strong + full recovers：367（27.85%）
- all-source weak：135（10.24%）

### Snow（5961 targets）

- ego weak + peer strong + full recovers：1009（16.93%）
- ego weak + peer strong + full still misses：753（12.63%）
- ego strong + full miss：1462（24.53%）
- ego strong + full recovers：1874（31.44%）
- all-source weak：863（14.48%）

这些 partition 是基于经验 strong/weak label 的诊断，不是机制证明。

尤其 Snow 中 `ego strong` 比例高，只说明：

> **存在大量无法被“低 quantity / 低 coverage”简单解释的漏检。**

它没有证明 local representation/detector failure 已成立。

---

## F5：Approximate visibility/free-space 第一层 Gate 未通过

`background_zero - target_zero` 的 high-traversal gap：

- Clean：0.0819
- Fog：0.0464
- Rain：0.0308
- Snow：0.0589

未达到预设的明显分离门槛。

因此：

> 暂不投入精确 CARLA beam reconstruction / semantic LiDAR replay。

同时现有 `endpoint_positive` 在 `known=False` 的定义下天然接近 0，不能把它当作独立 free-space 证据。

H-A3 当前应 **PAUSED / WEAKENED**，不是被永久证伪。

---

## F6：遮挡 × 天气存在条件性信号，但不是统一机制

Stage-1 high-vs-low occlusion weather-miss risk ratio：

- Fog：2.45
- Rain：2.69
- Snow：0.75

因此：

> 遮挡在 Fog/Rain 中可能显著放大天气损伤，但 Snow 方向不同。

当前把 occlusion 作为 **weather-dependent moderator / 分层变量**，不作为统一主机制。

注意：原报告中 Snow low-occlusion `relative N_eff loss = -1.66e6` 是近零 clean denominator 导致的数值爆炸，不能物理解读。

Stage-2 已改为同时报告：

- clean N_eff=0 样本比例；
- absolute delta；
- bounded symmetric delta；
- log1p delta；
- relative loss 仅在 clean N_eff>0 时计算。

bounded metric 是新指标，不能继续叫旧的 relative loss。

---

# 3. 当前六个竞争假设及最新状态

## H-A1：Quantity-dominant / Quantity-family hypothesis

### 当前状态：**SUPPORTED AS STRONG BASELINE / OPEN AS DOMINANT MECHANISM**

已知：quantity proxies 高度冗余，且 quantity AUC 很强。

仍需回答：

> quantity 是否已经解释 observation sufficiency 的大部分变化，还是 structure/task-usable evidence 在 quantity-matched 条件下仍有稳定独立作用？

### 不能声称

- 不能因为 quantity proxies 高相关，就说 quantity 不重要；
- 也不能因为 quantity AUC 高，就说 quantity 已经是唯一机制。

---

## H-A2：Structure-dominant incremental hypothesis

### 当前状态：**OPEN / PASSED FIRST CHEAP GATE**

真正竞争点不是“框内点数 vs 区域点数”，而是：

> 在 quantity 被匹配/控制后，目标表面覆盖、空间结构、局部 target-support pattern 是否仍提供稳定增量。

Stage-1 给出正信号，但还需要：

- scene/distance 分层；
- threshold sensitivity；
- 避免把雪散射造成的“伪 coverage”解释成正确目标结构。

如果后续结构指标只是在 Snow 噪声下虚高，则 H-A2 必须重新定义或降级。

---

## H-A3：Visibility/free-space hypothesis

### 当前状态：**WEAKENED / PAUSED**

Stage-1 approximate traversal 未形成足够强的分离。

暂不做高成本精确 ray reconstruction。

只有未来出现新的直接证据表明 free-space/unknown 区分具有明显增益，才重新打开该分支。

---

## H-A4：Occlusion × Weather interaction

### 当前状态：**PARTIALLY SUPPORTED AS MODERATOR**

Fog/Rain 有明显风险放大，Snow 不一致。

因此当前用途是：

- scene/case 分层；
- 解释 weather heterogeneity；
- 检查后续现象是否只集中在严重遮挡目标。

不把它单独升级为主创新。

---

## H-A5：Local downstream bottleneck

### 当前状态：**OPEN — NOT YET PROVEN**

当前观察：

> 一批 weather miss 在 `N_eff/coverage` 尺子下被判为 ego strong，Snow 尤其多。

但存在两个竞争解释：

### H-A5a：Evidence ruler overestimates local evidence

所谓“strong”只是：

- 点数不少；
- coverage 不低；
- feature magnitude 大；

但缺少正确的 target semantics。

### H-A5b：Task-usable evidence survives, downstream fails to use it

真正有效的信息仍存在，但在：

- PillarVFE / backbone；
- detector head；
- score / post-processing；

某处没有转化成正确检测。

Stage-2 必须先区分 H-A5a vs H-A5b，不能直接宣布“网络看到了但没认出来”。

后续若做 clean-information replacement，只能解释为：

> **某个 intervention site 的可恢复性。**

不能把逐层替换直接解释成：

> “错误最初发生在哪一层”。

不同替换位置的 recovery set 可以重叠，也不能相加成互斥故障比例。

---

## H-A6：Fusion bottleneck

### 当前状态：**OPEN — NEED SOURCE VALIDITY CHECK**

Stage-1 的 `ego weak + peer strong + full still miss` 只能称为：

> **metric-defined fusion candidates**

因为 `peer strong` 仍来自 N_eff/coverage proxy。

真正更强的机制候选必须满足：

> `ego miss + peer-alone correctly detects the same GT + full fusion miss`

定义为：

> **source-valid / fusion-invalid candidate**

只有这种 case 才说明：

1. 某个 peer 自身确实含有 detector 可利用的 task evidence；
2. full fusion 后该目标没有保留为正确检测。

即使如此，也只能说是 fusion-bottleneck candidate，不能直接说 AttFuse attention 是唯一原因。

---

# 4. Stage-2：证据有效性审查（当前正在执行）

新代码目录：`qa_evidence_validity/`

目标不是增加新假设，而是检查 Stage-1 中“strong/weak evidence”这把尺子是否可靠。

Stage-2 分两部分。

---

## E-B0：完全离线证据有效性重分析

文件：`qa_evidence_validity/offline.py`

不重跑模型，直接复用 Stage-1 `targets.jsonl`。

### B0.1 Threshold sensitivity

使用 clean-detected distance-bin：

- Q20
- Q25
- Q30
- Q40

重新定义 strong/weak，报告：

- ego-strong fraction range；
- peer-strong fraction range；
- 与 Q25 的 Jaccard；
- fusion candidate 集合稳定性。

### 目的

如果结论只在某一个阈值成立，则：

> strong/weak ruler 不稳定，不能进入机制解释。

阈值稳定只是必要条件，不等于语义正确。

---

## E-B1：Snow / weather N_eff 变化重新统计

修复原 near-zero denominator 问题。

同时报告：

- clean N_eff=0 cases；
- absolute difference；
- bounded symmetric difference；
- log1p difference；
- relative loss（只对 clean N_eff>0）。

不再用 epsilon 把零分母样本压进一个夸张相对比值。

---

## E-B2：Scene / distance concentration

对：

- ego strong weather miss；
- metric-defined fusion candidate；
- full recovery；

按 scene、distance 分层。

特别检查 top-3 scene share。

如果现象高度集中于少数场景，则先视为 scenario-specific，不能升级为一般机制。

---

## E-B3：Ego → Full degradation audit

直接统计：

- ego yes → full yes；
- ego yes → full no；
- ego no → full yes；
- ego no → full no。

重点检查：

> **自车本来能检测，但 full fusion 反而使目标丢失。**

并按天气、scene 汇报。

这是比“peer strong/full miss”更直接的 harmful-fusion 现象，但仍需要 source/score/duplicate 等后续解释。

---

## E-B4：Peer-alone task-usable evidence audit —— Stage-2 最关键实验

文件：`qa_evidence_validity/peer_audit.py`

对所有 Stage-1 weather-induced ego miss candidates：

> 每个 peer 的 pre-fusion multiscale features 单独经过原 frozen deblocks + cls/reg head + 原 post-processing。

注意：

- 不使用 AttFuse；
- 不加入 ego feature；
- 不是“ego+peer mask”伪单来源；
- peer features 已由原 pipeline 对齐到 ego frame；
- 评估同一 ego-frame GT；
- final criterion 为 IoU>=0.7 greedy matched target；
- 同时记录 best post-processed IoU / score / frame FP。

### 这一步回答

> Stage-1 所谓 `peer strong`，到底有多大比例真的能由该 peer 单独产生正确检测？

输出 proxy-vs-task truth 的：

- source-level precision / recall；
- target-level any-peer precision / recall；
- metric fusion candidates 中有多少被 peer-alone 验证；
- source-valid/full-miss 数量；
- source-valid/full-recover 数量；
- scene / distance 分布。

---

# 5. Stage-2 Gate：实验结束后如何决策

Stage-2 结束前，禁止设计新的 fusion/local 网络。

## Gate A：Evidence ruler 是否可信？

如果 Q20-Q40 非常敏感，或 `peer strong` 对 peer-alone detection precision 很低：

> 先修 evidence ruler；H-A5/H-A6 不能继续用旧 strong/weak 标签。

这意味着当前核心问题更接近：

> **sensing quality / feature magnitude ≠ task-usable evidence。**

---

## Gate B：是否存在可靠 fusion candidates？

如果存在跨多个 scene、多个 distance 的：

> `peer-alone detected + full miss`

则 H-A6 获得更强支持，可以进入小规模 source-combination / local-fusion intervention。

后续 intervention 必须同时记录：

- corrected misses；
- lost detections；
- FP change；
- score / IoU；
- scene consistency。

不能用某一个局部增强操作失败来否定所有融合方法，也不能只看 recall 不看误伤。

---

## Gate C：Local H-A5 是否值得进入 intervention？

Stage-2 仍不能直接证明 high local feature magnitude = 正确 target semantics。

只有当 local evidence validity 有进一步支持时，才做：

> **Hierarchical Clean-information Intervention**

它的正确问题是：

> “在某个层级注入 clean information，能否恢复目标？”

而不是：

> “错误最初发生在哪一层？”

因此结果只能汇报各 intervention site 的 recoverability，不能把不同层的恢复比例当成互斥 failure-origin percentage。

---

# 6. 当前优先级

当前不再扩展新的大假设。

优先级固定为：

1. **完成 Stage-2 evidence validity audit**；
2. 判断 strong/weak ruler 是否可靠；
3. 判断是否存在足够多、跨场景的 source-valid/full-miss；
4. 只有通过 Gate 的分支才进入有限 Oracle / Counterfactual intervention；
5. Oracle 有明显、稳定、低误伤空间后才设计可部署方法；
6. 最后才进入 Cheap Gate → Full Experiment → mechanism verification。

---

# 7. 当前一句话科学问题

当前最值得回答的问题已经收缩为：

> **恶劣天气下，我们观测到的高点数、高覆盖或高特征响应，究竟代表真正可用于检测的 task evidence，还是只是天气噪声/错误表征造成的“假强证据”；如果确实存在 task-usable evidence，它又是在 local downstream 还是 multi-agent fusion 中被丢失？**

这句话同时连接：

- H-A1：quantity；
- H-A2：structure；
- H-A5：local evidence validity / downstream；
- H-A6：source validity / fusion。

当前实验工作的目的就是把这几个解释进一步拆开，而不是马上增加新模块。
