# Q-A：恶劣天气下“观测不足”竞争假设研究地图

> 更新日期：2026-09-16  
> 本文件只执行 `idea/00_research_workflow.md` 的第 1 阶段：**现象 → 多个竞争假设 → 可证伪预测 → 最低成本验证顺序**。  
> 当前只研究 Q-A，不同时展开通信选择、CEIF 融合、lossless communication 或其它方法设计。

---

## 0. 研究边界

### 0.1 总研究方向没有改变

本项目的总研究方向始终是：

> **恶劣天气下协同感知鲁棒性。**

当前只聚焦其中一个基础问题：

> **Q-A：在没有 GT、没有 clean frame 的实际推理阶段，自车如何判断一个局部区域到底是“观测不足 / 未充分观测”，还是“已经充分观测，并且有理由相信那里为空”？**

后续所有通信和融合方法都应该建立在这个问题被澄清之后，而不是反过来用某个网络结构定义问题。

### 0.2 本阶段不解决什么

本文件暂时不研究：

- 应该向哪辆邻车请求信息；
- 哪一块邻车特征 task utility 更高；
- 通信预算与压缩；
- CEIF 如何融合；
- missing evidence 与 negative evidence 之后应该采取什么融合动作；
- 新的 attention / Transformer / MLP 应该怎么设计。

这些都属于后续问题。

当前只回答：

> **“我这里到底是真的缺证据，还是已经有足够证据认为这里没有目标？”**

---

## 1. 为什么必须重新定义“观测不足”

最简单的需求估计曾经是可靠性加权点数：

\[
N_{eff}(R)=\sum_{p\in R} r^p
\]

其中 `r^p` 来自 GSPR 点级 reliability。

但已有实验已经说明，不能直接把“点的质量/数量”写成“区域是否需要帮助”。

### 1.1 已确认事实

#### F1：GSPR 能很好地判断“已有点是否像天气噪声”

【事实】GSPR full_v1 的点级监督在当前训练/验证体系中表现稳定，并且相对原始 AttFuse 在 Fog/Rain/Snow 正式检测上获得稳定收益。

证据：`AI_CONTEXT.md §7.6–7.7`，以及 `attfuse_gspr/`、`gspr_supervision/`。

因此：

> **点级 weather reliability 本身是有价值的。**

但这不等于它已经解决区域观测充分性。

#### F2：GSPR 对“没有点”的区域没有直接意见

【事实】GSPR 只对体素化后保留的点产生 reliability / uncertainty；空 pillar 没有点级观测。

因此：

- 0 个点可能表示那里确实为空；
- 也可能表示目标被雾衰减掉；
- 也可能表示遮挡；
- 也可能表示距离过远、角分辨率不足；
- 还可能只是 LiDAR 没有射线有效覆盖。

所以：

> **absence of return 不能直接解释为 evidence of absence。**

#### F3：低 reliability 不等于低任务价值

【事实】已有 full-communication 删除诊断中，低 reliability 区域在 Clean/Fog/Rain 的多档删除总体降低 AP70，并且经常比等量随机删除更差；Snow 只在特定比例出现局部正结果。

证据：`AI_CONTEXT.md §12.1`、`gspr_evidence/HARM_DIAGNOSTIC.md`。

因此已经明显削弱：

> `低平均 reliability → 该区域无用 / 应删除 / 不需要保留`

这种直接映射。

#### F4：可靠地面点很多，不代表目标证据充分

【事实 + 定义边界】区域里的可靠点可能主要来自地面、建筑或其它背景；大量可靠点不能直接证明目标表面被充分观测。

反过来，车辆只需要少量但位置关键的车体回波，也可能已经足够支持检测。

因此：

> **“观测量”与“任务相关证据量”不是同一个量。**

#### F5：检测漏掉一个目标，也不自动意味着 sensing evidence 不足

【事实 + 待拆分】最终 detection miss 可能来自：

1. 物理回波真的丢失；
2. 天气噪声覆盖了有效回波；
3. 特征编码没有保留已有证据；
4. backbone / head 没有正确利用证据；
5. score / NMS / 定位误差使其在指标中表现为 miss。

所以在研究需求图之前，必须排除一个危险的偷换：

> **“模型没检测到” ≠ “自车没看到”。**

---

## 2. 当前核心现象

### P-A1：点级可靠性成功，但区域级“是否需要帮助”仍然未知

GSPR 已经说明“天气污染点”可以被学习和利用，但 low-r / high-u 到区域 task utility 的直接映射并不稳定。

真正需要解释的是：

> 为什么一个优秀的 sensing-quality signal，不能直接变成 observation-sufficiency signal？

### P-A2：最困难的不是“低质量点”，而是“没有证据时如何解释”

对于已经存在的点，可以讨论 reliability。

但对一个低密度甚至零回波区域，本地当前帧存在根本性歧义：

> **那里真的为空，还是因为天气/遮挡/采样导致没看见？**

这正是 Q-A 的核心。

### P-A3：需求不能简单由 detector confidence 定义

低检测置信度可能来自证据不足，也可能来自分类/定位困难；高置信背景也不意味着某个潜在目标区域被充分观测。

所以：

> detection confidence 可以是线索，但不能在没有验证的情况下被定义成“观测充分性”。

### P-A4：部分“观测不足”可能实际上是下游模型失败

如果某些 weather miss 中，自车仍然保留了足够车体回波或局部语义特征，那么真正的问题不在 sensing observation，而在 representation / detector。

如果这一类比例很高，那么继续设计“需求图”会解决错问题。

---

# 3. 六个竞争假设

下面六个假设不是六个准备组合起来的新模块。

它们是对同一个问题的**不同机制解释**。当前目标是尽可能便宜地把错误解释排除掉。

---

## H-A1：可靠证据“数量不足”就是主要原因

### 假设内容

恶劣天气下，大多数真正的 observation insufficiency，本质上仍然是有效回波支持量下降。

如果把普通点数升级为可靠性加权支持量，并控制距离、目标大小、LiDAR 角度等因素，那么 `N_eff` 应该能够较好地区分：

- 证据充分区域；
- weather-induced evidence loss 区域。

也就是说，之前失败的可能不是“数量思想”，而只是原始 density 太粗糙。

### 如果它是真的，应该观察到什么

1. 在控制距离、目标尺寸、视角后，weather miss 的 target-region `N_eff` 应显著低于正常检测目标。
2. 同一个 GT 目标在 clean→weather 发生 ego detection 退化时，`N_eff` 应同步明显下降。
3. 少量其它语义/跨车特征加入后，对判别能力的增量应该有限。
4. 高 `N_eff` 的目标极少出现“真正 sensing evidence 不足”。

### 什么结果会显著削弱/否定它

如果出现大量：

- `N_eff` 很高但 weather ego 仍明显缺失；
- `N_eff` 很低但目标仍稳定检测；
- 在相同 `N_eff` 下，目标结果仍因空间结构/语义分布产生巨大差异；

则“可靠点数量是主要变量”的解释应降级。

### 最便宜验证

不训练模型。

利用已有 paired clean/weather、GT、GSPR 输出和 ego prediction，按 target 做分层统计：

- raw point count；
- reliable point count；
- `N_eff`；
- distance；
- target size；
- clean/weather detection state。

先看分布和 AUROC，不设计网络。

---

## H-A2：真正决定充分性的不是点数，而是“任务相关语义证据”

### 假设内容

区域里有多少可靠点并不重要，关键是这些点是否形成了足以支持目标检测的结构/语义证据。

因此：

> **可靠地面点 100 个可能没有用；车体关键表面点 5 个可能已经足够。**

观测充分性应该更接近“是否存在 target-supporting evidence”，而不是“总证据质量”。

### 如果它是真的，应该观察到什么

1. 在 `N_eff` 相近的目标中，车体表面覆盖、局部 BEV 前景响应或弱 proposal 强度应明显区分成功/失败目标。
2. 少点但高度集中在目标几何表面的样本，可以稳定检测。
3. 多点但主要落在背景/地面的区域，仍可能对目标检测没有支持。
4. 语义/结构指标在控制 `N_eff` 后仍有明显增量预测能力。

### 什么结果会显著削弱/否定它

如果在控制可靠点数量、距离和几何之后，语义/结构特征几乎不能额外区分 observation sufficiency，那么它不是主要机制。

### 最便宜验证

仍然不训练网络。

对已有 target-region 统计：

- 目标框内/边界附近可靠点比例；
- 点的空间覆盖率；
- pillar occupancy pattern；
- frozen detector 的局部前景/logit/proposal 响应；
- 与 `N_eff` 做条件比较。

重点回答：

> **同样数量的可靠点，为什么有的目标够用、有的不够用？**

---

## H-A3：“未观测”与“可信为空”的关键区别来自 visibility / free-space 几何

### 假设内容

Q-A 最大的困难来自零回波歧义。

一个区域没有点时，仅靠点统计永远无法知道它是：

- 没被看到；
- 被遮挡；
- 被天气衰减；
- 还是射线确实经过该区域并在更远处获得可靠回波，因此这里有较强 free-space 证据。

如果这个假设成立，真正重要的不是“这里有没有点”，而是：

> **传感器是否有能力、是否有射线证据表明这个空间真的被观测过。**

### 如果它是真的，应该观察到什么

1. “充分观测且为空”的区域应比“未观测/证据不足”的区域拥有更强的 ray traversal / free-space support。
2. weather-induced miss 区域更容易出现射线提前终止、有效远端回波减少或局部可见性断裂。
3. 在 point count≈0 的区域中，visibility 几何仍能把部分“可信空闲”和“未知”分开。
4. 加入 visibility 后，对 Q-A 的区分能力应明显超过单纯 density/reliability。

### 什么结果会显著削弱/否定它

如果在严格控制距离、角度和遮挡后：

- free-space/ray 指标在“空闲”和“未观测”之间没有稳定差异；
- 或 weather 退化并不改变相关 visibility 统计；

则 ray geometry 不是主要答案。

### 最便宜验证

先做离线 ray/visibility 统计，不训练模型。

用 GT/paired clean-weather 只作为**分析标签**，比较：

- 空背景区域；
- clean 有目标而 weather ego 丢失的区域；
- weather 仍稳定检测的目标区域。

重点看 ray traversal / endpoint / occlusion / distance 分布。

---

## H-A4：真正的“需要帮助”在很多情况下只能通过跨车互补证据识别

### 假设内容

对于当前帧的单车来说，一块完全没有回波的区域存在不可消除的局部歧义。

如果 ego 看不到一个目标，它单靠自己可能无法知道那里是否应该有东西。

因此 observation insufficiency 并不总是一个纯 local property，而可能是：

> **ego evidence 与其它视角 evidence 的相对关系。**

### 如果它是真的，应该观察到什么

1. ego ambiguous / low-evidence 区域中，如果邻车在同一空间拥有稳定目标证据，该区域属于真实“需要帮助”的概率应显著升高。
2. ego-local 特征难以区分的一批样本，在加入跨车一致性/分歧后应明显可分。
3. weather-induced ego miss 会富集 `ego weak + neighbor strong` 的模式。
4. 单纯“不同车辆不一致”还不够；真正有用的应是**空间对齐且邻车自身证据可靠**的不一致。

### 什么结果会显著削弱/否定它

如果：

- local-only 指标已经能很好区分 Q-A；
- 邻车信息加入后几乎没有增量；
- cross-agent disagreement 大部分来自 pose error、噪声或普通视角差异，而不是 ego evidence loss；

则跨车互补不是 Q-A 的主要识别条件。

### 最便宜验证

不训练新模块。

在已有 paired / multi-agent 数据上建立 target-level contingency table：

- ego strong / neighbor strong；
- ego weak / neighbor strong；
- ego strong / neighbor weak；
- ego weak / neighbor weak。

再看每类中 weather-induced ego miss、GT target、正常 background 的比例。

---

## H-A5：一部分所谓“观测不足”其实是 representation / detector failure

### 假设内容

模型漏检并不一定意味着 LiDAR 没看到。

可能存在一批样本：

- ego 仍有足够可靠车体点；
- 局部几何覆盖没有明显丢失；
- 但经过 PillarVFE / backbone / head 后，目标仍然没有被检测出来。

如果这一类比例很高，那么“需求图”不能简单把所有 miss 都解释成 sensing deficiency。

### 如果它是真的，应该观察到什么

1. 部分 weather miss 的 target-region point support 与正常 TP 接近。
2. 这些样本在较早层仍存在目标结构，但在后续 feature/logit 中逐渐消失。
3. 单纯增加本地观测量不能解释这些 miss。
4. 对这类样本，即使 oracle 告诉你“这里观测不足”，向邻车请求更多同类证据也未必能恢复检测。

### 什么结果会显著削弱/否定它

如果绝大多数 weather-induced miss 都伴随清晰、显著、可重复的物理 evidence loss，并且 evidence restoration 能稳定恢复目标，那么 downstream failure 不是主要解释。

### 最便宜验证

从已有 miss 中抽小样本，不训练：

对比 TP vs weather miss 的：

- GT 框内原始点；
- GSPR 后有效点；
- pillar occupancy；
- backbone 局部 feature norm / response；
- detection logits。

目的是定位：

> **证据到底从哪一层开始消失？**

---

## H-A6：恶劣天气下不存在单一“观测不足”，而是多种状态混合

### 假设内容

Fog/Rain/Snow 对 LiDAR 的影响并不一定共享同一种机制。

一个标量 `need score` 可能把下面不同状态混在一起：

- **missing evidence**：有效目标回波减少；
- **corrupted evidence**：有回波但受污染；
- **spurious evidence**：天气制造额外假回波；
- **occluded / geometrically unobservable**：本来就不可见；
- **observed free**：真正观测过且为空。

如果这个假设成立，Q-A 最终需要的可能不是一个简单二值“需要/不需要”，而是先建立 observation state。

### 如果它是真的，应该观察到什么

1. Fog/Rain/Snow 的 evidence-loss signature 不同。
2. 同一个 `N_eff` 或 uncertainty 水平，在不同天气下对应不同的检测结果。
3. 某个单一 scalar 在一种天气有效，在其它天气明显失效。
4. 把样本按 missing / corrupted / spurious / free 等状态拆分后，很多此前矛盾的实验结果会变得更一致。

### 什么结果会显著削弱/否定它

如果在控制距离、遮挡和目标属性后，一个统一的 observation-sufficiency scalar 能在 Clean/Fog/Rain/Snow 上保持稳定关系，那么没有必要引入多状态解释。

### 最便宜验证

不用先训练分类器。

利用 paired clean/weather 数据做离线 taxonomy：

- clean target support → weather support 的变化；
- 新增点与丢失点；
- reliability/u；
- ray visibility；
- detection state。

观察不同天气的 failure composition 是否显著不同。

---

# 4. 六个假设之间真正竞争的是什么

| 假设 | 它认为最关键的信息是什么 | 如果成立，后续研究应主要看什么 |
|---|---|---|
| H-A1 | 可靠证据数量 | `N_eff` / density / distance-conditioned support |
| H-A2 | 目标相关语义结构 | target-supporting feature / proposal / spatial pattern |
| H-A3 | 是否真正被传感器观测 | visibility / ray / free-space evidence |
| H-A4 | 多视角互补关系 | ego–neighbor disagreement / complementary evidence |
| H-A5 | sensing 之外的模型失效 | feature propagation / detector decoding |
| H-A6 | 多种 observation state | missing / corrupt / spurious / free 的状态分解 |

注意：

最终真实机制可能包含多个因素，但**现在不能一开始就把六个全部拼成一个网络**。

第一阶段的目的，是先判断：

> 哪些因素拥有独立解释力，哪些只是相关但不必要，哪些已经可以排除。

---

# 5. 最低成本证伪实验

## E-A0：数据与配对审计

**目的**：先确认 clean/weather、scene、CAV、frame、pose、GT 的配对关系，避免伪现象。

**成本**：极低。

**不训练。**

如果基础配对不成立，后续所有 paired observation analysis 暂停。

---

## E-A1：Target-level evidence table

为每个 GT target 建立一行，不训练网络。

至少记录：

- weather；
- scene/frame/CAV；
- distance；
- target size；
- raw point count；
- reliable point count；
- `N_eff`；
- uncertainty；
- spatial coverage；
- ego detection state；
- clean paired detection/support；
- neighbor detection/support（若可得）。

**主要区分**：H-A1、H-A2、H-A4、H-A5、H-A6。

这是当前信息增益最高的实验。

---

## E-A2：Matched-pair 检验——同样 `N_eff`，结果为何不同？

寻找 `N_eff`、distance、target size 相近，但 detection outcome 不同的样本对。

比较：

- 车体点空间分布；
- pillar coverage；
- semantic response；
- neighbor evidence。

**主要区分**：H-A1 vs H-A2/H-A4/H-A5。

如果 H-A1 已经足够解释，不应急着引入复杂语义需求图。

---

## E-A3：零/低回波区域的 visibility/free-space 检验

只分析最有歧义的区域：

- point count≈0；或
- `N_eff` 很低。

加入 ray traversal / endpoint / occlusion 统计，测试能否区分：

- observed free；
- unknown / insufficient；
- weather-induced target loss。

**主要检验**：H-A3。

---

## E-A4：跨车 witness 检验

在 ego local evidence 相似的情况下，看邻车强证据是否显著提高“ego 确实缺证据”的后验概率。

**主要检验**：H-A4。

必须控制：

- pose alignment；
- distance；
- neighbor 自身证据质量。

禁止把普通 disagreement 直接解释为 ego failure。

---

## E-A5：Evidence-loss vs detector-failure 分层审计

对 weather ego miss 做小样本层级追踪：

`raw points → GSPR → pillar → backbone → head`

找“证据从哪里开始丢”。

**主要检验**：H-A5。

如果很多 miss 在 raw/GSPR 层并没有明显 evidence loss，则 Q-A 不能用“需求图”包办全部失败。

---

## E-A6：跨天气 failure taxonomy

基于 E-A1～E-A5 的字段，对 Fog/Rain/Snow 分别统计：

- missing；
- corrupted；
- spurious；
- observed-free；
- downstream-failure。

**主要检验**：H-A6。

先看组成是否真的不同，不训练天气分类器。

---

# 6. 实验优先级

| 优先级 | 实验 | 是否训练 | 成本 | 一次能区分多少解释 | 当前建议 |
|---|---|---:|---|---|---|
| A1 | E-A0 配对/协议审计 | 否 | 极低 | 排除全部伪现象 | 先做 |
| A2 | E-A1 target-level evidence table | 否 | 低 | H-A1/A2/A4/A5/A6 | **最高优先级** |
| A3 | E-A2 matched-pair | 否 | 低 | H-A1 vs A2/A4/A5 | 紧接 E-A1 |
| A4 | E-A5 evidence-loss 层级审计 | 否 | 低–中 | H-A5 与其它 sensing 假设 | 很重要 |
| B1 | E-A3 visibility/free-space | 否 | 中 | H-A3 | A2 后做 |
| B2 | E-A4 cross-agent witness | 否 | 中 | H-A4 | A2 后做 |
| B3 | E-A6 weather taxonomy | 否 | 低 | H-A6 | 汇总前面结果 |

当前阶段**没有任何需要重新训练完整模型的 Priority A 实验**。

---

# 7. 当前 Kill Criteria

为了避免再次陷入“不断加模块”，提前规定以下停止条件。

### Kill H-A1

如果 `N_eff` 在控制距离/目标大小后仍无法稳定区分充分/不足，并且大量高 `N_eff` miss 存在，则停止把 weighted density 当主线。

### Kill H-A2

如果语义/空间结构字段在 matched `N_eff` 条件下没有增量解释力，则不继续设计 semantic demand head。

### Kill H-A3

如果 ray/free-space 指标不能在低/零点区域稳定区分 observed-free 与 unknown，则不把 visibility geometry 作为核心方案。

### Kill H-A4

如果可靠 neighbor evidence 对 local ambiguous case 没有明显增量，或者 disagreement 主要由 alignment/noise 造成，则不继续走 cross-agent demand inference。

### Kill H-A5

如果绝大多数 weather miss 在最前端已经表现为明确 evidence loss，则把 downstream failure 降为次要问题。

### Kill H-A6

如果一个统一 scalar 在三种天气下关系稳定，则不为了“天气特异性”额外增加状态复杂度。

---

# 8. 本阶段结束时必须得到什么

完成第一阶段后，不能只得到一句：

> “我们应该结合 density + reliability + proposal + disagreement。”

这种结论没有价值，因为它只是把所有信号堆在一起。

我们真正需要得到的是：

1. **什么现象是真的；**
2. **哪几个竞争解释已经被排除；**
3. **哪一个或两个因素拥有独立、稳定的解释力；**
4. **“观测不足”应该被定义成什么，而不应该被定义成什么；**
5. **哪些样本属于 sensing insufficiency，哪些其实属于 downstream failure；**
6. **只有在这些问题明确后，才进入 `00_research_workflow.md` 的第 2、3、4 阶段。**

---

# 9. 当前一句话研究目标

> **先证明“观测不足”到底是什么，再研究怎么预测它；先证明怎么预测它，再研究如何请求邻车和如何融合。**

在完成上述证伪实验之前，不新增“观测不足识别网络”。
