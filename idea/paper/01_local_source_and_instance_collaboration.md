# 局部区域、来源选择与实例级协同：论文精读地图

> 更新日期：2026-09-20  
> 目的：回答“局部区域/实例/来源选择已经被做到什么程度，本项目还能在哪里创新？”

---

## 0. 结论

这一支文献已经非常拥挤。

目前至少有四种成熟思路：

1. **空间区域选择**：Where2comm、CodeFilling、CoSDH；
2. **整车/协作者选择**：Select2Col；
3. **实例/query 级协作**：QUEST、CoopDETR、INSTINCT；
4. **proposal/专家级动态路由**：ICPB，以及 2026 年进一步出现的多种 MoE collaborative fusion。

因此本项目不能再把以下内容单独包装成创新：

- “只在局部区域融合”；
- “只选择有贡献的 peer”；
- “根据质量决定是否协作”；
- “用 query 实现实例级协作”；
- “proposal 级 mixture-of-experts”；
- “中间融合 + 后期融合 hybrid”。

本项目更有价值的区别应是：

> **不是泛化地回答‘哪里/跟谁协作’，而是回答‘full collaboration 已经对某个局部目标产生了什么类型的错误，以及应该保持 full、修分数、修几何，还是修特征’。**

---

# 1. Where2comm — 空间选择的经典基线

**论文**  
[Where2comm: Communication-Efficient Collaborative Perception via Spatial Confidence Maps](https://proceedings.neurips.cc/paper_files/paper/2022/hash/1f5c5cd01b864d53cc5fa0a3472e152e-Abstract-Conference.html)  
NeurIPS 2022，DOI: 10.52202/068431-0352。

## 核心问题

通信带宽有限，哪些空间位置值得发送？

## 核心信号

Spatial Confidence Map。

直观地说：

> 每辆车先预测“哪些位置更可能包含重要感知信息”，然后只发送关键区域。

## 与本项目的重叠

高。

本项目早期很多“需求图 / confidence / reliability / block selection”思路都和 Where2comm 同一大类：

```text
局部质量/置信度
→
决定哪些区域需要通信或融合
```

## 与 Stage-3C 的关键区别

Stage-3C 的问题已经不是：

> “这个区域值不值得传？”

而是：

> “信息已经传过来了，full fusion 之后这个目标反而坏了；具体应该怎么修？”

所以后续不要继续围绕“更好的 Where2comm mask”打转。

---

# 2. CodeFilling — 信息需求与多来源冗余

**论文**  
[Communication-Efficient Collaborative Perception via Information Filling with Codebook](https://openaccess.thecvf.com/content/CVPR2024/html/Hu_Communication-Efficient_Collaborative_Perception_via_Information_Filling_with_Codebook_CVPR_2024_paper.html)  
CVPR 2024。

## 核心思想

它不只是问“哪里有高 confidence”，而是问：

> ego 当前缺什么信息？不同 collaborator 怎样一起填补这个 demand，同时避免重复发送？

另外，它使用 codebook 压缩消息表达。

## 对本项目的意义

它进一步压缩了“供需关系”这一创新空间。

以后如果我们说：

> “ego 在这个区域观测不足，所以请求邻车补充。”

这已经不够新。

## 仍未解决的部分

CodeFilling 的目标仍主要是：

- message selection；
- communication efficiency；
- 避免信息 overflow。

它并不针对：

- full fusion 后高分错位框压制正确框；
- score source 与 geometry source 不一致；
- 应该保持 full 还是进行局部修复；
- 修复动作本身带来的 FP / TP collateral damage。

---

# 3. CoSDH — Supply-Demand + Hybrid Fusion

**论文**  
[CoSDH: Communication-Efficient Collaborative Perception via Supply-Demand Awareness and Intermediate-Late Hybridization](https://openaccess.thecvf.com/content/CVPR2025/html/Xu_CoSDH_Communication-Efficient_Collaborative_Perception_via_Supply-Demand_Awareness_and_Intermediate-Late_Hybridization_CVPR_2025_paper.html)  
CVPR 2025。

## 核心思想

两点：

1. 建模 agent 间的 supply-demand，细化 collaboration region；
2. 中间融合性能不足时，用 late collaboration 进行补偿。

## 为什么必须重点读

它同时占了两个我们很容易想到的方向：

- “需求区域”；
- “feature fusion 不稳时加 late detection fallback”。

因此如果我们做：

```text
GSPR需求图
+
intermediate fusion
+
late fusion fallback
```

创新上非常危险。

## 本项目还能怎么区别

Stage-3C 提供了一个更细的现象：

> 同一个目标，full score、peer score、full geometry、peer geometry 可能各自有不同价值。

因此我们不应只做：

```text
intermediate vs late
```

而应该更细：

```text
KEEP full output
vs
use peer score evidence
vs
use peer geometry evidence
vs
local feature repair
```

也就是说，**不是选择 fusion paradigm，而是选择 repair action**。

---

# 4. Select2Col — “有贡献/有害 collaborator”已经有人做

**论文**  
[Select2Col: Leveraging Spatial-Temporal Importance of Semantic Information for Efficient Collaborative Perception](https://ieeexplore.ieee.org/document/10504998/)  
IEEE Transactions on Vehicular Technology 2024。  
DOI: 10.1109/TVT.2024.3390414。

## 核心思想

用轻量图神经网络估计每个 collaborator 的语义信息重要性，选择有贡献者，排除潜在带来负面影响的 agent。

还通过 historical prior hybrid attention 做多尺度/短时融合。

## 对本项目最大的警告

以下表述不再新：

> “不是所有邻车都有用，所以我们学习选择有用邻车、过滤有害邻车。”

项目自己的 Agent Selector v1 也已经证明：

- 在 AWA Locked 有收益；
- 但 OPV2V-W 跨域失败。

内部证据和外部文献都说明：

> **agent-level utility 不是最合适的新主线。**

## Stage-3C 为什么支持更细粒度

同一个 peer：

- 在一个目标附近可能有帮助；
- 在另一个目标附近可能造成 FP；
- 对同一目标的 score 和 geometry 价值还可能不同。

因此 collaborator 不应被简单贴：

```text
good / bad
```

标签。

---

# 5. SCOPE — 多尺度关键特征 + adaptive fusion

**论文**  
[Spatio-Temporal Domain Awareness for Multi-Agent Collaborative Perception](https://openaccess.thecvf.com/content/ICCV2023/html/Yang_Spatio-Temporal_Domain_Awareness_for_Multi-Agent_Collaborative_Perception_ICCV_2023_paper.html)  
ICCV 2023，DOI: 10.1109/ICCV51070.2023.02137。

## 核心内容

- 时间上下文增强；
- 多尺度关键空间信息；
- 多来源 complementary contribution adaptive fusion；
- 对 localization error 进行一定处理。

## 与本项目关系

它说明：

> “多尺度 + adaptive fusion + source contribution”本身不是新颖组合。

Stage-3C 中 scale 0、1、0+1 的干预结果可以作为方法设计证据，但不能简单推出：

> “所以我要做一个 multiscale attention。”

真正值得研究的是：

> 为什么某些目标必须 0+1 联合修改才恢复，以及这种必要性如何在**局部 action routing**中被识别。

---

# 6. S-AdaFusion — spatial-wise adaptive fusion 已经很早存在

**论文**  
[Adaptive Feature Fusion for Cooperative Perception Using LiDAR Point Clouds](https://openaccess.thecvf.com/content/WACV2023/html/Qiao_Adaptive_Feature_Fusion_for_Cooperative_Perception_Using_LiDAR_Point_Clouds_WACV_2023_paper.html)  
WACV 2023。

## 核心思想

通过 trainable feature selection / spatial-wise adaptive fusion 融合多车 feature。

## 对本项目的边界

以后不能用：

> “传统融合对所有位置处理一样，因此我们加入空间自适应权重。”

作为核心创新。

这是很成熟的命题。

---

# 7. QUEST — query cooperation 与实例级“融合/补全”分流

**论文**  
[QUEST: Query Stream for Practical Cooperative Perception](https://arxiv.org/abs/2308.01804)  
ICRA 2024。

## 核心思想

让 query 在 agent 间流动，实现 instance-level feature interaction。

特别值得注意的是：

- 对双方都感知到的实例做 fusion；
- 对单边没有看到的实例做 complementation。

## 为什么与本项目很接近

它实际上已经在做：

> 根据实例状态选择不同协作行为。

所以本项目如果只是：

> “先检测实例，再决定这个实例需要 fusion 还是 complement。”

会很危险。

## 本项目更细的空间

Stage-3C 不是简单：

```text
seen / unseen
```

而是：

```text
seen but score bad
seen but geometry bad
seen but ranking/NMS bad
feature fusion locally destructive
full is already best and should remain unchanged
```

所以动作语义更偏“repair state”，而不是“awareness state”。

---

# 8. CoopDETR — object query 级合作已经形成完整框架

**论文**  
[CoopDETR: A Unified Cooperative Perception Framework for 3D Detection via Object Query](https://arxiv.org/abs/2502.19313)  
ICRA 2025。

## 核心思想

- 单车把感知编码成 object query；
- Spatial Query Matching；
- Object Query Aggregation；
- 通过 object-level cooperation 降低通信。

## 边界

如果后续方法需要 proposal/object query：

> object query 只是表示形式，不能是创新本身。

更值得借的是：

- 如何跨 agent 匹配同一目标；
- 如何构建 source-specific proposal cluster。

这正好可以服务我们的：

```text
full / ego / peer proposals
→
match same candidate
→
比较 score / geometry / feature quality
→
route repair action
```

---

# 9. INSTINCT — 当前最需要防撞的一篇

**论文**  
[INSTINCT: Instance-Level Interaction Architecture for Query-Based Collaborative Perception](https://openaccess.thecvf.com/content/ICCV2025/html/Xu_INSTINCT_Instance-Level_Interaction_Architecture_for_Query-Based_Collaborative_Perception_ICCV_2025_paper.html)  
ICCV 2025。  
DOI: 10.1109/ICCV51701.2025.02362。

## 三个核心模块

1. quality-aware filtering；
2. dual-branch detection routing；
3. Cross Agent Local Instance Fusion。

## 为什么危险

这几个关键词几乎和我们最自然的第一反应重叠：

- quality-aware；
- routing；
- local；
- instance；
- cross-agent fusion。

因此后续方法绝不能把创新点写成：

> “质量感知实例级局部协同路由”。

这基本会正撞 INSTINCT。

## 真正可区分的地方

我们的方法要明确回答：

### INSTINCT 的 routing 在区分什么？

大体是 collaboration-irrelevant / collaboration-relevant instance。

### 我们应该区分什么？

```text
当前 full 是否应该保留？
如果不保留：
  score evidence 应该从哪来？
  geometry evidence 应该从哪来？
  是否只改输出就够？
  是否必须进入 feature repair？
```

核心差异应该是：

> **repair action routing，而非 collaboration routing。**

---

# 10. ICPB — proposal-wise MoE 已经被直接占用

**论文**  
[Perception balance with uncertainty-guided fusion and proposal-wise mixture-of-experts for robust multi-agent 3D object detection](https://www.sciencedirect.com/science/article/abs/pii/S0957417426002241)  
Expert Systems with Applications 2026。  
DOI: 10.1016/j.eswa.2026.131311。

## 核心模块

- feature-level uncertainty；
- proposal-level uncertainty；
- Hyper-dimensional Uncertainty-aware Self-Attention Fusion；
- Proposal-wise Uncertainty-aware Mixture-of-Experts；
- individual / collaborative teacher distillation。

## 直接结论

以下思路不能再单独声称新：

> “每个 proposal 根据 uncertainty 进入不同 expert，从而在 ego 和 collaboration 之间动态平衡。”

## 对本项目仍有用的启发

ICPB 可以作为一个非常好的“结构参考”，但我们的科学监督必须不同。

建议比较：

| 维度 | ICPB | 本项目候选 |
|---|---|---|
| 问题 | independent vs collaborative capability balance | weather collaboration 后局部输出被破坏 |
| 决策单位 | proposal | proposal/local target |
| 主要信号 | epistemic uncertainty | source-specific score/geometry disagreement + local feature evidence |
| expert/action | perception experts | keep / output repair / feature repair |
| 核心监督 | distillation + uncertainty | counterfactual recovery + collateral harm + detection quality |
| 天气机制 | 非核心 | 核心场景 |

因此：

> 可以借 MoE 作为实现工具，但不能把 MoE 作为论文科学贡献。

---

# 11. CoRA — feature fusion + object correction 已经有人组合

**论文**  
[CoRA: A Collaborative Robust Architecture with Hybrid Fusion for Efficient Perception](https://ojs.aaai.org/index.php/AAAI/article/view/37274)  
AAAI 2026，DOI: 10.1609/aaai.v40i4.37274。

## 核心结构

- feature-level fusion branch；
- object-level correction branch；
- object branch 利用语义关系修正 spatial displacement；
- 主要针对 pose error 等 robustness。

## 对我们有什么影响

如果我们提出：

> “我在 feature fusion 后再加一个 object-level correction branch。”

本身已经不新。

## 我们应该强调什么

不是：

```text
feature + object 两级融合
```

而是：

```text
检测到局部 collaborative failure
→
先判断失败类型/动作
→
按需进入 output repair 或 feature repair
```

CoRA 更像固定双分支结构；我们希望是**条件化 repair policy**。

---

# 12. Uncertainty-guided reliable CP 2026 — blind-region active query + local proposal fallback

**论文**  
[Uncertainty-guided and reliable collaborative perception for open heterogeneous systems](https://www.sciencedirect.com/science/article/pii/S0167865526001297)  
Pattern Recognition Letters 2026。  
DOI: 10.1016/j.patrec.2026.04.011。

## 核心方法

- Sparsity-Aware Active Query：ego 请求 blind region；
- Foreground-Aware Transmission：发送端过滤背景；
- Dual-stream Multi-level Fusion：中间协同 fusion 与单车 detection 并行；
- local proposals 提供稀疏通信下的可靠 fallback。

## 创新边界

它已经把：

```text
盲区需求
+
发送端前景筛选
+
local detection fallback
```

做成一套。

所以本项目不能再回到：

> “用 GSPR 找 blind region，然后 peer 发 foreground，最后和 ego late fusion。”

这非常接近已有工作。

---

# 13. UECP — 点密度/物理 uncertainty → fusion 的路线已经有人直接做

**论文**  
[UECP: Uncertainty-Enhanced Collaborative Perception](https://arxiv.org/abs/2606.23046)  
ECCV 2026 / arXiv 2026。

## 核心主张

作者认为 confidence map 与 detection head 耦合，不能提供独立物理证据。

因此：

- 用 LiDAR point density 直接监督 uncertainty map；
- uncertainty map 指导 pyramid fusion；
- 通过 uncertainty-weighted downsampling 和 uncertainty-guided residual fusion 强化可靠信息。

## 与本项目内部结果的关系

这篇非常适合放进 Related Work，也非常适合作为对照观点。

它的核心假设大致是：

> 物理 sensing uncertainty 可以指导 source weighting。

而本项目内部已经看到：

- density/reliability 与真实 task marginal utility 相关性有限；
- low-r/high-u 直接删/降权并不稳定；
- Snow 中 proxy 与 task truth 偏差明显；
- Stage-3C 显示 score/geometry/fusion action 不可被一个 scalar 质量完整表达。

因此我们可以形成一个很有价值的研究边界：

> **physical sensing uncertainty 有用，但不足以决定 task-specific collaborative repair。**

注意：这不是说 UECP 错，而是两者解决层级不同。

---

# 14. Among Us / ROBOSAC — “有害来源”与 subset counterfactual 的参考

**论文**  
[Among Us: Adversarially Robust Collaborative Perception by Consensus](https://openaccess.thecvf.com/content/ICCV2023/papers/Li_Among_Us_Adversarially_Robust_Collaborative_Perception_by_Consensus_ICCV_2023_paper.pdf)  
ICCV 2023。

## 核心思想

随机采样 collaborator subset，比对 collaboration 与 individual perception，寻找 consensus，从而避开恶意 agent。

## 为什么值得借鉴

它证明一种思路：

> 通过“有/无某些来源”的反事实结果来判断来源是否可信。

这与 Stage-3B 的 subset intervention 在思想上类似。

## 为什么不能直接搬

- ROBOSAC 面向 adversarial attackers；
- subset 粒度是 agent coalition；
- 需要多次 sampling/forward；
- 本项目 Stage-3C 已证明 whole-agent/whole-frame 修改太粗。

因此它更适合引用为：

> counterfactual source validation 的相关思想，

而不是最终实现。

---

# 15. COOPERTRIM — relevance / quantity 都已经有人做，且强调时间

**论文**  
[COOPERTRIM: Adaptive Data Selection for Uncertainty-Aware Cooperative Perception](https://proceedings.iclr.cc/paper_files/paper/2026/hash/c99e08e921b90e901e5eaa7ddee51d6c-Abstract-Conference.html)  
ICLR 2026。

## 核心问题

- 哪些 feature relevant？
- 应该发送多少？

## 核心信号

Conformal temporal uncertainty。

## 对本项目的边界

如果未来只是：

> “根据 uncertainty 动态选择信息量和特征区域。”

已经很难有空间。

用户当前不想优先引入 temporal branch，这也使其与本项目的核心路线有天然区别。

---

# 16. 方法查重矩阵

| 想法 | 已有工作 | 当前判断 |
|---|---|---|
| 空间置信图选区域 | Where2comm | 高度占用 |
| ego demand / peer supply | CodeFilling、CoSDH | 高度占用 |
| 选择有贡献 agent | Select2Col | 高度占用 |
| 实例/query 级协作 | QUEST、CoopDETR、INSTINCT | 高度占用 |
| quality-aware instance routing | INSTINCT | 极高撞车风险 |
| proposal uncertainty + MoE | ICPB | 极高撞车风险 |
| 中间+后期 hybrid | CoSDH、CoRA、Uncertainty-guided CP | 高度占用 |
| feature fusion + object correction | CoRA | 高度占用 |
| uncertainty map 指导 fusion | UECP、ICPB | 高度占用 |
| blind region active query | Uncertainty-guided CP | 已有 |
| subset consensus 去掉坏 agent | ROBOSAC | 已有，但任务不同 |
| temporal uncertainty feature selection | COOPERTRIM | 已有 |
| **source-specific score vs geometry repair** | 未发现直接等价方案 | 值得保留 |
| **keep/full vs score repair vs geometry repair vs feature repair 的动作建模** | 未发现直接等价方案；但与 INSTINCT/ICPB/CoRA 相邻 | 高价值，但必须精确写差异 |
| **counterfactual recovery + collateral harm 监督 repair action** | 与项目旧 counterfactual selector 相邻，外部未发现直接同构天气 CP 方法 | 有潜力，但必须解决旧 selector 标签错位问题 |
| **天气下 collaborative score-localization source mismatch** | 单车 quality alignment 很成熟，CP 中未发现直接等价主线 | 当前较好的科学问题 |

---

# 17. 对最终方法的硬约束

未来提出方法时，至少满足以下六条，否则大概率只是旧工作换名：

1. **粒度必须是 target/proposal/local region，而不是 agent-level。**
2. **必须把 KEEP_FULL 作为显式动作。**
3. **score quality 与 geometry quality 分开估计。**
4. **不能把 point density / GSPR reliability 当 task utility 真值。**
5. **监督必须同时考虑恢复和 collateral harm。**
6. **feature repair 只在 output repair 不足时触发，避免重新做全局 adaptive fusion。**

---

# 18. 精读时需要抽取的字段

后续读每篇论文，不要只记录“用了什么模块”，而要填下面这张表：

| 字段 | 要回答的问题 |
|---|---|
| Problem | 它真正解决的失败是什么？ |
| Unit | agent / region / instance / proposal / point？ |
| Observable signal | confidence / uncertainty / feature / query / history？ |
| Decision | transmit / select agent / fuse / route / correct？ |
| Supervision | detection loss / GT / distillation / uncertainty label？ |
| Safe fallback | 有没有保留 ego/full 的机制？ |
| Negative evidence | 怎样避免坏 peer/坏动作？ |
| Localization quality | 是否显式建模？ |
| Collateral harm | 是否约束 FP / 原 TP 损失？ |
| Weather | 是否针对天气？ |
| Counterfactual | 是否使用干预/有无某来源比较？ |
| Difference from us | 和 Stage-3C 的差异到底在哪？ |

这张表比继续收集更多“名字相似”的论文更重要。
