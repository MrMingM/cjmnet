# Stage-3C 后的定向文献地图与创新边界

> 更新日期：2026-09-20  
> 研究主线：恶劣天气下协同感知鲁棒性  
> 当前阶段：大规模诊断阶段基本收束，转入“定向文献 → 方法设计 → Oracle/训练标签设计 → Cheap Gate”  
> 本文件与 `idea/03_stage3c_results_and_conclusions.md`、`idea/02_hypothesis_graveyard.md` 配套使用。

---

## 0. 先给结论

现有证据已经足够支持停止继续横向扩展 Stage-3 诊断。

原因不是“唯一根因已经查明”，而是：

1. **局部修复比整图修复更符合当前问题。** Stage-3C 中 64 个焦点目标有 41 个可恢复；局部与全局对焦点目标的恢复集合一致，但“零原 TP 损失 + 零新增 FP”的机会从全局 8 个提高到局部 41 个。
2. **分数和定位不能分开看。** Fog/Rain 大量 NMS 失败表现为高分但位置不够准的竞争框；局部几何修复后，原抑制框可以变成正确框。Snow 又存在“几何仍可用但分数显著下降”的大量情况。
3. **特征/融合权重不能完全丢掉。** Snow 存在分数、几何单独替换都救不回、但权重干预可以恢复的目标。
4. **当前真正缺的是“无 GT 时怎么选动作”，而不是继续证明‘存在可恢复目标’。**
5. 文献已经把“区域选择、实例级协作、质量感知门控、proposal 级专家、IoU-aware 打分、混合中间/后期融合”等通用想法做得很密。后续创新必须建立在本项目已经观察到的**局部分数—几何—融合三类失效与可恢复动作**上，而不是重新包装这些已有模块。

因此接下来最值得研究的问题是：

> **对一个局部目标候选，模型能否在不依赖 GT 的情况下判断：保持原 full 结果，还是从某个 peer 修正分数、修正几何、联合修正输出，或者进入更深的局部特征修复？**

这个问题比“哪个 peer 好”“哪个区域置信度低”“哪个块 uncertainty 高”更具体，也更贴合 Stage-3C 的实验证据。

---

# 1. 本轮文献检索范围

本轮检索围绕四类问题进行，而不是泛搜“协同感知”：

1. **局部区域 / 实例 / proposal 级协作选择与路由**
2. **多来源有害信息、可靠性、不确定性与鲁棒融合**
3. **分类分数与定位质量不一致、IoU-aware score、NMS 竞争**
4. **恶劣天气下协同感知、天气去噪、域泛化、多模态天气鲁棒性**

检索来源包括：

- Consensus 学术检索；
- CVF Open Access（CVPR / ICCV / WACV）；
- NeurIPS / ICLR proceedings；
- IEEE Xplore；
- Elsevier / ScienceDirect；
- arXiv；
- 论文官方代码页。

时间重点覆盖 2020–2026，尤其检查 2025–2026 的新工作。

这不是 PRISMA 意义上的系统综述，但已经覆盖了当前方法设计最容易发生“撞车”的核心路线。

---

# 2. 与本项目最直接相关的论文分层

## 2.1 第一层：直接压缩创新空间，设计方法前必须精读

| 论文 | 年份/会议 | 核心做法 | 与本项目最直接的重叠 | 仍然留下的空间 |
|---|---|---|---|---|
| [INSTINCT: Instance-Level Interaction Architecture for Query-Based Collaborative Perception](https://openaccess.thecvf.com/content/ICCV2025/html/Xu_INSTINCT_Instance-Level_Interaction_Architecture_for_Query-Based_Collaborative_Perception_ICCV_2025_paper.html) | ICCV 2025 | 实例级 query；质量感知过滤；双分支 detection routing；局部跨车实例融合 | “实例级”“质量感知”“决定哪些实例需要协作”“局部跨车融合”都已经有人做 | 它没有围绕恶劣天气下**同一候选的分数/几何/融合失配**建立动作空间，也不是用 Stage-3C 式局部反事实恢复/误伤标签训练动作选择 |
| [Perception Balance with Uncertainty-Guided Fusion and Proposal-wise Mixture-of-Experts for Robust Multi-Agent 3D Object Detection](https://www.sciencedirect.com/science/article/abs/pii/S0957417426002241) | Expert Systems with Applications 2026 | feature uncertainty + proposal uncertainty；proposal-wise mixture-of-experts；在单车与协同能力间动态平衡 | “proposal 级门控/专家”“不确定性决定使用哪种感知结果”已被占据 | 其核心问题是独立/协同能力平衡和通信扰动，不是天气引发的 source-conditioned score/geometry repair；因此“MoE”不能当主创新，**动作定义和监督目标**必须不同 |
| [CoSDH: Communication-Efficient Collaborative Perception via Supply-Demand Awareness and Intermediate-Late Hybridization](https://openaccess.thecvf.com/content/CVPR2025/html/Xu_CoSDH_Communication-Efficient_Collaborative_Perception_via_Supply-Demand_Awareness_and_Intermediate-Late_Hybridization_CVPR_2025_paper.html) | CVPR 2025 | 建模供需关系选择协作区域；中间融合 + 后期融合混合 | “哪里需要协作”“区域需求”“中间+后期混合”已经不是空白 | 本项目应避免重新做“需求图+混合融合”；可以研究**协作已经发生后，局部候选为什么被破坏以及如何选择修复动作** |
| [UECP: Uncertainty-Enhanced Collaborative Perception](https://arxiv.org/abs/2606.23046) | ECCV 2026 / arXiv | 用 LiDAR 点密度直接监督 uncertainty map；不确定性引导多尺度协同融合 | “物理可靠性/点密度 → uncertainty → fusion weighting”高度重叠旧 GSPR/密度门控直觉 | 本项目已有实验证据说明 sensing quality 不能直接等同 task utility，因此不应回到“换一个 uncertainty map 再加权”的路线 |
| [Uncertainty-guided and reliable collaborative perception for open heterogeneous systems](https://www.sciencedirect.com/science/article/pii/S0167865526001297) | Pattern Recognition Letters 2026 | ego 主动查询盲区；发送端前景筛选；协同中间特征与单车 proposal 双流融合 | “盲区需求 + foreground filtering + single-agent fallback”已被覆盖 | 本项目仍可研究**局部候选的动作级修复**，尤其不是简单 fallback，而是分数/几何/特征不同修复方式 |
| [CoRA: A Collaborative Robust Architecture with Hybrid Fusion for Efficient Perception](https://ojs.aaai.org/index.php/AAAI/article/view/37274) | AAAI 2026 | feature-level fusion branch + object-level correction branch；对象级分支用于空间位移修正 | “特征融合 + 对象级纠正”这种两层混合思路已有人做 | 本项目不能把“加一个 box correction branch”本身当创新；应把天气下的**分数—几何质量不一致和来源动作选择**作为核心 |
| [CIA-SSD: Confident IoU-Aware Single-Stage Object Detector From Point Cloud](https://ojs.aaai.org/index.php/AAAI/article/view/16470) | AAAI 2021 | IoU-aware confidence rectification；距离相关 IoU-weighted NMS | 直接解决“分类分高但定位不准” | 单纯加 IoU head / 改 NMS 完全不够新；本项目要把质量对齐扩展到**协同来源与天气条件下的局部修复决策** |
| [VarifocalNet: An IoU-Aware Dense Object Detector](https://openaccess.thecvf.com/content/CVPR2021/html/Zhang_VarifocalNet_An_IoU-Aware_Dense_Object_Detector_CVPR_2021_paper.html) | CVPR 2021 | 直接学习 IoU-aware classification score，使候选排序同时体现存在概率与定位质量 | “把分类置信度和定位质量联合进最终排序”已有成熟工作 | 本项目可以借鉴 quality target，但不能把“quality-aware score”当创新点 |

### 这一层给出的最重要边界

以下说法已经不足以成为主创新：

- “我做实例级协同融合。”
- “我做 proposal-wise 门控。”
- “我用 uncertainty 选择单车还是协同。”
- “我判断哪里需要协作。”
- “我做中间融合和后期融合混合。”
- “我预测 IoU 修正分类分数。”
- “我通过质量分数改善 NMS。”
- “我用点密度/可靠性指导融合。”

如果最终方法只是上述某一项换网络名字，创新风险很高。

---

## 2.2 第二层：高度相关，主要用于确定设计边界

### A. 区域/来源选择与通信

1. [Where2comm: Communication-Efficient Collaborative Perception via Spatial Confidence Maps](https://proceedings.neurips.cc/paper_files/paper/2022/hash/1f5c5cd01b864d53cc5fa0a3472e152e-Abstract-Conference.html)  
   NeurIPS 2022。用空间置信图选择“在哪里通信”。  
   **边界：**空间 confidence map 和 sparse region selection 已是经典路线。

2. [CodeFilling: Communication-Efficient Collaborative Perception via Information Filling with Codebook](https://openaccess.thecvf.com/content/CVPR2024/html/Hu_Communication-Efficient_Collaborative_Perception_via_Information_Filling_with_Codebook_CVPR_2024_paper.html)  
   CVPR 2024。通过 information demand 做 message selection，用码本压缩消息。  
   **边界：**“填补 ego 信息需求”不是新的问题表述。

3. [Select2Col: Leveraging Spatial-Temporal Importance of Semantic Information for Efficient Collaborative Perception](https://ieeexplore.ieee.org/document/10504998/)  
   IEEE TVT 2024，DOI: 10.1109/TVT.2024.3390414。用图神经网络估计 collaborator 的 semantic importance，选择贡献者并排除潜在负贡献者。  
   **边界：**整车 collaborator selection 已有系统工作；本项目不要再退回 agent-level good/bad 分类。

4. [SCOPE: Spatio-Temporal Domain Awareness for Multi-Agent Collaborative Perception](https://openaccess.thecvf.com/content/ICCV2023/html/Yang_Spatio-Temporal_Domain_Awareness_for_Multi-Agent_Collaborative_Perception_ICCV_2023_paper.html)  
   ICCV 2023。多尺度关键空间特征、时序语义、多源 adaptive fusion。  
   **边界：**“多尺度 adaptive fusion”本身已经很常见。

5. [How2comm: Communication-Efficient and Collaboration-Pragmatic Multi-Agent Perception](https://proceedings.neurips.cc/paper_files/paper/2023/hash/4f31327e046913c7238d5b671f5d820e-Abstract.html)  
   NeurIPS 2023。互信息感知通信、空间-通道筛选、延迟补偿、协同 Transformer。  
   **边界：**“信息量/互信息决定传什么”并不直接等于本项目要解决的 task-level repair。

6. [COOPERTRIM: Adaptive Data Selection for Uncertainty-Aware Cooperative Perception](https://proceedings.iclr.cc/paper_files/paper/2026/hash/c99e08e921b90e901e5eaa7ddee51d6c-Abstract-Conference.html)  
   ICLR 2026。利用 temporal uncertainty 衡量 feature relevance，并动态决定共享量。  
   **边界：**不确定性驱动的 feature relevance / adaptive quantity 已被占据，而且依赖时序，与本项目当前空间局部修复不是同一问题。

### B. 实例/query 级协作

7. [QUEST: Query Stream for Practical Cooperative Perception](https://arxiv.org/abs/2308.01804)  
   ICRA 2024。query 在 agent 间传递；对共同感知实例做 fusion，对单边未知实例做 complementation。  
   **边界：**“实例级互补”和“co-aware / unaware 分流”已经存在。

8. [CoopDETR: A Unified Cooperative Perception Framework for 3D Detection via Object Query](https://arxiv.org/abs/2502.19313)  
   ICRA 2025。object query 级协作，包含跨车 query matching 与 aggregation。  
   **边界：**“object query collaboration”本身不是创新。

9. INSTINCT（见第一层）进一步把 query-based CP 推到 LiDAR 实例路由与局部实例融合。

因此如果本项目走实例级方案，真正需要强调的是：

> **实例只是干预粒度，不是创新本身；创新必须来自“天气协同失效动作的定义、来源条件化修复和反事实监督”。**

---

## 2.3 第三层：天气鲁棒协同感知的直接背景

1. [Weather-Aware Collaborative Perception With Uncertainty Reduction](https://ieeexplore.ieee.org/document/10738717)  
   IEEE TITS 2024。两阶段 Co-Denoising：单 agent 先做采样式粗去噪，协同阶段利用 Bayesian neural network 缓解天气噪声不确定性。  
   **启示：**天气噪声可能在 collaboration 中放大。  
   **边界：**“天气 + uncertainty + denoising”已有直接工作。

2. [V2X-DGW: Domain Generalization for Multi-Agent Perception Under Adverse Weather Conditions](https://github.com/Baolu1998/V2X-DGW)  
   ICRA 2025，DOI: 10.1109/ICRA55743.2025.11127945。Adaptive Weather Augmentation + weather-invariant alignment + agent-aware contrastive alignment，并发布 OPV2V-W / V2XSet-W。  
   **边界：**“天气增强 + 域泛化 + feature alignment”不是当前最值得重复的路线。

3. [DenoiseCP-Net: Efficient Collective Perception in Adverse Weather via Joint LiDAR-Based 3D Object Detection and Denoising](https://arxiv.org/abs/2507.06976)  
   2025/IV 2026。voxel-level noise filtering 与检测统一，通信前去除天气噪声。  
   **边界：**“先去噪再协同”已经有直接天气 CP 方法。

4. [V2X-R: Cooperative LiDAR-4D Radar Fusion with Denoising Diffusion for 3D Object Detection](https://openaccess.thecvf.com/content/CVPR2025/html/Huang_V2X-R_Cooperative_LiDAR-4D_Radar_Fusion_with_Denoising_Diffusion_for_3D_CVPR_2025_paper.html)  
   CVPR 2025。用 4D radar 作为天气鲁棒条件，引导 diffusion 去噪 LiDAR feature。  
   **边界：**如果不引入 4D radar，本项目不与其直接竞争；但“天气退化 feature enhancement”不是空白。

5. [Hybrid Robust Collaborative Perception with LiDAR-4D Radar Fusion under Adverse Weather Conditions](https://openaccess.thecvf.com/content/CVPR2026/html/Yang_Hybrid_Robust_Collaborative_Perception_with_LiDAR-4D_Radar_Fusion_under_Adverse_CVPR_2026_paper.html)  
   CVPR 2026。跨模态双向 gating 验证可靠性，并增强 degraded/suppressed regions。  
   **边界：**“reliability gating + degraded region enhancement”也已有天气多模态版本。

6. [Adver-City: Open-Source Multi-Modal Dataset for Collaborative Perception Under Adverse Weather Conditions](https://arxiv.org/abs/2410.06380)  
   ITSC 2025。提供多天气、多模态、多视点的协同感知数据。  
   **价值：**如果后续论文需要跨数据集/不同天气验证，它比继续只扩 OPV2V-W 内部诊断更有价值。

---

# 3. “分数—定位不一致”不是新现象，但“协同来源条件下的失配”仍有空间

Stage-3C 的 Fog/Rain 现象非常像经典 detector 的 classification-localization misalignment：

> 一个框分类分数很高，但几何不够准；NMS 按分数排序后，它会压掉位置更准但分数稍低的框。

这类问题在单车 detector 中已经被大量研究：

- 3D IoU-Net：预测 IoU，用定位质量辅助 NMS；
- CIA-SSD：IoU-aware confidence rectification + IoU-weighted NMS；
- AFDetV2：IoU score × classification heatmap；
- VarifocalNet：直接学习兼顾分类与定位的 IoU-aware classification score；
- WACV 2024 IoU-aware calibration：进一步把重复候选概率和 calibration 纳入后处理。

因此不能把论文创新写成：

> “现有模型分类分数与定位质量不匹配，所以我们加一个 IoU head。”

更有价值的问题是：

> **恶劣天气协同后，同一个局部目标的‘谁提供更可信的分类证据、谁提供更可信的几何证据’可能不是同一个来源；full fusion 甚至可能把两者组合得更差。能否显式建模这种 source-conditioned classification–localization quality mismatch？**

这和普通单车 IoU-aware scoring 有本质区别：

- 输入不再只有单个 proposal 自身特征；
- 需要比较 full / ego / peer 的局部证据；
- 需要判断“保持 full”还是从某个来源修复某个属性；
- 修分数可能把失败从 score-filtered 推到 NMS；
- 修几何可能把失败从 decode 问题推到 score-filtered；
- 因此需要**联合动作和连续障碍建模**，不能只优化一个 head。

---

# 4. 当前最有希望的创新交叉点

## 4.1 不建议继续做的“大方向”

### A. 再做一个 reliability / uncertainty map

原因：

- 项目内部已有 low-r / high-u / density / GSPR → task utility 的反例；
- 外部又有 UECP、Weather-Aware CP、COOPERTRIM 等大量 uncertainty-driven 方案。

结论：

> reliability/uncertainty 只适合作为一个辅助输入，不能再承担“决定哪个来源有用”的核心科学假设。

### B. 再做一个 agent selector

原因：

- Select2Col 已经明确选择 contributive collaborator；
- 项目自己的 agent selector 跨域失败；
- Stage-3C 证明同一个 peer 对不同目标、不同属性可能作用不同。

结论：

> agent-level good/bad 粒度太粗。

### C. 再做一个 spatial attention / adaptive fusion

原因：

- Where2comm、S-AdaFusion、SCOPE、How2comm、CoSDH 等已经覆盖得很密；
- 项目内部空间 utility/门控也没有稳定提升。

结论：

> “attention 更聪明”不是足够明确的研究问题。

### D. 单纯加 IoU head / 换 NMS

原因：

- detector 文献已经成熟；
- Stage-3B/3C 还明确看到简单放阈值会导致大量 FP；
- score / geometry / fusion 是连续障碍。

结论：

> IoU/quality 可以作为监督或安全分数，但不能单独成为主方法。

---

## 4.2 建议保留的核心问题

### 问题 P1：局部候选需要哪一种修复动作？

对一个候选目标区域，不再只输出“是否协作”，而是输出：

```text
KEEP_FULL
SCORE_REPAIR(source=j)
GEOMETRY_REPAIR(source=j)
JOINT_OUTPUT_REPAIR(source=j)
FEATURE_REPAIR(source=j, scale={0,1,0+1})
```

第一版不必一次实现全部动作。最现实的版本可以先做：

```text
KEEP_FULL
OUTPUT_REPAIR
FEATURE_REPAIR
```

其中 OUTPUT_REPAIR 内部再预测 score / geometry 的 source-specific correction。

这和已有 proposal-wise MoE 的差别不能写成“我的专家更多”，而应写成：

> **专家对应的是由反事实诊断定义的可解释修复动作，每种动作有明确的 target recovery / collateral harm 训练信号。**

### 问题 P2：谁的分数可信，谁的几何可信？

可以显式估计两个质量：

```text
q_cls(source, proposal)
q_loc(source, proposal)
```

它们不强制来自同一个 agent。

例如：

- peer A 提供更可信 objectness/classification；
- full fusion 提供更好的 box geometry；
- 或反过来；
- 如果两者都不可靠，再进入 feature repair。

这比简单“给 agent 一个 reliability”更符合 Stage-3C。

### 问题 P3：什么时候什么都不要改？

Stage-3C 的 17 个 Snow harmful score replacement case 很重要。

因此动作模型必须把：

```text
KEEP_FULL
```

当成**一等公民**，而不是默认“发现异常就必须修”。

训练目标也不能只有“恢复目标”，还必须处罚：

- 原 TP 丢失；
- 新增 FP；
- 其他目标被破坏；
- 候选排序恶化。

这也是项目历史 learned selector 缺失的重要约束之一。

### 问题 P4：输出层修不动时，何时才值得动融合特征？

Snow 存在 weight-only recoverable cases，因此 feature repair 不能完全删除。

但从工程和创新叙事上，更合理的是：

> **把 feature repair 变成少数难例的第二级动作，而不是对所有区域重新设计一套全局 fusion。**

这样可以自然形成：

```text
Stage 1: proposal/output-level cheap repair
    ↓ if unresolved / low quality
Stage 2: local feature-level repair
```

这也能避开“再造一个全图 attention fusion”的旧路。

---

# 5. 暂定方法雏形：Local Collaborative Repair Routing

> 以下是方法设计候选，不是已经验证的实验结果。暂定工作名，仅用于讨论。

## 5.1 核心输入

对匹配到同一局部候选的 full / ego / peer 信息，构造：

### 检测输出侧

- classification logit / score；
- box center / size / yaw；
- full-peer score difference；
- full-peer geometry disagreement；
- 不同 source 的 proposal rank；
- 预测 localization quality / IoU quality；
- NMS 邻域竞争强度。

### 特征侧

- full 与 peer 的局部 BEV feature difference；
- source-specific local feature norm / cosine similarity；
- scale 0/1 的局部一致性；
- GSPR reliability / uncertainty 作为辅助 sensing 信号。

注意：

> GSPR 不作为“谁有用”的真值，只提供天气 sensing 状态。

## 5.2 Router 输出

输出局部 action probability：

```text
P(keep)
P(score repair by peer j)
P(geometry repair by peer j)
P(joint repair by peer j)
P(feature repair by peer j)
```

第一版为降低复杂度可以合并为：

```text
keep / output-repair / feature-repair
```

然后 output-repair 再预测 source-specific score/geometry residual。

## 5.3 监督来源

训练时可以利用 GT，但正式推理不能使用 GT。

最关键的不是普通 detection loss，而是从已有 Stage-3C 思路扩展出来的**反事实动作标签**：

```text
utility(action)
=
target_recovery_gain
- λ1 * lost_original_TP
- λ2 * new_FP
- λ3 * degradation_of_other_targets
```

然后把“最优动作”作为 router teacher。

这里必须吸取项目旧 selector 的失败：

- 不能用单 block detection-loss gain 直接当 AP utility；
- 不能不建模集合副作用；
- 不能默认 repair 一定优于 keep；
- 不能只看 focal target 是否被救回来。

## 5.4 与现有论文的区别必须这样写

不能写：

> “我们首次提出 instance-level collaborative fusion。”

因为 INSTINCT / QUEST / CoopDETR 已有。

不能写：

> “我们首次用 uncertainty 决定是否 collaboration。”

因为 ICPB / UECP / COOPERTRIM 已有。

不能写：

> “我们首次联合 classification 和 localization quality。”

因为 CIA-SSD / VarifocalNet / AFDetV2 已有。

更可能成立的表述是：

> **我们从恶劣天气下的协同失效反事实分析出发，把局部 collaborative correction 建模为“保持/输出修复/特征修复”的动作选择问题；动作监督同时考虑目标恢复与旁路误伤，并显式解耦来源特定的分类质量和定位质量。**

是否最终能写“首次”，仍需在方法定稿后再进行一次针对性查重，不能现在提前下结论。

---

# 6. 推荐阅读优先级

## 第一优先级：决定创新边界

1. INSTINCT — 看 instance routing 和 local instance fusion 到底做到多细。
2. ICPB — 看 proposal-wise MoE 的输入、专家、监督方式。
3. CoSDH — 看 supply-demand + intermediate-late hybrid 与“需要帮助/保留 fallback”的边界。
4. CoRA — 看 feature fusion + object correction 的分工。
5. UECP — 看点密度 uncertainty map 如何监督和用于融合。
6. CIA-SSD — 看 3D detection 中 score-localization alignment 的标准做法。
7. VarifocalNet — 看 joint quality score 的监督目标。
8. WACV 2024 IoU-aware calibration — 看不用简单 NMS 的候选质量建模。

## 第二优先级：确定实例/区域协作的常见结构

9. QUEST
10. CoopDETR
11. Where2comm
12. CodeFilling
13. Select2Col
14. SCOPE
15. S-AdaFusion
16. Uncertainty-guided reliable CP 2026

## 第三优先级：论文 Related Work 和天气背景

17. Weather-Aware Collaborative Perception With Uncertainty Reduction
18. V2X-DGW
19. DenoiseCP-Net
20. V2X-R
21. HRCP
22. Adver-City
23. COOPERTRIM
24. How2comm
25. Among Us / ROBOSAC

---

# 7. 当前可以得出的结论

## 已经能得出

1. **继续泛化 Stage-3 诊断的边际收益已经下降。**
2. **局部修复是比整图干预更可靠的方法设计起点。**
3. **score、geometry、feature/fusion 不能被压成一个单一 reliability scalar。**
4. **已有文献已经把“区域选择、实例路由、不确定性门控、proposal MoE、IoU-aware score”分别做过，创新必须来自这些元素之间由本项目实验证据支持的新关系。**
5. 当前最值得保留的创新边界是：
   - adverse-weather；
   - target/proposal-local；
   - source-conditioned；
   - classification-quality 与 localization-quality 解耦；
   - keep/repair 动作显式选择；
   - counterfactual recovery + collateral harm 监督；
   - output repair 为主，feature repair 为难例 fallback。

## 还不能得出

1. 不能说 attention/query 是唯一根因。
2. 不能说 score+geometry 联合替换已经验证有效。
3. 不能说局部动作盲目执行就安全。
4. 不能说拟议 router 一定能从无 GT 信号学会 Oracle。
5. 不能说上面的方法一定具有论文级 novelty；还需要在结构定稿后做一次精确查重。

---

# 8. 下一步

现在不建议继续扩 Stage-3A/3B/3C 的大规模诊断。

更高收益的下一步是：

1. 精读第一优先级论文，抽取它们的**输入信号、动作空间、监督、粒度、fallback、质量定义**；
2. 用这些维度画“现有方法 vs 本项目候选方法”的 novelty matrix；
3. 把方法收缩成一个最小可实现版本；
4. 再设计训练标签/Oracle，而不是继续做无边界诊断；
5. 方法确定后，再进入 64–128 帧 Cheap Gate。

相关细分阅读见本目录：

- `01_local_source_and_instance_collaboration.md`
- `02_score_localization_quality_alignment.md`
- `03_adverse_weather_robust_collaboration.md`
- `04_method_design_after_stage3c.md`
