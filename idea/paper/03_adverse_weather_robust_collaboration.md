# 恶劣天气下协同感知：直接相关论文与当前创新边界

> 更新日期：2026-09-20  
> 目的：区分“天气鲁棒协同感知已经做过什么”和“Stage-3C 仍然新增了什么问题”。

---

## 0. 结论

恶劣天气协同感知已经不再是一个空白领域。

当前已经出现至少四条明确路线：

1. **天气噪声去除 / uncertainty reduction**
2. **天气域泛化 / weather-invariant representation**
3. **天气鲁棒模态补充，例如 4D radar**
4. **物理可靠性 / uncertainty 引导协同融合**

因此，本项目不应该再把主创新写成：

- “首次研究恶劣天气协同感知”；
- “用 uncertainty 缓解天气噪声”；
- “天气增强 + 特征对齐”；
- “点密度/可靠性指导融合”；
- “在通信前去掉天气噪声”。

更有价值的空白是：

> **天气不仅让 sensing 变差，还可能改变多来源信息在检测链中的相对价值，使 full collaboration 在局部目标上出现 score、geometry 和 feature-level 的不同失效。现有天气 CP 工作大多在“去噪/域泛化/可靠性加权”层解决问题，没有把这种局部 collaborative repair action 作为核心问题。**

---

# 1. Weather-Aware Collaborative Perception With Uncertainty Reduction

**论文**  
Weather-Aware Collaborative Perception With Uncertainty Reduction  
Ping Jiang, Xiaoheng Deng, Weishang Wu, Lixin Lin, Xuechen Chen, Chen Chen, Shaohua Wan  
IEEE Transactions on Intelligent Transportation Systems, 2024。

可检索入口：  
https://ieeexplore.ieee.org/

Consensus 记录：  
https://consensus.app/papers/weatheraware-collaborative-perception-with-uncertainty-jiang-deng/fb8610303bf050b3ae0a2037933912e2/

## 核心问题

天气噪声会造成 agent 的 feature representation 出现 aleatoric uncertainty，并且协同阶段可能进一步放大这种不确定性。

## 方法

Co-Denoising：

1. 每个 agent 先做 sampling-based noise filtering；
2. collaboration 阶段使用 Bayesian neural network 扩展/建模全局 feature uncertainty。

## 对本项目的价值

这篇非常适合支撑：

> “天气噪声在协同阶段不是自动被多车平均掉，反而可能进一步放大。”

## 创新边界

它已经占用了：

```text
weather
+
uncertainty
+
collaborative denoising
```

这个大方向。

所以本项目不应重新设计一个 Bayesian uncertainty fusion 然后称为主创新。

## 与 Stage-3C 的区别

它主要问：

> 怎样降低天气噪声造成的整体 feature uncertainty？

Stage-3C 更具体：

> 已经存在 task-usable peer evidence 时，为什么 full output 仍然可能在局部候选上丢失 score / geometry / ranking quality？

这是 downstream repair 问题。

---

# 2. V2X-DGW — 天气域泛化已经形成直接基线

**论文**  
[V2X-DGW: Domain Generalization for Multi-Agent Perception Under Adverse Weather Conditions](https://github.com/Baolu1998/V2X-DGW)  
ICRA 2025。  
DOI: 10.1109/ICRA55743.2025.11127945。

## 核心问题

模型只在 clean weather 上训练，面对 unseen adverse weather 出现 domain gap。

## 核心方法

- Adaptive Weather Augmentation；
- Trust-region Weather-invariant Alignment；
- Agent-aware Contrastive Alignment。

并构造：

- OPV2V-W；
- V2XSet-W。

## 与本项目关系

这是当前必须保留的恶劣天气 baseline/related work。

项目此前已经实际复现并使用 OPV2V-W，因此它不是只停留在文献层面的参考。

## 创新边界

以后不能再把：

> “clean 数据做天气增强，然后让 clean/weather feature 对齐”

作为核心创新。

项目自己的 AWA / consistency 路线也已经没有得到稳定提升。

因此外部文献和内部实验是一致的：

> **继续单纯扩天气增强/域对齐的收益可能有限。**

## Stage-3C 提供的新视角

V2X-DGW 主要解决：

```text
clean domain
→
weather domain generalization
```

Stage-3C 更像：

```text
weather 下 peer evidence 已存在
→
full collaboration 没有正确保留/利用
→
局部 output/fusion repair
```

二者可以在论文里形成清楚分工：

- V2X-DGW：提升 representation generalization；
- 本项目候选：修复 collaboration-induced local task failure。

---

# 3. DenoiseCP-Net — 通信前天气去噪已经有人直接做

**论文**  
[DenoiseCP-Net: Efficient Collective Perception in Adverse Weather via Joint LiDAR-Based 3D Object Detection and Denoising](https://arxiv.org/abs/2507.06976)  
2025 arXiv / 2026 IEEE IV。

## 核心观点

天气噪声不仅降低 detection：

> 噪声还会被通信和处理，增加通信/计算负担。

因此在 communication 之前 denoise 很合理。

## 方法

- voxel-level noise filtering；
- detection 与 denoising 共用 unified sparse convolution backbone；
- 去掉 non-informative weather noise。

## 创新边界

以下思路已经有直接对照：

> “天气点云有大量噪点，所以先去噪再通信。”

这与项目早期 Physics-Rain、GSPR、point reliability 的 sensing-side 研究属于相邻层级。

## Stage-3C 的不同

即使 source 已经能够单独检测目标：

> full collaboration 仍可能失败。

这说明至少一部分问题不能通过“把噪点删干净”解释完。

因此本项目可以明确说：

> sensing denoising 是必要背景，但 source-valid/full-miss 说明 downstream collaborative utilization 仍是独立问题。

---

# 4. UECP — 物理 uncertainty map 引导融合已被直接提出

**论文**  
[UECP: Uncertainty-Enhanced Collaborative Perception](https://arxiv.org/abs/2606.23046)  
2026。

## 核心批评

它认为普通 confidence map 与 detection head 共训练，因此可能带 detection bias。

作者希望用：

> 更“物理”的 perception quality evidence。

## 做法

- 直接用 LiDAR point density 监督 uncertainty map；
- Uncertainty-Weighted Downsampling；
- Uncertainty-Guided Residual Fusion；
- coarse-to-fine uncertainty-aware pyramid fusion。

## 为什么这篇对本项目尤其重要

因为它与我们曾经的直觉非常相似：

```text
点密度
→
物理可靠性
→
区域不确定性
→
决定 fusion 权重
```

而项目内部已经做了大量反例验证：

- density / confidence 与真实 marginal contribution 相关性有限；
- low-r/high-u hard filtering 多数条件伤害 AP；
- Stage-2 中 Snow 的 sensing proxy 与 peer-alone task truth 对应较差；
- Stage-3C 中同一个 source 对 score / geometry / feature 的价值可能不同。

因此它给我们一个很清晰的论文定位：

> **我们并不否认 sensing uncertainty；我们质疑的是“一个 sensing-quality scalar 足以决定 downstream collaborative utility”。**

这个边界比“我们设计了更好的 uncertainty”更有价值。

---

# 5. V2X-R — 4D Radar 作为天气稳定证据

**论文**  
[V2X-R: Cooperative LiDAR-4D Radar Fusion with Denoising Diffusion for 3D Object Detection](https://openaccess.thecvf.com/content/CVPR2025/html/Huang_V2X-R_Cooperative_LiDAR-4D_Radar_Fusion_with_Denoising_Diffusion_for_3D_CVPR_2025_paper.html)  
CVPR 2025。

## 核心问题

LiDAR/Camera 天气退化时，4D Radar 更稳。

## 核心做法

- 建 V2X-R 数据集；
- LiDAR-4D radar cooperative fusion；
- Multi-modal Denoising Diffusion；
- 以 radar feature 为条件去噪天气损坏的 LiDAR feature。

## 对本项目意义

如果未来想进一步提高天气上限：

> 引入天气鲁棒传感器是一个强方向。

但当前主线是 LiDAR-only / OPV2V-W，因此不建议此时改模态。

## 创新边界

“用可靠模态修复退化模态”不是空白。

---

# 6. HRCP — 天气下 reliability gating + degraded-region enhancement

**论文**  
[Hybrid Robust Collaborative Perception with LiDAR-4D Radar Fusion under Adverse Weather Conditions](https://openaccess.thecvf.com/content/CVPR2026/html/Yang_Hybrid_Robust_Collaborative_Perception_with_LiDAR-4D_Radar_Fusion_under_Adverse_CVPR_2026_paper.html)  
CVPR 2026。

## 核心模块

- hybrid collaboration strategy；
- Bidirectional Cross-Modal Gating；
- LiDAR / 4D radar 相互验证 feature reliability；
- Adaptive Feature Enhancement；
- 对 degraded/suppressed regions 进行 refinement。

## 对本项目的创新警告

以下词已经非常常见：

- reliability gating；
- adaptive enhancement；
- degraded region refinement。

因此未来写作时不能靠这些名字体现 novelty。

## 可以借鉴什么

“相互验证”这个思想值得借。

本项目虽然没有 radar，但有多个 source：

```text
full
ego
peer_j
```

可以研究：

> source 之间是否存在能够预测 score/geometry repair value 的一致性/冲突模式。

---

# 7. Uncertainty-guided reliable collaborative perception 2026

**论文**  
[Uncertainty-guided and reliable collaborative perception for open heterogeneous systems](https://www.sciencedirect.com/science/article/pii/S0167865526001297)  
Pattern Recognition Letters 2026。  
DOI: 10.1016/j.patrec.2026.04.011。

## 核心方法

1. Sparsity-Aware Active Query：针对 blind regions 请求信息；
2. Foreground-Aware Adaptive Transmission：发送前景、过滤背景；
3. Dual-stream Multi-level Fusion：
   - collaborative intermediate feature；
   - single-agent detection proposal；
   两路并行。

## 为什么它对 Q-A/Q-B 都有影响

对于 Q-A：

> “识别盲区后主动请求邻车”已有直接路线。

对于 Q-B：

> “协同不可靠时保留单车检测分支作为 fallback”也已有直接路线。

## 因此本项目不能只做

```text
观测不足区域检测
+
主动请求
+
late proposal fallback
```

这会非常接近已有工作。

## 更细的区别

本项目应研究：

> full collaboration 已经形成候选后，如何针对**具体错误属性**进行最小 repair，而不是笼统切换成 local proposal。

---

# 8. CoRA — 鲁棒协同中的 object-level correction

**论文**  
[CoRA: A Collaborative Robust Architecture with Hybrid Fusion for Efficient Perception](https://ojs.aaai.org/index.php/AAAI/article/view/37274)  
AAAI 2026。

## 核心动机

intermediate fusion 强，但现实扰动会造成 spatial displacement。

## 做法

- feature-level fusion branch；
- object-level correction branch；
- 用 semantic relevance 做空间位置修正。

## 和天气项目的关系

它不是天气论文，但对我们很重要：

> object-level correction 已被证明是鲁棒协同里合理的一层。

## 创新边界

如果我们做 output repair：

> 必须明确它是在修“天气导致的 source-conditioned score/geometry corruption”，而不是一般 pose correction。

---

# 9. ICPB — uncertainty + proposal-level routing 已经出现

**论文**  
[Perception balance with uncertainty-guided fusion and proposal-wise mixture-of-experts for robust multi-agent 3D object detection](https://www.sciencedirect.com/science/article/abs/pii/S0957417426002241)  
Expert Systems with Applications 2026。

## 为什么天气论文也必须读它

虽然它不是专门天气，但它已经覆盖：

- feature uncertainty；
- proposal uncertainty；
- uncertainty-aware fusion；
- proposal-wise experts；
- individual/collaborative balance。

所以如果本项目把天气问题写成：

> “weather uncertainty 高时更多依赖 peer，低时依赖 ego/full”

创新会非常危险。

## 更好的区分

不要把 weather 当 router 的唯一 context。

而是：

```text
weather sensing state
+
source-specific detection disagreement
+
local feature disagreement
+
proposal competition context
→
repair action
```

也就是说天气只是导致失败的条件之一，不是决策的全部。

---

# 10. Adver-City — 后续跨数据集验证值得考虑

**论文 / 数据集**  
[Adver-City: Open-Source Multi-Modal Dataset for Collaborative Perception Under Adverse Weather Conditions](https://arxiv.org/abs/2410.06380)  
ITSC 2025。

项目网站：  
https://labs.cs.queensu.ca/quarrg/datasets/adver-city/

## 数据特点

- CARLA + OpenCDA；
- 多种天气与昼夜；
- vehicle + roadside unit；
- LiDAR、RGB、semantic camera、GNSS、IMU；
- annotation schema 与 OPV2V 兼容。

## 对本项目的实际价值

如果后续方法在 OPV2V-W 有结果，审稿人很可能会问：

> 是否只对 OPV2V-W 的天气模拟有效？

因此 Adver-City 可以作为：

- cross-dataset 验证；
- 不同道路结构/密度；
- 不同天气组合；

的候选。

目前不建议立刻迁移代码，先把方法做通。

---

# 11. 天气 CP 文献按“解决层级”分类

| 层级 | 代表工作 | 解决什么 | 本项目是否还应主攻 |
|---|---|---|---|
| 点/voxel 去噪 | DenoiseCP-Net、Weather-Aware CP | 减少天气污染 | 不作为当前主线 |
| weather domain generalization | V2X-DGW | clean→weather 泛化 | 已做/已验证收益有限，不优先 |
| physical uncertainty | UECP | sensing quality→fusion | 内部反例较多，不作为主线 |
| robust modality | V2X-R、HRCP | radar 弥补 LiDAR | 高成本换模态，不优先 |
| active blind-region query | Uncertainty-guided CP | 哪里需要额外信息 | 已有工作较直接 |
| hybrid/fallback | Uncertainty-guided CP、CoRA、CoSDH | intermediate 不稳时用其他层级 | 结构本身不新 |
| **local collaborative repair** | **当前未发现完全对位工作** | **已有 source evidence 被 full 局部破坏后如何最小修复** | **当前最值得主攻** |

---

# 12. 本项目可以形成的天气特有叙事

建议不要说：

> “天气导致噪声，所以我们去噪。”

而是：

```text
天气改变不同来源的局部 evidence quality
→
full fusion 并不保证同时保留最好的 classification 与 localization evidence
→
一些 source-valid target 在 collaboration 后消失
→
局部修复可以保留恢复能力并大幅减少附带伤害
→
因此需要 weather-aware but task-driven local collaborative repair
```

这里：

> weather-aware

并不等于：

> 用天气标签作为输入。

它可以体现为：

- 使用 GSPR sensing state 作为辅助；
- 使用 source disagreement；
- 使用 proposal quality；
- 跨 Fog/Rain/Snow 学习共享 + 条件化行为。

---

# 13. Fog / Rain / Snow 不应强行用一个机制解释

当前证据：

## Fog / Rain

更突出：

- NMS competition；
- high-score / inaccurate geometry；
- geometry repair 有明显作用。

## Snow

更突出：

- score filtering；
- score repair 更重要；
- 仍有 weight-only recovery；
- 部分 geometry 修复后又被 score 卡住。

因此更合理的方法是：

> 同一 action space，允许不同天气自然出现不同 action distribution。

而不是：

```text
if fog:
  module A
if rain:
  module B
if snow:
  module C
```

否则容易过拟合天气标签。

---

# 14. GSPR 在新方法中的正确位置

GSPR 值得保留，但角色需要降级和重新定义。

## 不应该

```text
GSPR reliability
→
直接决定 peer/full utility
```

## 应该

```text
GSPR reliability
→
描述当前 sensing corruption state
→
作为 router / quality predictor 的辅助 context
```

例如：

```text
router input =
  source proposal disagreement
  + local feature disagreement
  + predicted q_cls/q_loc
  + GSPR reliability statistics
```

这样既利用已有资产，也尊重历史反证。

---

# 15. Related Work 建议结构

未来论文 Related Work 可以按三段写：

## 15.1 Collaborative perception and selective communication

Where2comm、How2comm、CodeFilling、CoSDH、Select2Col、QUEST、CoopDETR、INSTINCT。

重点指出：

> 这些方法主要决定何时、哪里、和谁协作，或如何降低通信；本项目研究 collaboration 后的局部 task failure repair。

## 15.2 Robust collaborative perception

SCOPE、CoAlign、ROBOSAC、CoRA、ICPB、uncertainty-guided reliable CP。

重点指出：

> 现有鲁棒方法主要处理 pose/time/attack/communication degradation 或 generic uncertainty；本项目关注 adverse weather 下 source-valid evidence 的局部损失。

## 15.3 Adverse-weather perception

Weather-Aware CP、V2X-DGW、DenoiseCP-Net、V2X-R、HRCP。

重点指出：

> 多数工作从 sensing denoising、domain generalization 或 robust modality 解决天气问题，而不是显式建模 collaborative score–localization repair。

---

# 16. 当前结论

### 可以得出

1. “恶劣天气协同感知”本身已经不是 novelty。
2. “uncertainty / reliability / density → fusion”已经有很新的直接论文。
3. “天气域泛化/增强”已有 V2X-DGW，项目内部也没有得到稳定收益。
4. Stage-3C 最有价值的新信息发生在 downstream task chain，而不是简单 sensing noise。
5. 因此方法应从“天气去噪”转向“天气条件下局部 collaborative repair”。

### 不能得出

1. 不能声称现有天气论文都没有考虑局部区域；
2. 不能声称所有天气 failure 都来自 fusion；
3. 不能说 GSPR 没价值；
4. 不能说 4D radar 路线不值得，只是当前项目成本/主线不适合立即切换；
5. 不能提前声称 source-conditioned repair 是首次，仍需在具体 architecture 定稿后做二次精准查重。
