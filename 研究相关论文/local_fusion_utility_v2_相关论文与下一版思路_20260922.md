# local_fusion_utility_v2 相关论文与下一版方法启发

> 更新日期：2026-09-22
> 目的：针对当前 local_fusion_utility_v2 在 Snow 上 AP@0.7 提升约 2.4~2.7 个百分点、但 Clean/Fog/Rain 提升明显偏小的问题，跨协同感知、多模态融合、鲁棒传感器融合、动态路由、多视图学习和通用视觉网络调研可迁移思路。
> 注意：当前 GitHub main 分支尚未包含 local_fusion_utility_v2/ 最新本地实现，因此本文对 v2 的具体流程以当前实验描述为准：局部摘要 → MLP 预测动作后果 → KEEP_FULL/peer-query 选择 → 在尺度0、1重新计算局部融合 → 原检测头重新预测。

---

## 0. 当前问题先说清楚

目前 v2 已经不是“检测后调框”，而是在特征融合阶段做局部干预。

当前动作大致是 KEEP_FULL、Peer1-query、Peer2-query、Peer3-query 等。MLP 的职责不是直接输出框偏移或 score，而是根据约 22 维局部摘要判断某个 peer-query 动作是否值得执行。真正的 box/score 变化来自局部融合特征改变后，原检测头重新计算。

当前完整门槛扫描中，utility 的较好统一门槛能让四种条件都不退化，但收益高度不均：

- Clean：约 +0.29 pp
- Fog：约 +0.23~0.28 pp
- Rain：约 +0.20~0.35 pp
- Snow：约 +2.4~2.7 pp

因此当前最重要的问题不是“v2 有没有用”，而是：为什么同样的局部融合动作在 Snow 有明显空间，在 Fog/Rain/Clean 上可利用空间很小？

我认为至少要区分三个潜在瓶颈，不能都归因到“MLP 太小”：

1. 输入信息瓶颈：完整局部 BEV 空间结构被压成少量统计量，模型可能不知道目标形状和来源间局部空间关系。
2. 动作空间瓶颈：模型只能从若干固定 query 产生的权重方案中选一个，不能形成新的、连续的来源权重组合。
3. 学习目标瓶颈：当前动作后果监督是否真正等价于“最终 AP 有益”，尤其 Fog/Rain 中 score、geometry、NMS 竞争链可能比 Snow 更复杂。

下一版最好通过消融分别回答这三件事，而不是一次性堆大网络。

---

# 1. 第一组：最直接支持“保留局部空间信息”的论文

## 1.1 TokenFusion — Multimodal Token Fusion for Vision Transformers

Wang et al., CVPR 2022  
论文：https://openaccess.thecvf.com/content/CVPR2022/html/Wang_Multimodal_Token_Fusion_for_Vision_Transformers_CVPR_2022_paper.html  
DOI: 10.1109/CVPR52688.2022.01187

核心思想：

- 不把一个模态整体判断为“好/坏”，而是细到 token 层。
- 动态发现信息不足的 token。
- 用另一个模态中投影、聚合后的信息去补。
- 保留位置对齐关系。

对当前 v2 的直接启发：

当前 v2 已经做到“局部区域 × peer”，但模型看到的还是统计摘要。TokenFusion 更深一层的启发是：局部决策应该直接接触局部空间结构，而不是只接触经过人工汇总后的标量。

可迁移到你的场景：Ego patch、Peer patch、Full patch 先经过轻量共享空间编码器，再判断来源贡献。这里不要求照搬 Transformer，5×5 或 7×7 BEV patch 加小卷积就能先验证“空间信息是否有价值”。

重叠风险：

如果最后方法只是“检测低质量 BEV cell，然后替换为其他来源 cell”，会与 TokenFusion 思路很近。你的区别应该落在同源多车 LiDAR、天气导致的局部 collaborative failure、不是硬替换 token，而是受约束地修正多来源融合权重。

---

## 1.2 AutoAlignV2 — Deformable Feature Aggregation for Dynamic Multi-Modal 3D Object Detection

Chen et al., ECCV 2022  
论文：https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/132_ECCV_2022_paper.php  
DOI: 10.1007/978-3-031-20074-8_36

核心思想：

- 不对整个区域做笨重的全局交互。
- 在局部选择少量有价值的采样点。
- 学习跨来源/跨模态对应关系。
- 只聚合真正有用的空间证据。

对当前 v2 的启发：

如果后续发现 7×7 patch 全部展开代价大、噪声多，可以不做整块卷积，而让网络学习“这个局部目标附近，哪些 BEV cell 才是判断 Peer1/Peer2 是否可靠的关键位置”。

---

## 1.3 DeepInteraction — 3D Object Detection via Modality Interaction

Yang et al., NeurIPS 2022  
论文：https://proceedings.neurips.cc/paper_files/paper/2022/hash/0d18ab3b5fabfa6fe47c62e711af02f0-Abstract-Conference.html  
DOI: 10.52202/068431-0145

核心思想：

过早把多个来源融合成一个统一表示，会丢掉各来源本身有用的独特信息。因此它长期保留 modality-specific representation，并在检测过程中多次交互。

对当前 v2 的启发：

你的 v2 虽然还拿得到 per-source 特征，但进入 MLP 前把它压成了摘要。可以借鉴它的原则：在决定“谁该占多少权重”之前，尽量晚一点丢掉 source-specific feature。

下一版可以让每个来源局部 patch 先经过共享编码器，各自得到一个小向量，再做来源间比较；不要一开始就手工压成 score、差值、均值等统计量。

这篇对当前问题很重要，因为它直接支撑“输入信息瓶颈”这个怀疑。

---

# 2. 第二组：最直接支持“动态来源权重，而不是固定 peer-query 套餐”的论文

## 2.1 QMF — Provable Dynamic Fusion for Low-Quality Multimodal Data

Zhang et al., ICML 2023  
论文：https://proceedings.mlr.press/v202/zhang23ar.html  
Consensus：https://consensus.app/papers/provable-dynamic-fusion-for-lowquality-multimodal-data-zhang-wu/36903a03d8ae59439c667c5c095ba9aa/

核心思想：

多来源输入质量会动态变化，低质量来源如果仍被同样融合，会污染最终结果。QMF 用不确定性/质量感知进行动态融合，而不是固定融合。

对当前 v2 的启发：

当前动作空间是 Ego-query、Peer1-query、Peer2-query 等几套离散权重。更灵活的下一步可以是：

原 AttFuse 权重 + 模型根据当前局部多来源特征预测的小幅修正。

一种可行形式是对原 attention logit 加一个有界残差，再做 softmax。残差用 tanh 和系数限制最大改动幅度。

关键提醒：

你自己的实验已经说明 density、confidence、GSPR reliability 都不能直接当成“真实协同价值”。所以 QMF 最值得学的是“质量变化应该进入动态融合”，而不是“直接把 uncertainty 当最终权重”。GSPR reliability 更适合作为 context feature，而不是权重答案。

---

## 2.2 Selective Sensor Fusion for Neural Visual-Inertial Odometry

Chen et al., CVPR 2019  
论文：https://openaccess.thecvf.com/content_CVPR_2019/html/Chen_Selective_Sensor_Fusion_for_Neural_Visual-Inertial_Odometry_CVPR_2019_paper.html  
DOI: 10.1109/CVPR.2019.01079

核心思想：

面对丢失、损坏、不同步的传感器输入，它直接从特征学习 soft mask 和 hard mask，决定不同来源应该保留多少。

为什么和你很像：

你现在其实已经在做“选择”，但选择对象是固定 peer-query。这篇给出的启发是：可以直接学习 feature-level soft gate，而不是先构造若干离散融合动作再让 MLP 选。

重叠风险：

“根据特征学习 gate”本身早已不新。你的研究差异应来自多车同模态协同、天气局部退化、基于 baseline AttFuse 的保守残差修正，以及显式关注 recovered target / lost TP / new FP。

---

## 2.3 Trusted Multi-View Classification / Dynamic Evidential Fusion

Han et al., ICLR 2021；后续 TPAMI 版本  
论文：https://arxiv.org/abs/2102.02051  
动态证据融合版本：https://arxiv.org/abs/2204.11423

核心思想：

不同 view 对不同样本的可信度不同，因此每个来源自己产生 evidence、建模 uncertainty，然后动态整合，而不是默认所有 view 一样可信。

对你的启发：

你的 CAV 可以类比成多个 view，但必须避免简单照搬“view uncertainty = CAV usefulness”。更值得借鉴的是：先保留每个来源自己的 evidence，再做关系判断。

---

## 2.4 Adaptive Feature Fusion for Cooperative Perception Using LiDAR Point Clouds

Qiao & Zulkernine, WACV 2023  
论文：https://openaccess.thecvf.com/content/WACV2023/html/Qiao_Adaptive_Feature_Fusion_for_Cooperative_Perception_Using_LiDAR_Point_Clouds_WACV_2023_paper.html

核心思想：

- trainable feature selection；
- spatial-wise adaptive fusion；
- 在 OPV2V 上提高协同感知性能。

对你的意义主要是创新边界：

如果下一版只是“给每辆车的 BEV 特征学一个空间 attention weight”，很危险，因为 AdaFusion 已经占了很大一块。

因此下一版必须与普通 spatial adaptive fusion 区分：不是重新学习一套从零开始的 fusion attention，而是针对已经训练好的 baseline，在识别到局部 full-collaboration failure 后，对原有权重做受约束 correction。

这个 baseline-preserving local residual correction 才可能形成更清楚的边界。

---

# 3. 第三组：支持“局部动态路由，而不是一个全局选择器”的论文

## 3.1 Dynamic Multimodal Fusion

Xue & Marculescu, CVPR Workshops 2023  
Consensus：https://consensus.app/papers/dynamic-multimodal-fusion-xue-marculescu/e558aadf97bf516f9851524bb57a06d9/

核心思想：

不同输入需要不同融合路径，因此用 gating function 在推理时动态决定哪些模态进入、哪些 fusion path 被激活。

对 v2 的启发：

v2 已经具备 router 的雏形。下一版可以把 router 从“动作选择”改成两阶段：先判断 KEEP / MODIFY；如果 MODIFY，再预测有限幅度的连续来源权重残差。这样可以保留显式 KEEP，而不是所有区域强制重新融合。

---

## 3.2 Router-Gated Cross-Modal Feature Fusion（音视频语音识别）

Lim et al., IEEE ASRU 2025  
Consensus：https://consensus.app/papers/improving-noise-robust-audiovisual-speech-recognition-lim-kim/d8d6709ee0d25a639cdb32eba818e6bd/

核心思想：

- 音频会被局部噪声破坏。
- router 估计 token-level corruption。
- gated cross-attention 在局部降低坏音频、增加视觉贡献。

为什么值得重点看：

它和你的问题结构很接近。音频某些 token 被噪声污染，类似某辆 CAV 某些 BEV 区域被天气污染；不能说整个模态都坏，也不能说整个 Peer 都坏；需要 local router。

这里最值得借鉴的是粒度：reliability / routing 不一定做到整 source，而可以做到局部 token / patch。

---

## 3.3 MetaBEV — Solving Sensor Failures for 3D Detection and Map Segmentation

Ge et al., ICCV 2023  
论文：https://openaccess.thecvf.com/content/ICCV2023/html/Ge_MetaBEV_Solving_Sensor_Failures_for_3D_Detection_and_Map_Segmentation_ICCV_2023_paper.html  
DOI: 10.1109/ICCV51070.2023.00801

核心思想：

MetaBEV 用 dense BEV queries 迭代、选择性地从不同传感器聚合特征，在 sensor corruption 和 sensor missing 情况下仍保持鲁棒。

对你的启发：

你现在是固定 query 候选再选哪个 query。MetaBEV 更像每个 BEV query 自己决定当前该从哪些来源拿多少信息。

因此下一版可以让局部目标区域自身形成一个“小 query”，同时读取 Ego/Peer 特征，再输出来源权重修正，而不是让某个 Peer 本身充当唯一 query。

重叠风险：

“BEV query 选择性聚合不同来源”已经有大量工作，因此不能把 query-based local fusion 本身当主创新。

---

# 4. 第四组：支持“不要只学习权重，还要让来源表示本身更稳定”的论文

## 4.1 EAU — Embracing Unimodal Aleatoric Uncertainty for Robust Multimodal Fusion

Gao et al., CVPR 2024  
论文：https://openaccess.thecvf.com/content/CVPR2024/html/Gao_Embracing_Unimodal_Aleatoric_Uncertainty_for_Robust_Multimodal_Fusion_CVPR_2024_paper.html

核心思想：

它没有把 uncertainty 只当一个最终融合权重，而是先利用 aleatoric uncertainty 让单模态表示本身更稳定，再进行鲁棒融合。

对你的启发：

如果 Fog/Rain 的问题不是“选错了一个 query”，而是各 source 特征本身混乱，那么只改融合权重上限会很低。

可以考虑在局部 fusion weight predictor 前增加非常轻的 source-local feature stabilization / normalization，例如共享小卷积、source-wise normalization 或 disagreement-aware residual feature correction。不过这个方向要谨慎，不要一下子把问题扩展成完整去噪网络。

---

## 4.2 HRCP — Hybrid Robust Collaborative Perception with LiDAR-4D Radar Fusion under Adverse Weather Conditions

Yang et al., CVPR 2026  
论文：https://openaccess.thecvf.com/content/CVPR2026/html/Yang_Hybrid_Robust_Collaborative_Perception_with_LiDAR-4D_Radar_Fusion_under_Adverse_CVPR_2026_paper.html

核心思想：

该工作在恶劣天气协同感知中使用 bidirectional cross-modal gating 和 adaptive feature enhancement，让不同模态互相验证可靠性，并增强被压制区域。

对当前项目的真正启发：

虽然你只有 LiDAR 多车，不是 LiDAR+Radar，但有一个重要思想可以迁移：来源可靠性不一定应该由来源自己单独判断，可以通过来源之间的相互验证得到。

例如 Peer1 patch 与 Ego 一致、与 Peer2 也一致，但 Full 反而偏离。这种“跨来源一致/不一致模式”可能正是 Fog/Rain 下 MLP 摘要没有充分表达的信息。

---

# 5. 第五组：和下一版非常接近，必须重点防止“重复创新”的论文

## 5.1 ICPB — Perception Balance with Uncertainty-Guided Fusion and Proposal-wise Mixture-of-Experts

Ni et al., Expert Systems with Applications, 2026  
DOI: 10.1016/j.eswa.2026.131311  
论文页：https://www.sciencedirect.com/science/article/abs/pii/S0957417426002241

核心思想：

- feature-level uncertainty-aware fusion；
- proposal-level mixture-of-experts；
- 在 multi-agent 3D detection 中根据 uncertainty 平衡独立感知和协同感知。

与潜在下一版的重叠：

如果你做“proposal-local features → uncertainty → MoE/router → 决定协同来源”，很容易和 ICPB 靠得太近。

建议保持的差异：

你的问题不是泛化的“单车和协同谁更好”，而是“在恶劣天气下，full collaboration 已经发生以后，哪些局部目标出现了来源特定的融合副作用；能否在尽量保持原 full 行为的前提下，对来源权重做最小必要修正”。

因此不建议把下一版包装为 proposal-wise MoE，也不建议以 epistemic uncertainty 作为唯一核心。更应该强调 baseline-preserving residual source reweighting、weather-induced local fusion failure 和 safe correction。

---

# 6. 通用视觉网络中两个轻量结构，也值得作为工程候选

## 6.1 Dynamic Convolution

Chen et al., CVPR 2020  
论文：https://openaccess.thecvf.com/content_CVPR_2020/html/Chen_Dynamic_Convolution_Attention_Over_Convolution_Kernels_CVPR_2020_paper.html  
DOI: 10.1109/CVPR42600.2020.01104

启发：

不是“网络越深越好”，而是根据当前输入动态混合多个简单算子，也可以显著提高表达能力。

如果担心 MLP 太弱，可以不直接上大 Transformer，而用少量动态分支处理不同 source relation / 不同局部尺度，这对计算量更友好。

---

## 6.2 Selective Kernel Networks

Li et al., CVPR 2019  
论文：https://openaccess.thecvf.com/content_CVPR_2019/html/Li_Selective_Kernel_Networks_CVPR_2019_paper.html  
DOI: 10.1109/CVPR.2019.00060

启发：

目标大小和天气污染范围可能不同，所以固定 3×3 或 7×7 patch 不一定适合所有目标。

可以让局部编码器同时观察 3×3、5×5、7×7，然后动态选择尺度。但这只能作为结构细节，不能作为论文主创新。

---

# 7. 综合这些论文后，我认为下一版最值得验证的结构

暂时不要急着命名成论文方法，可以内部叫：

Bounded Residual Local Source Fusion（受约束局部来源残差融合）

## 7.1 输入：不要再只剩 22 维摘要

对每个候选局部区域，在尺度0、1保留：

- Ego patch
- Peer1 patch
- Peer2 patch
- ...
- Full patch

先用共享轻量卷积编码每个 source，例如 1×1 降维后接 3×3 depthwise 或普通卷积，得到低维 spatial embedding。

然后显式构造：

- Ego ↔ Peer 差异；
- Peer ↔ Peer 一致性；
- Full ↔ 各 source 残差；
- 原始 AttFuse weight/logit；
- 当前 detection score/geometry 摘要；
- GSPR reliability 仅作为辅助 context。

这样既保留空间信息，也不会直接把完整 128/256 通道全部喂给大网络。

## 7.2 输出：不是直接预测全新的权重

让网络输出：

- KEEP gate g；
- 每个 source 的 attention-logit residual Δ_i。

残差要有界，例如用 tanh 和系数 β 限制最大幅度。

最终逻辑是：

原 AttFuse attention logit + 有界局部残差 → softmax → 新权重。

初始时让 g≈0、Δ≈0，使模型精确或近似恢复原 baseline。训练过程学习“哪里真的值得动”。

另一种更容易解释的实现是：

新权重 = (1-g) × 原权重 + g × 新预测权重，

并限制 0 ≤ g ≤ g_max。

## 7.3 为什么这种结构比当前 peer-query 动作更合理

当前是从几套现成权重套餐中选一个。下一版则是以原权重为中心，根据当前目标的局部多来源证据连续小幅移动。

假设真正合适的是 Ego 0.50、P1 0.10、P2 0.35、P3 0.05，而所有 query 都没有这套组合，当前 v2 永远做不到；残差权重修正可以做到。

---

# 8. 下一版实验不要一次加完，建议用四组实验判断瓶颈

## A. v2 原版

22维摘要 + 固定 peer-query 动作。

## B. 只改输入，不改动作

局部 BEV patch encoder + 仍然只选 KEEP / peer-query。

它回答：当前主要是不是“信息被摘要丢掉了”？

如果 B 明显超过 A，说明输入信息瓶颈成立。

## C. 不大改输入，只改动作空间

当前摘要 + 连续受约束 weight residual。

它回答：当前主要是不是“固定 peer-query 动作不够灵活”？

如果 C 明显超过 A，说明动作空间瓶颈成立。

## D. 两者都加

局部空间特征 + 连续受约束 weight residual。

只有 B/C 已经证明各自有价值以后，再跑 D。

这样最终论文才能回答：性能提升到底来自“看得更多”，还是“能做得更多”。

---

# 9. 训练方面的一个关键提醒

当前缓存里如果只有摘要和固定 query 动作后果，它无法直接训练新的连续融合模块。

新的模型需要：

- 原始或降维后的 per-source local BEV features；
- 原 AttFuse logits/weights；
- 融合后的 detection loss 或相应局部监督。

如果冻结 detector，冻结的是参数，不是对输入 feature 的梯度。因此不能把整个 detection head 前向都放在 torch.no_grad() 中，否则新的融合权重无法通过检测损失学习。

---

# 10. 优先精读顺序

如果只先读 8 篇，按对当前问题的价值排序：

1. TokenFusion, CVPR 2022 —— 看“局部空间 token 不应该过早压缩”。
2. QMF, ICML 2023 —— 看“低质量来源下动态融合应该如何建模”。
3. DeepInteraction, NeurIPS 2022 —— 看“为什么保留 source-specific representation 有价值”。
4. Selective Sensor Fusion, CVPR 2019 —— 看“feature-conditioned soft gate / hard gate”。
5. AdaFusion, WACV 2023 —— 看与你直接重叠的 spatial adaptive cooperative fusion，确定创新边界。
6. ICPB, ESWA 2026 —— 看 proposal-wise MoE + uncertainty 在 multi-agent detection 已经做到哪里，避免重复。
7. MetaBEV, ICCV 2023 —— 看 query 如何动态读取不同来源，而不是把某一个来源固定当 query。
8. EAU, CVPR 2024 —— 看 uncertainty 如何用于稳定表示，而不是只做简单权重。

第二梯队：

- AutoAlignV2, ECCV 2022；
- Router-Gated Cross-Modal Fusion, ASRU 2025；
- HRCP, CVPR 2026；
- Dynamic Multimodal Fusion；
- Trusted Multi-View Classification；
- Dynamic Convolution；
- Selective Kernel Networks。

---

# 11. 当前能得出什么结论

1. 单纯“把 MLP 加深”不是最有依据的下一步。现有文献反复说明，低质量多来源融合的瓶颈常常同时来自表示丢失和融合动作不够动态。
2. 当前 v2 的固定 peer-query 动作空间确实存在表达上限。MLP 再强也只能选已有权重组合，不能创造不存在的来源权重。
3. 保留局部空间特征非常值得验证。TokenFusion、AutoAlignV2、DeepInteraction、MetaBEV 等不同领域方法都说明：局部结构、来源特有表示和局部交互不应过早压缩。
4. 连续来源权重修正是合理下一步，但“自适应权重”本身不是创新。AdaFusion、QMF、Selective Sensor Fusion、MetaBEV、ICPB 等已经覆盖大量相关思想。
5. 最有希望形成自己研究边界的不是 attention/gating 这些模块名，而是问题定义和约束方式：恶劣天气、full collaboration 的局部负作用、source-conditioned local evidence、baseline-preserving、最小必要 residual correction、显式关注恢复目标同时不丢原 TP、不新增 FP。

---

# 12. 目前还不能得出什么结论

1. 不能根据 Snow +2.x pp 就断定“MLP容量不足”。
2. 不能断定 Fog/Rain 提升小一定是权重选择问题；也可能是 candidate coverage、监督目标或检测链竞争更复杂。
3. 不能宣称“局部连续权重修正”具有论文创新性；它与多篇动态/自适应融合工作存在明显邻接。
4. 不能把 GSPR reliability、confidence、uncertainty 直接等价为 source utility。
5. 不能在下一版同时加卷积、attention、MoE、连续权重、RepairNet 后只看最终 AP；这样无法知道真正贡献来自哪里。

---

# 13. 下一步最值得验证什么

按信息增益排序：

1. 先让 v2 完整结束，固定当前协议。
2. 做 B：空间信息增强，但动作空间不变。
3. 做 C：动作空间改成受约束连续残差权重，但尽量保持当前输入。
4. 比较 B/C 谁带来明显收益，再决定 D。
5. 所有实验同时看 AP@0.5 / AP@0.7、Clean/Fog/Rain/Snow、原 TP 损失、new FP、修改区域比例、权重改动幅度和运行开销。
6. 如果 Fog/Rain 仍然不涨，再检查“动作监督与最终 NMS/AP 是否不一致”，而不是继续无上限增加模型容量。

---

# 14. Consensus 本轮检索记录

本轮通过 Consensus 检索了低质量多模态融合、动态路由、局部跨来源 gating、鲁棒传感器融合等方向。Consensus 本月搜索额度在本轮多查询后达到上限，因此后续补充使用论文官方页面、CVF、PMLR、NeurIPS、ECCV/ECVA 和出版商页面进行交叉核验。

Consensus 重点返回：

- Dynamic Multimodal Fusion
- Provable Dynamic Fusion for Low-Quality Multimodal Data (QMF)
- Multimodal Fusion on Low-quality Data: A Comprehensive Survey
- Router-Gated Cross-Modal Feature Fusion
- Variance-Guided Spatial Attention Fusion
- Adaptive Sensor Fusion for Robust Perception in Dense Fog

其中 QMF、Dynamic Multimodal Fusion 和 Router-Gated Fusion 对当前问题最有直接方法启发。
