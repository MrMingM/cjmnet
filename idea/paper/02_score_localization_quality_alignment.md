# 分数—定位质量对齐、候选排序与 NMS：与 Stage-3C 的关系

> 更新日期：2026-09-20  
> 目的：厘清“高分但定位不准”是不是新问题，以及本项目怎样避免把成熟 detector 技术误包装成创新。

---

## 0. 结论

Stage-3C 在 Fog/Rain 中观察到的现象：

> 高分但定位不够准的竞争框压掉更合格的候选；局部几何修复后，原 suppressor 本身可以变成正确匹配。

这不是一个全新的 detector 现象。

单车 2D/3D detection 领域已经有大量工作专门解决：

> **classification confidence 与 localization quality 不一致。**

因此：

- 加一个 IoU prediction head，不新；
- 用 IoU 修正 score，不新；
- 用 quality-aware score 做 NMS，不新；
- 把 cls score 与 IoU score 相乘，不新；
- 改一个 localization-aware NMS，也很难成为主创新。

但 Stage-3C 比普通 detector 多了一层：

> **同一个目标的 classification evidence 和 geometry evidence 来自多个协同来源，full fusion 可能把不同来源的优点一起稀释。**

所以真正有潜力的是：

> **source-conditioned score–geometry quality alignment / repair。**

---

# 1. CIA-SSD：3D 点云里最直接的对照

**论文**  
[CIA-SSD: Confident IoU-Aware Single-Stage Object Detector From Point Cloud](https://ojs.aaai.org/index.php/AAAI/article/view/16470)  
AAAI 2021。  
DOI: 10.1609/aaai.v35i4.16470。

## 它解决什么

传统 single-stage detector：

```text
classification head
regression head
```

是两个相对独立的任务。

结果可能出现：

```text
分类分数很高
但框的位置并不好
```

而 NMS 又通常依赖 classification score 排序，于是会把几何更好的框压掉。

## 它怎么做

- Spatial-Semantic Feature Aggregation；
- IoU-aware confidence rectification；
- Distance-variant IoU-weighted NMS。

## 和 Stage-3C 的相似点

非常直接。

Stage-3C Fog/Rain 中：

> 原 NMS suppressor 在几何替换后变成最终正确框。

这说明“score 与 geometry 不一致”确实是一个合理解释。

## 但为什么不能直接照搬

CIA-SSD 假设只有一个 detector/source。

本项目有：

```text
ego
peer 1
peer 2
...
full fusion
```

因此更细的问题是：

> 哪个 source 的 score 更可信？哪个 source 的 geometry 更可信？

这不再只是一个单 proposal 的 IoU-aware rescore 问题。

---

# 2. 3D IoU-Net：直接预测 localization quality

**论文**  
[3D IoU-Net: IoU Guided 3D Object Detector for Point Clouds](https://arxiv.org/abs/2004.04962)  
2020。

## 核心做法

在：

```text
classification
+
box regression
```

之外增加：

```text
3D IoU prediction
```

并把预测 IoU 用作 NMS 的 detection confidence。

同时做：

- IoU-sensitive feature learning；
- IoU alignment；
- corner geometry encoding。

## 对我们的意义

如果我们需要一个：

```text
q_loc = predicted localization quality
```

它是很自然的技术参考。

但论文创新不能写：

> “我们预测 proposal IoU，改善 NMS。”

这早已有了。

## 可以怎样借

把 q_loc 从单一 proposal quality 扩展成：

```text
q_loc(source=j, proposal=k)
```

然后比较：

- full geometry quality；
- peer geometry quality；
- ego geometry quality。

这样 q_loc 是**动作选择输入**，而不是最终创新本身。

---

# 3. AFDetV2：classification × IoU 的直接结合

**论文**  
[AFDetV2: Rethinking the Necessity of the Second Stage for Object Detection from Point Clouds](https://ojs.aaai.org/index.php/AAAI/article/view/19980)  
AAAI 2022。  
DOI: 10.1609/aaai.v36i1.19980。

## 核心观察

如果 box regression 已经足够好，第二阶段很多时候主要是在：

> 把定位更好的框重新排到前面。

因此 AFDetV2 加 IoU prediction branch，并将：

```text
classification confidence
×
predicted IoU
```

形成最终 confidence。

## 创新边界

所以以下方法过于常规：

```text
final_score = cls_score * iou_score
```

即使在 collaborative perception 中直接加，也更像 detector engineering，而不是强创新。

## 对 Stage-3C 的启发

Stage-3C 中有两类不同失败：

### Fog/Rain 常见

```text
score高
geometry差
→
NMS竞争失败
```

### Snow 常见

```text
geometry仍合格
score掉太低
→
score filter失败
```

因此固定乘法未必合适。

更合理的模型应该知道：

> 当前主要需要“修 score”还是“修 geometry”。

---

# 4. VarifocalNet：联合表达 objectness 和 localization

**论文**  
[VarifocalNet: An IoU-Aware Dense Object Detector](https://openaccess.thecvf.com/content/CVPR2021/html/Zhang_VarifocalNet_An_IoU-Aware_Dense_Object_Detector_CVPR_2021_paper.html)  
CVPR 2021。  
DOI: 10.1109/CVPR46437.2021.00841。

## 核心问题

候选排序非常重要。

普通 classification score 并不能稳定代表：

- 这个目标存在的概率；
- 这个框到底定位得有多准。

## 核心做法

学习 IoU-Aware Classification Score：

> 一个同时编码 object presence confidence 和 localization accuracy 的分数。

并提出 Varifocal Loss。

## 对本项目的意义

它说明：

> “让 score 同时反映分类和定位”这一概念已经非常成熟。

但它也给本项目一个重要提示：

> 最终用于 ranking/NMS 的分数最好和 localization quality 有明确关系。

## 本项目应该怎样升级问题

不是预测：

```text
quality(full proposal)
```

而是预测：

```text
quality_cls(full)
quality_loc(full)

quality_cls(peer_j)
quality_loc(peer_j)
```

然后决定：

```text
KEEP
SCORE_FROM_j
GEOMETRY_FROM_j
JOINT_FROM_j
```

---

# 5. IoU-aware calibration：甚至“是否需要传统 NMS”都已经有人研究

**论文**  
[Do We Still Need Non-Maximum Suppression? Accurate Confidence Estimates and Implicit Duplication Modeling With IoU-Aware Calibration](https://openaccess.thecvf.com/content/WACV2024/html/Gilg_Do_We_Still_Need_Non-Maximum_Suppression_Accurate_Confidence_Estimates_and_WACV_2024_paper.html)  
WACV 2024。

## 核心做法

用 IoU-aware calibration 建模：

- detection confidence；
- duplicate likelihood；

从而减少对传统手工 NMS 的依赖。

## 对本项目的警告

不能把 Stage-3C 的 NMS failure 简化成：

> “NMS 算法落后，我换一个 NMS 就行。”

Stage-3C 现象更早：

- score 可能已经错；
- geometry 可能已经错；
- fusion source 可能已经错；
- NMS 只是最后把错误显性暴露出来。

所以 NMS 更适合：

> **mechanism-specific evaluation / quality-aware postprocess baseline**

而不是研究主线。

---

# 6. Structure-Aware Single-Stage 3D Detection：confidence 和 box spatial alignment 早已有工作

**论文**  
Structure Aware Single-Stage 3D Object Detection From Point Cloud，CVPR 2020。

## 相关点

该类工作已经明确关注：

> predicted bounding box 与 classification confidence 的 discordance。

并通过结构信息和 feature warping 让 confidence 与 box 对齐。

## 创新边界

因此“空间对齐 classification 和 regression feature”也不是新的大方向。

---

# 7. Stage-3C 为什么仍然比这些论文多了一层

普通 detector 只有：

```text
single feature
→
cls
→
reg
```

本项目是：

```text
ego feature
peer feature(s)
full fused feature
        ↓
多个 source 对同一目标产生不同 score / geometry evidence
        ↓
full collaboration 可能破坏其中一部分
```

这带来至少四个普通 IoU-aware detector 没回答的问题。

## 7.1 classification 和 geometry 最优来源可能不同

例如：

```text
full:
  score = 好
  geometry = 差

peer:
  score = 一般
  geometry = 好
```

最佳动作可能是：

```text
full score + peer geometry
```

Stage-3C 已经通过分开替换证明：

> 这种属性级来源差异值得研究。

但注意：

> score+geometry 联合替换尚未正式执行，因此不能写成已验证事实。

---

## 7.2 修复一道障碍后可能撞上下一道障碍

Snow 中已经观察到：

```text
score-filtered
  ↓ score repair
NMS failure
```

以及：

```text
no good geometry
  ↓ geometry repair
score-filtered
```

这说明问题不是单一 head。

更像：

```text
geometry
↕
score
↕
ranking/NMS
```

组成连续约束。

所以方法应避免：

> 单独优化某个 score head，然后期待整个链条自然恢复。

---

## 7.3 full 不一定比单 source 在每个属性上都好

传统 fusion 隐含希望：

> 多来源融合后得到一个更好的综合表示。

但 Stage-3C 的反事实说明：

> full output 的某个属性可能比 peer output 更差。

因此可以定义：

```text
Δq_cls(j) = q_cls(peer_j) - q_cls(full)

Δq_loc(j) = q_loc(peer_j) - q_loc(full)
```

真正有用的不是：

```text
reliability(peer_j)
```

而是：

> **peer_j 相对于当前 full，在当前 proposal 的某个属性上是否提供增量价值。**

---

## 7.4 “什么都不改”必须显式学习

普通 IoU-aware detector 基本默认：

> quality correction 总是会被执行。

但 Stage-3C 中盲目 local score replacement 可以产生新 FP。

因此我们需要：

```text
KEEP_FULL
```

这个动作。

这是 repair routing 和普通 quality calibration 的重要差异。

---

# 8. 建议的 quality 表达

> 以下是设计建议，不是已经验证的方法。

对于 proposal p、source s：

## 8.1 分类质量

```text
q_cls(s,p)
```

它回答：

> 这个 source 对“这里确实有目标”的证据有多可信？

输入可以包括：

- cls logit；
- local foreground feature；
- source/full logit difference；
- source-specific density/reliability（辅助）；
- 与其他 source 的一致性。

## 8.2 定位质量

```text
q_loc(s,p)
```

它回答：

> 这个 source 预测的 box geometry 有多可信？

训练时可使用 GT IoU 监督。

推理时只输出预测值。

## 8.3 冲突量

```text
d_cls(s,full)
d_geo(s,full)
```

例如：

- score difference；
- center difference；
- yaw difference；
- size difference；
- predicted IoU disagreement。

## 8.4 NMS competition context

不要只看一个 proposal。

还要看附近：

- 是否有更高分 competitor；
- competitor geometry quality；
- cluster 内 score margin；
- box-box IoU；
- source disagreement。

因为 Stage-3C Fog/Rain 的核心失败往往发生在候选竞争环境中。

---

# 9. 不建议直接做的方案

## 9.1 只加 IoU head

问题：

- 创新弱；
- 只能告诉“当前 box 定位质量”；
- 不能告诉该用哪个 peer；
- 不能决定是否改 score / geometry / feature。

## 9.2 只把 score 改成 score × IoU

问题：

- Snow 大量 score 本身就受天气影响；
- full/peer source quality 不一致；
- 一刀切乘法可能进一步压低低分但几何正确的候选。

## 9.3 直接换 Soft-NMS / DI-NMS / learned NMS

问题：

- 它处理的是 candidate competition；
- Stage-3C 已经证明部分错误发生在 score/geometry/fusion 更上游；
- threshold loosening 还会造成大量 FP。

## 9.4 只训练“是否有害”二分类

问题：

一个 source 可能：

- score 有益；
- geometry 有害；

或者相反。

所以：

```text
helpful / harmful
```

仍然太粗。

---

# 10. 更适合本项目的科学问题

建议论文问题写成：

> **在恶劣天气下，多车协同会造成来源特定的分类质量与定位质量失配。如何识别这种局部失配，并选择最小、副作用受控的修复动作？**

对应方法不是：

```text
IoU-aware detector
```

而是：

```text
Source-conditioned quality estimation
+
Local repair routing
+
Safety-aware action supervision
```

---

# 11. 最小可实现版本

为了避免一开始做太复杂，可以只做三个动作：

```text
A0 = KEEP_FULL
A1 = OUTPUT_REPAIR
A2 = FEATURE_REPAIR
```

其中 A1 内部输出：

```text
Δscore
Δbox
```

并通过 source attention / source routing 决定参考哪个 peer。

这样可以同时覆盖：

- Snow score problem；
- Fog/Rain geometry/NMS problem；
- 少量 weight-only Snow case。

后续如果 A1 有明显上限，再拆成：

```text
score-only
geometry-only
joint
```

不建议第一版就堆五个 expert。

---

# 12. 论文实验上应该怎样证明不是普通 IoU-aware detector

至少需要以下对照：

1. baseline full fusion；
2. baseline + IoU/quality head；
3. baseline + quality-aware NMS；
4. baseline + proposal uncertainty gate；
5. 本项目 source-conditioned repair；
6. 去掉 source disagreement；
7. 去掉 KEEP_FULL；
8. score-only vs geometry-only vs joint；
9. 不使用 GSPR vs 使用 GSPR auxiliary；
10. Fog/Rain/Snow 分天气机制统计。

机制指标至少包括：

- 原 NMS failure recovery；
- score-filtered recovery；
- no-IoU decode recovery；
- 原 TP lost；
- new FP；
- recovered target 的 q_cls/q_loc 变化；
- action distribution；
- source distribution；
- safe/harmful repair precision。

---

# 13. 最终边界

### 已有工作已经解决

- classification score 与 localization quality 的一般失配；
- IoU-aware score；
- localization-aware ranking；
- quality-aware NMS/calibration。

### 本项目仍可研究

- 多 agent / full 之间的 source-conditioned quality disagreement；
- score evidence 与 geometry evidence 来源解耦；
- repair action，而不只是 rescoring；
- adverse-weather-specific failure；
- recovery 与 collateral harm 联合监督；
- output repair 和 feature repair 的层级选择。

这才是 Stage-3C 对 detector 文献真正增加的新信息。
