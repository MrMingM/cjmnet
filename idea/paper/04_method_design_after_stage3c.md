# Stage-3C 之后的方法设计空间

> 更新日期：2026-09-20  
> 状态：方法候选，不代表已经验证。  
> 目标：把 Stage-3C 的事实转成一个与现有论文有明确边界、并且能落地实现的研究方案。

---

## 0. 结论

当前最值得优先设计的不是“新 fusion backbone”，而是：

> **局部协同修复路由：对同一个目标候选，显式判断保持 full、做输出层修复，还是进入特征层修复。**

推荐第一版只保留三个动作：

```text
A0: KEEP_FULL
A1: OUTPUT_REPAIR
A2: FEATURE_REPAIR
```

其中 OUTPUT_REPAIR 不只是“调分数”，而是同时预测：

```text
score correction
box correction
source contribution
```

方法的核心科学问题是：

> **恶劣天气下，full collaboration 会在局部目标上造成来源特定的分类质量与定位质量失配；能否根据无 GT 的多来源不一致性，选择最小且副作用受控的修复动作？**

这比“再做一个 uncertainty gate / attention fusion / agent selector / IoU head”更能保住创新边界。

---

# 1. 方法设计必须服从的五条实验事实

## F1：局部比全局安全

Stage-3C：

- 64 个焦点目标；
- 全局可恢复 41；
- 局部可恢复 41；
- 全局零代价可恢复 8；
- 局部零代价可恢复 41。

因此方法默认作用尺度应该是：

> proposal / target-local region。

而不是：

> whole frame / whole peer / whole BEV feature map。

---

## F2：score 与 geometry 是两个不同问题

Fog/Rain：

> high-score but inaccurate competitor 是重要路径。

Snow：

> geometry 合格但 score 太低的情况很突出。

因此一个 scalar utility 不够。

至少需要：

```text
classification quality
localization quality
```

两个维度。

---

## F3：修一道障碍可能撞上下一道

Snow：

- 7 个目标 score repair 后从 score-filtered 进入 NMS；
- 3 个目标 geometry repair 后从 no-good-IoU 进入 score-filtered。

因此不能认为：

```text
score repair = 最终恢复
```

或：

```text
geometry repair = 最终恢复
```

真正的目标应该是：

> detection-chain consistent repair。

---

## F4：feature/fusion 仍有独有价值

Snow 有 weight-only recoverable target。

因此不能把方法收缩成：

> 永远只在 box head 后处理。

但 feature repair 应该：

> 少量触发、局部触发、作为 fallback。

---

## F5：盲目修复会伤害

Snow local score replacement：

- 有安全恢复；
- 也有大量 unrecovered + new FP。

而简单 score difference 又无法清晰分开 safe / harmful。

因此：

> KEEP_FULL 必须是显式动作，且训练必须有 harmful repair 负样本。

---

# 2. 推荐主方案：Local Collaborative Repair Routing

暂定简称：

> **LCRR — Local Collaborative Repair Routing**

名字只用于内部讨论，最终论文定名以后再查重。

---

# 3. 整体流程

```text
Multi-agent features
      ↓
original full fusion
      ↓
baseline detection head
      ↓
candidate proposal clusters
      ↓
source-conditioned evidence extraction
      ↓
quality/disagreement encoder
      ↓
repair router
      ├── KEEP_FULL
      ├── OUTPUT_REPAIR
      └── FEATURE_REPAIR
      ↓
final proposals
      ↓
quality-aware ranking / NMS
```

核心原则：

> 先保留原 baseline，repair 是局部附加动作，而不是重新改写整个 fusion pipeline。

这样更符合 Stage-3C 的“局部修复、少误伤”。

---

# 4. 第一步：构造无 GT 的局部候选

正式推理不能使用 GT region。

因此第一步不能沿用 Stage-3C：

> GT rectangle × 1.0 / 1.5。

建议从检测输出构造候选 cluster。

## 4.1 来源

保留：

- full fused head proposals；
- ego head proposals；
- 各 peer-aligned head proposals。

不要求每个 source 都跑完整 postprocess。

可以保留：

- top-K anchors/proposals；
- score threshold 使用较低开发阈值；
- 然后只做 local matching。

## 4.2 跨 source 匹配

对 proposal 做：

- BEV center distance；
- box IoU；
- heading difference；
- class consistency；

形成：

```text
proposal cluster C_k
```

一个 cluster 尽量表示：

> 多个 source 对同一潜在目标的不同看法。

可以借 CoopDETR/QUEST 的 query/proposal matching 思想，但不直接复制其 fusion。

---

# 5. 第二步：source-conditioned evidence

对 cluster 中每个 source s，提取两类信息。

## 5.1 输出层证据

```text
score_s
box_s
rank_s
predicted_q_loc_s
```

以及相对 full 的差：

```text
Δscore_s
Δcenter_s
Δsize_s
Δyaw_s
Δq_loc_s
```

## 5.2 局部特征证据

从对应 BEV region pooling：

```text
f_s^0
f_s^1
f_full^0
f_full^1
```

计算：

- cosine similarity；
- feature norm ratio；
- full-peer residual；
- channel statistics；
- local spatial disagreement。

## 5.3 sensing-side 辅助

可以加入：

- GSPR mean reliability；
- GSPR low-reliability ratio；
- local point count；
- distance；
- occlusion proxy。

但这些只能作为 context：

> 不作为 action label。

---

# 6. 第三步：显式分开 classification quality 与 localization quality

## 6.1 Classification quality head

预测：

```text
q_cls(s,k)
```

训练目标可以是：

- objectness / class correctness；
- 或 IoU-aware classification target；
- 但需要避免仅复制 VarifocalNet。

它的作用是：

> 给 router 判断“这个 source 的分类证据有没有增量”。

---

## 6.2 Localization quality head

预测：

```text
q_loc(s,k)
```

训练时可用 GT IoU。

正式推理只用预测。

作用：

> 判断哪个 source 的 box geometry 更可信。

---

## 6.3 为什么必须分开

不能用：

```text
q = α q_cls + β q_loc
```

然后结束。

因为真正的动作可能是：

```text
score ← source A
geometry ← source B/full
```

即使第一版不显式允许两个不同 peer，也应在建模上保留这种可能性。

---

# 7. 第四步：Repair Router

## 7.1 第一版动作

```text
A0 KEEP_FULL
A1 OUTPUT_REPAIR
A2 FEATURE_REPAIR
```

为什么第一版不拆成五六个 expert？

因为现有数据还没有证明：

> score-only / geometry-only / joint / scale0 / scale1 / scale0+1

全部都需要独立专家。

先把大层级做通。

---

## 7.2 OUTPUT_REPAIR

预测：

```text
Δlogit
Δbox
```

但 correction 必须 source-conditioned。

一个简单方案：

```text
α_s = softmax(router_source_score_s)

reference =
Σ_s α_s * source_evidence_s
```

然后：

```text
logit' = logit_full + g_cls(reference, full)
box'   = box_full   + g_loc(reference, full)
```

关键点：

> 不直接把 peer output 生硬替换进 full。

因为 Stage-3C 的盲替换已经展示 harmful case。

更适合做：

> bounded residual correction。

但这里要和历史 C-006 区分：

历史 C-006 是：

> 给 A0B0 communication ranking score 加小 residual，但没有改变 ranking。

这里 correction 的对象是：

> target-local detection output，且有明确 counterfactual action supervision。

不是同一层级。

---

# 8. FEATURE_REPAIR

只有 router 选择 A2 时才触发。

## 8.1 作用区域

以 proposal cluster 为中心得到局部 BEV crop。

而不是 GT box。

region 大小可以：

- 根据 predicted box 扩张；
- 固定安全 margin；
- 不继续扫 1.0/1.5/2.0 一大堆超参。

## 8.2 来源

只对高 router source score 的 1–2 个 peer 做局部融合。

## 8.3 尺度

当前 Stage-3C 证据最支持：

- scale 0；
- scale 1；
- 某些 case 需要 0+1。

因此第一版 feature repair 建议只处理：

```text
scale 0 + scale 1
```

不要扩大到所有尺度。

## 8.4 结构

不建议重新设计大型 Transformer。

可以先用：

- gated residual fusion；
- source attention；
- small cross-attention；

重点不是结构，而是：

> router 什么时候触发它。

---

# 9. 最关键部分：反事实动作监督

这是方法能不能和 INSTINCT / ICPB / CoRA 拉开距离的关键。

## 9.1 为什么普通 detection loss 不够

普通 loss 只能说：

> 最终预测对不对。

它不能直接告诉 router：

> 当前应该保持 full，还是执行哪种修复。

---

## 9.2 训练时构造 action teacher

只在训练/development 数据上，可以使用 GT 做局部 counterfactual。

对每个候选 action a：

```text
U(a)
=
w1 * focal_target_gain
- w2 * lost_original_TP
- w3 * new_FP
- w4 * other_target_degradation
```

其中：

### focal_target_gain

不要只用 binary recovered。

可以综合：

- matched IoU；
- classification quality；
-是否过 score threshold；
- final detection match。

### collateral penalty

必须包含：

- 原 TP 丢失；
- 新增 FP；
- 周围目标恶化。

这是 Stage-3C 最重要的经验之一。

---

## 9.3 KEEP_FULL 的 teacher

如果任何 repair：

- 收益小；
- 风险高；
- 或没有稳定优势，

teacher 就应该输出：

```text
KEEP_FULL
```

不能人为让数据集中 repair action 总是“正类”。

---

# 10. 与项目历史 learned selector 的本质区别

这一步必须写清，否则很容易重复旧坑。

| 历史 selector | 新候选 |
|---|---|
| agent / block selection | target/proposal local repair |
| 决定“选哪些信息” | 决定“怎样修当前输出” |
| 单块 loss gain / utility proxy | action-specific final detection + collateral cost |
| top-k ranking | keep/repair categorical routing |
| 大范围替换通信集合 | 局部 bounded correction |
| 没有显式 score/geometry 分解 | q_cls/q_loc 分解 |
| domain transfer 较差 | weather context + task disagreement jointly建模 |
| harmful action 控制弱 | KEEP_FULL + harmful repair negatives |

因此这不是：

> “再训练一个更强 selector”。

而是：

> **换了科学问题、干预位置、动作空间和监督。**

---

# 11. 与外部论文的本质区别

## vs INSTINCT

INSTINCT：

```text
collaboration relevant?
→
local instance fusion
```

本项目：

```text
collaboration 后当前 proposal 哪里坏？
→
keep / output repair / feature repair
```

## vs ICPB

ICPB：

```text
uncertainty
→
individual/collaborative expert balance
```

本项目：

```text
source-conditioned score/geometry disagreement
+
local feature state
→
repair action
```

## vs CoRA

CoRA：

```text
feature fusion branch
+
object correction branch
```

本项目：

```text
conditional routing
→
只在必要区域执行 correction
```

## vs CIA-SSD / VarifocalNet

它们：

```text
single-source classification-localization alignment
```

本项目：

```text
multi-source classification-localization attribution and repair
```

---

# 12. 三个方法版本

## Version A：最小版本，优先推荐

### Local Output Repair

动作：

```text
KEEP
OUTPUT_REPAIR
```

只做：

- source-conditioned q_cls / q_loc；
- score residual；
- box residual；
- safety gate。

### 优点

- 最贴近 Fog/Rain score-geometry 证据；
- 实现成本低；
- 不动 fusion backbone；
- 很容易和 IoU-head baseline 比较。

### 缺点

- Snow weight-only case 可能无能为力。

### 使用条件

如果 Cheap Gate 已能在 Fog/Rain/Snow 显著改善且 FP 可控：

> 先把它做扎实，不急着加 feature repair。

---

## Version B：推荐完整版本

### Hierarchical Output-to-Feature Repair

```text
KEEP
OUTPUT_REPAIR
FEATURE_REPAIR
```

Output repair 解决多数 target。

只有：

- output quality 都低；
- source disagreement 大；
- router 预测 output repair 无法恢复；

才触发 feature repair。

### 优点

- 覆盖 Snow weight-only evidence；
- 计算集中在少量难例；
- 与“全图 adaptive fusion”有明确区别。

### 缺点

- action teacher 更复杂；
- 需要设计 feature-repair label。

---

## Version C：高风险，不优先

### Full Action MoE

动作：

```text
KEEP
SCORE_FROM_peer
GEOMETRY_FROM_peer
JOINT_FROM_peer
FEATURE_SCALE0
FEATURE_SCALE1
FEATURE_SCALE01
...
```

### 问题

- action space 太大；
- 与 ICPB / MoE 文献更接近；
- teacher label 稀疏；
- 容易把研究问题变成“训练 router 的工程”。

因此不建议一开始做。

---

# 13. Cheap Gate 设计

用户当前决定停止扩展诊断，不代表新方法不做 Cheap Gate。

区别是：

> 以后实验必须直接服务于方法可行性，而不是继续细分旧失败。

## Gate-1：q_cls / q_loc 是否真的可预测

在固定训练/开发集：

- q_loc 与真实 IoU 的相关；
- q_cls 与 true detection quality 的区分；
- 跨 Fog/Rain/Snow；
- 不要只看训练集。

Kill：

> 如果 q_cls/q_loc 都不能明显优于 raw score /简单差值，就不要训练大 router。

---

## Gate-2：repair action 是否可预测

只做：

```text
KEEP vs OUTPUT_REPAIR
```

64–128 帧即可。

看：

- safe repair precision；
- recovered targets；
- original TP loss；
- new FP。

Kill：

> 如果 router 不能明显区分 safe vs harmful repair，不扩 feature branch。

---

## Gate-3：feature repair 是否提供独立收益

仅对：

> output repair teacher 也失败的样本

测试 A2。

Kill：

> 如果 feature repair 独有增量很低，就停止，不为了 Snow 两个 case 做大模块。

---

# 14. 训练数据不能怎么做

## 不要

1. 在 OPV2V-W test 上生成 action label 再训练；
2. 用 Stage-3C 64 个 focal case 直接训练并宣称泛化；
3. 只采正向 recoverable case；
4. 忽略普通成功区域/背景；
5. 把 GT region 当推理输入；
6. 用单一 weather label 作为捷径；
7. 用单 block loss gain 当 action utility。

## 应该

训练 action teacher 时包含：

- baseline 正常成功 proposal；
- harmful repair；
- no-op；
- source-valid/full-miss；
- ordinary FP；
-不同距离/遮挡/天气。

---

# 15. 最重要的 baseline

如果以后真正做方法，baseline 至少需要：

### Detection quality baselines

- raw classification score；
- IoU head；
- score × IoU；
- quality-aware rescore/NMS。

### Collaborative routing baselines

- full fusion；
- ego；
- best peer confidence；
- uncertainty gate；
- INSTINCT-style relevance binary；
- proposal-wise simple MoE；
- simple source attention。

### Project historical baselines

- Agent Selector；
- spatial utility；
- GSPR reliability gate；
- full Where2comm。

这样才能证明：

> 新收益不是简单 IoU calibration、uncertainty gate 或 old selector 重新出现。

---

# 16. 论文机制分析应该提前设计

如果方法有效，不要只报 AP。

至少报告：

## Action 分布

```text
Fog:
  keep %
  output %
  feature %

Rain:
...

Snow:
...
```

预期不应该人为强制一致。

## Failure-stage recovery

- score-filtered；
- NMS-suppressed；
- no_iou70_after_decode。

## Safety

- lost original TP；
- new FP；
- safe repair precision。

## Source attribution

- ego/full/peer 哪一来源提供 score correction；
- 哪一来源提供 geometry correction。

## Quality alignment

- repaired score 与 IoU 的关系；
- competitor ranking 是否改善。

---

# 17. 当前最可能形成的论文叙事

```text
Adverse weather degrades not only sensing quality,
but also the relative usefulness of collaborative sources.

Task-usable peer evidence can exist while full collaboration loses the target.

Counterfactual intervention reveals that:
  (1) repair is local;
  (2) classification and localization failures differ;
  (3) feature-level repair remains necessary for a subset;
  (4) blind repair causes collateral harm.

Therefore, collaboration should not be treated as a binary
"use peer / do not use peer" problem.

We formulate local collaborative recovery as
a safety-aware repair routing problem.

The model estimates source-conditioned classification/localization quality,
then selects keep, output repair, or local feature repair.
```

这条叙事目前比：

> weather denoising + attention fusion

更有研究辨识度。

---

# 18. 当前明确不能写的 claim

- “首次实例级协同感知”
- “首次 proposal-level routing”
- “首次 mixture-of-experts collaborative perception”
- “首次 uncertainty-aware collaborative fusion”
- “首次 IoU-aware confidence”
- “首次 hybrid intermediate-late fusion”
- “首次 local feature fusion”
- “首次天气鲁棒 collaborative perception”
- “首次根据需求区域请求 peer”
- “attention/query 被证明是唯一失败根因”
- “score+geometry joint repair 已验证”

---

# 19. 当前最值得验证的三个问题

以后方法开发只围绕：

### Q1

> 无 GT 的 source-specific score/geometry disagreement 能否预测哪个 local output repair 是安全且有益的？

### Q2

> 将 KEEP_FULL 显式加入 action space，能否显著降低 blind repair 的 FP / TP collateral damage？

### Q3

> 在 output repair 已失败的 target 中，local feature repair 是否仍有足够独立 headroom，尤其是 Snow？

如果这三个问题都能回答“是”，就有希望形成完整方法。

---

# 20. 最终建议

当前优先级：

```text
1. 精读 INSTINCT / ICPB / CoRA / CIA-SSD / VarifocalNet
2. 定死 Version A 的输入、输出、teacher label
3. 做最小 action-label dataset
4. 先验证 KEEP vs OUTPUT_REPAIR
5. 只有 output 修复确实有上限，再加 FEATURE_REPAIR
```

不要再从：

> “换一个新的 fusion module”

开始。
