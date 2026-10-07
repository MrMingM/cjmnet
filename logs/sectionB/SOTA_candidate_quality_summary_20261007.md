# Section B：候选质量 / 重排序代表性方法阶段汇总（2026-10-07）

## 1. 研究问题

本阶段不再继续提出新的通用质量头，而是回答一个更基础的问题：在冻结当前 F 检测器、固定 decoded box geometry 和候选池的前提下，已有候选质量估计 / 重排序方法到底能追回多少 GT-IoU Oracle 的空间？

主评价以 **frame-order AP70** 为准。global-sort AP 仅作质量排序诊断，不能与 frame-order AP 混用。GT-IoU Oracle 只用于量化固定候选池的上界，推理时不可用。

公共思想是：尽量冻结 GSPR / PointPillar / Where2comm / F 融合 / 原分类与回归 / decoded boxes，仅修改候选的质量估计、排序或 NMS 决策。各实现均属于对论文核心机制的项目适配，不应统一表述为原论文完整 detector 的严格复现。

## 2. 协议边界：不能把四组绝对 AP 当成同一排行榜

| 方法 | 开发数据 | 天气协议 | 主要边界 |
|---|---|---|---|
| CIA-SSD-style IoU quality | 旧开发子集（与 90 帧/条件 F pilot 同口径） | 在线物理天气 | 固定 top256 + 原 rotated NMS；只移植 IoU-aware score，不移植完整 DI-NMS |
| Adapted LMD-core | 完整 OPV2V validation，1980 帧/条件 | Clean + 在线 Fog/Rain/Snow | 冻结 F/top256/geometry/NMS；meta-regression 预测 GT BEV IoU |
| SAQC | 完整 OPV2V validation，1980 帧/条件 | 本次实际 checkpoint 为 online_legacy；Clean 用 clean branch，Fog/Rain/Snow 用 weather branch | anchor-based 适配；7×7 局部 BEV patch 预测质量，`s'=s*q^1.5` |
| Learned 3D NMS | 完整 OPV2V validation | 固定离线物理天气 `/data/cjm/datasets/opv2v-physics-fixed-v1` | D2D-Rescore / GossipNet3D；固定 F/top256；分别测试 native Top-K 与原 rotated NMS |

因此：

- **LMD 与本次 SAQC 的 baseline 完全一致，可直接比较。**
- CIA-SSD 使用旧开发子集，不能用绝对 AP 与 1980 帧结果排名。
- Learned 3D NMS 使用固定离线天气 realization，与 LMD/SAQC 的在线天气不是同一输入 realization，只比较趋势，不作严格绝对排名。
- 本页全部属于 development / validation 基准；**不是统一的 OPV2V clean test + OPV2V-W Fog/Rain/Snow 正式测试主表。**

---

## 3. CIA-SSD-style IoU quality

### 3.1 实现过程

- 冻结 F、decoded boxes 和固定 top256 候选池。
- 只训练 CIA-SSD 风格 IoU quality head。
- 参考原作者当前生效实现采用 3×3 IoU branch；正 anchor 用匹配 GT 的 3D IoU 监督，target 为 `2*IoU3D-1`。
- 主分数使用 CIA 式 `classification_score * predicted_IoU^4`。
- 为保持固定候选几何和统一 NMS，本实验 **不移植完整 DI-NMS**，所以应称为 `CIA-SSD-style score benchmark`，不是完整 CIA-SSD detector 复现。
- official train 内按 scene 留出 20% 做 epoch 选择；development validation 不参与模型选择。

### 3.2 结果（frame-order AP70）

| 条件 | Original F | CIA score | GT-IoU Oracle | AP70 变化 | Oracle recovery |
|---|---:|---:|---:|---:|---:|
| Clean | 0.8223 | 0.8275 | 0.9354 | +0.52 pp | 4.63% |
| Fog | 0.7858 | 0.7876 | 0.9075 | +0.18 pp | 1.49% |
| Rain | 0.8127 | 0.8128 | 0.9315 | +0.01 pp | 0.13% |
| Snow | 0.6012 | 0.6116 | 0.8067 | +1.04 pp | 5.06% |

CIA score 的 score–GT-IoU Spearman 为 Clean/Fog/Rain/Snow = 0.7126 / 0.7229 / 0.7173 / 0.6500；IoU≥0.7 ROC-AUC = 0.8945 / 0.9022 / 0.8963 / 0.8822。

额外诊断很重要：直接 IoU regression 的 Spearman 更高（0.8733 / 0.8735 / 0.8756 / 0.8362），但 frame-order AP70 并不稳定，Clean 和 Rain 反而下降；只用 predicted IoU 排序更差。这说明“预测 IoU 更准”本身不能保证最终 AP 更高。

---

## 4. Adapted LMD-core

### 4.1 实现过程

- 完整 official OPV2V train、stride=1；冻结同一个 F。
- Clean + 在线物理 Fog/Rain/Snow 全部扫描；四条件训练一个共享 LMD meta model。
- train 内按 scene 留出 20% 做模型选择，同一 scene 的天气版本不跨侧。
- 完整 pre-NMS top256 候选很多，默认每个天气 reservoir sample 最多 50,000 个候选训练 meta model；检测前向仍扫描完整 train。
- 适配版主方法为 meta-regression：使用候选几何、分数和重叠 proposal 离散度预测真实 GT BEV IoU，再用该质量分数排序并走原 rotated NMS。
- 这是 adapted LMD-core，不是完整原版 LidarMetaDetect 特征体系。

### 4.2 结果（完整 1980 帧/条件，frame-order AP70）

| 条件 | Original F | LMD | GT-IoU Oracle | AP70 变化 | Oracle recovery |
|---|---:|---:|---:|---:|---:|
| Clean | 0.755802 | 0.749301 | 0.919738 | -0.650 pp | -3.97% |
| Fog | 0.726578 | 0.716814 | 0.890370 | -0.976 pp | -5.96% |
| Rain | 0.745695 | 0.738408 | 0.915273 | -0.729 pp | -4.30% |
| Snow | 0.629952 | 0.635059 | 0.835187 | +0.511 pp | +2.49% |

三种恶劣天气平均 AP70 变化约 **-0.398 pp**。

但 LMD 的质量预测本身非常强：score–GT-IoU Spearman = 0.8963 / 0.8957 / 0.8951 / 0.8742；IoU≥0.7 ROC-AUC = 0.9714 / 0.9664 / 0.9703 / 0.9574；PR-AUC = 0.9676 / 0.9592 / 0.9658 / 0.9410。

同时，LMD 相对 Original F 的 recovered / lost / new-FP 数为：Clean 575 / 826 / 6751；Fog 602 / 952 / 6083；Rain 611 / 873 / 6897；Snow 1246 / 1106 / 6789。它能救回大量 GT，但同时改变排序后损失更多原有正确结果并引入很多 FP，所以总体 AP 无法兑现。

**LMD 给出的核心证据：候选 IoU 可以预测得非常准，但把单候选质量直接替换成最终排序分数，仍然可能让检测结果更差。**

---

## 5. SAQC

### 5.1 实现与训练过程

本项目对原 center-based SAQC 做 anchor-based 适配：

1. 使用冻结 F 检测头真正消费的 fused BEV feature；
2. 将 decoded box `(x,y)` 中心映射到冻结 anchor BEV grid 的最近 cell；
3. 以该 cell 为中心裁 7×7 局部 BEV patch；
4. 加入 2 个相对坐标通道；
5. 两层 3×3 Conv + ReLU + average pooling 输出定位质量 q；
6. Smooth-L1 拟合 GT BEV IoU；
7. 主排序分数为 `s' = s * q^1.5`，再使用原 rotated NMS 和原输出预算。

本次实际完成的长训练属于 **旧 online-weather checkpoint**，不要被后来仓库 README 的 fixed-weather 版本覆盖解释：

- checkpoint：`/data/cjm/datasets/logs/fusion_detector_adaptation_20260927_103227/saqc_full_train_20260929_210426/saqc_quality.pth`
- 3 epochs；每 epoch 25,496 帧（6374 × Clean/Fog/Rain/Snow）
- epoch 3：843,848 个 quality samples
- mean loss = 0.00161195
- mean decoded-anchor same-cell ≈ 0.4291
- mean decoded-to-nearest-grid distance ≈ 0.3173
- optimizer updates = 76,488

开发评价过程中出现过两次工程问题：

- 新版 evaluator 一度要求 fixed-weather manifest，而老 checkpoint 没有该字段；已改为按 checkpoint 自动选择 online_legacy / fixed 协议。
- 更严重的是 evaluator 曾把 `branch='clean'` 写死，导致 Clean/Fog/Rain/Snow 四组结果几乎完全相同；该首轮结果 **作废**。修复后最终结果明确为 Clean `input_branch=clean`、Fog/Rain/Snow `input_branch=weather`，`logs/sectionB/SAQR/saqc_results.json` 为有效最终 development 结果。

### 5.2 最终结果（完整 1980 帧/条件，frame-order AP70）

| 条件 | Original F | SAQC top256 | GT-IoU Oracle | AP70 变化 | Oracle recovery |
|---|---:|---:|---:|---:|---:|
| Clean | 0.755802 | 0.756177 | 0.860443 | +0.038 pp | 0.36% |
| Fog | 0.726578 | 0.727038 | 0.848260 | +0.046 pp | 0.38% |
| Rain | 0.745695 | 0.746496 | 0.855756 | +0.080 pp | 0.73% |
| Snow | 0.629952 | 0.632846 | 0.801550 | +0.289 pp | 1.69% |

三种恶劣天气平均 AP70 仅约 **+0.138 pp**，平均只追回约 **0.93%** 的 frame-order Oracle 空间。

SAQC 单独预测质量 q 与 GT-IoU 的 Spearman = 0.8339 / 0.8233 / 0.8310 / 0.7926；融合后的最终 SAQC score Spearman = 0.7118 / 0.7161 / 0.7081 / 0.7101。说明模型确实学会了定位质量，但 `score × quality` 后真正用于 NMS/排序的信号只保留了一部分价值。

未做 Platt calibration 时，Q-ECE 从 raw score 的 0.2425 / 0.2501 / 0.2406 / 0.2289 变为 SAQC score 的 0.2992 / 0.3066 / 0.2974 / 0.2891，数值校准反而更差。

**SAQC 的核心证据：质量预测成功，但 quality → score fusion → NMS 的决策转换只兑现了不到 2% 的 frame-order Oracle 空间。**

---

## 6. Learned 3D NMS：D2D-Rescore / GossipNet3D

### 6.1 协议

- 冻结 F、decoded geometry 和 top256 候选。
- 使用固定数据集 `/data/cjm/datasets/opv2v-physics-fixed-v1`：Clean official train/validate + 固定 Fog/Rain/Snow train/validate；在线天气关闭。
- 每个结构训练一个跨天气共享模型。
- D2D-Rescore：全局候选 self-attention（默认 6 层 / 64 通道 / 4 heads）。
- GossipNet3D：5m 范围内局部候选消息传递（4 层 / 64 通道）。
- 同时报告 native Top-K 和 controlled original rotated-NMS；以下主表优先用 `original_nms` 路径，以保持与固定 NMS benchmark 更接近。
- 当前作者官方仓库不可访问，因此这是基于论文机制的项目实现，不能声称官方源码逐行复现。

### 6.2 D2D-Rescore（original NMS 路径）

| 条件 | Original F | D2D | GT-IoU Oracle | AP70 变化 | 按 frame AP70 计算的 Oracle recovery |
|---|---:|---:|---:|---:|---:|
| Clean | 0.755802 | 0.751024 | 0.919738 | -0.478 pp | -2.91% |
| Fog | 0.726541 | 0.721620 | 0.890617 | -0.492 pp | -3.00% |
| Rain | 0.747099 | 0.741817 | 0.915209 | -0.528 pp | -3.14% |
| Snow | 0.624367 | 0.630299 | 0.831605 | +0.593 pp | +2.86% |

恶劣天气平均 AP70 变化约 **-0.142 pp**。D2D 全局关系建模未形成跨天气稳定收益。

### 6.3 GossipNet3D（original NMS 路径）

`validation_gossip_fast_results.md` 内方法字段仍沿用了 `d2d_topk / d2d_original_nms` 名称，这是输出命名残留；该文件对应的实际模型是 GossipNet3D。

| 条件 | Original F | Gossip | GT-IoU Oracle | AP70 变化 | 按 frame AP70 计算的 Oracle recovery |
|---|---:|---:|---:|---:|---:|
| Clean | 0.755802 | 0.756897 | 0.919738 | +0.110 pp | +0.67% |
| Fog | 0.726541 | 0.725943 | 0.890617 | -0.060 pp | -0.36% |
| Rain | 0.747099 | 0.748264 | 0.915209 | +0.117 pp | +0.69% |
| Snow | 0.624367 | 0.643101 | 0.831605 | **+1.873 pp** | **+9.04%** |

三种恶劣天气平均 AP70 变化约 **+0.643 pp**，但收益高度集中在 Snow；Fog 仍有轻微下降。

这组结果是目前最明确的“候选之间关系建模比单框质量回归更有潜力”的信号，但它尚未证明跨天气通用稳定提升。

---

## 7. 横向汇总（只看 frame-order AP70 变化；不同协议不能当严格排行榜）

| 方法 | Clean ΔAP70 | Fog ΔAP70 | Rain ΔAP70 | Snow ΔAP70 | 恶劣天气平均 ΔAP70 | 协议备注 |
|---|---:|---:|---:|---:|---:|---|
| CIA-SSD-style score | +0.52 pp | +0.18 pp | +0.01 pp | +1.04 pp | +0.41 pp | 旧开发子集；不可与 1980 帧绝对排名 |
| LMD meta-regression | -0.650 pp | -0.976 pp | -0.729 pp | +0.511 pp | -0.398 pp | 1980 帧；online weather |
| SAQC | +0.038 pp | +0.046 pp | +0.080 pp | +0.289 pp | +0.138 pp | 1980 帧；本次实际 online_legacy |
| D2D-Rescore + original NMS | -0.478 pp | -0.492 pp | -0.528 pp | +0.593 pp | -0.142 pp | 1980 帧；fixed physics weather |
| GossipNet3D + original NMS | +0.110 pp | -0.060 pp | +0.117 pp | **+1.873 pp** | **+0.643 pp** | 1980 帧；fixed physics weather |

该表只能用于看“方向趋势”。尤其 CIA 和 Learned3D 的数据协议与 LMD/SAQC 不同，不能据此宣称谁是统一 SOTA 排名第一。

---

## 8. 阶段结论

### 8.1 最重要的事实：质量不是完全学不会，而是难以转成正确决策

LMD 在四条件的 GT-IoU Spearman 约 0.87–0.90，SAQC 的 predicted quality Spearman 约 0.79–0.83，CIA 的 direct IoU regression 约 0.84–0.88。也就是说，现有方法已经能相当准确地判断“哪个候选框定位得更好”。

但 frame-order AP 的 Oracle recovery 大多数仍接近 0、为负，或者只在 Snow 有几百分点。说明当前主要瓶颈不能再简单概括成“缺少更准的 IoU head”。

更准确的表述是：

> **候选定位质量可以较准确估计，但如何把质量信息转化为候选竞争、NMS、保留/删除决策，同时不损失已有 TP、不大量引入 FP，仍是主要难点。**

LMD 是最强的反例：质量预测非常准，但 Clean/Fog/Rain 的 AP70 均明显下降，因为重排序同时带来大量 lost GT 和 new FP。

### 8.2 Snow 是最有可利用空间的条件

CIA、LMD、SAQC、D2D、Gossip 几乎都在 Snow 上表现出比 Fog/Rain 更明显的正向空间。GossipNet3D 在 Snow 上达到 +1.873 pp，约追回 9.04% 的 frame Oracle gap，是本阶段最明显的可部署信号。

但这不能直接外推成“只做 Snow 方法”：Fog/Rain 仍是论文需要覆盖的恶劣条件，且 Gossip 在 Fog 仍略降。

### 8.3 候选关系 / 竞争建模比继续堆单候选质量头更值得研究

单候选方法（CIA-style、LMD、SAQC）共同暴露出 quality-to-decision 的转换问题；局部候选关系模型 Gossip 至少在 Snow 上明显比独立质量头更能兑现 Oracle。这与此前 H16/NMS 诊断中“真实单个 suppressor inversion 很稀疏”并不矛盾：有效关系机制可能不是只找一个错误 suppressor，而是需要在一组候选中联合决定谁应保留、谁应降权。

### 8.4 与方向 B 的关系

第 33/36 节的 source/task Oracle 已显示：逐目标、逐任务选择分类/回归信息源仍保留显著上界。这里的 SOTA benchmark 又显示：仅在检测末端给每个候选独立预测一个质量分数，通常无法兑现大 Oracle 空间。

两者共同支持下一步不要继续做“第五个通用 IoU/quality head”，而应优先研究：

- target-conditioned（逐目标）决策；
- classification / regression 任务分离；
- candidate competition-aware（候选竞争感知）决策；
- 必要时将 source selection 与候选关系决策结合，而不是只做一个独立标量可靠性分数。

---

## 9. 当前决策边界

1. **代表性候选质量 / 重排序方法的 development benchmark 阶段视为完成。** 不再因为某个方法只有 +0.x pp 就继续在同一 validation 上大规模调 beta、hidden size 或阈值。
2. 不能宣称“所有 SOTA 都失败”。严谨表述应是：**已测试的代表性独立质量估计 / 重排序方法只能追回固定几何 Oracle 的一小部分，而且收益高度依赖天气；当前没有方法在所有恶劣条件下稳定追回大比例 Oracle。**
3. 当前结果不能作为统一正式 OPV2V-W SOTA 排名。如果论文最终需要一张严格方法对照主表，应先冻结所有方法，再用相同正式协议跑一次 `OPV2V clean test + OPV2V-W Fog/Rain/Snow test`，关闭在线天气增强，使用非 global-sort AP。
4. 下一阶段方法设计优先回到“target/task/source/competition-aware decision”，而不是继续添加通用单候选质量预测器。
