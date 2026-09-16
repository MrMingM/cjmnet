# Q-A：恶劣天气下“观测不足”竞争假设研究地图

> 更新日期：2026-09-16  
> 本文件只执行 `idea/00_research_workflow.md` 的第 1 阶段：**现象 → 竞争假设 → 可证伪预测 → 最低成本验证顺序**。  
> 当前只研究 Q-A，不设计新网络，不进入通信选择、CEIF 方法设计或完整训练。

---

## 0. 研究问题与修订说明

### 0.1 总研究方向

本项目总方向始终是：

> **恶劣天气下协同感知鲁棒性。**

当前研究 Q-A：

> **在没有 GT、没有 clean frame 的实际推理阶段，自车如何判断一个局部区域是“观测不足 / 未充分观测”，还是“已经充分观测，并且有理由相信该区域为空”？**

这里先研究问题本身是否成立、主要瓶颈在哪里；不预设“需求图一定是最终解法”。

### 0.2 本轮审查后对旧六假设的处理

| 旧假设 | 处理 | 原因 |
|---|---|---|
| 旧 H-A1：可靠证据数量不足 | **保留并收紧定义** | 它可以作为 quantity-dominant 解释，但必须先做相关性/条件独立性门禁 |
| 旧 H-A2：任务相关语义证据 | **保留，但改成真正与 H-A1 冲突的 structural hypothesis** | 不能再只是“框内可靠点数”这种更细的计数版本 |
| 旧 H-A3：visibility/free-space | **保留，成本上调** | 对“未观测 vs 可信为空”有独立含义，但 ray analysis 工程成本和仿真外推限制之前被低估 |
| 旧 H-A4：需要帮助要看邻车是否有证据 | **从竞争假设中删除，降级为 Oracle/可恢复性标签** | 邻车证据不能在通信前直接作为 ego 本地需求判据，存在因果/部署顺序问题 |
| 旧 H-A5：representation/detector failure | **保留并拆清层级** | “模型漏检 ≠ sensing 没看到”是 Q-A 必须面对的竞争解释 |
| 旧 H-A6：不同天气机制不同 | **从独立假设中删除，降级为 moderator/分层变量** | 如果只是把其它实验按 Fog/Rain/Snow 分层，它没有独立因果内容 |
| 缺失项：融合端瓶颈 | **新增独立竞争假设** | full communication 也可能没有充分利用邻车正确证据，Q-A 不能默认“识别需求后问题就解决” |
| 缺失项：遮挡 × 天气交互 | **新增独立竞争假设** | 天气损伤可能强烈依赖目标原本的可见性/遮挡状态 |

修订后的体系仍保留 **6 个竞争假设**，但含义已经改变。

---

## 1. 当前必须先区分的两类问题

Q-A 其实包含两个不同诊断任务，不能混成一个标签。

### Q-A+：目标确实存在时，自车是否发生 sensing evidence insufficiency？

离线研究阶段可以用 GT 框定义目标，分析：

- 目标表面是否仍有足够回波；
- 天气是否导致支持证据下降；
- 遮挡是否放大天气损伤；
- raw evidence 是否存在但被后续网络丢失；
- 邻车证据是否存在但 full fusion 仍未恢复。

### Q-A−：区域没有目标时，这是“可信为空”还是“未知/未充分观测”？

这个问题不能只靠 GT=background 定义，因为推理阶段没有 GT。

它需要研究：

- 该空间是否真正被 LiDAR 覆盖；
- 是否存在射线穿越/free-space support；
- 是否被遮挡；
- 是否因为天气/距离导致没有返回。

**H-A1/H-A2/H-A4/H-A5/H-A6 主要先在 Q-A+ 上诊断；H-A3 是 Q-A− 最核心的独立假设，同时也能辅助 Q-A+。**

---

## 2. 已确认事实与尚不能声称的事实

### F1：GSPR 点级 reliability 有效，但不是 observation-sufficiency 标签

【事实】GSPR 能较稳定地区分天气污染点，并在 Fog/Rain/Snow 下改善检测。

【事实】low-r / high-u 区域直接硬过滤并没有形成稳定跨天气收益。

因此：

> `point reliability` 有 sensing 含义，但不能直接写成 `region needs help`。

### F2：空区域没有 GSPR 点级意见

GSPR 只对已有点输出 reliability。0 回波区域既可能为空，也可能是遮挡、天气衰减、距离/角分辨率不足或未被有效采样。

因此：

> **absence of return ≠ evidence of absence。**

### F3：当前没有 `区域总 N_eff` 与 `GT 框内可靠点数` 的 Pearson 统计

【事实】仓库当前没有逐目标配对表，也没有预计算 Pearson/Spearman 结果。

因此现在**不能声称相关系数大约是 0.8、0.9 或任何具体数值**。

这个统计被提升为本阶段第一个门禁实验 E-A0。

如果 Pearson 和 Spearman 都 > 0.85，则“区域总 N_eff”和“GT 框内可靠点数”在当前数据上应被视为高度冗余的 quantity family，不能拿来分别支撑 H-A1/H-A2。

### F4：模型漏检不等于 sensing 没看到

最终 miss 可能发生在：

1. raw LiDAR evidence 已经缺失；
2. raw evidence 仍在，但 PillarVFE/backbone 表征丢失；
3. detector/head 没有利用已有 evidence；
4. 邻车 evidence 已到达，但 fusion 抑制/稀释；
5. score/NMS/定位使目标在 AP 评测中表现为 miss。

### F5：现有协同实验要求我们把 fusion bottleneck 纳入竞争解释

【事实】learned matching 通信已经大幅改变选块（约 59%–70%），但正式天气 AP 仍低于 A0B0；因此“模型没改变消息”不能解释后期失败。

【事实】CEIF no-GT rule 的四条件干预均降低 AP，说明“拿到一个看似合理的额外证据并进行干预”不自动带来收益。

这些事实**不能单独证明 AttFuse full 一定存在融合失败**，但足以否定一个默认前提：

> “只要正确识别哪里缺信息并把邻车信息送过来，融合端自然会正确利用。”

因此 fusion bottleneck 必须成为独立竞争解释，而不是后续才考虑。

---

# 3. 修订后的六个竞争假设

## H-A1：Quantity-dominant —— 可靠任务证据的“量”主要决定观测充分性

### 假设内容

恶劣天气下，真正的 sensing insufficiency 主要来自有效回波支持量下降。

这里的 quantity family 可以包括：

- raw point count；
- reliable point count；
- `N_eff = Σ r^p`；
- GT 框内可靠点数/可靠质量。

但这些只是同一类解释的不同 proxy，不再人为拆成多个假设。

### 真正的可证伪预测

如果 H-A1 成立：

1. 控制 distance、target size、view angle、occlusion 后，quantity 指标仍应强预测 weather-induced sensing failure；
2. 同一 GT 目标 clean→weather 退化时，quantity 应同步下降；
3. **在 quantity 被匹配/控制后**，空间结构、表面覆盖、局部语义响应只能带来很小的额外判别增益；
4. 高 quantity 但真正 sensing evidence insufficiency 的样本应很少。

### 与 H-A2 的冲突点

- H-A1 预测：`P(failure | quantity, structure) ≈ P(failure | quantity)`；
- H-A2 预测：在 quantity 相同条件下，structure/semantic evidence 仍能显著改变 failure probability。

如果二者只是在比较“总 N_eff”和“框内可靠点数”，则不构成竞争关系，必须合并回 H-A1。

### Kill / downgrade criterion

若在 quantity-matched 样本中，结构变量仍稳定解释大量成功/失败差异，则 H-A1 从“主机制”降级为基础协变量。

---

## H-A2：Structure-dominant —— 同样数量的可靠点，空间/表面结构决定是否足够支撑任务

### 假设内容

观测充分性不是“多少可靠点”，而是这些点是否落在**对目标检测有判别价值的表面和空间结构**上。

H-A2 不再使用“GT 框内点数”作为核心区别，因为那仍然只是计数。

真正要测试的是：

- 目标表面覆盖是否完整；
- 点是否只集中在极小局部；
- 几何轮廓/边缘是否保留；
- pillar occupancy pattern 是否形成目标结构；
- frozen local feature/proposal 是否存在明确 target-supporting response。

### 真正的可证伪预测

如果 H-A2 成立：

1. 匹配 `N_eff`、距离、尺寸、遮挡等级后，成功检测与失败目标仍有明显结构差异；
2. 少量但覆盖关键目标表面的点可以优于大量集中/背景点；
3. structure features 对 quantity-only baseline 有稳定的条件增量；
4. clean→weather 中，即使 `N_eff` 变化不大，只要关键表面覆盖被破坏，miss 风险也明显升高。

### Kill / downgrade criterion

若 quantity matching 后，结构/语义指标在跨场景验证中几乎无额外信息，则 H-A2 降级，不再投入复杂结构建模。

---

## H-A3：Visibility/free-space —— “未观测”和“可信为空”的区别主要来自可见性与射线证据

### 假设内容

对 Q-A− 来说，0 点本身没有语义。

真正的区别是：

> **该空间是否被传感器有效覆盖，并且是否有足够几何证据支持“射线经过这里但没有目标”。**

因此 free-space / unknown 的区分需要 visibility 信息，而不是只看 density/reliability。

### 可证伪预测

如果 H-A3 成立：

1. point-count≈0 的背景区域中，observed-free 与 unknown 的 ray traversal / endpoint / occlusion pattern 明显不同；
2. visibility proxy 加入后能明显改善“可信为空 vs 未充分观测”的区分；
3. weather-induced target loss 更常伴随有效远端返回消失、射线提前终止或可见性覆盖下降；
4. 这种作用在控制 distance 后仍存在。

### 工程成本重新评估

**成本：中高，而不是中。**

原因：

- 需要每个 LiDAR 的 pose；
- 需要统一 world / ego / CAV 坐标；
- 若要精确复原 beam traversal，还需要 channels、vertical FOV、rotation frequency、points-per-second 等传感器参数；
- 多 CAV 必须分别在自己的传感器坐标系计算，再转换；
- 原始 OPV2V 没有直接提供完美的 per-cell occlusion/free-space 标签。

因此分两级做：

**Stage 1（低～中成本）**：仅用已有 point endpoints + LiDAR pose 构造近似 angular/ray traversal proxy，不追求恢复 CARLA 每条理论 beam。

**Stage 2（中高成本）**：只有 Stage 1 显示明显价值，才考虑小样本场景 replay / semantic LiDAR / 更精确 beam reconstruction。

### 外推限制

CARLA LiDAR 是 ray-cast simulator。它适合回答“在 OPV2V 仿真几何内部，visibility 是否有解释力”，但**不能直接证明真实 LiDAR 上 free-space 证据同样可靠**。

真实传感器中的多路径、beam divergence、表面反射、雨雾散射等传播细节并未由一个理想 ray-cast 几何模型完整覆盖。

因此如果 H-A3 后来成为论文核心机制，真实数据或至少第二种传感器/模拟器验证应成为后续外部有效性门禁。

---

## H-A4：Occlusion × Weather Interaction —— 天气损伤强烈依赖目标原本的遮挡/可见性

### 假设内容

恶劣天气对目标的影响不是一个与场景几何独立的统一衰减。

同样的 Fog/Rain/Snow 强度下：

- 原本完整可见的目标可能仍保留足够 evidence；
- 已被前车/建筑部分遮挡的目标本来就只剩少量关键表面；
- 天气再削弱少量回波后，可能出现**超线性/阈值式崩溃**。

因此所谓 observation insufficiency 可能主要由：

> `weather degradation × pre-existing occlusion`

共同决定。

### 可证伪预测

如果 H-A4 成立：

1. 控制 distance/size/weather 后，occluded targets 的 clean→weather evidence drop 明显更大；
2. weather 与 occlusion 存在显著 interaction，而不只是两个独立主效应相加；
3. 在 clean 中勉强可检出的部分遮挡目标，在 weather 中发生 miss 的比例显著高于无遮挡目标；
4. 邻车恢复收益更集中在 ego-occluded、neighbor-visible 的目标上。

### 数据可行性

OPV2V 本身覆盖 severe occlusion，并提供 LiDAR pose、点云和对象元数据，但**没有直接现成的完美 occlusion label**。

可以按成本递增得到 proxy：

1. clean frame 中 GT 框表面 hit count / surface coverage；
2. ego→target line-of-sight 上其它 GT box 的几何遮挡率；
3. 多车视角之间的 target hit-count / visibility contrast；
4. 必要时对少量场景 replay，加入 semantic LiDAR/depth 获取更强遮挡标签。

因此 H-A4 可检验，但不能假装“OPV2V 已经直接提供 occlusion flag”。

---

## H-A5：Local downstream bottleneck —— 一部分 apparent insufficiency 实际是 ego 表征/检测失败

### 假设内容

部分 weather miss 中，ego 原始 LiDAR 已经保留足够 target-supporting evidence，但 evidence 在以下某一层被丢失：

`raw points → pillar → BEV backbone → detection head → score/NMS`

这类样本不是“需要邻车补充”的典型 sensing insufficiency。

### 可证伪预测

如果 H-A5 成立：

1. 存在非小比例 weather miss，raw target evidence 与成功样本相当；
2. 这些样本的失败从中间表征/局部 proposal/head 才开始出现；
3. 即使不给邻车信息，通过保留/恢复 ego 自身已有 evidence 也存在明显 Oracle 上限；
4. quantity/visibility demand map 对这类样本不会有高 precision。

### Kill / downgrade criterion

如果绝大多数 ego weather miss 在 raw point 层已经明显缺证据，而中间层几乎只是忠实继承，那么 H-A5 不是主瓶颈。

---

## H-A6：Fusion bottleneck —— 邻车正确证据已经存在甚至已经通信，但 full fusion 仍没有充分利用

### 假设内容

Q-A 隐含了一个危险前提：

> “只要识别出 ego 缺证据并取得邻车信息，系统就会恢复。”

这个前提未被证明。

可能存在：

- neighbor 有清晰 target evidence；
- full communication 已包含该 evidence；
- 但 AttFuse 的 attention/source competition/feature scale 或下游 detector 仍把它稀释、抑制或错误解释；
- 最终 full fusion 仍 miss。

若这种情况占比高，Q-A 仍然是有意义的诊断问题，但它**不是当前系统的主要瓶颈**；此时“先做需求图再请求”不能作为论文主假设。

### 可证伪预测

如果 H-A6 成立：

1. 存在大量 `ego weak + at least one neighbor strong + full fusion still miss` 的目标；
2. 这些目标中，正确 source evidence 的局部 intervention / Oracle fusion 可以恢复一部分检测；
3. full fusion 的 attention/feature response 与真正有证据的 source 不一致；
4. 改善需求识别或通信 selection 但不改变融合机制，收益受明显上限限制。

### 什么结果会否定/削弱它

如果 `neighbor strong` 的目标在 full fusion 下几乎都能恢复，而 full miss 基本对应“所有车都弱/都缺证据”，则 fusion bottleneck 不是 Q-A 的主要竞争解释。

---

## 4. 被降级但仍保留的方法学变量

### M1：Neighbor evidence 不是本地需求 predictor，而是 Oracle recoverability label

旧 H-A4 的问题是部署顺序错误。

在“通信之前”的本地 demand estimation 中，ego 不能假设已经知道邻车当前区域 evidence。

因此邻车 evidence 只用于离线回答：

> **这个 ego miss 是否原则上可被协同恢复？**

它可以定义 recoverable / non-recoverable 子集、计算 Oracle 上限，也可以作为后续第二阶段“谁能帮助我”的研究对象，但不能偷偷作为 Q-A 本地 predictor 输入。

### M2：Weather type 是 moderator，不是独立机制假设

Fog/Rain/Snow 必须分层报告，但不再把“天气不同”本身当成 H-A6。

只有当出现一个**超出 H-A1～H-A6 的新机制**，并有独立可证伪预测时，才新增 weather-specific hypothesis。

目前天气类型只用于检查：

- effect 是否一致；
- 哪个假设在哪种天气更强；
- 是否存在 interaction。

---

# 5. 第一轮最低成本实验

## E-A0：Quantity proxy redundancy audit —— 第一优先级

### 目的

先回答审查提出的最基础问题：

> `区域总 N_eff` 和 `GT 框内可靠点数/可靠质量` 到底是不是几乎同一个量？

### 统计

逐 target、逐 weather 计算：

- region total `N_eff`；
- GT-box reliable point count；
- GT-box `Σr`；
- raw point count；
- distance；
- box size；
- occlusion proxy（若当前已有）；
- detection state。

报告：

- Pearson；
- Spearman；
- 按 weather 分层；
- 控制 distance/box size 后的 partial correlation。

### 决策

- 若 Pearson/Spearman > 0.85 且跨天气稳定：合并为 quantity family，不再人为区分；
- 若相关性明显较低：分析差异来自背景点、区域大小还是 target-support coverage。

**成本：低；不训练。**

---

## E-A1：Quantity-matched structural test —— 区分 H-A1 vs H-A2

建立 quantity-matched target pairs / bins：

- 相近 `N_eff`；
- 相近 distance；
- 相近 target size；
- 尽量匹配 occlusion。

然后比较成功/失败目标的：

- 表面 coverage；
- occupied pillar spatial pattern；
- point dispersion；
- frozen local feature/proposal response。

### 判读

- quantity 匹配后差异基本消失 → H-A1 ↑，H-A2 ↓；
- quantity 匹配后结构仍强区分 → H-A2 ↑，H-A1 降级为协变量。

**成本：低～中；优先利用已有输出。**

---

## E-A2：Failure-stage partition —— 一次区分 sensing / local downstream / fusion

对 paired clean/weather 的 GT targets，把 weather failure 分成至少四类：

1. **All-source weak**：ego 与 neighbors 都缺明显 evidence；
2. **Ego weak, neighbor strong, full recovers**：典型协同可恢复 sensing insufficiency；
3. **Ego weak, neighbor strong, full still misses**：支持 H-A6 fusion bottleneck；
4. **Ego raw strong, local/fused miss**：支持 H-A5 local downstream / detector failure。

先用简单、可审计的 evidence proxy，不训练新分类器。

### 价值

这是当前最重要的“Q-A 是否值得作为主线”的门禁。

如果第 2 类很大，Q-A 有直接价值；
如果第 3 类很大，应优先研究 fusion；
如果第 4 类很大，应优先研究 local representation/detector；
如果第 1 类很大，说明协同本身也缺信息，需求识别上限有限。

**成本：低～中；优先复用已有 full/no-fusion/GT/points。**

---

## E-A3：Occlusion × weather interaction audit —— 检验 H-A4

先不做精确 ray casting。

使用低成本 occlusion proxies：

- clean target hit count / surface coverage；
- ego→target LOS 上其它 GT box 的几何遮挡；
- 多车 target hit-count contrast。

比较 clean→weather 的：

- `ΔN_eff`；
- target point loss；
- ego miss probability；
- full recovery probability。

重点拟合/分层 `weather × occlusion` interaction，而不是只分别看 weather 和 occlusion 主效应。

**成本：低～中。**

---

## E-A4：Approximate visibility/free-space audit —— H-A3 的第一层门禁

不直接恢复 CARLA 全 beam model。

先基于：

- LiDAR pose；
- 实际 observed endpoints；
- angular bins；
- endpoint 前方的近似 traversal；

建立 conservative free-space proxy。

只回答：

> 在 point-count≈0 的区域里，它能否比 density/reliability 更好地区分“被观测过”与“几何上未知”？

如果没有明显增量，停止，不进入昂贵 replay。

**成本：中。**

---

## E-A5：Precise simulator visibility audit —— 条件触发

只有 E-A4 有明显正结果才做。

候选：

- 从 `data_protocol.yaml` / scenario replay 恢复 sensor setting；
- 小样本 CARLA/OpenCDA replay；
- semantic LiDAR/depth/actor geometry；
- 更严格 occlusion/free-space ground truth。

**成本：中高～高；不是 Priority A。**

---

# 6. 实验优先级

| 优先级 | 实验 | 主要区分 | 成本 | 是否训练 |
|---|---|---|---|---|
| A1 | E-A0 quantity proxy redundancy | H-A1/H-A2 是否真的独立 | 低 | 否 |
| A2 | E-A2 failure-stage partition | sensing vs local downstream vs fusion | 低～中 | 否 |
| A3 | E-A1 quantity-matched structural test | H-A1 vs H-A2 | 低～中 | 否 |
| A4 | E-A3 occlusion×weather audit | H-A4 | 低～中 | 否 |
| B1 | E-A4 approximate visibility | H-A3 | 中 | 否 |
| C1 | E-A5 precise replay/raycast | H-A3 深化 | 中高～高 | 否 |

当前**不应该训练任何新的 demand network**。

---

# 7. Q-A 的继续/停止门禁

在进入文献检索或方法设计前，至少回答三个问题：

### Gate 1：Quantity 与 structure 谁真正提供独立信息？

由 E-A0 + E-A1 回答。

### Gate 2：weather miss 主要发生在哪一层？

由 E-A2 回答。

这是最关键的门禁。

如果大多数失败属于：

- `ego weak + neighbor strong + full recovers`：Q-A 值得作为主线继续；
- `ego weak + neighbor strong + full still misses`：fusion bottleneck 更优先；
- `ego raw strong + local miss`：local representation/detector 更优先；
- `all-source weak`：协同恢复上限本身有限。

### Gate 3：遮挡是否是 weather failure 的主要 moderator？

由 E-A3 回答。

如果 occlusion×weather interaction 很强，后续 observation-sufficiency 模型不能只输入 weather reliability/density。

---

# 8. 当前明确不做的事

1. 不把 `N_eff` 直接定义成需求图。
2. 不把 GT-box reliable count 当作一个“新语义假设”，除非它与总 quantity 统计确实低相关且有独立意义。
3. 不把 neighbor evidence 当作通信前 ego 本地可用输入。
4. 不把 Fog/Rain/Snow 分层本身称为新假设。
5. 不直接做完整 ray casting/replay，先做低成本 proxy gate。
6. 不因为 matching/CEIF 失败就直接宣布 fusion bottleneck 已证明；要用 E-A2 把它和 sensing/local failure 分开。
7. 不默认“需求识别正确 → 通信/融合自然有效”。
8. 不训练新网络，直到 E-A0～E-A3 至少给出清楚的 failure partition。

---

# 9. 当前最重要的一句话

Q-A 现在不再被定义成：

> **“如何做一个更好的观测不足需求图？”**

而被定义成：

> **“恶劣天气导致的检测失败中，究竟有多少是真正的 ego sensing evidence insufficiency；它由 evidence quantity、evidence structure、visibility/occlusion 中哪一类因素决定；又有多少其实发生在 local representation 或 collaborative fusion 之后？”**

只有当这个问题被数据拆清楚，才值得进入第 3 阶段定向文献检索和后续方法设计。
