# Q-A：恶劣天气下“观测不足”竞争假设研究地图

> 更新日期：2026-09-17  
> 本文件服务于 `idea/00_research_workflow.md`：**现象 → 竞争假设 → 可证伪预测 → 最低成本验证 → 证据有效性审查 → 因果定位**。  
> 当前仍只研究 Q-A，不设计新网络，不把诊断现象直接写成机制结论。

---

## 0. 总研究方向与当前问题

本项目总方向始终是：

> **恶劣天气下协同感知鲁棒性。**

当前研究问题 Q-A：

> **在没有 GT、没有 clean frame 的实际推理阶段，自车如何判断一个局部区域是“观测不足 / 未充分观测”，还是“已经充分观测，并且有理由相信该区域为空”？**

第一轮诊断已经表明，最终 `weather miss` 不能直接等同于“自车没看到”。当前必须同时保留以下竞争解释：

1. sensing evidence 的量真的不足；
2. 数量相近，但目标相关结构不同；
3. “未观测”和“可信为空”需要 visibility/free-space 信息；
4. 遮挡与天气存在交互；
5. 当前 evidence-strength 指标可能把错误/无效证据误判为 strong；
6. 真正 task-usable 的 source evidence 可能存在，但 local downstream 或 fusion 没有正确利用。

当前目标不是证明某一条，而是继续排除解释。

---

# 1. Stage-1：第一轮 observation diagnostic（已完成）

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

H-A2 只通过了第一轮 Cheap Gate。

---

## F3：Stage-1 的 `strong / weak evidence` 只是经验标签，不是真值

Stage-1 使用 clean-detected targets 的 distance-bin Q25 阈值，将 `box N_eff + coverage4x4` 用作 strong/weak 判据。

weather-induced ego misses 中的 ego-strong fraction：

- Fog：≈ 0.359
- Rain：≈ 0.363
- Snow：≈ 0.560

当时不能据此声称：

> “这些目标已经保留足够正确的信息，只是网络没用好。”

同等重要的竞争解释是：

> **evidence-strength ruler 本身可能把天气噪声、散射或错误表征误判为 task-usable evidence。**

Stage-2 已专门审计这一点。

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

---

## F5：Approximate visibility/free-space 第一层 Gate 未通过

`background_zero - target_zero` 的 high-traversal gap：

- Clean：0.0819
- Fog：0.0464
- Rain：0.0308
- Snow：0.0589

未达到预设的明显分离门槛。

因此 H-A3 当前为 **PAUSED / WEAKENED**，不是被永久证伪。

---

## F6：遮挡 × 天气存在条件性信号，但不是统一机制

Stage-1 high-vs-low occlusion weather-miss risk ratio：

- Fog：2.45
- Rain：2.69
- Snow：0.75

因此：

> 遮挡在 Fog/Rain 中可能显著放大天气损伤，但 Snow 方向不同。

当前把 occlusion 作为 **weather-dependent moderator / 分层变量**，不作为统一主机制。

原报告中 Snow low-occlusion 的巨大 relative N_eff loss 是 near-zero clean denominator 导致的数值爆炸，不能物理解读。Stage-2 已改用 absolute / bounded / log1p 等更稳健指标复核。

---

# 3. Stage-2：Evidence Validity Audit（已完成）

代码：`qa_evidence_validity/`  
正式结果：`/data/cjm/datasets/logs/qa_evidence_validity_20260917_112301/evidence_validity_report.md`

Stage-2 的核心目的不是设计方法，而是回答：

> **Stage-1 的 N_eff / coverage strong-weak 尺子到底有多可信？以及是否真的存在“peer 单独有可检测证据，但 full fusion 丢掉目标”的 case？**

---

## S2-F1：strong/weak 结论对阈值有一定稳定性，但阈值稳定不等于语义正确

Q20-Q40 下 ego-strong fraction 波动范围 / peer-strong fraction：

| Weather | ego strong fraction range | peer strong fraction range |
|---|---:|---:|
| Fog | 0.184483 | 0.050341 |
| Rain | 0.225341 | 0.060698 |
| Snow | 0.207180 | 0.089415 |

因此旧尺子并非完全由某个 Q25 阈值偶然造成，但这仍不能证明 strong 就是真 task evidence。

---

## S2-F2：Q25 proxy 与 peer-alone task truth 并不等价，Snow 最明显

peer-alone 定义：

> 单个 peer 的 ego-aligned pre-fusion multiscale feature，绕过 AttFuse 和 ego，独立经过原 frozen deblocks + cls/reg head + 原 post-processing；对同一 GT 用 IoU≥0.7 greedy match 判断是否可独立检测。

Q25 proxy 与 peer-alone 的对应关系：

| Weather | source-level precision | source-level recall | any-peer precision | any-peer recall |
|---|---:|---:|---:|---:|
| Fog | 0.7317 | 0.8699 | 0.8922 | 0.9543 |
| Rain | 0.7894 | 0.8391 | 0.9402 | 0.9521 |
| Snow | 0.4963 | 0.4958 | 0.7075 | 0.7282 |

解释：

- Fog/Rain 中旧 proxy 有一定参考价值；
- Snow 中 source-level precision/recall 都只有约 0.5，说明 `N_eff/coverage strong` 经常不等于 detector 真能使用的证据；
- 因此旧 strong/weak 规则不再适合作为 H-A6 的主筛选条件。

当前处理：

> **N_eff / coverage 降级为 sensing-side auxiliary proxy，不再作为“真实证据存在”的主要判据。**

---

## S2-F3：Stage-1 的旧 metric fusion candidate 只覆盖了真实 source-valid/full-miss 的一部分

Stage-1 旧候选：`ego weak + peer strong + full miss`。

Stage-2 真正更强的候选：

> `any peer-alone detects same GT + full fusion misses`

旧候选被 task truth 验证的比例：

| Weather | Stage-1 metric candidate | validated by peer-alone | precision |
|---|---:|---:|---:|
| Fog | 166 | 61 | 36.75% |
| Rain | 32 | 13 | 40.63% |
| Snow | 753 | 313 | 41.57% |

旧规则对新的 source-valid/full-miss 的覆盖率：

- Fog：61 / 162 = 37.65%
- Rain：13 / 63 = 20.63%
- Snow：313 / 988 = 31.68%

因此不能把旧规则漏掉的 case 全部归因于“peer proxy 不准”；还有一部分是因为 ego 并不满足 metric-weak。

结论是：

> **整个旧 metric screening rule 不应继续作为 H-A6 主筛选器。**

---

## S2-F4：task-usable peer evidence 确实大量存在，而且 full fusion 在一部分 case 中没有保留它

Stage-2 对全部 weather-induced ego miss candidates 做 peer-alone 审计：

| Weather | weather ego misses | any peer-alone detects | full recovers | source-valid + full miss | no peer-alone detects |
|---|---:|---:|---:|---:|---:|
| Fog | 3377 | 2516 | 2354 | 162 | 861 |
| Rain | 1318 | 1106 | 1043 | 63 | 212 |
| Snow | 5961 | 3793 | 2805 | 988 | 2168 |

对应比例：

### 在全部 weather-induced ego misses 中

- Fog：any peer-valid 74.50%；source-valid/full-miss 4.80%
- Rain：any peer-valid 83.92%；source-valid/full-miss 4.78%
- Snow：any peer-valid 63.63%；source-valid/full-miss 16.57%

### 在已经确认 any-peer-valid 的 case 中

- Fog：full miss 6.44%，full recovery 93.56%
- Rain：full miss 5.70%，full recovery 94.30%
- Snow：full miss 26.05%，full recovery 73.95%

这些统计单位是：

> **target-frame occurrence（某个目标在某一帧的一次出现）**，不是独立车辆数。

并且子集限定为：

> clean ego detected + weather ego missed。

因此不能把这些比例写成全数据集漏检率，也不能直接当成 AP 上界。

---

## S2-F5：H-A6 的“现象”得到支持，但“具体机制”仍未定位

Stage-2 已经能确认：

> **在一个非平凡数量的 case 中，某个 peer 单独通过当前冻结 detector 可以正确检测同一 GT，但 full collaboration 后该目标消失。**

这比 Stage-1 的 proxy-defined candidate 强得多，因为它已经证明 source evidence 至少在当前 detector 下是 task-usable 的。

但它仍不能证明：

- AttFuse attention 是唯一原因；
- ego query 一定压制 peer；
- 某个 peer 一定是 harmful source；
- 失败一定发生在 fusion operator 本身；
- score/NMS 是根因。

正确表述是：

> **task-usable peer evidence individually exists but is not preserved by full collaboration in a nontrivial set; causal failure localization is still required.**

---

## S2-F6：H-A5 仍保持 OPEN

Stage-2 主要解决的是 peer source validity，因此它对 H-A6 的证据更直接。

对于 H-A5：

### H-A5a
Evidence ruler overestimates local evidence：

> 点数/coverage/feature magnitude 看起来强，但没有正确 target semantics。

### H-A5b
Task-usable local evidence survives, downstream fails to use it：

> 真正有效的信息仍存在，但 PillarVFE / backbone / detector / post-processing 没有转化成正确检测。

Stage-2 证明了旧 ruler 并不等价于 task truth，尤其 Snow 中误差明显，因此：

> **H-A5 仍不能在 Stage-2 后直接判定 A5a 或 A5b 哪一个成立。**

当前不把 H-A5 与 H-A6 混成同一个结论。

---

## S2-F7：Ego → Full degradation 现象存在，但与 source-valid/full-miss 不是同一个集合

Stage-2 直接统计 ego-detected → full-miss：

| Weather | ego-detected→full-miss | full recovery given ego miss | top-3 scene share of degradation |
|---|---:|---:|---:|
| Clean | 0.00852 | 0.73152 | 0.77474 |
| Fog | 0.01623 | 0.66775 | 0.82785 |
| Rain | 0.00803 | 0.71696 | 0.76642 |
| Snow | 0.05725 | 0.46995 | 0.75656 |

注意：

> 这里的 top-3 scene share 是一般 ego-detected→full-miss degradation，不是 source-valid/full-miss 162 / 63 / 988 的 scene concentration。

两者不能混用。

---

## S2-F8：修正后的 N_eff 统计否定了原先 Snow 巨大 relative loss 的物理解读

修正后同时报告 absolute / bounded / log1p / clean>0 relative。

代表性结果：

### Fog

- low-occ：absolute +0.3578，bounded +0.00277，log1p +0.00644
- high-occ：absolute +0.2932，bounded +0.04103，log1p +0.07706

### Rain

- low-occ：absolute +0.0870，bounded +0.00063，log1p +0.00122
- high-occ：absolute +0.1944，bounded +0.01292，log1p +0.02029

### Snow

- low-occ：absolute +7.0882，bounded -0.01265，log1p -0.09030
- high-occ：absolute +5.4538，bounded +0.04761，log1p +0.08846

因此：

> 原先 Snow 极端 relative N_eff loss 主要是 denominator artifact，不能继续作为机制证据。

---

# 4. 六个竞争假设的最新状态

## H-A1：Quantity-dominant / Quantity-family hypothesis

### 当前状态：**SUPPORTED AS STRONG BASELINE / OPEN AS DOMINANT MECHANISM**

quantity proxies 高度冗余，quantity AUC 仍然强。

但 Stage-2 进一步说明：

> quantity/coverage proxy 不能可靠等价于 task-usable evidence，尤其 Snow。

所以 quantity 可以继续作为 sensing strength baseline，但不能直接充当“目标证据是否真实存在”的真值。

---

## H-A2：Structure-dominant incremental hypothesis

### 当前状态：**OPEN / PASSED FIRST CHEAP GATE**

Stage-1 的 quantity-controlled structure signal 仍然保留。

但 Stage-2 显示 Snow 中 proxy 与 task truth 偏差较大，因此未来解释 structure 时必须避免把 weather-induced pseudo coverage 当作正确目标结构。

当前没有新增证据足以升级或否定 H-A2。

---

## H-A3：Visibility/free-space hypothesis

### 当前状态：**WEAKENED / PAUSED**

Stage-1 approximate traversal 未通过第一层 Gate。

当前不投入精确 beam reconstruction。

---

## H-A4：Occlusion × Weather interaction

### 当前状态：**PARTIALLY SUPPORTED AS MODERATOR**

Fog/Rain 有明显风险放大，Snow 方向不同。

继续作为分层变量使用，不升级为统一主机制。

---

## H-A5：Local downstream bottleneck

### 当前状态：**A5b DOMINANTLY SUPPORTED / SNOW PARTLY UNRESOLVED**

Stage-2 时 H-A5a / H-A5b 尚不能区分；2026-09-18 的独立 ego-local 全量审计已经直接检查了 clean-hit / weather-miss case 的本地检测证据。

在 q25 proxy-strong 子集中：

- Fog：1200 / 1211 = **99.09%** 有 A5b 型证据，unresolved 0.91%；
- Rain：477 / 478 = **99.79%** 有 A5b 型证据，unresolved 0.21%；
- Snow：2910 / 3336 = **87.23%** 有 A5b 型证据，unresolved 12.77%。

因此当前不能再把 H-A5 主要解释为“N_eff / coverage 虚高”。更准确的结论是：

> **Fog / Rain 几乎全部、Snow 的大多数 proxy-strong weather ego misses 中，目标级有效信息仍然存在，但没有被本地检测链路转化成最终 detection。H-A5b 是主导现象。**

但 Snow 仍有 12.77% q25 strong case 属于 upstream unresolved，因此：

> **A5a 没有被完全否定；该 Snow 子集也可能是更早的 PillarVFE / backbone 表征损坏，当前不能二选一。**

---

## H-A6：Fusion bottleneck

### 当前状态：**PHENOMENON SUPPORTED / CAUSE UNRESOLVED**

Stage-2 已完成 source validity check。

现在不再使用：

> `ego weak + peer strong + full miss`

作为主要机制候选。

当前主要候选定义为：

> **`peer-alone detects same GT + full fusion misses`**

正式 source-valid/full-miss 数量：

- Fog：162
- Rain：63
- Snow：988

因此 H-A6 的“失败现象”已经获得较强支持：

> 单个 peer 的 task-usable evidence 确实存在，但 full collaboration 在一部分 case 中没有保留最终检测能力。

但具体因果原因仍未确定。

---

# 5. Stage-2.5 / Stage-3：Source-valid Fusion Failure Causal Audit

代码已经写入：`gspr_evidence/`

主要文件：

- `stage3_analysis.py`
- `stage3_common.py`
- `stage3_runtime.py`
- `stage3_trace.py`
- `stage3a.py`
- `stage3b.py`
- `stage3_report.py`
- `test_stage3.py`
- `run_stage3a.sh`
- `run_stage3b.sh`
- `STAGE3.md`

实验边界：

- 只使用 OPV2V official validation + 与 Stage-1/2 一致的 online weather；
- 不使用 OPV2V-W test；
- 不训练；
- 不修改 GSPR / AttFuse / detector / postprocessor；
- Stage-3A 不自动执行 Stage-3B；
- 所有输出写到 `/data/cjm/datasets/logs/`。

---

## 5.1 Stage-2.5：Scene / distance denominator audit

Stage-2.5 只读 Stage-2 已有结果，不做 GPU 推理。

核心统计定义：

> `failure_rate_scene = #(source-valid + full-miss) / #(any peer-alone detected)`

同时记录：

- numerator；
- denominator；
- target-frame occurrences；
- unique frames；
- distance distribution。

这里强调：

> “某 scene 占全部失败很多”不等价于“该 scene failure propensity 很高”。必须同时看分母。

Stage-2.5 已集成进 `run_stage3a.sh` 的 offline report 流程。

---

## 5.2 Stage-3A：Detection-path disappearance audit

Stage-3A 的问题是：

> **对于已经确认 source-valid/full-miss 的目标，正确检测能力最后在检测流程的哪一步消失？**

它定位的是：

> **last-observed disappearance stage**

不是：

> **root cause / failure origin**。

当前 tracer 跟踪：

```text
psm → sigmoid score
rm + anchors → decoded boxes
→ score threshold
→ large-box / abnormal-z filtering
→ NMS top-1000
→ rotated NMS suppression
→ range filtering
→ final greedy GT matching
```

失败分类：

- `no_iou70_after_decode`
- `score_filtered`
- `geometry_filtered`
- `nms_topk_filtered`
- `nms_suppressed`
- `range_filtered`
- `matching_competition`
- `other`

所有 score / IoU 始终绑定同一 candidate；candidate ID 保持原 anchor lineage。

对于 NMS suppression 记录真实 suppressor、分数、GT IoU、框间 IoU；对 matching competition 记录最终被分配到哪个 GT。

---

## 5.3 Stage-3A：正式结果（已完成）

正式结果：

`/data/cjm/datasets/logs/qa_stage3a_20260917_193015/stage3a_report.md`

实验单位：

> **target-frame occurrence**，即某个目标在某一帧的一次出现；同一车辆可以跨帧重复出现。

候选范围：

| Weather | target occurrences | candidate frames | scenes |
|---|---:|---:|---:|
| Fog | 162 | 150 | 6 |
| Rain | 63 | 61 | 3 |
| Snow | 988 | 694 | 9 |

这些候选全部来自 Stage-2 已确认的：

> **peer-alone detects same GT + full fusion misses**

因此 Stage-3A 不是再用 N_eff / coverage proxy 猜测 evidence，而是只诊断已经通过 task-level source validity check 的真实 source-valid/full-miss case。

### 最后可观察到的失败阶段

| Stage | Fog | Rain | Snow |
|---|---:|---:|---:|
| no_iou70_after_decode | 3 (1.9%) | 1 (1.6%) | 63 (6.4%) |
| score_filtered | 45 (27.8%) | 6 (9.5%) | 798 (80.8%) |
| geometry_filtered | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| nms_topk_filtered | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| nms_suppressed | 114 (70.4%) | 56 (88.9%) | 127 (12.9%) |
| range_filtered | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| matching_competition | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| other | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |

主要表现：

- Fog：主要最后消失在 `nms_suppressed`，114 / 162 = 70.4%；
- Rain：主要最后消失在 `nms_suppressed`，56 / 63 = 88.9%；
- Snow：主要最后消失在 `score_filtered`，798 / 988 = 80.8%；
- 真正到 decode 后都没有 IoU≥0.7 proposal 的比例较低：Fog 1.9%、Rain 1.6%、Snow 6.4%；
- geometry filtering、NMS top-1000 截断、range filtering、matching competition 在本批 source-valid/full-miss 中均为 0。

因此 Stage-3A 最直接支持的观察是：

> **对于绝大多数 source-valid/full-miss case，正确几何候选并没有在 decode 前完全消失；它们大多已经形成 IoU≥0.7 proposal，最后主要在 score threshold 或 rotated NMS 阶段失去最终检测能力。**

但这仍然只是：

> **last-observed disappearance stage**

而不是：

> **root cause / failure origin**。

例如：

- full fusion 可能更早已经改变 feature，使正确框 score 被压低，最后才表现为 `score_filtered`；
- full fusion 也可能改变候选排序，使错误高分框在 NMS 中压制正确框，最后才表现为 `nms_suppressed`。

因此不能把 Stage-3A 结果直接解释成：

> “根因就是 score threshold / NMS”。

---

## 5.4 Stage-2.5：source-valid/full-miss 的 scene / distance 分布

Stage-2.5 给 source-valid case 提供了真实分母：

> `failure_rate = #(source-valid + full-miss) / #(any peer-alone detected)`

### Fog

按 scene：

- scene 8：2127 个 source-valid target occurrences，135 个 full-miss，失败率 6.35%；
- 其他 scene 也存在失败，例如 scene 2 为 14.63%、scene 7 为 11.11%、scene 6 为 9.09%；
- 因此 Fog 失败数量虽主要集中在大 scene，但不是只存在于单一 scene。

按距离：

| Distance | source-valid | full-miss | failure rate |
|---|---:|---:|---:|
| 0–20 m | 86 | 4 | 4.65% |
| 20–40 m | 265 | 20 | 7.55% |
| 40–60 m | 681 | 82 | 12.04% |
| 60–80 m | 946 | 17 | 1.80% |
| 80–100 m | 357 | 17 | 4.76% |
| 100+ m | 181 | 22 | 12.15% |

### Rain

按 scene：

- scene 8：977 个 source-valid target occurrences，59 个 full-miss，失败率 6.04%；
- scene 7：23 个 source-valid，3 个 full-miss，失败率 13.04%；
- Rain 总样本较少，scene 分布比 Fog/Snow 更集中。

按距离：

| Distance | source-valid | full-miss | failure rate |
|---|---:|---:|---:|
| 0–20 m | 27 | 0 | 0.00% |
| 20–40 m | 82 | 2 | 2.44% |
| 40–60 m | 361 | 46 | 12.74% |
| 60–80 m | 235 | 0 | 0.00% |
| 80–100 m | 98 | 1 | 1.02% |
| 100+ m | 303 | 14 | 4.62% |

### Snow

Snow 在多个 scene 都有明显失败：

| Scene | source-valid | full-miss | failure rate |
|---|---:|---:|---:|
| 0 | 329 | 161 | 48.94% |
| 1 | 204 | 73 | 35.78% |
| 2 | 207 | 81 | 39.13% |
| 3 | 284 | 130 | 45.77% |
| 4 | 50 | 13 | 26.00% |
| 5 | 27 | 8 | 29.63% |
| 6 | 62 | 26 | 41.94% |
| 7 | 98 | 32 | 32.65% |
| 8 | 2532 | 464 | 18.33% |

按距离：

| Distance | source-valid | full-miss | failure rate |
|---|---:|---:|---:|
| 0–20 m | 1695 | 552 | 32.57% |
| 20–40 m | 661 | 226 | 34.19% |
| 40–60 m | 603 | 146 | 24.21% |
| 60–80 m | 492 | 46 | 9.35% |
| 80–100 m | 221 | 13 | 5.88% |
| 100+ m | 121 | 5 | 4.13% |

因此 Snow 的 source-valid/full-miss 不是由单个 scene 或单个距离段独占；多个 scene 均出现较高失败率，而且近中距离（尤其 0–60 m）更明显。

---

## 5.4.1 Stage-3A 对 H-A5 / H-A6 的含义

### 对 H-A5

Stage-3A 仍然没有直接区分 ego-local 的 A5a / A5b，因为当前候选是由 **peer-alone task validity** 定义的，不是 ego-local task validity。

因此：

> **H-A5 仍保持 OPEN。**

### 对 H-A6

Stage-3A 把 H-A6 从“full collaboration 后目标消失”进一步细化为：

> **绝大多数 source-valid/full-miss case 在 decode 后仍存在 IoU≥0.7 proposal；最终检测能力主要在 score / NMS 阶段消失。**

这说明：

- 不能再把大多数 case 简单描述为“正确 peer feature 在 fusion 后完全消失”；
- 更准确的描述是：**full collaboration 后，正确目标几何候选通常仍然存在，但其置信度或候选竞争关系发生了不利变化，导致最终 detection 被过滤或抑制。**

具体根因仍未确定。

---

## 5.5 Stage-3B 当前状态

### 状态：**代码已实现，但尚未启动正式实验**

Stage-3B 只允许在 Stage-3A 完整报告人工阅读后手动运行。

它固定：

- per-frame encoded features；
- 原 AttFuse；
- 原 detector / postprocessor；
- ego 始终保留。

只改变：

> **参与融合的整车 source subset。**

最多 5 辆车，因此 ego-containing subsets 最多 16 个。

Stage-3B 必须区分：

1. **target-wise oracle recoverability**：是否存在某个 subset 能恢复某个 target；
2. **frame-wise realizable result**：一帧只能选择一个统一 subset，不能把不同目标各自最优 subset 拼成虚假的系统输出。

目前没有 Stage-3B 科学结果。

---

## 5.6 H-A5：Ego-local task evidence audit（2026-09-18 已完成）

结果目录：

`/data/cjm/datasets/logs/qa_local_evidence_full_20260918_112018`

本实验与 H-A6 的 Stage-3A 不同：

- H-A6 Stage-3A 从 **peer-alone valid + full miss** 出发，研究协同融合后正确检测为什么消失；
- 本 H-A5 审计从 **clean ego detected + weather ego missed** 出发，只研究自车本地链路，不使用 AttFuse。

旧 q20/q25/q30/q40 N_eff + coverage strong 标签只用于筛选和敏感性分析，不再当作 task truth。

### q25 主结果

| Weather | All ego misses | q25 strong | Late downstream loss | Regression/localization support | Upstream unresolved | A5b support |
|---|---:|---:|---:|---:|---:|---:|
| Fog | 3377 | 1211 | 1106 (91.33%) | 94 (7.76%) | 11 (0.91%) | 99.09% |
| Rain | 1318 | 478 | 461 (96.44%) | 16 (3.35%) | 1 (0.21%) | 99.79% |
| Snow | 5961 | 3336 | 2843 (85.22%) | 67 (2.01%) | 426 (12.77%) | 87.23% |

这里：

- **Late downstream loss**：最终 weather ego 漏检，但 decode 后已经出现 IoU≥0.7 的正确目标框，因此目标级几何证据已经存在，只是在后续检测流程中消失；
- **Regression/localization support**：没有正确 IoU≥0.7 decoded box，但 clean 中匹配该目标的同一 anchor 在天气分支中仍通过正常分类阈值，说明分类侧信号仍存活而定位失败；
- **Upstream unresolved**：两种信号都没有观察到，只能说明 A5a 或更早的局部特征损坏仍可能存在，不能直接判成 A5a。

三天气 q25 strong 合计 5025 个，其中：

- A5b-support：4587 / 5025 = **91.28%**；
- upstream unresolved：438 / 5025 = **8.72%**。

### 阈值敏感性

A5b-support 在 q20 / q25 / q30 / q40 下分别为：

- Fog：99.00% / 99.09% / 99.02% / 99.23%；
- Rain：99.82% / 99.79% / 99.73% / 99.63%；
- Snow：88.02% / 87.23% / 86.24% / 84.53%。

Snow unresolved 则从 q20 的 11.98% 上升到 q40 的 15.47%，因此 Snow unresolved 不能简单归因于“刚越过 strong 阈值的边缘样本”。

### A5 全量失败阶段补充统计

对现有 A5 全量 `targets.jsonl` 做离线汇总，不重新推理。

| Weather | q25 strong | 没生成正确框 | 分数过低 | NMS 竞争失败 |
|---|---:|---:|---:|---:|
| Fog | 1211 | 105 (8.67%) | 428 (35.34%) | 678 (55.99%) |
| Rain | 478 | 17 (3.56%) | 180 (37.66%) | 281 (58.79%) |
| Snow | 3336 | 493 (14.78%) | 2281 (68.38%) | 562 (16.85%) |

因此 q25 strong 中：

- Fog：**91.33%** 在最终漏检前已经生成 IoU≥0.7 正确框，主要最后消失在 NMS 竞争；
- Rain：**96.44%** 已经生成正确框，主要最后消失在 NMS 竞争；
- Snow：**85.22%** 已经生成正确框，但 **68.38%** 最后因分数过低被删除。

同一 clean 匹配 anchor 到天气分支后的中位变化：

| Weather | score Δ | logit Δ | IoU Δ | 仍过分数阈值 | 仍 IoU≥0.7 |
|---|---:|---:|---:|---:|---:|
| Fog | -0.080928 | -0.383998 | -0.056595 | 77.95% | 35.43% |
| Rain | -0.036148 | -0.191763 | -0.026554 | 75.52% | 42.26% |
| Snow | -0.593123 | -3.791191 | -0.142969 | 26.11% | 47.09% |

这说明 A5 的“目标信息仍存在”还可以进一步细化为：

> **Fog / Rain 主要表现为正确候选框的排序和竞争关系失稳；Snow 主要表现为正确候选框的分类置信度明显下降。**

这一模式与 H-A6 Stage-3A 独立得到的天气差异方向一致：

- H-A6 Fog / Rain 也主要最后消失在 NMS；
- H-A6 Snow 也主要最后消失在 score threshold。

但必须强调：A5 和 H-A6 Stage-3A 是不同候选集、不同问题，二者只能作为相互印证的现象，不能把比例直接合并，也不能据此宣布 NMS / score threshold 就是最早根因。

### 对 H-A5 的最新判断

本轮直接改变了此前“Stage-2 / Stage-3A 后 H-A5 仍 OPEN”的状态：

> **H-A5b 是主导现象，Fog / Rain 证据尤其强；Snow 仍保留一个约 13% 的 upstream-unresolved 子集。**

因此目前最稳妥的说法是：

> **很多恶劣天气自车漏检不是“完全没看到目标”，而是目标级有效信息已经存在，却没有成功变成最终检测。**

但必须继续保持边界：

> **A5b 主导 ≠ A5a 被完全否定；尤其 Snow unresolved 仍不能区分 sensing proxy 高估与更早表征损坏。**

---

# 6. 当前因果解释边界

目前可以说：

1. Stage-2 已确认 Fog/Rain/Snow 中存在真实的 `peer-alone detects + full misses`；
2. Stage-3A 已进一步确认：这些 case 中绝大多数在 decode 后仍存在 IoU≥0.7 的正确几何 proposal；
3. Fog/Rain 最后主要消失在 rotated NMS suppression；
4. Snow 最后主要消失在 score threshold；
5. Snow 的现象跨多个 scene 存在，并非单一场景异常。

目前不能说：

- 失败根因已经定位在 AttFuse attention；
- ego query 一定压制 peer；
- 某个 peer 一定是 harmful vehicle；
- `score_filtered` 就证明 classifier / score head 是根因；
- `nms_suppressed` 就证明 NMS 算法本身是根因；
- source subset selection 一定可以解决；
- H-A5a 在 Snow unresolved 子集中一定成立；
- H-A5a 已经被完全否定。

必须保持下面这个区分：

> **“正确 proposal 最后在哪一步消失” ≠ “最初是哪一步导致它变成低分或在候选竞争中失败”。**

当前最稳妥的机制描述是：

> **task-usable peer evidence individually exists；full collaboration 后，大多数目标仍保留正确几何 proposal，但 proposal confidence / ranking / competition 被改变，最终主要通过 score filtering 或 NMS suppression 失去检测能力。其更早的因果来源仍需进一步定位。**

---

# 7. 当前研究状态总结

当前 Q-A 已经从最初的：

> “如何做一个更好的 N_eff/coverage 需求图？”

推进到：

> **“哪些 sensing proxy 真能代表 task-usable evidence；当 peer 的 task-usable evidence 已经客观存在时，full collaboration 为什么会改变正确 proposal 的 score / ranking / competition，使其最终被过滤或抑制？”**

当前最重要的已确认事实是：

1. Quantity family 很强，但多个 quantity proxy 高度冗余；
2. Structure 有条件增量，但尚未证明主导；
3. Approximate visibility 第一层 Gate 未通过；
4. occlusion 是 weather-dependent moderator；
5. N_eff/coverage strong 并不等价于 task evidence，Snow 尤其明显；
6. source-valid/full-miss 现象真实存在：Fog 162、Rain 63、Snow 988；
7. Stage-3A 已完成，且绝大多数 source-valid/full-miss 在 decode 后仍存在 IoU≥0.7 proposal；
8. Fog：70.4% 最后消失在 NMS suppression，27.8% 消失在 score threshold，仅 1.9% decode 后无 IoU≥0.7 proposal；
9. Rain：88.9% 最后消失在 NMS suppression，9.5% 消失在 score threshold，仅 1.6% decode 后无 IoU≥0.7 proposal；
10. Snow：80.8% 最后消失在 score threshold，12.9% 消失在 NMS suppression，6.4% decode 后无 IoU≥0.7 proposal；
11. Snow 的失败跨多个 scene 存在，并且 0–60 m 距离段 failure rate 明显较高；
12. H-A6 的“现象”和“最后可观察消失阶段”均已获得较强证据，但更早的 root cause 仍未定位；
13. H-A5 ego-local 全量审计已完成：q25 strong 中 A5b-support 为 Fog 99.09%、Rain 99.79%、Snow 87.23%，因此 H-A5b 已获得主导性支持；
14. A5 全量失败阶段统计进一步表明：Fog 55.99%、Rain 58.79% 的 q25 strong 漏检最后表现为 NMS 竞争失败；Snow 68.38% 最后表现为正确框分数过低；
15. A5 与 H-A6 Stage-3A 虽是不同候选集，但都呈现 Fog/Rain 更偏候选竞争、Snow 更偏置信度下降的天气模式；
16. Snow 仍有 12.77% q25 strong case 属于 upstream unresolved，不能据此宣布 A5a 被完全否定；
17. Stage-3B 代码已实现，但尚未启动正式实验。

当前一句话科学问题：

> **恶劣天气下，当某个 peer 已经被证明包含可独立完成检测的 task-usable evidence 时，为什么 full collaboration 会让本来仍可形成正确几何 proposal 的目标变成低分或在候选竞争中被压制，以及这种变化最早由哪个协同环节引起？**
