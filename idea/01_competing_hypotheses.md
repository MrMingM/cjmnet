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

### 当前状态：**OPEN — A5a vs A5b 尚未判定**

Stage-2 已经证明：

> 旧 N_eff/coverage strong 标签并不能直接当作 local task evidence 真值。

因此当前仍不能说“ego 明明看到了，只是 detector 没用好”。

H-A5a 与 H-A5b 仍然都保留。

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

## 5.3 Stage-3A 当前运行进度

Stage-3A 已完成：

- Stage-3 单元测试开发；
- 原 OpenCOOD postprocessor parity 测试；
- Stage-2 lineage / source snapshot guard；
- Stage-2.5 offline report；
- 三天气 preflight smoke；
- Fog 全量正式运行已推进到第 125 / 150 个 source-valid candidate frame。

Fog 运行到：

> candidate frame 125 / 150，累计 target 135，elapsed ≈ 54.3 min

随后在下一候选 frame 被一致性保护主动中止：

```text
AssertionError: Stage-2 frame FP count changed
```

当前已定位到失败帧：

- weather：Fog
- sample_index：1813
- source-valid target：31
- 同一 frame 的 Stage-2 weather-induced ego-miss target indices：4, 19, 31

Stage-2 在 sample 1813 保存的 frame FP：

| Branch | Stage-2 frame FP |
|---|---:|
| ego | 6 |
| full | 5 |
| peer 1 | 9 |
| peer 2 | 8 |
| peer 3 | 6 |
| peer 4 | 10 |

其中 target 31：

- peer 2：matched
- peer 4：matched
- peer 1 / 3：not matched
- full：miss（因此属于 source-valid/full-miss）

当前正在单独重放 sample 1813，诊断：

1. 哪个 branch 的当前 frame FP 与 Stage-2 不一致；
2. 差值是 1 个边界 FP 还是明显变化；
3. target 31 的 matched / IoU / score 是否仍严格复现；
4. sensing statistics 是否仍严格复现。

截至目前：

> **Stage-3A 不能视为完成，也不能使用已经生成的部分 Fog 结果做科学结论。**

但前 125 个候选 frame 均通过此前的 replay consistency guard，说明大部分历史重放目前是稳定的。

---

## 5.4 当前一致性问题的解释边界

当前 `Stage-2 frame FP count changed` 不能直接解释成模型机制变化。

因为 Stage-3 的 replay guard 同时检查：

- full `psm/rm` 与原 frozen model；
- ego branch 与 none-mask；
- target-level matched state；
- matched score / matched IoU；
- best postprocessed IoU / corresponding score；
- sensing statistics；
- frame FP count。

当前报错发生在 frame FP guard。

因此需要先区分：

### 情况 A：只有一个与研究 target 无关的边界 FP 漂移

例如：

> Stage-2 FP=5，Stage-3 replay FP=6；target 31 的 match/IoU/score 和 sensing 全部一致。

这种情况更像是整帧贪心匹配在 IoU≈0.7 边界附近的数值差异，不等于 source-valid target replay 失败。

### 情况 B：target 31 或 sensing / branch output 也发生明显变化

这种情况说明历史天气输入或运行环境没有充分复现，Stage-3A 不能继续解释。

当前仍在等待 sample 1813 单帧诊断结果，因此尚未决定是否调整该 guard。

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

# 6. 当前因果解释边界

目前可以说：

> **Stage-2 已确认：在 Fog/Rain/Snow 中存在 task-usable peer evidence individually exists but is not preserved by full collaboration 的 case，Snow 最严重。**

目前不能说：

- 失败根因已经定位在 AttFuse attention；
- ego query 一定压制 peer；
- 某个 peer 是 harmful vehicle；
- 最后消失在 score/NMS 就说明 score/NMS 是根因；
- subset selection 一定能解决问题；
- H-A5a 或 H-A5b 已经判定。

尤其需要保持以下区分：

> **“最后在哪一步看不见了” ≠ “最初是哪一步造成了错误”。**

例如 fusion 改变 feature 后导致 score 下降，最后表现为 `score_filtered`，根因仍可能在更早的 feature interaction。

---

# 7. 当前研究状态总结

当前 Q-A 已经从最初的：

> “如何做一个更好的 N_eff/coverage 需求图？”

推进到：

> **“哪些 sensing proxy 真能代表 task-usable evidence；当 peer 的 task-usable evidence 已经客观存在时，为什么 full collaboration 仍会使一部分目标消失？”**

当前最重要的已确认事实是：

1. Quantity family 很强，但多个 quantity proxy 高度冗余；
2. Structure 有条件增量，但尚未证明主导；
3. Approximate visibility 第一层 Gate 未通过；
4. occlusion 是 weather-dependent moderator；
5. N_eff/coverage strong 并不等价于 task evidence，Snow 尤其明显；
6. source-valid/full-miss 现象真实存在：Fog 162、Rain 63、Snow 988；
7. 因此 H-A6 的“现象”已获得支持，但“具体原因”仍未定位；
8. H-A5 仍保持 OPEN；
9. Stage-3A 正在进行 detection-path disappearance audit；
10. 当前实验在 Fog sample 1813 被 frame-FP replay guard 主动中止，正在做单帧一致性诊断，尚无完整 Stage-3A 科学结论。

当前一句话科学问题：

> **恶劣天气下，哪些观测量真正代表可用于检测的 task evidence；当这种证据在某个 peer 中已经被证明确实存在时，它又是在 multi-agent collaboration 的哪个可观察阶段失去最终检测能力，以及这一现象背后的真正因果机制是什么？**
