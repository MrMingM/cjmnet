# 创新点墓地 / 假设墓地

> 建立日期：2026-09-16  
> 作用：记录“已经试过什么、失败发生在哪一层、未来什么不能换名字重做”。  
> 证据优先级：最新 `AI_CONTEXT.md` > 原始结果记录 > README/实验说明 > 代码与配置。只有代码而没有结果时，一律写成【代码存在】【结果未核验】，不能写成完成实验。

## 0. 使用说明

### 0.1 一句话规则

这个文件不是“失败项目清单”。它把三件事分开：**科学命题是否站得住、当前实现是否做对、实验记录是否足够完整**。

状态含义：

| 状态 | 本账本中的含义 |
|---|---|
| `SUPPORTED` | 已有较强项目内证据支持，但不是绝对证明 |
| `OPEN` | 科学问题仍未充分回答，或新实验正在进行 |
| `FAILED_IMPLEMENTATION` | 当前版本失败；上位科学问题不能据此埋掉 |
| `WEAKENED` | 关键预测没有稳定出现，可信度下降 |
| `FALSIFIED` | 在已经测试的明确条件下，命题被直接反证 |
| `PAUSED` | 有实现或历史痕迹，但证据不完整，或当前不投入 |
| `SUPERSEDED` | 被更新的方案、协议或认识取代 |

### 0.2 以后怎样使用

1. 新想法先查第 6、7、9、11 节，再谈结构。
2. 命中 `FALSIFIED` 的命题时，必须说明新增了什么证据或改变了什么适用条件。
3. 命中 `FAILED_IMPLEMENTATION` 时，必须说明输入、proxy（替代指标）、粒度、loss、ranking、融合位置中至少哪一项发生了实质改变。
4. `OPEN` 和 `PAUSED` 不能被写成“已证明无效”。
5. validation/global-sort 与正式 test/non-global-sort 不直接相减；正式协议以 `AI_CONTEXT.md §11` 为准。
6. 最新通信优先级以 `AI_CONTEXT.md §16` 为准：`lossless_comm` 正在 development 测试，完整结果尚未回传。

### 0.3 证据完整性

- 【事实】本地可核验方法代码、README、YAML 和测试入口。
- 【事实】GSPR、正式通信 benchmark、质量删除诊断、CEIF audit 的数值由 `AI_CONTEXT.md` 记录；本地没有相应服务器原始日志全集。
- 【事实】历史 Where2comm/weather gating、agent Oracle、spatial utility、旧 CURE、detector threshold/NMS 能找到代码或配置，但没有本地可核验结果，也没有最新上下文数值。
- 【结论】这些历史代码只能证明“想法被实现过或准备过”，不能证明“实验跑过”或“性能如何”。

## 1. 研究路线总表

| ID | 名称 | 所属阶段 | 原始科学假设 | 方法 | 当前状态 | 最关键证据 | 是否值得重启 |
|---|---|---|---|---|---|---|---|
| P-001 | detection-only GSPR | 点云/sensing | 点可靠软权重仅靠检测监督也能学到有用抑噪 | GSPR 接检测 loss，无点标签主监督 | `SUPERSEDED` | full_v1 相对它在 Fog/Rain/Snow 再增 `2.449/4.800/3.868` pp；`AI_CONTEXT.md §7.7` | 仅作消融，不重启为主线 |
| P-002 | point-supervised GSPR full_v1 | 点云/sensing | 直接点级可靠性监督、软加权和联合适配能提升恶劣天气检测 | evidential point reliability + soft-weight PillarVFE | `SUPPORTED` | 点 macro AUROC `.999833±.000043`；恶劣天气 AP70 相对 AttFuse 宏增益 `+5.307` pp；`AI_CONTEXT.md §7.6–7.7` | 保留，不修改冻结 v1 |
| P-003 | 旧 snow denoiser / weather feature adapter | 点云/表征 | 显式抑制雪噪 pillar 或恢复天气特征可改善检测 | pillar suppression、feature adapter | `PAUSED` | 代码见 `opencood/models/point_pillar_where2comm.py`、`snow_voxel_denoiser.py`；无结果记录 | 只有找到原结果或新 Oracle 才考虑 |
| C-001 | Where2comm baseline | 通信 | 空间检测置信度足以指导稀疏通信 | confidence map threshold/top-k | `PAUSED` | 配置/实现存在：`opencood/hypes_yaml/point_pillar_where2comm*.yaml`、`where2comm_fuse.py`；无项目内结果表 | 可作历史基线，不当创新重做 |
| C-002 | reliability/density weather gating | 通信/融合 | 点密度和孤立程度可作为天气可靠性，改善通信或特征融合 | `R_density×R_isolation` 软门控 | `PAUSED` | `opencood/utils/weather_reliability.py`；多个 weather YAML；无可核验 AP | 不在无结果审计前重启 |
| C-003 | A0B0 规则 | 通信 | 低自车置信需求 × 高发送端置信供给能抓住大部分有用信息 | 规则请求/响应、固定字节 top-k | `SUPPORTED` | 正式 none→A0B0 大幅增益；256 KiB 约省 `99.0260%` 载荷；`AI_CONTEXT.md §11.2` | 保留为强规则参照 |
| C-004 | A0B0 256 KiB 工作点 | 通信预算 | 256 KiB 可以近乎无损替代 full | 同 C-003，固定 256 KiB | `SUPERSEDED` | 正式 AP70 相对 full：Clean `-8.1364` pp；`AI_CONTEXT.md §12.3, §13, §16` | 不作为默认预算重启 |
| C-005 | 早期 A1B1/A1B0/A0B1 | 学习通信 | 检测 loss 的软反向可学会比规则更好的请求/响应 | 小型 A/B CNN，硬前向软反向 | `FAILED_IMPLEMENTATION` | 旧记录：修正约 `.099`、换块率 0、AP 与 A0B0 相同；`research/通信创新实验建议_20260909.md` | 不能原样重做 |
| C-006 | A0B0 + 有界 residual score | 学习通信 | 从规则起点加小残差可稳定改变边界排序 | `score=A0B0+0.1*tanh(net)` | `FAILED_IMPLEMENTATION` | 排名/选块未改变的历史记录；`gspr_communication/RESIDUAL.md`、`AI_CONTEXT.md §8.1` | 除非直接监督硬集合变化 |
| C-007 | legacy reviewer | 协同复核 | 邻车证据修正 ego 点权重后可改善检测 | 1473 参数点权重修正器 | `FAILED_IMPLEMENTATION` | 权重/PFN 有变化但未显示 AP 收益；`AI_CONTEXT.md §9.1–9.2` | 不原样重训 |
| C-008 | contrast / contrast_gain reviewer | 协同复核 | 与参考消息的差异及学习增益能增强真实消息敏感性 | 反事实差分与 gain | `FAILED_IMPLEMENTATION` | normal/neutral/permuted AP 与 protocol 相同；调整仅 `.000148/.000073`；`gspr_review/AMPLITUDE_SCAN.md` | 不再只换评分公式 |
| C-009 | reviewer BEV-location / fixed scale | 协同复核 | 相同证据作用到 BEV 而非点权重可能更有效 | 局部 BEV 缩放、fixed 0.05/0.10 对照 | `PAUSED` | 代码/配对协议存在；未找到 `location_comparison.json`/`fixed_comparison.json` 结果 | 先找回结果，不重跑 |
| C-010 | learned communication: matching | 学习通信 | 显式请求—响应证据匹配能提高固定预算块效用 | 风险请求 + 收益头 + matching | `FAILED_IMPLEMENTATION` | 正式天气 learned 相对 A0B0 `-9.6621/-6.4562/-8.1534` pp；换块 59%–70%；`AI_CONTEXT.md §11.2` | 只在完成失败归因后再议 |
| C-011 | learned communication: concat | 学习通信对照 | 普通拼接足以利用同样输入 | 同参数量 concat 网络 | `FAILED_IMPLEMENTATION` | 四条件均弱于 matching，天气也远弱于 A0B0；`AI_CONTEXT.md §11.2B–C` | 不原样扩容 |
| C-012 | learned communication: no_u | 学习通信消融 | 去掉 u 会显著变差，从而证明 u 的独立价值 | 移除 selector 的 u 通道后重训 | `FAILED_IMPLEMENTATION` | matching−no_u 仅 `+.9400/+.4398/+.0869/−.0646` pp；`AI_CONTEXT.md §11.2C` | 可保留作消融，不作主线 |
| C-013 | low-r filtering | 质量过滤 | 低平均可靠性块净效用为负，删除应优于随机 | full 通信上按低 r 删除 1/5/10% | `FALSIFIED` | Clean/Fog/Rain 全部下降且差于随机；Snow 仅 10% `+.7325` pp；`AI_CONTEXT.md §12.1` | 禁止作为通用规则重启 |
| C-014 | high-u filtering | 质量过滤 | 高 uncertainty 块普遍有害 | 按高 u 删除 1/5/10% | `WEAKENED` | Snow 10% `+2.6520` pp；Clean/Fog/Rain 约 `−2.4` 至 `−2.6` pp；development/global-sort | 仅保留 Snow 机制问题 |
| C-015 | random filtering control | 质量过滤对照 | 若定向过滤真实识别有害块，应优于等量随机删除 | 同每车删块数随机删除 | `SUPPORTED` | random 10% 四条件均负；区分稀疏化与质量选择；`AI_CONTEXT.md §12.1` | 作为必须保留的对照 |
| C-016 | 单块删除诊断 | 通信机制诊断 | 单块因果干预可揭示有害消息是否普遍 | 每次只删一个候选块，重跑融合/检测 | `SUPPORTED` | 改善/恶化仅 `13/136,24/146,15/107,25/137`；`AI_CONTEXT.md §12.1` | 值得继续分析既有日志 |
| C-017 | ΔL harm Oracle / 联合删除 | 通信反事实诊断 | 检测 loss 标记的有害 A0B0 块联合删除后应改善 AP | 逐块 ΔL、GT 辅助联合删除、随机对照 | `PAUSED` | `gspr_harm/README.md` 指向服务器运行目录，但本地和上下文无数值结果 | 先找回 summary，不重跑 |
| C-018 | A0B0 budget scan | 通信预算 | 增加稀疏预算可找到 12 项都不低于 full 的最小档位 | 256 KiB–32 MiB + full 曲线 | `SUPERSEDED` | 代码完成但未见结果；当前优先级被 `AI_CONTEXT.md §16` 覆盖 | 需要 selection 曲线时再做 |
| C-019 | lossless_full | 全覆盖压缩 | float32 BEV 有足够熵冗余，可 bit-exact 降真实 feature bytes | 三尺度 bit-exact zstd/zlib | `OPEN` | 代码/验收存在；development 正在跑，完整结果未回传；`AI_CONTEXT.md §16` | 当前应完成验证 |
| C-020 | level0_recompute | 全覆盖压缩/重算 | 只发 level0、接收端确定性重算 level1/2 可保 AP并进一步省字节 | peer 级冻结 block1/2 重算 | `OPEN` | raw values 理论少 `42.86%`，不是 wire bytes；结果待回传；`AI_CONTEXT.md §16` | 当前核心候选，不能提前庆祝 |
| F-001 | AttFuse baseline | 融合 | 逐位置多车注意力可作为稳定协同融合底座 | 多尺度 scaled dot-product attention | `SUPPORTED` | 原始正式基线和 full/original_full 一致性；`AI_CONTEXT.md §5, §11.2` | 保留冻结参照 |
| F-002 | 历史 CURE pre/post fusion restorer | 融合 | 反事实协同效用能指导特征恢复或门控 | pre/post restorer、fusion gate | `PAUSED` | 代码见 `counterfactual_*restorer.py`、`train_prediction_distillation.py`；无结果记录 | 不能据代码宣称已失败/成功 |
| F-003 | CEIF 科学假设 | 融合机制 | “未观测”与“可信反证”应触发不同的局部融合修正 | 端点下界、射线上界、冲突保留 | `OPEN` | 设计与有限候选正上限并存；`AI_CONTEXT.md §14–15` | 不允许因 rule 失败误埋 |
| F-004 | CEIF audit | 融合可行性审查 | 冻结模型局部干预可先测候选机会和规则风险 | rule、FP oracle、candidate/full hindsight | `SUPPORTED` | 完整 benchmark-audit 给出机会小、误伤大；`AI_CONTEXT.md §14.1` | 保留为训练前审查范式 |
| F-005 | CEIF no-GT rule | 融合规则 | 局部证据冲突可直接选中应替换/抑制区域 | 预测框覆盖 block 的单来源替换 | `FAILED_IMPLEMENTATION` | 四天气 AP70 `−.8215/−.5474/−1.3489/−1.3691` pp；多数标记为 TP | 禁止原尺度原规则重做 |
| F-006 | CEIF-min | 融合训练 | 局部查询和观测投影可减少旧 rule 的尺度误伤 | query head + local projection + aux_only | `OPEN` | 代码和 22 项本地测试完成；无真实训练/HIP/AP；`AI_CONTEXT.md §15` | 等真实结果，不提前判断 |
| D-001 | detector threshold / NMS 路线 | 检测后处理 | 调阈值或 NMS 可能吸收 evidence suppression 造成的 FP/重复框 | 固定 `score_threshold=.2`、`nms_thresh=.15`，Oracle 工具支持阈值覆盖 | `PAUSED` | 仅配置/后处理代码；未找到独立 sweep 或 AP 结果 | 不得声称已验证；test 上禁止调参 |
| X-001 | agent coalition Oracle / selector | 跨模块 | 不同帧最优协同 agent 子集不同，且可由部署字段预测 | coalition 枚举、frame AP/recall oracle、MLP selector | `PAUSED` | `oracle_eval.py`、`expert_oracle_eval.py`、`train_oracle_selector.py`；无结果文件 | 找回结果后再评价上限 |
| X-002 | spatial utility | 跨模块 | 每个邻车每个 cell 的边际效用可由局部特征学习 | Oracle cache + spatial utility head | `PAUSED` | `train_spatial_utility.py`、`spatial_utility_net.py`；无训练结果 | 需先证明标签与集合/AP一致 |

当前共收录 **32 条研究路线**。其中 `FAILED_IMPLEMENTATION` 8 条，`OPEN` 4 条；其余是已支持的基线/诊断、已削弱命题、被取代路线或证据不完整的历史资产。

## 2. 点云阶段

### P-001：detection-only GSPR

#### 1. 原始问题

在没有逐点可靠性标签时，检测任务本身能否教会 GSPR 抑制天气噪声？

#### 2. 原始科学假设

【假设】只要软权重能降低检测 loss，点质量表征就会自然形成。

#### 3. 为什么当时合理

检测目标与最终 AP 对齐，并且软权重位于 PFN pooling 前，理论上可通过端到端梯度找到有用点。

#### 4. 实际方法

GSPR 接入检测链，但不使用 full_v1 的直接 point reliability 主监督；作为 detection-only 对照训练。

#### 5. 当时期待

恶劣天气 AP 提升，并接近 point-supervised full_v1。

#### 6. 实际观察

【事实｜正式 test、非 global-sort】由差值可知 detection-only 相对原始 AttFuse 在 Fog/Rain/Snow 仍有约 `+.444/+3.586/+.775` pp，但 full_v1 又比它高 `+2.449/+4.800/+3.868` pp。证据：`AI_CONTEXT.md §7.7`。Clean 的原始 AttFuse 只保留两位小数，不做精确反推。

#### 7. 失败或限制发生层

`Training objective / Supervision`：检测 loss 能给部分信号，但不足以替代直接点级监督。

#### 8. 已经具体排除什么

排除“点级监督完全没有额外价值”。

#### 9. 没有排除什么

没有排除检测监督作为联合目标或无点标签数据的辅助价值。

#### 10. 剩余问题与 01 映射

对应 P2、H2.3、Q1：full_v1 的额外收益来自点监督还是联合表征适配，仍未完全拆开。

#### 11. 明确重启条件

只有研究弱监督/无标签迁移时，作为消融重启；不再替代 full_v1 主线。

#### 12. 状态

`SUPERSEDED`：被直接点监督 + 联合训练的 full_v1 取代，不等于完全无效。

### P-002：point-supervised GSPR full_v1

#### 1. 原始问题

如何在不同天气下识别受污染回波，同时尽量不删掉仍有用的弱证据？

#### 2. 原始科学假设

【假设】与 sensing corruption 直接对应的点级监督，加上连续软权重，比硬去点更稳。

#### 3. 为什么当时合理

问题、标签和干预对象都是“点”；reliability 不是由检测框间接猜测，且保留弱点而非不可逆删除。

#### 4. 实际方法

几何/辐射/物理证据融合，输出 reliability/uncertainty；Reliability-aware PillarVFE 在 pooling 前软加权；联合检测微调。

#### 5. 当时期待

点级判别稳定，三种恶劣天气 downstream AP 提升，跨 seed 不靠偶然。

#### 6. 实际观察

【事实】三次 full_v1 point macro AUROC `.999833±.000043`，balanced accuracy `.998437±.000092`；正式 AP70 相对 AttFuse Fog/Rain/Snow `+2.893/+8.386/+4.643` pp。证据：`AI_CONTEXT.md §7.6–7.7`。

#### 7. 成功发生层

`Hypothesis / Supervision / Representation` 均获得支持；但下游块级 task utility 不在其定义内。

#### 8. 已经具体排除什么

显著削弱“GSPR 没学到天气点质量”“收益只是单 seed 偶然”。

#### 9. 没有排除什么

没有证明真实天气逐点校准、跨传感器泛化，也没有证明 point reliability 等于 block utility。

#### 10. 剩余问题与 01 映射

P1、P2、H1.1–H1.4、Q1：点质量为什么不能直接转成通信价值。

#### 11. 明确重启条件

当前不重启、不改冻结源码；只有独立数据或 E1 指向可校准缺陷时，新建隔离实验。

#### 12. 状态

`SUPPORTED`：GSPR 是当前成功资产，通信 proxy 失败不能反向判定 GSPR 失败。

### P-003：旧 snow denoiser / weather feature adapter

#### 1. 原始问题

能否在 pillar/BEV 表征层直接压制雪噪或恢复 clean-like 特征？

#### 2. 原始科学假设

【假设】天气污染在 pillar/feature 上有可识别模式，局部门控或残差恢复可改善检测。

#### 3. 为什么当时合理

雪噪会产生孤立/异常 pillar；特征层比最终框更接近污染发生位置。

#### 4. 实际方法

`SnowVoxelDenoiser` 预测 pillar suppression；`WeatherFeatureAdapter` 做特征恢复；训练脚本含对应辅助 loss。

#### 5. 当时期待

天气检测改善且 clean identity 保持。

#### 6. 实际观察

【事实】代码存在于 `opencood/models/sub_modules/snow_voxel_denoiser.py`、`weather_feature_adapter.py` 和 `train_prediction_distillation.py`。【待验证】未找到本地结果或 `AI_CONTEXT.md` 数值。

#### 7. 失败或限制发生层

`Unknown`：证据不足，不能定位失败层。

#### 8. 已经具体排除什么

没有排除科学假设；只能排除“这是从未实现过的新想法”。

#### 9. 没有排除什么

未排除局部去噪、特征恢复、clean identity 约束的价值。

#### 10. 剩余问题与 01 映射

与 P5/H5.1–H5.3 有弱关联，但旧路线没有可核验结果，不能直接合并结论。

#### 11. 明确重启条件

先找回训练配置、checkpoint、正式/开发协议和结果；找不到时，新实验必须先做 low-cost Oracle，而不是复刻旧训练。

#### 12. 状态

`PAUSED`：代码资产存在，实验状态未知。

## 3. 通信阶段

### C-001：Where2comm baseline

#### 1. 原始问题

在带宽有限时，是否只传检测置信度高的空间区域就够了？

#### 2. 原始科学假设

【假设】前景 confidence 是空间信息价值的可用 proxy。

#### 3. 为什么当时合理

Where2comm 是成熟空间稀疏通信基线；目标区域通常对应高检测响应。

#### 4. 实际方法

对单车 confidence map 做 threshold/top-k mask，再进入 Where2comm fusion。

#### 5. 当时期待

显著降通信率而保持检测性能。

#### 6. 实际观察

【事实】配置和代码存在：`point_pillar_where2comm*.yaml`、`where2comm_fuse.py`。【待验证】仓库没有本项目 Where2comm baseline 的可核验 AP/通信结果。

#### 7. 失败或限制发生层

`Evaluation / Unknown`：不是已确认失败，而是记录不足。

#### 8. 已经具体排除什么

排除“空间 confidence top-k 是本项目第一次提出的创新”。

#### 9. 没有排除什么

没有排除它作为外部/历史基线的价值。

#### 10. 剩余问题与 01 映射

关联 P3、H3.2、Q2：confidence 规则在 Clean 与天气的效用密度是否不同。

#### 11. 明确重启条件

只作为统一协议基线复现，并明确载荷口径；不以改名字方式当新贡献。

#### 12. 状态

`PAUSED`：实现可见，结果不可核验。

### C-002：reliability/density weather gating

#### 1. 原始问题

Where2comm confidence 不知道回波是否被天气污染，能否加物理质量门？

#### 2. 原始科学假设

【假设】局部点密度不足和孤立点可稳定表示天气不可靠区域。

#### 3. 为什么当时合理

天气会改变点数和空间孤立性；该统计不需额外大网络。

#### 4. 实际方法

构造 `R_density`、`R_isolation`，乘成 `R_final`；可作用于 communication confidence 或 feature，带 reliability floor。

#### 5. 当时期待

恶劣天气减少有害消息，同时 clean 影响小。

#### 6. 实际观察

【事实】实现见 `opencood/utils/weather_reliability.py`、`where2comm_fuse.py`；多个 density/isolation YAML 存在。【待验证】无本地 AP 记录。

#### 7. 失败或限制发生层

`Proxy / Evaluation / Unknown`：density 是污染 proxy，不是 task utility；但本路线本身缺结果。

#### 8. 已经具体排除什么

后续 GSPR 删除实验排除了“质量均值可无条件直接当块价值”，但不能倒推此旧实现具体 AP。

#### 9. 没有排除什么

没有排除 density 作为辅助 sensing 特征或条件变量。

#### 10. 剩余问题与 01 映射

P1、H1.1/H1.2、Q1。

#### 11. 明确重启条件

必须先证明 density 在控制目标覆盖后预测边际 utility；否则不重训 gate。

#### 12. 状态

`PAUSED`：有代码、无可核验结果；与 low-r/high-u 失败直觉高度相似。

### C-003：A0B0 规则

#### 1. 原始问题

不用学习模块，规则供需能拿回多少协同收益？

#### 2. 原始科学假设

【假设】ego 低 confidence 区域需要信息，sender 高 confidence 区域能提供信息。

#### 3. 为什么当时合理

规则简单、可解释、无需额外训练，是检验复杂 selector 是否真的增值的硬基线。

#### 4. 实际方法

`request=1−ego confidence`；`response=request×sender confidence`；固定总字节 top-k，三尺度块打包。

#### 5. 当时期待

在极低载荷下明显优于 none，并成为学习模块必须超过的起点。

#### 6. 实际观察

【事实｜正式 test】none→A0B0 AP70：Clean `.577708→.751121`、Fog `.401465→.626587`、Rain `.406726→.626826`、Snow `.283999→.504870`；平均 `244003.7` bytes/frame。`AI_CONTEXT.md §11.2A`。

#### 7. 成功发生层

`Selection / Budget`：规则在 256 KiB 下极强；但不是无损。

#### 8. 已经具体排除什么

排除“少量通信几乎没有任务价值”；也给 learned selector 建立了强参照。

#### 9. 没有排除什么

没有证明 A0B0 是最优选择，也没有证明 256 KiB 合理。

#### 10. 剩余问题与 01 映射

P3、H3.1–H3.4、Q2。

#### 11. 明确重启条件

保留规则做对照；预算和表示方式可变，不能把旧 256 KiB 一起冻结成结论。

#### 12. 状态

`SUPPORTED`：强规则基线；“规则有效”和“旧预算可接受”必须分开。

### C-004：A0B0 256 KiB 工作点

#### 1. 原始问题

能否用约 256 KiB/场景替代约 25 MB full communication？

#### 2. 原始科学假设

【假设】A0B0 选中的少量块足够保持 full AP。

#### 3. 为什么当时合理

none→A0B0 已拿回大量协同收益，天气下与 full 的差距看起来较小。

#### 4. 实际方法

固定每场景 256 KiB 应用层预算，包括请求、索引、头部、三尺度特征。

#### 5. 当时期待

四天气、各 AP 指标近乎无损。

#### 6. 实际观察

【事实｜正式 test】相对 full AP70：Clean `−8.1364`、Fog `−1.3076`、Rain `−2.0681`、Snow `−1.3378` pp；载荷约少 `99.0260%`。`AI_CONTEXT.md §12.3`。

#### 7. 失败或限制发生层

`Budget / Selection / Fusion` 尚未拆开；不能只说“预算太小”。

#### 8. 已经具体排除什么

排除“256 KiB 近乎无损”的强命题。

#### 9. 没有排除什么

没有排除 A0B0 规则、更大预算、全覆盖压缩或重算。

#### 10. 剩余问题与 01 映射

P3、H3.1–H3.4、Q2；最低成本分析 E4。

#### 11. 明确重启条件

仅作为历史 budget point 或 selection 曲线端点；不得再当默认“无损”配置。

#### 12. 状态

`SUPERSEDED`：预算优先级被 `lossless_comm` 路线覆盖。

### C-005：早期 A1B1/A1B0/A0B1

#### 1. 原始问题

规则请求和响应没有直接表示 task utility，能否端到端学习更好的供需评分？

#### 2. 原始科学假设

【假设】检测 loss 通过软选择代理能让 A/B 改变真实 top-k 集合并提高 AP。

#### 3. 为什么当时合理

A/B 输入包含语义、质量、支持；学习模型理论上比乘法规则更灵活。

#### 4. 实际方法

冻结前端，只训练小型 A/B 网络；硬前向、软反向；A0B1/A1B0/A1B1 因子对照。

#### 5. 当时期待

学习组选块发生变化，并超过 A0B0。

#### 6. 实际观察

【事实｜历史开发记录，原始结果缺失】平均绝对修正约 `.099`，换块率 0，AP 与 A0B0 相同。`research/通信创新实验建议_20260909.md §训练如何改变`。

#### 7. 失败或限制发生层

`Ranking / Optimization`：连续分数变了，但没有跨过离散 top-k 边界。

#### 8. 已经具体排除什么

排除“只要分数数值变化，实际消息自然会变化”。

#### 9. 没有排除什么

没有排除 task-aware selection；也没有证明全部梯度错误或 GSPR 无用。

#### 10. 剩余问题与 01 映射

P4；这是后期大量换块失败之前的第一种失败，不能与 H4.1–H4.4 混为一谈。

#### 11. 明确重启条件

必须直接检查硬集合变化，并用能跨 ranking boundary 的监督/搜索；不能只换 CNN 深度。

#### 12. 状态

`FAILED_IMPLEMENTATION`：科学问题存活，旧软代理实现死亡。

### C-006：A0B0 + 有界 residual score

#### 1. 原始问题

能否避免 A1B1 从零学习的不稳定，直接在强规则边界上做小修正？

#### 2. 原始科学假设

【假设】小而有正负号的 residual 足以把少量边界块换得更好。

#### 3. 为什么当时合理

保留 A0B0 起点，减少探索空间，也允许被规则低估的块上升。

#### 4. 实际方法

3505 参数 correction；`score=request×confidence+0.1*tanh(output)`；零初始化。

#### 5. 当时期待

初始严格等于 A0B0，训练后只在边界附近发生少量有效替换。

#### 6. 实际观察

【事实｜历史记录】分数变化但实际 ranking/选块不变，AP 未变。设计边界见 `gspr_communication/RESIDUAL.md`；汇总结论见 `AI_CONTEXT.md §8.1` 和研究记录。

#### 7. 失败或限制发生层

`Ranking / Architecture amplitude`：0.1 有界残差未跨越 top-k margin。

#### 8. 已经具体排除什么

排除“在同一 proxy 上加一个很小 residual 就能解决选择”的版本。

#### 9. 没有排除什么

没有排除更好的标签、候选池或集合级目标。

#### 10. 剩余问题与 01 映射

P4；与后期 matching 的“大量换块但更差”形成明确两阶段证据。

#### 11. 明确重启条件

新方案必须展示 ranking margin、replacement fraction 和任务后果；不得只报告 score change。

#### 12. 状态

`FAILED_IMPLEMENTATION`。

### C-007：legacy reviewer

#### 1. 原始问题

邻车能否不只发 BEV，还发局部证据来复核 ego 的 GSPR 点权重？

#### 2. 原始科学假设

【假设】空间对齐的邻车可靠/噪声/未知证据能纠正 ego 的点质量判断，并传到检测。

#### 3. 为什么当时合理

多车观测互补；reviewer 直接作用于 GSPR 权重，链路短且参数少。

#### 4. 实际方法

量化证据格 + 1473 参数修正器；冻结 GSPR/AttFuse；复核占总 256 KiB 的一部分。

#### 5. 当时期待

review > protocol，最好超过 A0B0；真实消息优于 shuffled。

#### 6. 实际观察

【事实｜开发/两位小数 AP】正常复核每帧约改 Clean 864 点、Weather 412 点，PFN 相对 L1 变化约 `.5%`，但未显示检测收益；补回漏检数未变。`AI_CONTEXT.md §9.1–9.2`。

#### 7. 失败或限制发生层

`Representation / Architecture / Training objective`：链路生效，但幅度小、内容敏感性弱，且“权重改变”未对齐任务增益。

#### 8. 已经具体排除什么

排除“只要邻车证据改变点权重/PFN，检测必然改善”。

#### 9. 没有排除什么

没有排除多车证据复核的科学问题，也未证明消息完全未被使用。

#### 10. 剩余问题与 01 映射

与 Q1、H1.1、H2.3 相关：sensing correction 和 task utility 仍需分开。

#### 11. 明确重启条件

先有独立正确性标签或反事实 task gain，且真实消息必须明显优于置换/常量对照。

#### 12. 状态

`FAILED_IMPLEMENTATION`。

### C-008：contrast / contrast_gain reviewer

#### 1. 原始问题

legacy reviewer 是否因为共同偏置、未显式比较参考消息而忽略了邻车内容？

#### 2. 原始科学假设

【假设】真实证据相对 neutral reference 的差异，再乘可学习 gain，会产生更有任务意义的修正。

#### 3. 为什么当时合理

固定支持诊断显示置换消息输出差仅为调整幅度约 2.1%，提示内容敏感性弱。

#### 4. 实际方法

`contrast` 1472 参数、`contrast_gain` 1729 参数；normal/neutral/permuted/protocol 对照。

#### 5. 当时期待

真实消息 AP 高于 neutral/permuted，且 gain 不塌缩。

#### 6. 实际观察

【事实｜开发记录】legacy、contrast、contrast_gain 的 normal/neutral/permuted AP 与 protocol 相同；平均调整约 `.000148/.000073`，gain 约 `.4996`。`gspr_review/AMPLITUDE_SCAN.md`。

#### 7. 失败或限制发生层

`Architecture / Optimization / Magnitude`；是否仅幅度受限尚未由完成的 amplitude scan 结果证明。

#### 8. 已经具体排除什么

排除“把输入改成参考差分，就会自动带来检测收益”。

#### 9. 没有排除什么

未排除参考差分作为诊断；也未排除更直接任务监督。

#### 10. 剩余问题与 01 映射

映射 Q1；不属于后期 learned block selection 的同一实现。

#### 11. 明确重启条件

只有找回完整 amplitude/配对结果，或新实验能验证真实消息—置换消息的任务差异。

#### 12. 状态

`FAILED_IMPLEMENTATION`。

### C-009：reviewer BEV-location / fixed scale

#### 1. 原始问题

点权重改动太弱时，把相同证据作用到 BEV 特征是否更容易影响检测？

#### 2. 原始科学假设

【假设】失败主要是干预位置/幅度，而非证据信息本身。

#### 3. 为什么当时合理

PFN 变化只有约 `.5%`；BEV 更接近融合和检测头。

#### 4. 实际方法

将点级建议汇聚到三个尺度 BEV，执行局部 `feature×(1+adjustment)`；另有固定 `+5%/+10%` 缩放对照。

#### 5. 当时期待

bev_normal > protocol、bev_permuted，并最好超过 A0B0；learned 应优于固定缩放。

#### 6. 实际观察

【事实】代码、严格消息重放协议和服务器目录引用存在；`FIXED_BEV.md` 称复用已完成配对实验。【待验证】本地未找到最终 JSON，最新上下文没有 AP 结论。

#### 7. 失败或限制发生层

`Evaluation / Unknown`。

#### 8. 已经具体排除什么

只能排除“BEV 作用位置是未实现的新想法”。

#### 9. 没有排除什么

没有证据判定 BEV 位置成功或失败。

#### 10. 剩余问题与 01 映射

与 H2.3、Q1 弱关联；不能凭缺失结果纳入假设墓地。

#### 11. 明确重启条件

先找回 `location_comparison.json`、`fixed_comparison.json` 和协议哈希；结果不可恢复时才考虑小规模复现。

#### 12. 状态

`PAUSED`。

### C-010：learned communication — matching

#### 1. 原始问题

A0B0 只看 confidence，能否用 GSPR 证据缺口和真实换块收益选得更好？

#### 2. 原始科学假设

【假设】请求—响应显式匹配 + 单块 task gain 监督能提高固定预算效用。

#### 3. 为什么当时合理

输入区分可靠、噪声、未知和支持；教师用实际等字节替换的检测 loss 差，不再只靠软梯度。

#### 4. 实际方法

风险头生成请求，收益头预测候选替换 gain；matching 结构进行显式配对。

#### 5. 当时期待

同 protocol、同字节下 learned > protocol，并超过 A0B0，天气质量差异越大越受益。

#### 6. 实际观察

【事实｜正式 test】protocol→learned AP70：Clean `+.1723` pp，Fog/Rain/Snow `−5.4792/−3.0576/−3.9896` pp；相对 A0B0 天气更差。换块 59.25%–70.01%。`AI_CONTEXT.md §11.2`。

#### 7. 失败或限制发生层

至少包含 `Selection/Ranking`；候选请求、单块到集合、loss/AP、domain shift 尚未分开。

#### 8. 已经具体排除什么

排除“后期失败只是网络没改变消息”。

#### 9. 没有排除什么

没有排除 task-aware communication，也没有证明 matching 信号毫无价值；matching 仍略优于 concat。

#### 10. 剩余问题与 01 映射

P4、H4.1–H4.4、Q3；最低成本 E3。

#### 11. 明确重启条件

先完成候选池—教师—集合遗憾分解；若可部署字段 AUPRC 接近基率，则终止当前 selector 家族。

#### 12. 状态

`FAILED_IMPLEMENTATION`。

### C-011：learned communication — concat

#### 1. 原始问题

matching 的优势是否只是多了参数，而不是显式匹配机制？

#### 2. 原始科学假设

【假设】若普通拼接同样有效，复杂 matching 结构没有独立价值。

#### 3. 为什么当时合理

同参数量对照能隔离结构形式。

#### 4. 实际方法

使用相同输入和预算，以普通 concat 替代 matching，重新训练收益头。

#### 5. 当时期待

若 matching 机制成立，matching 应稳定高于 concat；两者至少应接近 A0B0。

#### 6. 实际观察

【事实｜正式 test】matching−concat AP70 为 Clean/Fog/Rain/Snow `+.5665/+1.1827/+.5848/+.8680` pp；但两者在天气都远低于 A0B0。`AI_CONTEXT.md §11.2C`。

#### 7. 失败或限制发生层

`Architecture` 对比显示 matching 有有限信号；整体失败主因不只是 concat 表达能力。

#### 8. 已经具体排除什么

排除“随便拼接这些输入就足以解决固定预算选择”。

#### 9. 没有排除什么

没有排除显式匹配；其小幅优势不足以证明可部署有效。

#### 10. 剩余问题与 01 映射

P4、H4.1–H4.4、Q3。

#### 11. 明确重启条件

不通过扩大 concat 网络重启；只有上位标签/集合问题先被修正后，concat 才可继续作对照。

#### 12. 状态

`FAILED_IMPLEMENTATION`。

### C-012：learned communication — no_u

#### 1. 原始问题

GSPR uncertainty `u` 是否给 selector 提供 reliability/probability 之外的独立价值？

#### 2. 原始科学假设

【假设】去掉 u 后四天气都应稳定变差。

#### 3. 为什么当时合理

u 表示证据不足，不等于低可靠概率，理论上能区分“有噪声”和“不知道”。

#### 4. 实际方法

去掉 selector 的 u 通道，保持冻结 GSPR 其余部分，重新训练同类网络。

#### 5. 当时期待

matching 在所有天气显著超过 no_u。

#### 6. 实际观察

【事实｜正式 test】matching−no_u AP70 为 `+.9400/+.4398/+.0869/−.0646` pp（Clean/Fog/Rain/Snow）；Snow 略反向。整体两者都弱于 A0B0。`AI_CONTEXT.md §11.2C`。

#### 7. 失败或限制发生层

`Proxy / Training / Domain`：u 的独立贡献不稳定，不能用本实现判定 u 本身无信息。

#### 8. 已经具体排除什么

排除“u 在当前 selector 中跨天气稳定必要”。

#### 9. 没有排除什么

没有排除 u 用于 sensing calibration、诊断或不同任务目标。

#### 10. 剩余问题与 01 映射

P1、P4、H1.4、H4.3/H4.4、Q1/Q3。

#### 11. 明确重启条件

必须先证明 u 对真实边际 utility 的条件信息，而不是再把 u 拼入更大网络。

#### 12. 状态

`FAILED_IMPLEMENTATION`；上位“未知与噪声应区分”仍 `OPEN`。

### C-013：low-r filtering

#### 1. 原始问题

全通信是否包含平均 point reliability 很低、删掉反而更好的块？

#### 2. 原始科学假设

【假设】低 r block 的净检测效用普遍为负。

#### 3. 为什么当时合理

GSPR 对天气点可靠性判别很强，低 r 看起来像最直接的有害消息标记。

#### 4. 实际方法

在 full communication 上按每邻车块级低 r 排序，硬删 1%、5%、10%，与等数量 random 对照。

#### 5. 当时期待

四天气至少不降，并优于随机删除。

#### 6. 实际观察

【事实｜完整 validation、在线天气、global-sort】Clean/Fog/Rain 所有比例都降且差于随机；Snow 只有 10% 约 `+.7325` pp。`AI_CONTEXT.md §12.1`。

#### 7. 失败或限制发生层

`Proxy / Granularity / Task objective`：点污染质量不等于整块边际 utility。

#### 8. 已经具体排除什么

在当前冻结系统和块定义下，排除“平均低 r 是跨天气通用删块规则”。

#### 9. 没有排除什么

不否定 GSPR 点级软权重；不否定更细粒度、target-aware utility。

#### 10. 剩余问题与 01 映射

P1、H1.1–H1.4、Q1；最低成本 E1。

#### 11. 明确重启条件

新方案须证明控制目标覆盖/唯一支持后仍有负 utility；否则禁止继续调 r 阈值。

#### 12. 状态

`FALSIFIED`：限定于“低平均 r 可直接做通用 block filtering”。

### C-014：high-u filtering

#### 1. 原始问题

证据不足的高 uncertainty 块是否比低 reliability 更能指示有害信息？

#### 2. 原始科学假设

【假设】高 u block 应在各种天气下优先删除。

#### 3. 为什么当时合理

u 与可靠概率不同，可能识别“判断本身不可信”的区域。

#### 4. 实际方法

在 full communication 上按高 u 删除 1%、5%、10%，同删块数随机对照。

#### 5. 当时期待

跨天气稳定优于 full 和 random。

#### 6. 实际观察

【事实｜开发 validation/global-sort】高 u 10%：Clean/Fog/Rain `−2.4138/−2.5761/−2.4561` pp，Snow `+2.6520` pp；Snow 同时 FP `+601`。`AI_CONTEXT.md §12.1`。

#### 7. 失败或限制发生层

`Dataset / Weather-specific mechanism / Proxy`。

#### 8. 已经具体排除什么

排除跨天气统一 high-u 删除规则。

#### 9. 没有排除什么

未排除模拟 Snow 中真实组合效应；也未在正式 OPV2V-W Snow 验证。

#### 10. 剩余问题与 01 映射

P5、H5.1–H5.4、Q4；E6 后才考虑 E7。

#### 11. 明确重启条件

先用已有 scene/单块日志证明非少数场景、非一般稀疏化；再做固定小样本正式 Snow，不扫比例。

#### 12. 状态

`WEAKENED`：通用命题死亡，Snow 局部机制未埋。

### C-015：random filtering control

#### 1. 原始问题

定向删除收益究竟来自“识别坏块”，还是仅来自稀疏化/注意力重归一化？

#### 2. 原始科学假设

【假设】若质量 proxy 有信息，定向过滤应稳定超过同删块数 random。

#### 3. 为什么当时合理

没有 random，就无法解释删除本身的正则化效应。

#### 4. 实际方法

按每车相同删除数量随机抽块，10% 结果作为主要对照。

#### 5. 当时期待

随机大多伤害；有用 proxy 应显著更好。

#### 6. 实际观察

【事实】random 10% 在 Clean/Fog/Rain/Snow 分别 `−1.1304/−1.2435/−1.1272/−.8218` pp。high-u Snow 明显优于它，low-r 多数条件比它更差。`AI_CONTEXT.md §12.1`。

#### 7. 成功发生层

`Evaluation / Control`：成功提供了最低必要反事实。

#### 8. 已经具体排除什么

排除“所有删除都天然提升”；也证明 low-r 伤害不是仅因少通信这么简单。

#### 9. 没有排除什么

普通 random 未匹配空间簇、目标覆盖和来源结构，不能彻底排除一般稀疏化解释。

#### 10. 剩余问题与 01 映射

P5、H5.2、E6。

#### 11. 明确重启条件

未来任何 filtering 都必须保留同预算 random，最好再做结构匹配 random。

#### 12. 状态

`SUPPORTED`：作为诊断对照成功，不是可部署方法。

### C-016：单块删除诊断

#### 1. 原始问题

批量过滤的 AP 变化是否由大量单个有害块累积而来？

#### 2. 原始科学假设

【假设】如果有害块普遍存在，单独删除候选时应经常改善检测。

#### 3. 为什么当时合理

固定其他消息只删一个块，比训练 selector 更直接地测边际作用。

#### 4. 实际方法

每邻车从 low-r/high-u/random 各取 2 个候选，去重后逐块重放，记录 loss、匹配和 FP 变化。

#### 5. 当时期待

定向候选有明显更高改善率。

#### 6. 实际观察

【事实】候选约 2.4–2.5 万/天气；改善/恶化仅 `13/136、24/146、15/107、25/137`。改善稀少，恶化更多。`AI_CONTEXT.md §12.1`。

#### 7. 成功或限制层

`Counterfactual diagnosis` 成功；但候选非穷举，事件指标不等于 AP，单块不等于联合。

#### 8. 已经具体排除什么

显著削弱“冻结 full 系统里到处都是可独立删除的坏块”。

#### 9. 没有排除什么

未排除 Snow 联合非加性、未抽中候选和 score/IoU margin 变化。

#### 10. 剩余问题与 01 映射

P5、H5.1/H5.4、E6、Q4。

#### 11. 明确重启条件

优先统计既有日志；只有字段不足时才做轻量 hook，不重新训练。

#### 12. 状态

`SUPPORTED`：支持的是“有害单块稀少”这一诊断，不是“所有消息都无害”。

### C-017：ΔL harm Oracle / 联合删除

#### 1. 原始问题

A0B0 已发送块中，是否存在按检测 loss 可定义的有害区域，联合删除能否提高 AP？

#### 2. 原始科学假设

【假设】单删使 loss 下降的块联合删除后仍会改善检测，并优于等量随机。

#### 3. 为什么当时合理

它直接使用固定背景反事实，不依赖 reliability proxy。

#### 4. 实际方法

逐块计算 `ΔL=L(base)−L(drop)`；将 `ΔL>ε` 区域联合删除；另构造新候选和随机对照。

#### 5. 当时期待

联合 AP 优于 A0B0/random，并给后续 utility 学习提供标签。

#### 6. 实际观察

【事实】`gspr_harm/README.md` 引用了完整 1980 帧服务器目录和重放入口。【待验证】本地无 `summary.json`，`AI_CONTEXT.md` 无数值，不能写成功或失败。

#### 7. 失败或限制发生层

`Evaluation provenance / Unknown`；另有单块 loss 与联合 AP 错位风险。

#### 8. 已经具体排除什么

排除“反事实有害性诊断从未实现”。

#### 9. 没有排除什么

科学结果全部未决；不能用后来的 full quality deletion 代替这组 A0B0/ΔL 结果。

#### 10. 剩余问题与 01 映射

与 H4.1/H4.3、E3 和 P5 的集合非加性相关。

#### 11. 明确重启条件

先找回服务器 `summary.json/regions.jsonl/frames.jsonl`；找不到时不凭 README 复述结论。

#### 12. 状态

`PAUSED`。

### C-018：A0B0 budget scan

#### 1. 原始问题

256 KiB 太小的话，最小无下降稀疏预算是多少？

#### 2. 原始科学假设

【假设】存在某个 1–32 MiB 档位，使四天气 12 项 AP 都不低于 full。

#### 3. 为什么当时合理

旧工作点在天气已接近 full，Clean 可能只需更多块。

#### 4. 实际方法

development 扫 256 KiB、1/2/4/8/16/32 MiB、full；只在 12 项原始值全通过时选最小档，再固定正式 test。

#### 5. 当时期待

获得零下降 selection budget 和预算—AP 曲线。

#### 6. 实际观察

【事实】代码与协议完成：`gspr_evidence/BUDGET_SCAN.md`。【待验证】无服务器结果。最新 `AI_CONTEXT.md §16` 明确其不再是 lossless_comm 前置门禁。

#### 7. 失败或限制发生层

`Research priority`，不是实验失败。

#### 8. 已经具体排除什么

排除“256 KiB 无需重新审视”。

#### 9. 没有排除什么

没有回答预算曲线，也没有否定稀疏 selection。

#### 10. 剩余问题与 01 映射

P3、Q2、E5；在 `01 §11` 已降级为 B 级 selection 对照。

#### 11. 明确重启条件

论文需要 selection curve，或 lossless 方案节省不足时再运行；不在 test 扫预算。

#### 12. 状态

`SUPERSEDED`：当前优先级被全覆盖压缩取代，代码仍可用。

### C-019：lossless_full

#### 1. 原始问题

不删任何空间信息，float32 三尺度 BEV 自身能压缩多少？

#### 2. 原始科学假设

【假设】特征位模式存在足够熵冗余，bit-exact 编码能显著降 feature bytes 且 AP 完全保持。

#### 3. 为什么当时合理

BEV 中有大量零值和重复结构；无损编码不改变任务信息。

#### 4. 实际方法

三尺度 float32 bit-exact 编码，完整 uint32 byte-shuffle 或 `+0.0` bitmap + 非零 words，zstd/zlib 二选一。

#### 5. 当时期待

解码 bit pattern、logits、AP 与 raw_full 一致，真实 feature payload 明显下降。

#### 6. 实际观察

【待验证】development 已启动，完整 Clean/Fog/Rain/Snow 结果未回传。代码见 `lossless_comm/`；`AI_CONTEXT.md §16`。

#### 7. 风险发生层

`Codec / Measurement / System cost`：协议字节口径和 D2H/H2D 尚需统一。

#### 8. 已经具体排除什么

尚未排除科学假设；本地单元测试不能替代服务器 AP。

#### 9. 没有排除什么

压缩率、端到端延迟、正式 benchmark 均未决。

#### 10. 剩余问题与 01 映射

P8、H8.1/H8.3/H8.4、Q2/Q6、更新后的 E12。

#### 11. 明确重启条件

当前不是重启而是等待已有 development；先 bit-exact，再 logits/AP，再统一 bytes/time。

#### 12. 状态

`OPEN`：不得放入墓地、不得提前宣称无损。

### C-020：level0_recompute

#### 1. 原始问题

三尺度 backbone 串行依赖时，是否无需重复发送可由 level0 确定性重算的 level1/2？

#### 2. 原始科学假设

【假设】只发完整 level0，在接收端逐 peer 重算可保持任务输出，同时比普通无损编码进一步省字节。

#### 3. 为什么当时合理

冻结 backbone 满足 `level0→level1→level2`；每 cell level1/2 占 `768/1792=42.86%` raw values。

#### 4. 实际方法

level0 bit-exact 传输；接收端对每个 peer 单独运行冻结 block1/2，再走原多尺度 AttFuse。

#### 5. 当时期待

AP30/50/70 不降，wire bytes 进一步下降；额外 recompute 成本可接受。

#### 6. 实际观察

【待验证】完整结果未回传；`42.86%` 只是 raw-value 理论减少。`AI_CONTEXT.md §16.3–16.9`。

#### 7. 风险发生层

`Numerical representation / System cost / Measurement`：peer batch shape 可引入浮点差，且带宽可能转成 receiver 计算。

#### 8. 已经具体排除什么

尚未排除；不能用结构关系直接证明 AP 无损。

#### 9. 没有排除什么

真实 wire reduction、正式天气稳定性、随 peer 数扩展的系统代价均未知。

#### 10. 剩余问题与 01 映射

P8、H8.2–H8.4、Q6、E12。

#### 11. 明确重启条件

完成当前 development；若数值/AP/bytes 任一门槛失败，不进入 learned codec 叠加。

#### 12. 状态

`OPEN`。

## 4. 融合阶段

### F-001：AttFuse baseline

#### 1. 原始问题

如何在共同 BEV 网格上融合多车同位置特征？

#### 2. 原始科学假设

【假设】逐位置 self-attention 能根据多车特征关系形成有效协同表示。

#### 3. 为什么当时合理

结构简单、已有 OPV2V 基线，且 full/original_full 可直接核对。

#### 4. 实际方法

多尺度逐位置 scaled dot-product attention，无显式天气质量门、未知/反证状态。

#### 5. 当时期待

稳定提供全通信协同上限和下游底座。

#### 6. 实际观察

【事实】正式 full 与 original_full 六位 AP 一致；full AP70 Clean/Fog/Rain/Snow `.832485/.639663/.647507/.518248`。`AI_CONTEXT.md §11.2`。

#### 7. 成功或限制层

`Fusion baseline` 成功；但对稀疏输入/质量证据没有显式语义。

#### 8. 已经具体排除什么

排除通信 full 路径与原模型语义明显不一致。

#### 9. 没有排除什么

没有证明 attention weight 等于来源因果 utility，也未证明 full 每个块都无害。

#### 10. 剩余问题与 01 映射

P3/H3.4、P5/H5.1、Q2。

#### 11. 明确重启条件

作为冻结 baseline 保留；新融合必须用相同输入/协议并解释超过它的机制。

#### 12. 状态

`SUPPORTED`。

### F-002：历史 CURE pre/post fusion restorer

#### 1. 原始问题

协同有时有害时，能否用反事实 utility 恢复天气特征并门控邻车贡献？

#### 2. 原始科学假设

【假设】clean/weather、all/agent-removed 的边际差可监督可部署的 recoverability/harm/utility 表征。

#### 3. 为什么当时合理

反事实比 density/confidence 更接近实际任务贡献。

#### 4. 实际方法

pre-fusion `CounterfactualUtilityRestorer`、post-fusion restorer、可选 fusion gate；多项恢复/伤害/utility/identity loss。

#### 5. 当时期待

在恶劣天气恢复有用协同特征，同时保持 clean identity。

#### 6. 实际观察

【事实】完整代码路径存在：`counterfactual_utility_restorer.py`、`counterfactual_fusion_restorer.py`、`train_prediction_distillation.py`。【待验证】没有本地训练结果或上下文 AP。

#### 7. 失败或限制发生层

`Unknown / Evidence provenance`。

#### 8. 已经具体排除什么

排除“反事实恢复/融合门控是从未实现的新概念”。

#### 9. 没有排除什么

没有证据判定 CURE 科学假设；也不能与当前 CEIF 混为同一机制。

#### 10. 剩余问题与 01 映射

与 Q3/Q5 的 task utility 和可识别性有关，但没有直接结果映射。

#### 11. 明确重启条件

先找回配置、checkpoint、结果和 protocol；否则只可复用思想，不得声称“重新发现”。

#### 12. 状态

`PAUSED`。

### F-003：CEIF 科学假设

#### 1. 原始问题

融合器把“没有观测”和“有可信自由空间反证”都当成缺特征，是否导致补全与纠错混淆？

#### 2. 原始科学假设

【假设】可信端点下界与可信射线上界应对局部特征产生不同方向的修正；未知应保持不动。

#### 3. 为什么当时合理

GSPR 能给点证据，多车几何提供互补观测；AttFuse 本身没有 missing/negative evidence 语义。

#### 4. 实际方法

设计层面：端点/射线观测区间、冲突显式保留、局部求解；见 `CEIF_创新融合模块设计_20260915.md`。

#### 5. 当时期待

补全 baseline FN，纠正有反证的虚假占用，同时不误伤未知和正确 TP。

#### 6. 实际观察

【事实】audit 的 candidate/full finite hindsight 均为正，但实际 rule 为负；正差混合 correct/complete。`AI_CONTEXT.md §14.1`。

#### 7. 当前瓶颈层

`Hypothesis` 尚开放；`Proxy / Granularity / Selection` 已暴露问题。

#### 8. 已经具体排除什么

排除“当前整框 rule 已经验证 CEIF 机制”。

#### 9. 没有排除什么

不能由 rule 失败否定 missing evidence ≠ negative evidence。

#### 10. 剩余问题与 01 映射

P6/P7、H6.2–H6.4、H7.3–H7.4、Q5。

#### 11. 明确重启条件

E8 拆 correct/complete，E9 测可识别性，E10 测局部 finite Oracle；结果支持后才扩大训练。

#### 12. 状态

`OPEN`。

### F-004：CEIF audit

#### 1. 原始问题

在花费完整训练前，冻结表示中是否真的有可恢复 AP，规则又会误伤多少？

#### 2. 原始科学假设

【假设】有限候选 counterfactual 能把“没有机会”和“不会选择”分开。

#### 3. 为什么当时合理

Oracle/干预比直接训练新网络便宜，也更容易设置 kill criterion。

#### 4. 实际方法

full baseline；evidence FP ideal cleanup；随机；no-GT rule；evidence/full pool hindsight；固定正式 benchmark-audit。

#### 5. 当时期待

先确认候选机会，再判断证据规则是否有 precision。

#### 6. 实际观察

【事实】evidence candidate hindsight `+.1558/+.6094/+.8369/+1.0375` pp；full finite pool `+.3616/+.7931/+1.2526/+2.0826` pp；有害率远高于有益率。`AI_CONTEXT.md §14.1`。

#### 7. 成功发生层

`Evaluation / Counterfactual diagnosis`：成功证明“有限机会存在，但规则失败”。

#### 8. 已经具体排除什么

排除“候选池完全没有正动作”和“现有 rule 足够安全”两个极端。

#### 9. 没有排除什么

不是 CEIF 理论上限；没有分 correct/complete，也未逐样本核验服务器 action 日志。

#### 10. 剩余问题与 01 映射

P6/P7、Q5；E8–E10。

#### 11. 明确重启条件

审查范式应被复用；不得把 hindsight 当预计训练增益。

#### 12. 状态

`SUPPORTED`：支持的是审查结论，不是 CEIF 最终有效。

### F-005：CEIF no-GT rule

#### 1. 原始问题

不使用 GT 时，局部端点/射线冲突能否直接决定应替换哪块特征？

#### 2. 原始科学假设

【假设】被证据冲突标记的高分框富集 FP，单来源替换能纠正它们。

#### 3. 为什么当时合理

可信穿越与预测占用冲突，看起来像明确反证。

#### 4. 实际方法

局部冲突触发预测框覆盖 block 的整块单来源 feature replacement。

#### 5. 当时期待

标记框错误率高，rule AP 至少在天气条件改善。

#### 6. 实际观察

【事实｜正式 benchmark-audit】rule AP70 四条件均降 `−.8215/−.5474/−1.3489/−1.3691` pp；冲突标记 FP率 Clean/Fog 仅 `0%/.51%`，Rain/Snow `6.61%/22.76%`。`AI_CONTEXT.md §14.1`。

#### 7. 失败或限制发生层

`Proxy / Granularity / Intervention`：局部证据触发整框覆盖块替换，错误基率低。

#### 8. 已经具体排除什么

排除当前 conflict rule + 整块替换方案。

#### 9. 没有排除什么

没有排除局部 cell 投影、补全动作或 CEIF 科学假设。

#### 10. 剩余问题与 01 映射

P6、H6.1–H6.4、Q5。

#### 11. 明确重启条件

新版本必须缩小作用尺度、分 correct/complete，并超过 aux_only/置换对照；禁止原样调阈值。

#### 12. 状态

`FAILED_IMPLEMENTATION`。

### F-006：CEIF-min

#### 1. 原始问题

旧 rule 的主要失败是否来自尺度过粗，而局部观测投影仍可学？

#### 2. 原始科学假设

【假设】可解码的局部占用查询 + 一步投影能在小邻域利用端点/射线约束，并优于同容量 aux_only。

#### 3. 为什么当时合理

旧 audit 已确认尺度错位是实现局限，finite hindsight 有正差。

#### 4. 实际方法

先训练并冻结 query head；训练 ceif 与同初始化 aux_only；只在查询附近局部投影，unknown 不动。

#### 5. 当时期待

`ceif > aux_only > baseline`，机制使用量非零，收益集中在理论样本类型。

#### 6. 实际观察

【事实】代码和 22 项本地合成/恢复测试通过。【待验证】真实数据训练、HIP、validation、正式 test 均无结果。`AI_CONTEXT.md §15`。

#### 7. 当前风险层

`Representation / Optimization / Evaluation` 均待真实数据检验。

#### 8. 已经具体排除什么

尚未排除科学假设；本地单测只证明实现契约。

#### 9. 没有排除什么

不能提前评价有效或无效；也不能把未来 baseline 增益自动归因于投影。

#### 10. 剩余问题与 01 映射

Q5、E11；机制成立必须超过 aux_only。

#### 11. 明确重启条件

当前是已授权但尚无结果的实验资产；正式扩大前应先看 E8–E10 低成本诊断。

#### 12. 状态

`OPEN`。

## 5. 跨模块与下游机制

### D-001：detector threshold / NMS 路线

#### 1. 原始问题

融合后的错误是否主要是低分 FP 或重复框，可否只靠后处理吸收？

#### 2. 原始科学假设

【假设】调整 score threshold/NMS 能解决 evidence suppression 或协同重复框问题。

#### 3. 为什么当时合理

AP 依赖分数排序和重复框；CEIF/通信的 corrected/lost 计数与 AP 排序并不一致。

#### 4. 实际方法

当前主要配置固定 `score_threshold=.2`、`nms_thresh=.15`；`oracle_eval.py` 支持运行时阈值覆盖。

#### 5. 当时期待

后处理能降低 FP/重复框，而不改融合表示。

#### 6. 实际观察

【事实】仓库只有配置、后处理和可覆盖入口；未找到独立 threshold/NMS sweep 的协议与结果。【事实】evidence FP ideal cleanup 的 AP70 上限也仅 `0–.2624` pp，但这不等于阈值实验。

#### 7. 失败或限制发生层

`Evaluation provenance / Unknown`。

#### 8. 已经具体排除什么

没有实验可排除；只能排除“阈值/NMS 已经被正式证明能解决 CEIF”的说法。

#### 9. 没有排除什么

没有排除后处理误差分解；但禁止在正式 test 上调阈值追 AP。

#### 10. 剩余问题与 01 映射

H3.3、E4；先拆排序、定位、重复框，再决定是否值得在 validation 预注册 sweep。

#### 11. 明确重启条件

必须有同一 validation、预注册阈值网格和冻结 test；目标是诊断 detector 层，不得冒充融合机制成立。

#### 12. 状态

`PAUSED`：证据不足，不能埋也不能宣传。

### X-001：agent coalition Oracle / selector

#### 1. 原始问题

“所有邻车都融合”是否总是最优，能否逐帧选择更有用的 agent 子集？

#### 2. 原始科学假设

【假设】不同帧的最优 coalition 不同，并可由无标注部署字段预测。

#### 3. 为什么当时合理

协同来源质量和可见性不同；反事实移除 agent 能直接测边际贡献。

#### 4. 实际方法

缓存编码后枚举 coalition，按 frame AP/recall/soft quality 做 Oracle；再训练 MLP selector；另有专家天气 Oracle。

#### 5. 当时期待

Oracle 相对 all 有稳定 headroom，selector 能恢复其中一部分。

#### 6. 实际观察

【事实】工具代码存在：`oracle_eval.py`、`expert_oracle_eval.py`、`train_oracle_selector.py`、`analyze_selector_margin.py`。【待验证】未找到结果 JSON/CSV/Markdown 或最新上下文数值。

#### 7. 失败或限制发生层

`Evaluation provenance / Unknown`。

#### 8. 已经具体排除什么

排除“agent selection/coalition Oracle 从未实现”。

#### 9. 没有排除什么

不能声称 Oracle 有上限、selector 有效或失败。

#### 10. 剩余问题与 01 映射

与 Q3 的可部署 utility 识别相关，但不是当前 block selector 正式结果。

#### 11. 明确重启条件

先找回 `oracle_selection.jsonl` 和协议；若没有 Oracle headroom，不训练 selector。

#### 12. 状态

`PAUSED`。

### X-002：spatial utility

#### 1. 原始问题

agent 级选择太粗时，能否学习每个邻车每个 BEV cell 的边际效用？

#### 2. 原始科学假设

【假设】局部可部署特征能预测 counterfactual spatial utility。

#### 3. 为什么当时合理

通信是空间选择问题，Oracle cache 可提供更细标签。

#### 4. 实际方法

从 Oracle spatial cache 训练 `SpatialUtilityNet`，按 utility threshold 参与选择/分析。

#### 5. 当时期待

在固定预算下比 confidence/density proxy 更接近真实边际贡献。

#### 6. 实际观察

【事实】`train_spatial_utility.py`、`spatial_utility_net.py` 和 `oracle_eval.py --spatial_utility_threshold` 存在。【待验证】没有训练结果、阈值结果或 AP。

#### 7. 失败或限制发生层

`Unknown`；潜在 `Training objective / Set interaction` 风险与后期 learned communication 相似。

#### 8. 已经具体排除什么

排除“spatial utility head 是完全未尝试的新点子”。

#### 9. 没有排除什么

没有判断其有效性；也不能把 matching 的失败自动套给该未核验实现。

#### 10. 剩余问题与 01 映射

P4、H4.1/H4.3、Q3：单 cell 标签是否转移到集合 AP。

#### 11. 明确重启条件

必须先验证 Oracle headroom、标签稳定性、集合非加性和跨场景可学习性。

#### 12. 状态

`PAUSED`。

## 6. 假设墓地

这里埋的是**命题**，不是整个研究领域。状态统计：`FALSIFIED` 10 条，`WEAKENED` 5 条，共 **15 条**。

| Hypothesis ID | 假设 | 状态 | 反证 | 能否换条件重新成立 | 不允许 AI 再怎样表述 |
|---|---|---|---|---|---|
| HG-001 | point reliability 可以直接当 block task utility | `FALSIFIED` | 低 r 删除多数条件比随机更差；P1/Q1 | 可：先控制目标覆盖、粒度和唯一支持 | 禁止用高 point AUROC 直接证明通信/删块效用 |
| HG-002 | 低平均 r 的块普遍有害，删掉会涨 AP | `FALSIFIED` | Clean/Fog/Rain 1/5/10% 全降 | 可：限定更细单元或特定条件，但须新证据 | 禁止再次直接按块均值 low-r top-k 删除 |
| HG-003 | 高 u 块应跨天气统一删除 | `FALSIFIED` | Clean/Fog/Rain 10% 均约跌 2.4–2.6 pp | 可：模拟 Snow 局部机制仍 OPEN | 禁止把 Snow validation 正结果写成通用规律 |
| HG-004 | u 相对普通概率在当前 selector 中总有独立价值 | `WEAKENED` | matching−no_u 在 Rain 近零、Snow 略负 | 可：换任务或做条件效用验证 | 禁止写“uncertainty 一定比 reliability 更有用” |
| HG-005 | 256 KiB A0B0 已近乎无损 | `FALSIFIED` | Clean 正式 AP70 `−8.1364` pp | 可：更高预算是不同命题 | 禁止只报 99.026% 省载荷而隐去 Clean 损失 |
| HG-006 | learned communication 失败只是因为分数/消息没变 | `FALSIFIED` | matching 换块 59%–70% 仍在天气更差 | 不可用于解释后期 matching；可描述早期 residual | 禁止把早期失败原因套到后期模型 |
| HG-007 | 增大 ranking flexibility 就会超过强规则 | `FALSIFIED` | matching/concat/no_u 大量换块仍不及 A0B0 | 可：只有 proxy/标签/集合目标发生实质变化 | 禁止“换 Transformer/更大 MLP”当作机制修复 |
| HG-008 | 单块 detection-loss gain 能直接相加成多块 AP gain | `WEAKENED` | 教师在规则上下文单块补位，推理换 60%–70%；正式 AP 下降 | 可：集合级 Oracle 通过后 | 禁止不测集合交互就把 gain 称为 AP utility |
| HG-009 | reviewer 只要改变点权重/PFN 就应改善检测 | `FALSIFIED` | 点和 PFN 确实变化，但补回/AP 未改善 | 可：需正确性/任务增益对照 | 禁止用 `changed_points` 代替机制成功 |
| HG-010 | 真实 reviewer 消息的微弱内容敏感性足以支撑任务收益 | `WEAKENED` | 置换差仅约调整幅度 2.1%；contrast 对照 AP 同 protocol | 可：若真实消息稳定优于置换并改变任务事件 | 禁止把非零敏感性直接写成有效利用 |
| HG-011 | CEIF 局部冲突标记的高分框大多是 FP | `FALSIFIED` | 标记内 FP率 Clean/Fog 0%/.51%，Rain/Snow仍多数 TP | 可：更精细几何子群可另验 | 禁止“射线穿过局部=整框错误” |
| HG-012 | evidence FP suppression 是 CEIF 的主要大收益来源 | `FALSIFIED` | evidence FP ideal cleanup 最大仅 Snow `+.2624` pp | 可：补全/局部表征修正另论 | 禁止拿 generic FP 直觉替代 correct/complete 分析 |
| HG-013 | 当前 learned gain 标签已与正式 AP utility 对齐 | `WEAKENED` | loss 监督与 AP 指标不一致，正式 learned 天气全面下降 | 可：E3 证明相关性后 | 禁止把 `gain=L_rule−L_swap` 直接叫预计 AP 提升 |
| HG-014 | 通信更多会让所有简单目标计数单调变好 | `FALSIFIED` | Fog/Snow A0B0 corrected/lost 计数优于 full，但 AP 更低 | AP 总体仍可随通信改善；命题仅针对简单计数 | 禁止用 corrected/lost 单一计数代替 AP |
| HG-015 | full communication 中额外消息一定无害 | `WEAKENED` | 模拟 Snow high-u 批量删除 `+2.6520` pp | 可：正式 Snow 尚未验证，且其他天气删除伤害 | 禁止由单一 Snow 开发结果宣称 full 普遍有害 |

## 7. 失败实现墓地

这里保存“上位问题可能没死，但这个版本不要重复”。

| Implementation ID | 实现 | 上位假设 | 为什么失败 | 已尝试过哪些变体 | 下次不能原样重复什么 |
|---|---|---|---|---|---|
| FI-001 | 早期 A1B1 软选择 | task-aware selection | score 变但硬 top-k 不变 | A1B0/A0B1/A1B1 | 同 detection loss + 同软反向，只换网络层数 |
| FI-002 | `A0B0+0.1*tanh(residual)` | 规则边界微调 | 有界幅度未跨 ranking margin | residual、零初始化 | 再给 A0B0 加“小残差”却不监督 replacement |
| FI-003 | legacy point reviewer | 多车证据纠正 sensing | 改动小、内容敏感性弱、未转成检测收益 | normal/protocol/shuffled/固定支持诊断 | 用 changed_points/PFN change 当成功指标 |
| FI-004 | contrast / contrast_gain reviewer | 参考差异增强内容利用 | normal/neutral/permuted AP 均同 protocol，修正幅度极小 | legacy/contrast/contrast_gain | 只换参考消息或 gain 公式，不改变监督 |
| FI-005 | matching selector | 证据匹配预测 block utility | protocol 已丢信息；单块教师到大规模重选错位；正式天气下降 | matching | 不做 E3 就继续续训/扩网络 |
| FI-006 | concat selector | 同输入普通融合 | 比 matching 更差且远弱于 A0B0 | concat | 把拼接换成更深 CNN 当新机制 |
| FI-007 | no_u selector | 验证 u 的独立价值 | u 贡献跨天气不稳定，整体路线失败 | no_u | 简单加回 u 就声称解决 selection |
| FI-008 | CEIF no-GT 整框 rule | missing/negative evidence 区分 | 局部冲突触发整块单来源替换，大量误伤 TP | rule/random/FP oracle/hindsight | 原冲突阈值 + 原干预尺度再扫参数 |

失败实现共 **8 条**。这 8 条与总表中的 `FAILED_IMPLEMENTATION` 一一对应。

## 8. 成功经验 / Positive Lessons

### 8.1 GSPR 为什么值得保留

| 经验 | 证据性质 | 对未来研究的要求 |
|---|---|---|
| 问题定义清楚：判断回波污染，不假装预测目标 utility | 【事实】输入、标签、输出在点级对齐 | 新模块先写清“究竟预测什么” |
| 监督与问题直接对应 | 【事实】point reliability 有直接标签，三 seed 点指标稳定 | 能直接监督就不要用远端 AP proxy 绕一圈 |
| 干预位置与 corruption 接近 | 【事实】在 PFN pooling 前软加权点 | 干预尺度不能远大于证据尺度 |
| 不做不可逆硬删除 | 【事实】GSPR 软权重保留弱证据 | 质量不等于无用时，先软干预 |
| 有独立点级指标，也看 downstream AP | 【事实】AUROC/BAcc 与正式 AP 都报告 | 单一中间指标不能代替最终任务 |
| 多 seed、跨天气、正式 test | 【事实】三次训练和 OPV2V-W 正式测试 | 机制结论要跨条件，而非只挑一个天气 |
| 点监督带来 detection-only 之外的增益 | 【事实】三天气均明显 | 成功模式是“直接机制信号 + 下游联合适配” |
| 成功不自动外推到通信 | 【推断】后续删除诊断给出直接反例 | 每跨一个层级，都要重新验证 proxy 对齐 |

### 8.2 其他应复用的正经验

- 【事实】A0B0 的价值在于简单、强、可解释；任何 learned selector 都必须在同协议下超过它。
- 【事实】random filtering 是必要对照，它避免把一般稀疏化误写成 quality selection。
- 【事实】CEIF audit 先做冻结反事实再训练，成功避免把“有无机会”和“能否选择”混在一起。
- 【事实】full/original_full、protocol/learned、baseline/aux_only 等成对对照能隔离工程路径与机制。
- 【推断】最有价值的结果往往不是总 AP，而是把失败定位到 proxy、粒度、ranking、集合交互或协议。

## 9. 不要再默认相信的科研直觉

| 直觉 | 当前状态 | 为什么不能再默认成立 | 重新使用需要什么证据 |
|---|---|---|---|
| reliability 低 = task utility 低 | 已否定直接等同 | low-r 多数条件比 random 更差 | 控制目标覆盖、距离、唯一支持后的边际效用 |
| uncertainty 高 = 应删除 | 通用命题已否定 | 仅模拟 Snow 局部正，其他天气明显负 | 正式 Snow、场景稳定性、结构匹配 random |
| learned selector 一定优于规则 | 已否定 | 三个 learned 版本正式天气都弱于 A0B0 | 同协议、同字节、跨天气正式提升 |
| 网络确实改变 message = AP 应提高 | 已否定 | matching 换 59%–70% 仍更差 | replacement 与真实集合收益一致 |
| score 变了 = ranking 变了 | 已否定 | early/residual score 变但换块为 0 | 必须报告 top-k 集合变化 |
| 少通信主要伤恶劣天气 | 已否定 | 256 KiB 最大损失在 Clean | 预算曲线与 error decomposition |
| attention/confidence 高 = 来源因果重要 | 未被证明 | AttFuse softmax、corrected 计数与 AP 不一致 | 删除/替换 counterfactual |
| 一条可信射线冲突 = 整框是假目标 | 已否定 | CEIF 标记框多数是 TP | 局部几何富集度和 cell-level Oracle |
| 调 detector threshold/NMS 就能修好 evidence suppression | 无证据，不能默认 | 没有正式 sweep；FP ideal cleanup 上限也很小 | validation 预注册 sweep + 冻结 test，且与融合机制分开 |
| 单块 loss gain 可直接监督多块 AP selection | 明显削弱 | 单块补位教师配合 60%–70% 重选失败 | 集合级 Oracle、gain/AP 相关与跨场景验证 |
| 代码存在 = 实验做过 | 明确禁止 | 多条历史路线只有代码/README | 必须有结果文件或最新上下文数值 |
| `42.86% raw values` = `42.86% wire bytes` | 明确禁止 | codec、零值、头部和计费口径会改变结果 | 统一 feature payload/protocol total 实测 |

## 10. 仍然存活、不允许误埋的方向

| 存活问题 | 当前状态 | 为什么不能埋 | 最便宜下一步 | 对应 01 |
|---|---|---|---|---|
| GSPR sensing reliability 能否经条件校准预测 downstream utility | `OPEN` | 被否定的是直接等同，不是所有映射 | 既有单删日志做 E1 | Q1、P1、H1.* |
| 严格零 AP 下降下，通信瓶颈是信息不足还是表示冗余 | `OPEN` | 256 KiB 失败不等于全覆盖压缩失败 | 完成当前 E12 development | Q2、P8 |
| level0 重算是否真正省系统成本 | `OPEN` | 结果未回传，带宽—计算交换未量化 | 数值/AP/bytes/time 分项 | Q6、H8.* |
| learned selector 的失败主因 | `OPEN`，当前实现已死 | 请求、标签、集合、domain 尚未分解 | 只读教师/cache/log 做 E3 | Q3、P4 |
| 模拟 Snow high-u 批量收益的来源 | `OPEN`，分支暂停 | 局部正结果不能外推，也不能凭跨天气失败抹掉 | E6 场景/非加性分析 | Q4、P5 |
| CEIF 机会主要来自 correction 还是 completion | `OPEN` | rule 失败，hindsight 正；动作混合 | 既有 actions 日志做 E8 | Q5、P6/P7 |
| 局部尺度观测投影是否有上限 | `OPEN` | 旧整块干预尺度不匹配 | E9/E10 后再评价 CEIF-min | Q5、H6.2/H7.4 |
| 历史 agent/spatial Oracle 到底有无 headroom | `PAUSED` | 代码存在但结果缺失 | 找回结果文件，避免重跑 | 与 Q3 相关 |

明确边界：`PAUSED` 不是 `FALSIFIED`；`FAILED_IMPLEMENTATION` 也不是上位科学命题死亡。

## 11. AI 新想法查重规则

### 11.1 必查流程

#### Step 1：它解决的科学问题是什么？

必须用一句可证伪的话写出，例如“预测 block 对当前 ego 检测的边际效用”，不能只写“提升鲁棒性”。

#### Step 2：它依赖什么核心假设？

明确 proxy、干预对象和因果链。若核心仍是“低 r/high u/density = 无用”，直接查 HG-001–HG-004。

#### Step 3：这个假设是否已经出现在墓地？

- 命中 `FALSIFIED`：必须提供新适用条件或新证据，否则停止。
- 命中 `WEAKENED`：先做最低成本诊断，不训练完整模型。
- 命中 `OPEN`：查看 `01_competing_hypotheses.md` 的 falsification criterion。

#### Step 4：方法是不是旧实现换了名字？

按八个维度比较：输入信号、proxy、干预对象、空间粒度、loss、ranking/gating、fusion 位置、scientific mechanism。只换 CNN/Transformer/MLP 名字不算新路线。

#### Step 5：它比旧实现多了什么可观察预测？

如果不能设计一个能同时区分旧解释和新解释的低成本实验，标记：`HIGH RISK OF REPEATED IDEA`。

### 11.2 五组“不同名字但本质类似”的历史方法

| 相似组 | 历史名字 | 共同内核 | 真正差异 |
|---|---|---|---|
| G1 物理质量门控 | density gating、weather reliability、low-r、high-u、quality filtering | 用 sensing quality proxy 降权/删消息 | density/r/u 的定义和软/硬作用不同；不能互相冒充已有结果 |
| G2 学习块排序 | A1B1、residual、matching、concat、no_u、spatial utility | 在固定预算下学习 top-k utility/ranking | 早期没换块；后期大量换错；spatial utility 无结果 |
| G3 协同证据反馈 | legacy reviewer、contrast、contrast_gain、BEV-location、fixed BEV | 邻车摘要驱动 ego 点/BEV residual correction | 参考差分、gain、作用位置和幅度对照不同 |
| G4 反事实效用 | ΔL harm、agent Oracle、spatial utility、历史 CURE、CEIF audit hindsight | 用删除/替换反事实估计边际价值 | agent/block/cell 粒度、GT 用途和是否可部署不同 |
| G5 抑制/去噪 | snow voxel suppression、reliability feature gate、CEIF rule、detector threshold/NMS | 识别疑似坏信息后压制 | 发生在 point/pillar/BEV/box/postprocess 的不同层；旧层级失败不能自动否定其他层级 |

发现 **5 组**同名不同或异名同核的历史方法。查重时既要识别共同内核，也不能把不同实验的结果互相移植。

### 11.3 最容易被 AI 再次提出的五个旧想法

1. **“把 GSPR reliability/uncertainty/density 输入一个新 MLP/Transformer，再按分数删块。”** 高风险重复 G1 + G2；low-r/high-u 和多个 learned selector 已给出反证。
2. **“给 A0B0 score 加一个小 residual，保留规则稳定性。”** 精确重复 C-006；必须先解决 ranking margin 和硬集合监督。
3. **“用单块 counterfactual loss 当 utility 标签，训练更强 selector。”** 重复 matching/spatial utility/CURE 家族；必须先证明集合非加性和 AP 对齐。
4. **“看到射线冲突或高 uncertainty 就抑制对应框/BEV，再调 threshold/NMS。”** 重复 CEIF rule/quality filtering；局部证据到整框抑制已严重误伤 TP。
5. **“让邻车发证据摘要，修正 ego 的 GSPR 点权重或 BEV 特征。”** 重复 reviewer 家族；必须有真实消息优于置换、并有直接任务或正确性监督。

## 12. 与 01_competing_hypotheses.md 的映射

| 本墓地资产 | 01 中对应现象/假设/问题 | 关系 |
|---|---|---|
| P-002 + C-013/C-014/C-016 | P1、H1.1–H1.4、Q1、E1 | GSPR 成功，但 direct block utility 命题入墓 |
| P-001/P-002 | P2、H2.1–H2.4、E2 | 保存 detection-only 与 full_v1 的历史层级 |
| C-003/C-004/C-018/C-019/C-020 | P3/P8、H3.*、H8.*、Q2/Q6、E4/E5/E12 | A0B0 规则、旧预算和新全覆盖路线分开 |
| C-005/C-006/C-010–C-012 | P4、H4.1–H4.4、Q3、E3 | 早期“没换块”和后期“换错块”分开 |
| C-014–C-016 | P5、H5.1–H5.4、Q4、E6/E7 | Snow 局部异常保留，不外推 |
| F-003–F-006 | P6/P7、H6.*、H7.*、Q5、E8–E11 | 科学假设、audit、rule、min 四层分开 |
| X-001/X-002/F-002/C-017 | H4.1/H4.3/H7.3 相关 | 历史 counterfactual 资产存在，但因无结果不写成事实 |
| D-001 | H3.3、E4 | detector/postprocess 仍需误差分解，不与 fusion 结论混写 |

分工边界：

- `01_competing_hypotheses.md` 回答“现在还不知道什么、怎样最便宜证伪”。
- 本文件回答“已经试过什么、哪些命题和具体实现不能再重复付费”。
- 当 `01` 的新实验完成后，应先更新 `AI_CONTEXT.md` 的事实，再决定本文件中的 `OPEN/PAUSED` 是否迁移到其他状态。

## 13. 当前科研边界总结

1. 【事实】GSPR full_v1 是成功且冻结的 sensing 资产；其 point reliability 不能直接等同于 block utility。
2. 【事实】A0B0 是强规则基线；被否定的是 256 KiB “近乎无损”，不是规则本身。
3. 【事实】学习通信有两种不同失败：早期 score 变但 ranking 不变；后期 ranking 大变但 AP 更差。
4. 【事实】low-r/high-u 不能作为跨天气通用 hard filtering；Snow 的局部正现象仍只属于 development/global-sort。
5. 【事实】CEIF audit 支持“有限机会存在、当前 rule 误伤严重”；这不否定 missing evidence 与 negative evidence 应区分。
6. 【待验证】CEIF-min 没有真实 AP；不能埋，也不能宣布有效。
7. 【待验证】lossless_full/level0_recompute 正在测试；不能把 raw-value 理论减少写成 wire-byte 或系统收益。
8. 【事实】历史 Oracle、spatial utility、CURE、density gating、detector threshold/NMS 多数只有代码痕迹；没有结果就不做科学结论。
9. 【研究纪律】下一项完整训练之前，先查假设墓地和失败实现墓地；若只是旧 proxy、旧粒度、旧 loss、旧 ranking 换网络名，直接标记 `HIGH RISK OF REPEATED IDEA`。
10. 【当前优先级】通信先完成 `lossless_comm` development 的数值/AP/bytes/time 门槛；其他路线按 `01` 的信息增益/成本顺序处理，不自动启动实验。
