# 科研代码与实验上下文

更新：2026-10-09。本文保留环境、关键结果、证据路径和研究决定；操作细节查各实验目录 README。沿用历史 §1–§38 编号，重复的 §27 改为 §27A/§27B。

## 当前状态与执行约束

- **方向 B 继续：**逐目标来源选择的 Oracle（借助 GT 判断理想动作的研究诊断）三天气平均 AP70 空间为 Same−Shared +6.36 pp、Task−Shared +7.72 pp、Task−Same +1.36 pp（§33）；限制到 Shared top256 候选区域后，Task−Same 仍 +1.379 pp（§36）。这些数值不能当作可部署方法的预期收益。
- **最新实测为 S0–S3（§38）：**当前最终输出特征配合线性模型未可靠学会分类/定位来源收益，结论 `S1_WEAK / S2_WEAK`；弱动作铺到所有候选会放大误伤，保守校准全部 KEEP。
- **下一阶段为 S4 Source Evidence Audit：**代码在 `local_fusion_source_evidence_audit/`，检查更底层的几何、语义和 GSPR 证据。代码已实现，尚无服务器实验结果；入口与复现要求见该目录 README。优先级为几何证据 → 语义证据 → 候选竞争，不直接扩大正式网络。
- 通信保留全覆盖无损编码与层级重算（§16）；旧 learned block selection、B1 Gate、当前候选 distortion 排序头均已停止扩大。历史设计不构成重启授权。
- **实验只在服务器运行：本地编写 → 整目录同步 → 服务器执行。**本地仅做静态或纯 Python/NumPy 检查，不跑真实数据、Torch/OpenCOOD 科研实验。长任务给后台命令。
- 冻结 GSPR 源码及既有模型/实验，新增工作放独立目录。服务器可没有 `.git`；运行前提使用源码哈希，不能依赖 Git/分支/commit。源码和 `source_snapshot.json` 必须整套同步，不能为通过校验手改哈希。
- 正式 test 不用于调阈值、epoch、结构或选择模型。旧 OPV2V-W 已多次被研究查看，不能称为完全未接触的测试集。S4 不访问正式 test。

结果表 AP 默认 0–1；`pp` 为百分点，0.005 AP = 0.5 pp。C/F/R/S 简写对应 Clean/Fog/Rain/Snow。恢复/丢失原 TP/新增 FP 身份是匹配结果，新增 FP 身份不等于总 FP 净增加，计数不能直接换算 AP。

## 1. 服务器与设备

服务器 `cjm@admin1`；项目 `/home/cjm/OpenCOOD-main/cjmnet`，父目录 `/home/cjm/OpenCOOD-main`。8 张 HIP HCU（0–7）；原分配中 7 固定占用，使用前核对当前空闲卡。

只设 `ROCR_VISIBLE_DEVICES=<物理卡号>`，unset `HIP_VISIBLE_DEVICES`、`CUDA_VISIBLE_DEVICES`，程序内使用 `torch.cuda` / `cuda:0`。同时设置多个可见设备变量会二次过滤，可能报 `No HIP GPUs are available`。

## 2. 软件与本地环境

服务器 Conda `opencood`，解释器 `/home/cjm/miniconda3/envs/opencood/bin/python`，Python 3.10、PyTorch 2.5.1、NumPy 1.26.4、Shapely 2.0.0、OpenCOOD。本地 Windows 仓库 `D:\Collaborative perception\mymodule\cjmnet`，不具备完整科研运行环境。

## 3. 数据与冻结前端

| 用途 | 路径与规模 |
|---|---|
| 官方 Clean | `/data/scd/datasets/opv2v_official_data_dumping/{train,validate,test}`；43/9/16 场景，train 6374 帧、validate 1980 帧、test 2170 帧 |
| 固定正式天气 test | `/data/cjm/datasets/opv2v-w/{fog,rain,snow}/test` |
| 固定物理天气开发数据 | `/data/cjm/datasets/opv2v-physics-fixed-v1/{fog,rain,snow,mixed}/{train,validate}` |
| 默认冻结前端 | `/data/cjm/datasets/logs/gspr_joint_full_v1_seed20260907_20260907_161155/{config.yaml,net_best_validation.pth}` |

默认前端 checkpoint SHA256：`93598c7f6cde8c8caa54155cde98bbe75ff2921481b24928a2172e2d6c5dd4c1`。后续各实验的 F/Shared 起点还需查各自协议，不可只凭名称判断权重相同。

固定物理天气数据生成日志已报告 Complete，尚无独立全量完整性扫描。每种天气 train 19575 个 PCD、validate 10478 个；元数据链接官方 Clean，mixed PCD 链接单天气 PCD，源目录必须保留。生成器 `opencood/tools/materialize_physics_weather.py`，说明 `opencood/tools/PHYSICS_WEATHER_DATASET.md`，中断可 `RESUME=1`，已有 complete 数据无需重建。它是固定 epoch-0 realization；旧在线天气逐轮重采样，PCD 强度 RGB 量化、加载时点顺序重排，两种协议不能混写。

## 4. 存储与操作

大数据、权重、缓存和大型日志放 `/data/cjm/datasets/logs` 等外部数据盘，不写代码仓库。启动前指定物理卡与 opencood 解释器；长期任务后台运行并记录 RUN/PID/LOG。单个进程失败后先确认已有缓存与源码契约，再断点恢复，避免覆盖已完成结果。

## 5. 原始 AttFuse 基线（2026-09-03）

代码 `attfuse_code/{point_pillar_intermediate.py,att_bev_backbone.py,self_attn.py}`，配置 `attfuse_config.yaml`，入口 `train_attfuse.py`。PointPillar 多尺度逐 BEV cell 来源 attention；Adam lr=.002、15 epochs、batch=4、max CAV=5、compression=0、MultiStep[10,15] gamma=.1。

运行 `/data/cjm/datasets/logs/attfuse_opv2v_20260903_200923`，按最低 Clean validation loss 冻结 `net_epoch13.pth`。正式结果位于 `opv2v_w_best_epoch13/{weather}/eval.yaml` 与 `opv2v_clean_best_epoch13/eval.yaml`。

| 条件 | AP30 | AP50 | AP70 |
|---|---:|---:|---:|
| Clean | 0.91 | 0.90 | 0.82 |
| Fog | 0.701255859 | 0.688508387 | 0.610343363 |
| Rain | 0.681250285 | 0.658894030 | 0.561978678 |
| Snow | 0.580290729 | 0.557711838 | 0.468391929 |

Clean 只有日志舍入值，不能伪造精确差值。`too many cavs` 表示按 max CAV 截断，单独出现不代表进程失败。

## 6. GSPR 方法与接口

GSPR 为点提供连续可靠性和不确定性，再在 PFN 前做软加权。几何分支先建模，再结合反射/物理证据、传感器统计 FiLM 和 pillar XY Haar 高低频上下文；两类 Dirichlet evidence 产生可靠性/不确定性。保留点槽位，无硬删除、无时间分支。

这是 TripleMixer 思路与监督接口的项目适配，不能称其完整方法复现。冻结输出：`point_reliability`、`point_uncertainty`、`point_evidence`、`pillar_reliability`、`pillar_uncertainty`。旧分阶段设计已被 full_v1 完成结果覆盖。

## 7. GSPR 预训练、联合训练与三次结果（已完成）

早期逐点数据：`/data/cjm/datasets/gspr_opv2v_weather_{smoke,development}`；development 24/12 源 PCD、72/36 NPZ，仅 moderate，Val Rain 只有 27 个噪声点，近满分不能作为可靠结论。点预训练 `/data/cjm/datasets/logs/gspr_point_pretrain_20260905_120258/gspr_best.pth`：AUROC .994094、AUPRC .999798，平衡阈值 .588589；固定 .5 阈值 noise recall=0 属校准偏移，主路径使用软权重。

联合开发 `/data/cjm/datasets/logs/gspr_point_joint_20260905_193746/net_epoch2.pth`，Clean val loss .259608；`frozen_epoch2_all_weather` 三天气平均 AP70 .584316，较 GSPR Detection-only +2.140 pp。

完整逐点语料 `/data/cjm/datasets/gspr_opv2v_weather_full_v1`：train 516 源 PCD/4644 NPZ，val 108/972，3 天气×3 强度；已校验无重复。Rain light/moderate 噪声稀少，Fog heavy 噪声约34%。采样九组等概率，每组零噪样本占比最多20%（自然更低则保持），validation 保持自然分布；预训练按九组 macro AUROC 选权重，不人为改物理参数。

seed20260906 点预训练5轮，epoch5：val loss .059142、macro AUROC .999871。联合训练 epoch1 冻结检测端、epoch2–5 解冻，按最低 Clean val detection loss 且 point macro AUROC≥.995 选权重。三次最佳 epoch 4/3/4；val loss .272177±.000409、point macro AUROC .999833±.000043。

三次正式检测均值±样本标准差：

| 条件 | AP@0.3（均值 ± 样本标准差） | AP@0.5（均值 ± 样本标准差） | AP@0.7（均值 ± 样本标准差） |
|---|---:|---:|---:|
| Clean | 0.920849 ± 0.002051 | 0.912204 ± 0.002133 | 0.831512 ± 0.001838 |
| Fog | 0.718529 ± 0.001066 | 0.708345 ± 0.000560 | 0.639272 ± 0.001119 |
| Rain | 0.760051 ± 0.000276 | 0.743675 ± 0.000511 | 0.645840 ± 0.000853 |
| Snow | 0.642076 ± 0.001294 | 0.622683 ± 0.002086 | 0.514817 ± 0.004604 |

相对原始 AttFuse，AP70 的 Fog/Rain/Snow 增量约 +2.893/+8.386/+4.643 pp，三天气平均 +5.307 pp；相对 **GSPR Detection-only epoch11** 为 +2.449/+4.800/+3.868 pp，平均 +3.706 pp，Clean −.353 pp。两种参照不能混用。

seed20260906 pilot 的随机种子设置与后两次不同；严格三种子论文统计需披露差异或按统一代码重跑。默认 seed20260907 权重见 §3；seed20260908 为 `/data/cjm/datasets/logs/gspr_joint_full_v1_seed20260908_20260907_214827/net_best_validation.pth`。逐种子完整表与原始报告优先作为精确数值依据。

## 8. 冻结边界与通信阶段状态

禁止修改 `attfuse_gspr/reliability.py`、`pillar_vfe.py`、`point_pillar_gspr_attfuse.py`、`gspr_supervision/*`、`pretrain_gspr.py`、`train_gspr_multitask.py`。既有14文件审计用于检查冻结源；新功能使用独立模块、入口和配置。旧联合优化计划不构成解冻授权。当前通信路线见 §16，旧 review/selection 仅保留为历史对照。

## 9. 通信 review 原型（历史）

`gspr_review/`：1473 参数 reviewer，发送 height-cell 的可靠性/噪声/不确定性/support；总256 KiB，review≤12.5%，含实际序列化。3轮小样本 Clean/Weather AP 约 .93/.92/.81 与 .84/.82/.71。

固定 mask 内容扰动只产生约原更新2.1%的影响，PFN特征相对 L1 变化约 .5%，说明内容依赖较弱。`contrast`/`contrast_gain` 为1472/1729参数；cfvqa/KalmanNet/InterSHAP仅概念适配，见 SOURCE_ADAPTATION，不称完整复现。最终判断以 §11–12 为准。

## 10. Evidence block matching（历史）

`gspr_evidence/`：risk 11940参数、gain 11521参数；训练标签用等字节真实 block replacement 的检测损失变化，保留负收益；GT仅监督。17字段请求、256 blocks、总256 KiB包含metadata/feature。teacher完整扫描train/val，每peer/frame最多8候选block，不能误写成只用8帧。matching/concat/no_u对照；早期 global-sort 开发结果不能冒充正式测试。

## 11. 统一正式协议与 learned communication 结果（已完成）

正式比较固定 OPV2V clean test + OPV2V-W Fog/Rain/Snow test，关闭在线天气/数据增强，读取 `processed_lidar`，全场景全帧 stride=1。AP30/50/70 使用 **BEV 平面多边形 IoU** 与 `global_sort_detections=False`（按帧累计）；它与跨帧统一分数排序 global-sort AP、height-aware 3D IoU 不同。validation+在线天气只作开发，不能与 test AP直接相减。

历史正式入口 `gspr_evidence/run_benchmark.sh`，训练run `/data/cjm/datasets/logs/gspr_evidence_seed20260913_20260913_094932`；未提供精确测试输出时间/checkpoint哈希，不补造。72行结果：3 variants×4天气×6 controls，各2170帧，full=original_full到6位。

| 条件 | full AP30 | full AP50 | full AP70 | A0B0 AP70 | none AP70 | matching protocol AP70 | matching learned AP70 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Clean | .921226 | .912739 | .832485 | .751121 | .577708 | .751891 | .753614 |
| Fog | .719748 | .709023 | .639663 | .626587 | .401465 | .584758 | .529966 |
| Rain | .759965 | .743751 | .647507 | .626826 | .406726 | .592840 | .562264 |
| Snow | .644169 | .624993 | .518248 | .504870 | .283999 | .463232 | .423336 |

A0B0 244003.7 vs full 25052662.3 bytes/frame（约少99.026%），但 AP70损失 C/F/R/S为8.1364/1.3076/2.0681/1.3378 pp，不能称全条件近无损。matching protocol/learned均244913.4 bytes/frame。learned−A0B0 为 +.2493/−9.6621/−6.4562/−8.1534 pp。

matching−concat learned AP70 为 +.5665/+1.1827/+.5848/+.8680 pp；matching−no_u 为 +.9400/+.4398/+.0869/−.0646 pp，u非全条件有效。matching实际替换59–70% blocks，失败不能解释成没有执行。单block teacher与大规模联合替换存在状态差异，尚不能确定唯一根因。

## 12. 过滤/收益诊断与停止 learned selection（已完成）

`/data/cjm/datasets/logs/gspr_harm_validation_both_20260914_144741/{clean,fog,rain,snow}/diagnosis.md`，完整validation+在线天气、global-sort。高u删除10%相对full的 AP70 Δ C/F/R/S = −2.4138/−2.5761/−2.4561/+2.6520 pp，随机10% = −1.1304/−1.2435/−1.1272/−.8218 pp。Snow高u10% .693438→.719959，recover1732/lost277/newFP601；未形成跨天气收益。

低r过滤也未稳定超过随机。单动作采样候选 C/F/R/S 24525/23917/24570/25430，有益13/24/15/25，有害136/146/107/137；非穷举，不排除批量交互。用户已停止当前 learned通信/过滤，不能自动重启。曾保留A0B0，后由 §16 全覆盖压缩优先取代。

## 13. A0B0 budget scan（历史设计，无完成结果）

`gspr_evidence/{budget_scan.py,budget_report.py,run_budget_scan.sh,BUDGET_SCAN.md}`：预算256KiB、1/2/4/8/16/32MiB/full；只在development选择四天气12项未舍入AP均不低于full的最小预算，失败回退full，禁止test选预算。full/original_full一致性及非global-sort检查保留；已非当前前置门禁。

## 14. CEIF 可行性审计（已完成）

`ceif_audit/`，run `/data/cjm/datasets/logs/ceif_opv2vw_audit_20260915_165200`，正式每天气2170帧，full复现 §11。几何端点/ray traversal只能提供局部证据，不能把整cell称为free；GT hindsight仅研究诊断。

| 干预 ΔAP70/pp | Clean | Fog | Rain | Snow |
|---|---:|---:|---:|---:|
| 理想evidence FP清理 | .0000 | .0004 | .0486 | .2624 |
| 无GT规则 | −.8215 | −.5474 | −1.3489 | −1.3691 |
| evidence候选+GT选择 | .1558 | .6094 | .8369 | 1.0375 |
| 全finite候选+GT选择 | .3616 | .7931 | 1.2526 | 2.0826 |

有效动作3022/5799/5402/6211，有益39/134/204/297、有害575/762/1239/1383。evidence与整ROI替换尺度不一致是已知限制，未证明唯一根因，未宣告CEIF成功。

## 15. 最小可训练 CEIF（已完成，未通过）

`ceif_min/`：冻结GSPR/backbone/head，先用Clean端点/ray伪标签预训练occupancy head 1轮；ceif/aux_only同残差结构训练3轮，lr2e−4、batch1、约1024queries，只有ceif执行投影。GT框不作occupancy标签；online天气训练、validation选checkpoint、固定正式test。

结果 `/data/cjm/datasets/logs/ceif_min_train_20260916_101742_opv2vw_test_20260917_155212/results.md`：

| AP70 | Clean | Fog | Rain | Snow |
|---|---:|---:|---:|---:|
| baseline | .832568 | .639395 | .646653 | .518487 |
| ceif | .821628 | .631742 | .635622 | .508052 |
| aux_only | .823145 | .631884 | .637272 | .508243 |

全部12项指标 ceif<aux_only<baseline，预定+5pp失败；不依据test继续调参。几何额外通信单独统计。

## 16. 全覆盖无损通信＋层级重算（正式完成，保留）

`lossless_comm/`，结果 `/data/cjm/datasets/logs/lossless_comm_benchmark_20260916_182339/results.md`。对照raw_full/lossless_full/level0_recompute/original_full；四天气12项AP均同 §11 full到6位。

float32逐位编码采用byte shuffle或+0 bitmap/nonzero，zstandard可用则用、否则zlib，保留−0。接收peer level0后用冻结block1/2重算：每粗cell原1024+512+256=1792个值，去掉768=42.86%原始值冗余；此比例与wire压缩率不同。

| MiB/frame | Clean | Fog | Rain | Snow |
|---|---:|---:|---:|---:|
| raw_full | 23.8921 | 23.8921 | 23.8921 | 23.8921 |
| lossless_full | 3.5508 | 1.5016 | 1.8139 | 1.2960 |
| level0_recompute | 2.8540 | 1.1632 | 1.4260 | .9901 |
| level0减少比例 | 88.05% | 95.13% | 94.03% | 95.86% |

level0逐位相同；level1/2因sender batch/receiver逐peer运算可有浮点漂移，需测量，不能放宽容差掩盖。CPU encode lossless221–456ms、level0159–360ms，decode55–71/42–56ms，重算7–8ms；暂不以codec时间否决。raw含request而新路径主要计feature，论文wire统计需统一；计时未完整计D2H/H2D，不能称端到端延迟。AP优先，不自动增加learned codec。

## 17. H-A5 自车证据审计（全量完成）

结果 `/data/cjm/datasets/logs/qa_local_evidence_full_20260918_112018/{a5_report.md,a5_failure_stage_report.md}`。官方validation+在线天气，ego-only冻结，无AttFuse/训练/test。对象为Clean ego检出而weather ego漏检；q25 N_eff/coverage只是strong sensing proxy分层。

| Weather | All ego misses | q25 strong | Late detection-path loss | Regression/localization support | Upstream unresolved | A5b support |
|---|---:|---:|---:|---:|---:|---:|
| Fog | 3377 | 1211 | 1106 (91.33%) | 94 (7.76%) | 11 (0.91%) | 1200 / 1211 = 99.09% |
| Rain | 1318 | 478 | 461 (96.44%) | 16 (3.35%) | 1 (0.21%) | 477 / 478 = 99.79% |
| Snow | 5961 | 3336 | 2843 (85.22%) | 67 (2.01%) | 426 (12.77%) | 2910 / 3336 = 87.23% |

| Weather | q25 strong | 没生成 IoU≥0.7 正确框 | 正确框分数过低被删 | 正确框在 NMS 竞争中被压掉 |
|---|---:|---:|---:|---:|
| Fog | 1211 | 105 (8.67%) | 428 (35.34%) | 678 (55.99%) |
| Rain | 478 | 17 (3.56%) | 180 (37.66%) | 281 (58.79%) |
| Snow | 3336 | 493 (14.78%) | 2281 (68.38%) | 562 (16.85%) |

strong集合总4587/5025=91.28%有A5b下游证据；q20–40范围Fog99.00–99.23%、Rain99.63–99.82%、Snow84.53–88.02%。同anchor score变化中位数 F/R/S −.080928/−.036148/−.593123，IoU变化 −.056595/−.026554/−.142969。

A5b为主导，Snow仍有12.77% upstream unresolved，不能称A5a完全否定或proxy一定失真。最后失败环节不等于最早根因。A6“peer-alone检出而fusion漏检”属于另一分母。

## 18. Stage-3B 机制干预（已完成）

`/data/cjm/datasets/logs/qa_stage3b_diagnostic_20260918_110054/mechanism_report.md`。选定失败目标 F162/150帧、R63/61帧、S988/694帧，非全量AP。

| 可恢复机会 | Fog | Rain | Snow |
|---|---:|---:|---:|
| 保ego来源子集 | 96 | 43 | 358 |
| peer分数+full几何 | 58 | 21 | 617 |
| full分数+peer几何 | 120 | 53 | 244 |
| peer query全尺度 | 58 | 18 | 581 |
| uniform | 54 | 15 | 417 |

机会不能相加。Fog/Rain目标附近低IoU高分抑制框114/56次，支持score–quality错位；不证明物理对象身份。Snow798次score过滤，最好好框中位score .0893，低于.2阈值；query机会581 vs subset358，至少223为query独有。

整帧query-all recover/lost/newFP F57/114/144、R18/14/39、S574/1586/703；hindsight无误伤可恢复17/7/67。NMS放宽至.5只多恢复2/2/5，却新增FP84/43/454；Snow score .1/.05恢复280/390、新FP4502/18336。支持局部选择，不能全局替换或直接放阈值。自车也存在类似失败，attention唯一根因尚未成立。

## 19. 局部 utility v1（历史，已由v2覆盖）

`local_fusion_utility/`，run `/data/cjm/datasets/logs/local_fusion_utility_seed20260913_20260920_203329`。冻结前端，s0/1局部tile peer-query重算全部来源权重、s2保持；utility监督recover/lost/newFP，loss_gain对照。完整train/val、8采样动作/branch/frame。

开发AP70 baseline C/F/R/S .7724/.7331/.7630/.5871，utility .7709/.7328/.7618/.6085，loss_gain .7719/.7326/.7622/.6115。utility约23.48 tiles/frame、loss_gain405.3，动作密度不等，不能纯归因为监督。旧正式启动路径错误不代表模型损坏；不要重启当时待完成计划，v2已完成正式测试。

## 20. 局部 utility v2（正式完成，收益有限）

`/data/cjm/datasets/logs/local_fusion_utility_v2_20260921_152505`；本地 `logs/v2/` 的all_results/calibration/history/pipeline_status。全train/val seed20260913、8轮；6374训练帧与5898更新不能混写，dev1980/test2170帧每条件。tile高度修为96:100，逐branch active/low/background采样、穷举peer，FP32。

校准整帧联合输出：Clean下降≤.1pp、每条件lost/newFP各≤baseline TP的1%，可行中最大化四条件平均AP70，KEEP显式。utility阈值.02、loss_gain.05、confidence选择KEEP。单动作标签与联合执行仍有差异。

正式完整AP：

| 条件 | 方法 | AP30 | AP50 | AP70 |
|---|---|---:|---:|---:|
| Clean | baseline | 0.921226 | 0.912739 | 0.832485 |
| Clean | utility | 0.920135 | 0.911584 | 0.830790 |
| Clean | loss_gain | 0.919761 | 0.911197 | 0.830971 |
| Fog | baseline | 0.719748 | 0.709023 | 0.639663 |
| Fog | utility | 0.719626 | 0.708735 | 0.639304 |
| Fog | loss_gain | 0.719380 | 0.708667 | 0.639368 |
| Rain | baseline | 0.759965 | 0.743751 | 0.647507 |
| Rain | utility | 0.757841 | 0.742068 | 0.648675 |
| Rain | loss_gain | 0.759022 | 0.742851 | 0.647239 |
| Snow | baseline | 0.644169 | 0.624993 | 0.518248 |
| Snow | utility | 0.643063 | 0.623590 | 0.525104 |
| Snow | loss_gain | 0.643342 | 0.624173 | 0.519211 |

utility ΔAP70 C/F/R/S −.1695/−.0359/+.1168/+.6856 pp，四条件平均+.1493pp、恶劣平均+.2555pp；全部AP30/50下降。累计recover/lost/newFP utility460/167/551，loss_gain126/8/390。开发Snow utility1626/295/287，noKEEP2224/1731/1723，说明KEEP减少误伤但不保证正式安全。

全流程24.86h：train cache2.95、val cache1.62、训练两模型.17、calibration14.04、development4.89、benchmark1.19h。raw feature25,035,148.39 bytes/frame，不含协议；稀疏动作不代表省通信。action head2.5–3.6ms不含完整推理。结论限于小幅Snow收益，不支持通用主创新，禁止test后调阈值。

## 21. v3 结构与协议（结果见§22）

`local_fusion_v3/`：baseline为冻结GSPR+AttFuse；attention/residual使用同样s0/1局部空间编码（1×1降维32、3×3卷积），来源自身/均值/full表示与原权重预测source softmax，s2不改。

attention直接替换；residual混合 `w_new=w_original+g*(w_proposed-w_original)`，`g=.5*sigmoid`，零权重/bias−4初始化约.009。全BEV、无GT筛选，cls/reg共用融合特征；单车保持原路径。旁路不保证无退化。

两模型同完整train、Clean+在线mixed天气、batch1、8轮、AdamW lr=.0003 wd=.0001、梯度裁剪5；只训练融合，冻结前端/检测端eval但允许梯度穿过。loss=PointPillar检测损失+.01权重变化惩罚，按配对validation loss选best。

validation决定整模型启用/KEEP：Clean AP70下降≤.1pp、每条件AP50下降≤.1pp、lost/newFP各≤baseline TP的1%、四条件mean AP70增益>.01pp。原模型与deployed结果均报告。dev与calibration同validation，不独立。`run_all.sh`已完成训练→calibration→dev→正式test，旧待运行状态作废。

## 22. v3 实测（正式完成，部署回退KEEP）

run `/data/cjm/datasets/logs/local_fusion_v3_20260922_182832`；本地 `logs/v3/{all_results.json,calibration.json,history.json,pipeline_status.json,resolved_experiment.yaml}`。源码哈希一致，所有阶段0，dev1980/test2170帧每条件。

正式AP：

| 条件 | 方法 | AP30 | AP50 | AP70 |
|---|---|---:|---:|---:|
| clean | baseline | 0.921226 | 0.912739 | 0.832485 |
| clean | attention | 0.896608 | 0.886584 | 0.806429 |
| clean | residual | 0.907509 | 0.898058 | 0.817383 |
| fog | baseline | 0.719748 | 0.709023 | 0.639663 |
| fog | attention | 0.719399 | 0.706859 | 0.638826 |
| fog | residual | 0.721623 | 0.709584 | 0.640845 |
| rain | baseline | 0.759965 | 0.743751 | 0.647507 |
| rain | attention | 0.758039 | 0.742279 | 0.646272 |
| rain | residual | 0.761572 | 0.745123 | 0.652774 |
| snow | baseline | 0.644169 | 0.624993 | 0.518248 |
| snow | attention | 0.651834 | 0.631745 | 0.524225 |
| snow | residual | 0.652212 | 0.631949 | 0.533444 |

residual较baseline恶劣平均AP70 +.7215pp、四条件+.1636pp，Clean−1.5102pp；正式四条件均高于attention。开发residual ΔAP70 C/F/R/S −2.2932/−1.2979/−2.4724/+3.4621pp，平均−.6503pp；仅Snow升。正式累计attention recover/lost/newFP1846/730/5123，residual1802/545/3662，旁路有价值但不充分安全。

两模型calibration均enabled=false/feasible=false：residual开发每条件newFP/TP6.89/6.16/7.29/11.34%均超1%，Clean/Fog/Rain AP50下降也超限。deployed全部等于baseline；KEEP只说明整模型开关拒绝，不能称学会逐目标安全决策。

仅回传一份无variant的history，按gate推测为residual，不能比较两组曲线。每轮6374帧/5898更新；epoch8 val loss .310424最低（未读checkpoint epoch元数据），gate均值.2384，代表混合系数而非修改区域比例。loss下降不证明AP最好。

总66.71h：attention28.78、residual28.62、cal3.90、dev4.08、test1.33h。residual80–106ms/frame含编码/上下文/模块/后处理，排除加载/H2D/指标；不可与v2单head计时直接比较。背景误检、保护目标与checkpoint目标错位均属待证假设，不能据此test后放宽KEEP。

## 23. 方案A局部决策（确认复验完成，降级）

代码 `local_fusion_decision_pilot/`、`local_fusion_decision_pilot_v2/`、`local_fusion_decision_confirm/`。冻结v3和前端，训练三类轻量选择器；每帧≤6 tiles，residual及最多两个peer-query，GT仅单动作标签/评价。联合执行选中动作，学习KEEP top4可不动。

第一轮2772标签98.92%无变化；第二轮6876标签有益242/有害183，Snow72帧学习+4.7108pp但简单confidence+4.7450pp。confidence定义使同tile总选residual，是较弱对照；相同动作数仍不等动作类型/计算量。

确认结果 `logs/v3/confirmation_results.json`，源run `/data/cjm/datasets/logs/decision_pilot_v2_20260926_113127`。原72帧覆盖全部9场景，只能同场景抽207新帧，排除旧帧±2索引；不能称跨场景独立验证。

| 条件 | 原融合 | 学习 KEEP top-4 | 等动作数置信度 | 学习相对原融合/百分点 | 学习相对置信度/百分点 |
|---|---:|---:|---:|---:|---:|
| Clean | 0.807455 | 0.805994 | 0.805416 | -0.1461 | +0.0578 |
| Fog | 0.763782 | 0.767535 | 0.764480 | +0.3752 | +0.3054 |
| Rain | 0.796900 | 0.795299 | 0.793721 | -0.1602 | +0.1578 |
| Snow | 0.543492 | 0.588914 | 0.586289 | +4.5422 | +0.2624 |

Snow学习比配平confidence仅+.2624pp，未达预定+.5pp；Clean/Rain退化≤.2pp虽通过，整体判定仍失败。Snow学习recover192/lost5/newFP39，非安全更新。冻结复验支持局部动作有效，未支持选择器必要性或外部泛化。停止扩大当前方案A，保留动作与简单排序为对照。

## 24. 正确框保持约束（小型完成，停止扩大）

`local_fusion_preservation_pilot/`，成功 `/data/cjm/datasets/logs/preservation_pilot_20260926_111704/decision_results.json`。数值来自用户JSON，未本地核验完整run。两组同v3起点，continued用原loss，protected另加.1教师保持；只在训练GT确认的原TP anchor约束logit/7维回归。冻结GSPR/原融合/检测头，96 train帧×2轮，validation48帧/条件，固定末轮，未访问test。

protected−continued AP70 C/F/R/S +.0007/+.0212/+.0002/+.0002 pp；Clean原TP损失/新FP完全不变，Fog少1新FP但AP30/50各约−.117pp。两组共有Snow+8.3pp不能算保护项效果。保护覆盖约91%，近零增益不能归因于完全没找到保护位置；标量loss比例不能推导梯度强弱。

当前版本停止扩大。anchor保持未直接约束其他位置背景/竞争/NMS，只能否定当前方案，不否定所有正确结果保护方法。无同48帧起始checkpoint AP，不能单独估计继续训练收益。

## 25. F vs F+D 与候选质量 Oracle（完成，F+D停止扩大）

`local_fusion_detector_adaptation/`，run `/data/cjm/datasets/logs/fusion_detector_adaptation_20260927_103227`。两组同v3 residual `/data/cjm/datasets/logs/local_fusion_v3_20260922_182832/residual/best.pth`；F只训练融合，F+D额外训练deblocks/cls/reg，编码冻结、归一化统计固定，320 train帧×2轮、90 validation帧/条件、在线天气，无test。

| 条件 | F AP70 | F+D AP70 | F+D-F / pp |
|---|---:|---:|---:|
| Clean | 0.822254 | 0.801742 | -2.0513 |
| Fog | 0.785814 | 0.772347 | -1.3467 |
| Rain | 0.812677 | 0.797414 | -1.5263 |
| Snow | 0.601223 | 0.592535 | -0.8688 |

F+D三天气平均−1.2473pp，expand=false；loss更低未转化为AP。397个新增FP中classification_only356（89.67%）、regression_only32、joint/NMS9。

`oracle_quality.py` / `run_quality_oracle.sh`，本地 `logs/候选A/quality_oracle.json`。强制原AP复现≤1e−6。固定decoded几何：quality_scorepass仅原scorepass池，真实GT最大IoU作score并删除IoU0背景；quality_all绕过原score阈值；ceiling70_all只留GT IoU≥.7。均不可部署，非所有方法数学上界。

| 条件 | baseline 原始→quality_scorepass / pp | F 原始→quality_scorepass / pp | F+D 原始→quality_scorepass / pp |
|---|---:|---:|---:|
| Clean | 0.821706→0.905053，**+8.3347** | 0.822254→0.909009，**+8.6755** | 0.801742→0.909051，**+10.7309** |
| Fog | 0.782841→0.871544，**+8.8704** | 0.785814→0.881441，**+9.5627** | 0.772347→0.884021，**+11.1673** |
| Rain | 0.823700→0.896799，**+7.3100** | 0.812677→0.901115，**+8.8438** | 0.797414→0.900800，**+10.3386** |
| Snow | 0.535813→0.628436，**+9.2623** | 0.601223→0.707071，**+10.5847** | 0.592535→0.720512，**+12.7977** |

完美quality_scorepass后FD−F C/F/R/S +.0042/+.2579/−.0315/+1.3441pp；原差距大幅消失。FD scorepass的IoU≥.7覆盖比F多4/6/3/40个GT，但all-decoded覆盖1730→1730、1703→1696、1731→1726、1683→1680，说明主要让已有好框跨阈值，同时抬高错误候选。

F scorepass→all decoded覆盖 C1633→1730、F1586→1703、R1625→1731、S1337→1683（每条件1753GT）。直接quality_all AP70反降为 .6946/.6679/.6993/.5756，每帧约6–7万候选，低非零IoU框仍极多。ceiling70_all F=.985739/.970337/.986880/.958928，仅显示几何潜力。

主要瓶颈为分数无法稳定表达候选质量；不能推出所有检测适应无效、分类/定位必须不同来源，或推理特征必然可学到GT质量。先找到现实规模候选池，后续见 §32–37。

## 26. B0 Shared/Split（完成，全局方案未通过）

`local_fusion_task_split_pilot/`，run `/data/cjm/datasets/logs/task_split_pilot_20260927_120637`。复用F+D相同320 train/90 val索引、2轮。两组同初始化/参数量，两Router均前向、双解码；Shared平均两套来源权重共用于cls/reg，Split各用一套，检测端冻结。change penalty两Router取平均；Collapse将训练后权重重新平均，Swap交换，仅诊断。

预定第二seed门槛：Split−Shared三天气mean AP70≥.5pp、≥2天气正；Clean不低于Shared及v3_start超过.1pp；四条件AP50下降≤.1pp，lost/newFP各≤Shared TP的1%。

实测Split−Shared F/R/S −.1335/−.1096/+.2468pp，mean约+.0012pp，1/3正，expand=false。Split−Collapse +.1467/+.0793/+.3620pp，Swap−Split −.2051/−.1471/−.4521pp；全局top-source task disagreement C/F/R/S约3.85/3.32/3.52/4.10%。全局方法弱，但有局部差异，触发task-gap审计。

## 27A. B0 task-gap 审计定义（完成结果见§28）

`local_fusion_task_split_pilot/{audit_task_gap.py,run_task_gap_audit.sh}`，原90帧/条件，不训练。强制Shared/Split/Collapse AP复现≤1e−6。主比较Split vs Collapse，GT分split_only/reference_only/both/neither；精确oriented footprint与1.5×context统计mean_abs、`TV=.5*sum_source|w_cls−w_reg|`、top-source disagreement，报告AUROC及高四分位富集。GT仅后验定位，关联不构成因果或可部署触发阈值。

## 27B. B1 Local/Global Gate 协议（结果见§29）

`local_fusion_task_split_v2/`，冻结B0 Split端点，`w_task=w_common+g*(w_task_raw−w_common)`，g=0为Collapse、g=1为Split。输入TV、top disagreement、Collapse分类activity；不读GT/天气。两组同466参数，3×3 conv8→SiLU→1×1→sigmoid，bias−4；Local读空间图，Global先均值广播，replicate padding保证有效参数一致。

同B0索引，seed20260927、4轮、lr=.001 wd=.0001、loss=检测+.01 mean(g)，固定末轮。冻结所有Router/前端/检测端，单来源无梯度分支跳更新。Local−Global门槛：三天气mean≥.5pp、≥2天气正且mean高于B0 Split；Clean/AP50下降≤.1pp，lost/newFP各≤TP1%。仅expand=true才允许正式test。实际失败，未访问test，不重启旧运行计划。

## 28. B0 task-gap 审计结果（已完成）

审计 §26 run。三天气Split vs Collapse目标：split_only11（11帧）、reference_only1、both4359、neither888。split_only/both/neither box TV .02947/.02153/.01607；top disagreement .08047/.03064/.03240。split_only 81.82%框内有top-source冲突，both36.52%。

split_only vs both/all-other AUROC：TV .676/.693，mean_abs .737/.746，top disagreement .741/.743；高task-gap四分位富集2.15–2.18倍。Snow6 vs0、mean_abs AUC .855，Rain仅2例。支持局部冲突动机但样本很少，不能据此部署。

## 29. B1 Gate 结果（已完成，学习方案失败）

run `/data/cjm/datasets/logs/task_split_v2_20260927_163353`。Local/Global均4轮×305更新，四条件AP与Collapse完全一致到保存精度，AP70 C/F/R/S .821899/.783012/.810788/.600072；Split .822057/.784479/.811581/.603692。Local−Split三天气mean−.1960pp，expand=false，无OPV2V-W。

Gate塌缩：Global mean .01131→.00042，Local .01161→.00056；validation约.00054–.00055。scale1 corr(g,TV) F/R/S −.503/−.493/−.509，冲突cell反而更关；scale0主要跟activity相关.71–.79。当前检测loss+.01稀疏项未学会局部打开，不能据此否定所有局部task split上限。

## 30. B1 固定 sweep / hindsight Oracle 定义（结果见§31）

`local_fusion_task_split_v2/{oracle_sweep.py,run_oracle_sweep.sh}`，不训练、不test。常数g=0/.05/.10/.25/.50/.75/1，端点AP≤1e−6复现。Oracle仅对Split检出而Collapse漏检GT的box/1.5×context置g=1，其余0。使用GT+hindsight，只诊断端点局部选择上限，不能作为部署阈值或直接训练标签。输出 `$B1_RUN/oracle_sweep/oracle_sweep_results.json`。

## 31. B1 sweep结果与更强来源 Oracle 设计

本地 `logs/候选B/oracle_sweep_results.json`。三天气mean AP70按g=0/.05/.10/.25/.5/.75/1为 .731290/.731501/.731501/.732259/.733204/.733339/.733250；事后最好g=.75仅比Split+.0089pp。

Oracle-Box/Context相同，较Split F/R/S +.0900/+.1546/+.0573pp，mean+.1006pp；较Collapse+.2966pp。保留3/2/6独有恢复、相对Collapse零lost/newFP，只开约1e−5–6e−5 cells。端点额外上限小，停止B1 penalty/bias/epoch/hidden调参。

转到 `local_fusion_task_source_oracle/`：KEEP_SHARED、single:i、query:i，GT目标1×/1.5×ROI；Same要求cls/reg同来源，Task可分开。预注册方向B继续条件：三天气Task−Shared mean≥1.5pp；≥2天气各≥1pp；Task−Same mean≥.5pp，结果后不得改。

首轮 `/data/cjm/datasets/logs/task_source_oracle_20260927_195742` 在Fog80/90后极慢，最多约240次完整postprocess/GT/GT目标，密集帧数千CPU/Shapely NMS，不能据未完成运行下科学结论。修正版所有组合先GT-aware anchor proxy评分，仅shortlist做完整NMS，k=6，保留overall/task-separated/same三组；非完整post-NMS穷举上界。最终结果见§33。

## 32. top256候选池与无训练重评分（已完成）

`local_fusion_detector_adaptation/{candidate_audit.py,candidate_rescore.py}`及对应run脚本；复用§25 F/FD的90 validation帧，无训练/test，先复现原AP≤1e−6。top256按融合score取几何有效框，不改变geometry；agreement=`max_source(score_source*IoU_source,fused)`，GT仅评价。

| 范围内IoU≥.7覆盖GT（每条件1753） | Clean | Fog | Rain | Snow |
|---|---:|---:|---:|---:|
| F score>.2→top256 | 1632→1690 | 1586→1640 | 1625→1686 | 1337→1544 |
| FD score>.2→top256 | 1636→1698 | 1592→1639 | 1628→1689 | 1377→1560 |

p90≤256，覆盖达到all-geometry池的90%，并非覆盖90%全部GT；Snow绝对覆盖88.08/88.99%。NMS前覆盖不等最终TP。

用户摘录FD top256 frame-order AP70 C/F/R/S36.54/35.61/36.95/25.12%，global-sort88.15/84.70/87.86/73.14%；前者早帧低分FP排在后帧TP前，候选数/帧顺序影响巨大。两口径不能互比为收益。

同global-sort口径F top256较原AP70 +.62/+.73/+.63/+3.43pp；agreement再减.51/.75/.32/.40pp，收益来自扩池，非该来源公式。完整candidate_rescore.json未回传，本节重评分数值为用户摘录，不称本地逐字段核验，不推造FD差值。Stage0路径为§25 run下 `candidate_stage0_20260927_212316/candidate_audit.json`。

## 33. 逐目标 Task/Source Oracle（完成，方向B通过）

成功 `/data/cjm/datasets/logs/task_source_oracle_v2_20260928_113748/death_test_results.json`。复用90 validation帧/条件、在线天气，报告基线名Shared，实际权重/协议按run核对（勿与B0其他表仅凭名称混用）；程序要求原B0保存AP≤1e−6复现。

候选与shortlist定义见§31，GT+hindsight贪心依次看整帧匹配、保留原GT、新FP、目标恢复/score/IoU、简单动作。强研究诊断，非可部署性能或任意权重/顺序的数学绝对上界。

| 条件 | Shared | Oracle-Same | Oracle-Task | Task-Shared | Task-Same |
|---|---:|---:|---:|---:|---:|
| Clean | 0.822254 | 0.860024 | 0.871309 | +4.9055 pp | +1.1285 pp |
| Fog | 0.785814 | 0.830193 | 0.840669 | +5.4855 pp | +1.0476 pp |
| Rain | 0.812677 | 0.851623 | 0.859157 | +4.6480 pp | +0.7534 pp |
| Snow | 0.601223 | 0.708617 | 0.731479 | +13.0255 pp | +2.2861 pp |

| 条件 | 方法 | AP30 | AP50 | AP70 |
|---|---|---:|---:|---:|
| Clean | Shared | 0.913628 | 0.893022 | 0.822254 |
|  | Oracle-Same | 0.910690 | 0.895493 | 0.860024 |
|  | Oracle-Task | 0.912750 | 0.901338 | 0.871309 |
| Fog | Shared | 0.895184 | 0.867144 | 0.785814 |
|  | Oracle-Same | 0.889514 | 0.873912 | 0.830193 |
|  | Oracle-Task | 0.889654 | 0.876363 | 0.840669 |
| Rain | Shared | 0.911185 | 0.885773 | 0.812677 |
|  | Oracle-Same | 0.906696 | 0.889730 | 0.851623 |
|  | Oracle-Task | 0.909868 | 0.893536 | 0.859157 |
| Snow | Shared | 0.761571 | 0.729659 | 0.601223 |
|  | Oracle-Same | 0.772873 | 0.760603 | 0.708617 |
|  | Oracle-Task | 0.786517 | 0.772728 | 0.731479 |

| 条件 | Oracle-Same recovered / lost / newFP | Oracle-Task recovered / lost / newFP |
|---|---|---|
| Clean | 39 / 0 / 1 | 52 / 0 / 2 |
| Fog | 39 / 0 / 0 | 50 / 0 / 0 |
| Rain | 43 / 0 / 0 | 55 / 0 / 1 |
| Snow | 105 / 0 / 0 | 140 / 0 / 1 |

三天气mean Task−Shared +7.7197pp、Same−Shared +6.3573pp、Task−Same +1.3624pp，三天气均Task−Shared>1pp，预注册三项全通过：direction_b_survives=true。主要增量在AP70，Fog/Rain AP30反降，不能宣称所有门槛改善。

每条件1753GT，Task changed C/F/R/S1687/1634/1681/1480，task-separated1265/1194/1226/973。大量动作偏好不等新增检出；旧11个Split-only只是一类终态事件，旧Router弱不代表目标级来源空间小。

当前方向为**逐目标、逐任务来源选择**：先识别候选的来源证据，再判断分类/定位收益。不能直接把Oracle source ID当训练标签、把7.72pp当未来收益，或宣称仅天气导致偏好（Clean也有空间）。

## 34. score–geometry distortion直接审计与回放（完成，主张未成立）

§25 run下 `top256_score_geometry_20260928_155556/extraction/`；本地 `logs/候选A/top256/{candidate_audit.json,candidate_hypothesis_replay.json/.md,direct_pair_distortion.json/.md}`，输入/脚本哈希核对一致。代码candidate_audit、candidate_hypothesis_replay、candidate_pair_distortion_audit。固定F为主，每条件90×256候选、9个反复使用场景，无test；单车输出/逐peer移除是关系代理，移除会重归一化，非可相加因果贡献。

配对仅含Clean/weather均入池且source数相同anchor（F16972/R19179/S14953），有选择偏差。score降≥.05且几何支持变化≤.05的低分好框比例 F/R/S7.5/1.7/12.8%，低分坏框2.7/1.6/4.7%；存在不同步，尚不能归因协同融合独有。

同帧IoU>.15潜在竞争好坏对 C/F/R/S135/153/170/168；分数变化指标在部分条件有信号，Fog/Snow场景区间常跨50%，几何一致性在Fog/Snow方向相反，15×4比较未校正多重检验。

按scene五折逻辑探针，固定top256、原NMS/逐帧输出预算，全部填满率1，原AP复现≤1e−6。候选自身+简单一致性→全变化/干预特征的global-sort AP70 C/F/R/S87.11→85.16、83.29→82.69、86.01→84.89、68.62→68.74%；Snow+.13pp区间[−.64,+.83]跨零。

全特征frame-order AP70 .7897/.7686/.7810/.6084，原 .8223/.7858/.8127/.6012；Snow自身探针.6099也高于全特征。相对“自身+来源基础量”，全部特征增量−.43/+.04/+.23/+.34pp区间均跨零。描述性现象成立，稳定纠错、独立distortion贡献、协同因果尚未成立；9场景bootstrap不覆盖重训练/选特征不确定性。

## 35. 候选排序头与倒挂对挽救（完成，停止扩大）

训练导出§25 run下 `candidate_ranker_train_20260929/`，评价同§34；候选geometry/检测器固定，四组同容量MLP：自身、简单一致性、来源基础量、distortion；三种子20260929/20260930/20260931。278658有标签候选、127742竞争对，真正低分好/高分坏倒挂1253对。初轮整体重排distortion mean AP70 .8125/.7689/.7942/.6005，预定门槛0/3天气通过。

`candidate_ranker_rescue.py` 固定6轮末模型，倒挂/普通对等量训练；推理仅原NMS直接抑制的低/高分多来源范围内对，每帧最多交换一对排序再完整NMS/原预算。阈值.70/.80/.90/.95，主阈值预先指定.90，主AP保留原分数/frame-order。

| 天气 | 原流程 | 候选自身 | 简单一致性 | 来源基础量 | 加入 distortion | distortion 平均交换数／净找回 GT |
|---|---:|---:|---:|---:|---:|---:|
| Clean | 0.8223 | 0.8214 | 0.8210 | 0.8183 | 0.8189 | 53.7／-4.0 |
| Fog | 0.7858 | 0.7807 | 0.7809 | 0.7820 | 0.7822 | 55.0／-3.0 |
| Rain | 0.8127 | 0.8094 | 0.8098 | 0.8105 | 0.8102 | 55.7／-2.3 |
| Snow | 0.6012 | 0.6048 | 0.6039 | **0.6079** | 0.6067 | 65.7／+4.7 |

继续门槛：≥2天气较原流程和最佳同容量对照均≥.5pp、净恢复正、剔任一场景仍领先；Clean下降≤.1pp。实际0/3天气通过，Clean亦失败；Snow约+.55pp但来源基础量+.67pp更好。

推理范围直接抑制对9921/10430/10128/8823，真实倒挂仅24/23/37/41；每条件54–66次交换只命中约1.0/1.7/1.3/3.3真实倒挂，多数模糊质量对不进入报告AUC，高AUC不等实际高命中。结果/逐帧/逐边见 `logs/候选A/top256/candidate_ranker_rescue{.md,.json,_frames.jsonl,_edges.jsonl}`。当前distortion无稳定独立收益，不扩大论文核心网络；同9场景继续调阈值不产生独立证据。

## 36. Shared top256区域受限 Oracle（完成，空间保留）

`local_fusion_task_split_pilot/proposal_constrained_oracle.py`；本地 `logs/候选A/车辆互换/{proposal_oracle_results.json,protocol.json}`。同§33 90 validation帧；Shared top256先由推理生成，GT为每目标选择最高重叠proposal（IoU≥.1）限定ROI，GT仍选source/task，非部署。

| 天气 | Shared AP70 | 候选区域 Same AP70 | 候选区域 Task AP70 | Task−Same（百分点） | 原 §33 Task−Same（百分点） | 原 Shared 漏检目标中有候选区域 |
|---|---:|---:|---:|---:|---:|---:|
| Clean | 0.8223 | 0.8635 | 0.8723 | +0.878 | +1.129 | 157/169（92.9%） |
| Fog | 0.7858 | 0.8307 | 0.8411 | +1.037 | +1.048 | 162/231（70.1%） |
| Rain | 0.8127 | 0.8524 | 0.8606 | +0.825 | +0.753 | 164/180（91.1%） |
| Snow | 0.6012 | 0.7097 | 0.7324 | +2.275 | +2.286 | 384/486（79.0%） |

三天气Task−Same +1.379pp vs原+1.362pp；逐目标greedy路径不同，不能逐点当因果差。相同Task状态下Task动作排序胜出 C/F/R/S1264/1194/1227/974，但真正多匹配目标仅11/11/11/34，不能把排序胜出说成检出。Fog/Snow漏检中69/102无IoU≥.1候选，仍有proposal限制。Clean/Fog/Rain AP30略退化。

四个新脚本/配置SHA与本地一致，Shared关键AP相同；服务器引用旧death_test JSON hash `bc09b3486509b9b190eb3f239eb5f8374266f92c8495a09402f88f52dd5c6973`，本地旧JSON `8512b0b995f260a3d074e1f18bb700fc5745fce7cbde35ae6fc75f79b6e5e70c`，只能确认关键AP一致，未确认旧报告逐字节同一，严溯源需复制服务器原件。

## 37. 代表性候选质量/重排序基准（development完成）

冻结F、top256、decoded geometry，适配论文核心机制；各组非完整原detector严格复现。主frame-order AP70，global-sort只诊断；GT Oracle不部署。协议不同，不能排列统一SOTA榜：

| 方法 | 协议/实现边界 |
|---|---|
| CIA-SSD-style | 旧90帧子集+在线天气；3×3 IoU branch，正anchor GT 3D IoU target=2IoU−1，score=s*q^4；未移植完整DI-NMS |
| Adapted LMD-core | 完整1980 validation/在线天气；几何、分数、proposal离散度回归GT BEV IoU，每天气reservoir≤50000；train scene留20%选模型 |
| SAQC | 同LMD baseline、1980帧，实际checkpoint online_legacy；7×7 fused BEV patch+相对坐标，质量回归，score=s*q^1.5 |
| D2D / GossipNet3D | 1980帧固定离线物理天气；全局6层64通道4head / 局部5m四层64通道；native TopK和original NMS都报，主表取后者；官方仓库不可访问，论文机制适配 |

LMD与本次SAQC baseline一致可直接比较；Learned3D天气realization不同，只比较趋势。均未形成统一正式test主表。

| 条件 | Original F | CIA score | GT-IoU Oracle | AP70 变化 | Oracle recovery |
|---|---:|---:|---:|---:|---:|
| Clean | 0.8223 | 0.8275 | 0.9354 | +0.52 pp | 4.63% |
| Fog | 0.7858 | 0.7876 | 0.9075 | +0.18 pp | 1.49% |
| Rain | 0.8127 | 0.8128 | 0.9315 | +0.01 pp | 0.13% |
| Snow | 0.6012 | 0.6116 | 0.8067 | +1.04 pp | 5.06% |

| 条件 | Original F | LMD | GT-IoU Oracle | AP70 变化 | Oracle recovery |
|---|---:|---:|---:|---:|---:|
| Clean | 0.755802 | 0.749301 | 0.919738 | -0.650 pp | -3.97% |
| Fog | 0.726578 | 0.716814 | 0.890370 | -0.976 pp | -5.96% |
| Rain | 0.745695 | 0.738408 | 0.915273 | -0.729 pp | -4.30% |
| Snow | 0.629952 | 0.635059 | 0.835187 | +0.511 pp | +2.49% |

| 条件 | Original F | SAQC top256 | GT-IoU Oracle | AP70 变化 | Oracle recovery |
|---|---:|---:|---:|---:|---:|
| Clean | 0.755802 | 0.756177 | 0.860443 | +0.038 pp | 0.36% |
| Fog | 0.726578 | 0.727038 | 0.848260 | +0.046 pp | 0.38% |
| Rain | 0.745695 | 0.746496 | 0.855756 | +0.080 pp | 0.73% |
| Snow | 0.629952 | 0.632846 | 0.801550 | +0.289 pp | 1.69% |

| 条件 | Original F | D2D | GT-IoU Oracle | AP70 变化 | 按 frame AP70 计算的 Oracle recovery |
|---|---:|---:|---:|---:|---:|
| Clean | 0.755802 | 0.751024 | 0.919738 | -0.478 pp | -2.91% |
| Fog | 0.726541 | 0.721620 | 0.890617 | -0.492 pp | -3.00% |
| Rain | 0.747099 | 0.741817 | 0.915209 | -0.528 pp | -3.14% |
| Snow | 0.624367 | 0.630299 | 0.831605 | +0.593 pp | +2.86% |

| 条件 | Original F | Gossip | GT-IoU Oracle | AP70 变化 | 按 frame AP70 计算的 Oracle recovery |
|---|---:|---:|---:|---:|---:|
| Clean | 0.755802 | 0.756897 | 0.919738 | +0.110 pp | +0.67% |
| Fog | 0.726541 | 0.725943 | 0.890617 | -0.060 pp | -0.36% |
| Rain | 0.747099 | 0.748264 | 0.915209 | +0.117 pp | +0.69% |
| Snow | 0.624367 | 0.643101 | 0.831605 | **+1.873 pp** | **+9.04%** |

| 方法 | Clean ΔAP70 | Fog ΔAP70 | Rain ΔAP70 | Snow ΔAP70 | 恶劣天气平均 ΔAP70 | 协议备注 |
|---|---:|---:|---:|---:|---:|---|
| CIA-SSD-style score | +0.52 pp | +0.18 pp | +0.01 pp | +1.04 pp | +0.41 pp | 旧开发子集；不可与 1980 帧绝对排名 |
| LMD meta-regression | -0.650 pp | -0.976 pp | -0.729 pp | +0.511 pp | -0.398 pp | 1980 帧；online weather |
| SAQC | +0.038 pp | +0.046 pp | +0.080 pp | +0.289 pp | +0.138 pp | 1980 帧；本次实际 online_legacy |
| D2D-Rescore + original NMS | -0.478 pp | -0.492 pp | -0.528 pp | +0.593 pp | -0.142 pp | 1980 帧；fixed physics weather |
| GossipNet3D + original NMS | +0.110 pp | -0.060 pp | +0.117 pp | **+1.873 pp** | **+0.643 pp** | 1980 帧；fixed physics weather |

CIA直接IoU回归Spearman .84–.88却未稳定提高AP。LMD质量Spearman .87–.90、ROC-AUC .957–.971，仍 C/F/R降；recover/lost/newFP C575/826/6751、F602/952/6083、R611/873/6897、S1246/1106/6789，质量准确不保证决策安全。

SAQC有效checkpoint `/data/cjm/datasets/logs/fusion_detector_adaptation_20260927_103227/saqc_full_train_20260929_210426/saqc_quality.pth`：3轮，每轮25496帧、共76488更新，epoch3 843848 quality samples、loss .00161195。最初evaluator把天气branch写死clean的结果作废；有效最终 `logs/sectionB/SAQR/saqc_results.json`，Clean=clean、其余=weather，不能用后来fixed README重解释旧online checkpoint。q Spearman .79–.83，最终score .71左右；未Platt时Q-ECE变差，质量转分数只追回<2% frame Oracle。

Gossip报告方法名残留d2d_topk/d2d_original_nms，实际为GossipNet3D。Snow+1.873pp/9.04% Oracle recovery是最明显信号，Fog仍略降。不能称所有SOTA失败；当前共同难点是把质量转成竞争、NMS和保留/删除决定。benchmark阶段结束，不在同validation反复调beta/hidden/阈值；若论文需正式统一主表，应冻结方法后同协议一次test，不由本页自动授权执行。

## 38. S0–S3 Action Utility 审计（2026-10-09完成）

代码 `local_fusion_action_utility_audit/`，run `/data/cjm/datasets/logs/action_utility_audit_20261007_210243`；本地报告 `logs/sectionB/all/FINAL_RESULTS.md`。复用B0 train/val scenes/indices，不访问test。train scenes固定拆probe-fit/calibration，validation只最终development；Shared完整验证AP先≤1e−6复现，S0/S3共用精确FP32快照并核对缓存/输入哈希。

Shared top256、ROI expansion1.5、KEEP_SHARED/single:i/query:i；每动作从同一原始Shared出发，未用逐GT greedy路径作标签。GT仅association/outcome/评价；S3-B proposal/action决策GT字段0。S1/S2只用单层Linear pairwise ranker。

**S0：**恶劣天气关联候选最佳cls/reg来源不同率69.79%（预定明显差异线10%）。固定tie-break可能把等效outcome选成不同来源，不能解释为69.79%目标必须分离，也不等AP增量。

**S1/S2：**最终S1_WEAK、S2_WEAK。加competition特征pairwise accuracy增量cls C/F/R/S约.0039/.0047/.0044/.0073，reg .0037/.0025/.0024/.0052，regret几乎不变。只否定当前output手工特征+线性模型，不证明收益本身或底层证据不可学习。

**S3 AP70：**

| Method | Clean | Fog | Rain | Snow | adverse mean ΔAP70 |
|---|---:|---:|---:|---:|---:|
| Shared | 0.822254 | 0.785814 | 0.812677 | 0.601223 | +0.0000 pp |
| Assisted-Cls | 0.820579 | 0.786624 | 0.811931 | 0.604857 | +0.1233 pp |
| Assisted-Reg | 0.822254 | 0.785814 | 0.812677 | 0.601223 | +0.0000 pp |
| Assisted-Task | 0.820579 | 0.786624 | 0.811931 | 0.604857 | +0.1233 pp |
| Proposal-Cls-Greedy | 0.810538 | 0.780065 | 0.799685 | 0.593451 | -0.8838 pp |
| Proposal-Reg-Greedy | 0.822254 | 0.785814 | 0.812677 | 0.601223 | +0.0000 pp |
| Proposal-Task-Greedy | 0.810538 | 0.780065 | 0.799685 | 0.593451 | -0.8838 pp |
| Proposal-Task-Conservative | 0.822254 | 0.785814 | 0.812677 | 0.601223 | +0.0000 pp |

四条件总recover/lost/newFP/changed：Assisted-Cls/Task17/11/30/2083，Assisted-Reg全0；Proposal-Cls23/49/158/11437，Proposal-Task23/49/158/11438，Conservative全0。Reg仅1次Proposal修改，Assisted-Task与Cls完全相同，定位来源尚未学会。

GT只限定关联proposal时恶劣mean+.1233pp；全部proposal执行变−.8838pp。保守校准没有可靠margin，全KEEP。首要瓶颈为**当前输出信号未可靠识别来源收益**，候选扩散/竞争再放大错误；不能称NMS唯一根因或直接跳复杂GNN。

下一步S4按 **Evidence→Utility→Action→Competition** 顺序检查：先判断某来源在当前候选中提供什么几何/语义信息，再预测任务收益、执行动作、处理竞争。复用旧labels/split/action pool/线性模型，仅增加底层evidence。per-source BEV、预测ROI的有效点/voxel支持与覆盖、GSPR可靠性/不确定性、重算leave-one-out响应均须核对坐标与validity；来源移除响应不等可相加因果贡献。

S4代码状态与操作详见 `local_fusion_source_evidence_audit/README.md`：默认BASE_RUN为上述旧run；先R0复现原模型/指标≤1e−6，再GEO/SEM/ABLATIONS/REPLAY/FINAL；主组固定all_evidence。旧缓存/模型/源码/标签必须完整校验；整目录同步含source_snapshot，不更换旧标签、不放宽复现门槛。**尚无S4实测结论**，不能把代码实现或静态检查当性能验证。

维护本文件时，每个实验只保留一份协议、关键表、结果路径和决定；结果回传后直接更新“待运行”状态，避免继续叠加重复进度公告。精细训练曲线、完整逐帧数据、重复命令优先保留在原报告/README。
