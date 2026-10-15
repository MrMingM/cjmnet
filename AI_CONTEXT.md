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

`gspr_evidence/`：risk 11940参数、gain 11521参数；训练标签用等字节真实 block replacement 的检测损失变化，保留负收益；17字段请求、256 blocks、总256 KiB包含metadata/feature。teacher完整扫描train/val，每peer/frame最多8候选block，不能误写成只用8帧。matching/concat/no_u对照；早期 global-sort 开发结果不能冒充正式测试。

## 11. 统一正式协议与 learned communication 结果（已完成）

正式比较固定 OPV2V clean test + OPV2V-W Fog/Rain/Snow test，关闭在线天气/数据增强，读取 `processed_lidar`，全场景全帧 stride=1。AP30/50/70 使用 **BEV 平面多边形 IoU** 与 `global_sort_detections=False`（按帧累计）；它与跨帧统一分数排序 global-sort AP、height-aware 3D IoU 不同。validation+在线天气只作开发，不能与test AP直接相减。

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
|---|---:|---:|---:|---:|
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

## 39. S4 Source Evidence Audit（2026-10-13完成）

结果文件 `logs/s4/FINAL_RESULTS.md`；代码 `local_fusion_source_evidence_audit/`。本轮不访问正式 test，仍属 development/在线天气协议。复用 S0 的 counterfactual action label、Shared top256、候选池、fit/calibration/validation split、线性 pairwise ranker、训练超参数和 replay 规则；唯一主要变量是把旧 output-level feature 扩展为 source-local geometry、semantic、GSPR 与 leave-one-out（LOO，移除单来源后重算）证据。归一化只用 fit，保守阈值只用 train calibration；GT 仅用于旧标签、Assisted 关联和最终评价，inference feature 中 GT=0。

### 39.1 预注册结论

最终判定：`GEO_EVIDENCE_WEAK / SEM_EVIDENCE_WEAK / REPLAY_FAIL`。因此不能宣称更底层 source evidence 已经稳定解决来源效用学习；同时也不能把当前失败解释成“source utility 不可学习”，因为本轮仍固定为手工统计 evidence + 线性模型。

### 39.2 S4-GEO：几何 evidence 未改善整体 regression source ranking

恶劣天气 mean pairwise accuracy / regret：

| Feature group | Pairwise acc | Regret |
|---|---:|---:|
| output_only | 0.820068 | 0.127281 |
| output_plus_geometry | 0.816143 | 0.126536 |
| output_plus_gspr | 0.819124 | 0.126311 |
| output_plus_geometry_semantic | 0.814746 | 0.126392 |
| all_evidence | 0.812621 | 0.126953 |
| all_evidence_without_loo | 0.814865 | 0.127046 |

`all_evidence` 相对 output_only 的 pairwise accuracy 反降约 0.745 pp，regret 仅轻微改善；没有达到 `pairwise +0.02` 与 `regret reduction ≥20%` 的 evidence gate。当前点/柱支撑、预测框覆盖、跨来源几何一致性、GSPR 与 LOO 统计不能用简单线性关系稳定回答“哪个 source 的 regression 更好”。这不等于几何证据无用，只否定当前表达与线性组合。

GSPR 对 GEO 的独立收益很小：`output_plus_gspr` 相对 output_only 的 regret reduction 约 0.76%；`all_evidence` 去掉 GSPR 也没有恶化。LOO 对 GEO 也几乎无增益，说明“某来源影响有多大”不等于“该影响方向正确”。后续不把 GSPR/LOO 强制视为核心组件。

### 39.3 S4-SEM：存在中等语义信号，但未越过预注册门槛

恶劣天气 mean pairwise accuracy / regret：

| Feature group | Pairwise acc | Regret |
|---|---:|---:|
| output_only | 0.843626 | 0.175226 |
| output_plus_semantic | 0.851801 | 0.155693 |
| output_plus_gspr | 0.845380 | 0.165068 |
| output_plus_geometry_semantic | **0.852211** | **0.147179** |
| all_evidence | 0.850238 | 0.151047 |
| all_evidence_without_loo | 0.849711 | 0.155564 |

最佳 `output_plus_geometry_semantic` 相对 output_only 的 pairwise accuracy 约 +0.86 pp，regret 从 0.175226 降至 0.147179，约下降 16%；说明更底层 semantic evidence 确有可利用趋势，但仍未达到预注册的 pairwise 增益 +0.02 和 regret reduction 20% 门槛，因此正式判定仍是 `SEM_EVIDENCE_WEAK`。不能据此宣称 semantic evidence 已经“可学习成功”，但它比 GEO 的平均 ranking 信号更值得继续验证非线性交互。

### 39.4 Replay：最重要的新现象是 regression 第一次被真正激活

AP70：

| Method | Clean | Fog | Rain | Snow | adverse mean ΔAP70 |
|---|---:|---:|---:|---:|---:|
| Shared | 0.822254 | 0.785814 | 0.812677 | 0.601223 | +0.0000 pp |
| Evidence-Assisted-Cls | 0.823646 | 0.787343 | 0.812926 | 0.600294 | +0.0283 pp |
| Evidence-Assisted-Reg | 0.822849 | 0.785814 | 0.812677 | 0.604846 | **+0.1208 pp** |
| Evidence-Assisted-Task | 0.824239 | 0.787343 | 0.812926 | 0.604706 | **+0.1754 pp** |
| Evidence-Proposal-Cls-Greedy | 0.817099 | 0.779646 | 0.806460 | 0.593726 | -0.6628 pp |
| Evidence-Proposal-Reg-Greedy | 0.820828 | 0.785059 | 0.812677 | 0.603476 | **+0.0499 pp** |
| Evidence-Proposal-Task-Greedy | 0.815726 | 0.778683 | 0.806364 | 0.597680 | -0.5662 pp |
| Evidence-Proposal-Task-Conservative | 0.822254 | 0.785814 | 0.812677 | 0.601223 | +0.0000 pp |

Fog/Rain/Snow 计数汇总：Assisted-Reg `5 recovered / 0 lost / 0 newFP / 772 changed`；Proposal-Reg-Greedy `5 / 2 / 2 / 1784`。这是相对 S3 的关键变化：S3 的 Assisted-Reg 完全 0 修改、Proposal-Reg 仅约 1 次修改，而 S4 底层 evidence 让 regression source action 第一次出现少量真实、低误伤收益。Snow Assisted-Reg AP70 从 0.601223 提升到 0.604846，约 +0.362 pp。平均 ranking gate 虽失败，但这批高价值动作说明 regression 路线尚未死亡。

Classification 仍是主要风险源。Proposal-Cls-Greedy 共 `19 recovered / 35 lost / 89 newFP / 8155 changed`；Proposal-Task-Greedy 为 `24 / 36 / 90 / 8289`，导致恶劣天气 mean AP70 -0.5662 pp。说明模型并非完全找不到有益 action，而是扩展到全部 proposal 后错误 action 增长更快，WHEN-TO-ACT 与后续 candidate competition 仍未解决。

Conservative 版本再次全部 KEEP：0 changed、AP 与 Shared 完全一致。当前模型仍找不到一个满足校准安全约束的高置信度修改区间，因此问题不只是 source 排名，还包括“什么时候应该修改 Shared”。

### 39.5 当前研究判断

S4 后方向 B 的瓶颈应拆为三层：

1. **Evidence → Utility：**有弱/中等信号，但手工 evidence + Linear 尚未稳定；SEM 趋势强于 GEO 的平均 ranking。
2. **Utility → Action / WHEN-TO-ACT：**仍明显不足。尤其 classification 修改过多，保守策略只能全 KEEP。
3. **Action → Competition：**全部 proposal 执行后 lost TP 与 new FP 被进一步放大；candidate competition 仍重要，但不应在 source utility 尚弱时直接跳复杂 GNN。

S4 不支持继续盲目堆点数、IoU、cosine、GSPR、LOO 等手工特征。下一步优先做严格单变量的**小型非线性 utility probe**：保持 S4 evidence、labels、split、action、candidate pool 与 replay 不变，只把 Linear 替换为很小的 MLP，分别验证 regression 与 classification 的 evidence 是否存在非线性交互。优先级为 regression → semantic/classification。只有当 utility 明显变得可学而 Proposal replay 仍失败时，再进入 joint action / candidate competition。

若小型非线性模型仍不能明显提高 Assisted 与安全 Proposal replay，则应重新审视“整体 source-level regression/classification utility”标签是否过粗，特别是 regression 可能需要按 `x/y/size/yaw` 等可观测维度细分，而不是继续扩大网络容量。

## 40. Source Utility Kill Protocol：Stage 1 预注册

预注册日期：2026-10-13。当前 source utility 状态：**PROBATION / 待判死**。本节取代 §39.5 的后续开发建议：从现在开始，K3 tiny MLP、新 source 模型开发、x/y/yaw 标签细分、source GNN、joint-action source network 全部冻结。只有未来 **K1 PASS AND K2 PASS** 才允许恢复 K3。

阶段 1 仅包含 **K0′ Group-level Label Margin Audit** 与 **K1-0 Oracle Outcome-space Decomposition**，二者并行执行。代码放 `local_fusion_source_kill_stage1/`；不训练、不修改 GSPR/detector/B0/S4，不访问正式 test，不启动 K1/K2/K3，不建立 branch/worktree。结果与 cache 放 `/data/cjm/datasets/logs/`；运行依赖源码哈希，不依赖 Git。

K0′ 只做标签结构审计。K1-0 只做 Oracle 收益来源分解。二者无论结果如何，本阶段都不自动执行 K3。真正 source-specificity 判定由下一阶段 K1 完成；label stability 判定由下一阶段 K2 完成。本阶段结束后停下，输出阶段 1 结果，等待人工判断。

从 Stage 1 正式开始日起，K0′ + K1-0 + K1 + K2 的 source kill suite 总项目投入上限为 **14 个自然日**。这是 **resource stop**，不构成科学意义的 hypothesis rejection。首次服务器 preflight 通过时保存项目开始日与截止日，后续 RUN 共用时钟；实现/本地静态检查不算正式开始。

### 40.1 K0′ 固定协议

直接读取 §38 原 S0 run 的完整 per-group action outcomes，不重新运行 detector，不重新生成 counterfactual label。启动时核对 S0 complete、源码/协议/缓存哈希、B0 train/validation indices 与 scene split、原浮点修复的已有 provenance。group 为 split/weather/frame/Shared proposal/task/focal association 的完整 KEEP/single/query 动作集合。primary scope 固定 validation-associated；train 检查趋势，background 单列。

严格调用原 `path_independent_action_key()` / `action_keys()`：顺序为 action_tp、-lost、-new_fp、focal_detected_after、recovered、-action_fp、focal_score_after、focal_iou_after、KEEP preference。`discrete6=key[:6]`、`outcome8=key[:8]`、`full9=key[:9]`。按 full9 排序比较第一/第二：discrete6 不同为 DISCRETE_MARGIN；仅 outcome8 不同为 CONTINUOUS_ONLY_MARGIN；outcome8 相同为 EXACT_OUTCOME_TIE，KEEP 简化偏好不冒充真实 margin。KEEP-only 单列并纳入 exact tie，runner-up 不存在单独计数。统计最大 outcome8 等价动作数量、均值/中位数/多重并列比例，不把任意并列 source 叫作唯一 best。

相对 KEEP，存在 non-KEEP discrete6 更优为 beneficial discrete set；无离散更优而存在 discrete6 相同且 outcome8 更优为 continuous-only benefit；存在 discrete6 更差为 harmful discrete exists。输出所有 split/weather/task/stratum 的计数与比率、离散 margin 与 beneficial set 的 scene 覆盖及单 scene 最大占比。重新核对旧 LABEL_AUDIT 的 action-level TP 增加/减少/不变，按原两位百分数舍入精度逐项一致才通过；不一致立即失败并检查 schema。这里只称“能直接带来净 TP 增长的 action 比例”。

K0_SIGNAL 仅描述：cls/reg 各自 adverse mean discrete-margin 均 ≥50% 为 STRONG；均 <20% 为 WEAK；其余 MIXED。此命名不构成 STOP，不触发/取消 K1/K2，K3 保持冻结。adverse mean 为 Fog/Rain/Snow 等权均值，cls/reg 分列；同时保留 pooled group 统计。

### 40.2 K1-0 固定协议

直接适配 §33 `run_oracle()`，沿用 build_candidate_pool、build_proxy_pool、compose_prediction、_target_order、_action_key、_shortlist_actions。seed=20260927、target_iou=.7、ROI=[1.0,1.5]、families=[single,query]、shortlist_k=6。Same 保留 rows[:6]；Task 保留 overall/separated/same 各 top-k 后去重，绝不强制截成 6。

每个 mode 的 FULL/A_ONLY/B_ONLY 都独立从 Shared 开始，保留原目标顺序；每个 target 在自身 current greedy state 下重新 shortlist、compose、postprocess、_state、_action_key。KEEP key 用 current_state 对自身；A 为 action[:6] != KEEP[:6]；B 为 [:6] 相同且 [:8] 不同；E 为 [:8] 相同。A_ONLY/B_ONLY 只允许 KEEP 加当时对应组；FULL 允许全部原 shortlist。禁止继承 S0 静态 Group A/B。适配器执行原 run_oracle 函数体，只令被禁止动作返回当前 KEEP key，从而不能胜出；原模块文件保持不动。

先完成四天气 Shared/Full-Same/Full-Task 的 AP30/50/70 复现门禁，分别与原 §33 保存结果 abs diff ≤1e−6；失败 STOP Stage 1，不解释或继续 A/B。通过全部 FULL 门禁后才运行 restricted trajectories。每个 weather×mode×variant 独立原子保存并按哈希 resume。

报告四天气及 adverse mean 的全部 AP30/50/70；核心 AP70 pp 为各 variant 的 Task−Shared 与 Task−Same。先报绝对 pp 后报 retention；分母绝对值 <.5pp 标记 UNSTABLE_RATIO。decomposition 是 **restricted-policy attribution，不是线性可加分解**；A/B 收益不可简单相加解释为 Full。按 weather/mode/variant 输出 targets、changed、shortlisted/selected A/B/E/KEEP、availability、task-separated、recovered/lost/new_fp，终态相对 Shared 与逐步后果分别记录。

自动文字描述固定使用 .5pp 为“明显收益”，Full retention ≥.7 为“大部分”、≤.3 为“很弱”，仅辅助描述；不用于 kill 或恢复开发。A 为主时下一阶段重点 N2 random source mixing；B 为主时重点 N1 matched output perturbation 与 score-only calibration；两者均明显时完整验证 N1+N2+score-only。报告保留所有原数值与边界，不宣称 source utility 已证明可学。

### 40.3 执行与输出边界

POSIX sh/LF；preflight→纯单元测试→K0 CPU 与 K1-0 GPU 并行→独立 PID/status→全部成功才汇总。所有入口 realpath 拒绝正式 test 及其别名，dataset 根与嵌套链接先检查再加载；不读取正式 test 内容。GPU 默认 ROCR_VISIBLE_DEVICES=2，unset HIP_VISIBLE_DEVICES/CUDA_VISIBLE_DEVICES；K0 隐藏全部 GPU。完整报告为 K0_GROUP_MARGIN、K1_0_ORACLE_DECOMPOSITION、STAGE1_FINAL 的 JSON/Markdown。原子写入与独立 worker 状态避免中断污染及并发 manifest 丢更新。仅实现与检查完成后交付 launch 命令，由用户决定正式启动；不得自动运行长实验或失败后改变定义/ROI/shortlist/标签。

## 41. Source Utility Kill Protocol：Stage 1 结果（2026-10-14完成）

结果目录 `logs/k/`，关键文件为 `K0_GROUP_MARGIN.{md,json}`、`K1_0_ORACLE_DECOMPOSITION.{md,json}`、`STAGE1_FINAL.{md,json}`。本阶段严格沿用 §40 预注册：K0′ 与 K1-0 并行，不训练，不修改 GSPR/detector/B0/S4，不访问正式 test，不自动启动 K1/K2/K3。K3 与所有新 source 模型开发继续冻结。

### 41.1 实验过程与一致性门禁

**K0′** 直接读取 §38 S0 的 per-group counterfactual action outcomes，不重新运行 detector。主范围固定为 `validation-associated`，每个 group 比较同一候选/任务下 KEEP、single:i、query:i 的完整 action key；`discrete6` 用于离散检测后果，`outcome8` 再包含 focal score/IoU。报告 best-vs-runner-up 的 DISCRETE_MARGIN、CONTINUOUS_ONLY_MARGIN、EXACT_OUTCOME_TIE，以及是否存在 non-KEEP 在离散结果上优于 KEEP 的 beneficial discrete set。旧 LABEL_AUDIT 的 action-level TP increase/decrease/unchanged 也重新核对，只作为背景风险比例，不与 group-level margin 混用。

**K1-0** 适配 §33 的逐目标 GT-hindsight greedy Oracle。Full、AOnly、BOnly 三条轨迹均独立从 Shared 开始；每一步都在各自 current greedy state 下重新 shortlist、compose、postprocess 并与当前 KEEP 比较。A 表示 `discrete6` 与 KEEP 不同，B 表示 `discrete6` 相同但 `outcome8` 不同；没有把 S0 的静态 A/B 标签直接套到 §33。四天气 Shared/Full-Same/Full-Task 的 AP30/50/70 均先通过对原 §33 结果的 `abs diff <= 1e-6` 复现门禁，再运行 restricted trajectories。

### 41.2 K0′：组级标签结构为 `K0_SIGNAL_WEAK`

`validation-associated` 主表：

| Weather | Task | Groups | Discrete margin | Continuous-only | Exact tie | Beneficial-set |
|---|---|---:|---:|---:|---:|---:|
| Clean | cls | 1741 | 5.28% | 91.96% | 2.76% | 5.92% |
| Clean | reg | 1741 | 0.80% | 96.38% | 2.81% | 1.38% |
| Fog | cls | 1684 | 4.93% | 91.51% | 3.56% | 6.83% |
| Fog | reg | 1684 | 1.07% | 96.26% | 2.67% | 1.72% |
| Rain | cls | 1737 | 4.84% | 92.17% | 2.99% | 6.39% |
| Rain | reg | 1737 | 0.86% | 96.49% | 2.65% | 1.67% |
| Snow | cls | 1651 | 8.30% | 78.38% | 13.33% | 12.30% |
| Snow | reg | 1651 | 2.00% | 85.40% | 12.60% | 3.45% |
| adverse mean | cls | — | **6.02%** | **87.35%** | **6.63%** | **8.50%** |
| adverse mean | reg | — | **1.31%** | **92.72%** | **5.97%** | **2.28%** |

按 §40 的描述性门槛，cls/reg 的 adverse mean discrete-margin 均低于 20%，因此 K0′ 记为 `K0_SIGNAL_WEAK`。这说明“给每个候选预测唯一最佳 source”的广泛 ranking 目标结构很弱：绝大多数 best-vs-runner-up 只由 focal score/IoU 连续量拉开，真正有离散检测后果 margin 的组很少；regression 尤其稀疏。

Action-level 背景同样显示强烈风险不对称。non-KEEP 中，能直接带来净 TP 增长的动作不到 1%，而净 TP 减少约 9–19%：Clean cls 0.38% vs 10.06%、Fog cls 0.35% vs 11.43%、Rain cls 0.42% vs 9.60%、Snow cls 0.86% vs 19.03%；Clean reg 0.35% vs 8.95%、Fog reg 0.46% vs 10.77%、Rain reg 0.45% vs 8.97%、Snow reg 0.98% vs 14.57%。这里不能把“TP increase <1%”偷换成“所有有益动作 <1%”，因为 TP 身份不变但分数/IoU 改善仍可能影响 AP；该统计只说明 source intervention 是极端高风险、低基率问题，不能再把失败单纯解释成 Linear/MLP 容量不足。

### 41.3 K1-0：Oracle headroom 几乎全部由少量离散 outcome-changing 动作贡献

AP70：

| Weather | Shared | Full-Same | Full-Task | AOnly-Same | AOnly-Task | BOnly-Same | BOnly-Task |
|---|---:|---:|---:|---:|---:|---:|---:|
| Clean | 0.822254 | 0.860024 | 0.871309 | 0.859931 | 0.871223 | 0.822269 | 0.822298 |
| Fog | 0.785814 | 0.830193 | 0.840669 | 0.830171 | 0.840632 | 0.785799 | 0.785801 |
| Rain | 0.812677 | 0.851623 | 0.859157 | 0.851572 | 0.859124 | 0.812709 | 0.812664 |
| Snow | 0.601223 | 0.708617 | 0.731479 | 0.708514 | 0.731349 | 0.601315 | 0.601224 |
| adverse mean | 0.733238 | 0.796811 | 0.810435 | 0.796753 | 0.810368 | 0.733275 | 0.733230 |

对应 AP70 收益（pp）：

| Weather | Full Task−Shared | A Task−Shared | B Task−Shared | Full Task−Same | A Task−Same | B Task−Same |
|---|---:|---:|---:|---:|---:|---:|
| Clean | +4.9055 | +4.8969 | +0.0044 | +1.1285 | +1.1292 | +0.0029 |
| Fog | +5.4855 | +5.4818 | -0.0013 | +1.0476 | +1.0461 | +0.0002 |
| Rain | +4.6480 | +4.6447 | -0.0013 | +0.7552 | +0.7552 | -0.0046 |
| Snow | +13.0255 | +13.0125 | +0.0001 | +2.2861 | +2.2835 | -0.0091 |
| adverse mean | **+7.7197** | **+7.7130** | **-0.0009** | **+1.3624** | **+1.3616** | **-0.0045** |

AOnly 的 adverse mean `Task−Shared` 保留 Full 的约 **99.91%**，`Task−Same` 约 **99.94%**；BOnly 基本为 0。Full Task 每条件 changed C/F/R/S 为 1687/1634/1681/1480，而 AOnly 仅 65/69/69/168，却保留同样的终态 recovered/lost/newFP：Clean 52/0/2、Fog 50/0/0、Rain 55/0/1、Snow 140/0/1。也就是说，原 Full Oracle 中大量 B 类连续 score/IoU 改动对最终 AP70 几乎是陪跑，真正的 headroom 集中在几十到一百多个会改变离散检测后果的关键动作。

K1-0 因而明确反驳了一个先前的重要怀疑：§33 的 +7.72 pp **并不是主要由大量 continuous score/IoU ranking 微调制造的**。在当前 restricted-policy attribution 下，几乎全部收益来自 A 类离散 outcome-changing action。注意 AOnly/BOnly 是各自独立 greedy trajectory，不是线性可加分解，不能把二者收益简单相加解释 Full。

### 41.4 当前生死判断与边界

Stage 1 的综合判断为：**K0′ = `K0_SIGNAL_WEAK`，K1-0 = 机制分解 PASS；source utility 仍处于 PROBATION，不判死，也不恢复模型开发。**

K0′ 的坏消息是 source 决策信号极度稀疏，广泛“每候选选最佳 source”的学习目标不合适；这与 S3/S4 Conservative 反复全 KEEP、Greedy 大量误伤相一致。K1-0 的好消息是，这些极少数关键 A 类动作却解释了几乎全部 Oracle AP70 headroom，因此不能把原 Oracle 简化成普通 ranking hindsight 假象。

但 K1-0 **没有证明 source identity 特异性**：仍可能是任意匹配幅度的局部输出扰动，或随机重新混合 CAV 权重，也能偶然制造类似的离散修复，再由 GT hindsight 挑中。它也没有证明标签稳定或可学习。因而不能把 `AOnly≈Full` 当作继续投入新模型的授权。

当前最合理的问题定义已从“对所有候选学习 source utility ranking”收缩为：**绝大多数候选保持 KEEP，只识别极少数值得执行的安全 source intervention。** 这是结构性认识，不授权直接设计 rare-event MLP/GNN。

### 41.5 下一阶段固定决策

下一步只允许并行执行 **K1 source-specificity** 与 **K2 label stability**；K3 tiny MLP、x/y/size/yaw 标签细分、source GNN、joint-action source network 继续冻结。

- **K1** 必须包含 N1 matched output-space perturbation、N2 random source mixing 和 score-only 对照，并与 Source Oracle 使用同一 shortlist/GT-aware proxy/greedy 流程、按帧 CAV 数对齐动作数；随机对照至少 3 seeds。Stage 1 显示 A 类动作为主，因此 N2“是否必须选对 source/CAV”是核心审判，N1 仍用于排除 generic perturbation。
- **K2** 只在预注册的 Group A/B 范围内检查轻微扰动后的 top-1、top-2、KEEP-vs-MODIFY 与 A/B membership 稳定性，并设置最小有效 group/scene 数；样本不足导致不可判定时按预注册 FAIL 处理，不能成为继续投入理由。
- **最终停止规则不再后改：K1 FAIL 或 K2 FAIL/INCONCLUSIVE → source utility 主线 STOP，不运行 K3；只有 K1 PASS 且 K2 PASS 才恢复 K3 资格。**

本阶段结果仍来自反复使用的 development/validation 场景与 GT-hindsight Oracle，不能当独立泛化或可部署收益；正式 test 继续禁止用于 source kill suite 调参和判方向。

## 42. Source Utility Kill Protocol：Stage 2 最终冻结协议（2026-10-14）

本节在 K1/K2 正式运行前冻结，**若与 §41.4–§41.5 的旧表述冲突，以本节为准**。P0 前置审计完整结果以 `logs/k/P/` 为主；顶层 `logs/k/P0_SOURCE_REDUNDANCY.*` 的早期不完整版本只保留历史，不参与判断。P0-A 显示 validation adverse beneficial groups 中跨 source 等价最佳比例 cls=20.28%、reg=26.09%，因此多数 beneficial group 的最佳离散结果仍只来自一个 source；P0-B 显示 validation adverse joint Same positive targets=259、same-state true Task-positive=64。K1-0 只记为 **A-dominant descriptive decomposition**，没有 PASS/FAIL 身份。

### 42.1 两层科学假设与停止逻辑

Stage 2 不再把 Same 与 Task 绑成单一生死结论：

- **H1 / Source-specific intervention：**某些目标是否真的需要特定 source，而不是任意匹配幅度的扰动或随机 source mixing 都能达到类似收益。由 K1-Same + K2-Same 判定。
- **H2 / Task-specific source intervention：**在 H1 成立的前提下，classification / regression 是否真的需要分开选择 source。由 K1-Task + K2-Task 判定。

停止规则固定：

1. `K1-Same FAIL/INCONCLUSIVE` 或 `K2-Same FAIL/INCONCLUSIVE` → **整个 source utility 主线 STOP**，不运行 K3，不再以 MLP/GNN/标签细分继续解释。
2. H1 PASS，但 `K1-Task FAIL/INCONCLUSIVE` 或 `K2-Task FAIL/INCONCLUSIVE` → **Task separation STOP**，只允许保留 Same-only source intervention 作为候选；不能再宣称 cls/reg 分源是核心机制。
3. 只有对应层的 K1 与 K2 均 PASS，且未来通过 §42.5 的 K3 data-adequacy gate，才获得对应 K3 学习实验资格。

14 天 resource stop 保持不变：项目时钟截止 `2026-10-27T03:23:56.508866+00:00`。超时记 `RESOURCE_STOP`，不是科学意义上的 hypothesis rejection，但项目层面停止继续投入本主线。

### 42.2 K1：source-specificity 的最终协议

K1 主评价只使用 Fog/Rain/Snow development/validation，Clean 只做 Source 复现与每类 null 一个 sanity seed，不参与 PASS/FAIL。Source 必须严格复现 §33/§40 的 Shared、Same、Task、GT-aware proxy、target order、ROI、`shortlist_k=6` 与完整 postprocess/NMS；AP30/50/70 复现门禁 `abs diff <= 1e-6`。

Null families 固定三类：

1. **N1 matched output-space perturbation：**不保留 source identity。扰动必须从既有 train source-action 的实际输出变化中采样，至少匹配 `Δscore` 与 box 的 center/size/yaw 变化；禁止只在权重空间“看起来差不多”。按 weather/task/action family 可行时分层采样，validation 不用于调 null 强度。
2. **N2 random source mixing：**使用当前帧同一批有效 CAV，随机混合权重；每个 frame/target 生成的 non-KEEP 候选数量必须与 Source action space 一致，即与当时 `single:i + query:i` 的数量对齐，再加同一个 KEEP。N2 与 Source 必须走**完全相同**的 GT-aware proxy、同样 shortlist 规则、同样 top-6 容量、同样完整 NMS 与 greedy；禁止 random 直接随机抽 6 个而 Source 先过 GT proxy。
3. **Score-only matched null：**保持 geometry 不变，仅生成与 source-induced score 变化幅度相匹配的候选；action count、proxy、shortlist、greedy 与 Source 对齐，用于判断仅分数校准能否复制收益。

N1/N2/score-only 随机 null 均至少 3 seeds；正式 seeds 在运行前写入 config 并冻结。每个 null family 先对 3 seeds 求 per-weather/per-scene gain，再以 seed 平均作为主比较，同时完整报告每个 seed 与 worst seed。N2 额外报告与 Source 的输出变化幅度：`|Δscore|`、center shift、size change、yaw change、candidate IoU change 的 median/p90；若实现发现 N2 明显处于更弱 perturbation 尺度，必须在运行前仅用 train、且只用 output delta 不看 AP/GT outcome 的固定校准规则调整随机混合分布。正式 validation 结果出来后禁止再调。

K1-Same 主效应：`Source Same−Shared` 对 `Null Same−Shared`。K1-Task 主效应：`Source Task−Same` 对 `Null Task−Same`。Source 与 Null 均用 Fog/Rain/Snow 三天气等权 mean AP70（pp）。对每个 null family 独立比较；只要任一 family 命中任一失败条件，对应 H1/H2 的 K1 即 FAIL：

- `Null_gain / Source_gain >= 0.60`；
- `Source_gain - Null_gain < 0.50 pp`；
- paired scene-bootstrap 的 `Source_gain - Null_gain` 95% CI 包含 0。

Bootstrap 固定以 validation scene 为重采样单位，Fog/Rain/Snow 的同 scene 配对保留；至少 5000 次，随机种子 `20261014`。若 Source gain <=0、复现门禁失败、null 不公平/无法完成必要的 action-count/shortlist/output-scale 对齐，则对应 K1 记 `INCONCLUSIVE`，按上述项目停止逻辑处理，不能据此继续开发模型。

### 42.3 K2：label stability 的最终协议

K2 与 K1 并行；不训练模型。Primary 对象固定为 P0 已核出的 **validation joint Oracle records**，不把 S0 cls/reg positive 简单相加冒充 joint Same。K2-Same 使用 adverse pooled `joint Same positive targets`；K2-Task 使用 adverse pooled `same-state true task-positive`。train/S0 可作为辅助稳定性统计，但不进入 H1/H2 主 PASS/FAIL。

**样本资格门槛先检查，结果出来后不得改：**

- K2-Same：`n >= 200`、覆盖 `>=7` validation scenes、单 scene 最大占比 `<=40%`。当前 P0 为 n=259、9 scenes、21.24%（55/259），具备运行资格。2026-10-14 经人工确认，按已通过 manifest 哈希校验的正式 P0 `same_scene_coverage` 修正基准描述；此前 8 scenes / 29.69% 引用了 Task 字段，所有判定门槛保持不变。
- K2-Task：`n >= 50`、覆盖 `>=5` validation scenes、单 scene 最大占比 `<=40%`。当前 P0 为 n=64、8 scenes、29.69%，具备运行资格，但必须报告原始 flip counts，不能只报百分比。

K2 perturbation 在正式运行前固定为小扰动，不得根据结果调强度。以 baseline focal proposal/ROI 为对象，GT evaluation 本身不移动：

1. proposal center：x/y 分别 `±0.10 m`，四个单轴变体；
2. proposal yaw：`±2°`，两个变体；
3. ROI/context 尺度：对 baseline ROI multiplier 乘 `0.97` 与 `1.03`，两个变体。

每个 perturbation 都从同一 baseline current state 重新 shortlist、compose、postprocess、算 action key；不能只在保存的 label 上做字符串替换。报告 top-1、top-2 set/Jaccard、KEEP-vs-MODIFY、A/B membership、source identity/source-set 与 Task-vs-Same 状态变化。

K2-Same 的 primary stability gate：

- KEEP-vs-MODIFY agreement 在每类 perturbation 上均 `>=90%`；
- perturbed winner source 落在 baseline equivalent-best source set 的比例在每类 perturbation 上均 `>=70%`；
- A/B membership agreement 在每类 perturbation 上均 `>=80%`；
- paired scene-bootstrap 95% lower bound：KEEP-vs-MODIFY `>=80%`，equivalent-best-source retention `>=60%`。

任一条件失败 → `K2-Same FAIL`；样本资格不满足、复现失败或实现无法在 current state 重算 → `INCONCLUSIVE`。

K2-Task 的 primary stability gate：

- baseline same-state true Task-positive 在 perturbation 后仍保持 Task 优于 Same 的比例，每类 perturbation 均 `>=70%`；
- 若 Task 仍为正，perturbed cls/reg source pair 与 baseline equivalent-best pair/set 一致的比例，每类 perturbation 均 `>=60%`；
- scene-bootstrap 95% lower bound：Task-positive retention `>=50%`。

任一条件失败 → `K2-Task FAIL`。由于 n 较小，必须同时报告 64 个基准样本中每类 perturbation 的 stable/flip 原始计数、逐 scene 计数与 bootstrap 区间；不得用 pooled 百分比掩盖某个 scene 的集中翻转。

### 42.4 Stage 2 执行、输出与边界

K1 与 K2 允许并行。代码放新的独立目录（建议 `local_fusion_source_kill_stage2/`），不得修改 Stage 1/P0 原脚本与冻结模型。必须读取并校验 `logs/k/P/` 的完整 P0 输出与 manifest；不得把顶层早期不完整 P0-A 当正式输入。

所有正式工作只在 development/validation；严格拒绝正式 test 及其软链接/realpath 别名。GPU/CPU 任务分别记录 PID、status、源码哈希、输入哈希、config 哈希；支持原子保存与 resume。Shell 必须 POSIX `sh`、LF，禁止 `pipefail`、`[[ ]]` 与 Bash 数组。服务器运行时只设 `ROCR_VISIBLE_DEVICES=<物理卡>`，unset `HIP_VISIBLE_DEVICES`、`CUDA_VISIBLE_DEVICES`。

建议输出至少：`K1_SOURCE_SPECIFICITY.{json,md}`、`K2_LABEL_STABILITY.{json,md}`、`STAGE2_FINAL.{json,md}`、逐 scene/bootstrap 与 null seed 明细，以及完整 protocol/config snapshot。Stage 2 完成后必须停下等待人工判断，禁止自动启动 K3 或新模型。

### 42.5 K3 Data Adequacy Gate（仅预注册，不在 Stage 2 实现/运行）

K3 继续冻结。即使 K1/K2 PASS，也必须先生成真正 joint train labels，再按以下 gate 决定是否有资格训练：

- **K3-Same：**true train joint Same positive targets `>=200`，覆盖 `>=10` train scenes，单 scene 最大占比 `<=40%`。
- **K3-Task：**true train same-state Task-positive targets `>=100`，覆盖 `>=10` train scenes，单 scene 最大占比 `<=40%`。

达不到门槛记 `K3_DATA_INADEQUATE`，不运行对应 K3；不能在失败后以“MLP 容量不够/标签再细分/再加特征”为理由继续扩大。该 data gate 只决定当前数据是否足以做可信学习实验，不等同于科学上证明 H1/H2 为假。
