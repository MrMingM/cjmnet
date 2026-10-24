# 科研代码与实验上下文

更新：2026-10-19。本文保留环境、关键结果、证据路径和研究决定；操作细节查各实验目录 README。沿用历史编号，当前至 §49，重复的 §27 改为 §27A/§27B。

## 当前状态与执行约束

- **V2X-R Clean GSPR训练/评测已完成，当前配置为负结果（§49.5）：**随机初始化、20轮、Clean O选中epoch17；C/F/S各829帧，O AP70=.6611949125627008/.41628316004934424/.27924909081082266，较纯AttFuse下降4.4203/2.5843/.9673 pp。O下9项AP全部下降，O*/F AP70也全下降；weather gap缩小伴随Clean下降，不能称鲁棒性提升。产物归档`logs/v2xr-gspr-clean-train-v2/`，可用哈希及选择/评测身份核对一致。保留纯AttFuse主基线，未自动追加训练或启动诊断；此结论只针对Clean-only、无逐点标签的本次配方，不推断所有GSPR设置无效。
- **最新 V2X-R Clean AttFuse训练及最终天气评测已归档核对（§48.2–48.5）：**8083帧完整train、随机初始化、20轮全部提交，仅按Clean validation F AP70选中epoch20。最终Clean/Fog/Snow各829帧，summary.status=complete、full_validation=true；F AP70=0.7050576560830986/0.2568649945625255/0.266408159808131，天气下降44.8193/43.8649 pp；O AP70=0.7053976613825419/0.4421260304638024/0.2889221792378152。官方GT为6224/3673/5854，固定GT各6224，None帧43/164/114。训练6份产物和最终RESULTS.md/summary.json位于`logs/train/`，文件及身份关联核对见§48.5；权重记录SHA256=`ccf0211d04f732246d53b70faf8c3f6eb494c19963177d9cedf3d2b106cf01d3`，整理报告为`logs/train/FINAL_WEATHER_RESULTS.md`。
- **Source utility 主线已停止（§43）：**Stage 2 的 K1/K2 均为 INCONCLUSIVE，按预注册规则执行 SOURCE UTILITY MAINLINE STOP，K3 与新 source 模型继续冻结。§33/§36 的 Oracle 收益保留为历史诊断，不能据此恢复模型开发。
- **当前天气开发工作（§44）：**已接入论文引用的公开物理天气模拟器，在原始 OPV2V validate 上完成1980帧纯 AttFuse 评测。Fog/Rain/Snow 的 AP70 下降分别为28.705/6.790/61.904 pp；Rain未达到≥10 pp门槛，Snow退化偏重，当前配方尚未通过开发集筛选。
- **V2X-R 早期迁移实验（§45）：**validate 的配对 Clean/Fog/Snow LiDAR 子集已提取并校验，6场景、829协作帧、20车辆包。12帧检查及829帧全量AttFuse迁移评测均已完成；全量Clean/Fog/Snow AP70为0.799440/0.381473/0.244982，Fog/Snow下降41.797/55.446 pp，三条件GT一致、几何检查通过。这些是OPV2V权重的旧协议迁移结果，不能直接与§48域内训练结果比较；官方公开数据未提供Rain版本。
- **V2X-R 官方 Clean 复现 D−1A 已完成（§46）：**官方V2X-R权重、官方源码/配置/GT/AP路径，全validate 829帧；AP30/AP50/AP70=0.85993605/0.82279434/0.66625452，距官方参考绝对差0.6964/0.3006/0.0655 pp。结果接近官方参考，标记REFERENCE DISTANCE WITHIN 2pp; NOT A STATISTICAL PASS。该阶段后续状态见§48，不再沿用当时“未训练”的状态。
- **D−1B/C历史协议审计：**§47保留最初实现记录；后续完成证据及D−1D继承关系见§48和`v2xr_train_attfuse/README.md`。
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
| 原版模拟器生成的 OPV2V validate | `/data/cjm/datasets/opv2v-reference-weather-v1/{clean,fog,rain,snow}/validate`；1980协作帧，每条件10478个车辆点云；生成及AttFuse评测完成，详见§44 |
| 原版模拟器8帧检查数据 | `/data/cjm/datasets/opv2v-reference-weather-smoke-v1/{clean,fog,rain,snow}/validate`；8协作帧、每条件40个车辆点云；评测和独立几何检查完成 |
| 原版模拟器资源 | `/data/cjm/datasets/reference-weather-assets-v1`；Fog积分表、LISA Mie系数、雪粒压缩包和所选64个雪粒文件；资源准备完成 |
| V2X-R 原始下载 | `/data/cjm/datasets/V2X-R/{train,validate,test}`；按场景/车辆存放`.7z`；validate与train已提取，原包保留；test未报告提取 |
| V2X-R 完整 train | 默认根目录`/data/cjm/datasets/v2xr-official-train-v2`，loader目录`train/`；2026-10-18回传全量提取完成，8083协作帧、32有数据场景、92场景-CAV组合、24088单车帧；详见§48 |
| V2X-R 配对 LiDAR validate 子集 | `/data/cjm/datasets/v2xr-lidar-weather-validate-v1`；6场景、829协作帧，Clean/Fog/Snow各3041个PCD及3041份共享YAML；合计12164个提取文件，约12.661 GiB，已完成校验，详见§45 |
| V2X-R 官方 Clean validate 视图 | `/data/cjm/datasets/v2xr-official-validate-v1/validate`；复用Clean/YAML链接，在独立`auxiliary/`补提取官方loader读取的真实雷达/四相机文件；D−1A已完成829帧，详见§46 |
| V2X-R 官方源码/模型/结果 | runtime=`/data/cjm/datasets/v2xr-official-runtime-v1`；model_dir=`/data/cjm/datasets/logs/v2xr-official-attfuse-model-v1`；完整结果=`/data/cjm/datasets/logs/v2xr-official-attfuse-clean-repro-v1`；用户原始权重/YAML位于`/data/cjm/datasets/V2X-R/`，原权重文件名未回传，不推测 |
| 历史 GSPR 逐点监督语料 | `/data/cjm/datasets/gspr_opv2v_weather_{smoke,development,full_v1}`；规模和用途见§7，属于已有派生语料 |
| V2X-DGW 下载目录 | `/data/cjm/datasets/V2X-DGW`；目录存在，内容未进一步核验；用户确认官方仅发布OPV2V-W天气test，未提供本任务需要的天气train/validate，当前不作为训练来源 |
| 其他历史数据目录 | `/data/cjm/datasets/counterroute_m7`、`/data/cjm/datasets/opv2v_synthetic_validation`；用户目录列表确认存在，本轮未核验内容或使用，不据名称推断可用训练规模 |
| 默认冻结前端 | `/data/cjm/datasets/logs/gspr_joint_full_v1_seed20260907_20260907_161155/{config.yaml,net_best_validation.pth}` |

默认前端 checkpoint SHA256：`93598c7f6cde8c8caa54155cde98bbe75ff2921481b24928a2172e2d6c5dd4c1`。后续各实验的 F/Shared 起点还需查各自协议，不可只凭名称判断权重相同。

固定物理天气数据生成日志已报告 Complete，尚无独立全量完整性扫描。每种天气 train 19575 个 PCD、validate 10478 个；元数据链接官方 Clean，mixed PCD 链接单天气 PCD，源目录必须保留。生成器 `opencood/tools/materialize_physics_weather.py`，说明 `opencood/tools/PHYSICS_WEATHER_DATASET.md`，中断可 `RESUME=1`，已有 complete 数据无需重建。它是固定 epoch-0 realization；旧在线天气逐轮重采样，PCD 强度 RGB 量化、加载时点顺序重排，两种协议不能混写。

上述 `opv2v-physics-fixed-v1` 是旧近似天气协议；`opv2v-reference-weather-v1` 是此次接入公开原版模拟器后建立的新协议，二者的结果与训练数据不能混写。V2X-R 子集的 `payload/validate/<scene>/<cav>/` 保存所选原文件，`{clean,fog,snow}/validate/` 是供读取的配对链接视图；payload和链接目标都需保留。`/data/cjm/datasets/logs` 是权重/结果目录，`/data/cjm/datasets/tmp` 是临时目录，均不作为新的原始数据集。

## 4. 存储与操作

大数据、权重、缓存和大型日志放 `/data/cjm/datasets/logs` 等外部数据盘，不写代码仓库。启动前指定物理卡与 opencood解释器；长期任务后台运行并记录 RUN/PID/LOG。单个进程失败后先确认已有缓存与源码契约，再断点恢复，避免覆盖已完成结果。

2026-10-13至15日天气生成期间，其他用户任务填满home所在磁盘，导致Open3D导入时无法创建临时文件，报 `No usable temporary directory found`；清理空间后用相同输出目录及 `RESUME=1` 恢复，最终全量任务完成。临时目录统一放数据盘：`/data/cjm/datasets/tmp/{reference_weather,v2xr_subset,v2xr_attfuse}`，通过TMPDIR指定，不把共享home剩余空间当作长期保证。V2X-R提取使用独立7-Zip入口 `/data/cjm/tools/7zip-26.04/bin/7zz`，提取命令的PATH须包含该bin目录；AttFuse评测本身无需7-Zip。

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

seed20260906 点预训练5轮，epoch5：val loss .059142、macro AUROC .999871。联合训练 epoch1 冻结检测端、epoch2–5 解冻，按最低 Clean val detection loss且 point macro AUROC≥.995 选权重。三次最佳 epoch 4/3/4；val loss .272177±.000409、point macro AUROC .999833±.000043。

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
|---|---:|---:|---:|---:|
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

## 43. Source Utility Kill Protocol：Stage 2 实测与主线停止（2026-10-15）

结果入口为 `logs/k/STAGE2_FINAL.md`、`logs/k/K1_SOURCE_SPECIFICITY.{md,json}`；Stage 2 仍只使用 development/validation，未访问正式 test，K3 全程冻结。最终自动汇总为：`K1_SAME=INCONCLUSIVE`、`K1_TASK=INCONCLUSIVE`、`K2_SAME=INCONCLUSIVE`、`K2_TASK=INCONCLUSIVE`，因此按 §42 预注册停止逻辑，`H1_SOURCE_SPECIFIC=STOP`、`H2_TASK_SPECIFIC=NOT_APPLICABLE`，**SOURCE UTILITY MAINLINE STOP**。

### 43.1 K1：source-specificity 未进入有效 Null 比较

K1 先完成 Source 条件复现，再准备 N1 matched output-space perturbation、N2 random source mixing 与 score-only matched null。正式失败原因不是某个 Null 已经复制了 Source 收益，而是在 Null 候选真正生成/比较之前，逐帧配对门禁发现 Source 与 Null 重放的 `model_input_sha256`、`shared_prediction_sha256` 或 `source_count` 至少一项不一致，程序抛出：`Source/Null frame input, Shared prediction or CAV count mismatch`。

因此 K1 没有产生可解释的 N1/N2/score-only AP、`Null/Source` 比例、`Source−Null` 差值或 paired scene-bootstrap 置信区间；不能写成“K1 科学上证明 random null 能复制 source”。正式结论只能是：**K1-Same / K1-Task = INCONCLUSIVE，source-specificity 未建立。** 该门禁在 §42 已预注册为 INCONCLUSIVE 条件，不能在结果后放宽配对要求以继续模型开发。

### 43.2 K2：baseline A_ONLY 轨迹复现失败，稳定性实验未完成

K2 的样本资格本身通过：Same baseline 为 259 个正例、9 个 validation scenes、单 scene 最大占比 21.24%；Task baseline 为 64 个正例、8 个 scenes、29.69%。但正式 center/yaw/ROI perturbation 之前，程序要求当前 A_ONLY baseline trajectory 与 Stage 1 保存轨迹一致。

在 Fog、frame/sample 1297、target 48 上出现 shortlist 差异：Stage 1 记录为 `single:2、single:2、query:2、query:2、query:0、query:3`，当前重放为 `single:2、single:2、query:2、query:2、query:0、query:0`。发生替换的两个动作均属于 B 类，在 A_ONLY 轨迹中禁止选用；合法 A 类候选与最终选中的 `single:2` 保持一致。因此这次中断**没有证明最终 source 选择已经改变，也没有证明 K2 稳定率低于门槛**。但完整 baseline trajectory 不再满足预注册的严格复现门禁，K2 在正式小扰动稳定性统计前中止。

当前只观察到约 `3×10^-6` 量级连续数值差异，微小浮点变化影响 top-6 边界是合理推测，但原因尚未被证实，不能写成已确定根因。正式结论只能是：**K2-Same / K2-Task = INCONCLUSIVE，标签稳定性未知。**

### 43.3 最终项目结论：source utility 主线停止，但不是科学意义的“已证伪”

§42 在 Stage 2 开跑前已写死：`K1-Same FAIL/INCONCLUSIVE` 或 `K2-Same FAIL/INCONCLUSIVE` 任一出现，就停止整个 source utility 主线，不运行 K3，不再用 MLP/GNN、x/y/size/yaw 标签细分或 joint-action source network 解释失败。本次 K1 与 K2 都触发 INCONCLUSIVE，因此项目层面正式执行 **SOURCE UTILITY MAINLINE STOP**。

这不等于科学上证明 source information 永远无用，也不等于证明 source label 一定不稳定。可以严谨写成：**在当前冻结 development/validation 协议下，严格验证未能建立 source-specificity，也未能完成 label-stability 验证；结合此前多轮可学习性与 replay 失败，继续投入 source utility 的预期收益不足。**

支持停止的累计证据不是单一报错，而是整条链条：§34–§35 的 source/distortion 代理和候选重排没有稳定独立收益；§37 中多个代表性质量/重排序方法即使质量预测较准也未稳定兑现 AP；§38–§39 的 source utility/evidence 学习为 `S1_WEAK / S2_WEAK`、`GEO_EVIDENCE_WEAK / SEM_EVIDENCE_WEAK / REPLAY_FAIL`，Conservative 多次退化为全 KEEP；§41 的 K0′ 进一步显示 beneficial/discrete signal 极稀疏、错误干预风险远高于净 TP 增长；K1-0 只说明 Oracle 收益集中在少量 A 类离散动作，不构成 source-specificity 或可学习性的 PASS；最终 Stage 2 又没有通过严格可重复性/配对门禁。

因此从 2026-10-15 起，**不再把“预测每个 candidate/source 的 utility 或 cls/reg 分源选择”作为论文主线继续开发**。K3 tiny MLP、source GNN、按 x/y/size/yaw 细分 source utility、joint-action source network 均保持冻结。后续若开启新方向，应把“Oracle opportunity ≠ learnable opportunity”“有益干预极稀少、错误干预代价高”作为设计约束，而不是直接重启 source ranking。

### 43.4 历史表述校正

§41.4 中旧文字 `K1-0 = 机制分解 PASS` 已被 §42 与本节正式覆盖，正确口径为 **`K1-0 = A-dominant descriptive decomposition`，无 PASS/FAIL 身份**。另外 §41.3 表内 Rain 的 `Full Task−Same` 旧写为 `+0.7552 pp`，原始 K1-0 报告的正确值是 **`+0.7534 pp`**；`+0.7552 pp` 对应 AOnly Task−Same。历史表本身未重写时，以本节校正说明和 `logs/k/K1_0_ORACLE_DECOMPOSITION.md` 为准。

## 44. 公开原版物理天气模拟器：OPV2V validate 已完成（2026-10-15）

本轮优先解决开发天气与正式测试天气退化量级不一致的问题。用户此前指出：旧90帧在线天气与旧固定物理天气的Fog/Rain影响过弱，无法可靠指导正式恶劣天气研究。新工作放独立目录 `reference_weather/`，接入 V2X-DGW 论文引用的 [LiDAR_fog_sim](https://github.com/MartinHahner/LiDAR_fog_sim)、[LISA](https://github.com/velatkilic/LISA)、[LiDAR_snow_sim](https://github.com/SysCV/LiDAR_snow_sim)。保留上游许可证和版本/适配记录，运行依赖源码哈希，不依赖Git。

### 44.1 数据和固定协议

- Clean源：`/data/scd/datasets/opv2v_official_data_dumping/validate`，1980协作帧、10478个车辆点云。
- 全量输出：`/data/cjm/datasets/opv2v-reference-weather-v1`；条件目录为 `{clean,fog,rain,snow}/validate/`，逐PCD记录在 `records/<weather>/validate/`，完整协议及汇总在 `manifest.json`。
- 资源：`/data/cjm/datasets/reference-weather-assets-v1`；已经下载、提取并记录哈希，后续复用。
- Fog：alpha=0.06、gamma=1e-6、noise=10，保留原版ParameterSet构造行为和积分表。
- Rain：LISA `python_old`，33 mm/h、905nm、strongest回波模式，原版Mie表。
- Snow：1.5 mm/h、下落速度1m/s、Gunn分布，公开80m半径雪粒场。用户明确同意 `opv2v_elevation_no_focal_v1`：按仰角重建通道、关闭真实HDL-64专属焦距/通道强度修正，路面平面改用OPV2V下包络拟合，保留原版遮挡/回波计算。该适配假设必须在报告中披露，不能宣称与官方OPV2V-W完全一致。
- seed=20261013；默认4个生成进程、每个Snow进程2个通道线程。已生成PCD/记录匹配时，`RESUME=1`会核对后复用；不能为续跑改物理参数、源码快照或删已完成输出。
- 本轮仅生成validate并评测，**尚未生成这套原版模拟器的train，未做天气训练或GSPR重训**。正式test不读取、不参与调参。

评测使用纯AttFuse：`attfuse_code.point_pillar_intermediate.PointPillarIntermediate`，配置 `/data/cjm/datasets/logs/attfuse_opv2v_20260903_200923/config.yaml`，指定权重 `/data/cjm/datasets/logs/attfuse_opv2v_20260903_200923/net_epoch13.pth`，严格加载全部参数。该权重SHA256为 `06e953cc15391e5b7569e825d33112f79b1e8c15a70b8914a489bb437cb81aba`。

### 44.2 全量结果与当前判断

主指标为项目现有BEV多边形IoU、`global_sort=False`，`global_sort=True`另存，不能混用。下表保留用户回传日志的显示精度：

| 条件 | 帧数 | AP30 | AP50 | AP70 | Clean−天气 AP70 / pp |
|---|---:|---:|---:|---:|---:|
| Clean | 1980 | 0.872966 | 0.848902 | 0.768961 | — |
| Fog | 1980 | 0.557366 | 0.548712 | 0.481908 | 28.705 |
| Rain | 1980 | 0.823157 | 0.791118 | 0.701059 | 6.790 |
| Snow | 1980 | 0.216328 | 0.207663 | 0.149920 | 61.904 |

**生成及全量评测已完成；开发集筛选未通过。**完整validate条件为True，但Fog/Rain均≥10 pp的最低筛选门槛为False，因为Rain仅下降6.790 pp。Fog已出现明显退化，Snow退化很重，仍需检查删除数、天气回波替换数及强度拟合；不能仅凭低AP认定物理模拟正确，也不能直接认定适配错误。此前请求的全量点云统计尚未收到回传。OPV2V-W作者完整天气配方仍未知；AP降幅相近也不能证明天气分布相同。

证据和入口：

- 全量日志：`/data/cjm/datasets/logs/reference_weather_full_v1.log`。
- 全量报告：`/data/cjm/datasets/opv2v-reference-weather-v1/evaluation/RESULTS.md`。
- 机器可读结果：同目录 `summary.json`、`protocol.json`、`{clean,fog,rain,snow}/result.json`。
- 代码/说明：`reference_weather/{run_server.sh,config.yaml,README.md,UPSTREAM.json,source_snapshot.json}`。

### 44.3 小规模检查和运行故障记录

8帧检查输出在 `/data/cjm/datasets/opv2v-reference-weather-smoke-v1`，每条件40个车辆点云。Snow阶段出现SciPy `linregress` 的sqrt RuntimeWarning，随后仍完成生成和四条件评测。独立检查脚本 `audit_reference_weather.py` / `run_reference_weather_audit.sh` 复用该数据，在NMS/AP中检查多边形和IoU；共核查13586次IoU，四条件AP与原结果差值均为0。

检查报告：`/data/cjm/datasets/opv2v-reference-weather-smoke-v1/geometry_audit_20261013T040738_858391Z/audit.json`；日志 `/data/cjm/datasets/logs/reference_weather_geometry_audit_v1.log`。这份独立几何审计覆盖8帧，不能扩大表述为原版模拟器1980帧已全部经过同一独立审计。

全量期间还发生home磁盘满导致的临时目录异常，按§4恢复后完成。恢复日志再次出现Fog/Rain计数时，匹配的既有记录用于校验/复用；不能把计数重现直接理解为重新模拟全部点云。

## 45. V2X-R 配对天气子集与 AttFuse 迁移评测（2026-10-15进度）

目的：利用作者已生成的天气点云，增加另一套可训练/验证的数据来源，检查Fog/Snow退化，并减少对自行适配Snow的单一依赖。这项天气数据检查不恢复§43已经停止的source utility主线。

### 45.1 原始数据、选择范围与提取结果

原始下载目录 `/data/cjm/datasets/V2X-R`，用户确认包含 `train/`、`validate/`、`test/`；各场景按车辆保存如 `-1.7z`、`200.7z` 的压缩包及 `data_protocol.yaml`。本轮只列出并提取validate，原始train/test压缩包保留，**尚未提取训练子集，训练规模未定**。官方说明提供Clean/Fog/Snow三种LiDAR版本，没有提供预生成Rain；若后续用LISA自行添加Rain，必须独立标注来源和开发参数，不能称为作者官方Rain。

提取代码 `v2xr_subset/`，在服务器通过7-Zip合成选择检查后完成plan及20个车辆包的提取/哈希核验：

| 项目 | 实际结果 |
|---|---|
| 输出根目录 | `/data/cjm/datasets/v2xr-lidar-weather-validate-v1` |
| 场景/协作帧/车辆包 | 6场景、829协作帧、20个车辆包；保留每帧所有协作车辆 |
| 所选天气版本 | `fog_0.060`、`snow_2.5_2.0` |
| 文件数 | Clean/Fog/Snow各3041个PCD，另有3041份共享逐帧YAML，共12164个提取文件 |
| 所选内容大小 | plan统计12.661 GiB；对应压缩包全部内容为20.328 GiB，未选中相机/雷达不落盘 |
| 原文件存放 | `payload/validate/<scene>/<cav>/` |
| 三条件读取视图 | `{clean,fog,snow}/validate/<scene>/<cav>/<timestamp>.{pcd,yaml}`，通过链接引用payload |
| 清单/进度/格式样例 | `plan.json`、`inventory.json`、`manifest.json`、`samples.json`、`records/validate/` |

plan时数据盘剩余899.392 GiB仅为当时观测，后续操作重新检查空间。提取日志为 `/data/cjm/datasets/logs/v2xr_subset_validate_extract_v1.log`，20/20 Verified并输出Complete；规划日志 `/data/cjm/datasets/logs/v2xr_subset_validate_plan_v1.log`。

实际样例确认：PCD三条件均为 `FIELDS x y z rgb`，Clean为ASCII、Fog/Snow为binary；原读取器使用RGB红色通道表示0–1强度。第一份样例Clean/Fog均58266点，Snow为51361点；这是单个样例，不能外推全量点数比，也不能仅凭点数推断天气影响。YAML顶层包含 `lidar_pose`、`ego_speed`、`true_ego_pos`、`RSU`、`vehicles`；车辆包含 `location`、`angle`、`center`、`extent`，接口与现有世界坐标投影一致，extent为半尺寸。

### 45.2 评测协议和已完成的12帧检查

独立入口 `v2xr_attfuse/` 直接读取已提取的三条件视图，使用§44同一个OPV2V纯AttFuse epoch13权重，严格加载；没有进行V2X-R训练、微调或GSPR重训。

沿用本项目现有OpenCOOD处理顺序、最多5车、70m通信范围及GT范围 `[-140,-40,-3,140,40,1]`。无相机FOV裁剪、无雷达输入、不根据天气点云可见性改变GT；在线天气增强和定位/时延噪声关闭。逐帧核对Clean/Fog/Snow真值框数组完全一致，检查模型输出、参与NMS/AP的框/多边形/IoU有效性；RuntimeWarning按帧捕获汇总。**本协议是OPV2V模型的跨数据集迁移基线，不能直接等同作者官方V2X-R分数，也不证明OPV2V-W分布已复现。**

服务器回传显示：8项合成检查通过，12164个提取文件/字段校验通过；六个场景各取两帧，共12帧的三条件评测已全部完成。

| 条件 | 帧数 | AP30 | AP50 | AP70 | Clean−天气 AP70 / pp |
|---|---:|---:|---:|---:|---:|
| Clean | 12 | 0.912834 | 0.912834 | 0.776281 | — |
| Fog | 12 | 0.472981 | 0.463278 | 0.364955 | 41.133 |
| Snow | 12 | 0.343720 | 0.343720 | 0.238727 | 53.755 |

三条件真值框相同，几何检查通过；Clean/Fog/Snow分别检查12670/5939/4079次IoU。每条件记录48条按帧过滤后的警告，不等于未经去重的全部发生次数；Open3D关于CUDA不可用的ImportWarning未阻止当前HIP模型评测完成。小样本覆盖各场景，但仍不能替代完整829帧结论。

12帧结果目录 `/data/cjm/datasets/logs/v2xr-attfuse-transfer-smoke-v1`，报告 `RESULTS.md`，详细输出为 `verification.json`、`protocol.json`、`{clean,fog,snow}/result.json`、`summary.json`；日志 `/data/cjm/datasets/logs/v2xr_attfuse_transfer_smoke_v1.log`。

### 45.3 829帧全量迁移评测结果（2026-10-15完成）

用户已回传全量报告，`完整已准备 validate=True`，Clean/Fog/Snow均完成829帧；三条件GT逐帧一致，所有参与NMS/AP的多边形与IoU通过有效性检查。全量使用 `FRAMES_PER_SCENE=0`，权重仍为§44的同一个OPV2V epoch13，未做V2X-R训练。

| 条件 | 帧数 | AP30 | AP50 | AP70 | Clean−天气 AP70 / pp |
|---|---:|---:|---:|---:|---:|
| Clean | 829 | 0.879262 | 0.867048 | 0.799440 | — |
| Fog | 829 | 0.431531 | 0.427571 | 0.381473 | 41.797 |
| Snow | 829 | 0.344714 | 0.332199 | 0.244982 | 55.446 |

数值按报告显示精度保留。完整结果根目录 `/data/cjm/datasets/logs/v2xr-attfuse-transfer-v1`，报告 `RESULTS.md`，机器可读文件 `summary.json`、`protocol.json`、`verification.json`、`{clean,fog,snow}/result.json`；日志 `/data/cjm/datasets/logs/v2xr_attfuse_transfer_full_v1.log`。代码/完整操作说明为 `v2xr_attfuse/{run_server.sh,config.yaml,README.md,UPSTREAM.json,source_snapshot.json}`。

结论：12帧中观察到的明显天气退化在全量中保持，当前模型在作者预生成Fog/Snow上分别下降41.797/55.446 pp。该结果确立了本项目读取协议下的V2X-R迁移基线；训练效果仍未知。作者预生成Snow同样造成严重退化，说明严重下降也会出现在这套现成天气数据上；§44的原版模拟器Snow适配仍需独立诊断。两套数据的场景、天气参数和样本不同，不能把AP差额归因于某项适配，也不能直接与作者官方V2X-R分数等同。

### 45.4 当前待办与训练边界

2026-10-15曾建议先查看已写入 `verification.json` 的三条件点数统计（尚未回传），再考虑train提取规模与Clean/混合天气训练对照。随后用户将本轮范围明确限定为§46的D−1A官方Clean复现；该阶段现已完成并停在此处，不自动执行后续天气分析或训练。此前训练对照仅为建议，**尚未提取训练子集或启动训练，也不恢复source utility主线**。

Rain仍需要单独的数据/物理模拟开发方案；V2X-R的Fog/Snow结果不能视为Rain门槛已经解决。正式OPV2V-W/V2X-R test继续封存，不用于选参数或模型。本地仅完成代码、语法/静态/打包检查；本节的实际数据和模型结果均来自用户回传的服务器日志。

## 46. D−1A：V2X-R 官方 AttFuse Clean validation（2026-10-16完成）

### 46.1 范围、代码与协议

用户明确只做D−1A，使用放在`/data/cjm/datasets/V2X-R/`的官方LiDAR AttFuse checkpoint与V2XR_attfuse.yaml。
独立代码`v2xr_official_repro/`，未修改§44/§45工具、原AttFuse、GSPR或既有实验；无branch/worktree运行前提。
官方仓库`https://github.com/ylwhxht/V2X-R`，固定commit=`f3ebbad95561444a53a17358735b0f0dc4ba89f7`。
完整源码快照328个文件与归档逐字节一致；源码SHA256见`UPSTREAM.json`、`source_snapshot.json`。
本地仅做AST、导入依赖静态扫描、LF/sh语法与哈希检查；实际编译、数据读取和推理均在服务器。

官方推理源码复制至数据盘runtime，wrapper直接执行原版`inference.main()`；实际导入opencood模块路径与源码哈希校验，拒绝混入本项目OpenCOOD实现。
官方注册名`IntermediateFusionDatasetV2XR`实际对应`opencood.data_utils.datasets.opv2v.intermediate_fusion_dataset.IntermediateFusionDataset`；
模型为官方`opencood.models.point_pillar_intermediate.PointPillarIntermediate`，使用processed_lidar。
FOV=True、proj_first=True；按ego camera0视野裁点并根据逐车框内至少1点筛选GT，再按目标ID去重。
score threshold=.20（严格大于）、NMS=.15、COM_RANGE=70m、voxel=[.4,.4,4]。
实际GT使用anchor范围`[-140.8,-40,-3,140.8,40,1]`，不能误写成声明但该路径未使用的GT_RANGE_OPV2V常量±140。
官方YAML未设max_cav，BaseDataset默认7；未为迁就项目改为5，验证数据每场景2–5个agent。

实际评测模块为`eval_utils_opv2v.py`（官方inference条件中非空字符串恒真），不是另一份`eval_utils.py`。
AP为BEV多边形IoU、逐帧分数排序贪心匹配、按帧累计、global_sort=False、VOC2010积分。
官方预测返回None时跳过该帧AP累计的行为原样保留；所有帧GT总数与AP分母分别记录。
本阶段只使用Clean，eval_sim/train_sim均False；没有混入§45的fixed-GT评测。

### 46.2 数据、权重与环境准备

原始validate为压缩包，旧paired目录缺少官方loader即使纯LiDAR也要读取的真实radar.pcd与四张图片。
新视图`/data/cjm/datasets/v2xr-official-validate-v1/validate/<scene>/<agent>/`链接Clean/YAML，
缺失依赖从原validate包补提取至同根`auxiliary/`；原始数据与paired目录保留，不重复复制Clean点云。
入口要求核对6场景、20个scene-agent、829协作帧及3041单agent帧，逐项比对官方loader实际ego、CAV与时间戳顺序。
完整manifest及哈希在新视图根目录，实际JSON尚未回传本地。

独立model_dir=`/data/cjm/datasets/logs/v2xr-official-attfuse-model-v1`。
用户本地`C:\Users\25313\Downloads\V2XR_attfuse.yaml`与固定官方模板逐字节一致，SHA256=
`7e1ce936d779138b5b1ca6462ebc79d9e604920d9a29b7ef1c3a64e50d96e346`。
运行副本只调整root_dir/validate_dir；原YAML和权重不改写。按官方README保留0.pth链接，
另建官方loader实际打开的net_epoch0.pth链接；两个别名指向同一原件。
官方loader原本允许部分加载；wrapper先验证完整key/shape/dtype吻合，调用官方loader后逐tensor核对加载值，禁止缺参/丢参/随机补参。
本次官方权重SHA256=`4e92bff533cba4d8e2020f4176bd9f684b5a2337d59f853535f612edd6aa3e94`，与12帧smoke相同。

构建故障依次为缺gflags、缺glog与系统GCC<9；在独立`/data/cjm/tools/v2xr-build-deps-v1`补齐后，服务器预检确认GCC11.4、头文件检查通过，Torch2.5.1/HIP6.3.25405。
全量pcdet扩展构建随后在PointNet2的旧THC/转换后THH.h处失败。静态依赖检查覆盖63个模块，纯AttFuse链只需box_overlaps和roiaware_pool3d；
build wrapper保留原版扩展对象、源码与编译参数，仅排除未使用的PointNet2/pcdet iou3d_nms。
官方GT的ROI算子保留，未补假头文件、未换近似实现；选择与HIP转换源码哈希记录在runtime/native_build_selection.json。
build-v4日志回传ROI扩展完成复制，之后smoke及全量推理均已完成。
还按官方INSTALL指定klintan/pypcd来源提供固定commit `796e545df2ee5e7836df6fe2f274aae3428d1ac2`与python-lzf=.2.6安装指令，后续实测已通过导入阶段。

### 46.3 Smoke与全量结果

12帧smoke：AP30=.99565217、AP50=.97965514、AP70=.70466531；GT230、预测229、官方跳帧0。
smoke仅检查运行路径，不用于官方全validate对齐判定。

用户已回传全量报告，本地证据`logs/v2x-r/RESULTS.md`（当前该目录仅此文件）：

| 指标 | 本次AP | 官方参考 | 绝对差值 / pp |
|---|---:|---:|---:|
| AP30 | 0.85993605 | 0.8669 | 0.6964 |
| AP50 | 0.82279434 | 0.8258 | 0.3006 |
| AP70 | 0.66625452 | 0.6656 | 0.0655 |

完整验证集829/829帧；全部已读帧GT=6224、预测=7034；官方AP实际累计GT=6223，预测None跳过43帧。
因此这43个跳过帧合计包含1个GT，不能把43帧解释成43个漏检目标；本实验没有改写官方跳帧口径。
官方Clean主指标接近参考值，报告为`REFERENCE DISTANCE WITHIN 2pp; NOT A STATISTICAL PASS`，
没有触发AP70绝对偏差>2pp的NOT YET ALIGNED标记。只称数值接近，不称统计意义PASS或完全逐位复现。
旧§45的OPV2V权重+fixed-GT结果与本节权重/GT/处理协议不同，不直接比较AP以推断模型优劣或GT筛选的因果贡献。

### 46.4 结果路径与当前停止位置

- 全量服务器结果：`/data/cjm/datasets/logs/v2xr-official-attfuse-clean-repro-v1`。
- 全量日志：`/data/cjm/datasets/logs/v2xr_official_clean_full_v1.log`。
- Smoke结果/日志：`/data/cjm/datasets/logs/v2xr-official-attfuse-clean-smoke-v1`、`/data/cjm/datasets/logs/v2xr_official_clean_smoke_v1.log`。
- 构建日志：`/data/cjm/datasets/logs/v2xr_official_build_v4.log`；预检/编译记录根`/data/cjm/datasets/logs/v2xr-official-build-v4`。
- 应保留的服务器证据：`summary.json`、`protocol.json`、`data_verification.json`、`frames.jsonl`、`official_result_stat.json`、`official_eval.yaml`、`contract.json`与`preflight.json`。

当前本地只实际读取了用户回传RESULTS.md；没有独立读取完整机器可读输出、实际运行YAML/hash清单或逐帧记录，后续归档应补齐这些文件。
本轮D−1A全量Clean验证已完成，停在这里；未实现/运行官方Fog/Snow分析、D−1B/D−1C、D0/D1/D2或训练，正式test未访问。
后续维护实验状态写入本文件或结果目录，避免仅为更新进度改写独立实验包而改变source_snapshot/contract身份。


## 47. D−1B：官方天气 GT 审计（代码已实现，服务器未运行）

用户授权在D−1A已验证前提下继续审计；新增入口 `v2xr_official_repro/gt_audit/run_server.sh`，说明见同目录README。
继承固定官方源码/runtime、native ROI、model/config处理和D−1A数据文件；天气使用官方eval_sim/sim_weather选择。
输出目录下仅建立额外链接视图，不改D−1A视图、paired数据、冻结源码或历史结果。
逐帧直接调用原版dataset/collate/generate_gt_bbx，无detector，无prediction=None跳帧。
记录原YAML IDs、逐CAV候选框、native点数/零点移除、最终evaluation IDs/中心/尺寸/yaw/角点。
按ID做三条件两两比较，提供Clean→Fog/Snow retained/removed/added、数量/几何变化与距离/点数/CAV证据分布。
官方空体素随机临时点保留并单独标记；逐帧重设seed以支持恢复，实际Clean每帧GT数和CAV数必须与D−1A一致。
前12帧三条件还与无观察器原版输出逐角点核对；全量Clean GT必须6224。真实数据与这些检查均仅在服务器执行。
默认输出 `/data/cjm/datasets/logs/v2xr-official-gt-audit-v1`，支持按完整三条件帧原子保存并RESUME=1恢复。
源码/输入/native扩展/环境/运行配置身份由SHA256与契约记录，不依赖服务器Git。
D−1A既有包文件逐字节保留；顶层source_snapshot更新纳入新文件，旧manifest另存gt_audit/d1a_source_snapshot.json。
本地已通过AST/LF/哈希检查；Windows的sh.exe未能启动，POSIX sh语法由服务器入口sh -n自检。
尚无829帧Fog/Snow统计结果，不能宣称官方GT一致或不一致。

## 48. V2X-R 完整训练数据提取完成（2026-10-18）

用户回传服务器成功日志：`Train extraction complete: ... FULL TRAIN`。
已核对`v2xr_train_attfuse/extract.py`：只有逐包提取/CRC/哈希/内容检查完成、核对输出文件集合与场景归属并原子写出`manifest.json`后，才打印此行。
本次未在本地运行数据处理、Torch/OpenCOOD或科研实验；以下计数依据用户日志，服务器manifest尚未回传本地独立核验。

| 项目 | 完成计数 |
|---|---:|
| 原始场景数 / 有数据场景数 / 空场景数 | 33 / 32 / 1 |
| 原始CAV压缩包 / 空目录占位包 | 97 / 5 |
| 有数据的场景-CAV组合 | 92 |
| 协作帧（scene + timestamp） | 8083 |
| 单车帧 / Clean PCD / YAML | 各24088 |
| Fog PCD / Snow PCD / Radar PCD | 各24088 |
| 全部PCD（Clean + Fog + Snow + Radar） | 96352 |
| PNG（每单车帧4张） | 96352 |

协作帧包含同一时刻多个CAV，因此24088个单车帧不等于24088个协作帧；全部PCD数也不能作为训练帧数。
`TRAIN_DATA_ACCEPTANCE.json`记录此前用户已接受实际8083与公开8084相差1帧，固定原inventory身份和四份证据哈希；不认定官方数字错误或已确定缺帧身份，不补造/重复帧。
空目录包和空场景通过内容确认、保留证据及CRC结果，不建立官方loader中的空目录；有内容但损坏/缺文件的包不适用此排除规则。

默认数据根目录`/data/cjm/datasets/v2xr-official-train-v2`：`train/`、`manifest.json`、`plan.json`、`data_acceptance.json`、`records/`。
用户此条日志未重复提供TRAIN_ROOT；此路径依据当前launcher/README默认值，若服务器使用自定义TRAIN_ROOT则沿用其实际值。
提取日志默认`/data/cjm/datasets/logs/v2xr_train_extract_full_v2.log`。

现有`v2xr_train_attfuse/README.md`记录D−1B/C已完成（本地证据在`logs/D/D1/no1/`及`logs/D/D1/`），并已实现D−1D；§47“未运行”描述早期状态，不能作为最新授权或进度结论。
该训练代码按随机初始化的官方Clean配方训练20轮，仅按Clean validation F AP70选择epoch，最后显式评测天气；本次日志仅确认数据提取，未证明已开始或完成训练。
训练只读取Clean `<timestamp>.pcd`；提取出来的Fog/Snow不进入训练或epoch选择。
下一步按该README先做服务器`MODE=self-test`（若尚未完成），再做独立输出的`MODE=train SMOKE=1`；成功后才执行正式20轮。
正式训练默认输出`/data/cjm/datasets/logs/v2xr-clean-attfuse-train-v1`，smoke输出`/data/cjm/datasets/logs/v2xr-clean-attfuse-train-smoke-v1`。
提取已含完整性检查，无新增异常时不要求为继续训练重复全量解压或独立verify；训练入口仍会检查manifest与输入哈希。

### 48.1 训练 smoke 完成，等待正式训练

用户回传上述`self-test && train SMOKE=1`后台命令正常完成，训练日志包含`Training smoke completed. Never use this directory for full model selection.`；由于使用`&&`，训练能启动也说明前面的self-test返回成功，具体自检明细未单独回传。
随机初始化、lr=.002，执行2个训练batch，loss=109.9693/109.4479；2个验证batch平均loss=181.140854。
12帧Clean AP的F/O/O*一致：AP30=.003195311602996658、AP50=0、AP70=0；这是运行链检查，不用于判断20轮训练性能，也不用于选模。
PyTorch的`scheduler.step(epoch)`弃用警告保留；该调用来自已核对的官方训练行为，未为消除警告修改调度器。
smoke结果在`/data/cjm/datasets/logs/v2xr-clean-attfuse-train-smoke-v1`，日志`/data/cjm/datasets/logs/v2xr_clean_train_smoke_v1.log`；不从smoke权重恢复正式训练。
下一步使用独立`/data/cjm/datasets/logs/v2xr-clean-attfuse-train-v1`、`MODE=train SMOKE=0 RESUME=0`从随机初始化训练20轮，每轮完整829帧Clean AP，结束后按预定Clean F AP70选权重；暂不启动最终天气评测。
正式训练若中断，以相同源码、GPU、数据/输出目录和`SMOKE=0 RESUME=1`恢复最近完整提交的epoch。

### 48.2 正式20轮训练完成，Clean选中epoch20（2026-10-18回传）

用户提供服务器训练末尾日志：`D-1D EPOCH COMMITTED 20/20`、`Training Finished`、`Training complete. Clean-only selection locked: 20`。
最后一轮官方日志从0编号为epoch19，对应保存/选择轮次epoch20；两者是同一轮。
最后一轮Clean validation loss=0.414521（日志舍入值）；不将其作为选模指标。
按当前代码`finish_training`/`selection`逻辑，完成行在20个epoch提交记录和Clean全829帧结果检查、选择锁定及training_summary写入之后打印。
本次仅核对本地代码并记录用户日志，尚未独立读取服务器训练JSON、逐轮记录或checkpoint文件，不补造权重哈希。

| 选中epoch20的Clean协议 | AP30 | AP50 | AP70 |
|---|---:|---:|---:|
| F（固定官方Clean GT，None帧计入） | 0.8761284122350801 | 0.846663119111817 | 0.7050576560830986 |
| O*（官方Clean GT，None帧计入） | 0.8761284122350801 | 0.846663119111817 | 0.7050576560830986 |
| O（官方GT，None整帧跳过） | 0.8765509142824365 | 0.8470714118874587 | 0.7053976613825419 |

F/O*一致，符合Clean目标集一致的协议设计。O的AP70比F高约0.0340 pp；具体None帧及GT计数等待该轮summary，不能由AP差反推数量。
模型从随机初始化训练，train仅Clean；固定seed20261017、20轮中Clean F AP70最大且并列取最早epoch的规则保持不变，日志报告选中20。
这是单次种子的Clean validation选模结果，不等于正式test分数，也尚不能说明训练后天气鲁棒性改善。
旧§45迁移结果使用不同GT/FOV/max_cav等设置，不直接相减解释域内训练收益。

服务器训练根目录：`/data/cjm/datasets/logs/v2xr-clean-attfuse-train-v1`。
选中权重应为同目录`net_epoch20.pth`，确切路径与SHA256以`selected.json`为准；原始权重不需为本次文档维护复制到本地。
训练日志：`/data/cjm/datasets/logs/v2xr_clean_train_full_v1.log`。
待回传归档：`selected.json`、`training_summary.json`、`history.json`、`protocol.json`、`config.yaml`、`selection_policy.json`，用于核对权重身份、20轮曲线、配置及完成状态。
官方末尾自动inference命令已被拦截，日志`Official automatic inference replaced ...`确认未由该调用自动运行推理。
本条日志没有最终`final_evaluation/summary.json`，选中权重的最终Clean/Fog/Snow O/O*/F评测尚未报告完成；不填入天气结果或自动启动后续实验。

### 48.3 训练机器可读文件已归档，最终天气评测需独立启动

用户已将6份训练产物放到本地`D:\Collaborative perception\mymodule\cjmnet\logs\train`：`selected.json`、`training_summary.json`、`history.json`、`protocol.json`、`config.yaml`、`selection_policy.json`。
本次读取并核对：training_summary.status=complete、epochs=20、selected_epoch=20、initialization=random、weather_used_for_selection=false；history含20个epoch，选中记录确为候选中最大Clean F AP70。
`config.yaml`、`history.json`、`protocol.json`、`selected.json`、`selection_policy.json`的本地SHA256全部与training_summary.output_hashes相符。
这是回传文件之间的身份一致性检查；未读取实际服务器checkpoint、20份逐轮AP summary或contract.json，不能声称已独立复查整条训练链。
selected.checkpoint=`/data/cjm/datasets/logs/v2xr-clean-attfuse-train-v1/net_epoch20.pth`，记录SHA256=`ccf0211d04f732246d53b70faf8c3f6eb494c19963177d9cedf3d2b106cf01d3`。
epoch20精确validation_loss=0.4145206095813547、train_loss=0.21819189440200826、train_updates=1010；Clean AP与§48.2一致。

代码确认：每轮调用`clean-ap`只评测Clean，训练结束只锁定选择/写训练完成记录；不会自动调用`final()`。
因此训练完成日志不代表Fog/Snow评测已运行；如用户另行执行过MODE=final，其结果可能独立存在于服务器，当前回传6份训练文件不能证明该独立任务是否运行或完成。
显式`MODE=final`会固定选中epoch20，评测Clean/Fog/Snow各829帧，输出O/O*/F三协议；不重新训练或重新按天气选模。
全量结果目录：`/data/cjm/datasets/logs/v2xr-clean-attfuse-train-v1/final_evaluation/`，主报告`RESULTS.md`，完成状态`summary.json`；部分12帧检查目录为`final_evaluation_smoke12`，不能混当全量。
最终评测日志：`/data/cjm/datasets/logs/v2xr_clean_selected_validation_v1.log`。
下一步用户在服务器检查该结果目录；未启动时按README后台运行MODE=final，已中断时保持相同参数加RESUME=1。

### 48.4 选中epoch20最终Clean/Fog/Snow评测结果已回传

用户确认评测结束，并提供三条件O/O*/F的完整AP30/50/70 replay值。
日志前缀“D-1C replay”来自D−1D复用的冻结D−1C评测器，不应误记为旧作者权重或OPV2V迁移实验。
本次只读取文档/代码并整理用户日志、计算差值，没有运行Torch/OpenCOOD或真实数据实验。

模型为§48.3已锁定的V2X-R Clean随机初始化训练epoch20；checkpoint记录SHA256=`ccf0211d04f732246d53b70faf8c3f6eb494c19963177d9cedf3d2b106cf01d3`。
正式评测命令目标为同一validate的829配对协作帧×Clean/Fog/Snow；实际frame_count/full_validation、最终checkpoint/GT/prediction身份及None计数仍待服务器final_evaluation/summary.json核对。
当前证据足以记录用户确认的完成与AP，不声称已独立验证该最终目录全部完成链。

O使用当前天气官方GT、None整帧跳过；O*使用相同GT且None按零预测计入；F使用同帧官方Clean选择的固定GT且None按零预测计入。
三协议共用同一预测，BEV IoU/VOC2010/global_sort=False不变；F仍有官方Clean的FOV/范围/目标选择条件，不包含原始YAML所有框。
F作为天气鲁棒性主结果，O用于官方协议比较，O*用于区分None跳帧影响。

| 条件 | 协议 | AP30 | AP50 | AP70 |
|---|---|---:|---:|---:|
| clean | O | 0.8765509142824365 | 0.8470714118874587 | 0.7053976613825419 |
| clean | O* | 0.8761284122350801 | 0.846663119111817 | 0.7050576560830986 |
| clean | F | 0.8761284122350801 | 0.846663119111817 | 0.7050576560830986 |
| fog | O | 0.5635459741481517 | 0.5507777853843197 | 0.4421260304638024 |
| fog | O* | 0.55480050163889 | 0.5422304579226098 | 0.4352648315157016 |
| fog | F | 0.3274071726413508 | 0.319989150377532 | 0.2568649945625255 |
| snow | O | 0.46624621034876207 | 0.4352340884702022 | 0.2889221792378152 |
| snow | O* | 0.45708694929817917 | 0.4266840508593218 | 0.28324639334571894 |
| snow | F | 0.4299143639446658 | 0.4013188357536122 | 0.266408159808131 |

| 协议 | Clean AP70 | Fog AP70 | Snow AP70 | Clean−Fog / pp | Clean−Snow / pp |
|---|---:|---:|---:|---:|---:|
| O | 0.70539766 | 0.44212603 | 0.28892218 | 26.3272 | 41.6475 |
| O* | 0.70505766 | 0.43526483 | 0.28324639 | 26.9793 | 42.1811 |
| F | 0.70505766 | 0.25686499 | 0.26640816 | 44.8193 | 43.8649 |

| 天气 | None跳帧影响 O−O* AP70 / pp | 天气GT集合影响 O*−F AP70 / pp | F比O多显示的天气下降 / pp |
|---|---:|---:|---:|
| fog | 0.6861 | 17.8400 | 18.4921 |
| snow | 0.5676 | 1.6838 | 2.2174 |

天气下降差值为(CleanF−WeatherF)−(CleanO−WeatherO)，需计入Clean自身O−F约0.0340 pp；不能直接把它等同于天气O−F。
本次Clean域内训练后，固定同一批目标的Fog/Snow AP70仍较Clean下降44.8193/43.8649 pp。
Fog中，控制预测与None处理后仅天气GT集合变化的AP70差为17.8400 pp，Snow为1.6838 pp；这些是给定模型预测下的协议效应，不等于模型可获得的提升或整个论文高分的唯一原因。
单种子validation结果不代表test或统计显著性；未取得同协议OPV2V权重对照前，不用旧§45分数直接量化域内训练收益/域偏移贡献。
不从AP差反推出GT删除数、None帧数、恢复TP/新增FP数量。本数据没有Rain条件，本轮未训练天气模型、未评测正式test或启动D0/D1/D2。

服务器原始最终输出：`/data/cjm/datasets/logs/v2xr-clean-attfuse-train-v1/final_evaluation/{RESULTS.md,summary.json,...}`。
本地整理报告：`D:\Collaborative perception\mymodule\cjmnet\logs\train\FINAL_WEATHER_RESULTS.md`，标明来自用户日志，不覆盖/冒充服务器原始报告。
下一步只需回传最终RESULTS.md、summary.json补齐归档与计数；本次未自动启动任何新实验。

### 48.5 最终报告/summary已归档并核对

已读取本地`logs/train/RESULTS.md`及`logs/train/summary.json`，核对status=complete、full_validation=true、frame_count=829、conditions=[clean,fog,snow]；§48.4的27项AP及天气下降与原始报告/summary一致。
最终summary的checkpoint_sha256与已归档selected.json相同；associated_d1d_training.training_contract_sha256与选模记录相同；frozen_d1c_gt_identity与训练protocol一致。
RESULTS.md实际SHA256与summary.output_hashes一致：`bec4695339c2429c3603dc80a81f53e17f31238b2a33916fccdc890466696a06`。
summary.json实际SHA256：`89312cdef48b02443455b411f9f4adba63f89b4424900c8f017410720276ac34`。
冻结D−1C GT身份：`d5244c4a8fa78dc1d26df1a7ee1f4592c1dd24f91a3f9c575313a081bbeab776`。
最终prediction_cache_identity：`6e59325d5faf7cfd5f25af3254ef89d2eb28799225750bf671ec6844e8850d84`；gt_identity：`2054c792df9ed2168476b6d7deb5300ac48df17c637be3cd9ef3e980b366dccd`。
上述后两项为summary中的记录，未在本地读取完整预测缓存或GT manifest重算其身份。

| 条件 | 官方GT | 固定GT | 天气过滤GT | None帧 | None帧内官方GT | None帧内固定GT | 预测框总数 | O实际GT分母 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Clean | 6224 | 6224 | 0 | 43 | 3 | 3 | 6718 | 6221 |
| Fog | 3673 | 6224 | 2551 | 164 | 57 | 169 | 2239 | 3616 |
| Snow | 5854 | 6224 | 370 | 114 | 115 | 117 | 3482 | 5739 |

Fog/Snow从固定Clean目标中分别过滤40.9865%/5.9447%；计数是整个validate中逐帧目标实例总和，不等于跨时间去重的独立车辆数。
O*的GT分母分别为6224/3673/5854，F均为6224；已核对O分母=官方GT−None帧内官方GT、固定GT=官方GT+过滤GT，以及9组AP的全部天气gap计算。
43/164/114个None帧合计分别涉及3/57/115个官方GT，不能把None帧数直接当作漏检目标数。
Snow另外有1个empty non-None帧，官方按有值但零预测处理，不属于114个None帧；Clean/Fog无此类帧。
Fog/Snow被过滤目标的一对一removed_detected在IoU .3/.5/.7均为0，removed_recall均0；这描述该权重的预测和该匹配规则，不推断所有模型都会如此。
summary还记录Fog保留框最大角点差2.384185791015625e-07 m、未逐位相同；Clean/Snow角点逐位一致。保留此浮点差异，不将Fog写成逐位相同。

本地已核对归档JSON/报告、指标计算与身份关联，未重新执行推理或重算原始匹配。服务器最终protocol/各manifest/逐帧缓存继续保留；`logs/train/protocol.json`仍是训练protocol，不能当作最终评测protocol或用它比对最终summary.output_hashes.protocol.json。
已同步更新本地整理报告`logs/train/FINAL_WEATHER_RESULTS.md`，当前无需为确认这些汇总数字再补传文件；本次未启动任何新实验。

### 48.6 用户要求单独按作者协议评测：新增独立导出/重新推理脚本

用户希望查看按作者协议移除无点GT后的AP，并要求测试代码/脚本。无需重新训练，已有D−1D的O就是OFFICIAL_EXACT：
Clean AP30/AP50/AP70=0.8765509142824365/0.8470714118874587/0.7053976613825419；
Fog=0.5635459741481517/0.5507777853843197/0.4421260304638024；
Snow=0.46624621034876207/0.4352340884702022/0.2889221792378152。

新增独立目录`v2xr_official_selected_eval/`，未修改D−1D、官方快照或历史评测包/权重：
`run.py`、`run_server.sh`、`README.md`、`check_static.py`、`source_snapshot.json`。
ACTION=export（默认）从服务器已完成final_evaluation核验并单独导出O，不跑模型；核对选中权重真实SHA256、训练选择/完成记录、全部已登记最终产物哈希，以及stat_O与原版official_runs/<condition>/official_result_stat.json相同。
ACTION=rerun通过冻结`v2xr_train_attfuse/run_server.sh`的evaluate-checkpoint入口，使用已锁定权重重新执行官方推理；输出新缓存，不覆盖原final_evaluation。支持12帧smoke、全829帧、RESUME=1。
底层仍保存原有O*/F审计，但新报告只展示O，不重新实现GT/后处理/匹配/AP，不改变官方None整帧跳过的行为。
官方配置仍为IntermediateFusionDatasetV2XR、fov/proj_first=true、score>.20、NMS=.15、COM70/max_cav7、BEV/VOC2010/global_sort=False。
默认独立报告根目录`/data/cjm/datasets/logs/v2xr-attfuse-official-protocol-v1`，完整报告`OFFICIAL_O.md/.json`；重新推理缓存`inference_full/`，12帧缓存`inference_smoke12/`与报告`OFFICIAL_O_SMOKE12.md/.json`。
TMPDIR=`/data/cjm/datasets/tmp/v2xr_official_selected_eval`，仅ROCR_VISIBLE_DEVICES选择GPU，POSIX sh/LF；服务器不依赖Git。
本地已做AST/LF/源码哈希静态检查，没有运行导出、模型、Torch/OpenCOOD或服务器实验；shell语法在服务器入口通过sh -n自检。
本次只交付独立脚本和服务器命令，尚无这套新入口的实际运行结果；已知分数来自§48.5已归档的原官方O结果。

### 48.7 独立官方O重新评测已回传，与作者权重及公开表比较

用户回传`logs/train/OFFICIAL_O.md`，标记FULL VALIDATION、每条件829帧、epoch20；来源为`/data/cjm/datasets/logs/v2xr-attfuse-official-protocol-v1/inference_full/summary.json`，表明报告关联独立重新推理目录。
报告权重记录SHA256与既有epoch20一致，9项AP按报告8位小数与上一轮D−1D O完全一致；当前未回传新目录OFFICIAL_O.json或summary，不声称已独立核验该新目录的全部缓存/哈希。
本次OFFICIAL_O.md实际SHA256=`6913d355b8e0d720de6bfa78152a81d672acd279f48d3236736246005f1a0f2a`。

作者普通LiDAR AttFuse在官方冻结README的Clean validation公布86.69/82.58/66.56%；本次87.655091/84.707141/70.539766%，分别高0.9651/2.1271/3.9798 pp。
同一README的Testing为91.21/89.51/80.01%，不能与本次validation直接解释为同集差异。
作者普通Clean训练权重在我们同829帧/同官方O协议复测的AP30/AP50/AP70（来自logs/D/D1/RESULTS.md）与本次对照如下：

| 条件 | 作者权重复测AP30/AP50/AP70 | 本次重训AP30/AP50/AP70 | 差值AP30/AP50/AP70 / pp |
|---|---|---|---|
| Clean | 0.85993605 / 0.82279434 / 0.66625452 | 0.87655091 / 0.84707141 / 0.70539766 | +1.6615 / +2.4277 / +3.9143 |
| Fog | 0.44374745 / 0.41430254 / 0.27543253 | 0.56354597 / 0.55077779 / 0.44212603 | +11.9799 / +13.6475 / +16.6694 |
| Snow | 0.46365775 / 0.42977816 / 0.27207667 | 0.46624621 / 0.43523409 / 0.28892218 | +0.2588 / +0.5456 / +1.6846 |

这里“作者权重复测”是我们的运行结果，不应称作作者在论文/README报告的逐天气原值。作者/本次权重不同，None帧与O分母也可能不同；O定义相同。
本次Clean−Fog/Snow O AP70下降26.3272/41.6475 pp；作者权重39.0822/39.4178 pp。Snow绝对AP提高但gap更大，因为Clean提高幅度更大，不能据gap单项判断性能。

官方README还另列“Multi-modal Diffusion Denoising Results (AttFuse train with fog simulation)”：纯LiDAR的Normal/Fog/Snow AP70=76.09/61.52/45.54%。
这组使用Fog模拟训练，本次只用Clean训练，训练条件不同；该天气表本身未明确validation/test，不能凭上下文猜测split。
若仅列数值差，本次Normal/Fog/Snow分别低5.5502/17.3074/16.6478 pp；不能将其称为同设置复现失败，也不能承诺Fog训练必然复现。
当前支持本次在同验证集官方协议下优于作者发布的普通Clean训练权重，不声称超过作者天气训练或多模态方法。
单种子且使用Clean validation选模，不能据此声明统计显著或正式test泛化优势。

完整本地比较报告：`logs/train/OFFICIAL_O_COMPARISON.md`。作者公开表来源为固定commit f3ebbad95561444a53a17358735b0f0dc4ba89f7的`v2xr_official_repro/upstream/readme.md`；本次只读取/计算文档指标，未运行或追加实验。

## 49. V2X-R Clean GSPR-AttFuse：代码交付，服务器实验待运行

用户要求加入GSPR，在V2X-R重新训练并按O协议评测；随后明确选择“只用V2X-R Clean训练，保持随机初始化、20轮和Clean选模，单独比较加入GSPR的影响”。
本授权仅针对新的GSPR训练/评测，旧GSPr/OPV2V及Source Utility历史实验和源码保持不变，不恢复Source Utility方向。
新目录`v2xr_gspr_clean/`，本地只做文件/AST/LF/JSON/哈希检查，不导入或执行Torch/OpenCOOD/实际科研数据。

### 49.1 模型与训练约定

旧`attfuse_gspr/point_pillar_gspr_attfuse.py`依赖OPV2V骨干，不能整套作为官方V2X-R模型使用。
本次从冻结`attfuse_gspr/reliability.py`按哈希加载GeometryFirstPillarReliability，新增`integration.PointPillarGsprOfficial`继承官方PointPillarIntermediate。
官方VFE完成点特征装饰/padding mask后，在第一层PFN前乘连续GSPR可靠性，原PFN/scatter/AttBEVBackbone/检测头保持官方实现；原点槽位/坐标/点数不变。
GSPR在dataset/GT过滤之后执行，不参与官方框内点数检查，不因权重低而删除点或GT。
新GSPR初始化在官方共享参数初始化之后进行，并恢复额外初始化前的CPU RNG，保证共享参数及全局随机状态可与普通AttFuse对齐；服务器自检会核对该性质。
模型和损失工厂被显式适配，contract/config记录gspr_adapter、actual_model_class；不修改runtime源码。config的model.core_method保留官方解析名称，实际实例是上述GSPR子类。

数据只读取已接受的8083个V2X-R Clean train帧，使用原提取目录/CRC/hash与接受契约；不生成新训练帧、不使用Fog/Snow/Rain点云训练，也不加载OPV2V/作者/旧epoch20权重初始化。
训练seed20261017、batch8、20轮、Adam lr=.002、官方增强/16workers/MultiStep10和15及scheduler.step(epoch)顺序均沿用成功D−1D。
没有天气噪声逐点标签，因此是检测损失驱动的GSPR版本，不冒称旧多任务天气监督GSPR；可靠性/不确定性没有因本轮训练就自动获得天气标签校准。
损失=官方检测损失+0.02×(有效点可靠性均值−.9)^2+0.0001×有效点evidence平方均值，沿用原GSPR检测训练正则；全部设置保存在settings.json与config.yaml.gspr_experiment。

选模改为20轮Clean validation **O AP70最大**，未舍入精确并列取最早epoch；不使用天气选模，不追加轮次。
普通AttFuse虽原按F选择epoch20，但已读取history确认按O排序也选epoch20；最终比较程序会再次校验。
完整epoch提交、模型/optimizer/scheduler/RNG恢复及独立子进程每轮829帧Clean AP沿用D−1D冻结实现。
继承的部分日志仍写D-1D，当前身份以新目录、contract.stage=V2XR_GSPR_CLEAN_O、实际模型类/设置/哈希为准。

### 49.2 官方O评测与对照

按冻结D−1C生成器实际执行原版inference.main，使用新GSPR模型，其余官方dataset、camera/FOV、GT点数过滤、范围、score>.20、NMS=.15、COM70/max_cav7、BEV/VOC2010/global_sort=False保持不变。
O为主结果：当前天气GT、None整帧跳过；O*/F仅保留审计，O replay必须与同次原版官方inference统计完全相同。
最终Clean/Fog/Snow各829帧validate；当前“测试”指此验证基准，不读取正式test。
最终选择锁定后评测天气，并与固定普通AttFuse权重`ccf0211d04f732246d53b70faf8c3f6eb494c19963177d9cedf3d2b106cf01d3`进行同协议比较。
纯AttFuse O AP70参照C/F/S=.7053976613825419/.4421260304638024/.2889221792378152；baseline目录/完成文件哈希固定在PINS.json。
本次无真实服务器结果，不预设GSPR提升，单种子validation差异不称统计显著或test泛化收益。

### 49.3 路径、脚本与检查

新包文件：auditlib.py、integration.py、training.py、train_acceptance.py、evaluation.py、results.py、main.py、server_tests.py、run_server.sh、check_static.py、settings.json、PINS.json、source_snapshot.json、README.md。
auditlib/训练/评测桥接只加载冻结D−1D模块，不在服务器复制/篡改旧源码。train_acceptance桥接将继承模块的POLICY_PATH明确指回原TRAIN_DATA_ACCEPTANCE.json，避免因新HERE误读不存在的新授权。
新包snapshot+PINS固定GSPR、继承引擎与基线结果；服务器不依赖Git。
数据：`/data/cjm/datasets/v2xr-official-train-v2/train`；正式输出：`/data/cjm/datasets/logs/v2xr-gspr-clean-train-v1`；基线：`/data/cjm/datasets/logs/v2xr-clean-attfuse-train-v1`。
临时目录：`/data/cjm/datasets/tmp/v2xr_gspr_clean`；仅ROCR_VISIBLE_DEVICES设置GPU，unset HIP/CUDA_VISIBLE_DEVICES，POSIX sh/LF。
先服务器MODE=self-test：合成体素检查共享权重/RNG、单位权重官方前向一致、检测损失到GSPR梯度、padding和严格模型/optimizer重载。
随后MODE=train SMOKE=1、独立`v2xr-gspr-clean-train-smoke-v1`目录，2个训练batch、2个验证batch、12帧Clean AP。
两项通过后MODE=all，在两个独立进程中依次正式20轮train、选定权重final；训练失败不进入天气，RESUME=1保持身份恢复。
最终`final_evaluation/RESULTS.md`、summary.json优先展示O并附O*/F；`COMPARISON_O.md`、comparison_O.json给出同纯AttFuse的三天气AP差。
smoke日志`/data/cjm/datasets/logs/v2xr_gspr_clean_smoke_v1.log`；正式训练/评测日志`/data/cjm/datasets/logs/v2xr_gspr_clean_full_v1.log`。
本地已通过AST/LF/JSON及冻结源SHA256静态检查；实际shell语法由服务器入口sh -n检查，模型自检/训练/推理均待用户在服务器运行。

### 49.4 首次服务器损失创建失败已修复，改用独立v2目录

用户日志确认：完整train SHA检查进行到216792文件末段、train/val长度8083/829、proj_first=true，成功创建integration.PointPillarGsprOfficial（6593656参数）；在官方train.py第116行create_loss失败。
报错为新适配器ValueError('Unexpected official detection loss')。初版错误检查point_pillar_loss，并继承了opencood.loss.point_pillar_loss.PointPillarLoss；实际官方配置和已归档纯AttFuse config均明确为point_pillar_loss_opencood。
这是新增适配代码选择错误，与用户数据/服务器环境无关；原静态检查及初版直接构造损失的自检没有覆盖真实损失工厂，是此前检查遗漏。
已完整读取官方point_pillar_loss_opencood.py，修复GSPR损失继承PointPillarLossOpenCOOD、forward(epoch=1,train=True)接口和工厂名称检查；保留官方检测计算及500次运行均值日志行为。
固定GSPR正则项不变；额外正则日志累积使用criterion buffer，加入原有严格state_dict恢复，避免引入未保存的运行状态。
服务器自检改为真正调用train_utils.create_loss(parameters)，核对实际父类、检测梯度、总损失日志及criterion buffer严格重载。
本地静态检查新增官方YAML→工厂常量→导入父类→settings链核对，不导入Torch；PINS额外固定实际官方损失源码和官方YAML。
settings.implementation_revision=2，记录准确base_detection_loss。INITIAL_LOSS_FACTORY_FAILURE.json保存初版snapshot/integration哈希和失败阶段。

此异常发生在optimizer创建和训练循环之前，没有该次训练更新或可继续的epoch；用户回传未说明本次MODE是smoke还是full，不据日志猜测。
由于新源码身份改变，禁止手改旧request/contract/snapshot或直接RESUME旧v1。保留原失败目录，改用新输出：
正式默认`/data/cjm/datasets/logs/v2xr-gspr-clean-train-v2`；smoke=`/data/cjm/datasets/logs/v2xr-gspr-clean-train-smoke-v2`；自检在新正式根后加-self-test。
新日志`/data/cjm/datasets/logs/v2xr_gspr_clean_smoke_v2.log`及`v2xr_gspr_clean_full_v2.log`。
只需整目录同步v2xr_gspr_clean（含新snapshot/PINS）并首次RESUME=0，自检通过后smoke，再MODE=all正式20轮及最终O评测。
已提取train、原始包、纯AttFuse及官方runtime保持原样，无需重新解压或重训纯AttFuse。修复后的服务器数值检查/实际训练尚未运行；本地仅AST/LF/JSON/配置绑定/哈希检查通过。

### 49.5 修复后的20轮训练与最终评测已完成：GSPR当前Clean配方无收益（2026-10-19）

用户已回传服务器`/data/cjm/datasets/logs/v2xr-gspr-clean-train-v2/`的training_summary、selected、history、config、selection_policy、训练protocol、COMPARISON_O/comparison_O以及final_evaluation的RESULTS/summary/protocol。
本地保存于`logs/v2xr-gspr-clean-train-v2/`，已读取并逐项核对可用记录，新增ANALYSIS.md和ARTIFACT_AUDIT.json；没有运行Torch/OpenCOOD、点云推理或新训练。

**完成和身份：**training_summary.status=complete、epochs=20；history含20轮，每轮1010个更新，共20200次；实际Clean train8083帧、seed20261017、随机初始化。
按预先固定Clean O AP70最大选epoch17，确为候选中最高值；checkpoint=`/data/cjm/datasets/logs/v2xr-gspr-clean-train-v2/net_epoch17.pth`，记录SHA256=`87941d2d31e10e707c94055c3bebd1ce301e847c58f3e0c62981821aebf13fd5`。
最终summary.status=complete、full_validation=true、每条件829帧；权重身份与selected匹配，冻结GT身份与纯AttFuse一致，训练契约关联一致，实际模型integration.PointPillarGsprOfficial。
protocol/settings记录implementation_revision=2、正确PointPillarLossOpenCOOD父类、无点监督；归档integration源码SHA256与本地修复版一致，未发现旧损失误选或权重/协议混用的记录。
核对5份可用训练产物哈希与training_summary一致；最终RESULTS/protocol哈希与summary一致；comparison中两侧summary及比较报告哈希匹配，20轮选模与AP差值一致。
实际.pth、原始数据、完整逐帧缓存/GT manifest未在本地重算，不能把上述元数据核验称为独立重跑整条训练/评测。

| 条件 | 指标 | 纯AttFuse | GSPR | GSPR−基线 / pp |
|---|---|---:|---:|---:|
| Clean | AP30 | 0.87655091 | 0.85186399 | -2.4687 |
| Clean | AP50 | 0.84707141 | 0.82524928 | -2.1822 |
| Clean | AP70 | 0.70539766 | 0.66119491 | -4.4203 |
| Fog | AP30 | 0.56354597 | 0.55213470 | -1.1411 |
| Fog | AP50 | 0.55077779 | 0.53726272 | -1.3515 |
| Fog | AP70 | 0.44212603 | 0.41628316 | -2.5843 |
| Snow | AP30 | 0.46624621 | 0.43636937 | -2.9877 |
| Snow | AP50 | 0.43523409 | 0.41014700 | -2.5087 |
| Snow | AP70 | 0.28892218 | 0.27924909 | -0.9673 |

本次O下9项指标全部下降。O* AP70的C/F/S差值=-4.4713/-2.7369/-.9817 pp，F差值=-4.4713/-1.6151/-.9233 pp，负结果不只存在于动态GT或None分母下。
GSPR O Clean−Fog/Snow差距24.4912/38.1946 pp，小于基线26.3272/41.6475；缩小1.8360/3.4530 pp来自Clean下降幅度更大，天气绝对AP仍下降，不能当作鲁棒性改善。
两模型官方GT均6224/3673/5854，固定GT各6224；GSPR None帧43/149/99（基线43/164/114），GSPR None帧中官方GT为8/74/122，O实际分母6216/3599/5732。
GSPR预测框C/F/S=6709/2352/3320（基线6718/2239/3482）；同F真值独立一对一TP@.7为4936/1814/2086（基线5077/1865/2191）。差值是净数量，不能当作具体目标恢复/丢失身份。
Fog/Snow被过滤目标2551/370的removed_detected在IoU .3/.5/.7仍全为0；保留目标检出数也减少。更少None帧未带来更高AP。

**曲线：**后期epoch16–20 Clean O AP70=.65152013/.66119491/.65944821/.66112605/.65797225；epoch19比17低约.0069 pp，17选择正确。末轮不再持续改善，当前无直接追加轮数的证据。
早期epoch5从epoch4的.42126612降至.14074760、val loss升至1.03448，第6轮恢复；记录波动，不据此单独判定BN/学习率/数据或GSPR为根因。
epoch17 train loss=.24702923、val loss=.41785772；epoch20 train=.24068216、val=.41922023。含GSPR正则的总loss不能直接与纯AttFuse检测loss比较大小解释原因。

**解释边界与后续建议：**本次检验Clean-only、无逐点标签、随机初始化的GSPR结构+固定正则组合；旧OPV2V多任务/预训练+天气点监督的收益不能直接移用，也不能由此次结果推断所有GSPR设置无效。
当前保留纯AttFuse为主基线，记录本次负结果，不宣称统计显著，不自动改阈值、追加轮数、重训天气或恢复Source Utility。
归档文件缺少最终可靠性/不确定性分布、逐batch正则占比和梯度分析，尚不能确认压低有效点、可靠性塌缩/常数化、正则主导或训练推理统计偏移等根因。
若用户授权继续，优先对选中epoch17做小范围诊断，核对r分布与点数/距离/强度、正则占比、同权重单位权重的特征变化；单位权重干预会改变训练分布/BN，只作机制诊断，不能代替匹配训练消融。
后续新模型开发宜从train按场景划分开发集，避免继续围绕已反复分析的829帧选方法；正式test仍隔离。

### 49.6 对照对象澄清：作者完整AttFuse+MDD L+4DR不等于此前LiDAR基线

用户引用作者AttFuse+MDD（L+4DR）的Normal/Fog/Snow AP30/50/70=90.99/89.64/81.96、87.37/83.90/68.64、83.78/81.19/66.86%。
已读取本地论文`研究相关论文/融合相关论文/01_02_V2X-R_2025.pdf`第6–8页文本并渲染检查第7/8页，确认第7页Table5完全包含上述数字。
PDF表5原顺序为Snow/Fog/Normal；用户引用表按Normal/Fog/Snow排列，数字均对应正确。
第5.4节明确：训练模型时每帧随机选择[normal,fog]的LiDAR点云，Fog为seen域、Snow为unseen域。
因此作者完整方法与本项目当前实验有三个实质差异：LiDAR+4D Radar输入（当前纯LiDAR）、MDD多模态扩散去噪模块（当前普通AttFuse或GSPR）、Normal/Fog混合训练（当前仅Clean训练）。
官方冻结MDD配置`L_4DR_Fusion_with_MDD/V2XR_AttFuse.yaml`同时记录train_sim=true、use_DeLidar=true、model=point_pillar_intermediate_lrf_MDD、batch2/30轮，与本次batch8/20轮的纯LiDARClean配方不同；仅核对代码，不改任何实验。

论文表2的普通LiDAR AttFuse Clean validation 86.69/82.58/66.56%，与此前所说本项目纯AttFuse AP70高约3.98 pp对应。该结论只针对普通LiDAR Clean验证基线，不代表超过AttFuse+MDD完整方法。
表5中的Fog模拟训练纯LiDAR AttFuse为Normal/Fog/Snow AP70=76.09/61.52/45.54%；L+4DR无MDD为81.75/62.89/63.71%；L+4DR有MDD为81.96/68.64/66.86%。这些训练设置不能与当前Clean-only 829帧结果直接当作匹配对照。
本次论文阅读还确认表7、表8的标题明确“V2X-R testing with fog-simulation”，其中同一MDD Fog AP70=68.64%。表5本身未单列split，严格对照仍须核对全部天气的实际测试流程；当前本项目仅829帧validate，不据数值接近就称同集复现。

**GT过滤边界的纠正：**被冻结的IntermediateFusionDataset在纯LiDAR下检查LiDAR框内点；当use_Lidar和use_Radar均true时，使用LiDAR有点 OR Radar有点保留GT。
MDD配置未关闭任一模态，属于多模态分支。因此此前实测Fog过滤2551/6224（40.99%）、Snow370/6224（5.94%）的结论仅针对已审计的纯LiDAR路径，不能未经实测直接归因到作者L+4DR+MDD表5那行。
这并不撤销LiDAR O/O*/F审计及GSPR负结果，但限制了“作者论文成绩由删GT造成”的解释范围；多模态GT及正式test尚未审计/复现。
以上为论文/代码只读核对和参照澄清，未授权或启动雷达/MDD训练、test评测或新的模型实验。

## 50. V2X-R Clean+Fog 混合训练 L0–L2 实测完成（2026-10-21 归档）

### 50.1 证据来源、实验设置和完成范围

用户确认服务器 L0→L1→L2 全流程已运行完成；已读取 GitHub main 上的机器可读结果 `logs/train/FINAL_RESULTS.json`，其中顶层 `status=complete`、六个变体的 `status=complete`、每组 Clean/Fog/Snow 的 `full_validation=true`、`frame_count=829`。本节数字直接由该 JSON 提取/换算；**未独立登录服务器检查实际 .pth、完整逐帧缓存、训练日志或重算 AP**。因此“元数据报告完成”不等于已完成独立字节级与逐帧复现审计。

代码和实验协议：`v2xr_weather_l012/`，以 `settings.json`、`PROTOCOL.md` 为固定实现说明；正式服务器输出约定为 `/data/cjm/datasets/logs/v2xr-weather-l012-v1/`，报告约定为 `reports/FINAL_RESULTS.json` 等；本次实际提交仓库的汇总副本位于 `logs/train/FINAL_RESULTS.json`，不能据此推断其他服务器原始产物已经上传 GitHub。

训练集采用已验收的 V2X-R 8083 个协作帧（24088 份单车点云），Clean/Fog 配对；**实际作者源码是每辆 CAV 每次读取独立地以 3/7 概率抽 Fog、4/7 概率抽 Clean，并非协作帧级 1:1 或每帧各半**，同一协作帧可以混合天气。沿用官方训练时依当前点云决定 GT 的规则；每轮 drop_last 后常规抽样索引为 8080，原版遇空 GT 可重采样。C/D 为随机初始化20轮、seed=20261017、batch8、Adam lr=.002、官方原配置与调度；D 在原 AttFuse 点特征前端加入 `v2xr_gspr_clean` 的 GSPR 和原有固定正则，无逐点天气标签。每轮以完整829帧 **Clean 官方 O AP70** 选择 checkpoint，随后完整评测 Clean/Fog/Snow；未使用正式 test。总流程和模型配置详细见 `v2xr_weather_l012/PROTOCOL.md`。L2-BN 使用全部训练 Fog 协作帧重新累计 BN 统计量，不更新权重；L2-FiLM 冻结原主干，仅训练新增64通道FiLM三轮，按 Clean O AP70 选择。

对照命名：C=混合天气纯AttFuse；D=相同配方 GSPR-AttFuse；C_BN/D_BN=分别对 C/D 做 Fog 训练集 BN 校准；C_FiLM/D_FiLM=分别在 C/D 上独立训练 FiLM，BN/FiLM 不叠加。O=原版官方天气 GT 且预测 None 整帧跳过；O*=天气 GT 且 None 按零预测；F=三天气统一固定 Clean GT 且 None 按零预测；其余主要评测设置沿用固定官方 D−1C（BEV/VOC2010、逐帧排序、score>.20、NMS=.15、COM70、max_cav7）。

### 50.2 六组完整结果

**官方 O：各单元格为 AP30 / AP50 / AP70，单位是 0–1 的 AP 小数；每种天气829帧。**

| 模型 | Clean | Fog | Snow |
|---|---|---|---|
| C | 0.854908 / 0.821546 / 0.684721 | 0.875527 / 0.854783 / 0.675034 | 0.685194 / 0.639444 / 0.427073 |
| D | 0.855342 / 0.823801 / 0.698510 | 0.890636 / 0.873004 / 0.756856 | 0.623958 / 0.591801 / 0.428650 |
| C_BN | 0.835745 / 0.804922 / 0.651312 | 0.864850 / 0.850086 / 0.664333 | 0.652716 / 0.599808 / 0.393505 |
| D_BN | 0.808513 / 0.775783 / 0.636405 | 0.866912 / 0.847916 / 0.705336 | 0.509813 / 0.479105 / 0.332710 |
| C_FiLM | 0.846785 / 0.817626 / 0.680741 | 0.876383 / 0.853111 / 0.673519 | 0.679638 / 0.633888 / 0.433029 |
| D_FiLM | 0.852305 / 0.820301 / 0.694579 | 0.891509 / 0.870741 / 0.757324 | 0.624066 / 0.588402 / 0.428047 |

**固定 Clean GT 的 F 协议：AP70（0–1 小数），是核查天气退化的必要独立审计。**

| 模型 | Clean F AP70 | Fog F AP70 | Snow F AP70 |
|---|---:|---:|---:|
| C | 0.684391 | 0.398405 | 0.399008 |
| D | 0.698398 | 0.446205 | 0.400964 |
| C_BN | 0.650893 | 0.391937 | 0.367267 |
| D_BN | 0.636303 | 0.415677 | 0.307906 |
| C_FiLM | 0.680413 | 0.396978 | 0.404016 |
| D_FiLM | 0.694468 | 0.446481 | 0.400469 |

**相对于对应原模型的官方 O AP70 差值（百分点 pp；不代表统计显著）。**

| 对照 | Clean | Fog | Snow |
|---|---:|---:|---:|
| D-C | +1.3789 | +8.1822 | +0.1578 |
| C_BN-C | -3.3409 | -1.0701 | -3.3568 |
| D_BN-D | -6.2104 | -5.1519 | -9.5941 |
| C_FiLM-C | -0.3979 | -0.1515 | +0.5956 |
| D_FiLM-D | -0.3931 | +0.0468 | -0.0604 |

旧 Clean-only 纯AttFuse 官方 O AP70（§48/§49）为 Clean=0.705398、Fog=0.442126、Snow=0.288922。混合天气模型相对它的差值（pp）：

| 对照 | Clean | Fog | Snow |
|---|---:|---:|---:|
| C - Clean-only AttFuse | -2.0677 | +23.2908 | +13.8150 |
| D - Clean-only AttFuse | -0.6888 | +31.4730 | +13.9728 |

### 50.3 预设门槛和结果解释

- **L0 通过**：C 的 Fog O AP70=0.675034，相对旧 Clean-only AttFuse +23.2908 pp，超过事先预设的 +5 pp 门槛；该门槛仅用于决定是否推进实验，不是数据或科学结论正确性的充分证明。
- **L1 通过**：D−C 官方 O AP70 在 Clean/Fog/Snow 分别为 +1.3789/+8.1822/+0.1578 pp，满足 Fog≥+1.5 pp、至少两天气非负、Clean≥−1 pp。20/20 轮记录 `epoch_trace_equal=true`，包含实际读路径、worker Python/NumPy 随机状态、重复读取、GT和增强后输入哈希；这证明上述**观测采样轨迹**配对，不证明 ROCm/GPU 算子逐位确定或两模型仅有一个训练参数变化。
- **GSPR 在 Fog 有实际固定 GT 收益**：Fog O AP70 D−C=+8.1822 pp；对应 F AP70 D−C=+4.7800 pp（C=0.398405→D=0.446205）。O 的增益不能全部归因于模型对固定目标检测能力的同等幅度提升，但 F 下仍有明确正差值。
- **需防止官方 Fog GT 过滤误读**：Clean 固定 GT=6224；Fog 官方GT=3673，剔除2551/6224（40.99%）；C/D 的 Fog 官方 O AP70 分别0.675034/0.756856，但固定 GT F AP70 仅为 0.398405/0.446205。不能将 O 数值直接解释为对原6224个目标的检测率，更不能未经重新匹配/相同split与作者公开论文数值断言超越。
- **未见 Snow 的一致改善**：D−C Snow O 的 AP30/AP50/AP70 为 -6.1236/-4.7643/+0.1578 pp；虽然 AP70 微升，固定真值下预测的一对一匹配计数 IoU≥0.7 从 C 的 3370 降为 D 的 3123。一对一匹配数、排序 AP、目标召回定义不同，不把数量差直接等同目标身份变化。
- **BN 负收益**：C_BN−C 的 Fog O AP70=-1.0701 pp，D_BN−D=-5.1519 pp；两组 Clean/Fog/Snow AP70 均下降，当前不支持全训练 Fog 的无监督 BN 重校准作为有效增益模块。
- **FiLM 收益很小/不稳定**：C_FiLM−C 的 Fog O AP70=-0.1515 pp，D_FiLM−D=+0.0468 pp；C_FiLM 在 Snow AP70 +0.5956 pp，但 Clean/Fog 下降；D_FiLM Fog 仅约+0.05 pp，尚不足以支持有效的独立模块结论。
- **与 §49 Clean-only GSPR 的区别**：§49 GSPR 相对于匹配 Clean-only AttFuse 在三天气 O AP70 均下降；本轮改为 Clean+Fog 混合训练后出现 Fog +8.18 pp，提示训练天气分布可能影响 GSPR 的可用性。两轮模型/训练条件不同，不构成单独的“天气×GSPR”严格因果分解；需要进一步跨种子与模块诊断。
- **研究局限**：同一829帧验证集上单种子比较，没有置信区间、独立种子或正式test；本节无真实硬件日志独立核验；Rain 未纳入该 V2X-R 配对数据实验。下一步先做 Fog 逐目标、定位、打分与可靠性分布诊断，并用新随机种子和独立test确认能否稳定泛化。

### 50.4 权重身份及下一步作者基线同集评测

机器可读汇总中六个模型的 `checkpoint_sha256`：
- C：`d33781c7fcd548e17b66f1959bd50442603cf0d4e84e24d0c9f204afc23cf569`
- D：`bcf3e0ad3b63697b076cdddf9b49939ca15528525fca86bc002a1f45184a2c6f`
- C_BN：`b67f2a8fe8c743c65fefd5fa5ad2a63d9734770c3a304f1f1b0030a2b6caa915`
- D_BN：`fff2521527df0df1bcd3a4d770fbc262b1b1de62a107abfa63659147383c923c`
- C_FiLM：`eacb6f7f92f424adef8266446a04c1839bde7434b4864feac40813247543551b`
- D_FiLM：`71b150fa2efcfddbde133bd99d60bf8ca21adacde6e9cf67024f956c810af109`

下一阶段**不需要重训**。用户已将作者官方 Clean+Fog 天气训练版纯 LiDAR 权重 `AttFuse_LiDAR_train_w_fog.pth` 放到服务器 `/home/cjm/OpenCOOD-main/cjmnet/logs/` 下；这一文件存在性及内容尚未由本地工具直接检查。官方 Hugging Face 文件：`https://huggingface.co/hx24/V2X-R_Benchmark/blob/main/benchmark/l%2Br_train_with_fog_sim/AttFuse_LiDAR_train_w_fog.pth`；官方页面公布 SHA256=`195a873a814cfeb9e392ea7e3b6ae12a142b009719194d3c0a5980cbb4d089f3`。后续服务器须自行校验本地实际 SHA256、格式、模型key/shape/dtype，不应未经校验宣称是作者权重。

下一步使用这一冻结权重，**只推理不训练**，在同一 V2X-R validation 829帧 Clean/Fog/Snow 上复用 D−1C 冻结官方预测和 O/O*/F 重放，输出与 C/D 的匹配对照与逐帧GT/None审计。作者论文表5所报天气训练纯 LiDAR AttFuse Normal/Fog/Snow AP70=0.7609/0.6152/0.4554，但论文天气结果与本项目 validation 的数据划分/目标过滤是否一致尚未确认，不能直接把不同来源AP差额视作同集SOTA优势；作者多模态 L+4DR+MDD 更不属于同模态匹配对照。

归档真值以 `logs/train/FINAL_RESULTS.json` 机器可读副本为准，旧 §48/§49 历史段落保持原样。

## 51. GSPR 后端可靠性聚合与观测支撑框架（2026-10-23，本地实现，服务器待验收）

用户本轮授权新增GSPR后端A/B/C及两阶段对照，未重启§43停止的source utility。新增独立 `gspr_repe/`，未创建分支/worktree，旧冻结代码、配置、权重和结果未修改。

核心：`modules.py`实现柱内几何/可靠性聚合A、Q/S观测双头B、八邻域可信来源残差增强C；`models.py`继承官方GSPR适配器，A替代其PFN前软加权入口，B/C接Scatter之后、原官方AttFuse之前。A启用时填充点排除于BN和池化；A关闭保留旧路径。新模块门控bias−5，不能声称初始化严格等价；仅全关闭B0要求与旧GSPR严格等价。

`data.py`新增训练集Clean参考读取，保存/恢复Python/NumPy随机状态，官方重新抽帧或车辆数量不齐时屏蔽配对监督。Q以停止梯度GSPR作为弱代理；S优先采用训练Clean的相对有效点数，在有参考/有观测位置监督，未对齐时使用距离分层代理。各辅助损失权重.01。Clean不作为完整场景真值；C暂只增强有观测单元，不声称补全完全空区域。

五组B0/B1/B2/B3/B4分别为旧GSPR、+A、普通增强、+A+B+目标状态门控普通增强、+A+B+C约束增强。全部匹配随机初始化共享主干/GSPR、seed20261017、20轮、batch8、官方Adam和调度；Clean/Fog每CAV抽样4/7、3/7，只按829帧Clean O AP70选模型；Snow只在锁定后验证，正式test不读取。B3/B4同参数量和邻域循环。B3−B2包含A/B共同改变，不能由此单独归因B或排名三个模块贡献；没有统计显著性结论。

唯一入口 `gspr_repe/run_all.sh`，POSIX sh、LF。MODE=preflight/smoke/all；默认RESUME=1。串行顺序：数据/环境与冻结依赖哈希→服务器结构/梯度/参数更新/重载/失败恢复测试→五组真实数据16帧一轮smoke及12帧三天气推理→第一阶段验收→五组正式20轮及官方O/O*/F评测→自动报告。技术失败停止；负性能保留并继续。恢复核验阶段产物和完整模型/优化器/调度器/损失/RNG状态，不依赖.git。

沿用 `v2xr_weather_l012` 的训练/数据审计副本与 `v2xr_train_attfuse`/`v2xr_protocol_audit` 冻结官方评测，不重写AP/NMS/匹配。新增 `diagnostics.py`直接用官方匹配轨迹和已有D−1B框内点证据汇总距离/点数分层召回、目标恢复/丢失、误检净变化。速度记录区分同步推理与host训练计时；缓存恢复不把本次部分前向误称全量性能。

论文/源码审查见 `gspr_repe/SOURCE_REVIEW.md`。扫描本地资料实际找到PointASNL与HEDNet/SAFDNet压缩源码；TransBridge仅找到论文。TransBridge原方法使用多帧密集重建监督，本次双头为适配与推断，不能称完整复现。新增算子为纯PyTorch，不安装新依赖；实际ROCm兼容性尚待服务器验证。

本地通过：AST、模块间导入符号静态核对、配置、SHA256、LF/禁用Shell语法扫描、sh -n。未运行任何Torch/OpenCOOD科研实验、真实数据、ROCm测试或训练，不能登记第一/第二阶段成功或编造AP。完整状态见 `gspr_repe/LOCAL_VALIDATION.md`。

默认服务器输出 `/data/cjm/datasets/logs/gspr-repe-v1`；启动日志 `/data/cjm/datasets/logs/gspr_repe_v1.launch.log`。成功第一阶段报告 `stage1/STAGE1_REPORT.md`；正式报告 `reports/{STAGE2_RESULTS.json,STAGE2_REPORT.md,ABLATION_TABLE.csv,FAILURE_CASES.md,FINAL_RESEARCH_ASSESSMENT.md,STRATIFIED_DETECTION.json}`。当前这些服务器结果均未生成。未完成研究项包括Q/S与具体目标变化的因果干预、空白区域补全和跨种子稳定性。


### 51.1 NVIDIA租用服务器适配（2026-10-24，本地静态通过，远端未执行）

依据AI_REMOTE.md新增gspr_repe/cuda_runtime.py和cuda_evaluation.py，run_all.sh默认BACKEND=cuda，BACKEND=rocm保留旧海光路径。新默认Python/runtime为/workspace/cjmnet_cuda_env下的cu124环境和已编译official-runtime-cu124；数据根目录仍为/data/cjm/datasets中已迁移的train、配对validate及官方validate。新增独立CUDA输入审计与结果目录，不要求原.7z、D−1A/B/C运行产物或旧纯AttFuse权重。原提取文件/记录和验收身份仍核对，不改历史JSON/哈希。

CUDA初次预检实际执行native CPU/CUDA框内点及框重叠，并重新用官方观察器审计三天气829帧GT。新评测外层直接调用官方inference_intermediate_fusion、原后处理/AP及已有匹配跟踪器；不声称新CUDA与旧HIP逐位一致。批次8/20轮和Clean/Fog配方保持，内存不足时技术失败停下，不隐式减批次。

补充独立train_acceptance.py，将策略文件固定指向原v2xr_train_attfuse目录，避免当前独立auditlib.HERE导致的错误策略路径。旧冻结包未改。新增test_migration_static.py只测试语法树、原提取哈希编码和模拟场景索引，无科研模型/数据。新打包器delivery_tools/build_repe_cuda_source_bundle.py仅打包源码及冻结依赖。上传/启动见gspr_repe/CUDA_REMOTE.md；服务器结果/显存/AP仍未获得。


### 51.2 交付压缩包布局调整（2026-10-24）

按用户要求，新的`cjmnet_repe_cuda_flat.zip`取消cjmnet外层，直接解压到`/workspace/gspr_source/cjmnet`。打包器、CUDA_REMOTE.md及AI_REMOTE.md已同步说明，新增ZIP顶层断言及逐文件校验。旧包的父目录解压命令不适用于flat包。此次仅调整交付布局和说明，没有运行实验。

## 52. 作者天气训练纯 LiDAR AttFuse 同集评测结果（2026-10-24 归档）

### 52.1 文件来源、完成记录和评测设置

本节记录用户回传的作者固定权重评测结果，数据来自压缩包内机器可读结果及逐帧记录。本地只读取和核对已生成文件，未运行模型或实验；既有章节保持原样。

- 本地结果包：`logs/v2xr-gspr-clean-train-v2/v2xr-author-fog-attfuse-val-v1-results.tar`；共28个文件，大小14684160字节。
- 结果包 SHA256：`189572309293a420b1deabede89f02124d3ec95c4170b739541aeffaabd3ffb2`。
- 服务器输出目录：`/data/cjm/datasets/logs/v2xr-author-fog-attfuse-val-v1/`。
- `completion.json` 记录 `status=complete`、`mode=all`、`full_evaluation=true`；日志中的运行日期为2026-10-21（UTC），本节归档日期为2026-10-24。
- 运行身份：`e3246b5b36fe9c96448682a7dffa6ce51d08764f8a5c66ffa8d9f046659482b3`。
- 作者模型名称为 `Author_Fog_AttFuse`；C、D沿用§50的归档模型。本次为固定外部权重评测，无训练、微调或本地训练选模，未读取正式test。
- Clean/Fog/Snow各829个配对协作帧，共2487条逐帧评测记录、2487条GT清单记录和2487条预测清单记录；天气移除目标记录2921条。
- 复用冻结的官方俯视框交并比（BEV IoU）和VOC2010平均精度（AP）计算，`global_sort=False`；每种天气的三套协议共用同一份预测。
- O：天气条件下的官方真实目标，预测为None时跳过整帧。O*：使用相同天气真实目标，None按零预测计入。F：使用对应Clean帧固定真实目标，None按零预测计入。
- 冻结GT身份：`d5244c4a8fa78dc1d26df1a7ee1f4592c1dd24f91a3f9c575313a081bbeab776`；数据身份：`d470d8f033464537e29ba38e07d13a00710ef3be7b6bc0e6ef0f80d2c3304cf5`。
- 作者、C、D的原始审计来源均记录为完整829帧；对照记录中的协议、随机种子、加载进程数、软件版本、官方源码与配置、输入清单、帧序列及计算设置检查均为 `direct_artifact_verified`。
- 推理随机种子与 `PYTHONHASHSEED` 均为20261016，数据加载进程数16；服务器GPU为K100_AI、物理卡2，HIP版本6.3.25405，Torch版本2.5.1+das.opt1.dtk25042。

### 52.2 作者权重加载记录

- 原始文件：`/home/cjm/OpenCOOD-main/cjmnet/logs/AttFuse_LiDAR_train_w_fog.pth`。
- 官方来源：https://huggingface.co/hx24/V2X-R_Benchmark/blob/main/benchmark/l%2Br_train_with_fog_sim/AttFuse_LiDAR_train_w_fog.pth。
- 实际权重SHA256：`195a873a814cfeb9e392ea7e3b6ae12a142b009719194d3c0a5980cbb4d089f3`，与任务指定的官方文件哈希一致。
- 文件格式为直接参数字典（`plain_state_dict`）；统一移除 `module.` 命名前缀，转换记录为一一对应。
- 模型类：`opencood.models.point_pillar_intermediate.PointPillarIntermediate`，纯LiDAR AttFuse。
- 使用 `torch.load(map_location=cpu, weights_only=True)`；严格加载142个参数/缓冲张量，缺失键和多余键均为空，全部shape、dtype及加载后数值逐项一致。
- 官方配置仅 `root_dir`、`validate_dir` 两项路径与模板不同；配置SHA256为 `c4a4d135594b577b0108b0f49a81937f95ee80c422bfc559a443f8a771bbd76a`。

对照权重身份：

- Author_Fog_AttFuse：`195a873a814cfeb9e392ea7e3b6ae12a142b009719194d3c0a5980cbb4d089f3`。
- C：`d33781c7fcd548e17b66f1959bd50442603cf0d4e84e24d0c9f204afc23cf569`。
- D：`bcf3e0ad3b63697b076cdddf9b49939ca15528525fca86bc002a1f45184a2c6f`。

C/D完整精度参考来源为服务器 `v2xr_author_fog_eval/reference/FINAL_RESULTS.json`，其原始归档来源为 `logs/train/FINAL_RESULTS.json`；SHA256为 `13f6ac5de347144cde676270bac651a0e2af8b3a90d89f1a580bcc4f5f78f848`。

### 52.3 同集绝对成绩

以下AP均为0—1小数，按机器可读JSON整理，表中保留10位小数；完整精度保存在原结果包的 `author_evaluation/summary.json` 和 `reports/MATCHED_COMPARISON.json`。

| 模型 | 天气 | 协议 | AP30 | AP50 | AP70 |
|---|---|---|---:|---:|---:|
| Author_Fog_AttFuse | clean | O | 0.8471793813 | 0.8068832768 | 0.6173229034 |
| Author_Fog_AttFuse | clean | O* | 0.8471793813 | 0.8068832768 | 0.6173229034 |
| Author_Fog_AttFuse | clean | F | 0.8471793813 | 0.8068832768 | 0.6173229034 |
| Author_Fog_AttFuse | fog | O | 0.8523427707 | 0.8157794636 | 0.5996286321 |
| Author_Fog_AttFuse | fog | O* | 0.8521107145 | 0.8155573619 | 0.5994653790 |
| Author_Fog_AttFuse | fog | F | 0.5142583603 | 0.4851925794 | 0.3537654783 |
| Author_Fog_AttFuse | snow | O | 0.6594025263 | 0.6087887260 | 0.3736474303 |
| Author_Fog_AttFuse | snow | O* | 0.6559106441 | 0.6055648704 | 0.3716687712 |
| Author_Fog_AttFuse | snow | F | 0.6211083032 | 0.5719646787 | 0.3499213362 |
| C | clean | O | 0.8549079280 | 0.8215460712 | 0.6847206685 |
| C | clean | O* | 0.8544958580 | 0.8211500817 | 0.6843906297 |
| C | clean | F | 0.8544958580 | 0.8211500817 | 0.6843906297 |
| C | fog | O | 0.8755274093 | 0.8547829271 | 0.6750336026 |
| C | fog | O* | 0.8745739354 | 0.8538520445 | 0.6742984721 |
| C | fog | F | 0.5168412064 | 0.5044241768 | 0.3984052963 |
| C | snow | O | 0.6851941652 | 0.6394441468 | 0.4270725323 |
| C | snow | O* | 0.6806293253 | 0.6351840987 | 0.4242273276 |
| C | snow | F | 0.6411340785 | 0.5974241185 | 0.3990081580 |
| D | clean | O | 0.8553416529 | 0.8238005836 | 0.6985099588 |
| D | clean | O* | 0.8552042265 | 0.8236682249 | 0.6983977303 |
| D | clean | F | 0.8552042265 | 0.8236682249 | 0.6983977303 |
| D | fog | O | 0.8906360298 | 0.8730036652 | 0.7568556591 |
| D | fog | O* | 0.8894236203 | 0.8718152583 | 0.7558253628 |
| D | fog | F | 0.5250606735 | 0.5146669385 | 0.4462050645 |
| D | snow | O | 0.6239576654 | 0.5918006733 | 0.4286504340 |
| D | snow | O* | 0.6205468958 | 0.5885656850 | 0.4263072816 |
| D | snow | F | 0.5836570579 | 0.5535770437 | 0.4009644644 |

作者模型的帧数、预测和真实目标计数：

| 天气 | 帧数 | 官方GT | 固定Clean GT | 天气移除GT | None帧 | 预测框 | O计分GT | O*计分GT | F计分GT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| clean | 829 | 6224 | 6224 | 0 | 34 | 7001 | 6224 | 6224 | 6224 |
| fog | 829 | 3673 | 6224 | 2551 | 88 | 3939 | 3672 | 3673 | 6224 |
| snow | 829 | 5854 | 6224 | 370 | 57 | 5058 | 5823 | 5854 | 6224 |

作者模型三种天气的非None空预测帧数均为0。Clean/Fog/Snow的None帧内，官方GT数量分别为0/1/31，固定Clean GT数量分别为0/80/32。

### 52.4 模型差值

差值单位为百分点，定义为100×（左模型AP−右模型AP），表中保留6位小数。

| 对照（左−右） | 天气 | 协议 | ΔAP30 | ΔAP50 | ΔAP70 |
|---|---|---|---:|---:|---:|
| Author_Fog_AttFuse-C | clean | O | -0.772855 | -1.466279 | -6.739777 |
| Author_Fog_AttFuse-C | clean | O* | -0.731648 | -1.426680 | -6.706773 |
| Author_Fog_AttFuse-C | clean | F | -0.731648 | -1.426680 | -6.706773 |
| Author_Fog_AttFuse-C | fog | O | -2.318464 | -3.900346 | -7.540497 |
| Author_Fog_AttFuse-C | fog | O* | -2.246322 | -3.829468 | -7.483309 |
| Author_Fog_AttFuse-C | fog | F | -0.258285 | -1.923160 | -4.463982 |
| Author_Fog_AttFuse-C | snow | O | -2.579164 | -3.065542 | -5.342510 |
| Author_Fog_AttFuse-C | snow | O* | -2.471868 | -2.961923 | -5.255856 |
| Author_Fog_AttFuse-C | snow | F | -2.002578 | -2.545944 | -4.908682 |
| D-Author_Fog_AttFuse | clean | O | +0.816227 | +1.691731 | +8.118706 |
| D-Author_Fog_AttFuse | clean | O* | +0.802485 | +1.678495 | +8.107483 |
| D-Author_Fog_AttFuse | clean | F | +0.802485 | +1.678495 | +8.107483 |
| D-Author_Fog_AttFuse | fog | O | +3.829326 | +5.722420 | +15.722703 |
| D-Author_Fog_AttFuse | fog | O* | +3.731291 | +5.625790 | +15.635998 |
| D-Author_Fog_AttFuse | fog | F | +1.080231 | +2.947436 | +9.243959 |
| D-Author_Fog_AttFuse | snow | O | -3.544486 | -1.698805 | +5.500300 |
| D-Author_Fog_AttFuse | snow | O* | -3.536375 | -1.699919 | +5.463851 |
| D-Author_Fog_AttFuse | snow | F | -3.745125 | -1.838763 | +5.104313 |
| D-C | clean | O | +0.043372 | +0.225451 | +1.378929 |
| D-C | clean | O* | +0.070837 | +0.251814 | +1.400710 |
| D-C | clean | F | +0.070837 | +0.251814 | +1.400710 |
| D-C | fog | O | +1.510862 | +1.822074 | +8.182206 |
| D-C | fog | O* | +1.484968 | +1.796321 | +8.152689 |
| D-C | fog | F | +0.821947 | +1.024276 | +4.779977 |
| D-C | snow | O | -6.123650 | -4.764347 | +0.157790 |
| D-C | snow | O* | -6.008243 | -4.661841 | +0.207995 |
| D-C | snow | F | -5.747702 | -4.384707 | +0.195631 |

包内AP70协议差值的算术分项如下。None分项为两个模型各自“O−O*”之差，GT分项为两个模型各自“O*−F”之差；三项之和等于O差值。

| 对照 | 天气 | F差值 | None分项 | GT分项 | O差值 |
|---|---|---:|---:|---:|---:|
| Author_Fog_AttFuse-C | clean | -6.706773 | -0.033004 | +0.000000 | -6.739777 |
| Author_Fog_AttFuse-C | fog | -4.463982 | -0.057188 | -3.019328 | -7.540497 |
| Author_Fog_AttFuse-C | snow | -4.908682 | -0.086655 | -0.347173 | -5.342510 |
| D-Author_Fog_AttFuse | clean | +8.107483 | +0.011223 | +0.000000 | +8.118706 |
| D-Author_Fog_AttFuse | fog | +9.243959 | +0.086704 | +6.392040 | +15.722703 |
| D-Author_Fog_AttFuse | snow | +5.104313 | +0.036449 | +0.359538 | +5.500300 |
| D-C | clean | +1.400710 | -0.021781 | +0.000000 | +1.378929 |
| D-C | fog | +4.779977 | +0.029517 | +3.372712 | +8.182206 |
| D-C | snow | +0.195631 | -0.050205 | +0.012365 | +0.157790 |

### 52.5 一对一匹配与逐帧计数

下表为Fog、框交并比≥0.7时的一对一匹配计数。“正确匹配”和“误检”来自原匹配器记录，漏检为相应协议下未匹配的真实目标。

| 模型 | 预测框 | None帧 | 协议 | 正确匹配 | 误检 | 漏检 | 计分GT |
|---|---:|---:|---|---:|---:|---:|---:|
| Author_Fog_AttFuse | 3939 | 88 | O | 2866 | 1073 | 806 | 3672 |
| Author_Fog_AttFuse | 3939 | 88 | O* | 2866 | 1073 | 807 | 3673 |
| Author_Fog_AttFuse | 3939 | 88 | F | 2866 | 1073 | 3358 | 6224 |
| C | 3879 | 111 | O | 3028 | 851 | 641 | 3669 |
| C | 3879 | 111 | O* | 3028 | 851 | 645 | 3673 |
| C | 3879 | 111 | F | 3031 | 848 | 3193 | 6224 |
| D | 3922 | 103 | O | 3211 | 711 | 457 | 3668 |
| D | 3922 | 103 | O* | 3211 | 711 | 462 | 3673 |
| D | 3922 | 103 | F | 3212 | 710 | 3012 | 6224 |

Fog的F协议正确匹配按保留/移除目标分组：作者2866/0，C为3028/3，D为3211/1；两组相加分别为2866、3031、3212。C在Fog有3帧非None空预测，作者和D均为0。

作者模型在F协议匹配中，保留/移除目标的正确匹配数如下，三元组按交并比≥0.3/0.5/0.7排列：

| 天气 | 保留目标匹配数 | 移除目标匹配数 |
|---|---|---|
| clean | 5688 / 5514 / 4778 | 0 / 0 / 0 |
| fog | 3378 / 3310 / 2866 | 24 / 9 / 0 |
| snow | 4142 / 3969 / 3088 | 8 / 5 / 1 |

以下逐帧计数采用固定Clean GT、交并比≥0.7的一对一匹配；每个对照和天气均为829帧。新增检出/不再检出来自逐帧目标ID记录。

| 对照（左−右） | 天气 | 匹配数增加帧 | 匹配数减少帧 | 匹配数相同帧 | 新增检出目标 | 不再检出目标 | 匹配数净变化 |
|---|---|---:|---:|---:|---:|---:|---:|
| Author_Fog_AttFuse-C | clean | 116 | 248 | 465 | 209 | 447 | -238 |
| Author_Fog_AttFuse-C | fog | 55 | 157 | 617 | 122 | 287 | -165 |
| Author_Fog_AttFuse-C | snow | 148 | 284 | 397 | 427 | 708 | -281 |
| D-Author_Fog_AttFuse | clean | 255 | 92 | 482 | 451 | 193 | 258 |
| D-Author_Fog_AttFuse | fog | 211 | 19 | 599 | 407 | 61 | 346 |
| D-Author_Fog_AttFuse | snow | 207 | 208 | 414 | 608 | 574 | 34 |
| D-C | clean | 167 | 141 | 521 | 267 | 247 | 20 |
| D-C | fog | 161 | 34 | 634 | 280 | 99 | 181 |
| D-C | snow | 164 | 286 | 379 | 440 | 687 | -247 |

九份 `reports/frame_gains_*.jsonl` 各829条记录。作者全部逐帧指标保存在 `author_evaluation/per_frame_metrics.jsonl`，天气移除目标记录保存在 `author_evaluation/removed_gt_metrics.jsonl`。

### 52.6 论文公开数字的并列记录

包内 `paper_reference` 记录的来源为论文v5 Table 5：https://arxiv.org/html/2411.08402v5#S5.T5；官方仓库来源为 https://github.com/ylwhxht/V2X-R。

包内来源说明：Table 5未明确标注validation或testing，相关Table 7/8标注testing；论文表的指标名称为3D mAP，本次冻结实现使用俯视框交并比和VOC2010 AP。本次数据为829帧validation，未读取test。下表按原报告并列记录公开数字和本次数值；LiDAR+4D Radar+MDD不在本次模型列表内。

| 天气 | 论文纯LiDAR AP70 | 本次作者权重O AP70 | 本次−论文（百分点） |
|---|---:|---:|---:|
| clean | 0.7609 | 0.6173229034 | -14.357710 |
| fog | 0.6152 | 0.5996286321 | -1.557137 |
| snow | 0.4554 | 0.3736474303 | -8.175257 |

### 52.7 本地文件核对范围

已完整读取包内28个文件，解析全部JSON及逐行JSON记录；包内可用的22项输出哈希引用核对一致。三个天气的逐帧数量、预测框/GT汇总、九组逐帧变化汇总和按完整精度AP计算的对照差值与报告一致。

包内包含预检、权重加载审计、完整评测汇总、协议、GT/预测清单、逐帧指标、移除目标指标、对照报告、九份逐帧差值及五份运行日志。运行日志包含Shapely交集计算警告，未包含Traceback。

本包未包含原始权重、单帧预测框缓存、C/D原始审计文件；也未包含完成记录所引用的 `smoke/summary.json`、`author_evaluation/contract.json`、三天气预测子清单和九份 `stat_*.json`。权重加载及历史C/D原始文件核验结果按包内服务器审计记录登记。

## 53. 加入Snow训练的T1/T2四组对照及E0复用结果（2026-10-24 归档）

### 53.1 结果来源与完成记录

本节记录加入雪数据训练的T1/T2四组对照，来源为同目录下另一份服务器结果包。§52保留作者固定权重评测的记录。本节仅整理已生成的设置、成绩和计数。

- 本地结果包：`logs/v2xr-gspr-clean-train-v2/gspr-e01-results-20261024.tar.gz`，大小2129958字节，共43个文件，全部已读取。
- 压缩包SHA256：`2d5dd4a1bdb1d96a6f6bdf15a96988b0be58ba4266be334ef65d3a31e9f8ae6b`。
- 服务器结果目录：`/data/cjm/datasets/logs/gspr-e01-formal-warmup-v1/`。
- `completion.json`记录 `status=complete`、`scientific_completion=true`、每组正式训练20轮；四组均有完整20轮历史和最终评测结果。
- 运行身份：`ccd2c876316568d84c9589a7362589c55dc565b33a1271e89a11e8f826adfe81`。
- E0从 `/data/cjm/datasets/logs/gspr-e01-v1/E0` 复用，`reuse_existing=true`。新增训练模型为T1-C、T1-D、T2-C、T2-D；T0-C、T0-D为§50的历史Clean+Fog模型。
- C为纯AttFuse，D为同配方GSPR-AttFuse。四组先完成正式第1—2轮，全部通过预运行检查后继续第3—20轮；前2轮计入总轮数，原BN和原评测设置保留。

### 53.2 天气训练比例、训练设置与选中权重

天气按每辆协作车（CAV）的每次点云读取独立抽取，以下为抽样概率；实际读取次数另列。

| 配方 | Clean | Fog | Snow | 模型 |
|---|---:|---:|---:|---|
| T0（历史对照） | 4/7 | 3/7 | 0 | T0-C、T0-D |
| T1 | 3/7 | 3/7 | 1/7 | T1-C、T1-D |
| T2 | 3/7 | 2/7 | 2/7 | T2-C、T2-D |

- 训练集8083个协作帧；批次大小8，丢弃不足一批的末尾3帧，每轮实际消耗8080帧、更新1010次。四组各20轮、20200次更新。
- 随机种子20261017，加载进程16；沿用官方Adam初始学习率0.002和原学习率调度。每轮使用完整829帧Clean官方O AP70选模，完整精度并列时选最早轮。
- 最终在Clean/Fog/Snow各829帧validation上评测，未读取正式test。O/O*/F定义沿用§52.1；官方GT数量分别6224/3673/5854，固定Clean GT均6224。
- 四组训练审计均记录完整状态恢复通过；C/D逐轮数据配对通过，D与对应C的公共初始142个张量检查为精确一致。
- 固定GT身份：`d5244c4a8fa78dc1d26df1a7ee1f4592c1dd24f91a3f9c575313a081bbeab776`。

| 模型 | 已完成轮数 | 选中轮次 | 选中Clean O AP70 | 权重SHA256 |
|---|---:|---:|---:|---|
| T1-C | 20 | 18 | 0.6640649544 | `da4bde3cb12e9daa40ccfdb31bd5f7cb47ca85c068c0181f9e727674701c7e54` |
| T1-D | 20 | 19 | 0.6918542150 | `8dac5a2dbb143aeaeb6feffe578b8459ea4e2a6a5de4c49e9247d3c4c4cf6e2f` |
| T2-C | 20 | 20 | 0.6910672479 | `740d673340371f7cad1ac4368813d3df452b3a1eed42c1b62e760a3befd9779f` |
| T2-D | 20 | 19 | 0.7089986244 | `66329b532a9a19f85a471f4261e5be0e0445db91ae59a0214106ed93e03c1547` |

20轮实际天气点云读取次数（C/D同配方一致）及训练GT累计数量：

| 配方 | Clean读取 | Fog读取 | Snow读取 | 训练GT累计数 |
|---|---:|---:|---:|---:|
| T1 | 250515 | 250968 | 83723 | 1335860 |
| T2 | 248933 | 165995 | 166201 | 1375800 |

### 53.3 六组模型的三天气绝对成绩

AP为0—1小数，表中按AP30 / AP50 / AP70排列，保留10位小数；完整精度见 `E1/E1_RESULTS.json`。

| 模型 | 协议 | Clean AP30 / AP50 / AP70 | Fog AP30 / AP50 / AP70 | Snow AP30 / AP50 / AP70 |
|---|---|---|---|---|
| T0-C | O | 0.8549079280 / 0.8215460712 / 0.6847206685 | 0.8755274093 / 0.8547829271 / 0.6750336026 | 0.6851941652 / 0.6394441468 / 0.4270725323 |
| T0-C | O* | 0.8544958580 / 0.8211500817 / 0.6843906297 | 0.8745739354 / 0.8538520445 / 0.6742984721 | 0.6806293253 / 0.6351840987 / 0.4242273276 |
| T0-C | F | 0.8544958580 / 0.8211500817 / 0.6843906297 | 0.5168412064 / 0.5044241768 / 0.3984052963 | 0.6411340785 / 0.5974241185 / 0.3990081580 |
| T0-D | O | 0.8553416529 / 0.8238005836 / 0.6985099588 | 0.8906360298 / 0.8730036652 / 0.7568556591 | 0.6239576654 / 0.5918006733 / 0.4286504340 |
| T0-D | O* | 0.8552042265 / 0.8236682249 / 0.6983977303 | 0.8894236203 / 0.8718152583 / 0.7558253628 | 0.6205468958 / 0.5885656850 / 0.4263072816 |
| T0-D | F | 0.8552042265 / 0.8236682249 / 0.6983977303 | 0.5250606735 / 0.5146669385 / 0.4462050645 | 0.5836570579 / 0.5535770437 / 0.4009644644 |
| T1-C | O | 0.8566335752 / 0.8224593203 / 0.6640649544 | 0.8992050389 / 0.8723124691 / 0.7017348780 | 0.8098423440 / 0.7590876079 / 0.5225384718 |
| T1-C | O* | 0.8564959412 / 0.8223271771 / 0.6639582602 | 0.8984705943 / 0.8715999895 / 0.7011617213 | 0.8094273240 / 0.7586985982 / 0.5222706865 |
| T1-C | F | 0.8564959412 / 0.8223271771 / 0.6639582602 | 0.5302189095 / 0.5143616262 / 0.4137800454 | 0.7613090544 / 0.7135960144 / 0.4912231039 |
| T1-D | O | 0.8625582246 / 0.8281419309 / 0.6918542150 | 0.9112427902 / 0.9001368294 / 0.7521518557 | 0.8181109732 / 0.7750922138 / 0.5748385945 |
| T1-D | O* | 0.8624196388 / 0.8280088747 / 0.6917430559 | 0.9100023291 / 0.8989114866 / 0.7511279627 | 0.8175519633 / 0.7745625984 / 0.5744458110 |
| T1-D | F | 0.8624196388 / 0.8280088747 / 0.6917430559 | 0.5370241894 / 0.5304790955 / 0.4432668713 | 0.7697370794 / 0.7287009961 / 0.5402965581 |
| T2-C | O | 0.8644845900 / 0.8390697831 / 0.6910672479 | 0.8888274562 / 0.8723015826 / 0.7037956551 | 0.8091964056 / 0.7674325101 / 0.5723539788 |
| T2-C | O* | 0.8640679040 / 0.8386653471 / 0.6907341500 | 0.8883434772 / 0.8718266021 / 0.7034124285 | 0.8090581759 / 0.7673014146 / 0.5722562074 |
| T2-C | F | 0.8640679040 / 0.8386653471 / 0.6907341500 | 0.5242425437 / 0.5144953582 / 0.4151082664 | 0.7611499145 / 0.7216874167 / 0.5382371205 |
| T2-D | O | 0.8585826853 / 0.8291529836 / 0.7089986244 | 0.8688853965 / 0.8546812574 / 0.7399780011 | 0.8149218717 / 0.7714448743 / 0.5937454201 |
| T2-D | O* | 0.8584447382 / 0.8290197649 / 0.7088847107 | 0.8684122763 / 0.8542158715 / 0.7395750727 | 0.8143650409 / 0.7709177510 / 0.5933397177 |
| T2-D | F | 0.8584447382 / 0.8290197649 / 0.7088847107 | 0.5253537458 / 0.5052930027 / 0.4366144876 | 0.7715502114 / 0.7307027345 / 0.5585071398 |

### 53.4 同配方D−C及相对历史模型的差值

单位为百分点，按ΔAP30 / ΔAP50 / ΔAP70排列；由100×（左模型AP−右模型AP）计算，保留6位小数。

| 对照 | 协议 | Clean | Fog | Snow |
|---|---|---|---|---|
| T0-D − T0-C | O | +0.043372 / +0.225451 / +1.378929 | +1.510862 / +1.822074 / +8.182206 | -6.123650 / -4.764347 / +0.157790 |
| T0-D − T0-C | O* | +0.070837 / +0.251814 / +1.400710 | +1.484968 / +1.796321 / +8.152689 | -6.008243 / -4.661841 / +0.207995 |
| T0-D − T0-C | F | +0.070837 / +0.251814 / +1.400710 | +0.821947 / +1.024276 / +4.779977 | -5.747702 / -4.384707 / +0.195631 |
| T1-D − T1-C | O | +0.592465 / +0.568261 / +2.778926 | +1.203775 / +2.782436 / +5.041698 | +0.826863 / +1.600461 / +5.230012 |
| T1-D − T1-C | O* | +0.592370 / +0.568170 / +2.778480 | +1.153173 / +2.731150 / +4.996624 | +0.812464 / +1.586400 / +5.217512 |
| T1-D − T1-C | F | +0.592370 / +0.568170 / +2.778480 | +0.680528 / +1.611747 / +2.948683 | +0.842803 / +1.510498 / +4.907345 |
| T2-D − T2-C | O | -0.590190 / -0.991680 / +1.793138 | -1.994206 / -1.762033 / +3.618235 | +0.572547 / +0.401236 / +2.139144 |
| T2-D − T2-C | O* | -0.562317 / -0.964558 / +1.815056 | -1.993120 / -1.761073 / +3.616264 | +0.530687 / +0.361634 / +2.108351 |
| T2-D − T2-C | F | -0.562317 / -0.964558 / +1.815056 | +0.111120 / -0.920236 / +2.150622 | +1.040030 / +0.901532 / +2.027002 |
| T1-C − T0-C | O | +0.172565 / +0.091325 / -2.065571 | +2.367763 / +1.752954 / +2.670128 | +12.464818 / +11.964346 / +9.546594 |
| T1-C − T0-C | O* | +0.200008 / +0.117710 / -2.043237 | +2.389666 / +1.774795 / +2.686325 | +12.879800 / +12.351450 / +9.804336 |
| T1-C − T0-C | F | +0.200008 / +0.117710 / -2.043237 | +1.337770 / +0.993745 / +1.537475 | +12.017498 / +11.617190 / +9.221495 |
| T1-D − T0-D | O | +0.721657 / +0.434135 / -0.665574 | +2.060676 / +2.713316 / -0.470380 | +19.415331 / +18.329154 / +14.618816 |
| T1-D − T0-D | O* | +0.721541 / +0.434065 / -0.665467 | +2.057871 / +2.709623 / -0.469740 | +19.700507 / +18.599691 / +14.813853 |
| T1-D − T0-D | F | +0.721541 / +0.434065 / -0.665467 | +1.196352 / +1.581216 / -0.293819 | +18.608002 / +17.512395 / +13.933209 |
| T2-C − T0-C | O | +0.957666 / +1.752371 / +0.634658 | +1.330005 / +1.751866 / +2.876205 | +12.400224 / +12.798836 / +14.528145 |
| T2-C − T0-C | O* | +0.957205 / +1.751527 / +0.634352 | +1.376954 / +1.797456 / +2.911396 | +12.842885 / +13.211732 / +14.802888 |
| T2-C − T0-C | F | +0.957205 / +1.751527 / +0.634352 | +0.740134 / +1.007118 / +1.670297 | +12.001584 / +12.426330 / +13.922896 |
| T2-D − T0-D | O | +0.324103 / +0.535240 / +1.048867 | -2.175063 / -1.832241 / -1.687766 | +19.096421 / +17.964420 / +16.509499 |
| T2-D − T0-D | O* | +0.324051 / +0.535154 / +1.048698 | -2.101134 / -1.759939 / -1.625029 | +19.381815 / +18.235207 / +16.703244 |
| T2-D − T0-D | F | +0.324051 / +0.535154 / +1.048698 | +0.029307 / -0.937394 / -0.959058 | +18.789315 / +17.712569 / +15.754268 |

包内另记录（T1或T2的D−C）−（T0的D−C），其AP70差值如下，单位为百分点：

| 配方 | 协议 | Clean | Fog | Snow |
|---|---|---:|---:|---:|
| T1 | O | +1.399997 | -3.140508 | +5.072222 |
| T1 | O* | +1.377769 | -3.156065 | +5.009517 |
| T1 | F | +1.377769 | -1.831294 | +4.711715 |
| T2 | O | +0.414209 | -4.563971 | +1.981354 |
| T2 | O* | +0.414346 | -4.536425 | +1.900356 |
| T2 | F | +0.414346 | -2.629355 | +1.831371 |

### 53.5 预测、None帧及固定GT匹配计数

以下为各组在固定Clean GT（F协议）、框交并比≥0.7时的一对一匹配计数。每种天气均829帧、6224个固定目标。

| 模型 | 天气 | 预测框 | None帧 | 正确匹配 | 误检 | 漏检 |
|---|---|---:|---:|---:|---:|---:|
| T0-C | clean | 6836 | 32 | 5016 | 1820 | 1208 |
| T0-C | fog | 3879 | 111 | 3031 | 848 | 3193 |
| T0-C | snow | 5100 | 81 | 3370 | 1730 | 2854 |
| T0-D | clean | 6700 | 32 | 5036 | 1664 | 1188 |
| T0-D | fog | 3922 | 103 | 3212 | 710 | 3012 |
| T0-D | snow | 4549 | 70 | 3123 | 1426 | 3101 |
| T1-C | clean | 7000 | 31 | 4994 | 2006 | 1230 |
| T1-C | fog | 3877 | 102 | 3063 | 814 | 3161 |
| T1-C | snow | 6309 | 32 | 4054 | 2255 | 2170 |
| T1-D | clean | 6875 | 42 | 5030 | 1845 | 1194 |
| T1-D | fog | 3883 | 114 | 3165 | 718 | 3059 |
| T1-D | snow | 6026 | 49 | 4196 | 1830 | 2028 |
| T2-C | clean | 6665 | 45 | 5026 | 1639 | 1198 |
| T2-C | fog | 3786 | 109 | 3081 | 705 | 3143 |
| T2-C | snow | 6035 | 27 | 4200 | 1835 | 2024 |
| T2-D | clean | 6988 | 37 | 5117 | 1871 | 1107 |
| T2-D | fog | 3923 | 104 | 3200 | 723 | 3024 |
| T2-D | snow | 6193 | 40 | 4285 | 1908 | 1939 |

固定目标ID配对变化如下，均按框交并比≥0.7统计；每个对照、天气为829帧。

| 对照（左−右） | 天气 | 新增检出 | 不再检出 | 匹配净变化 | 匹配增加帧 | 匹配减少帧 | 匹配相同帧 |
|---|---|---:|---:|---:|---:|---:|---:|
| T1 D-C | clean | 278 | 242 | 36 | 154 | 127 | 548 |
| T1 D-C | fog | 247 | 145 | 102 | 137 | 77 | 615 |
| T1 D-C | snow | 469 | 327 | 142 | 240 | 157 | 432 |
| T2 D-C | clean | 307 | 216 | 91 | 188 | 117 | 524 |
| T2 D-C | fog | 222 | 103 | 119 | 137 | 52 | 640 |
| T2 D-C | snow | 401 | 316 | 85 | 210 | 151 | 468 |
| T1-C − T0-C | clean | 251 | 273 | -22 | 153 | 153 | 523 |
| T1-C − T0-C | fog | 212 | 180 | 32 | 111 | 86 | 632 |
| T1-C − T0-C | snow | 959 | 275 | 684 | 394 | 74 | 361 |
| T1-D − T0-D | clean | 234 | 240 | -6 | 127 | 130 | 572 |
| T1-D − T0-D | fog | 118 | 165 | -47 | 59 | 101 | 669 |
| T1-D − T0-D | snow | 1246 | 173 | 1073 | 473 | 41 | 315 |
| T2-C − T0-C | clean | 280 | 270 | 10 | 162 | 153 | 514 |
| T2-C − T0-C | fog | 211 | 161 | 50 | 120 | 92 | 617 |
| T2-C − T0-C | snow | 1044 | 214 | 830 | 442 | 54 | 333 |
| T2-D − T0-D | clean | 271 | 190 | 81 | 164 | 102 | 563 |
| T2-D − T0-D | fog | 106 | 118 | -12 | 62 | 70 | 697 |
| T2-D − T0-D | snow | 1321 | 159 | 1162 | 484 | 34 | 311 |

### 53.6 包内预设筛选记录

`E1_WEATHER_TRADEOFF.json`中的筛选量全部采用F协议AP70，单位为百分点。这里只登记文件中的阈值、实测数值和布尔字段。

| 配方 | Snow D相对T0-D（阈值≥2） | Snow D−C（阈值≥1） | Clean D相对T0-D（阈值≥−1） | Fog D相对T0-D（阈值≥−1） | all_passed |
|---|---:|---:|---:|---:|---|
| T1 | +13.933209 | +4.907345 | -0.665467 | -0.293819 | true |
| T2 | +15.754268 | +2.027002 | +1.048698 | -0.959058 | true |

### 53.7 同包E0复用数值及本地核对范围

E0在本轮为已有结果复用，无新增E0训练。其R0/R1/R2/R3方案定义保留在包内原审计文件中；下表只摘录各方案的F协议AP70及相对R0的差值。

| 方案 | Clean F AP70 | Fog F AP70 | Snow F AP70 | Clean相对R0 | Fog相对R0 | Snow相对R0 |
|---|---:|---:|---:|---:|---:|---:|
| R0 | 0.6983977303 | 0.4462050645 | 0.4009644644 | +0.000000 | +0.000000 | +0.000000 |
| R1 | 0.5601507925 | 0.2246063702 | 0.2016780488 | -13.824694 | -22.159869 | -19.928642 |
| R2 | 0.6242861601 | 0.2629966526 | 0.2410118919 | -7.411157 | -18.320841 | -15.995257 |
| R3 | 0.5962998530 | 0.2524020053 | 0.2255730736 | -10.209788 | -19.380306 | -17.539139 |

E0点云记录：Snow固定目标共6224，零真实点观测目标366；在共同ego坐标下，Clean占用单元4291667，Snow缺失对应Clean单元972613。

本地已读取压缩包全部43个文件，核对四组各20轮完整历史、选中轮次及权重身份、完成文件引用的四项根目录哈希、包内可用的16项输出哈希引用，以及全部63行模型差值。T0完整精度参考为原始字节归档 `gspr_e0_diagnosis/original_FINAL_RESULTS.json`，SHA256为 `13f6ac5de347144cde676270bac651a0e2af8b3a90d89f1a580bcc4f5f78f848`。

本包保存汇总、训练审计、目标分析、历史、选模记录和配置；未包含四组原始权重、完整恢复状态、逐帧预测缓存、逐轮独立提交文件及 `.stages/` 完成文件。对应服务器检查结果按包内报告登记。本地未运行训练、推理或新的科研实验。
