# 科研代码与实验上下文

更新于 2026-11-12。第 1–7 节和第 45–69 节保留原编号及详细记录。第 8–44 节的旧实验详见 [AI_CONTEXT_ARCHIVE.md](AI_CONTEXT_ARCHIVE.md)。归档为整理前的完整原文，旧编号仍可查。

## 当前状态与执行约束

- **当前研究主线为 GSPR 与 GGCA A1。** GSPR 给激光雷达点赋连续可靠性并在点柱编码前软加权（§6–7、§49–50）。A1 在原 GSPR 点柱编码上增加柱内近邻聚合（§56、§59）。V2X-R 正式对照中，A0 选第 19 轮，A1 选第 20 轮。A1 的 Clean/Fog/Snow O AP70 为 0.712182 / 0.787111 / 0.450783，F AP70 为 0.711838 / 0.464589 / 0.421884。
- **A1 直接相关工作。** A0–A3 正式结果及机制检查见 §58–63。E0 复用原 A1，E1/E2 各完成 20 轮并分别选中第 20/15 轮，正式评测和完整验证集诊断状态见 §64–66。早期 REPE 及其与 A0 的关系见 §55、§57。
- **V2X-R 双模态扩展。** LR0 是官方激光雷达与雷达 AttFuse 对照，20 轮完成，选中第 18 轮（§67）。LR-G 在激光雷达支路接入 GSPR，LR1 在该支路进一步接入 A1，定义和执行入口见 `v2xr_lr1/README.md`。R-A/R-B 的 F0、F1、F2、F3 各完成 20 轮和三天气正式评测，结果见 §69。POT-CF 独立融合对照 P1–P3 已完成训练、评测和消融，完整结果见 §68。
- **数据和评测依据。** V2X-R 官方复现、天气真值审核及数据提取见 §45–48。纯激光雷达训练、作者权重同集对照和加入雪数据的实验见 §49–53。SCOR-W 数据集和验证证据见 §54、§60。
- **实验执行。** 本地编写和静态检查，真实数据训练、推理和科研实验在服务器运行。长任务使用后台命令。所有新实验日志统一写入 `/data/new_disk/cjm_logs/` 下各自的实验子目录。冻结源码、权重和既有结果按原合同复用，新实验使用独立目录。服务器版本用源码哈希核验，存在源码清单时随整套源码同步。
- **评测口径。** 正式 test 集只用于最终评测。O 使用当天气真值并跳过无预测返回帧，O* 计入这些帧且视为零预测，F 使用固定 Clean 真值并计入这些帧。纯激光雷达与双模态的固定真值数量分别为 6,224 和 6,235。AP 默认用 0–1 小数，表格另有标注时按其单位。pp 为百分点。

## 1. 服务器与设备

服务器 `cjm@admin1`；项目 `/home/cjm/OpenCOOD-main/cjmnet`，父目录 `/home/cjm/OpenCOOD-main`。8 张 HIP HCU（0–7）；原分配中 7 固定占用，使用前核对当前空闲卡。

只设 `ROCR_VISIBLE_DEVICES=<物理卡号>`，unset `HIP_VISIBLE_DEVICES`、`CUDA_VISIBLE_DEVICES`，程序内使用 `torch.cuda` / `cuda:0`。同时设置多个可见设备变量会二次过滤，可能报 `No HIP GPUs are available`。

## 2. 软件与本地环境

服务器 Conda `opencood`，解释器 `/home/cjm/miniconda3/envs/opencood/bin/python`，Python 3.10、PyTorch 2.5.1、NumPy 1.26.4、Shapely 2.0.0、OpenCOOD。本地 Windows 仓库 `D:\Collaborative perception\mymodule\cjmnet`，不具备完整科研运行环境。

## 3. 数据与冻结前端

| 用途 | 路径与规模 |
|---|---|
| 官方 Clean | `/data/scd/datasets/opv2v_official_data_dumping/{train,validate,test}`；43/9/16 场景，train 6374 帧、validate 1980 帧、test 2170 帧 |
| SCOR-W-Fog v1完整数据 | `/data/cjm/datasets/scorw-fog-v1-coveragefix/{clean,fog}/{train,validate,test}`；43/9/16个scene、6374/1980/2170协作帧；Fog各19575/10478/5985份PCD；D0–D7完成，固定AttFuse验证结果见§54 |
| 固定正式天气 test | 新盘 `/data/new_disk/opv2v-w/{fog,rain,snow}/test`；原访问路径 `/data/cjm/datasets/opv2v-w/{fog,rain,snow}/test` 通过绑定挂载保留（2026-11-11） |
| 固定物理天气开发数据 | `/data/new_disk/opv2v-physics-fixed-v1/{fog,rain,snow,mixed}/{train,validate}`；原目录 `/data/cjm/datasets/opv2v-physics-fixed-v1` 通过绑定挂载保留（2026-11-11） |
| 原版模拟器生成的 OPV2V validate | `/data/new_disk/opv2v-reference-weather-v1/{clean,fog,rain,snow}/validate`；1980协作帧，每条件10478个车辆点云；生成及AttFuse评测完成，详见§44；原目录 `/data/cjm/datasets/opv2v-reference-weather-v1` 通过绑定挂载保留（2026-11-11） |
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

上述 `opv2v-physics-fixed-v1` 是旧近似天气协议；`opv2v-reference-weather-v1` 是此次接入公开原版模拟器后建立的新协议，二者的结果与训练数据不能混写。V2X-R 子集的 `payload/validate/<scene>/<cav>/` 保存所选原文件，`{clean,fog,snow}/validate/` 是供读取的配对链接视图；payload和链接目标都需保留。`/data/cjm/datasets/logs` 是既有实验的权重与结果目录，`/data/cjm/datasets/tmp` 是临时目录，均不作为新的原始数据集。

## 4. 存储与操作

所有新实验的日志统一写入 `/data/new_disk/cjm_logs/`，按实验建立独立子目录，启动时将标准输出和错误输出重定向到对应日志。大数据、权重和缓存放外部数据盘，不写代码仓库。启动前指定物理卡与 opencood解释器；长期任务后台运行并记录 RUN/PID/LOG。单个进程失败后先确认已有缓存与源码契约，再断点恢复，避免覆盖已完成结果。

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

### 7.1 OPV2V-W 与 V2X-R：GSPR 跨数据集结果对照（2026-10-29 汇总）

本节紧接早期OPV2V-W GSPR原始结果，对照后续V2X-R的Clean-only、混合天气训练和作者原始权重同集评测。**只在同一个数据集、同一训练配方和评测协议下计算GSPR相对普通AttFuse的增益；不同数据集的绝对AP不能直接相减，不代表已证明跨数据集泛化。** 全部AP70值为0–1小数，增益为百分点（pp）。

**A. OPV2V-W：早期带逐点天气监督的GSPR（本节原始实验，三次完整训练）。** GSPR先用Fog/Rain/Snow×三强度的天气派生逐点标签预训练，再联合检测训练；旧OPV2V-W正式test每条件2170协作帧。三次检测结果均值及样本标准差见上表，下表按AP70列出与§5的原始AttFuse对照。原Clean基线仅记录到约0.82，不计算精确Clean增益。

| 条件 | 原AttFuse AP70 | GSPR AP70（三次均值） | GSPR−AttFuse / pp |
|---|---:|---:|---:|
| Clean | ≈0.82（原记录舍入） | 0.831512 | 不精算（基线舍入） |
| Fog | 0.610343 | 0.639272 | +2.893 |
| Rain | 0.561979 | 0.645840 | +8.386 |
| Snow | 0.468392 | 0.514817 | +4.643 |

相对原始AttFuse，Fog/Rain/Snow分别约+2.893/+8.386/+4.643 pp，平均约+5.307 pp；原逐点预训练的noise AUROC和多seed结果见本节上文。**此实验使用逐点天气监督，不能当作V2X-R无逐点标签实验的直接复现。**

**B. V2X-R：无逐点天气标签、Clean+Fog混合训练的GSPR。** 训练集8083协作帧，作者源码按每辆CAV每次读取以Clean 4/7、Fog 3/7随机选择，20轮、batch8、seed20261017；每轮829帧Clean validation官方O AP70选checkpoint。C=同配方纯LiDAR AttFuse，D=同配方GSPR-AttFuse；Clean/Fog/Snow各829帧validation。O为官方天气GT和None跳帧，F为三天气固定Clean GT且None计零。

| 条件 | C（AttFuse）O AP70 | D（GSPR）O AP70 | D−C O / pp | D−C F / pp |
|---|---:|---:|---:|---:|
| Clean | 0.684721 | 0.698510 | +1.379 | +1.401 |
| Fog | 0.675034 | 0.756856 | +8.182 | +4.780 |
| Snow | 0.427073 | 0.428650 | +0.158 | +0.196 |

Fog在O下提升+8.182 pp、在固定GT的F下仍提升+4.780 pp，Clean也有正差值；Snow O AP70仅+0.158 pp，且Snow O AP30/AP50为−6.124/−4.764 pp，不能声称跨天气一致增强。相反，**§49 V2X-R Clean-only GSPR（同为无逐点标签）对比匹配AttFuse的Clean/Fog/Snow O AP70下降−4.420/−2.584/−0.967 pp**。此差异说明训练条件很关键，但两个训练方案不构成“只改变天气混合比例”的完整机制因果鉴别。

**C. V2X-R：与作者官方Clean+Fog训练的纯LiDAR AttFuse预训练权重做同集对照（§52，2026-10-24已完成）。** 作者权重为`AttFuse_LiDAR_train_w_fog.pth`，SHA256 `195a873a814cfeb9e392ea7e3b6ae12a142b009719194d3c0a5980cbb4d089f3`；记录严格加载142个张量。作者、C、D均在相同829帧、相同O/O*/F评测器下评测。

| 条件 | 作者权重 O AP70 | D（GSPR）O AP70 | D−作者 O / pp | D−作者 F / pp |
|---|---:|---:|---:|---:|
| Clean | 0.617323 | 0.698510 | +8.119 | +8.107 |
| Fog | 0.599629 | 0.756856 | +15.723 | +9.244 |
| Snow | 0.373647 | 0.428650 | +5.500 | +5.104 |

Fog F协议IoU≥0.7逐目标一对一匹配正确数：作者2866、C3031、D3212；D较作者+346、较C+181。该结果与Fog F AP70的正差值共同说明**优势不完全由天气GT过滤或None跳帧解释**；但并未证明是GSPR可靠性加权这一机制单独导致。相比作者权重的差值同时包含训练实现、模型选择等差异，论文方法消融必须优先引用**D−C匹配对照**。Fog官方GT为3673、固定Clean GT为6224；不能把O AP70直接理解为对原始6224个目标的召回率。

**D. 后续Snow混合训练补充（见§53）。** 同配方D−C的Clean/Fog/Snow O AP70增益：T1（Clean/Fog/Snow=3/7、3/7、1/7）分别+2.779/+5.042/+5.230 pp；T2（3/7、2/7、2/7）分别+1.793/+3.618/+2.139 pp。它们支持加入Snow训练后GSPR在Snow也有正向匹配增益，但更改训练比例也改变了基线，不能与T0直接混算。

**多数据集结论与限制。** 旧OPV2V-W采用天气逐点监督、多种子、正式天气test，Fog/Rain/Snow AP70都提升；V2X-R无逐点标签在Clean-only为负、Clean+Fog时Fog明显提升、混入Snow时Snow出现正收益。结果说明GSPR对训练分布和监督方式敏感，**不能断言同一模型已直接实现跨数据集泛化**。V2X-R现为单seed、829帧validation，尚无正式test和多seed确认；V2X-R不含Rain对应评测，不可虚构Rain结果。作者论文Table 5列出的3D mAP与本次BEV IoU/829帧validation不保证同一指标和split，不能直接宣称超越作者官方论文测试基准。

原始证据：OPV2V-W见§5–§7及§11；V2X-R Clean-only见§49；天气混合六模型详表/BN/FiLM见§50及`logs/train/FINAL_RESULTS.json`；作者权重、严格加载、逐帧与O/O*/F完整表见§52及`logs/v2xr-gspr-clean-train-v2/v2xr-author-fog-attfuse-val-v1-results/reports/MATCHED_COMPARISON.json`；Snow训练比例对照见§53。**此节仅将已有实验做跨数据集索引对照，不新增训练、修改历史结果或宣称独立重算服务器预测。**


### 7.2 OPV2V-W：GSPR＋A1 的三种子评测（仅用 Clean 检测监督）

本组实验对应海光服务器 `/data/cjm/datasets/logs/opv2v-gspr-a1-v2/`。先做 5 轮 GSPR 逐点预训练，再做 5 轮 GSPR 与 A1 联合检测训练。检测框监督来自 Clean，模拟 Fog、Rain、Snow 继续用于 GSPR 的逐点可靠性监督。三个种子均完成 Clean、Fog、Rain、Snow 测试，每种天气各 2170 帧。逐种子数值来自服务器 `FINAL_RESULTS.json` 的回显，完整精度以服务器原文件为准。

| 种子 | 条件 | AP30 | AP50 | AP70 |
|---|---|---:|---:|---:|
| 20260906 | Clean | 92.28% | 91.37% | 83.22% |
| 20260906 | Fog | 71.79% | 70.81% | 63.97% |
| 20260906 | Rain | 76.01% | 74.41% | 64.50% |
| 20260906 | Snow | 64.31% | 62.45% | 51.64% |
| 20260907 | Clean | 92.05% | 91.19% | 83.18% |
| 20260907 | Fog | 72.00% | 70.93% | 64.00% |
| 20260907 | Rain | 76.09% | 74.49% | 64.66% |
| 20260907 | Snow | 64.41% | 62.52% | 51.84% |
| 20260908 | Clean | 91.87% | 91.01% | 82.87% |
| 20260908 | Fog | 71.80% | 70.80% | 63.78% |
| 20260908 | Rain | 75.94% | 74.31% | 64.46% |
| 20260908 | Snow | 64.06% | 61.99% | 50.90% |

下表按回显的两位小数计算均值与样本标准差，差值为 GSPR＋A1 均值减去本节开头的历史 GSPR 均值，单位为百分点。

| 条件 | AP30 均值 ± 标准差（%） | AP50 均值 ± 标准差（%） | AP70 均值 ± 标准差（%） | ΔAP30 | ΔAP50 | ΔAP70 |
|---|---:|---:|---:|---:|---:|---:|
| Clean | 92.067 ± 0.206 | 91.190 ± 0.180 | 83.090 ± 0.192 | −0.018 | −0.030 | −0.061 |
| Fog | 71.863 ± 0.118 | 70.847 ± 0.072 | 63.917 ± 0.119 | +0.010 | +0.012 | −0.011 |
| Rain | 76.013 ± 0.075 | 74.403 ± 0.090 | 64.540 ± 0.106 | +0.008 | +0.036 | −0.044 |
| Snow | 64.260 ± 0.180 | 62.320 ± 0.288 | 51.460 ± 0.495 | +0.052 | +0.052 | −0.022 |

四种天气的 AP70 与历史 GSPR 接近，差距均小于 0.07 个百分点。Fog、Rain、Snow 等权平均 AP70 从 59.998% 到约 59.972%，下降约 0.03 个百分点。这次仅用 Clean 检测监督的实验未观察到 A1 带来额外的 AP70 提升。历史 GSPR 的首个种子使用了不同的随机种子设置，这些差值仅作历史比较，A1 的独立作用需用匹配对照判断。

## 第 8–44 节归档

这部分的原文与历史交叉引用见 [AI_CONTEXT_ARCHIVE.md](AI_CONTEXT_ARCHIVE.md)。

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


### 51.3 海光ROCm入口与最小同步包确认（2026-10-25）

用户要求在原海光服务器运行本次REPE。核对当前run_all.sh、auditlib.py、training.py及evaluation.py，BACKEND=rocm仍走原海光Python、HIP检查、原官方runtime及D−1A/B/C证据；CUDA适配没有删除该入口。默认BACKEND为cuda，所以海光必须显式BACKEND=rocm，不激活NVIDIA环境。

当前sources()还无条件记录v2xr_pointcloud_only工具源码，故海光需要同步完整gspr_repe/和v2xr_pointcloud_only/。使用原完整数据时POINTCLOUD_ONLY_ROOT设为空，不运行精简数据生成，不改变训练配方。旧v2xr_weather_l012、v2xr_gspr_clean、v2xr_train_attfuse、v2xr_protocol_audit、v2xr_official_repro、冻结GSPR及历史环境/数据/权重保留，不在更新包内覆盖。

新增delivery_tools/build_repe_rocm_update.py和verify_repe_rocm_update.py，生成无cjmnet外层的最小更新包：delivery/repe_rocm_20261025_100147/gspr_repe_rocm_update.zip，132228字节，SHA256=d402b43546643e947273bb3eac690d3d3ab76902e751c66de34884804811aa9d。包内35个文件，校验工具核对624个既有依赖源码及各包完整文件集合。本地通过AST、gspr_repe静态检查、所选更新内容和既有冻结依赖逐文件核对，以及ZIP CRC/文件集合/哈希核对；未执行Torch、真实数据、GPU或海光训练。

建议上传到/data/cjm/datasets/packages/gspr-repe-rocm-20261025-v1/，解压到/home/cjm/OpenCOOD-main/cjmnet，校验输出rocm_update_source_integrity_passed后以BACKEND=rocm GPU=<空闲物理卡> MODE=all RESUME=1启动。新输出根/data/cjm/datasets/logs/gspr-repe-rocm-20261025-v1，启动日志/data/cjm/datasets/logs/gspr_repe_rocm_20261025.launch.log。MODE=all依次执行第一阶段检查/smoke/验收及第二阶段，第一阶段技术失败自动停止。海光真实数据/环境及实验结果仍待用户执行回传。


### 51.4 海光更新包历史参考校验范围修正（2026-10-25）

用户在海光执行verify_repe_rocm_update.py时，仅报四个required_existing_sources差异：attfuse_gspr/README.md、gspr_communication/RESIDUAL.md、gspr_evidence/stage3_sources.json、gspr_supervision/README.md。该命令为静态源码校验，没有启动本次训练。

核对调用链：这四项来自v2xr_official_repro/frozen_baseline.json的旧归档引用；REPE运行的auditlib.sources→legacy.sources→common.verify_package/support.verify_sources只严格校验当前包快照和直接PINS，并不读取上述四项。原官方check_static仍有完整旧归档审计职责，保持原样。本次部署检查器错误地扩大了运行前置范围，未发现模型源码或训练配置差异的回传。

仅修改delivery_tools中的更新包构建器/校验器，增加四项明确历史参考名单；必须同时不属于当前直接PINS、不在交付files或当前包快照、且原expected哈希与旧frozen_baseline清单一致，才降为historical_reference_differences记录。差异报告保留expected/actual哈希和原因。模型代码、当前配置/协议、直接PINS、各包完整文件集合、frozen_baseline.json本身仍严格阻断。没有覆盖四项旧文件、没有重写旧哈希，也未修改gspr_repe源码/快照。

新增delivery_tools/test_repe_rocm_update.py，7项标准库模拟文件测试通过：正常通过、复现四项历史差异并证明不写文件、模型差异阻断、其他JSON/MD差异阻断、直接PINS禁止降级、旧清单本身严格检查、当前包多余文件阻断。重新打包及ZIP CRC/逐文件哈希通过；没有运行科研模型或真实数据。

提供三文件修复包delivery/repe_rocm_20261025_100514/gspr_repe_rocm_verifier_fix.zip，只替换项目根目录verify_repe_rocm_update.py、REPE_ROCM_UPDATE.json、REPE_ROCM_README.md。完整更新包同目录保留。用户可解压修复包后重新执行原校验；通过仅表示本次运行源码身份校验通过，不替代数据/环境/GPU/训练验收。


### 51.5 海光第一阶段结构测试非有限值失败与数值风险修复（2026-10-25）

用户回传server_tests.py在一次optimizer.step后的eval前向触发冻结v2xr_gspr_clean/integration.py的ValueError('Nonfinite GSPR output')。此前REUSE COMPLETE fixture是注入故障恢复测试的日志，不能当作整阶段通过。回传截断日志未显示当前变体，尚不能确认服务器唯一根因，也不能将错误归因于ROCm或BN。

审查发现gspr_repe/modules.py的加权空间离散统计使用sqrt(max(variance,0))；单点/重合点variance=0时反向导数无穷，零上游梯度亦可能生成NaN，沿可靠性路径污染GSPR。原结构测试只检查新增repe_*模块梯度，未检查GSPR/主干全部梯度就step，存在漏检。现改为sqrt(max(variance,0)+1e-6)-1e-3，以1毫米尺度平滑，保持零方差输出0、导数有界；空可信点范围在相减及范数前移除inf哨兵。冻结GSPR和官方代码不修改，原非有限值异常保留，训练批次/学习率/数据不变。

server_tests新增单点/重合点/无可信点/零上游梯度/零初始化观测头反向回归；在optimizer前检查全模型每项已产生的梯度，并在前向/step前后检查全部浮点参数和buffer。新增numerical_progress.json记录变体和阶段，避免下一次截断日志无法定位。远端数值测试尚未执行；本地仅AST、3项标准库平滑公式/检查顺序测试、9项迁移静态测试、7项部署校验器模拟测试，以及包快照/依赖/CRC/哈希检查通过，没有本地Torch或科研实验。

新完整更新包delivery/repe_rocm_20261025_102734/gspr_repe_rocm_update.zip，136223字节，SHA256=566b80ddc33f707dbaf328c277908dce081558ebc7feb7f68af38bc0794b2f4f。包含36个文件及完整新版gspr_repe快照，保留上一轮四项历史参考差异的只记录策略。更新后需使用新输出/data/cjm/datasets/logs/gspr-repe-rocm-20261025-v2，保留v1失败目录，不手改旧契约或恢复旧源码身份。第一阶段仍未通过，第二阶段不得提前运行。


### 51.6 v2目录回传仍为旧结构测试栈，先核实部署身份（2026-10-25）

用户回传v2/self_test再次Nonfinite GSPR output，堆栈仍是server_tests.py第90行expected=model(copy.deepcopy(batch))，且没有新增STRUCTURE TEST变体输出。核对本地102734数值修复zip：该语句位于第147行，第90行为def run；zip确含全模型梯度检查、变体输出、平滑方差公式。zip SHA256仍为566b80ddc33f707dbaf328c277908dce081558ebc7feb7f68af38bc0794b2f4f。因此该次进程执行旧版测试代码；仅凭该日志不能判断新数值修复是否有效，也不能确认是旧包未解压还是更新前进程/日志。

先请求远端只读源码/压缩包哈希核对，不继续重复全量数据扫描。当前修复版server_tests.py SHA256=00c0f3410e470b41840515489240281deda5a4a85c417258c1f198445aabfbff；modules.py SHA256=20b7661f7d88b7ee31918d1b41ab701325681a0fa2369f6d33259f7b46f5eb23。保留v1/v2失败产物和原异常，未新增实验结果。scheduler.step(epoch)的提示是官方配方保留的弃用警告，本次不据此改学习率调度。


### 51.7 NVIDIA完整重打包与新精简数据路径复核（2026-10-25）

用户要求将历次修复合入新的完整远端包，并澄清其所说REMOTE_CONTEXT.md实际指AI_REMOTE.md。随后用户更新该文件：精简数据已在海光生成、6组样本输入/随机状态一致检查通过且用户确认已传至NVIDIA；租用服务器原三个完整数据副本已删除。当前唯一数据总根为/data/cjm/datasets/v2xr-pointcloud-only-v1，其下保留三个同名数据集子目录、POINTCLOUD_ONLY.json、EQUIVALENCE.json。NVIDIA目标端全文件内容/CUDA/训练验收仍无通过记录，不能按旧完整目录启动。

复核现有run_all.sh、camera_metadata.py、cuda_runtime.py：显式POINTCLOUD_ONLY_ROOT会切换训练/官方验证/配对验证子目录，按元数据读取真实图片尺寸，原图像素缺失仅在原尺寸/哈希记录匹配时允许；保持雷达、相机视野筛选、冻结源码和训练配方。EQUIVALENCE.json必须匹配适配器源码及元数据身份，不修改旧清单绕过检查。

本轮只更新delivery_tools/build_repe_cuda_source_bundle.py的打包/发布配置，不修改正在海光运行的gspr_repe科研源码或快照。新完整包沿用flat格式，并使用带发布时间的唯一文件名，含完整冻结依赖、数值修复、全模型检查、精简数据工具和数据校验辅助工具。新增包内根目录run_remote.sh部署入口，按AI_REMOTE显式启用精简根并覆盖可能残留的旧TRAIN_ROOT/VIEW_ROOT/PAIRED_ROOT，先源码校验、目标端全部精简文件哈希校验，再调用原两阶段run_all.sh；MODE=all默认串行执行，技术失败停止。新的CUDA_INPUT_ROOT和OUTPUT_ROOT带发布标识，保留旧失败/预检目录。

新增RELEASE_IDENTITY.json记录关键源码哈希、明确数值修复存在、当前数据模式及远端尚未验证的状态。本地仅做静态、模拟文件测试和打包CRC/逐文件哈希，未读取真实数据、未导入科研模型、未运行GPU/训练；结果以新包外SUMMARY.json与STATIC_CHECKS.json记录为准。用户最新海光状态为smoke_eval_B3运行中，不据此登记正式训练或全阶段完成。


2026-10-25 15:23 CUDA远端部署更新：用户授权停止旧227/1935，已正常退出。最终修复版REPE_CUDA_20261025_152140在独立源码/workspace/gspr_source/cjmnet_repe_cuda_20261025_152140后台启动，PID6872；当前仅全量数据校验4000/163556，尚未训练。具体源码、结果、输入审计及日志见AI_REMOTE.md第14节、delivery/repe_cuda_20261025_152140/REMOTE_DEPLOYMENT.json。训练诊断被清空问题已修复，Python3.10远端检查3项通过；不改写旧结果/源码身份。


## 15. CUDA实验已按用户要求停止，恢复点已验证（2026-10-26 16:13，Asia/Shanghai）

用户明确要求停止当前CUDA实验并保留进度。核对后向主进程6872发送SIGTERM，确认6872、训练子进程68645及所属16个数据加载进程全部退出。未删除源码、数据或结果。

B0最近完整保存轮次为16/20，下一次从第17轮重跑。已逐字节核对最近恢复状态和权重哈希通过。恢复文件：`/data/cjm/datasets/logs/gspr-repe-cu124-slim-20261025_152140/B0/resume_states/16.pth`，SHA256=`be30ca88be299727d6f4592783830308b96457cb7fcdaf5226f0f484481b0168`。未完成的第17轮不保证保存批次进度。

远端停止记录：`/data/cjm/datasets/logs/gspr-repe-cu124-slim-20261025_152140/USER_STOP_RECORD.json`。原流水线的STOP/failure日志保留，SIGTERM在旧实现中可能记录为technical_failure；本次原因是用户主动停止，不是训练数值失败。

后续恢复使用同一源码`/workspace/gspr_source/cjmnet_repe_cuda_20261025_152140/run_remote.sh`，保持MODE=all RESUME=1及原OUTPUT_ROOT/CUDA_INPUT_ROOT默认值，后台启动并追加原launch.log。不得用新包/新源码直接续跑该目录。当前未重新启动。


## 54. SCOR-W-Fog v1全量构建与固定AttFuse验证结果（2026-10-27复核）

### 54.1 完成记录、路径与数据规模

服务器最终状态`DATASET_COMPLETE`，D0–D7八个阶段均记录`passed`。最终状态文件修改时间2026-10-27 00:56:46（Asia/Shanghai）。本次为全量MODE=full；源码目录`scorw_dataset/`，修订`scorw-dataset-v1.1-timing-coverage`。

- 原始输入：`/data/scd/datasets/opv2v_official_data_dumping/{train,validate,test}`。
- 数据输出：`/data/cjm/datasets/scorw-fog-v1-coveragefix`；`clean/`、`fog/`保留原始split/scene/CAV/timestamp结构。
- 日志：`/data/cjm/datasets/logs/scorw-fog-v1-coveragefix/full.log`。
- 来源清单：`SOURCE_INVENTORY.json`及`stages/D0/{TRAIN,VALIDATE,TEST}_SOURCE.json`；输出清单：`{TRAIN,VALIDATE,TEST}_MANIFEST.json`及`manifests/`；逐点来源：`records/`。

| 划分 | scene数 | 协作帧 | 每条件PCD数 | Clean/Fog每条件点数 | 天气回波替换数 |
|---|---:|---:|---:|---:|---:|
| train | 43 | 6374 | 19575 | 1119783879 | 89349395 |
| validate | 9 | 1980 | 10478 | 582505715 | 53429144 |
| test | 16 | 2170 | 5985 | 342258299 | 27248719 |

合计68个scene、10524个协作帧、36038份Fog PCD。输入和Fog输出均为2044547893个回波；无返回计数0、空点云0。scene跨split重复记录为false。

Clean为原始数据符号链接，Fog PCD为实际文件；Fog YAML及其他元数据链接原来源，读取路径及SHA256检查通过。PCD字段为x,y,z,rgb，OpenCOOD读取RGB红通道归一化强度。位姿、标注和object ID字节不因天气改写，数据视图依赖原始数据及元数据链接目标。

### 54.2 天气场、接收器与来源统计

每scene一个固定共享世界场，全部CAV及全部时间戳共用；68个scene的seed和场哈希各为68个唯一值。主seed=20261023，按split及scene身份派生场seed。场算法`bounded-random-fourier-tanh-v1`，mean_alpha=0.06、length_m=25、contrast=0.8、modes=32。世界边界为全scene原始XYZ与传感器原点包围盒加2m边距。正式生成覆盖全部原始点，未按通信范围或GT框抽样。

接收器沿用R3的`single-scattering-finite-pulse-strongest-total-peak-v1`：晴天已有回波为条件、单次散射、有限脉冲，硬/软波形相加后选择最强峰。硬回波积分步长0.5m，响应步长0.1m，beta/alpha=0.015355177231985366，gamma=1e-6，脉冲半宽20ns，重叠区间0.9–1.0m，强度量化间隔1/255。阈值状态`uncalibrated_approximation`，未使用标定的硬件接收阈值。

下表记录模拟器选峰及软/硬分量规则定义的来源类型；逐点车辆语义标注未提供：

| 来源/统计项 | 数量 | 占全部输入回波 |
|---|---:|---:|
| 原始硬回波返回 | 1872795462 | 91.59949094% |
| 模拟天气返回 | 170027258 | 8.31612987% |
| 来源不确定unknown | 1725173 | 0.08437919% |
| 无返回 | 0 | 0.00000000% |
| 光学厚度大于1 | 461243189 | 22.55966664% |

近场未建模标记共27625条。按最终输出距离统计，0–3m点数由60175427增加至230202685。

全数据按原始Clean光路距离分组如下；强度比例为该原始距离组输出信号之和/原始信号之和，包含替换后的天气信号。

| 原始距离/m | 原始回波数 | 天气替换率 | 输出/原始强度总和 |
|---|---:|---:|---:|
| 0–3 | 60175427 | 0.00000000% | 75.85166033% |
| 3–10 | 1183251099 | 0.00000000% | 49.72041317% |
| 10–20 | 452653278 | 0.00000000% | 25.82107612% |
| 20–40 | 208144565 | 24.53846297% | 10.15069192% |
| 40–60 | 71762069 | 74.06362545% | 4.62679062% |
| 60–80 | 34105225 | 92.96694861% | 5.60162517% |
| 80–120 | 34452218 | 98.95325752% | 10.58786409% |
| 120–200 | 4012 | 100.00000000% | 15.42650523% |

各scene天气替换率范围：Train 1.98127610%–16.07052087%，Validate 2.82542308%–11.35669974%，Test 3.10294884%–15.75276692%。完整68个scene统计保存在服务器DATASET_QA.json及本地ANALYSIS_SUMMARY.json。

### 54.3 数值与读取检查记录

9项工程验收通过。原逐点接收器与批量查询比较覆盖3组合成条件各36条光路、2份Train抽样各147条光路，共402条输入；分别比较batch=1/8/16。报告中光学厚度、硬信号、软信号及天气XYZI最大绝对误差均为0，来源与有效性一致，最终PCD SHA256一致。零雾回退、确定性场查询、空PCD实际读取检查通过。

Hahner硬回波ADC、软回波查表及替换/无返回重放检查通过；新均匀软响应与独立积分、波形非负及面积、常数/非均匀场空间收敛检查通过。40条合成回归中，原版记录19个替换、8个零码无返回；新引擎记录12个替换、15个零码无返回。两者选峰规则不同，回归未要求最终返回逐点相同。

Clean与Fog各36038份PCD均执行实际OpenCOOD读入、体素预处理和collation。Train/Validate各scene固定首末帧另执行原dataset取样及frame collation；Test执行完整索引和点云预处理，未解释Test目标标签。

IoU审计记录206943次调用、8873887次结果有限的几何相交警告；非法几何0、非法或非有限结果0。geometry_repairs=false、values_changed=false，原计算值保留。

### 54.4 固定AttFuse验证集检测结果

模型为`attfuse_code.point_pillar_intermediate.PointPillarIntermediate`；固定权重`/data/cjm/datasets/logs/attfuse_opv2v_20260903_200923/net_epoch13.pth`，checkpoint SHA256=`06e953cc15391e5b7569e825d33112f79b1e8c15a70b8914a489bb437cb81aba`，配置SHA256=`10d9d20a607a08d70c68b67e524c188014559412a44176545ee43384e7ee4a92`。本轮training=false。

完整Validate 1980个协作帧，同一批54621个按帧计数的标注目标实例；Clean/Fog固定同一Clean GT。沿用R3同步project-first适配器、70m通信范围、冻结模型CAV上限、120m目标范围、俯视平面框匹配及全局分数排序。GT_RANGE记录为[-140,-40,-3,140,40,1]；未以天气点数决定GT可见性。

AP按0–1记录，下降为100×（Clean−Fog），单位pp：

| 指标 | Clean | Fog | 下降/pp |
|---|---:|---:|---:|
| AP30 | 0.8841046710578971 | 0.6752106844283291 | 20.88939866 |
| AP50 | 0.8716086110190255 | 0.6648634939292744 | 20.67451171 |
| AP70 | 0.8244272274894610 | 0.6222479003669350 | 20.21793271 |

| 框重合阈值 | 条件 | 正确检出 | 误检 | 漏检 | 标注实例 |
|---|---|---:|---:|---:|---:|
| 0.3 | clean | 50944 | 7396 | 3677 | 54621 |
| 0.3 | fog | 37792 | 6527 | 16829 | 54621 |
| 0.5 | clean | 50269 | 8071 | 4352 | 54621 |
| 0.5 | fog | 37182 | 7137 | 17439 | 54621 |
| 0.7 | clean | 48045 | 10295 | 6576 | 54621 |
| 0.7 | fog | 34988 | 9331 | 19633 | 54621 |

阈值0.7下，漏检净增加13057，误检计数变化-964；召回率由87.96067447%变为64.05594918%。这些是汇总计数差，未登记为逐目标转移事件数。

检测报告status=passed，标签UNCALIBRATED_EXPLORATORY；fixed_same_Clean_GT=true、native_dataset_GT_filter_used=false、Test_AP_computed=false、AP_decrease_required=false。本轮没有单车对照或逐scene AP汇总；验证帧最多的两个scene合计1205/1980帧（60.85858586%）。

### 54.5 时间、存储与证据范围

运行合同文件修改时间2026-10-23 14:53:42，最终门槛文件修改时间2026-10-27 00:56:46（Asia/Shanghai），跨度82.05119885小时。该跨度包含各阶段及可能的暂停/恢复，未登记为纯CPU计算小时。

2026-10-27只读du -s -x -B1记录输出目录实占82424754176字节（82.424754176 GB），不跟随Clean符号链接，不包含外部原始数据或冻结权重。D0运行前预算为20.66826788天、648.04509818 CPU小时、261702130304字节输出空间；这些为预算估计。

2026-10-27通过已核验SSH读取完成报告，本地18份JSON副本与服务器发布清单SHA256一致；协议—合同—最终门槛身份及D7质量报告回执核对一致。本轮复核未重新遍历全量PCD或逐点sidecar，完整数据检查状态来自D0–D7保存的审计。未重跑天气生成、检测或训练。

- 服务器核心报告：DATASET_QA.json、VALIDATION_GATES.json、PHYSICS_AUDIT.json、COMPATIBILITY_AUDIT.json、VALIDATION_ATTFUSE_RESULTS.json、REPORT_MANIFEST.json。
- 小报告包：`/data/cjm/datasets/scorw-fog-v1-coveragefix/SMALL_REPORTS.tar.gz`，目录记录107211字节。
- 本地JSON副本：`logs/scorw-fog-v1-review-20261027/verified_reports/`；下载校验`REPORT_DOWNLOAD_AUDIT.json`；汇总数据`ANALYSIS_SUMMARY.json`。
- DATASET_QA.json SHA256=`3ea7302f491484d6a5778b66f3779cf3eb3cacd935123dead193f0528006c187`。
- VALIDATION_ATTFUSE_RESULTS.json SHA256=`61d3ef1dc40140a03689a30b3bdff7adfe215d7ebd9ac238ba34f5f08bed42b7`。
- VALIDATION_GATES.json SHA256=`b716c91513b9a6c143a4017fe2bff5af1843d897cecf402e44d1b13f13be4347`。

### 54.6 报告中的物理状态与未记录项

REAL_WORLD_VALIDATION_MISSING=true，真实天气标定未完成。当前模型以晴天已有返回为条件，未恢复逐束发射时间、真实接收波形或未观测遮挡物体；接收阈值状态未标定。高光学厚度路径使用同一单次散射近似。来源标签按模拟器分量及选峰规则记录。本次全量数据无返回计数为0，DATASET_QA.zero_no_return_is_allowed=true。未登记真实雾天分布复现、Test AP、逐scene检测AP、单车/协作因果对照或新模型训练结果。


## 55. 当前 GSPR REPE 的 B0–B4 实验方法（2026-10-27，按实现核对）

本节只记录方法，不记录本轮 AP、耗时、完成轮次或效果结论。它是§51方案的具体实现说明；旧章节中的历史状态不替代本节的方法定义。依据为 `gspr_repe/models.py`、`modules.py`、`data.py`、`training.py`、`checkpoint.py`、`evaluation.py`、`cuda_evaluation.py`、`orchestrator.py`、`settings.json`，以及冻结依赖 `v2xr_gspr_clean/integration.py` 和官方 V2XR_attfuse.yaml。B0–B4 是本实验的模型组号，不与其他历史实验中的同名编号混用。

### 55.1 总体任务、共同主干及模块位置

任务为多车协同的纯激光雷达三维目标检测。每车点云先按官方方式处理及投影，再经过 GSPR 点可靠性估计、柱体编码、俯视网格特征提取、官方注意力融合和检测头。柱体是地面网格向竖直方向延伸形成的小格子；俯视网格特征也称 BEV，是把柱体特征放回地面二维网格后的表示。

共同检测器为官方 PointPillarIntermediate / AttFuse。原有 GSPR 从柱体内 XYZI（空间坐标和反射强度）、点数及柱体坐标产生逐点可靠性 r、不确定性 u、证据和有效点掩码。A 位于柱体特征汇总处；B/C 位于 scatter（将柱体特征放回俯视网格）之后、官方后续主干及车间注意力融合之前。新增邻域操作在单车内执行，车间融合仍交给官方网络。

五组保留相同官方检测头、后处理、标注生成和评测路径，不增加图片、雷达输入或新的原生算子。保留的“冻结源码”是指历史依赖文件不修改；本实验共同主干和 GSPR 的参数仍参与训练，并非冻结权重。

### 55.2 五组模型的准确组成

| 组别 | GSPR及官方检测器 | A：柱内自适应聚合 | B：观测状态估计 | 普通/约束邻域增强 | 设计用途 |
|---|---|---|---|---|---|
| B0 | 保留 | 无 | 无 | 无 | 本轮共同基准，不能用纯 AttFuse 或历史 Clean-only GSPR 的成绩代替 |
| B1 | 保留 | 有 | 无 | 无 | 比较柱内聚合整条实现路径的影响 |
| B2 | 保留 | 无 | 无 | 普通八邻域增强，来源只受有点掩码限制 | 检查普通局部增强这一对照 |
| B3 | 保留 | 有 | 有 | 普通八邻域增强，加目标状态门控；来源只受有点掩码限制 | 完整来源约束的匹配对照 |
| B4 | 保留 | 有 | 有 | 与 B3 相同的邻域结构和目标门控，进一步用邻居质量、支持度、未知度限制来源 | 完整方案 |

实现开关 `FLAGS` 的顺序为 A/B/C：B0=(False,False,False)，B1=(True,False,False)，B2=(False,False,True)，B3/B4=(True,True,True)。这里 C 开关表示启用 Enhancement 类，不代表 B2/B3 已启用 B4 的可信来源约束。B3 和 B4 的增强模式由 variant 单独区分，不能只看三项布尔开关。

### 55.3 B0：原有 GSPR 接入方式

GSPR 先对原始柱体点计算可靠性。随后在官方第一层柱体特征网络 PFN 前，将官方装饰点特征乘以 r，再走原有线性映射、归一化、激活和汇总路径。“装饰点特征”是官方编码器为每点拼接的坐标、相对位置等特征。GSPR 权重不删点，不修改官方记录的点数，也不据此改写真实目标标注。

B0 不实例化 A/B/C，代表当前 Clean/Fog 混合训练配方下的原有 GSPR 检测器。它与历史只用 Clean 训练的 GSPR 不是同一实验配方。

### 55.4 A：可靠性感知柱内聚合（B1/B3/B4）

1. 用真实点数建立有效位置掩码，补齐槽位置零。替换原 PFN 汇总路径：只将有效点送入官方线性层、归一化和 ReLU，再对有效点做最大值汇总得到 h_max；补齐位置不参与归一化及最大值。训练时若仅一个有效点，使用保存的归一化统计量，避免单点批归一化错误。
2. 在 GSPR 后的柱体点坐标上，用有效点的普通均值求中心。对每点构造6维打分输入：相对中心的三维坐标、相对距离、可靠性 r、不确定性 u。打分网络为 Linear(6,16) → SiLU → Linear(16,1)。
3. 打分加上 log(max(r,1e-6))，将可靠性作为一次显式先验。屏蔽无效槽，沿柱内点做 softmax，再乘有效掩码并重新归一化，得到每点权重 a_i。
4. 对装饰点特征作独立线性映射到64维，得到 value_i，计算 h_weighted=sum(a_i*value_i)。该 value 分支不额外显式乘一次可靠性；A 启用时不再调用 B0 的“PFN前装饰特征乘r”钩子路径。
5. 输出 h_A=h_max+sigmoid(g)*(h_weighted-h_max)，g 是可学习标量，初值为−5，使初始混合幅度较小。

因此 B1−B0 检查的是“A的完整编码实现”，同时包含有效点归一化/最大池化处理和自适应聚合，不能将差异全部归结为某一个点权重公式。关闭 A 时走原有 GSPR/PFN 路径。

### 55.5 柱体统计量与 B 的输入

B/C 启用时，每个有点柱体计算7类统计，并按柱体坐标写入俯视网格：有效点数 n；有效可靠性和 m=sum(r)；平均可靠性 m/n；平均不确定性；可信点包围范围的对角线长度；可靠性加权空间离散程度；可靠性加权中心的水平距离。

可信点条件为有效且 r≥0.5、u<0.75。没有可信点时将范围上下界安全置零后再计算长度。加权中心按 max(m,1e-6) 归一化；离散程度使用 sqrt(max(spread,0)+1e-6)−1e-3，保持零离散时数值为零，并限制反向梯度。无点网格保持全零，occupied 表示 n>0。

编码时对点数/可靠性和、范围/离散程度用 log1p；平均可靠性和平均不确定性保留原值，距离除以150。64通道网格特征先通过1×1卷积压到8通道，与7通道统计拼成15通道。

### 55.6 B：质量、支持度及未知状态估计（B3/B4）

网络为 Conv(15,16,3×3,padding=1) → SiLU → Conv(16,4,1×1)，四个输出经 sigmoid 分别得到 q、s、e、u_B。q 表示质量，s 表示观测支持程度，e 是可学习的候选证据，u_B 表示未知程度。u_B 与原 GSPR 的逐点不确定性 u 是不同变量。B 最后一层权重和偏置置零，原始四头初值均为0.5。

无点位置强制 q=0、u_B=1。候选增强门控定义为 c=(1−s)*e*(1−u_B)*occupied：已有一定观测、支持不足且不太未知的位置才允许增强。完全空网格 c=0，本轮不声称实现完全无点区域或缺失目标的补全。

B 的监督为两个弱监督项，使用二元交叉熵：

- 质量目标：有点位置的平均 GSPR 可靠性 m/n；作为标签时停止梯度，避免该标签分支反向改变自身目标。
- 支持度默认目标：按距离0–30、30–60、60–100、100米以上分段，以当前批次该段有点位置的 m 中位数（至少为1）作参考，计算 clip(m/reference,0,1)，标签停止梯度。
- 有可对齐的训练 Clean 参考时，以同格 Clean 原始点数 n_clean 为分母，计算 clip(m/max(n_clean,1),0,1)。只在 Clean 至少2点且当前天气有点的位置监督；若该有效配对掩码存在，用此项替换默认支持度损失，否则使用距离段弱代理。

e 和 u_B 没有单独的真实标签或专用监督项，其作用通过增强路径及检测损失学习。q/s 是弱代理，不是物理场景的真实质量、缺失比例或完整目标占据标签；监督标签停止梯度不等于冻结 GSPR，检测及增强路径仍可传回梯度。

### 55.7 C：八邻域残差增强及 B2/B3/B4 的区别

对每车64通道俯视特征 x，用无偏置1×1卷积分别生成8通道 key 和64通道 value；key 用来比较相邻网格特征是否相似。另用1×1卷积生成目标增强门，权重初值0、偏置−5。

只访问周围8个邻居，不含中心自身；使用补零与切片避免边界环绕，不跨车辆交换。邻居 j 到目标 i 的未归一化贡献为：

`w_ij = exp(cosine(key_i,key_j)) * source_j / (1+dx²+dy²)`。

其中余弦相似度计算的 eps=1e-6。累加 N_i=sum(w_ij*value_j)、D_i=sum(w_ij)，构造消息 N_i/max(D_i,1e-6)。目标门由 sigmoid(gate(x_i)) 产生，且 D_i≤1e-6 时关闭更新。最终输出为 x_i 加上目标门乘邻居消息。

| 组别 | source_j | 是否再乘候选目标门 c_i |
|---|---|---|
| B2 | occupied_j | 否 |
| B3 | occupied_j | 是 |
| B4 | q_j*s_j*(1−u_B,j) | 是 |

B2 因此可向原本无点的目标格传播邻居特征；B3/B4 因 c_i 含 occupied_i，限定在原本有点的目标格增强。B3/B4 具有相同模块参数、8邻域循环、目标门及辅助损失；差别集中在来源权重是否使用 B 的状态估计。B4 对邻居贡献采用连续权重抑制，不额外使用硬阈值筛选。

### 55.8 训练数据、Clean参考和共同训练设置

- 使用同一 V2X-R 训练划分，8083协作帧、24088组单车 Clean/Fog 配对。每辆参与车辆独立执行官方天气采样：randint(0,6)≤2读Fog（3/7），否则读Clean（4/7）。同一协作帧可混合不同天气车辆；Snow不参与训练。
- 所有组的数据包装器均额外读取同索引的训练 Clean 参考以保持共同数据流程，参考只在 B 启用时用于支持度监督，不送入检测主干进行双路融合。读取参考前保存、后恢复 Python/NumPy 随机状态。只在天气与Clean均未重抽帧、整批对齐且两边车辆数量向量一致时提供参考，否则退回默认弱代理；不使用验证集Clean作训练参考。
- 保留官方空标注重采样、投影、车辆顺序和标注过滤。最多7辆车，先投影到共同坐标再融合。官方路径虽然构造了数据增强器，但本次使用的数据集实现没有调用配置中的世界翻转/旋转/缩放；记录实际调用次数，不把YAML配置当作已执行增强。
- 柱体尺寸0.4×0.4×4米，范围x=[−140.8,140.8]、y=[−40,40]、z=[−3,1]；每柱最多32点；训练/评测最大柱数32000/70000。柱体和scatter特征64通道，俯视网格200×704。
- 五组分别从随机初始化训练，不加载旧AttFuse或旧GSPR训练权重。共享主干/GSPR采用匹配初始化；新增模块构造前后保存/恢复CPU随机状态。固定训练seed=20261017，不预先承诺不同硬件或不同组的所有随机轨迹逐位相同，最终核查实际采样记录。
- 每组20轮、batch=8、16个加载进程、预取因子4，打乱训练数据、pin_memory=True、drop_last=True；每轮1010次更新，实际消费8080个请求样本。最后不足一批的3帧不参与该轮更新，空标注重采样另行记录。
- Adam学习率0.002、eps=1e-10、weight_decay=1e-4。沿用官方MultiStep调度，milestones=[10,15]、gamma=0.1，在轮末调用 scheduler.step(从0开始的epoch)，不自行更换调度语义。
- 总损失为官方检测损失，加既有GSPR正则：0.02*(有效点平均可靠性−0.9)²+0.0001*有效点证据平方均值；B3/B4再加0.01*L_q+0.01*L_s。不使用逐点噪声真值标签。B0/B1/B2不含B辅助项。
- 本轮只运行B0–B4，不运行旧配置遗留的FiLM、归一化再校准或其他L0–L2分支。参数形状/环境不满足要求时停止，不自动减小batch或更改实验配方。

### 55.9 两阶段执行、权重选择与评测协议

第一阶段：检查源码/环境/数据身份 → 服务器结构、前后向及完整状态恢复测试 → 依次执行B0–B4各16个训练请求样本的一轮小训练（batch8，共2次更新）和验证 → 每组对Clean/Fog/Snow各12帧推理 → 验收。训练诊断必须保留并通过检查，不能以低AP为由造数或跳过技术验收。

第二阶段：依次对B0–B4执行完整20轮训练及最终评测。每轮计算Clean验证损失，并另起评测进程，在完整829帧Clean验证集计算AP。只按官方O协议的Clean AP70选轮次，完全相同则选较早轮次；须完成全部20轮后正式锁定权重。验证损失或Fog/Snow AP都不用于选轮次。

锁定后用同一权重在Clean、Fog、Snow各829帧完整验证集评测；Snow为训练未见天气。正式test不读取，不以其调结构、阈值或轮次。AP30/AP50/AP70表示预测框与真实框的交并比阈值为0.3/0.5/0.7时的平均精度，保留官方后处理、AP及匹配流程。

| 协议 | 标注和无预测帧处理 | 用途 |
|---|---|---|
| O | 官方随当前输入点云过滤的标注，沿用官方跳过预测返回None帧的行为 | 官方结果对照及Clean选权重 |
| O* | 与O相同的当前天气标注，将预测None帧按空预测计入 | 审计跳过无预测帧的影响 |
| F | 固定使用同帧Clean真实目标标注，预测None帧按空预测计入 | 在相同目标集合下比较天气退化和模块影响 |

天气可能改变O/O*的可见目标集合，跨天气判断退化时应同时查看F，不能只凭某天气O更高就认定更稳健。F采用数据集中可观测的Clean参考标注，不代表完整物理场景全部目标。

### 55.10 对照解释边界、诊断与复现记录

B1−B0考察A完整路径；B2−B0考察普通邻域增强；B3−B2同时引入A、B及目标状态门控，不能分离B的独立贡献；B4−B3考察可信来源加权相对普通来源的影响；B4−B0考察完整方案的组合影响。本轮没有A+B但关闭增强等全部组合，不能由五组结果给三个模块作严格独立贡献排名。只有一个固定训练种子，不宣称统计显著性。

保留每批实际天气读取、输入哈希、重采样、增强调用及随机状态摘要；输出质量/支持状态、增强幅度和来源质量，并利用既有匹配记录按距离、框内点数等分层检查目标恢复/丢失。AP变化本身不能证明q/s的物理意义或目标恢复的因果机制。训练主进程的轮耗时包含验证、评测及保存；同步前向耗时另记，不能混称纯GPU训练速度。

每轮完整提交模型、优化器、调度器、损失累计状态和Python/NumPy/Torch/设备随机状态。恢复只从完整提交轮次继续，未完成轮次可重跑。源码、配置、数据和已完成产物用SHA256绑定，不依赖服务器存在Git；身份变化时使用新输出目录。主流程单GPU串行组织，轮末评测另起进程时训练父进程仍保留部分显存，不保证任意时刻只有一个GPU进程。

海光使用ROCm/HIP环境及完整数据；CUDA使用独立CUDA运行环境、精简点云数据及图片尺寸记录适配，并重新生成环境/标注审计。两边共享上述模型和训练配方，但仍须核对数据、配置及实际记录，不宣称跨硬件逐位相同。本节不新增任何实验完成或效果判断。

## 56. GGCA 模块A独立消融交付与海光验收（2026-10-27）

用户要求重新设计模块A，新增独立gspr_ggca/，只做A0—A3；不修改旧gspr_repe及冻结模型/权重/结果，不创建branch/worktree。已完整阅读本上下文、原接入/训练/评测/恢复代码、官方common_modules/pillar_vfe.py，以及本地PointASNL原文/压缩源码；通过CVF原文核对PointDSC的跨点云对应边长度约束，未直接移植到单柱。

模型：A0复用唯一指定/data/cjm/datasets/logs/gspr-repe-rocm-20261025-v4/B0。A1保留原GSPR装饰特征乘r及官方Linear/BN/ReLU/Max完整前向，读取同一次BN输出构造有梯度的点特征，进行4近邻关系聚合和逐柱有界残差；不替换pfn.forward，不额外更新原BN，不使用未加权特征主路径或全局混合标量。A2仅增加局部间隔/特征支持的有界几何先验修正，无邻居为未知、不按全柱中心或邻居数量判噪。A3仅增加有效观测数、可靠性、不确定性、覆盖、真实逐车传感器距离和共识的充分性控制；一两点严格保留原输出。三组同参数结构、零初始化输出投影，无新增辅助损失、无B/C、无推理GT/晴天参考。算法、适配边界见METHOD_DESIGN.md。

公平性：A1/A2/A3独立从B0的resume_states/00.pth严格对齐全部共同张量及初始RNG，禁止将B0最佳权重用作预训练；共同20轮/batch8/seed20261017/官方Adam及调度/逐CAV Clean4/7 Fog3/7；每轮Clean完整829帧O AP70选模，正式三天气O/O*/F沿用旧官方推理、GT、匹配/AP。训练、数据和checkpoint引擎通过固定源码和三个明确插入点复用，不编辑旧引擎。输出路径独立放/home/cjm/datasets/logs，新的路径保护允许该根并拒绝与数据、旧A0及其他历史输入重叠。

A0直接只读核对：已确认training_summary=complete、20轮，history有20轮。selected选epoch19，checkpoint实际SHA256=7c75ddef20d3042c27f57f07f9f3366f86d9758bcbab5a5e1730aec2bb00c1c6；路径由selected实际取得，为B0/epochs/19/checkpoint_<hash>.pth。初始状态实际SHA256=a4b953d907a9b25f279e19731a5f191b673751ce1f976dafa20b0bfe69fed164，与00.json一致。配置合同身份及相关模型/训练源码与本地核对一致。B0正式三天气summary=complete、full_validation=true、829帧，权重身份一致；O AP70 Clean/Fog/Snow=.6790423426742578/.7552091642488489/.39197021920951053，F=.678715040773897/.445432654556161/.3675350577292191。上述为旧B0结果，不是GGCA新成绩。完整逐轮/预测缓存/数据审计仍以新preflight通过文件为准。

代码：baseline.py只读审计指定A0，合格旧评测直接引用，缺少完整评测时仅补评测；训练/初始化/协议不合格直接停止，不找替代或重训。run_all.sh唯一入口支持MODE=preflight/smoke/all、RESUME=1，默认BACKEND=rocm、解释器/home/cjm/miniconda3/envs/opencood/bin/python。串行完成预检、A0引用、结构及三组真实小训练/推理、第一阶段门禁、三组正式训练、完整评测、选模权重后验机制诊断、自动FINAL_RESULTS.md/json/CSV与失败案例。模块不读GT；diagnose.py仅在前向后关联固定Clean框，额外首12帧/天气，不重算A0 AP或优化A0。完整829帧分层召回、恢复/丢失及定位变化来自原匹配轨迹；实际未缓存正式前向记录分布/开销，不伪造缓存帧统计。

本地及海光Python3.10通过18项标准库静态/模拟文件测试、436项依赖哈希和POSIX sh -n。科研Torch/真实数据均未在本机执行。海光物理卡2上的独立合成验收通过：15种退化柱体/模式组合、A0严格等价、A1—A3关闭和零残差的整模型输出/原PFN/BN精确一致、全部新增参数收到梯度、参数更新、严格重载、模型/Adam/调度器/损失累计及Python/NumPy/Torch CPU+设备RNG下一步恢复。证据/home/cjm/datasets/logs/gspr-ggca-engineering-v1/SYNTHETIC_ONLY.json；这份验收不包含真实数据训练，不能替代流水线smoke。

用户随后明确授权完成代码后自动上传、服务器测试并启动，覆盖最初不自动启动的限制。参考其指定对话01a1760c-afc0-7891-bd7e-0809b7e56784取得海光172.19.165.3:22的已核验主机信息；原root密钥受限只读。用户执行一次脚本给cjm普通账户添加公钥后，已通过cjm正常连接并上传，未改root只读限制、磁盘迁移或其他任务。启动前GPU2使用率及显存0，其他GPU存在任务；选择物理卡2。home所在文件系统当时剩55GiB。

首次gspr-ggca-v1全量输入检查后，因新preflight目录尚未创建而无法写config.yaml，门禁停止，未训练A1—A3。已修复创建顺序并增加回归检查，旧失败目录保留。当前运行源码/home/cjm/OpenCOOD-main/cjmnet/gspr_ggca；修复包delivery/ggca_20261027_223133/gspr_ggca_rocm_20261027_223133.zip，SHA256=a1712512bed4d6f4fdc1ee6705e07fbf1fde5a389dbd7d08a06545d0a2e58244。

2026-10-27 22:36（Asia/Shanghai）状态：新流水线PID498318，GPU2，MODE=all；结果/home/cjm/datasets/logs/gspr-ggca-v2，日志/home/cjm/datasets/logs/gspr_ggca_v2.launch.log，仍在preflight全量输入检查，训练文件计数已到215000/216792。未登记真实smoke或正式A1—A3完成，不编造AP。实际后续状态应直接读取current_stage.json和日志。代码版本改变必须新输出，不能改旧合同或哈希。交付包构建器delivery_tools/build_ggca_source_bundle.py核对当前快照后仅打包新目录，不重写旧依赖哈希。已下载7份原始JSON并逐文件校验到delivery/ggca_20261027/remote_evidence，含旧B0元数据、合成验收及v1部署记录；v2部署记录在远端packages/ggca_20261027/DEPLOYMENT_v2.json。


56节状态更新（2026-10-27 22:39，Asia/Shanghai）：v2的preflight已complete，A0_reference已complete并直接引用已有正式三天气评测，没有补评测或重训。PID498318继续self-test，其重复完整数据检查尚在运行；实际smoke与正式A1—A3未确认。18项标准库检查、海光独立合成验收均通过。交付及部署汇总见delivery/ggca_20261027_223133/VALIDATION_AND_DEPLOYMENT.json。

56节状态更新（2026-10-27 23:13，Asia/Shanghai）：v2的完整self-test已通过，含真实数据加载、旧B1三因素八组合诊断、A0逐值等价、原PFN和BN状态保持、三组新参数梯度/更新/严格恢复、完整训练状态下一步恢复。A1真实小训练已执行两次优化更新；initialization_audit.json确认179个共享张量与B0初始00.pth完全一致，trained_A0_weight_used=false。随后轮末Clean评测因evaluation.py提前导入models触发旧引擎源码隔离检查而停止；没有进入正式训练。已把该导入延后到旧评测完成环境引导之后，并新增静态回归检查，保留所有失败记录，不改旧合同或哈希。

当前v3包为delivery/ggca_20261027_230038/gspr_ggca_rocm_20261027_230038.zip，SHA256=dec80ca8ac260f408be0987546b8fe4f422bc4792e39eebdacf838f59c93d654，28个新目录文件；19项标准库静态/模拟测试通过。本机没有执行Torch、数据或科研实验。新包已经上传并部署到/home/cjm/OpenCOOD-main/cjmnet/gspr_ggca；仍使用海光原opencood解释器及ROCm环境，无新专用算子。v3在物理GPU2后台MODE=all、RESUME=1运行，主PID591672；输出/home/cjm/datasets/logs/gspr-ggca-v3，主日志/home/cjm/datasets/logs/gspr_ggca_v3.launch.log。23:12读取进程仍存活，preflight与A0_reference已经完成，当前self-test进行中。完整A1/A2/A3小训练和三天气推理验收尚未全部完成，正式20轮训练也未确认开始；流水线会在验收通过后自动继续，无需逐个启动。

另外在服务器用v2的A1两步小训练权重验证v3评测入口，独立目录/home/cjm/datasets/logs/gspr-ggca-engineering-eval-v3/clean/summary.json实际status=complete，Clean12帧及O/O*/F回放完成，日志有REAL_CLEAN_SMOKE_EVALUATION_PASSED。该工程检查没有训练A0，也没有把小训练权重用于正式A1/A2/A3初始化；其低AP不代表正式模型效果。初次独立诊断驱动缺TMPDIR，补齐与统一入口相同环境后完成，统一run_all.sh原本即设置TMPDIR。

实际证据已下载并逐文件SHA256核对到delivery/ggca_20261027_230038/remote_evidence：v2完整self-test、B1三因素诊断、A1初始化审计、v3完整预检和A0复用审计/引用、12帧Clean推理结果及部署记录。最终正式FINAL_RESULTS.md/json只在完整三组训练、829帧三天气评测和机制分析结束后自动生成；当前没有GGCA正式消融结论。以后只有在当前进程退出后，才用同一入口RESUME=1恢复，避免重复启动。

56节最终状态复核（2026-10-27 23:15，Asia/Shanghai）：v3的完整self-test已实际complete，summary及B1三因素诊断已下载校验。主PID591672已自动进入smoke_A1，子PID630078，STOP.json不存在。后续A2/A3小规模验收、正式20轮及完整评测仍待后台顺序完成。最新实证以delivery/ggca_20261027_230038/VALIDATION_AND_DEPLOYMENT.json和remote_evidence为准。



## 57. REPE B1/B2正式结果归档及旧B3/B4暂停（2026-10-28直接读取）

依据用户明确授权，本轮直接连接海光cjm@172.19.165.3，先停止旧REPE的B3/B4流水线，再只读收集B1/B2完整结果和选中权重。读取时间：2026-10-28T09:24:01.119003+08:00。本节记录实测结果；§55继续只保留方法。

### 57.1 停止范围与保留进度

核对current_stage为B3_train、命令行及输出根与/data/cjm/datasets/logs/gspr-repe-rocm-20261025-v4一致后，向旧REPE主进程333928发送SIGTERM。主进程、B3子进程1604852及16个数据加载进程已全部退出；停止后复查两个主要PID均不存在。没有删除、移动或修改原源码、模型权重、数据、既有结果与完成记录。原流水线可能把信号15记为technical_failure；本次原因是用户主动停止，不能当作训练数值失败。
B3已完整提交5/20轮；未完成的第6轮不保证保留批次进度。B4正式训练尚未开始，根目录下B4/不存在；第一阶段已有的B4小规模验收不在本次删除或重跑范围。旧流水线停止后不会自动继续B4。
B3最近恢复点：`/data/cjm/datasets/logs/gspr-repe-rocm-20261025-v4/B3/resume_states/05.pth`。SHA256：`86af42fc657f8c3e869ad9f45c93402f0fee3356fd6a3078a6086afc18141a1c`，已实际核验通过；同轮模型权重也已核验。所有旧进度保留，未启动新实验。
独立GGCA任务不属于停止范围。读取时PID591672仍为gspr_ggca/orchestrator.py、输出/home/cjm/datasets/logs/gspr-ggca-v3；未向其发送信号或修改其源码、配置及结果。

### 57.2 B1/B2身份、完成状态及选中权重

两组均training_summary.status=complete、20轮完整提交，正式评测status=complete，Clean/Fog/Snow各829帧、full_validation=true，评测权重SHA256与selected.json一致。已实际读取并核验选中权重SHA256、训练合同、选模记录、选中轮次、正式评测阶段绑定的summary哈希以及summary声明的输出文件哈希。B0/B1/B2的冻结GT身份相同。没有重新加载模型推理或重算AP；未在本机执行科研实验。

| 模型 | 完成轮数 | 选中轮次 | Clean O AP70 | 训练及验证累计/小时 | 平均每轮/分钟 |
|---|---:|---:|---:|---:|---:|
| B1 | 20/20 | 20 | 69.360099% | 18.918420 | 56.755260 |
| B2 | 20/20 | 13 | 69.974354% | 19.129584 | 57.388753 |

耗时包含轮内训练、验证、Clean评测及保存；不是纯GPU计算耗时。两组都只按完整Clean O AP70选权重，平局选更早轮次。

B1 选中权重：

`/data/cjm/datasets/logs/gspr-repe-rocm-20261025-v4/B1/epochs/20/checkpoint_a6be322cb8c10f6a49d829877b4bd5e563358a00b53bc125ba425c49fbab7c2d.pth`

SHA256：`a6be322cb8c10f6a49d829877b4bd5e563358a00b53bc125ba425c49fbab7c2d`（已实际核验）。

选择记录：`/data/cjm/datasets/logs/gspr-repe-rocm-20261025-v4/B1/selected.json`；完整评测：`/data/cjm/datasets/logs/gspr-repe-rocm-20261025-v4/B1/final_evaluation/summary.json`。

B2 选中权重：

`/data/cjm/datasets/logs/gspr-repe-rocm-20261025-v4/B2/epochs/13/checkpoint_509b3eb309dcd28816d95b8f99f2bb53f548cff93f6bd742a6ee15d3c8976df5.pth`

SHA256：`509b3eb309dcd28816d95b8f99f2bb53f548cff93f6bd742a6ee15d3c8976df5`（已实际核验）。

选择记录：`/data/cjm/datasets/logs/gspr-repe-rocm-20261025-v4/B2/selected.json`；完整评测：`/data/cjm/datasets/logs/gspr-repe-rocm-20261025-v4/B2/final_evaluation/summary.json`。

### 57.3 完整正式评测表

AP以下用百分数显示；精确未舍入值保存于本机REMOTE_RESULTS.json和服务器原summary.json。

| 模型 | 天气 | 协议 | AP30 | AP50 | AP70 |
|---|---|---|---:|---:|---:|
| B1 | clean | O | 85.289159% | 82.327709% | 69.360099% |
| B1 | clean | O* | 85.275455% | 82.314482% | 69.348955% |
| B1 | clean | F | 85.275455% | 82.314482% | 69.348955% |
| B1 | fog | O | 86.394738% | 83.693988% | 68.598951% |
| B1 | fog | O* | 86.394738% | 83.693988% | 68.598951% |
| B1 | fog | F | 51.202147% | 49.604469% | 40.547468% |
| B1 | snow | O | 57.975610% | 54.921940% | 35.366791% |
| B1 | snow | O* | 57.648792% | 54.612336% | 35.167422% |
| B1 | snow | F | 54.238140% | 51.365780% | 33.076814% |
| B2 | clean | O | 85.485481% | 82.258046% | 69.974354% |
| B2 | clean | O* | 85.444277% | 82.218397% | 69.940626% |
| B2 | clean | F | 85.444277% | 82.218397% | 69.940626% |
| B2 | fog | O | 85.981089% | 84.509416% | 74.270758% |
| B2 | fog | O* | 85.887453% | 84.417383% | 74.189876% |
| B2 | fog | F | 52.616585% | 51.141797% | 44.233905% |
| B2 | snow | O | 65.800942% | 61.901224% | 45.139612% |
| B2 | snow | O* | 65.441251% | 61.562850% | 44.892863% |
| B2 | snow | F | 61.657532% | 57.962835% | 42.224104% |

### 57.4 与本轮B0的AP70比较

参照为同一服务器、同一REPE根目录B0的选中第19轮完整正式评测，不能替换成历史纯AttFuse或Clean-only GSPR结果。差值用未舍入原数计算，单位为百分点。

| 天气 | B0 F AP70 | B1 F AP70 | B1−B0 F | B2 F AP70 | B2−B0 F | B1−B0 O | B2−B0 O |
|---|---:|---:|---:|---:|---:|---:|---:|
| clean | 67.871504% | 69.348955% | +1.477451 | 69.940626% | +2.069122 | +1.455865 | +2.070120 |
| fog | 44.543265% | 40.547468% | -3.995797 | 44.233905% | -0.309360 | -6.921966 | -1.250158 |
| snow | 36.753506% | 33.076814% | -3.676692 | 42.224104% | +5.470598 | -3.830231 | +5.942590 |

描述性结论：本轮B1在Clean上提升、Fog/Snow下降；B2在Clean/Snow提升，Fog仍略低于B0。B2没有加入A/B，其改善不能归因于A。以上是单种子、当前配方下的事实，不构成统计显著性或因果机制判断。B1同时替换原GSPR/PFN加权及归一化路径，单看gate混合比例或AP变化不能判定具体退化原因。B3/B4暂停是用户研究决定，不据此把未完成实验写成失败结果。

### 57.5 本地证据

原始读取及精确数值：`delivery/repe_stop_20261028/REMOTE_RESULTS.json`。包含B0/B1/B2的selected、summary、精确AP差值、权重及输出哈希核验标志，B3最后完整提交及恢复哈希核验、旧进程退出复查和独立GGCA进程信息。本轮只停止指定旧REPE进程，并记录/读取既有结果，未重训、未推理、未修改模型。

## 58. GGCA A2点对几何修订与保留A1的版本续跑（2026-10-28）

用户明确授权连接海光、仅停止本次GGCA及其子进程、上传部署、服务器测试和后台续跑；优先保留A1，禁止改历史合同/哈希/完成标记，禁止本机科研实验、分支/worktree及影响其他任务。已读取本节之前上下文、现行GGCA模型/训练/评测/断点/阶段/汇总实现，以及本地PointDSC-master.zip的59个文件哈希和全部Python源码语法；重点阅读PointDSC.py、common.py、loss.py、trainer.py和三类数据集对应标签生成。

11:39首次直接核实：旧主进程591672、A1训练子进程908558，物理GPU2，A1完整提交7/20轮，正在第8轮约248/1010。A2/A3正式目录不存在。逐项核对前7轮complete.json所绑定的模型、完整状态、Clean选模评测和采样SHA256；07.pth包含192个模型张量、Adam的123个参数状态、原调度器、损失状态/累计以及Python/NumPy/Torch CPU/设备RNG，实际字段齐全。第8轮当时仅有sampling.jsonl.tmp，不存在逐批恢复能力。/home文件系统当时可用约49GB；A1当时约1.14GB，完整旧B0约3.01GB，原地引用权重，预计新增约10GB量级，有余量，不删除旧数据或记录。

保留旧/home/cjm/OpenCOOD-main/cjmnet/gspr_ggca全部字节及源码快照，历史snapshot SHA256=04a7c16fb5c9816cf4298f93df6b519575c88adefc0ad4fda4852712cc54533d。只复制少量源码，所有A1大权重/状态原地引用。旧版本/状态清单和只读源码副本位于/home/cjm/datasets/logs/gspr-ggca-geometry-v2-20261028/migration/。已启动边界监视器PID3089875，目标为第8轮完整提交后冻结已核实的A1训练进程组，核验checkpoint/resume/Clean评测/trace后仅终止旧GGCA监督器及其组；不假设SIGTERM能保存，不声称逐批无损。实际完成必须以boundary_stop.json为准，当前本段登记时仍等待，不能据此声称已经停止。

A2修订：旧c=sum(g*p)/sum(g)在等p及整体g缩小时抵消问题确实存在。新版保留A1原GSPR-PFN、原NN参数/初始化/BN及关系网络，定义点对紧支撑距离关系B（半径clip(.15+.002*逐车传感器距离,.15,.6)米），结合局部二跳支持和已受GSPR影响的特征余弦得到G。最终交换w_ij=a_A1,ij*G_ij，不再按sum(G)重新归一化；剩余1-sum(w)保留自身f_i。柱内查询只增加有界正支持奖励normalize[p_i*(1+.5*s_i)]，无支持保留原先验。分别记录绝对几何支持s、条件邻居可信q、实际交换量和自身保留量，缺邻居不判噪。PointDSC借鉴的是NonLocalBlock的几何直接约束点对消息、局部多点兼容思想；不移植跨云边长度差、配准/SVD/谱匹配及依赖真实对应标签的损失，无新增辅助损失和专用CUDA依赖。

A3最终采用稀疏保护修订2：同时检查N_eff和绝对先验量P=sum(p)，E_count=min(clip((N_eff-2)/4),clip((P-2)/4))，再乘原不确定性/覆盖距离因子及mean(s)，仅将同一A2残差强度乘E；n<=2严格回原输出。修正原因：N_eff对统一降低p近似不变，不能单独表示充分的可信证据。P不是有效点真值，不引入点噪声标签。A1算法及正式训练源码全部不变；新命名空间禁止启动A1训练，A1只能由原入口按原合同继续。A1读出/工程诊断仍可用同等价架构。

初稿源码gspr_ggca_geo_v2/及输出gspr-ggca-geometry-v2-20261028/已经保留。其工程测试曾在空闲卡1通过：等/近等p结构作用、非零输出、整体支持与可信度分离、双表面、冲突高可靠/受支持低可靠、异常群失败边界、刚体不变性、退化输入、原PFN/BN/关闭/零残差、全新参数梯度/更新/保存恢复，以及实际A1 07状态只读重建和全部RNG逐值相同。该初稿A3没有最后的P检查，因此不作为最终R2的A3验收。仅等待中的初稿调度器3227477已主动停止，无训练子进程；其run_identity和STOP保留，原A1及边界监视器未受影响。

最终R2源码独立部署/home/cjm/OpenCOOD-main/cjmnet/gspr_ggca_geo_v2_r2；输出独立/home/cjm/datasets/logs/gspr-ggca-geometry-v2-20261028-r2。实际包delivery/ggca_geometry_v2_20261028_122532/gspr_ggca_geo_v2_20261028_122532.zip，SHA256=d8b53f9b3e9ac4ea7182efb814136732c6580f863869bb49ca29bfb0030706f9，解压顶层gspr_ggca_geo_v2_r2，无cjmnet外壳。独立迁移计划引用旧计划文件SHA及原边界记录，不改历史合同。唯一run_all.sh调度原A1旧路径/原目录、全新A2/A3路径，并最终逐项核对初始化/种子/配置/数据/选模/评价协议后统一汇总。

R2本机仅14项静态控制检查；海光另有8项A0假文件合同测试和5项调度/接管/空闲卡测试，合计27项通过。未在本机加载Torch模型或科研数据。最终R2的完整Torch/几何/低可信绝对量/真实状态工程验收已在海光卡6启动（PID3286906），12:32仍在运行；此处不提前记通过。卡1随后被无关PID3211087占用时测试检查拒绝启动，未停止该任务；实际核对卡6仅3MiB后选择它，TEST_GPU_OVERRIDE.json记录实际选择，正式训练及A1保持卡2。主机128核、负载约24、空闲内存约1.3TB，额外空闲卡测试有资源余量。

统一续跑设计：等待原A1第8轮提交并安全停止→R2预检/A0直接引用/实际停止状态重建与完整结构测试→若额外卡空闲，独立A1_lane立即按旧代码在卡2从完整状态继续，同时新A2/A3小训练/三天气推理在额外卡顺序完成；否则单卡顺序。A1_lane记录PID、启动时间、原命令及完整状态哈希，控制器重启可核对接管，不重复启动。新测试失败保留原A1继续，并明确报告新测试失败。所有正式新训练仍受R2真实验收门禁控制，A2/A3从B0原初始00.pth独立开始。当前本段不宣称停止/恢复已完成、不宣称正式GGCA效果。

58节工程状态更新（2026-10-28 12:35，Asia/Shanghai）：最终R2在海光卡6的完整工程验收已实际完成，含绝对先验量低但N_eff高时A3严格回原输出、旧新A1非零输出/梯度/Adam一步一致、A1实际07状态只读逐值恢复、原PFN/BN/零残差及三组13个新增参数梯度/更新/严格重载、15个退化组合。R2与初稿A2非零输出精确一致，变化只在A3保护。原始证据已下载校验到delivery/ggca_geometry_v2_20261028_122532/remote_evidence，工程范围说明见ENGINEERING_VALIDATION.md。正式真实小训练/三天气推理仍待执行。

最终R2统一入口PID3319576，主日志/home/cjm/datasets/logs/gspr_ggca_geometry_v2_r2.launch.log，MODE=all、RESUME=1、正式GPU2、额外测试优先GPU6（卡1已被其他任务占用）。12:35原A1仍第8轮927/1010，边界监视器仍等待。尚不能声称已经安全停止或已恢复第9轮；这些后续状态必须实际复核再追加。旧A1源码与初稿源码均原地保留。

58节最终续跑验收（2026-10-28 14:19，Asia/Shanghai，额度恢复后直接读取）：原A1第8轮已于12:48:34完成整轮训练、验证及Clean选模并提交。边界工具在该提交后冻结/核验并停止原主进程591672、训练组908558，12:48:35核验该组无存活进程。next_epoch_artifacts=[]，第9轮尚未创建；未丢弃当前第8轮、未重跑已完成轮次，没有逐批恢复声明。旧监督器因主动信号生成的STOP/failure原样保留。

实际完整恢复点/home/cjm/datasets/logs/gspr-ggca-v3/A1/resume_states/08.pth，SHA256=220591f64cdbe003ee8f6f69877e8eb5443ae2b31659e30102185d9ae229233f。对应epoch08模型checkpoint SHA256=d0e72a4e80b5d9203e5b806cd52bad4de5b7cb71ed745571d2c1d44ca18574de。实际逐值核对完整状态中的模型与该权重一致，Adam123个参数状态的step全部8080，原调度器last_epoch=7、lr=.002，损失/累计和全部RNG齐全。R2 self-test进一步重建真实08状态，模型/优化器/调度器/损失及RNG逐值匹配，对保留A1执行的测试优化步数0。旧28源码及前7轮所有继承产物哈希再次核对不变。

A1已由原/home/cjm/OpenCOOD-main/cjmnet/gspr_ggca/main.py按原合同、原配置、原GPU2、原输出目录从08恢复，真实训练PID3418600。14:18日志已经推进至第9轮约883/1010，14:19 rocm-smi明确把3418600映射到物理卡2，显存约30GB；不是只看到nohup返回。唯一主控PID3319576仍存活，已在新根current_stage标记preserved_A1_train，并真实接管同一A1子进程，未重复启动。

最终R2的preflight、A0_reference、完整self-test、新A2/A3各两次真实V2X-R优化更新以及Clean/Fog/Snow各12帧推理均complete，stage1_acceptance.complete，stage2_allowed=true。实际小训练卡6，两组均独立对齐B0原始00状态，未使用训练后A0/A1权重；源合同method.revision=3、几何revision=2、稀疏保护revision=2。逐文件核验两组评测output_hashes、checkpoint和输入/冻结GT身份，结果都一致。原根A2/A3正式目录仍不存在，新根A2/A3正式训练也尚未开始；当前等待保留A1完成20轮，后续自动正式评测A1，再独立训练/评测新A2/A3并统一汇总。A0没有重训，原正式评测直接引用。低AP来自两步小训练，不构成正式效果结论。

最终输出/home/cjm/datasets/logs/gspr-ggca-geometry-v2-20261028-r2；主日志/home/cjm/datasets/logs/gspr_ggca_geometry_v2_r2.launch.log；原A1续跑日志位于新根A1_lane/train.log，权重/完整状态继续写旧根gspr-ggca-v3/A1。实际停止与续跑终审在新根migration/FINAL_CONTINUATION_VERIFICATION.json，包含所有状态/哈希、Adam步数、真实PID及新smoke身份。当前home可用49204842496字节，未为迁移复制大权重、未删除旧记录、未停止其他任务或磁盘迁移。

最终部署、验收及全部33份原始JSON/Markdown已下载SHA256核对至delivery/ggca_geometry_v2_20261028_122532/remote_evidence；总记录VALIDATION_AND_DEPLOYMENT.json。以后主控退出才执行单条后台恢复命令：nohup env BACKEND=rocm GPU=2 GGCA_TEST_GPU=6 MODE=all RESUME=1 sh /home/cjm/OpenCOOD-main/cjmnet/gspr_ggca_geo_v2_r2/run_all.sh >> /home/cjm/datasets/logs/gspr_ggca_geometry_v2_r2.launch.log 2>&1 < /dev/null &。已有原A1存活时会核对PID启动时间、原命令后接管，避免重复进程。所有科研测试均在海光完成；本机仅编写、静态检查、打包和读取证据。当前没有正式GGCA消融结果或性能提升结论。

58节实时推进补证（2026-10-28 14:25，Asia/Shanghai）：已再次确认主控3319576、原A1子进程3418600均存活，原A1第9轮已实际推进至950/1010。ACTUAL_RUNNING_PROGRESS.json记录真实日志片段及摘要SHA，全部34份证据已下载核验；交付汇总CONTINUATION_AUDIT.md和VALIDATION_AND_DEPLOYMENT.json保存于delivery/ggca_geometry_v2_20261028_122532。没有重复启动原实验。


## 59. GGCA A1正式结果与A0对照记录（2026-10-29）

读取时间：2026-10-29 09:40:32（北京时间；精确 UTC 时间见 JSON）。本次仅远程读取已有文件、核对哈希及汇总计数，未运行训练或推理，未修改服务器文件或进程。

A1：20/20 轮训练完成；Clean、Fog、Snow 正式评测状态 complete，每种天气 829 帧。选模使用 Clean validation O AP70，A1 选中第 20 轮，A0 选中第 19 轮。A0 为已有 B0 结果。

AP 为原记录数值乘 100，单位 %；差值为 A1−A0，单位百分点，保留四位小数。

### 59.1 O协议

| 天气 | 指标 | A0 | A1 | A1−A0 |
|---|---|---:|---:|---:|
| clean | AP30 | 83.9791 | 86.3213 | +2.3421 |
| clean | AP50 | 81.7420 | 83.5675 | +1.8255 |
| clean | AP70 | 67.9042 | 71.2182 | +3.3139 |
| fog | AP30 | 88.2748 | 89.6151 | +1.3403 |
| fog | AP50 | 87.0029 | 88.6534 | +1.6505 |
| fog | AP70 | 75.5209 | 78.7111 | +3.1902 |
| snow | AP30 | 60.2126 | 65.4365 | +5.2239 |
| snow | AP50 | 56.2044 | 61.5846 | +5.3802 |
| snow | AP70 | 39.1970 | 45.0783 | +5.8812 |

### 59.2 O*协议

| 天气 | 指标 | A0 | A1 | A1−A0 |
|---|---|---:|---:|---:|
| clean | AP30 | 83.9387 | 86.2797 | +2.3410 |
| clean | AP50 | 81.7026 | 83.5273 | +1.8247 |
| clean | AP70 | 67.8715 | 71.1838 | +3.3123 |
| fog | AP30 | 88.2267 | 89.5663 | +1.3396 |
| fog | AP50 | 86.9555 | 88.6051 | +1.6496 |
| fog | AP70 | 75.4798 | 78.6683 | +3.1885 |
| snow | AP30 | 60.0274 | 65.1123 | +5.0849 |
| snow | AP50 | 56.0316 | 61.2795 | +5.2479 |
| snow | AP70 | 39.0765 | 44.8549 | +5.7784 |

### 59.3 F协议

| 天气 | 指标 | A0 | A1 | A1−A0 |
|---|---|---:|---:|---:|
| clean | AP30 | 83.9387 | 86.2797 | +2.3410 |
| clean | AP50 | 81.7026 | 83.5273 | +1.8247 |
| clean | AP70 | 67.8715 | 71.1838 | +3.3123 |
| fog | AP30 | 52.0657 | 53.1023 | +1.0366 |
| fog | AP50 | 51.3155 | 52.3429 | +1.0274 |
| fog | AP70 | 44.5433 | 46.4589 | +1.9156 |
| snow | AP30 | 56.4590 | 61.4106 | +4.9517 |
| snow | AP50 | 52.7006 | 57.6986 | +4.9980 |
| snow | AP70 | 36.7535 | 42.1884 | +5.4349 |

协议定义：O 按当前天气过滤真值框，预测为 None 时跳过整帧；O* 使用同样真值框，None 帧保留真值并记零预测；F 固定使用 Clean 真值框，None 帧保留真值并记零预测。O 为实验主协议。

### 59.4 检测计数（IoU 0.7）

| 协议 | 天气 | A0 正确匹配/误检/漏检 | A1 正确匹配/误检/漏检 |
|---|---|---:|---:|
| O | clean | 4998/2010/1223 | 5102/1718/1119 |
| O | fog | 3222/769/449 | 3284/649/387 |
| O | snow | 3027/1865/2809 | 3327/1643/2498 |
| O* | clean | 4998/2010/1226 | 5102/1718/1122 |
| O* | fog | 3222/769/451 | 3284/649/389 |
| O* | snow | 3027/1865/2827 | 3327/1643/2527 |
| F | clean | 4998/2010/1226 | 5102/1718/1122 |
| F | fog | 3222/769/3002 | 3286/647/2938 |
| F | snow | 3027/1865/3197 | 3327/1643/2897 |

### 59.5 计分真值数量

| 天气 | A0 O/O*/F | A1 O/O*/F |
|---|---:|---:|
| clean | 6221/6224/6224 | 6221/6224/6224 |
| fog | 3671/3673/6224 | 3671/3673/6224 |
| snow | 5836/5854/6224 | 5825/5854/6224 |

### 59.6 训练及选模

| 轮次 | A0 Clean O AP70 (%) | A1 Clean O AP70 (%) |
|---:|---:|---:|
| 1 | 0.9507 | 2.0048 |
| 2 | 13.6557 | 11.7924 |
| 3 | 30.2238 | 28.9201 |
| 4 | 38.3714 | 36.8495 |
| 5 | 42.1482 | 14.2292 |
| 6 | 47.7084 | 50.8622 |
| 7 | 49.3842 | 1.8053 |
| 8 | 50.4977 | 53.3699 |
| 9 | 55.7834 | 60.0003 |
| 10 | 60.2086 | 62.6159 |
| 11 | 0.0000 | 63.3825 |
| 12 | 64.6743 | 68.6914 |
| 13 | 66.2751 | 69.4713 |
| 14 | 65.4199 | 68.2519 |
| 15 | 65.7175 | 69.9118 |
| 16 | 67.6154 | 70.4787 |
| 17 | 67.8600 | 70.1916 |
| 18 | 67.5653 | 70.3272 |
| 19 | 67.9042 | 70.7021 |
| 20 | 67.7378 | 71.2182 |

种子均为 20261017；每轮记录 8083 个接受帧、实际消费 8080 帧、1010 次更新。两组均完成 20 轮。

### 59.7 核对及原始文件

三天气逐帧编号与帧标识一致；冻结真值身份一致；protocols、official_source_hashes、official_config_sha256、输入清单及推理种子等字段一致。protocol.json 全文件不相同；不同顶层字段为 associated_d1d_training、checkpoint_sha256、gt_identity、gt_manifest_sha256、identity、physical_gpu、prediction_cache_identity、source_hashes、stage。原字段完整保留在 JSON。

#### A0

- 训练目录：`/data/cjm/datasets/logs/gspr-repe-rocm-20261025-v4/B0`
- 正式结果：`/data/cjm/datasets/logs/gspr-repe-rocm-20261025-v4/B0/final_evaluation/summary.json`
- 选模记录：`/data/cjm/datasets/logs/gspr-repe-rocm-20261025-v4/B0/selected.json`
- 选用权重：`/data/cjm/datasets/logs/gspr-repe-rocm-20261025-v4/B0/epochs/19/checkpoint_7c75ddef20d3042c27f57f07f9f3366f86d9758bcbab5a5e1730aec2bb00c1c6.pth`
- 权重 SHA256：`7c75ddef20d3042c27f57f07f9f3366f86d9758bcbab5a5e1730aec2bb00c1c6`
- 实际权重哈希与选模记录一致：True
- 正式评测权重与选模记录一致：True
- summary 声明的全部输出文件哈希核对一致：True

#### A1

- 训练目录：`/home/cjm/datasets/logs/gspr-ggca-v3/A1`
- 正式结果：`/home/cjm/datasets/logs/gspr-ggca-v3/A1/final_evaluation/summary.json`
- 选模记录：`/home/cjm/datasets/logs/gspr-ggca-v3/A1/selected.json`
- 选用权重：`/home/cjm/datasets/logs/gspr-ggca-v3/A1/epochs/20/checkpoint_4b267b6c71aa7a15aacc48d1b73af4f1d380f086acef8b94be887371f832799a.pth`
- 权重 SHA256：`4b267b6c71aa7a15aacc48d1b73af4f1d380f086acef8b94be887371f832799a`
- 实际权重哈希与选模记录一致：True
- 正式评测权重与选模记录一致：True
- summary 声明的全部输出文件哈希核对一致：True

完整原始选模记录、评测汇总、协议文件、20 轮摘要、计数及核对记录：A1_A0_RECORD.json。未加入机制判断或效果归因。

本节对应读取时点的事实记录，更新第58节中A1尚未完成的历史进度；第58节原文保留。A2/A3正式结果未纳入本次记录。

本地证据：

- 结果表：delivery/ggca_A1_results_20261029/A1_A0_RESULTS.md。
- 原始记录及哈希：delivery/ggca_A1_results_20261029/A1_A0_RECORD.json。

## 60. SCOR-W评测协议与天气点云全链路诊断（2026-10-30读取）

### 60.1 完成范围与核验

服务器结果目录 `/data/cjm/datasets/logs/scorw-diagnostics-v1.2/full`，源码 `/home/cjm/OpenCOOD-main/cjmnet/scorw_diagnostics`。最终状态 `DIAGNOSTICS_COMPLETE`；tests、A1、A、B_smoke、B、reports均为passed，11项服务器CPU单元测试通过。完整Validate共9个scene、1980帧、10478份CAV点云，原始与Fog输出均582505715条点记录；活跃CAV点云6852份，未激活3626份。本次复用D6缓存，没有训练、天气生成或GPU推理。

本次读取并核验REPORT_MANIFEST声明的17份根层报告哈希，全部一致；未重新执行点云追踪或检测。REPORT_MANIFEST SHA256=`76096aaad0ee6c516a6f84f6f1adee7f67f2cfda7418497a0d3f8a56d85748a1`。输入冻结记录D6历史源码检查通过，固定GT目标实例54621；输入清单身份=`c8834852ad94a447ad90921edd36e186a5acc18788d90073211ac1b54c399a82`。

### 60.2 固定缓存的AP排序与GT对照

以下AP为0–1值。全局排序与按帧累计使用完全相同的D6预测框、置信度、GT和匹配记录。原全局排序六个AP值均复现。

|天气|指标|全局排序|按帧累计|全局减按帧|
|---|---|---:|---:|---:|
|clean|AP30|0.8841046711|0.8268666270|+0.0572380441|
|clean|AP50|0.8716086110|0.8065651025|+0.0650435086|
|clean|AP70|0.8244272275|0.7377555281|+0.0866716994|
|fog|AP30|0.6752106844|0.6016536837|+0.0735570007|
|fog|AP50|0.6648634939|0.5831711202|+0.0816923738|
|fog|AP70|0.6222479004|0.5204779871|+0.1017699132|

固定顺序为“D6全局排序→D6按帧累计→原生GT→原生匹配器”，下表为逐项AP变化，分解顺序固定；完整两套GT×两种匹配×两种排序交叉表在MATCHING_DIFFERENCE.json。

|天气|指标|排序变化|GT变化（按帧累计条件下）|匹配变化（原生GT条件下）|
|---|---|---:|---:|---:|
|clean|AP30|-0.0572380441|+0.0444607099|+0.0000000000|
|clean|AP50|-0.0650435086|+0.0445254323|+0.0000000000|
|clean|AP70|-0.0866716994|+0.0430241610|+0.0000000000|
|fog|AP30|-0.0735570007|+0.0349074606|+0.0000000000|
|fog|AP50|-0.0816923738|+0.0290446773|+0.0000000000|
|fog|AP70|-0.1017699132|+0.0055714019|+0.0000000000|

两种匹配器在本次交叉审计中的逐帧TP序列差异计数为0。D6 Clean AP30−AP70：全局排序0.0596774436，按帧累计0.0891110989。

|Clean指标|D6预测＋原生GT/匹配＋按帧累计|§44历史原生推理|前者减历史|
|---|---:|---:|---:|
|AP30|0.8713273369|0.8729662430|-0.0016389062|
|AP50|0.8510905348|0.8489018777|+0.0021886570|
|AP70|0.7807796891|0.7689611007|+0.0118185884|

历史原生预测框未逐帧核实，记录状态 `UNVERIFIED_NO_NATIVE_PREDICTION_CACHE`；上述剩余差值未完成归因。本次没有原生重推理，AP间距未作为定位精度的独立证据。

### 60.3 GT身份与几何对照

D6 GT累计54621个目标帧实例，原生GT累计54232个；共同object ID实例52721个，仅D6存在1900个，仅原生存在1511个。1964帧的GT数量不同，529帧的活跃CAV名单不同。共同ID中，角点集合误差超过0.0001m的实例43162个，最大角点集合误差0.2259015888m。数量均按帧累计，不表示独立道路目标数。

D6从Clean元数据构造原始三维角点，按距任一活跃CAV120m及GT_RANGE的XY角点范围过滤。原生流程实际执行getitem、collate及GT生成，先投影并重建yaw框，执行XYZ八角点范围过滤，再执行最终XY过滤。原生流程先按最大CAV数截断，再做70m筛选；D6先筛70m范围，再截断。原生会打乱点顺序，D6不打乱。上述为代码及实际执行路径记录，未将单一差异指定为全部几何或AP差异的原因。

### 60.4 全Validate点流与有效网络输入

“进入网络”在本诊断中定义为：四维点特征位于原生体素化、collate后送入PillarVFE的有效槽位；不表示该点必然影响后续最大池化特征。逐点通过original_index[emitted]关联；未用XYZ最近邻。独立携带ID的体素化与正式四维结果逐文件核对坐标、点数、特征和槽位顺序；逐帧输入哈希与D6缓存一致，来源关联失败计数0。

|最终去向|Clean点记录|Fog点记录|
|---|---:|---:|
|PillarVFE有效槽位|267775138|254584448|
|自车过滤|8332664|24168102|
|未激活CAV|202533843|202533843|
|无返回|0|0|
|空间范围过滤|27287489|14545734|
|体素边界|2|2|
|单体素点数限制|76576579|86673586|
|体素总数限制|0|0|

### 60.5 原始目标位置关联点与近场天气点

使用固定D6 GT框在原始Clean位置关联，重叠框点单列并从唯一归属统计中排除。活跃CAV的唯一归属点中，Clean有效槽位27094627条，Fog保留原始目标位置的硬回波有效槽位24920617条。来源0为模拟器硬回波分类，框内点没有逐点车辆语义真值。

|目标到ego距离/m|目标帧实例|原始唯一归属点（活跃CAV合计）|Clean有效槽位|Fog硬回波有效槽位|天气替换点|
|---|---:|---:|---:|---:|---:|
|0-20|11974|37650738|12528374|12318123|192764|
|20-40|10735|8486126|4396660|4028672|365490|
|40-60|13717|17060653|6445716|5549577|888241|
|60-80|8856|7046573|3419769|2978826|439157|
|80-120|7352|285387|284929|45419|239510|

距离分组使用目标到ego的三维距离；点到各自传感器的距离统计另存POINT_FLOW_BY_DISTANCE.json，不混用。上表0–120m范围分组未覆盖范围外目标。

Validate来源1天气点共53429144条，全部位于各自CAV局部传感器0–3m范围。去向如下；占比分母为全部53429144条近场天气点，包含未激活CAV。

|近场天气点去向|数量|占比|
|---|---:|---:|
|PillarVFE有效槽位|2759795|5.16533636%|
|自车过滤|15835438|29.63820270%|
|未激活CAV|24557563|45.96286064%|
|无返回|0|0.00000000%|
|空间范围过滤|177963|0.33308226%|
|体素边界|0|0.00000000%|
|单体素点数限制|10098385|18.90051804%|
|体素总数限制|0|0.00000000%|

同一批原始点在Clean中进入有效槽位15950363条，替换为天气回波后进入有效槽位2759795条；此配对依据原始索引。

### 60.6 逐scene记录与描述性关联

|scene|帧|CAV点云|Clean目标有效槽位|Fog硬回波目标有效槽位|Clean AP70匹配目标|Fog AP70匹配目标|
|---|---:|---:|---:|---:|---:|---:|
|2021_08_20_21_48_35|112|560|1216501|1190742|2255|1854|
|2021_08_21_17_30_41|157|471|1399383|1345456|2988|2446|
|2021_08_22_13_37_16|135|270|567454|554744|1453|1217|
|2021_08_22_22_01_17|202|404|733034|724862|1134|1038|
|2021_08_23_10_51_24|64|128|419579|413499|1144|853|
|2021_08_23_13_17_21|48|96|218004|211068|598|485|
|2021_08_23_19_42_07|57|114|253593|253575|469|466|
|2021_09_09_19_27_35|459|3213|2512019|2267627|8869|4729|
|2021_09_11_00_33_16|746|5222|19775060|17959044|29135|21900|

报告记录：目标帧层面“硬回波有效槽位变化”与“检测匹配状态变化”的描述性相关系数0.2897429700；9个scene层面的槽位保留比与检测率变化相关系数0.8277735839。连续帧和目标实例未作为独立scene；causal_claim=false，没有反事实干预或唯一原因证明。真实雾天验证仍缺失。

### 60.7 证据位置

服务器根目录下AP_PROTOCOL_AUDIT.json、AP_PROTOCOL_COMPARISON.md、GT_DIFFERENCE.json、MATCHING_DIFFERENCE.json、FACTOR_CONTRIBUTIONS.json、POINT_FLOW_SUMMARY.json、POINT_FLOW_BY_DISTANCE.json、OBJECT_POINT_FLOW.json、NEAR_FIELD_WEATHER_FLOW.json、OBSERVATION_DETECTION_ASSOCIATION.json、EXAMPLE_TARGETS.md及FINAL_DIAGNOSIS.md保存完整结果。逐帧GT/匹配分片在A/frames，逐点流统计分片在B/frames。

本机核验记录与精确数值：`logs/scorw-diagnostics-review-20261030/VERIFIED_RESULTS.json`。本次仅只读获取服务器既有结果并修改本机文档，未重跑实验、未修改服务器结果。

## 61. GGCA A2内容、权重与三天气评测记录（2026-10-30）

A2在A1可靠性感知聚合上增加柱内点对几何约束：由距离、共同邻域支持和特征相似性得到G，以w_ij=a_ij·G_ij控制邻居交换，剩余权重保留自身特征；柱内汇总增加有界几何支持奖励。保留原GSPR可靠性加权和官方PFN完整路径，不额外执行原BatchNorm，不新增参数或辅助损失。A2不包含A3的稀疏观测保护。具体公式及版本见第58节和gspr_ggca_geo_v2/METHOD_DESIGN.md。

训练完成20/20轮，按Clean validation O AP70选中第20轮，选模值70.1797%。权重：

`/home/cjm/datasets/logs/gspr-ggca-geometry-v2-20261028-r2/A2/epochs/20/checkpoint_f29e34b8bfa5f22456472b81b5921aa580b62fd29098161db866b7b8646ac727.pth`

选模记录所报checkpoint SHA256：`f29e34b8bfa5f22456472b81b5921aa580b62fd29098161db866b7b8646ac727`。

用户在物理卡1使用该权重完成独立三天气完整验证评测（Clean/Fog/Snow各829帧，非正式test集），未重新训练。脚本输出COMPLETE A2 evaluation；按脚本完成条件，两组汇总status=complete、full_validation=true、frame_count=829，三天气齐全，A2评测权重身份与选模记录一致，A1/A2冻结真值身份一致。本节依据用户粘贴的服务器输出登记，本次未再次连接服务器或独立核验权重文件字节哈希。

AP单位为%，差值为A2−A1，单位百分点；以下保留服务器输出的四位小数，差值不由已舍入的AP重新计算。A1参照为第59节选中第20轮的正式评测。

| 协议 | 天气 | A2 AP30 | A2 AP50 | A2 AP70 | ΔAP30 | ΔAP50 | ΔAP70 |
|---|---|---:|---:|---:|---:|---:|---:|
| O | Clean | 85.4711 | 82.8457 | 70.1797 | -0.8502 | -0.7218 | -1.0384 |
| O | Fog | 88.5384 | 87.4817 | 76.9111 | -1.0767 | -1.1716 | -1.8000 |
| O | Snow | 63.8460 | 59.6957 | 41.7613 | -1.5905 | -1.8889 | -3.3170 |
| O* | Clean | 85.4573 | 82.8324 | 70.1685 | -0.8223 | -0.6949 | -1.0154 |
| O* | Fog | 88.4661 | 87.4103 | 76.8483 | -1.1002 | -1.1948 | -1.8199 |
| O* | Snow | 63.5079 | 59.3796 | 41.5401 | -1.6044 | -1.8999 | -3.3148 |
| F | Clean | 85.4573 | 82.8324 | 70.1685 | -0.8223 | -0.6949 | -1.0154 |
| F | Fog | 52.4530 | 51.6535 | 45.4002 | -0.6493 | -0.6894 | -1.0587 |
| F | Snow | 59.9247 | 55.9468 | 39.1579 | -1.4859 | -1.7518 | -3.0305 |

结果目录：`/home/cjm/datasets/logs/gspr-ggca-geometry-v2-20261028-r2/A2/manual_evaluation_gpu1_20261030_162538/`；完整结果为其中的`summary.json`，A2−A1对照为`A2_MINUS_A1.txt`。A1参照文件：`/home/cjm/datasets/logs/gspr-ggca-v3/A1/final_evaluation/summary.json`。本记录为独立手动评测输出，不据此登记统一调度器A2_evaluate阶段完成或A3结果。

## 62. GGCA A0—A3完整实验结束与正式结果（2026-11-01只读核验）

### 62.1 完成状态及记录范围

远程读取UTC时间：2026-11-01T01:10:12.437041+00:00。服务器current_stage为final/complete，STOP.json不存在；FINAL_RESULTS.json为complete。最终阶段完成时间（北京时间）：2026-11-01T01:43:45.400722+08:00。
本次只读取已有报告、选模、权重、评测及诊断文件，核验SHA256并汇总；未启动训练或推理，未修改服务器代码、合同、完成标记或进程。A0复用旧B0，A1保留原源码续跑，A2/A3使用几何修订2，A3额外启用稀疏保护修订2。四组均有20轮记录，三天气各829帧完整验证评测；正式test集未参与。
本节更新第58节的历史运行进度，并补全第59节A1、第61节A2记录。第61节手动评测原文保留；本次统一调度器A2_evaluate已完成，正式final_evaluation的AP与第61节所列四位小数一致。以下全部采用统一正式结果目录，不以手动评测替代阶段完成标记。

### 62.2 训练、选模及权重核验

| 模型 | 完整轮数 | 选中轮次 | 选模Clean O AP70 (%) | 轮内累计耗时(h) | 峰值已分配显存(GiB) |
|---|---|---|---|---|---|
| A0 | 20 | 19 | 67.904234 | 19.049116 | 18.032147 |
| A1 | 20 | 20 | 71.218161 | 29.995630 | 18.761112 |
| A2 | 20 | 20 | 70.179732 | 32.285480 | 18.759445 |
| A3 | 20 | 18 | 70.033229 | 33.424479 | 18.763037 |

轮内耗时为原history.duration_seconds累计，包含该实现轮内训练、验证、Clean选模和保存；不是纯训练计算耗时。峰值为原记录peak_memory_bytes，不等同整卡占用。选模规则均为完整829帧Clean validation O AP70最大值，精确同分选较早轮次，种子20261017。训练20轮、batch size 8、8083协作帧、每轮1010次更新、Clean/Fog=4/7与3/7、Snow=0；各车独立天气采样。

A0：训练目录`/data/cjm/datasets/logs/gspr-repe-rocm-20261025-v4/B0`；选模记录`/data/cjm/datasets/logs/gspr-repe-rocm-20261025-v4/B0/selected.json`。
选中权重`/data/cjm/datasets/logs/gspr-repe-rocm-20261025-v4/B0/epochs/19/checkpoint_7c75ddef20d3042c27f57f07f9f3366f86d9758bcbab5a5e1730aec2bb00c1c6.pth`；实际SHA256=`7c75ddef20d3042c27f57f07f9f3366f86d9758bcbab5a5e1730aec2bb00c1c6`。
正式评测`/data/cjm/datasets/logs/gspr-repe-rocm-20261025-v4/B0/final_evaluation/summary.json`；summary SHA256=`28604af612aa2aad6bf88ce2bbca01d8b9970b8d02c88d864562932f2ca59b0c`。权重/选模/评测身份一致=True，summary声明的输出文件哈希不匹配数=0。

A1：训练目录`/home/cjm/datasets/logs/gspr-ggca-v3/A1`；选模记录`/home/cjm/datasets/logs/gspr-ggca-v3/A1/selected.json`。
选中权重`/home/cjm/datasets/logs/gspr-ggca-v3/A1/epochs/20/checkpoint_4b267b6c71aa7a15aacc48d1b73af4f1d380f086acef8b94be887371f832799a.pth`；实际SHA256=`4b267b6c71aa7a15aacc48d1b73af4f1d380f086acef8b94be887371f832799a`。
正式评测`/home/cjm/datasets/logs/gspr-ggca-v3/A1/final_evaluation/summary.json`；summary SHA256=`0ff1f01addcb59f2145c84e0ec123f461531fd5aad43f60b8fd34df37960f09a`。权重/选模/评测身份一致=True，summary声明的输出文件哈希不匹配数=0。

A2：训练目录`/home/cjm/datasets/logs/gspr-ggca-geometry-v2-20261028-r2/A2`；选模记录`/home/cjm/datasets/logs/gspr-ggca-geometry-v2-20261028-r2/A2/selected.json`。
选中权重`/home/cjm/datasets/logs/gspr-ggca-geometry-v2-20261028-r2/A2/epochs/20/checkpoint_f29e34b8bfa5f22456472b81b5921aa580b62fd29098161db866b7b8646ac727.pth`；实际SHA256=`f29e34b8bfa5f22456472b81b5921aa580b62fd29098161db866b7b8646ac727`。
正式评测`/home/cjm/datasets/logs/gspr-ggca-geometry-v2-20261028-r2/A2/final_evaluation/summary.json`；summary SHA256=`d1e441a4c074dacf56db1e5043c891c14c5f5849d5908c9d9c6208dfe8f8f2e2`。权重/选模/评测身份一致=True，summary声明的输出文件哈希不匹配数=0。

A3：训练目录`/home/cjm/datasets/logs/gspr-ggca-geometry-v2-20261028-r2/A3`；选模记录`/home/cjm/datasets/logs/gspr-ggca-geometry-v2-20261028-r2/A3/selected.json`。
选中权重`/home/cjm/datasets/logs/gspr-ggca-geometry-v2-20261028-r2/A3/epochs/18/checkpoint_b262f9a36f3f7e6c4ae725aaad675288da80afc344aad11796991145a9936d87.pth`；实际SHA256=`b262f9a36f3f7e6c4ae725aaad675288da80afc344aad11796991145a9936d87`。
正式评测`/home/cjm/datasets/logs/gspr-ggca-geometry-v2-20261028-r2/A3/final_evaluation/summary.json`；summary SHA256=`fb1bd7c2fb422d75d6881cb5749dd650a8e7ae7d68d58b9d9af7f78f4d58d084`。权重/选模/评测身份一致=True，summary声明的输出文件哈希不匹配数=0。

### 62.3 三天气完整AP表

AP为原值乘100，单位%。O为主协议：按天气过滤真值，预测None跳过整帧；O*使用相同真值但保留None帧；F固定Clean真值并保留None帧。

| 模型 | 天气 | 协议 | AP30 | AP50 | AP70 |
|---|---|---|---|---|---|
| A0 | clean | O | 83.979137 | 81.742002 | 67.904234 |
| A0 | clean | O* | 83.938659 | 81.702602 | 67.871504 |
| A0 | clean | F | 83.938659 | 81.702602 | 67.871504 |
| A0 | fog | O | 88.274805 | 87.002906 | 75.520916 |
| A0 | fog | O* | 88.226738 | 86.955532 | 75.479794 |
| A0 | fog | F | 52.065683 | 51.315500 | 44.543265 |
| A0 | snow | O | 60.212562 | 56.204389 | 39.197022 |
| A0 | snow | O* | 60.027419 | 56.031570 | 39.076498 |
| A0 | snow | F | 56.458951 | 52.700645 | 36.753506 |
| A1 | clean | O | 86.321270 | 83.567534 | 71.218161 |
| A1 | clean | O* | 86.279663 | 83.527254 | 71.183833 |
| A1 | clean | F | 86.279663 | 83.527254 | 71.183833 |
| A1 | fog | O | 89.615108 | 88.653389 | 78.711113 |
| A1 | fog | O* | 89.566312 | 88.605116 | 78.668253 |
| A1 | fog | F | 53.102254 | 52.342937 | 46.458874 |
| A1 | snow | O | 65.436458 | 61.584579 | 45.078253 |
| A1 | snow | O* | 65.112293 | 61.279496 | 44.854941 |
| A1 | snow | F | 61.410634 | 57.698606 | 42.188436 |
| A2 | clean | O | 85.471060 | 82.845710 | 70.179732 |
| A2 | clean | O* | 85.457327 | 82.832399 | 70.168456 |
| A2 | clean | F | 85.457327 | 82.832399 | 70.168456 |
| A2 | fog | O | 88.538423 | 87.481748 | 76.911127 |
| A2 | fog | O* | 88.466107 | 87.410296 | 76.848308 |
| A2 | fog | F | 52.452955 | 51.653541 | 45.400209 |
| A2 | snow | O | 63.845982 | 59.695694 | 41.761274 |
| A2 | snow | O* | 63.507884 | 59.379574 | 41.540126 |
| A2 | snow | F | 59.924698 | 55.946823 | 39.157894 |
| A3 | clean | O | 86.762302 | 83.084006 | 70.033229 |
| A3 | clean | O* | 86.720482 | 83.043959 | 69.999473 |
| A3 | clean | F | 86.720482 | 83.043959 | 69.999473 |
| A3 | fog | O | 90.462210 | 88.760387 | 77.405861 |
| A3 | fog | O* | 90.412952 | 88.712056 | 77.363712 |
| A3 | fog | F | 53.649067 | 52.507187 | 45.800558 |
| A3 | snow | O | 69.158442 | 64.409248 | 45.658541 |
| A3 | snow | O* | 68.804026 | 64.079170 | 45.424555 |
| A3 | snow | F | 64.790906 | 60.324588 | 42.724188 |

### 62.4 消融差值

差值由未舍入原数计算，单位百分点，符号保持；不作效果归因。

| 对照 | 天气 | 协议 | ΔAP30 | ΔAP50 | ΔAP70 |
|---|---|---|---|---|---|
| A1-A0 | clean | O | +2.342133 | +1.825532 | +3.313926 |
| A1-A0 | clean | O* | +2.341004 | +1.824652 | +3.312329 |
| A1-A0 | clean | F | +2.341004 | +1.824652 | +3.312329 |
| A1-A0 | fog | O | +1.340303 | +1.650483 | +3.190196 |
| A1-A0 | fog | O* | +1.339573 | +1.649584 | +3.188459 |
| A1-A0 | fog | F | +1.036571 | +1.027437 | +1.915609 |
| A1-A0 | snow | O | +5.223896 | +5.380190 | +5.881231 |
| A1-A0 | snow | O* | +5.084875 | +5.247926 | +5.778443 |
| A1-A0 | snow | F | +4.951683 | +4.997961 | +5.434930 |
| A2-A1 | clean | O | -0.850210 | -0.721825 | -1.038429 |
| A2-A1 | clean | O* | -0.822335 | -0.694856 | -1.015377 |
| A2-A1 | clean | F | -0.822335 | -0.694856 | -1.015377 |
| A2-A1 | fog | O | -1.076686 | -1.171641 | -1.799985 |
| A2-A1 | fog | O* | -1.100205 | -1.194821 | -1.819945 |
| A2-A1 | fog | F | -0.649299 | -0.689396 | -1.058665 |
| A2-A1 | snow | O | -1.590475 | -1.888885 | -3.316979 |
| A2-A1 | snow | O* | -1.604409 | -1.899922 | -3.314815 |
| A2-A1 | snow | F | -1.485936 | -1.751783 | -3.030541 |
| A3-A0 | clean | O | +2.783165 | +1.342004 | +2.128995 |
| A3-A0 | clean | O* | +2.781823 | +1.341357 | +2.127969 |
| A3-A0 | clean | F | +2.781823 | +1.341357 | +2.127969 |
| A3-A0 | fog | O | +2.187405 | +1.757481 | +1.884944 |
| A3-A0 | fog | O* | +2.186214 | +1.756524 | +1.883918 |
| A3-A0 | fog | F | +1.583385 | +1.191687 | +1.257292 |
| A3-A0 | snow | O | +8.945880 | +8.204860 | +6.461519 |
| A3-A0 | snow | O* | +8.776607 | +8.047600 | +6.348057 |
| A3-A0 | snow | F | +8.331955 | +7.623943 | +5.970682 |
| A3-A2 | clean | O | +1.291242 | +0.238296 | -0.146503 |
| A3-A2 | clean | O* | +1.263154 | +0.211560 | -0.168983 |
| A3-A2 | clean | F | +1.263154 | +0.211560 | -0.168983 |
| A3-A2 | fog | O | +1.923788 | +1.278639 | +0.494733 |
| A3-A2 | fog | O* | +1.946845 | +1.301760 | +0.515403 |
| A3-A2 | fog | F | +1.196113 | +0.853646 | +0.400348 |
| A3-A2 | snow | O | +5.312460 | +4.713554 | +3.897268 |
| A3-A2 | snow | O* | +5.296141 | +4.699596 | +3.884429 |
| A3-A2 | snow | F | +4.866208 | +4.377765 | +3.566293 |

### 62.5 检测计数与计分真值

以下正确匹配、误检、漏检使用F协议、IoU=0.7，在829帧逐帧原记录上求和。O/O*计数保留在本地完整JSON中。

| 模型 | 天气 | 正确匹配 | 误检 | 漏检 | 计分GT O/O*/F |
|---|---|---|---|---|---|
| A0 | clean | 4998 | 2010 | 1226 | 6221/6224/6224 |
| A0 | fog | 3222 | 769 | 3002 | 3671/3673/6224 |
| A0 | snow | 3027 | 1865 | 3197 | 5836/5854/6224 |
| A1 | clean | 5102 | 1718 | 1122 | 6221/6224/6224 |
| A1 | fog | 3286 | 647 | 2938 | 3671/3673/6224 |
| A1 | snow | 3327 | 1643 | 2897 | 5825/5854/6224 |
| A2 | clean | 5074 | 1819 | 1150 | 6223/6224/6224 |
| A2 | fog | 3271 | 764 | 2953 | 3670/3673/6224 |
| A2 | snow | 3264 | 1739 | 2960 | 5823/5854/6224 |
| A3 | clean | 5083 | 1787 | 1141 | 6221/6224/6224 |
| A3 | fog | 3244 | 660 | 2980 | 3671/3673/6224 |
| A3 | snow | 3427 | 1710 | 2797 | 5824/5854/6224 |

### 62.6 按距离及点数分层的召回

F协议、IoU=0.7。距离为目标中心到自车的平面距离；点数为各协作车原始天气点云落框点数之和，不去重。表内为匹配数/真值数（召回率%）。

| 天气 | 分层 | 区间 | A0 | A1 | A2 | A3 |
|---|---|---|---|---|---|---|
| clean | distance_ego_m | 0-30 | 1914/2138 (89.5229) | 1932/2138 (90.3648) | 1950/2138 (91.2067) | 1932/2138 (90.3648) |
| clean | distance_ego_m | 30-60 | 1883/2045 (92.0782) | 1931/2045 (94.4254) | 1892/2045 (92.5183) | 1910/2045 (93.3985) |
| clean | distance_ego_m | 60+ | 1201/2041 (58.8437) | 1239/2041 (60.7055) | 1232/2041 (60.3626) | 1241/2041 (60.8035) |
| clean | native_points_sum_cavs | 1-5 | 174/683 (25.4758) | 203/683 (29.7218) | 197/683 (28.8433) | 188/683 (27.5256) |
| clean | native_points_sum_cavs | 21+ | 4017/4488 (89.5053) | 4084/4488 (90.9982) | 4077/4488 (90.8422) | 4077/4488 (90.8422) |
| clean | native_points_sum_cavs | 6-20 | 807/1053 (76.6382) | 815/1053 (77.3979) | 800/1053 (75.9734) | 818/1053 (77.6828) |
| fog | distance_ego_m | 0-30 | 1915/2138 (89.5697) | 1920/2138 (89.8036) | 1940/2138 (90.7390) | 1908/2138 (89.2423) |
| fog | distance_ego_m | 30-60 | 1068/2045 (52.2249) | 1118/2045 (54.6699) | 1079/2045 (52.7628) | 1084/2045 (53.0073) |
| fog | distance_ego_m | 60+ | 239/2041 (11.7099) | 248/2041 (12.1509) | 252/2041 (12.3469) | 252/2041 (12.3469) |
| fog | native_points_sum_cavs | 0 | 0/2551 (0.0000) | 2/2551 (0.0784) | 3/2551 (0.1176) | 7/2551 (0.2744) |
| fog | native_points_sum_cavs | 1-5 | 1/33 (3.0303) | 1/33 (3.0303) | 3/33 (9.0909) | 2/33 (6.0606) |
| fog | native_points_sum_cavs | 21+ | 3203/3574 (89.6195) | 3267/3574 (91.4102) | 3250/3574 (90.9345) | 3221/3574 (90.1231) |
| fog | native_points_sum_cavs | 6-20 | 18/66 (27.2727) | 16/66 (24.2424) | 15/66 (22.7273) | 14/66 (21.2121) |
| snow | distance_ego_m | 0-30 | 1170/2138 (54.7240) | 1327/2138 (62.0674) | 1266/2138 (59.2142) | 1369/2138 (64.0318) |
| snow | distance_ego_m | 30-60 | 1270/2045 (62.1027) | 1378/2045 (67.3839) | 1353/2045 (66.1614) | 1402/2045 (68.5575) |
| snow | distance_ego_m | 60+ | 587/2041 (28.7604) | 622/2041 (30.4753) | 645/2041 (31.6022) | 656/2041 (32.1411) |
| snow | native_points_sum_cavs | 0 | 0/370 (0.0000) | 0/370 (0.0000) | 2/370 (0.5405) | 0/370 (0.0000) |
| snow | native_points_sum_cavs | 1-5 | 309/1209 (25.5583) | 368/1209 (30.4384) | 348/1209 (28.7841) | 347/1209 (28.7014) |
| snow | native_points_sum_cavs | 21+ | 2141/3609 (59.3239) | 2355/3609 (65.2535) | 2306/3609 (63.8958) | 2477/3609 (68.6340) |
| snow | native_points_sum_cavs | 6-20 | 577/1036 (55.6950) | 604/1036 (58.3012) | 608/1036 (58.6873) | 603/1036 (58.2046) |

### 62.7 逐目标变化及定位记录

每个目标按帧计数，F协议、IoU=0.7。恢复=右侧模型未匹配、左侧匹配；丢失反之。共同匹配目标的IoU差值只在两组均匹配的目标上统计。

| 对照 | 天气 | 恢复 | 丢失 | 误检差 | 漏检差 | 共同匹配数 | 共同IoU平均差 |
|---|---|---|---|---|---|---|---|
| A1-A0 | clean | 314 | 210 | -292 | -104 | 4788 | +0.007962 |
| A1-A0 | fog | 168 | 104 | -122 | -64 | 3118 | +0.008974 |
| A1-A0 | snow | 610 | 310 | -222 | -300 | 2717 | +0.007308 |
| A2-A1 | clean | 220 | 248 | 101 | 28 | 4854 | -0.003240 |
| A2-A1 | fog | 103 | 118 | 117 | 15 | 3168 | -0.006286 |
| A2-A1 | snow | 394 | 457 | 96 | 63 | 2870 | -0.005443 |
| A3-A0 | clean | 276 | 191 | -223 | -85 | 4807 | +0.004936 |
| A3-A0 | fog | 149 | 127 | -109 | -22 | 3095 | +0.004433 |
| A3-A0 | snow | 686 | 286 | -155 | -400 | 2741 | +0.005862 |
| A3-A2 | clean | 219 | 210 | -32 | -9 | 4864 | -0.000009 |
| A3-A2 | fog | 107 | 134 | -104 | 27 | 3137 | +0.001087 |
| A3-A2 | snow | 542 | 379 | -29 | -163 | 2885 | +0.003485 |

| 模型 | 天气 | 匹配目标数 | 已匹配目标平均IoU |
|---|---|---|---|
| A0 | clean | 4998 | 0.851704 |
| A0 | fog | 3222 | 0.858325 |
| A0 | snow | 3027 | 0.832656 |
| A1 | clean | 5102 | 0.858322 |
| A1 | fog | 3286 | 0.866285 |
| A1 | snow | 3327 | 0.836034 |
| A2 | clean | 5074 | 0.855473 |
| A2 | fog | 3271 | 0.860573 |
| A2 | snow | 3264 | 0.831035 |
| A3 | clean | 5083 | 0.855827 |
| A3 | fog | 3244 | 0.862603 |
| A3 | snow | 3427 | 0.833336 |

已匹配目标平均IoU的样本集合随模型变化；逐帧恢复/丢失目标ID、共同匹配IoU差以及帧标识完整保留于REMOTE_FINAL_RECORD.json。

### 62.8 诊断范围、保护及参数更新记录

mechanism状态=complete，实际144份记录（4模型×3天气×12帧），全部哈希一致；该阶段新优化步数0，AP_recomputed=False，使用匹配输入，真值仅在前向完成后用于分组统计。可靠性/不确定性直方图、点数、共识、修正强度及目标保护点数逐帧保留在JSON。A1旧共识与A2/A3新绝对支持的定义不同，不合并为同一量。

| 模型 | 训练诊断记录数 | 存在非零梯度的记录数 | 存在非零参数更新的记录数 |
|---|---|---|---|
| A0 | 20200 | 0 | 0 |
| A1 | 20200 | 20200 | 20200 |
| A2 | 20200 | 20200 | 20200 |
| A3 | 20200 | 20200 | 20200 |

本表统计新增聚合分支的诊断字段；A0不包含该分支，缺失字段按0条计数，不表示A0主干训练梯度或更新为零。

| 天气 | 诊断目标记录数 | 有保护点的目标记录数 | 保护点数合计 | 其中恢复记录数 | 其中丢失记录数 |
|---|---|---|---|---|---|
| clean | 230 | 230 | 63850 | 17 | 8 |
| fog | 230 | 216 | 61036 | 20 | 11 |
| snow | 230 | 227 | 42657 | 34 | 14 |

上表仅前12帧；目标保护点数为落入目标框的保护点累计，不能当作全829帧统计，也不表明恢复或丢失由保护操作导致。

### 62.9 匹配输入推理开销原记录

| 模型 | 天气 | 帧数 | 参数量 | 同步前向均值(ms) | 峰值已分配显存(MiB) |
|---|---|---|---|---|---|
| A0 | clean | 12 | 6593656 | 432.417716 | 320.379395 |
| A0 | fog | 12 | 6593656 | 29.130562 | 506.482422 |
| A0 | snow | 12 | 6593656 | 29.620471 | 506.482422 |
| A1 | clean | 12 | 6599226 | 125.542340 | 501.184082 |
| A1 | fog | 12 | 6599226 | 105.502939 | 506.482422 |
| A1 | snow | 12 | 6599226 | 107.933359 | 506.482422 |
| A2 | clean | 12 | 6599226 | 132.268736 | 506.482422 |
| A2 | fog | 12 | 6599226 | 117.689067 | 506.482422 |
| A2 | snow | 12 | 6599226 | 119.824783 | 506.482422 |
| A3 | clean | 12 | 6599226 | 134.732320 | 506.482422 |
| A3 | fog | 12 | 6599226 | 120.626233 | 506.482422 |
| A3 | snow | 12 | 6599226 | 122.854147 | 680.200195 |

按服务器原报告直接登记，未剔除首帧、未重新计时；A0 Clean均值包含原记录中的首帧4.860109953秒。上述值不代表纯模块开销或整卡显存。

### 62.10 版本、阶段与证据

version_fairness.status=complete；A0_retrained=False；A1_restarted_from_epoch_zero=False；sampling_trace_equal=True。合同核对字段：epochs, policy, train_manifest_sha256, train_manifest_identity, train_data_acceptance, pair_identity, config_sha256, input_identity, frozen_gt, loader, scheduler, training_GT, gspr_frozen_settings。
- A1合同`/home/cjm/datasets/logs/gspr-ggca-v3/A1/contract.json`，SHA256=`8eb034a17485571e25ded5f98b5c146fdfc7e0d4790d501c56002c85721919e2`，source_revision=original-A1。
- A2合同`/home/cjm/datasets/logs/gspr-ggca-geometry-v2-20261028-r2/A2/contract.json`，SHA256=`c67d70be710f7f52831233606d7952b3126fbeb27b8f3f79c36daaf406a400b4`，source_revision=pair-exchange-v2。
- A3合同`/home/cjm/datasets/logs/gspr-ggca-geometry-v2-20261028-r2/A3/contract.json`，SHA256=`a43de85c46b357893306a162a52e11b76e9d5b97440c24d10c79a57a01e34b8a`，source_revision=pair-exchange-v2。

实际读取到的完成阶段：A0_reference, A2_evaluate, A2_train, A3_evaluate, A3_train, final, mechanism, preflight, preserved_A1_evaluate, preserved_A1_train, self-test, smoke_A2, smoke_A3, smoke_eval_A2, smoke_eval_A3, stage1_acceptance。
正式报告：`/home/cjm/datasets/logs/gspr-ggca-geometry-v2-20261028-r2/reports/FINAL_RESULTS.json`（389063154字节），SHA256=`aca3558e891be65b22ff58233eedd8ae5063b9e12383bae18f9921e3f30dc2c8`。配套FINAL_RESULTS.md、ABLATION_TABLE.csv、FAILURE_CASES.md由服务器原结果脚本生成。
本地完整提取证据：`delivery/ggca_final_20261101/REMOTE_FINAL_RECORD.json`，保留全部逐目标变化、分层记录、144帧诊断、阶段记录、精确AP和哈希核验；为避免重复复制389MB逐批训练数据，仅提取训练history的逐轮摘要及梯度/更新记录计数。
范围限制按原报告登记：单训练种子；验证集兼用于选模；未评测test；几何量为局部统计量，无点噪声真值；目标关联诊断仅覆盖各天气前12帧。未补做实验或加入机制成立、因果关系、统计显著性结论。

## 63. A2/A3已有权重低成本机制诊断（2026-11-01海光完成）

独立输出`/home/cjm/datasets/logs/ggca-fixed-weight-diagnostic-20261101_094700-r4/`，最终status=complete；两套原选模权重、三天气各固定24个均匀索引、每组5个开关，共720次前向，0次训练优化。GPU2，进程1226512完成后退出；原模型、BN及源码哈希未变。不同开关共用首次GSPR/PFN特征前缀，记录但不采用后续HIP重复计算微小差异。初稿/r2因重复PFN位差中止，r3错误沿用Clean配置目录导致天气PCD缺失，全部保留且排除结果；最终r4使用已审计三天气overlay，并对所有选中PCD核对路径和SHA256。

A3固定权重内部开启保护未改变几何支持或交换关系（逐值检查），保护使总残差范数降低94.807145%，平均残差门从0.286766降至0.005133；1/2点柱体100%关闭残差，3–5/6–10/11–20/21–32点柱体范数分别降低99.889/98.679/95.636/89.105%。每套权重共统计428405个逐帧柱体记录，不是独立物理目标数。

以下F协议AP70单位%，每天气仅24帧；统一削弱在公式上匹配保护的残差总量，浮点舍入误差最大1.90735e-6；不能当作829帧正式成绩。

| 权重 | 天气 | 完整修正 | 自适应保护 | 等量统一削弱 | 完全关闭 |
|---|---|---:|---:|---:|---:|
| A2 | clean | 64.9526 | 61.7176 | 61.6340 | 61.6820 |
| A2 | fog | 45.3236 | 42.5326 | 43.9553 | 42.5158 |
| A2 | snow | 38.7602 | 35.4623 | 34.6503 | 31.8218 |
| A3 | clean | 65.8800 | 66.5072 | 66.5072 | 66.5072 |
| A3 | fog | 44.0284 | 45.1165 | 45.1165 | 45.3800 |
| A3 | snow | 40.0568 | 39.6117 | 39.6117 | 39.6117 |

A3权重下自适应保护与等量统一削弱三天气AP70相同；Clean/Snow还与完全关闭相同，Fog低于完全关闭0.2635个百分点。A2权重加保护在三天气均低于完整修正，因此不能将所有修正笼统称为有害。本次没有验证几何判断准确率提高；训练权重的间接影响未被该推理开关隔离。单个打乱种子、小样本，不作显著性结论。

精确32点数分组、消息交换/自身保留、残差强度、逐目标ID、错误记录和指令在`delivery/ggca_fixed_diagnostic_20261101/{REMOTE_SUMMARY.json,DIAGNOSTIC_RESULTS.md,RUN_RECORD.md}`。这份诊断只用于形成新实验假设，不是新E1/E2的结果。

## 64. E0/E1/E2独立消融：设计、部署与验收（2026-11-01）

用户最后明确要求上传海光并并行启动两组，覆盖同条任务前部“不自动启动正式训练”的原限制。按此授权执行带验收门禁的MODE=all：结构和真实数据小训练/推理/开关测试全部通过之前，不得进入正式训练。没有创建branch或worktree，没有以Git状态作为服务器前提；旧GSPR、GGCA和REPE源码/权重/结果保持只读。

### 64.1 三组组成及初始化

- E0=原GGCA A1，直接引用`/home/cjm/datasets/logs/gspr-ggca-v3/A1/`已有选模第20轮及三天气829帧正式结果，不重新训练。权重SHA256=`4b267b6c71aa7a15aacc48d1b73af4f1d380f086acef8b94be887371f832799a`。
- E1=A1+点级自特征保留S，不使用A2几何共识，不乘旧A3的整柱残差充分性门。
- E2=A1+旧B2的普通八邻域Enhancement，在Scatter后接入；不使用旧B1 Aggregation、Observation状态网络、额外质量来源门或辅助损失。
- E1/E2独立对齐A1原`resume_states/00.pth`全部192个共享张量，SHA256=`8e360fe0eccf6dcdd37076e8f978c09660a055e771ab93e62c374839b6c7079f`。E2额外C张量只取B2第0轮，SHA256=`08179a633d0e538bc2e268cb27f4fa127e13c603c95a51a407648cdb90235aad`；两份初始状态的共同主干/GSPR逐值相同。没有把训练后A1/B2最佳权重用于新组预训练。

### 64.2 S的数学定义与边界

从A1同一次官方PFN/BN输出取得已受GSPR影响的点特征f；原GSPR装饰特征乘r、Linear/BN/ReLU/Max及A1邻居关系/可靠性池化/柱体残差全部保留。只在原A1邻居消息m之后增加一次凸混合，不重新执行原BN，不替换PFN forward。

p_i=r_i(1-u_i)；q_i=sum_j(a_ij*p_j)；e_i=sum_j(valid_ij*p_j)/K，K=min(4,槽位数)。q为实际聚合邻居的可信程度，e为绝对可信邻居观测量，不把邻居少直接判噪。

alpha_i = 0.5*p_i*sigmoid(b + softplus(w0)*(p_i-q_i) + softplus(w1)*(1-e_i) - softplus(w2)*u_i)

m_new_i = alpha_i*f_i + (1-alpha_i)*m_A1_i

仅4个参数，b初值−1.5，正向增益初值[2,1,1]；alpha≤0.5*p，因此邻居消息至少保留一半。可信自身与低质邻居冲突时提高自保留，当前点不可信/不确定时不盲目保留。无邻居时A1已用自身，alpha记0且新消息不变；不删点、不造点、不把一两点柱体整体残差关掉。原A1整柱残差门完全未改。初始合成输入实际alpha为.319927/.082091/.012487，对应可信自身+差邻居/可信邻居/低可信自身；不是检测AP结果。

### 64.3 C及实验公平性

E2直接调用原`gspr_repe/modules.py::Enhancement`，`constrained=False,state_gate=False`。occupied仅由真实体素坐标与voxel_num_points>0构造；只允许已有观测网格作来源，目标网格不限有点与否。保留key/value、8邻域、exp(cosine)/(1+dx²+dy²)、归一化和sigmoid残差门（原bias=−5）；补零切片不环绕、不跨车辆。关闭C或零C残差应严格恢复A1。

沿用8083训练协作帧、20轮、batch8、seed20261017、Adam .002/eps1e-10/weight_decay1e-4、官方[10,15]调度语义；每车独立Clean4/7与Fog3/7，Snow0；每轮完整829帧Clean O AP70选模，同分早轮。完整三天气829帧O/O*/F、GT、匹配、后处理和检测/GSPR损失均复用原引擎，无辅助损失。最终自动报告E1−E0、E2−E0，并在旧B0/B2配方/初始化核对后输出描述性差中差(E2−E0)−(B2−B0)，不作单种子显著性或机制已成立保证。

### 64.4 机制诊断与验收设计

E1同一锁定权重比较adaptive、匹配有效邻居点平均alpha的uniform、按柱体点数分组打乱alpha、off；所有开关回放同一缓存GSPR/PFN前缀。E2同一权重比较C on/off。正式诊断预先固定24均匀索引/天气；工程小验收使用0/828两帧/天气，不据Snow性能调系数。

统计自保留均值/最大值/逐柱门控分布、原残差门及实际修正RMS；E2统计有点/空目标网格的接受数、非零增强数、强度和观测来源掩码。完整正式结果按F协议保留正确检测/误检/漏检、恢复/丢失ID、共同匹配IoU差、0–30/30–60/60+米召回，以及目标原始点数0/1–5/6–20/21+召回。Snow的1–5点指各CAV原始天气点云落框点数之和；与每柱截断到32槽的柱体点数分开统计。

测试包含S门控方向及参数更新、空/单/双/多点及填充NaN稳定性，C空目标传播/无观测不能作来源/无跨车及边界环绕；在非零A1残差下检查关闭/零新增修正的整模型输出和原BN逐位一致；新增参数梯度、Adam实际更新、完整模型/Adam/调度/损失累计/所有RNG状态及下一步恢复；独立小训练各两步和三天气各12帧推理。

### 64.5 工程状态、代码及路径（正式结果尚未完成）

本地独立源码`gspr_e1e2/`；服务器最终源码`/home/cjm/OpenCOOD-main/cjmnet/gspr_e1e2_v5`。唯一入口run_all.sh，POSIX sh/LF，支持MODE=preflight/smoke/all与RESUME=1。输出`/home/cjm/datasets/logs/gspr-e1e2-20261101-v5`；主日志`/home/cjm/datasets/logs/gspr_e1e2_v5.launch.log`。检查时2/4/6卡空闲，选择E1物理卡2、E2物理卡4，实际每阶段重新核对，拒绝抢占外部任务。

v1交付在启动前发现仅旧REPE历史测试文件/清单差异而拒绝；v2预检缺数据合同显式转接；v3未训练时主动停止以补原评测导出接口；v4完整预检及E0审计通过，正常HIP下一步微小位差触发合成恢复门禁停止。所有旧失败版本及记录保留，没有改旧哈希绕过检查。最终v5 S/C算法及初始系数从首稿起未根据Snow成绩变化。

海光额外对照：原A1/E1/E2默认下一步最大位差4.38e-8/7.82e-8/5.77e-8，保存状态本身精确恢复；严格确定性设置下三者下一步均0。HIP诊断histc不支持该模式，恢复单元测试只将无梯度直方图放CPU计算，其余实际模型计算保持，正式训练仍沿用原非确定性配置。关闭/零修正和BN检查没有放宽数值容差。v5辅助完整模型合成测试已实际通过，包含全部保存状态精确恢复、下一步模型/Adam精确、梯度及初始化匹配；主流程仍重新执行全量验收，辅助报告不能代替真实小训练门禁。

2026-11-01 11:01读取：v5主控PID1401477，MODE=all，仍在preflight，训练与三天气小验收未完成，E1/E2正式训练均未开始。后续以实际阶段日志及新追加的完成记录为准，不能将此处设计/已启动入口写作正式实验完成。

交付包`delivery/e1e2_20261101_v5/gspr_e1e2_v5.zip`，SHA256=`96e526880b5d9fb332d8df7d7a6cf40611631f750487b50af602702a85331b93`；24个新文件、464个实际依赖哈希，只有本机静态AST/POSIX/LF/哈希检查，所有Torch/数据/训练测试均在海光。只承诺完整轮次断点恢复，不声称逐批恢复。父控制器在两条独立GPU通道内完成小训练/验收后再并行正式训练及各自评测/机制诊断，最终自动汇总，技术失败保留已提交状态并停止本次通道，不停止其他实验。

仅在当前控制器退出后，用一条后台命令恢复：

```sh
nohup env BACKEND=rocm GPUS=2,4 MODE=all RESUME=1 sh /home/cjm/OpenCOOD-main/cjmnet/gspr_e1e2_v5/run_all.sh >> /home/cjm/datasets/logs/gspr_e1e2_v5.launch.log 2>&1 < /dev/null &
```

原始证据按文件哈希下载在`delivery/e1e2_20261101_v5/remote_evidence/`，观察时间及存在/缺少的文件见OBSERVATION.json；完整状态恢复对照保留在远端v4/resume_investigation/并随证据归档。不得用尚缺少的acceptance、selected或FINAL_RESULTS文件宣称通过或效果。

64节状态更新（2026-11-01 11:08，Asia/Shanghai，直接读取）：v5 preflight和self-test已complete，包含E0复用/全部数据身份审计、关闭与零新增修正整模型及BN精确等价、S/C数值/梯度/更新、完整保存状态与严格确定性下一步恢复。主控1401477进入parallel_smoke；E1/GPU2通道控制器1432981、E2/GPU4通道控制器1432984。两组实际小训练/三天气推理/两帧机制开关尚在执行，stage1_acceptance未完成，正式训练未启动。

2026-11-01 11:33补充实测：两组均完成2次真实优化更新，共享初始化192张量逐值一致，E1新增S为4参数，E2新增C为4673参数；总可训练参数6599230/6603899。第二次更新S梯度L2=2.488642394382623e-6，更新L2=0.0039993226528167725；C的key/value/gate权重及bias均有非零梯度、更新。该小训练峰值已分配显存19.19709539413452/26.818678379058838 GiB，不能作为全训练峰值保证。两组Clean/Fog小规模推理已完成，Snow及开关诊断仍在执行；正式20轮未开始。证据在delivery/e1e2_20261101_v5/remote_evidence/v5/smoke/{E1,E2}/history.json。

### 64.6 联合验收完成与正式并行入口（2026-11-01 11:45）

海光stage1/acceptance.json实际status=complete、stage2_allowed=true、E0_retrained=false。E1/E2各2次真实优化更新，完整模型/Adam/调度/损失/累积量/随机状态已保存；Clean/Fog/Snow各12帧O/O*/F推理完成；每天气索引0/828的E1 adaptive/uniform/shuffled/off及E2 on/off开关完成，模型及BN状态未变化。主控1401477于UTC03:45:10进入parallel_formal，E1通道1542778/训练进程1542803/GPU2，E2通道1542781/训练进程1542804/GPU4。当前正在正式训练启动前全量输入校验，尚未登记正式优化完成；正式成绩仍待后续20轮及完整评测。验收原文和SHA归档delivery/e1e2_20261101_v5/remote_evidence/。

### 64.7 正式优化已确认、交付归档（2026-11-01 11:52）

远程观测UTC：2026-11-01T03:52:18.876222+00:00。两组均已打印optimizer.step之后的`epoch=1/20 batch=0/1010`标记，随后逐批日志已到E1的12/1010与E2的13/1010，损失有限。仍处第1轮，未把这些批次写为已完成轮次、正式选模或最终天气成绩。

| 组别 | 物理GPU | 通道PID | 训练PID | 本次第0轮完整状态SHA256 |
|---|---:|---:|---:|---|
| E1 | 2 | 1542778 | 1542803 | `06b69ce7401853c25e80ffa445ebfffd1f58cfab80f08968bb7e4c86ff9cd136` |
| E2 | 4 | 1542781 | 1542804 | `3c12642ce51ddcfeef28357e6ff992665d85e7001e10b27d0c8a8197be22c992` |

主控1401477，stage=parallel_formal/running。观测时卡2/4显存使用13832/26946 MiB，利用率3.3%/94.2%，属于单次采样；输出盘剩余43.47734069824219 GiB。E1/E2正式初始化再次核对192个共享张量一致，未加载训练后的E0权重；E2额外C初始化与B2第0轮一致。

交付包：`delivery/e1e2_20261101_v5/gspr_e1e2_v5.zip`，SHA256=`96e526880b5d9fb332d8df7d7a6cf40611631f750487b50af602702a85331b93`。完整验收、实际进程命令、ROCR物理卡映射、源码/数据/合同/初始状态与阶段证据见`VALIDATION_AND_DEPLOYMENT.json`、`DELIVERY_REPORT.md`、`remote_evidence/OBSERVATION.json`（每份下载记录逐项校验SHA）。服务器代码快照和已完成状态不在归档阶段修改。

正式结果将由唯一入口自动写入`/home/cjm/datasets/logs/gspr-e1e2-20261101-v5/reports/FINAL_RESULTS.json`及`.md`；包含E0引用、E1/E2三天气三协议AP、差值、匹配/误检/漏检、距离与目标点数召回、恢复/丢失ID、机制控制和采样轨迹。当前无E1/E2正式效果结论。

主日志：`/home/cjm/datasets/logs/gspr_e1e2_v5.launch.log`；实时分组日志：`/home/cjm/datasets/logs/gspr-e1e2-20261101-v5/lanes/{E1,E2}/logs/train.log`。完整已提交轮次可恢复模型、Adam、调度、criterion、累积量、全部随机状态；未提交轮次重跑，不声称逐批恢复。后续主控退出后使用同一入口，不在仍存活时重复启动：

```sh
nohup env BACKEND=rocm GPUS=2,4 MODE=all RESUME=1 sh /home/cjm/OpenCOOD-main/cjmnet/gspr_e1e2_v5/run_all.sh >> /home/cjm/datasets/logs/gspr_e1e2_v5.launch.log 2>&1 < /dev/null &
```

交付记录已传至海光独立目录`/home/cjm/datasets/logs/gspr-e1e2-20261101-v5/deployment/20261101_1152/`，含VALIDATION_AND_DEPLOYMENT.json、DELIVERY_REPORT.md、AI_CONTEXT_E12_UPDATE.md、remote_evidence.zip、原始源码zip，逐份SHA已核对。最终再次执行服务器静态检查通过；实时状态存于`delivery/e1e2_20261101_v5/FINAL_LIVE_STATUS.txt`。未修改根目录历史源码/合同，未停止其他任务；当前两组独立正式训练继续运行。

最后只读快照UTC2026-11-01T03:54:11.861014+00:00（北京时间11:54）：E1/E2仍为第1/20轮，日志分别推进到33/1010与35/1010批，训练PID和物理GPU映射保持上述值，联合验收仍complete，未见STOP记录。

## 65. E0/E1/E2正式实验完成与客观结果（2026-11-03海光只读核验）

### 65.1 完成状态、范围及权重

读取时间UTC：2026-11-03T01:01:06.320775+00:00。主流程final/complete，FINAL_RESULTS.status=complete，STOP.json不存在。E1/E2各完成20轮；E0为原GGCA A1复用，E0_retrained=False。三组采样轨迹逐轮哈希一致：True。

输出根目录：`/home/cjm/datasets/logs/gspr-e1e2-20261101-v5`。所有正式天气评测均为validation，Clean/Fog/Snow各829帧；本次只读取、校验和记录，未启动任何训练或推理。

| 模型 | 已完成轮数 | 选中轮 | checkpoint SHA256 |
|---|---:|---:|---|
| E0 | 20 | 20 | `4b267b6c71aa7a15aacc48d1b73af4f1d380f086acef8b94be887371f832799a` |
| E1 | 20 | 20 | `98852dbbf45b03def5a9c187f500e09349dd96d33eaae6faa14538b5e326b931` |
| E2 | 20 | 15 | `c8d84dc7012a1d2f9511d21c117bb78f881ad5b10631b6dd0eff7b1163bde906` |

E1/E2选中权重文件实算SHA与selected.json一致；每组均按20个候选的Clean O AP70选模，同分取早轮。初始化、20轮/batch8/seed20261017、每车独立Clean4/7与Fog3/7、Adam及原调度规则沿用§64；无Snow训练或Test评测。

### 65.2 三天气三协议正式AP

单位为%；O为原官方计分，O*补入None帧的官方真值，F使用固定Clean真值。协议细节以归档protocol.json为准。

| 模型 | 天气 | 协议 | AP30 | AP50 | AP70 |
|---|---|---|---:|---:|---:|
| E0 | clean | O | 86.321270 | 83.567534 | 71.218161 |
| E0 | clean | O* | 86.279663 | 83.527254 | 71.183833 |
| E0 | clean | F | 86.279663 | 83.527254 | 71.183833 |
| E0 | fog | O | 89.615108 | 88.653389 | 78.711113 |
| E0 | fog | O* | 89.566312 | 88.605116 | 78.668253 |
| E0 | fog | F | 53.102254 | 52.342937 | 46.458874 |
| E0 | snow | O | 65.436458 | 61.584579 | 45.078253 |
| E0 | snow | O* | 65.112293 | 61.279496 | 44.854941 |
| E0 | snow | F | 61.410634 | 57.698606 | 42.188436 |
| E1 | clean | O | 86.271596 | 83.474618 | 69.787094 |
| E1 | clean | O* | 86.257735 | 83.461206 | 69.775881 |
| E1 | clean | F | 86.257735 | 83.461206 | 69.775881 |
| E1 | fog | O | 89.862365 | 88.631760 | 77.803540 |
| E1 | fog | O* | 89.813434 | 88.583499 | 77.761174 |
| E1 | fog | F | 53.020044 | 52.276220 | 45.889588 |
| E1 | snow | O | 61.494743 | 57.672526 | 41.871102 |
| E1 | snow | O* | 61.001020 | 57.209491 | 41.534932 |
| E1 | snow | F | 57.571988 | 53.856904 | 39.065792 |
| E2 | clean | O | 84.626401 | 82.041944 | 69.205047 |
| E2 | clean | O* | 84.599207 | 82.015581 | 69.182809 |
| E2 | clean | F | 84.599207 | 82.015581 | 69.182809 |
| E2 | fog | O | 88.113279 | 87.374767 | 77.073319 |
| E2 | fog | O* | 87.993331 | 87.255825 | 76.968400 |
| E2 | fog | F | 51.927941 | 51.492713 | 45.421744 |
| E2 | snow | O | 67.105630 | 64.156193 | 47.984970 |
| E2 | snow | O* | 66.727344 | 63.794533 | 47.714470 |
| E2 | snow | F | 62.760583 | 60.002121 | 44.877974 |

### 65.3 相对E0的AP差值

单位为百分点；直接记录差值，不作机制成立或统计显著性判断。

| 对比 | 天气 | 协议 | ΔAP30 | ΔAP50 | ΔAP70 |
|---|---|---|---:|---:|---:|
| E1-E0 | clean | O | -0.049674 | -0.092916 | -1.431067 |
| E1-E0 | clean | O* | -0.021928 | -0.066048 | -1.407952 |
| E1-E0 | clean | F | -0.021928 | -0.066048 | -1.407952 |
| E1-E0 | fog | O | +0.247257 | -0.021629 | -0.907573 |
| E1-E0 | fog | O* | +0.247122 | -0.021617 | -0.907079 |
| E1-E0 | fog | F | -0.082210 | -0.066717 | -0.569287 |
| E1-E0 | snow | O | -3.941715 | -3.912053 | -3.207151 |
| E1-E0 | snow | O* | -4.111273 | -4.070006 | -3.320009 |
| E1-E0 | snow | F | -3.838646 | -3.841702 | -3.122643 |
| E2-E0 | clean | O | -1.694870 | -1.525590 | -2.013114 |
| E2-E0 | clean | O* | -1.680456 | -1.511673 | -2.001024 |
| E2-E0 | clean | F | -1.680456 | -1.511673 | -2.001024 |
| E2-E0 | fog | O | -1.501830 | -1.278622 | -1.637794 |
| E2-E0 | fog | O* | -1.572980 | -1.349291 | -1.699853 |
| E2-E0 | fog | F | -1.174313 | -0.850223 | -1.037130 |
| E2-E0 | snow | O | +1.669172 | +2.571614 | +2.906716 |
| E2-E0 | snow | O* | +1.615050 | +2.515037 | +2.859529 |
| E2-E0 | snow | F | +1.349949 | +2.303515 | +2.689538 |

### 65.4 固定真值检测计数与目标分层

以下均为F协议、匹配IoU=0.7；TP由距离分层匹配数求和，FN=GT−TP，FP=正式prediction_count−TP。

| 模型 | 天气 | GT | TP | FP | FN |
|---|---|---:|---:|---:|---:|
| E0 | clean | 6224 | 5102 | 1718 | 1122 |
| E0 | fog | 6224 | 3286 | 647 | 2938 |
| E0 | snow | 6224 | 3327 | 1643 | 2897 |
| E1 | clean | 6224 | 5067 | 1775 | 1157 |
| E1 | fog | 6224 | 3248 | 691 | 2976 |
| E1 | snow | 6224 | 3131 | 1636 | 3093 |
| E2 | clean | 6224 | 4972 | 1638 | 1252 |
| E2 | fog | 6224 | 3184 | 592 | 3040 |
| E2 | snow | 6224 | 3429 | 1413 | 2795 |

目标点数为各协作车原始天气点云落入该目标框的点数之和，与柱体截断后的槽位点数分开。

| 模型 | 天气 | 分层依据 | 分组 | 匹配目标/目标总数 | 召回率% |
|---|---|---|---|---:|---:|
| E0 | clean | target_distance_m | 0-30 | 1932/2138 | 90.364827 |
| E0 | clean | target_native_points_sum_cavs | 21+ | 4084/4488 | 90.998217 |
| E0 | clean | target_distance_m | 30-60 | 1931/2045 | 94.425428 |
| E0 | clean | target_native_points_sum_cavs | 6-20 | 815/1053 | 77.397911 |
| E0 | clean | target_native_points_sum_cavs | 1-5 | 203/683 | 29.721816 |
| E0 | clean | target_distance_m | 60+ | 1239/2041 | 60.705537 |
| E0 | fog | target_distance_m | 0-30 | 1920/2138 | 89.803555 |
| E0 | fog | target_native_points_sum_cavs | 21+ | 3267/3574 | 91.410185 |
| E0 | fog | target_distance_m | 30-60 | 1118/2045 | 54.669927 |
| E0 | fog | target_native_points_sum_cavs | 0 | 2/2551 | 0.078401 |
| E0 | fog | target_native_points_sum_cavs | 6-20 | 16/66 | 24.242424 |
| E0 | fog | target_native_points_sum_cavs | 1-5 | 1/33 | 3.030303 |
| E0 | fog | target_distance_m | 60+ | 248/2041 | 12.150906 |
| E0 | snow | target_distance_m | 0-30 | 1327/2138 | 62.067353 |
| E0 | snow | target_native_points_sum_cavs | 21+ | 2355/3609 | 65.253533 |
| E0 | snow | target_distance_m | 30-60 | 1378/2045 | 67.383863 |
| E0 | snow | target_native_points_sum_cavs | 6-20 | 604/1036 | 58.301158 |
| E0 | snow | target_native_points_sum_cavs | 1-5 | 368/1209 | 30.438379 |
| E0 | snow | target_native_points_sum_cavs | 0 | 0/370 | 0.000000 |
| E0 | snow | target_distance_m | 60+ | 622/2041 | 30.475257 |
| E1 | clean | target_distance_m | 0-30 | 1925/2138 | 90.037418 |
| E1 | clean | target_native_points_sum_cavs | 21+ | 4071/4488 | 90.708556 |
| E1 | clean | target_distance_m | 30-60 | 1912/2045 | 93.496333 |
| E1 | clean | target_native_points_sum_cavs | 6-20 | 806/1053 | 76.543210 |
| E1 | clean | target_native_points_sum_cavs | 1-5 | 190/683 | 27.818448 |
| E1 | clean | target_distance_m | 60+ | 1230/2041 | 60.264576 |
| E1 | fog | target_distance_m | 0-30 | 1912/2138 | 89.429373 |
| E1 | fog | target_native_points_sum_cavs | 21+ | 3231/3574 | 90.402910 |
| E1 | fog | target_distance_m | 30-60 | 1082/2045 | 52.909535 |
| E1 | fog | target_native_points_sum_cavs | 0 | 0/2551 | 0.000000 |
| E1 | fog | target_native_points_sum_cavs | 6-20 | 16/66 | 24.242424 |
| E1 | fog | target_native_points_sum_cavs | 1-5 | 1/33 | 3.030303 |
| E1 | fog | target_distance_m | 60+ | 254/2041 | 12.444880 |
| E1 | snow | target_distance_m | 0-30 | 1154/2138 | 53.975678 |
| E1 | snow | target_native_points_sum_cavs | 21+ | 2179/3609 | 60.376836 |
| E1 | snow | target_distance_m | 30-60 | 1355/2045 | 66.259169 |
| E1 | snow | target_native_points_sum_cavs | 6-20 | 587/1036 | 56.660232 |
| E1 | snow | target_native_points_sum_cavs | 1-5 | 365/1209 | 30.190240 |
| E1 | snow | target_native_points_sum_cavs | 0 | 0/370 | 0.000000 |
| E1 | snow | target_distance_m | 60+ | 622/2041 | 30.475257 |
| E2 | clean | target_distance_m | 0-30 | 1925/2138 | 90.037418 |
| E2 | clean | target_native_points_sum_cavs | 21+ | 4037/4488 | 89.950980 |
| E2 | clean | target_distance_m | 30-60 | 1882/2045 | 92.029340 |
| E2 | clean | target_native_points_sum_cavs | 6-20 | 770/1053 | 73.124406 |
| E2 | clean | target_native_points_sum_cavs | 1-5 | 165/683 | 24.158126 |
| E2 | clean | target_distance_m | 60+ | 1165/2041 | 57.079863 |
| E2 | fog | target_distance_m | 0-30 | 1905/2138 | 89.101964 |
| E2 | fog | target_native_points_sum_cavs | 21+ | 3174/3574 | 88.808058 |
| E2 | fog | target_distance_m | 30-60 | 1040/2045 | 50.855746 |
| E2 | fog | target_native_points_sum_cavs | 0 | 0/2551 | 0.000000 |
| E2 | fog | target_native_points_sum_cavs | 6-20 | 9/66 | 13.636364 |
| E2 | fog | target_native_points_sum_cavs | 1-5 | 1/33 | 3.030303 |
| E2 | fog | target_distance_m | 60+ | 239/2041 | 11.709946 |
| E2 | snow | target_distance_m | 0-30 | 1469/2138 | 68.709074 |
| E2 | snow | target_native_points_sum_cavs | 21+ | 2561/3609 | 70.961485 |
| E2 | snow | target_distance_m | 30-60 | 1398/2045 | 68.361858 |
| E2 | snow | target_native_points_sum_cavs | 6-20 | 572/1036 | 55.212355 |
| E2 | snow | target_native_points_sum_cavs | 1-5 | 296/1209 | 24.483044 |
| E2 | snow | target_native_points_sum_cavs | 0 | 0/370 | 0.000000 |
| E2 | snow | target_distance_m | 60+ | 562/2041 | 27.535522 |

### 65.5 逐目标恢复、丢失及定位变化

F协议IoU0.7，相对E0，统计单位为目标在某协作帧中的一次观测；共同匹配项IoU差为算术均值。

| 对比 | 天气 | 恢复数 | 丢失数 | 共同匹配数 | 共同IoU均值差 |
|---|---|---:|---:|---:|---:|
| E1-E0 | clean | 194 | 229 | 4873 | -0.00109024 |
| E1-E0 | fog | 81 | 119 | 3167 | -0.00195104 |
| E1-E0 | snow | 341 | 537 | 2790 | -0.00118724 |
| E2-E0 | clean | 178 | 308 | 4794 | -0.00743702 |
| E2-E0 | fog | 88 | 190 | 3096 | -0.00925240 |
| E2-E0 | snow | 532 | 430 | 2897 | +0.00005745 |

完整frame_key、恢复/丢失object_id、共同目标IoU差及FP/FN变化保留在COMPACT_RESULTS.json的target_changes；未将数量变化解释为机制原因。

### 65.6 固定权重机制诊断

每天气预先固定24帧，索引0,36,…,828；这些结果来自诊断子集，不等同于829帧正式评测。E1比较adaptive/uniform/shuffled/off，uniform匹配每帧有邻居有效点的平均alpha，shuffled在柱体点数分组内打乱。E2比较C on/off。两组记录均optimization_steps=0、model_and_BN_unchanged=true。

| 模型 | 天气 | 开关 | AP30% | AP50% | AP70% | TP | FP | FN | 1—5点目标匹配/总数 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| E1 | clean | adaptive | 86.520281 | 83.267948 | 64.745023 | 149 | 55 | 42 | 6/22 |
| E1 | clean | off | 86.523146 | 83.267948 | 64.747506 | 149 | 55 | 42 | 6/22 |
| E1 | clean | shuffled | 86.523146 | 83.267948 | 64.747506 | 149 | 55 | 42 | 6/22 |
| E1 | clean | uniform | 86.523146 | 83.267948 | 64.747506 | 149 | 55 | 42 | 6/22 |
| E1 | fog | adaptive | 56.081392 | 56.081392 | 46.073545 | 103 | 26 | 88 | 0/2 |
| E1 | fog | off | 56.081392 | 56.081392 | 46.073545 | 103 | 26 | 88 | 0/2 |
| E1 | fog | shuffled | 56.081392 | 56.081392 | 46.073545 | 103 | 26 | 88 | 0/2 |
| E1 | fog | uniform | 56.081392 | 56.081392 | 46.073545 | 103 | 26 | 88 | 0/2 |
| E1 | snow | adaptive | 55.765665 | 54.378254 | 37.275351 | 92 | 47 | 99 | 12/37 |
| E1 | snow | off | 56.201279 | 54.383291 | 37.275351 | 92 | 48 | 99 | 12/37 |
| E1 | snow | shuffled | 55.765665 | 54.378254 | 37.275351 | 92 | 47 | 99 | 12/37 |
| E1 | snow | uniform | 56.195732 | 54.378254 | 37.275351 | 92 | 48 | 99 | 12/37 |
| E2 | clean | off | 80.456921 | 78.239350 | 63.894254 | 145 | 51 | 46 | 5/22 |
| E2 | clean | on | 81.941302 | 79.655636 | 61.877726 | 142 | 60 | 49 | 6/22 |
| E2 | fog | off | 55.542121 | 55.542121 | 44.890903 | 100 | 28 | 91 | 0/2 |
| E2 | fog | on | 55.201850 | 55.201850 | 44.450553 | 99 | 24 | 92 | 0/2 |
| E2 | snow | off | 57.201296 | 57.201296 | 49.134931 | 105 | 29 | 86 | 9/37 |
| E2 | snow | on | 59.258903 | 57.401852 | 48.402461 | 105 | 34 | 86 | 11/37 |

E1按柱体数加权汇总自保留alpha、原残差门eta、修正RMS；跨24帧合并，同一柱体点数分组的完整结果另存PILLAR_RETENTION.csv。

| 天气 | 模式 | 柱体数 | alpha柱体均值 | eta均值 | 修正RMS均值 |
|---|---|---:|---:|---:|---:|
| clean | adaptive | 161921 | 0.07078752 | 0.42658379 | 0.21621796 |
| clean | uniform | 161921 | 0.08778416 | 0.42658379 | 0.21624850 |
| clean | shuffled | 161921 | 0.07131412 | 0.42658379 | 0.21625786 |
| clean | off | 161921 | 0.00000000 | 0.42658379 | 0.21629109 |
| fog | adaptive | 135476 | 0.07052422 | 0.43391669 | 0.18438976 |
| fog | uniform | 135476 | 0.08581718 | 0.43391669 | 0.18434464 |
| fog | shuffled | 135476 | 0.07108783 | 0.43391669 | 0.18434515 |
| fog | off | 135476 | 0.00000000 | 0.43391669 | 0.18441442 |
| snow | adaptive | 131008 | 0.06369592 | 0.42721699 | 0.22366303 |
| snow | uniform | 131008 | 0.07493350 | 0.42721699 | 0.22372253 |
| snow | shuffled | 131008 | 0.06400035 | 0.42721699 | 0.22374803 |
| snow | off | 131008 | 0.00000000 | 0.42721699 | 0.22382480 |

E2传播统计按网格数加权；同一帧的各车辆网格及各帧重复观测分别计数。

| 天气 | 目标类型 | 总网格数 | 接收网格数 | 非零增强数 | 平均强度 | 平均修正RMS |
|---|---|---:|---:|---:|---:|---:|
| clean | observed | 161921 | 157480 | 157480 | 0.04878218 | 0.11784029 |
| clean | empty | 9694079 | 255914 | 255914 | 0.00044653 | 0.00049066 |
| fog | observed | 135476 | 134758 | 134758 | 0.03306135 | 0.05987304 |
| fog | empty | 9720524 | 152244 | 152244 | 0.00026492 | 0.00018760 |
| snow | observed | 131008 | 121802 | 121802 | 0.04327301 | 0.07668936 |
| snow | empty | 9724992 | 235461 | 235461 | 0.00040954 | 0.00042928 |

来源规则原文：original occupied grid only; target occupancy unrestricted。门控直方图、每帧alpha最大值、邻居可信程度和观测证据、来源质量及具体目标匹配见归档机制summary.json。

### 65.7 训练、参数更新及资源记录

轮次耗时为训练记录duration_seconds之和，含原引擎验证及Clean选模耗时，不作为纯GPU训练时间；显存为torch记录的峰值已分配显存，不等同于驱动保留显存。

| 模型 | 更新数 | 参数量 | 轮次耗时合计小时 | 最大已分配显存GiB |
|---|---:|---:|---:|---:|
| E0 | 20200 | 6599226 | 29.995630 | 18.761112 |
| E1 | 20200 | 6599230 | 36.050630 | 18.764005 |
| E2 | 20200 | 6603899 | 32.802870 | 26.122989 |

逐轮参数梯度/更新的正值批次数与最大值保留在COMPACT_RESULTS.json/training_records/parameter_diagnostics；原始逐批值在remote/E1/history.json及remote/E2/history.json。

### 65.8 原始证据、校验与记录边界

原FINAL_RESULTS.json SHA256：`01e615235ae8611e9860676373facebb778f28d695d5310eaaf9ce7b426f4d23`。下载归档内所有文件均逐项SHA校验，两个选中checkpoint均在海光实算SHA核对。

本地证据：`delivery/e1e2_results_20261103/evidence.zip`及`remote/`，下载时间、路径、哈希、缺失文件列表见remote/OBSERVATION.json；紧凑副本COMPACT_RESULTS.json保留完整正式成绩、目标变化、分层和机制记录，并将逐批训练诊断汇总为逐轮统计。

单随机种子；使用validation选模及报告成绩，未评Test；开关对照为固定权重推理干预，未重新训练对照。上述为数值和记录范围，不据此写入稀疏保护或互补机制已被证明的结论。旧实验代码、权重、合同和完成标记均未修改。

§65补充：原报告描述性差中差(E2−E0)−(B2−B0)，单位百分点，不作统计显著性判断。

| 天气 | 协议 | ΔΔAP30 | ΔΔAP50 | ΔΔAP70 |
|---|---|---:|---:|---:|
| clean | F | -3.186074 | -2.027468 | -4.070147 |
| clean | O | -3.201214 | -2.041634 | -4.083234 |
| clean | O* | -3.186074 | -2.027468 | -4.070147 |
| fog | F | -1.725215 | -0.676521 | -0.727770 |
| fog | O | +0.791887 | +1.214868 | -0.387636 |
| fog | O* | +0.766305 | +1.188858 | -0.409935 |
| snow | F | -3.848632 | -2.958676 | -2.781060 |
| snow | O | -3.919208 | -3.125221 | -3.035873 |
| snow | O* | -3.798782 | -3.016243 | -2.956836 |

正式训练新增参数的梯度/更新非零批次数（各组共20200更新；同一参数张量L2>0计一次）：
- E1 gradient_l2/s.logits：20199/20200；最大L2=0.003239268437。
- E1 update_l2/s.logits：20200/20200；最大L2=0.006574355997。
- E2 C_gradient_l2/gate.bias：20200/20200；最大L2=0.3222372532。
- E2 C_gradient_l2/gate.weight：20200/20200；最大L2=5.823885918。
- E2 C_gradient_l2/key.weight：20200/20200；最大L2=0.1367684901。
- E2 C_gradient_l2/value.weight：20200/20200；最大L2=3.720870972。
- E2 C_update_l2/gate.bias：20028/20200；最大L2=0.002827167511。
- E2 C_update_l2/gate.weight：20200/20200；最大L2=0.0261628069。
- E2 C_update_l2/key.weight：20200/20200；最大L2=0.1060358062。
- E2 C_update_l2/value.weight：20200/20200；最大L2=0.1854130477。


## 66. E2完整验证集网格开关诊断代码（2026-11-03，尚未运行）

按用户要求新增独立gspr_e2_grid_diagnostic/，海光放置/home/cjm/OpenCOOD-main/cjmnet/gspr_e2_grid_diagnostic。只读原E2 selected.json最佳权重及20轮记录，不重训，不修改旧源码/合同/结果。

四个模式：on全部开启；off全部关闭；observed_only仅向原本有点目标网格加增强（关闭空网格增强）；empty_only仅向原本空目标网格加增强（关闭有点网格增强）。始终保持原有点网格来源掩码、八邻域、key/value、相似度和残差门一致。仅对目标位置使用torch.where屏蔽增量；同一帧四开关回放相同首次Scatter输入/原始占据掩码/增强输出。

默认三个天气各完整829帧，官方O/O*/F后处理与计分；输出AP30/50/70、TP/FP/FN、on−off和off−E0差值、原有点/空网格增量统计、F协议IoU0.7的逐目标恢复/丢失/共同IoU差，以及距离×目标原始点数召回。近距离预定义<30米，密集>=21原始点，稀疏1—5原始点，均非柱体槽位点数。

on−off衡量给定E2联合训练权重下推理增强的条件贡献；off−E0包含GSPR、柱内聚合、主干、融合及检测头的整体训练差异，不单独解释为纯主干因果效应。不假定有点/空网格增强收益可以线性相加。

唯一入口run_all.sh：默认GPU=6/MODE=all，先执行GPU目标掩码精确检查及每天气0/828两帧真实smoke，通过后执行完整829帧；MODE=smoke只小验收。完整帧记录有合同、输入及载荷哈希，可从完整帧续算，未提交帧重跑四模式。结束时核对模型/BN和原权重SHA不变。

已完成本地AST/LF检查、上传SHA校验及海光sh -n/AST静态检查；scientific_tests_executed=false、started=false。用户选择自行启动，因此尚无新增诊断成绩，不能登记运行验收通过。

输出/home/cjm/datasets/logs/e2-grid-diagnostic-v1；smoke为同路径加_smoke。原始代码包及部署记录在delivery/e2_grid_diagnostic_v1/。

```sh
nohup env GPU=6 MODE=all sh /home/cjm/OpenCOOD-main/cjmnet/gspr_e2_grid_diagnostic/run_all.sh >> /home/cjm/datasets/logs/e2_grid_diagnostic_v1.launch.log 2>&1 < /dev/null &
```

部署包SHA256：`4ba0b4a728a6bed130db67791b4ffcab7fcdc01abbb7c6ff0e0398c3cf92a99c`。

## 67. LR0 双模态 AttFuse 正式结果（2026-11-08 海光只读核验）

LR0 在海光物理卡 1（K100_AI）完成 20 轮。正式评测的 Clean、Fog、Snow 各 829 帧，使用同一份第 18 轮权重。按相同的双模态 Clean 真值计分（F），Fog 和 Snow 的 AP70 分别比 Clean 低 22.7092 和 21.7178 个百分点。使用天气真值（O）时，Fog AP70 高于 Snow，使用固定 Clean 真值时顺序反转。

### 模型、数据和训练

- 直接复用官方双模态 `point_pillar_intermediate_lrf`。激光雷达和 4D 雷达各用一个 PointPillars 编码器，各输出 64 通道鸟瞰特征，沿通道拼成 128 后交给官方 AttFuse 主干和原检测头。参数量 25,105,680。未加载预训练权重，未加入 GSPR、A1 或扩散去噪。
- 全量核验 8,083 个训练协作帧、24,088 个单车帧及 120,440 个所需文件，829 个验证协作帧、3,041 个单车帧及 18,246 个所需文件。两种传感器按场景、车辆、时间戳配对。天气切换只改变激光雷达文件，雷达始终读取对应原始帧。雷达原文件字段为 `x y z rgb`，官方读取位置和红色通道，无显式速度输入。
- 沿用 A1 的种子 20261017、批大小 8、Adam 初始学习率 0.002、权重衰减 0.0001、eps 1e-10、原 [10,15] 调度和逐车独立 Clean 4/7、Fog 3/7 抽样。Snow 不参与训练。20 轮各 1,010 次更新，共 20,200 次。轮内记录累计约 13.30 小时，含每轮验证及保存。训练峰值已分配显存约 20.03 GiB。
- 使用官方双模态检测损失 `point_pillar_loss`，A1 的检测损失入口为 `point_pillar_loss_opencood`。LR0 验证损失在逐帧推理时计算后平均，A1 采用单独的批量验证损失。官方双模态原配方为 30 轮，本实验统一为 20 轮。
- 空模态处理保留车辆和真实点，空柱分支的鸟瞰特征为零，取代官方随机补点。20 轮累计记录激光雷达 15,107 次、雷达 55,729 次空分支，统计的是每次读入后的车辆分支。几何无效预测框在进入官方去重前排除。正式三天气均记录排除 0 个。日志仍有 Shapely 求交警告，流程完成，未将警告视为不存在。
- `SMOKE.json` 已完成。两次真实批量更新、编码和拼接检查、三天气各 12 帧、官方 O/O*/F 复核和完整恢复状态往返均通过。

### 选模与正式评测

20 轮权重、恢复状态和验证摘要哈希全部核对一致。完整 Clean O AP70 最高的是第 18 轮（0.7043143771），其次为第 20 轮（0.7041298685）和第 19 轮（0.7023565774）。第 18 轮训练/验证损失分别为 0.25714755 / 0.40228328。锁定权重后独立进行的 Clean 正式 O AP70 为 0.7037151939，比该轮选模读数低约 0.0599 个百分点。同一权重哈希已核验，两次读数分别保留，差值成因未单独验证。

O 使用该天气真值并跳过没有预测返回的 None 帧。O* 使用同一套天气真值，把 None 帧按零预测计入。F 固定双模态 Clean 真值，也把 None 帧按零预测计入。下表 AP 为 0–1 小数。匹配、误检、漏检的三元组按 AP70 对应的 0.7 重叠阈值统计。

| 天气 | 协议 | AP30 | AP50 | AP70 | 参与真值 | None帧 | 匹配 / 误检 / 漏检 |
|---|---|---:|---:|---:|---:|---:|---:|
| Clean | O | 0.854182 | 0.830191 | 0.703715 | 6,233 | 36 | 5,121 / 1,616 / 1,112 |
| Clean | O* | 0.853908 | 0.829924 | 0.703489 | 6,235 | 36 | 5,121 / 1,616 / 1,114 |
| Clean | F | 0.853908 | 0.829924 | 0.703489 | 6,235 | 36 | 5,121 / 1,616 / 1,114 |
| Fog | O | 0.764650 | 0.719134 | 0.516750 | 5,695 | 41 | 3,846 / 1,682 / 1,849 |
| Fog | O* | 0.763042 | 0.717622 | 0.515663 | 5,707 | 41 | 3,846 / 1,682 / 1,861 |
| Fog | F | 0.716117 | 0.671599 | 0.476398 | 6,235 | 41 | 3,860 / 1,668 / 2,375 |
| Snow | O | 0.712554 | 0.680414 | 0.497694 | 6,086 | 69 | 3,838 / 1,610 / 2,248 |
| Snow | O* | 0.709988 | 0.677963 | 0.495902 | 6,108 | 69 | 3,838 / 1,610 / 2,270 |
| Snow | F | 0.698386 | 0.666080 | 0.486312 | 6,235 | 69 | 3,840 / 1,608 / 2,395 |

| 天气 | 冻结纯 LiDAR 天气真值 | LR0 天气真值 | 仅 LR0 目标 | 真值不同的帧 | 推理帧/秒 | 峰值已分配显存 GiB |
|---|---:|---:|---:|---:|---:|---:|
| Clean | 6,224 | 6,235 | 11 | 11 | 15.16 | 1.226 |
| Fog | 3,673 | 5,707 | 2,034 | 596 | 16.62 | 1.223 |
| Snow | 5,854 | 6,108 | 254 | 139 | 16.64 | 1.220 |

三种天气均无仅纯 LiDAR 保留的目标。表中比较的是过滤后的天气真值，尚未扣除 O 的 None 帧。Fog 双模态真值比纯 LiDAR 多约 55.4%，因此跨模态分数差同时受评测目标集合影响。此记录是单次运行结果。推理速度包含按帧同步的模型和官方后处理，未计数据加载。显存为本进程峰值，区别于整卡用量。

### 产物与身份

海光目录 `/data/cjm/datasets/logs/v2xr-lr0-v3/`。`RESULTS.md` 和 `FINAL_RESULTS.json` 含所有阈值的匹配、误检、漏检数。`history.json`、`selected.json`、`CONTRACT.json`、`config.yaml`、`DATA.json` 和 `SMOKE.json` 分别记录轮次、选模、源码、配置、输入和自测证据。`final_evaluation/gt_comparison.json` 保存与纯 LiDAR 的逐帧真值差异。服务器日志末尾为 `COMPLETE LR0`。

- 第 18 轮权重 `epochs/18/weights.pth` SHA256 `0eb0ec8bd6a6e0bc368c7cc8aacd49a33cfe014413463f8b314fdbf76c6defd2`。
- 正式结果 `FINAL_RESULTS.json` SHA256 `c98d327ab834535916e5f3bc5dfd0ba94afda09fb16730c06d81d46c598e9758`。
- 运行合同 `CONTRACT.json` SHA256 `5479e679e12f5dda760d55d8039c22d99e38e73a2f30ddc550372d705f4bf652`。有效配置 `config.yaml` SHA256 `4b8a3626d6fa94fd4ef841ba67dddb01f6030f2c964d0ed2d142c60eca8d59b9`。
- 数据验收 `DATA.json` SHA256 `0615a36a5ad056cba1053b91ffd406af1b6c0d03cb589c6e0201a09d99a7beab`。

## 68. POT-CF 三卡正式实验与消融结果（2026-11-11 海光核验）

- P1（局部普通注意力）、P2（局部非平衡最优传输）和 P3（P2 加未匹配观测保留路径）分别使用海光物理 GPU 4、5、6，均完成 20 轮训练、正式评测和消融汇总。P1 第 5 轮提交后从同一检查点续跑。汇总产物于 2026-11-11 06:09（北京时间）生成。
- 正式训练每轮 1,010 次更新，沿用第 67 节的训练数据和协议。按完整 Clean 验证集的 O AP70 选取权重。正式评测每种天气各 829 帧。F 固定使用同一份双模态 Clean 真值，共 6,235 个目标。O 和 O* 的计分定义见第 67 节。
- P0 为第 67 节中物理 GPU 1 上的历史正式结果。P0 同权重在卡 4 的复测列在消融表中。

| 组别 | 选中轮次 | 选模 Clean O AP70 | 权重 SHA256 |
|---|---:|---:|---|
| P1 | 16 | 0.691426 | `6a5e84cda2795e267a1481f4ea489fec4ade7a5e7713f8f57bca192a2209a3a5` |
| P2 | 19 | 0.701185 | `c2b2da5fbd82be5b53d6225171be7f535637f879256995fbacd62905c94baa9e` |
| P3 | 18 | 0.702547 | `5864ac849677b83c7f68d6cabf83c7f07045cde48083dac5df9db6d8917b5e1d` |

### 正式评测

| 组别 | 天气 | 协议 | AP30 | AP50 | AP70 | 推理帧/秒 | 峰值显存 GiB |
|---|---|---|---:|---:|---:|---:|---:|
| P1 | clean | F | 0.840679 | 0.824885 | 0.691079 | 15.92 | 1.226 |
| P1 | clean | O | 0.841759 | 0.825944 | 0.691966 | 15.92 | 1.226 |
| P1 | clean | O* | 0.840679 | 0.824885 | 0.691079 | 15.92 | 1.226 |
| P1 | fog | F | 0.669844 | 0.642012 | 0.471436 | 16.92 | 1.224 |
| P1 | fog | O | 0.723361 | 0.693933 | 0.512651 | 16.92 | 1.224 |
| P1 | fog | O* | 0.719812 | 0.690528 | 0.510135 | 16.92 | 1.224 |
| P1 | snow | F | 0.656699 | 0.635090 | 0.465886 | 17.42 | 1.221 |
| P1 | snow | O | 0.672882 | 0.650744 | 0.477274 | 17.42 | 1.221 |
| P1 | snow | O* | 0.670018 | 0.647974 | 0.475242 | 17.42 | 1.221 |
| P2 | clean | F | 0.850313 | 0.825468 | 0.700151 | 11.72 | 1.225 |
| P2 | clean | O | 0.850722 | 0.825865 | 0.700488 | 11.72 | 1.225 |
| P2 | clean | O* | 0.850313 | 0.825468 | 0.700151 | 11.72 | 1.225 |
| P2 | fog | F | 0.705961 | 0.662309 | 0.472441 | 12.79 | 1.224 |
| P2 | fog | O | 0.754629 | 0.709415 | 0.511370 | 12.79 | 1.224 |
| P2 | fog | O* | 0.753306 | 0.708172 | 0.510474 | 12.79 | 1.224 |
| P2 | snow | F | 0.676408 | 0.638913 | 0.466819 | 11.72 | 1.220 |
| P2 | snow | O | 0.689866 | 0.652347 | 0.476885 | 11.72 | 1.220 |
| P2 | snow | O* | 0.688059 | 0.650638 | 0.475636 | 11.72 | 1.220 |
| P3 | clean | F | 0.861912 | 0.837930 | 0.702133 | 11.71 | 1.225 |
| P3 | clean | O | 0.862327 | 0.838334 | 0.702471 | 11.71 | 1.225 |
| P3 | clean | O* | 0.861912 | 0.837930 | 0.702133 | 11.71 | 1.225 |
| P3 | fog | F | 0.718371 | 0.675860 | 0.477446 | 12.40 | 1.224 |
| P3 | fog | O | 0.768384 | 0.723971 | 0.519580 | 12.40 | 1.224 |
| P3 | fog | O* | 0.766499 | 0.722195 | 0.518305 | 12.40 | 1.224 |
| P3 | snow | F | 0.710825 | 0.672581 | 0.485724 | 13.72 | 1.220 |
| P3 | snow | O | 0.725336 | 0.686256 | 0.496818 | 13.72 | 1.220 |
| P3 | snow | O* | 0.723318 | 0.684346 | 0.495435 | 13.72 | 1.220 |

### 消融 F AP70

| 设置 | Clean | Fog | Snow |
|---|---:|---:|---:|
| `P0/single` | 0.523759 | 0.327291 | 0.352914 |
| `P0/same_weights_gpu4` | 0.704088 | 0.475973 | 0.486369 |
| `P1/single` | 0.532142 | 0.316720 | 0.353022 |
| `P2/single` | 0.534441 | 0.331507 | 0.345184 |
| `P2/dustbin_inference` | 0.700847 | 0.471550 | 0.467235 |
| `P3/single` | 0.523008 | 0.333863 | 0.354791 |
| `P3/matched_only` | 0.701969 | 0.479241 | 0.485234 |
| `P3/unmatched_only` | 0.701380 | 0.479693 | 0.485230 |
| `P3/module_off` | 0.700820 | 0.480032 | 0.485004 |
| `P3/window5` | 0.702209 | 0.479650 | 0.485267 |
| `P3/radar_off` | 0.677051 | 0.394165 | 0.406443 |
| `P3/epsilon_0.1` | 0.702770 | 0.479220 | 0.485578 |
| `P3/epsilon_0.4` | 0.701785 | 0.479361 | 0.485367 |

### 固定真值逐目标统计

“仅邻车有实点”按当帧两路点是否落入固定真值框内统计。误检数变化为后者减前者。

| 对照 | 天气 | 恢复 | 丢失 | 恢复且仅邻车有实点 | 仅邻车有实点的全部目标 | 误检数变化 | 新增误检（空间去重） |
|---|---|---:|---:|---:|---:|---:|---:|
| P0 single to multi | clean | 919 | 52 | 162 | 404 | -680 | — |
| P0 single to multi | fog | 900 | 103 | 154 | 486 | -675 | — |
| P0 single to multi | snow | 803 | 224 | 98 | 524 | -769 | — |
| P1 single to multi | clean | 882 | 74 | 129 | 404 | -487 | — |
| P1 single to multi | fog | 937 | 103 | 136 | 486 | -431 | — |
| P1 single to multi | snow | 777 | 242 | 72 | 524 | -446 | — |
| P2 single to multi | clean | 881 | 67 | 150 | 404 | -600 | — |
| P2 single to multi | fog | 864 | 96 | 133 | 486 | -556 | — |
| P2 single to multi | snow | 777 | 224 | 86 | 524 | -681 | — |
| P3 single to multi | clean | 916 | 68 | 161 | 404 | -624 | — |
| P3 single to multi | fog | 896 | 101 | 156 | 486 | -653 | — |
| P3 single to multi | snow | 810 | 187 | 106 | 524 | -693 | — |
| P2 to P3 | clean | 249 | 225 | 30 | 404 | -15 | 733 |
| P2 to P3 | fog | 329 | 250 | 42 | 486 | 54 | 833 |
| P2 to P3 | snow | 531 | 286 | 42 | 524 | 200 | 938 |
| same P3 weights, unmatched off to on | clean | 6 | 3 | 2 | 404 | 1 | 11 |
| same P3 weights, unmatched off to on | fog | 9 | 14 | 2 | 486 | 5 | 36 |
| same P3 weights, unmatched off to on | snow | 12 | 9 | 1 | 524 | 3 | 20 |

### 服务器产物

海光目录 `/data/cjm/datasets/logs/v2xr-fusion-pot-v1/`。各组的 `selected.json` 保存所选权重路径，各组的 `TRANSPORT.json` 保存传输统计。汇总日志 `handoff.log` 末尾记录 `ALL PARALLEL GROUPS AND REPORT COMPLETE`。

- `FINAL_COMPARISON.md` SHA256 `07bd220bb0e6d811e4f519ffebfe59cac04ba825a6b373016a770d1ce12dfef0`。
- `FINAL_RESULTS.json` SHA256 `6ef4b33797dfcf27db493adb9b816318c734a3792cd2c6490abd3e88ca4f2a0a`。

## 69. R-A/R-B 的 F0–F3 正式结果（2026-11-11 至 11-12 海光核验）

F0 为 LR1（激光雷达分支接入 GSPR 和 A1），F1 在 F0 上加入 R-A，F2 在 F1 上加入 R-B。三组均未使用配对天气监督。三组各完成 20 轮、每轮 1,010 次更新，从零初始化，批大小 8，种子 20261017。训练使用 8,083 个协作帧，按车独立抽样 Clean 4/7、Fog 3/7。按每轮完整 Clean 验证的 O AP70 选权重。锁定权重后，三种天气各评测 829 帧。

| 组别 | 海光物理卡 | 选中轮次 | 选模时 Clean O AP70 | 参数量 | 轮内累计小时 | 训练峰值已分配显存 GiB |
|---|---:|---:|---:|---:|---:|---:|
| F0 | 1 | 20 | 0.724206 | 25,120,570 | 42.87 | 20.91 |
| F1 | 2 | 19 | 0.716363 | 25,126,875 | 40.52 | 20.97 |
| F2 | 3 | 16 | 0.715375 | 25,135,452 | 44.94 | 20.99 |

轮内累计时间来自各轮记录，包含训练、验证和保存。表中选模分数取训练期间的 Clean 验证，正式 Clean 分数取锁定权重后的独立评测。

### 正式评测

O 使用对应天气真值并跳过返回 None 的帧。O* 使用对应天气真值，None 按零预测计入。F 固定双模态 Clean 真值，None 按零预测计入。已逐帧核对三组的 F 真值目标身份与几何记录，与 LR0 一致。每组每种天气的 F 真值均为 6,235 个。下表 AP 为 0–1 小数，匹配、误检和漏检按 0.7 重叠阈值统计。

| 天气 | 协议 | 组别 | AP30 | AP50 | AP70 | 参与真值 | None 帧 | 匹配 / 误检 / 漏检 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Clean | O | F0 | 0.858094 | 0.835604 | 0.725339 | 6234 | 40 | 5168 / 1578 / 1066 |
| Clean | O | F1 | 0.846029 | 0.823672 | 0.717303 | 6235 | 44 | 5139 / 1621 / 1096 |
| Clean | O | F2 | 0.846387 | 0.826426 | 0.715889 | 6227 | 53 | 5054 / 1241 / 1173 |
| Clean | O* | F0 | 0.857956 | 0.835470 | 0.725222 | 6235 | 40 | 5168 / 1578 / 1067 |
| Clean | O* | F1 | 0.846029 | 0.823672 | 0.717303 | 6235 | 44 | 5139 / 1621 / 1096 |
| Clean | O* | F2 | 0.845301 | 0.825365 | 0.714970 | 6235 | 53 | 5054 / 1241 / 1181 |
| Clean | F | F0 | 0.857956 | 0.835470 | 0.725222 | 6235 | 40 | 5168 / 1578 / 1067 |
| Clean | F | F1 | 0.846029 | 0.823672 | 0.717303 | 6235 | 44 | 5139 / 1621 / 1096 |
| Clean | F | F2 | 0.845301 | 0.825365 | 0.714970 | 6235 | 53 | 5054 / 1241 / 1181 |
| Fog | O | F0 | 0.764604 | 0.727004 | 0.571953 | 5696 | 44 | 3982 / 1525 / 1714 |
| Fog | O | F1 | 0.757403 | 0.718870 | 0.555088 | 5693 | 49 | 3936 / 1584 / 1757 |
| Fog | O | F2 | 0.727585 | 0.705505 | 0.561084 | 5682 | 65 | 3826 / 1061 / 1856 |
| Fog | O* | F0 | 0.763130 | 0.725603 | 0.570851 | 5707 | 44 | 3982 / 1525 / 1725 |
| Fog | O* | F1 | 0.755545 | 0.717106 | 0.553726 | 5707 | 49 | 3936 / 1584 / 1771 |
| Fog | O* | F2 | 0.724398 | 0.702415 | 0.558626 | 5707 | 65 | 3826 / 1061 / 1881 |
| Fog | F | F0 | 0.713403 | 0.678765 | 0.530766 | 6235 | 44 | 4002 / 1505 / 2233 |
| Fog | F | F1 | 0.708032 | 0.671455 | 0.514997 | 6235 | 49 | 3956 / 1564 / 2279 |
| Fog | F | F2 | 0.673763 | 0.653631 | 0.518788 | 6235 | 65 | 3844 / 1043 / 2391 |
| Snow | O | F0 | 0.729193 | 0.696684 | 0.535344 | 6089 | 71 | 4017 / 1600 / 2072 |
| Snow | O | F1 | 0.718788 | 0.686913 | 0.535365 | 6080 | 73 | 4034 / 1582 / 2046 |
| Snow | O | F2 | 0.695052 | 0.674079 | 0.515998 | 6061 | 87 | 3746 / 1114 / 2315 |
| Snow | O* | F0 | 0.726924 | 0.694516 | 0.533678 | 6108 | 71 | 4017 / 1600 / 2091 |
| Snow | O* | F1 | 0.715493 | 0.683764 | 0.532911 | 6108 | 73 | 4034 / 1582 / 2074 |
| Snow | O* | F2 | 0.689704 | 0.668893 | 0.512028 | 6108 | 87 | 3746 / 1114 / 2362 |
| Snow | F | F0 | 0.714545 | 0.682798 | 0.523537 | 6235 | 71 | 4019 / 1598 / 2216 |
| Snow | F | F1 | 0.705364 | 0.673379 | 0.523478 | 6235 | 73 | 4039 / 1577 / 2196 |
| Snow | F | F2 | 0.676942 | 0.656629 | 0.502294 | 6235 | 87 | 3748 / 1112 / 2487 |

| 天气 | 协议 | F1−F0 AP70（百分点） | F2−F1 AP70（百分点） | F2−F0 AP70（百分点） |
|---|---|---:|---:|---:|
| Clean | O | -0.8036 | -0.1414 | -0.9450 |
| Clean | O* | -0.7919 | -0.2333 | -1.0252 |
| Clean | F | -0.7919 | -0.2333 | -1.0252 |
| Fog | O | -1.6866 | +0.5996 | -1.0870 |
| Fog | O* | -1.7125 | +0.4900 | -1.2225 |
| Fog | F | -1.5769 | +0.3792 | -1.1977 |
| Snow | O | +0.0022 | -1.9367 | -1.9345 |
| Snow | O* | -0.0767 | -2.0883 | -2.1651 |
| Snow | F | -0.0058 | -2.1184 | -2.1242 |

| 组别 | Clean / Fog / Snow 推理帧每秒 | Clean / Fog / Snow 峰值已分配显存 GiB |
|---|---:|---:|
| F0 | 2.39 / 2.93 / 2.84 | 1.245 / 1.241 / 1.235 |
| F1 | 3.82 / 4.57 / 4.46 | 1.589 / 1.239 / 1.234 |
| F2 | 3.85 / 4.02 / 4.27 | 1.592 / 1.241 / 1.235 |

推理计时包含按帧同步的模型执行及后处理。三组正式评测记录的雷达空分支均为每种天气 598 次，激光雷达空分支依 Clean/Fog/Snow 为 111/193/115 次，计数单位为每帧的车辆分支。

### 产物核验

海光目录为 `/data/cjm/datasets/logs/v2xr-radar-ra-rb-v1/`。各组目录下的 `GROUP_COMPLETE.json`、`history.json`、`selected.json` 和 `FINAL_RESULTS.json` 已核对，选中权重实算哈希与正式评测记录一致。全部阈值的匹配统计与目标级诊断见各组 `FINAL_RESULTS.json` 及 `final_evaluation/`。

- F0 第 20 轮权重 SHA256 `a5e12d10cde649c5a198b814277134ae06f8069a5b8e4d32505438c19b1fe8b1`。`FINAL_RESULTS.json` SHA256 `167569795733df2f3eecaac50ff0fc9301ea958e2e0081c72a54bd876ba27562`。
- F1 第 19 轮权重 SHA256 `1beb4961b575d25bbec36ce8b0dd61fcb8f33096b3be8738b7b075216ef551d9`。`FINAL_RESULTS.json` SHA256 `7fcb0c29d420fe986a334bd84eb41b133b310ab2eb2289329b9a8c4e0e9203a3`。
- F2 第 16 轮权重 SHA256 `841a583931c0cd40f351c793399d89807a9a23e3599efda923a6797fa02959c7`。`FINAL_RESULTS.json` SHA256 `7adb9c9be53f5653b0e92485e402be2a61d28e0e1f5b1bf21bd4009d34601ea8`。


### F3 正式结果（2026-11-12 海光核验）

R-A/R-B 消融的 F3（F2 加训练期 Clean/Fog 配对监督）已在海光卡 1 完成 20 轮，每轮 1,010 次更新。按每轮完整 Clean 验证的 O AP70 选中第 16 轮，选模分数为 0.723163。该权重独立正式复测的 Clean O AP70 为 0.723264。模型参数量 25,135,452，轮内累计用时 49.15 小时（含训练、验证与保存），训练峰值已分配显存 20.99 GiB。正式评测完成标记的时间为 11 月 12 日 17:27（北京时间）。

三种天气各 829 帧。三天气的 F 协议固定真值均为 6,235 个，逐帧目标身份与几何记录已核对，与 F2 一致。AP 为 0–1 小数，匹配、误检和漏检按 0.7 重叠阈值统计。

| 天气 | 协议 | AP30 | AP50 | AP70 | 参与真值 | None 帧 | 匹配 / 误检 / 漏检 |
|---|---|---:|---:|---:|---:|---:|---:|
| Clean | O | 0.844348 | 0.825383 | 0.723264 | 6233 | 43 | 5095 / 1158 / 1138 |
| Clean | O* | 0.844077 | 0.825118 | 0.723032 | 6235 | 43 | 5095 / 1158 / 1140 |
| Clean | F | 0.844077 | 0.825118 | 0.723032 | 6235 | 43 | 5095 / 1158 / 1140 |
| Fog | O | 0.736964 | 0.701145 | 0.548497 | 5683 | 66 | 3810 / 1151 / 1873 |
| Fog | O* | 0.733865 | 0.698196 | 0.546191 | 5707 | 66 | 3810 / 1151 / 1897 |
| Fog | F | 0.683422 | 0.649684 | 0.503470 | 6235 | 66 | 3819 / 1142 / 2416 |
| Snow | O | 0.671225 | 0.644421 | 0.501505 | 6080 | 71 | 3702 / 1099 / 2378 |
| Snow | O* | 0.668148 | 0.641467 | 0.499206 | 6108 | 71 | 3702 / 1099 / 2406 |
| Snow | F | 0.655139 | 0.628401 | 0.489038 | 6235 | 71 | 3702 / 1099 / 2533 |

| 天气 | F3−F2 O AP70（百分点） | F3−F2 O* AP70（百分点） | F3−F2 F AP70（百分点） | F3 推理帧/秒 | 峰值已分配显存 GiB |
|---|---:|---:|---:|---:|---:|
| Clean | +0.7375 | +0.8062 | +0.8062 | 4.00 | 1.590 |
| Fog | −1.2586 | −1.2435 | −1.5318 | 4.32 | 1.239 |
| Snow | −1.4493 | −1.2822 | −1.3257 | 4.73 | 1.234 |

差值用未四舍五入的正式 AP 计算。推理计时包含按帧同步的模型执行和后处理。这里的 F3 是本节 R-A/R-B 消融，下一节的 F3 属于细尺度跨车交互实验。

海光证据目录为 `/data/cjm/datasets/logs/v2xr-radar-ra-rb-v1/F3/`。已核对 `GROUP_COMPLETE.json` 登记的文件哈希、20 轮记录和实际权重哈希。

- 第 16 轮 `epochs/16/weights.pth` SHA256 `cb74085abde82de65e908adcd055fea0b85a9dd3333ce2ca251b544219da89bc`。
- `FINAL_RESULTS.json` SHA256 `14526677e7b986b2abea49efe9f49a1a51e64c9e35d49b4f5ce460dfed888178`。

## 70. 细尺度跨车交互 F2/F3/F4（2026-11-11 海光实测）

本节 F2 指细尺度低秩交互，与第 69 节 R-A/R-B 的 F2 属于不同实验。源码位于 `v2xr_fusion_interaction/`。三组分别计划使用海光物理 GPU 2、3、4，独立加载同一份 P0 第 18 轮权重，SHA256 `0eb0ec8bd6a6e0bc368c7cc8aacd49a33cfe014413463f8b314fdbf76c6defd2`。

D1 海光正式结果支持在邻车特征内容方向继续验证。仅替换内容后的 Fog/Snow F AP70 为 50.19%/54.45%，对应 P0 47.60%/48.64%。D1–D3 最终假设判定均为证据不足，历史汇总见 `/data/cjm/datasets/logs/v2xr-fusion-diagnostics-v1/SUMMARY.md`。

- F2 在尺度 0、1 的原 AttFuse 输出上加秩 16 的同格主车—邻车乘积修正。F3 按车对直接柱体支撑乘 1、0.25 或 0。F4 再保护恰有一辆车提供直接支撑的格。三组各新增 18,432 个可训练参数，原模型及批归一化状态冻结。
- 海光三卡内存验收通过。初始与关闭分支的前后融合张量、两个检测头和官方最终候选框与 P0 逐元素一致。三组使用完全相同的两批真实数据和初始投影。两尺度 W 首步更新，U/V 第二步有非零梯度。冻结状态、空雷达、三种天气各 3 帧和完整内存状态恢复均通过。训练验收峰值已分配显存约 9.40、9.41、9.41 GiB。
- 正式入口 `v2xr_fusion_interaction/run_all.sh` 已同步海光并通过 `sh -n`。2026-11-11 16:16 核验，指定输出 `/data/new_disk/cjm_logs/v2xr-fusion-interaction-v1/` 尚未创建，父目录为 root 管理且 `cjm` 无写权限。入口在落盘前退出，20 轮正式训练和全量评测尚未开始。内存验收回传日志位于 `v2xr_fusion_interaction/evidence/`。
- 目录开通后，入口会执行三组正式 smoke 和启动门槛，随后并行训练、评测及等幅度功能消融。全量结果、逐目标统计及方法判定由服务器在指定目录生成，完成后生成 `AI_CONTEXT_APPEND.md` 和对应历史追加记录。

- 2026-11-11 16:27 海光确认本轮目录由 cjm 持有并可写，统一入口启动。16:29 三组正式 smoke 通过同初始化和真实批次顺序的统一检查，每组完成两个 batch、三种天气各 12 帧以及检查点恢复。F2/F3/F4 分别在卡 2/3/4 开始正式训练，第 1 轮均已记录至少 25 / 1,010 次更新。正式启动和验收日志已回传至 v2xr_fusion_interaction/evidence/official/。

### 70.1 正式结果与执行状态（2026-11-12 14:14，北京时间）

服务器目录为 `/data/new_disk/cjm_logs/v2xr-fusion-interaction-v1/`。F2、F3、F4 均已完成 20 轮、20,200 次更新。按每轮完整 Clean 验证的 O AP70 选择权重，三个选中权重的 SHA256 实算均与记录一致。

| 组别 | GPU | 选中轮次 | 选模 Clean O AP70 | 训练峰值已分配显存 GiB | 核验状态 |
|---|---:|---:|---:|---:|---|
| F2 | 2 | 3 | 0.705343 | 9.11 | 三天气全量评测完成，机制诊断失败 |
| F3 | 3 | 9 | 0.704740 | 9.12 | 全量评测进行中 |
| F4 | 4 | 15 | 0.705906 | 9.12 | 三天气全量评测完成，机制诊断失败 |

以下仅录入完整落盘的三天气评测，每种天气各 829 帧。逐帧核对 F 的目标 ID 和框身份与历史 LR0 一致，每种天气均为 6,235 个固定目标。AP 为 0–1 小数，匹配、误检和漏检按 0.7 重叠阈值计数。

| 组别或同权重控制 | 天气 | 协议 | AP30 | AP50 | AP70 | 匹配 / 误检 / 漏检 |
|---|---|---|---:|---:|---:|---|
| F2 | clean | O | 0.854230 | 0.829615 | 0.705343 | 5129 / 1617 / 1104 |
| F2 | clean | O* | 0.853956 | 0.829349 | 0.705116 | 5129 / 1617 / 1106 |
| F2 | clean | F | 0.853956 | 0.829349 | 0.705116 | 5129 / 1617 / 1106 |
| F2 | fog | O | 0.762880 | 0.722434 | 0.517306 | 3835 / 1670 / 1860 |
| F2 | fog | O* | 0.761276 | 0.720915 | 0.516218 | 3835 / 1670 / 1872 |
| F2 | fog | F | 0.714333 | 0.675496 | 0.477483 | 3851 / 1654 / 2384 |
| F2 | snow | O | 0.714645 | 0.681623 | 0.498541 | 3846 / 1609 / 2242 |
| F2 | snow | O* | 0.712305 | 0.679391 | 0.496909 | 3846 / 1609 / 2262 |
| F2 | snow | F | 0.700637 | 0.667497 | 0.487522 | 3849 / 1606 / 2386 |
| F4 | clean | O | 0.854462 | 0.831633 | 0.705906 | 5131 / 1605 / 1102 |
| F4 | clean | O* | 0.854188 | 0.831366 | 0.705680 | 5131 / 1605 / 1104 |
| F4 | clean | F | 0.854188 | 0.831366 | 0.705680 | 5131 / 1605 / 1104 |
| F4 | fog | O | 0.764406 | 0.716727 | 0.515713 | 3840 / 1697 / 1855 |
| F4 | fog | O* | 0.762799 | 0.715220 | 0.514629 | 3840 / 1697 / 1867 |
| F4 | fog | F | 0.715901 | 0.669453 | 0.475451 | 3854 / 1683 / 2381 |
| F4 | snow | O | 0.712570 | 0.679501 | 0.499141 | 3847 / 1613 / 2239 |
| F4 | snow | O* | 0.710004 | 0.677054 | 0.497343 | 3847 / 1613 / 2261 |
| F4 | snow | F | 0.698400 | 0.665177 | 0.487720 | 3849 / 1611 / 2386 |
| F4 权重按 F3 公式 | clean | O | 0.854743 | 0.831746 | 0.704530 | 5129 / 1612 / 1104 |
| F4 权重按 F3 公式 | clean | O* | 0.854469 | 0.831479 | 0.704304 | 5129 / 1612 / 1106 |
| F4 权重按 F3 公式 | clean | F | 0.854469 | 0.831479 | 0.704304 | 5129 / 1612 / 1106 |
| F4 权重按 F3 公式 | fog | O | 0.764104 | 0.716769 | 0.516143 | 3844 / 1700 / 1851 |
| F4 权重按 F3 公式 | fog | O* | 0.762498 | 0.715261 | 0.515057 | 3844 / 1700 / 1863 |
| F4 权重按 F3 公式 | fog | F | 0.715597 | 0.669463 | 0.475845 | 3858 / 1686 / 2377 |
| F4 权重按 F3 公式 | snow | O | 0.712105 | 0.679025 | 0.498515 | 3847 / 1616 / 2239 |
| F4 权重按 F3 公式 | snow | O* | 0.709540 | 0.676579 | 0.496720 | 3847 / 1616 / 2261 |
| F4 权重按 F3 公式 | snow | F | 0.697966 | 0.664713 | 0.487108 | 3849 / 1614 / 2386 |
| F4 权重等幅度统一缩放 | clean | O | 0.854937 | 0.832104 | 0.705142 | 5129 / 1610 / 1104 |
| F4 权重等幅度统一缩放 | clean | O* | 0.854663 | 0.831837 | 0.704916 | 5129 / 1610 / 1106 |
| F4 权重等幅度统一缩放 | clean | F | 0.854663 | 0.831837 | 0.704916 | 5129 / 1610 / 1106 |
| F4 权重等幅度统一缩放 | fog | O | 0.763748 | 0.717386 | 0.516297 | 3843 / 1692 / 1852 |
| F4 权重等幅度统一缩放 | fog | O* | 0.762142 | 0.715877 | 0.515211 | 3843 / 1692 / 1864 |
| F4 权重等幅度统一缩放 | fog | F | 0.715285 | 0.669998 | 0.475986 | 3857 / 1678 / 2378 |
| F4 权重等幅度统一缩放 | snow | O | 0.712201 | 0.679338 | 0.497113 | 3843 / 1614 / 2243 |
| F4 权重等幅度统一缩放 | snow | O* | 0.709636 | 0.676891 | 0.495323 | 3843 / 1614 / 2265 |
| F4 权重等幅度统一缩放 | snow | F | 0.698068 | 0.665022 | 0.485739 | 3845 / 1612 / 2390 |

| 组别 | 推理帧/秒 Clean / Fog / Snow | 推理峰值已分配显存 GiB Clean / Fog / Snow |
|---|---|---|
| F2 | 12.03 / 12.28 / 11.86 | 1.225 / 1.223 / 1.220 |
| F4 | 12.81 / 13.14 / 13.76 | 1.228 / 1.226 / 1.222 |

在 F4 同权重控制中，F4 对按 F3 公式执行的 F AP70 差值为 Clean +0.138、Fog −0.039、Snow +0.061 个百分点。对等幅度统一缩放的差值为 Clean +0.076、Fog −0.053、Snow +0.198 个百分点。Fog 和 Snow 的变化方向不同，逐目标独有观测保护收益待机制诊断。表中的 F4 权重按 F3 公式是功能控制，独立训练的 F3 仍在评测。

F2 和 F4 的正式预测与评分已保存。两组在逐目标机制诊断调用 `support['objects']` 时抛出 `KeyError: 'objects'`，详见各自 `FAILURE.json`。统一 P0 复测及总汇总尚未生成。F2–F4 相对本轮 P0 的正式增益和机制结论待补。

核验文件哈希：
- F2 选中权重 SHA256 `c6dbed26a6059ca681734859a9c45dc8086ccb6fc830517a64faca7b07f722c0`。
- `F2/final_evaluation/summary.json` SHA256 `2eeb48285237395a54c03bef79c73ff951ef197a98279e971def80952d2db5ab`。
- F3 选中权重 SHA256 `3ed223b2a30e7991412f190c1883800a55c58385873262cdc0f969226c960e64`。
- F4 选中权重 SHA256 `3a70e8d1bc416b728620d78ea67096b03dc739c1da0a77033323dfd6a0e2d76e`。
- `F4/final_evaluation/summary.json` SHA256 `2d9c24b2f81afd59e8552789eb38b2a7fa30dc57ab27d12a119c66ff68b4d1c4`。
- `F4/controls/F3/summary.json` SHA256 `39172f1614d2068a36aa184ab558ce12c672d7c6d171fb65acd9b87203c2555c`。
- `F4/controls/uniform3/summary.json` SHA256 `07a638ce3c1f4687ad9f68c75b6a442b0ec6b777ebc14a01fb729f28c24b2e6e`。

### 70.2 正式复测与机制诊断（2026-11-12 14:59，北京时间）

已补齐同环境 P0、F3 及三组机制诊断。原报错来自 `object_support()` 的返回值已经是以目标 ID 为键的字典。修复后直接读取并逐帧核验固定目标 ID。恢复流程核验原 20 轮完成记录、Clean O AP70 选中权重及冻结参数哈希，复用已有完整评测，只补缺项。三组均未重新训练。F3 第一次接续遇到旧进程退出时显存尚未释放，经设备空闲检查后重试成功。

每种天气评测 829 帧。固定真值协议 F 使用同一批 6,235 个目标。下表为 F AP70，完整 O、O*、F 三协议 AP30/50/70、误检、漏检、帧率和显存见 `SUMMARY.md`。

| 组别 | Clean | Fog | Snow |
|---|---:|---:|---:|
| P0 | 70.349% | 47.640% | 48.631% |
| F2 | 70.512% | 47.748% | 48.752% |
| F3 | 70.451% | 47.359% | 48.754% |
| F4 | 70.568% | 47.545% | 48.772% |

按场景内相邻 16 帧成块，使用相同帧块做 2,000 次配对重采样。Fog/Snow 中 F2、F3、F4 对 P0 的 95% 区间都覆盖零。Clean 的 F4 对 P0 为 +0.219 个百分点，区间 [+0.104, +0.381]。Fog 的独立训练 F3 对 F2 为 −0.390 个百分点，区间 [−0.772, −0.025]。这些区间仅反映本验证集的样本波动，未校正多重比较，未覆盖跨训练种子的波动。

| 组别 | Fog 修正 / 原融合幅度（0.8 m / 1.6 m） | Snow 修正 / 原融合幅度（0.8 m / 1.6 m） | 仅邻车有点目标相对 P0 恢复 / 新增丢失（Fog；Snow） | 历史目标相对 P0 恢复 / 新增丢失（Fog；Snow） |
|---|---|---|---|---|
| F2 | 2.552% / 14.510% | 2.406% / 14.545% | 4 / 6；5 / 3 | 12 / 2；15 / 6 |
| F3 | 0.410% / 0.280% | 0.477% / 0.323% | 2 / 4；0 / 0 | 3 / 2；6 / 7 |
| F4 | 0.201% / 0.217% | 0.246% / 0.259% | 1 / 3；0 / 1 | 0 / 2；5 / 5 |

- F2 的修正在 1.6 米尺度约为原融合均方根幅度的 14.5%，目标有恢复也有新增丢失，尚无稳定 AP 收益的证据。冻结原网络且关闭分支的真实样本等价检查通过，P0 作为全量关闭对照。
- F3 在 Fog/Snow 的两尺度中有 91.9%–95.8% 的主车–邻车对格子双方均无直接观测。同权重下，对无约束公式的 Fog/Snow 差值为 −0.103 / −0.115 个百分点，对等幅缩放为 −0.224 / +0.140 个百分点，四个区间均覆盖零。
- F4 将同权重 F3 在所有主车独有格、邻车独有格上的非零修正置零。Fog/Snow 中，F4 对同权重 F3 公式的仅邻车有点目标恢复 / 新增丢失为 0 / 2、0 / 1，对同权重等幅缩放均为 0 / 1。F4 对等幅缩放的 AP70 差值为 −0.053 / +0.198 个百分点，区间均覆盖零。当前数据对独有观测保护的独立检测收益证据不足。格子独有观测和框内仅邻车有点分别统计。

**F3 正式结果补充。** 第 9 轮权重的正式评测、同权重控制和机制统计均已完成。下表为固定真值协议 F 的 AP70，差值单位为百分点。

| F3 及其对照 | Clean | Fog | Snow |
|---|---:|---:|---:|
| F3 正式成绩 | 70.451% | 47.359% | 48.754% |
| F3 权重，关闭观测约束 | 70.001% | 47.462% | 48.869% |
| F3 权重，等幅度统一缩放 | 70.404% | 47.583% | 48.615% |
| F3 相对本轮 P0 复测 | +0.102 | −0.281 | +0.123 |

F3 相对本轮 P0 的变化随天气而异。Fog 和 Snow 的配对区间均覆盖零，两个同权重对照也未提供观测约束有独立收益的证据。

服务器结果目录 `/data/new_disk/cjm_logs/v2xr-fusion-interaction-v1/`。全量汇总见 `FINAL_RESULTS.json`，配对结果见 `PAIRED_RESULTS.json`、`PAIRED_COMPARISON.md`。各组 `MECHANISM.json`、`MECHANISM_ANALYSIS.md` 和 `TARGET_COMPARISONS.jsonl` 保存分层统计及逐目标记录。报告和恢复日志的本地副本在 `v2xr_fusion_interaction/evidence/recovery/`。

## 71. V1/V2/V3 冻结 P0 的跨车融合假设诊断（2026-11-12 启动记录）

本轮不训练神经网络。V1 检查逐车候选覆盖，V2 比较位置、局部特征与区域结构，V3 检查特征变化和目标框误差的跨车相关性。冻结 P0 第 18 轮，权重 SHA256 与第 67 节一致。三个任务共享三天气的逐帧缓存。物理卡 2 负责共享提取，随后 V1、V2、V3 分别在物理卡 1、2、3 并行运行。

海光完成 3 项单元测试与三组真实帧烟测。每种天气用 3 帧，共用 59 个固定目标。三天气多车目标匹配与同环境 P0 档案一致。主车单车重放与历史 P0/single 在 Snow 有 1 个目标匹配差异，Clean 和 Fog 均为 0。全量报告按正式 P0 档案分组，逐帧缓存保留重放差异。烟测只核对链路，全量结果和研究判断待运行完成。

代码位于 v2xr_fusion_hypothesis_audit/。本轮日志和产物在 /data/cjm/datasets/logs/v2xr-fusion-hypothesis-audit-v1/。统一日志 run_all.log，各任务日志 V1.log、V2.log、V3.log。成功后服务器 AI_CONTEXT.md 追加正式报告。
