# 租用 NVIDIA 服务器配置与迁移上下文

海光端精简任务路径修订（2026-10-24，尚待服务器执行）：依据 AI_CONTEXT.md §1/§4，独立代码放 `/home/cjm/OpenCOOD-main/cjmnet_pointcloud_delivery`，保留原项目；压缩包放 `/data/cjm/datasets/packages/v2xr-pointcloud-only-v1/cjmnet_repe_cuda_flat.zip`，精简数据仍为 `/data/cjm/datasets/v2xr-pointcloud-only-v1`，日志/PID/退出码放 `/data/cjm/datasets/logs/v2xr-pointcloud-only-v1`，TMPDIR 为 `/data/cjm/datasets/tmp/v2xr_pointcloud_only_v1`。不再使用此前建议的 `~/cjmnet_pointcloud_delivery`、`~/v2xr_slim_prepare.log` 等位置。新启动器 `v2xr_pointcloud_only/run_hygon_prepare.sh` 串行执行生成和抽样检查，不训练。

2026-10-24 新增待执行工作：用户要求海光端先精简图片再迁移。新增 `v2xr_pointcloud_only/slim_data.py`，保留图片真实尺寸及原SHA256，将非图片文件硬链接到新根 `/data/cjm/datasets/v2xr-pointcloud-only-v1`，不删除或改写原数据。雷达、点云、YAML与原验收记录保留。`check_equivalence.py` 在海光原环境比较少量优化前后的官方批次/标注和随机状态；通过后写 EQUIVALENCE.json。GSPR/REPE CUDA 新增显式 `POINTCLOUD_ONLY_ROOT` 模式，仅在进程内替换图片读取函数并沿用原视野裁剪，输出和预检使用新的独立路径。原始官方和历史GSPR快照不修改。尚未在服务器执行预处理、等价检查或训练；最新目标可用68G，需先获得精简大小再确定迁移空间，不自动删除远端旧完整副本。详情见 `v2xr_pointcloud_only/README.md`。

最新补充（2026-10-24，本轮数据校验）：三个目录的续传已回传 `to-chk=0/284775`、`TRANSFER_EXIT=0`，传输任务完成。随后跨机 rsync 内容校验因远端断开连接失败（255），尚无内容校验通过结论。重新测试 SSH 已输出 `SSH_READY`；目标 rsync 为 `/usr/bin/rsync`，最新磁盘为 300G 总量、233G 已用、68G 可用。以下早期章节中的“传输中”和旧磁盘数字作为历史快照保留，以此补充为最新状态。

本轮新增 `v2xr_cuda_delivery/file_integrity.py`：两台服务器分别后台生成三个目录全部文件的 SHA256 清单，随后仅传输清单并比较，不依赖长时间 SSH 连接。跟随软链接读取真实文件，检查缺失、多余及内容变化；清单须具有完成标记。不写入数据目录、不导入模型、不运行科研实验。仅完成本地临时文本样例测试，远端尚待执行。其结果用于核对数据复制，不能代替改进版 GSPR/REPE 自身的输入验收或 GPU 试跑。

更新：2026-10-24（Asia/Taipei）。依据本次对话中的服务器输出及本地安装、打包脚本整理，未直接登录远端检查。磁盘用量和传输进度为最近一次回传快照，会继续变化。

本文记录租用服务器。原海光服务器的研究历史、实验结论与冻结证据继续查阅 `AI_CONTEXT.md`，本文不替代或改写那些记录。

## 当前状态

- 独立 Python 3.10 / PyTorch 2.5.1+cu124 / CUDA 12.4 环境已通过用户回传版本确认。
- 两个项目原生算子已编译成功；尚无真实 GPU 算子执行和模型训练成功的回传记录。
- 原 V2X-R 精简源码已解压到 `/workspace/cjmnet`，源码完整性检查通过。
- 用户已确认 GSPR 源码解压于 `/workspace/gspr_source/cjmnet`；本次 REPE 继续使用该位置。
- 三个数据目录正在从海光服务器传入 `/data/cjm/datasets`。首次连接中断，续传已启动；尚未回传 `TRANSFER_EXIT=0`。
- **已新增独立 GSPR REPE CUDA 入口（§9），仅通过本地静态检查，服务器验收待执行。** 原海光脚本仍不能直接使用；新入口使用本次CUDA环境和重新生成的输入/GT记录。数据传输完成、GPU执行、显存和训练速度均待服务器验证。
- 截至本记录，租用服务器没有已确认完成的科研训练或评测结果。

## 1. 服务器与连接

| 项目 | 已知配置或状态 |
|---|---|
| 平台 | Compshare；AutoDL 无可用资源后改用此平台 |
| 实例名称 / 终端主机名 | `cpod-1wth1jwjqyow` |
| 登录用户 | `root` |
| SSH 地址 | `cpod-1wth1jwjqyow.podtcp.compshare.cn` |
| SSH 端口 | `28374` |
| 系统 | 登录回显 Ubuntu 22.04.5 LTS，x86_64 |
| 登录回显内核 | `6.12.101-compshare` |
| 计划使用的 GPU | 单张 RTX 4090，24GB；实际分配、驱动、显存和 GPU 可见性尚待 `nvidia-smi` 确认 |
| CPU、分配核数、内存 | 未回传，不能按平台常见配置推测 |
| 准备方式 | 最初通过无卡模式创建实例、安装环境和编译算子；当前是否切换为有卡模式未确认 |

登录：

```bash
ssh -p 28374 root@cpod-1wth1jwjqyow.podtcp.compshare.cn
```

SSH 已从海光服务器成功连接。平台重建实例后地址、端口可能变化，应以当时控制台为准。密码和私钥不写入本文。

## 2. Python 与 CUDA 环境

平台原镜像标识为 `cuda132_torch2130_py312`，页面显示 Ubuntu 22.04、CUDA 13.2、PyTorch 2.13.0、Python 3.12。原环境提示符为 `(py312)`，实测输出 `2.13.0+cu132 13.2`。

项目使用另外创建的环境，保留平台原环境：

| 项目 | 项目环境 |
|---|---|
| 环境名称 | `cjmnet-cu124` |
| 环境根目录 | `/workspace/cjmnet_cuda_env` |
| Python | 3.10；具体补丁版本待远端记录 |
| Python 解释器 | `/workspace/cjmnet_cuda_env/envs/cjmnet-cu124/bin/python` |
| PyTorch | `2.5.1+cu124`，已回传验证 |
| PyTorch 所带 CUDA 运行库版本 | `12.4`，已回传验证 |
| torchvision | 安装脚本固定 `0.20.1+cu124` |
| CUDA 编译器 | `/workspace/cjmnet_cuda_env/cuda-12.4/bin/nvcc` |
| 编译目标 | 安装脚本固定 `TORCH_CUDA_ARCH_LIST=8.9`，对应 RTX 4090 |
| 环境激活脚本 | `/workspace/cjmnet_cuda_env/activate.sh` |
| 安装和构建日志目录 | `/workspace/cjmnet_cuda_env/logs` |

每次新开终端先激活：

```bash
source /workspace/cjmnet_cuda_env/activate.sh
python -c "import sys, torch; print(sys.executable); print(torch.__version__, torch.version.cuda)"
```

预期输出项目解释器和 `2.5.1+cu124 12.4`。仅看到 `(py312)` 时，尚不能当作已进入项目环境。

安装脚本中的主要依赖固定值：NumPy 1.26.4、SciPy 1.15.3、Shapely 2.0.0、Open3D 0.18.0、Cython 0.29.37、spconv 2.3.6、cumm 0.4.11。完整版本清单以 `cuda_env_setup/setup_cjmnet.sh` 及远端 `/workspace/cjmnet_cuda_env/environment_check.json` 为准；该 JSON 内容尚未完整回传。

spconv/cumm 在此路径用于 CPU 点云体素化；模型运算使用 PyTorch GPU。pypcd 使用 klintan 分支固定源码版本 `796e545df2ee5e7836df6fe2f274aae3428d1ac2`，不以 PyPI 同名旧包替代。安装自检曾因单点 ASCII 样例失败，已改用双点样例；真实数据读取仍须单独验证。

CUDA 编译工具和 PyTorch 运行库已准备，不代表宿主 NVIDIA 驱动和 GPU 执行已验证。不从平台镜像名称推断驱动版本。

## 3. 源码位置与交付包

| 用途 | 路径与状态 |
|---|---|
| 原 V2X-R 精简源码 | `/workspace/cjmnet`，已解压并通过 `verify_source_bundle.py` |
| 原包服务器上传路径 | `/workspace/cjmnet_v2xr_source.zip`，用户已执行解压 |
| GSPR 专用源码包建议上传路径 | `/workspace/cjmnet_gspr_source.zip`，用户已执行解压 |
| GSPR 专用源码建议解压位置 | `/workspace/gspr_source/cjmnet`，用户已执行解压 |
| 环境安装脚本约定上传位置 | `/workspace/setup_cjmnet.sh`；源码内副本为 `cuda_env_setup/setup_cjmnet.sh` |
| 本地工作区 | `D:\Collaborative perception\mymodule\cjmnet` |
| 海光原项目 | `/home/cjm/OpenCOOD-main/cjmnet` |

原包回传校验：`source_integrity_passed`，635 个清单文件、6 个源码快照包，`training_executed=false`、`cuda_migration_complete=false`。

GSPR 专用交付包本地路径：

```text
delivery/gspr_source_20261023_203000/cjmnet_gspr_source.zip
```

大小 5,172,189 字节；609 个文件；5 项静态检查通过。SHA256：

```text
a7763b666c162d830a1207df15ff90b08443153c0c86bcc4525e4dfdc0d19d48
```

该包包括 `v2xr_gspr_clean`、冻结 GSPR 文件、`v2xr_train_attfuse` 共享训练引擎、`v2xr_protocol_audit`、`v2xr_official_repro` 官方源码及校验依赖、环境和数据检查辅助文件。不含数据、权重和历史训练结果，不含 `v2xr_weather_l012` 天气混合实验入口。GSPR 专用包按现有 Clean-only 实现交付，不应与天气混合配方混记。

解压到新目录以免覆盖已有源码，服务器不要求存在 `.git`：

```bash
python -m zipfile -e /workspace/cjmnet_gspr_source.zip /workspace/gspr_source
source /workspace/cjmnet_cuda_env/activate.sh
python /workspace/gspr_source/cjmnet/verify_source_bundle.py
```

上述目标目录须为空或不存在。源码校验成功仅说明文件与交付清单一致。切换源码目录后，已有运行副本不会自动更新。

## 4. 已编译算子与运行副本

官方源码来自冻结的 V2X-R 版本，保存在项目 `v2xr_official_repro/upstream` 内。环境脚本将其复制到独立运行目录编译，保留上传源码不变。

| 项目 | 路径或结果 |
|---|---|
| 官方运行副本 | `/workspace/cjmnet_cuda_env/official-runtime-cu124` |
| CPU 框重叠算子 | `box_overlaps` |
| 框内点检查算子 | `roiaware_pool3d_cuda`，包含 CPU/CUDA 实现 |
| 构建回传结果 | `SUCCESS: build-ops` |
| 成功构建日志 | `/workspace/cjmnet_cuda_env/logs/build-ops_20261023T101742Z_28054.log` |
| 构建检查记录约定路径 | `/workspace/cjmnet_cuda_env/native_check.json`，内容尚未完整回传 |

只编译本模型实际需要的扩展，未编译无关的旧 PointNet2 算子。不复制海光编译的 `.so` 到 NVIDIA 环境。无卡编译成功不等于 GPU 算子执行通过。

## 5. 磁盘与数据路径

最近一次回传：

```text
Filesystem  Size  Used  Avail  Use%  Mounted on
overlay     300G  217G   84G   73%  /
```

`df -hT /data` 显示 `/data` 位于根目录的 overlay 文件系统，与环境和源码共用容量。最初为 92G，扩容后显示 300G；未确认独立挂载数据盘。平台停止、释放或重建实例时的持久化规则尚未核实。

数据来源为海光服务器 `cjm@admin1`。源端和目标端使用相同的目录名称：

| 数据 | 两端路径 | 最近源端统计 | 最近目标端统计 |
|---|---|---|---|
| 完整训练集及验收文件 | `/data/cjm/datasets/v2xr-official-train-v2` | 162G | 162G |
| 配对天气验证集 | `/data/cjm/datasets/v2xr-lidar-weather-validate-v1` | 13G | 26G |
| 官方验证视图及辅助文件 | `/data/cjm/datasets/v2xr-official-validate-v1` | 7.8G | 7.3G |

目标端统计发生在传输过程中，不能据此确认任一目录完整。源端使用 `du -shL`，目标端使用 `du -sh`；原目录包含软链接和共享文件，数值不能直接作为内容一致性证明。

训练读取目录为 `/data/cjm/datasets/v2xr-official-train-v2/train`。海光既有验收记录为 8083 个协作帧、32 个有数据场景、92 个场景车辆组合、24088 个单车帧。保留 `manifest.json`、`plan.json`、`data_acceptance.json`、`records/` 等记录。公开参考 8084 与实际 8083 的既有接受记录不修改，不补造或重复帧。

验证集为 6 场景、829 个协作帧、3041 个单车帧。配对根目录包含 `payload/validate` 与 `clean/fog/snow` 读取视图；官方 Clean 读取目录为 `/data/cjm/datasets/v2xr-official-validate-v1/validate`，辅助文件位于同根 `auxiliary/`。模型虽以 LiDAR 为输入，当前官方加载器仍会读取雷达和四张相机图片，必须保留。

旧协议另有天气验证读取目录，真实位置记录在海光服务器：

```text
/data/cjm/datasets/logs/v2xr-official-gt-audit-v1/input_manifest.json
```

其中 `overlay` 字段尚未回传，不能假定已包含在本次三个目录中。原始 `.7z`、历史 D−1A/B/C 证据目录、纯 AttFuse 基线权重及完整日志未确认迁移；旧入口可能依赖这些外部文件。正式 test 未列入本次传输。

## 6. 数据传输进度与规则

从海光服务器推送到租用服务器，使用 `rsync -rltL --partial --append-verify`。`-L` 将软链接指向的内容复制为实际文件，可能让共享文件在目标端占用多份空间。因此，先前按约 183G 推算“传完剩约 96G”只作早期估计，不能作容量保证。配对验证目录已显示更大目标占用，但增长原因尚未逐文件核验。

- 传输专用 SSH 密钥约定位置：海光用户目录 `~/.ssh/cjmnet_transfer`；不在本文保存密钥内容。
- 首次日志：海光服务器 `~/v2xr_transfer.log`，连接被远端关闭，`Broken pipe (32)`、`TRANSFER_EXIT=10`，未完成。
- 续传日志：海光服务器 `~/v2xr_transfer_resume.log`；最近回传约 22,462,843,346 字节、10%、15.16MB/s，仍有 `ir-chk`，尚未确认完成。
- 每次运行会重新扫描目录，进度百分比重新计算；目录未扫描完时总量仍会变化。百分比不能直接推算剩余磁盘需求。
- 已完成且大小、修改时间一致的文件通常跳过；不完整文件继续传输。相同目标路径的续传不会自动创建另一套目录。
- 不同时启动多个相同传输任务，不删除源数据或未完成文件。以本次日志最后出现 `TRANSFER_EXIT=0` 判断进程成功，再进行数据完整性校验。
- `setlocale: LC_ALL ... zh_CN.UTF-8` 警告未造成此前 SSH 或 rsync 失败。

查看传输与空间，不启动新任务：

```bash
# 海光服务器
tail -n 5 ~/v2xr_transfer_resume.log

# 租用服务器
df -h /data
```

## 7. 训练前尚需完成

1. 等待数据传输成功，确认最终空间并核对数据、目录和哈希。基础文件检查通过不等于官方加载器和全部内容校验通过。
2. 确认最终使用 `/workspace/cjmnet` 还是 GSPR 独立副本，记录其源码哈希，避免混用多个版本。
3. 独立适配 CUDA 训练入口。原入口包含 ROCm/HIP 强制检查、海光绝对路径、历史环境及算子身份约束，不能只更换解释器后直接运行。
4. 核实历史数据验收、天气读取目录和基线证据等实际所需外部输入；不伪造旧记录或改写旧哈希使检查通过。
5. 分配 GPU 后核对型号、驱动及可见性，检查真实算子执行，再运行少量训练和验证批次，确认损失、显存、保存及恢复正常。
6. 试跑通过后再正式训练。RTX 4090 24GB 能否容纳现有 batch8 及原验证流程尚未实测，不自动减小批次或改模型掩盖失败。

速度比较需保持模型、数据、批次大小、精度、优化器和训练配置一致；分别记录训练、数据读取、验证和评测耗时。CPU、磁盘及加载进程差异也影响总耗时，不能将整轮耗时直接当作纯 GPU 性能。

## 8. 后续协作约束

- 本地编写和静态检查，上传远端后运行实验；不在本地运行真实模型、数据处理或科研训练。
- 冻结的 GSPR 和既有科学实验不直接修改，CUDA 适配放独立目录，并记录实际源码和配置身份。
- 服务器允许没有 `.git`；用源码 SHA256 校验版本，不要求分支或 commit 检查才能运行。
- 长任务提供后台命令，保留 PID、日志和退出状态；检查已有任务后再恢复，避免重复运行。
- 海光服务器的 `ROCR_VISIBLE_DEVICES` 规则仅适用于旧环境。NVIDIA 入口需按实际 GPU 分配设置 CUDA 可见性，不照搬 ROCm 限制或覆盖平台设备隔离配置。
- 不使用正式 test 调参或选模型；不改变既有训练和评测定义来解决迁移问题。
- 更新本文时明确区分“计划”“脚本约定”“用户回传已验证”，不得将编译成功写成训练成功，或将数据传输中写成验收完成。


## 9. GSPR后端REPE的独立CUDA入口（2026-10-24本地适配）

本轮任务是`gspr_repe/`可靠性聚合A、观测Q/S模块B和邻域增强C的两阶段实验，采用Clean/Fog混合训练B0–B4；与§3早期Clean-only GSPR专用包分开。新增`cuda_runtime.py`和`cuda_evaluation.py`，默认`BACKEND=cuda`；原海光`BACKEND=rocm`保留。冻结GSPR、旧实验及原快照不修改。

路径默认采用本文已有环境：解释器`/workspace/cjmnet_cuda_env/envs/cjmnet-cu124/bin/python`，runtime=`/workspace/cjmnet_cuda_env/official-runtime-cu124`；训练、配对验证、官方验证三个根目录仍为§5路径。脚本不安装包或编译算子，不使用平台py312，也不覆盖平台已有CUDA_VISIBLE_DEVICES。实际GPU要求只有一张可见卡。

CUDA新增数据视图/审计默认`/data/cjm/datasets/gspr-repe-cu124-inputs-v1`，结果默认`/data/cjm/datasets/logs/gspr-repe-cu124-v1`。初次全量预检核对已提取训练文件及原验收/提取记录、配对验证payload和视图文件SHA256；rsync -L普通文件按内容核对。历史原始.7z、D−1A/B/C输出及纯AttFuse旧权重不作为新CUDA运行前提；保留历史归档身份记录，明确未在NVIDIA重新做过归档CRC。

使用冻结官方GT观察器重新读取三天气829帧，检查帧序、几何和6224/3673/5854目标数量，并记录本次环境、native.so、源码和数据身份。CUDA模式实际执行GPU/CPU框内点和CPU框重叠算子，编译成功不自动算作运行通过。评测包装调用原官方inference_intermediate_fusion、后处理、AP和匹配跟踪；没有重写AP/NMS。新外层包装与旧inference.main有差异，尚未实测跨CUDA/HIP逐位等价。

所有B0–B4保留batch8/20轮/seed20261017/原优化器和调度，Clean选模，Snow仅锁定后验证。4090的24GB能否容纳该配方仍待实测，不自动改小batch。技术失败停在阶段记录中，不进入后续阶段。

新交付打包器`delivery_tools/build_repe_cuda_source_bundle.py`生成`delivery/repe_cuda_*/cjmnet_repe_cuda_source.zip`，包含REPE及全部冻结依赖，不含数据/权重/历史结果。旧GSPR包不含本轮模块和完整天气混合依赖，不能继续用旧包代替。用户指定新包继续使用`/workspace/gspr_source/cjmnet`；更新前先将旧源码目录保留到旁边的备份位置，再解压到同一路径，已经编译的runtime继续保留。详细上传/后台启动命令见`gspr_repe/CUDA_REMOTE.md`。

本地验证范围：AST、模块间符号核对、POSIX sh语法、源码快照、纯标准库迁移合同测试、交付包CRC和逐文件哈希；没有运行Torch、真实数据、GPU或任何科研训练。未连接远端，未确认传输完成，不登记服务器阶段成功或AP。


### 9.1 源码路径与opencood布局澄清（2026-10-24）

用户明确继续使用`/workspace/gspr_source/cjmnet`，此前`/workspace/repe_cuda/cjmnet`只是隔离旧副本的建议，代码并未要求该名称。启动说明与打包器已统一为用户指定目录。压缩包包含cjmnet顶层，因此解压参数为`/workspace/gspr_source`，避免多套一层cjmnet。

项目根目录没有opencood是当前隔离布局的预期结果。官方原始源码位于`v2xr_official_repro/upstream/opencood`；实际导入`/workspace/cjmnet_cuda_env/official-runtime-cu124/opencood`，其中已有CUDA扩展。训练配置从该runtime的`opencood/hypes_yaml/V2X-R/LiDAR/V2XR_attfuse.yaml`读取，再生成本次带数据路径的输出config.yaml。无需移动官方配置或opencood到GSPR项目根目录；不覆盖已编译runtime。该说明依据本地代码，远端目录实际存在性仍由预检确认。


### 9.2 当前交付改为无cjmnet外层的压缩包（2026-10-24）

用户要求压缩包直接包含目标文件。新包名`cjmnet_repe_cuda_flat.zip`，顶层直接是gspr_repe、v2xr_gspr_clean等目录及源码校验文件，解压目标为`/workspace/gspr_source/cjmnet`。此前带cjmnet外层的旧包仍应解压到`/workspace/gspr_source`，两种格式不要混用。打包器和CUDA_REMOTE.md已更新并明确检查zip内没有cjmnet/前缀。

旧包解压到已有父目录时，会合并同名cjmnet目录，同名文件可能被覆盖，旧包中多余文件不会自动删除。新flat包覆盖已有项目也存在相同问题；应在没有旧任务运行时先保留旧cjmnet目录，再解压到原路径。没有删除远端文件或改变已编译runtime。
