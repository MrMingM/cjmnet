# 实验环境与配置上下文

> **2026-09-17 通信阶段最新结论（必读第16节）：`lossless_comm` 正式 benchmark 已完成。`lossless_full` 与 `level0_recompute` 在 Clean/Fog/Rain/Snow 的 AP30/AP50/AP70 共12项指标上均与 `raw_full` 完全一致到结果表六位小数；`level0_recompute` 将当前表中通信量从 23.8921 MiB/frame 降至 0.9901–2.8540 MiB/frame，对应减少 88.05%–95.86%。当前通信主线已从 learned block selection 转为“保留全覆盖、压缩表示冗余/重算层级冗余”。CPU codec 耗时较高暂不作为否决项；其设备/实现归因尚未单独验证。旧第8/12/13节的 selection/budget 内容保留为历史对照。**

本文档用于让 AI 快速了解本项目的系统环境、数据集、实验记录和当前研究约束。历史记录保留供追溯；当前执行边界以最新更新为准。

> **最新进度（2026-09-17，必读第16节）：通信压缩正式 benchmark 已完成。`lossless_full` 与 `level0_recompute` 在固定 Clean/Fog/Rain/Snow 正式测试上的 AP30/AP50/AP70 均与 raw/full 基线一致到结果表六位小数；其中 `level0_recompute` 的 Mean MiB/frame 为 Clean 2.8540、Fog 1.1632、Rain 1.4260、Snow 0.9901，对当前 raw_full 23.8921 MiB/frame 分别减少 88.05%/95.13%/94.03%/95.86%。当前优先保留这条“全覆盖无损编码＋接收端层级重算”路线，不再把 A0B0 budget scan 作为前置门禁。**

> **正式测试协议（2026-09-14，必读第11节）：与历史实验比较必须使用 OPV2V clean test 和已生成的 OPV2V-W fog/rain/snow test，关闭在线天气增强，完整测试、非全局排序 AP。validation＋在线模拟天气仅作开发验证，禁止称为 OPV2V-W 正式测试或与历史 test AP 直接相减。当前正式入口为 `bash gspr_evidence/run_benchmark.sh`，复用现有权重。**

> **当前最高优先级约束（2026-09-07）：用户正在进行 GSPR 正式多种子训练，GSPR-v1 源代码冻结。禁止修改 `attfuse_gspr/reliability.py`、`attfuse_gspr/pillar_vfe.py`、`attfuse_gspr/point_pillar_gspr_attfuse.py`、`gspr_supervision/` 下全部文件、`pretrain_gspr.py`、`train_gspr_multitask.py`。后续通信原型只能在隔离的新模块、新入口、新配置中开发，不得影响当前训练；详见第8节。旧记录中有关修改上述训练脚本的计划不构成解冻授权。**

注意，用户下面的这些环境是实际进行实验的环境，而你（也就是现在正在执行ai指令的你）所处的环境是用户在自己的win电脑上，用户是先根据你修改的代码，将修改代码后的文件通过远程服务器替换到实验服务器上，而当前的win电脑只有一个python 3.11的环境，没有安装代码所需要的包，所以说你如果在用户的win电脑上测试一个文件的话大概率会因为缺失包而不会跑通

## 1. 硬件与系统环境
- **远程训练服务器**：
  - **用户**：`cjm`
  - **主机**：`admin1`
  - **远程工作区根目录**：`/home/cjm/OpenCOOD-main` （新代码若有变动，请以此类推）
- **加速卡 (显卡) 配置**：
  - **系统可见设备**：共有 8 张加速卡（显示为 HCU 0–7）。
  - **可用设备**：当前 **HCU 0–6 可用**供实验分配，HCU 7 有固定占用，请勿使用。
  - **PyTorch 接口**：服务器环境底层的加速后端是 HIP，但在代码层面依然使用 PyTorch 原生的 `torch.cuda` API（或者 `--device cuda`）。如果设备不可见，报错信息会包含 `No HIP GPUs are available`。
  - **多卡隔离规则（极重要）**：
    - 在 Linux 环境中分配显卡时，**只能使用 `ROCR_VISIBLE_DEVICES=<physical_index>`** 来进行物理卡隔离。
    - 代码内部直接调用 `cuda:0` 即可。
    - **禁止**同时设置 `HIP_VISIBLE_DEVICES` 或 `CUDA_VISIBLE_DEVICES` 为相同编号，否则会引发二次过滤，导致脚本找不到设备。

## 2. 软件环境
- **Conda 虚拟环境**：`opencood`
- **Python 版本**：3.10
- **远程解释器路径**：`/home/cjm/miniconda3/envs/opencood/bin/python`
- **核心依赖栈**：PyTorch 2.5.1, NumPy 1.26.4, Shapely 2.0.0, 并且安装有 OpenCOOD。
- **本地开发环境**：
  - **Windows 本地路径**：`D:\Collaborative perception\mymodule\cjmnet`
  - **提示**：本地 Windows 环境并没有安装 `torch`。所以涉及到包含 `import torch` 等依赖运行时的单元测试，无法在本地环境完全跑通，需移步服务器环境运行。

## 3. 数据集与预训练模型路径
- **官方 OPV2V Clean 数据集**：
  - Train 路径（43 个场景）: `/data/scd/datasets/opv2v_official_data_dumping/train`
  - Validate 路径（9 个场景）: `/data/scd/datasets/opv2v_official_data_dumping/validate`
  - Test 路径（16 个场景）: `/data/scd/datasets/opv2v_official_data_dumping/test`
- **OPV2V-W (或 V2X-DGW) 恶劣天气测试数据集**：
  - Fog test: `/data/cjm/datasets/opv2v-w/fog/test`
  - Rain test: `/data/cjm/datasets/opv2v-w/rain/test`
  - Snow test: `/data/cjm/datasets/opv2v-w/snow/test`

## 4. 实验操作与存储规范
- **大文件存储纪律（强制）**：
  - **严禁**将产生或下载的占用内存较大的数据文件、大型日志或大权重文件存放在代码目录（即 workspace/ 项目路径）内。
  - 所有的大文件必须放置在外部数据盘：`/data/cjm/datasets/logs` 目录下。
- **实验调度**：若需使用终端命令跑实验，请确保指定了对应的 `ROCR_VISIBLE_DEVICES`，并使用 `/home/cjm/miniconda3/envs/opencood/bin/python` 解释器执行。

## 5. AttFuse 基线实验记录（2026-09-03 至 2026-09-04）

### 5.1 代码核验结论

- 本地开发目录：`D:\Collaborative perception\mymodule\cjmnet`
- 服务器部署目录：`/home/cjm/OpenCOOD-main/cjmnet`
- 当前目录不是完整的 OpenCOOD 仓库，而是从 OpenCOOD 中独立抽取的经典 PointPillar Attentive Fusion（AttFuse）核心实现及训练包装。
- 核心文件：
  - `attfuse_code/point_pillar_intermediate.py`
  - `attfuse_code/att_bev_backbone.py`
  - `attfuse_code/self_attn.py`
  - `attfuse_config.yaml`
  - `train_attfuse.py`
- 三个模型文件已经与官方 OpenCOOD 的经典实现联网核对。模型结构与官方 PointPillar Intermediate/Attentive Fusion 一致；本地版本主要将模块导入路径改为 `attfuse_code`，以便作为独立目录调用。
- AttFuse 在每个 BEV 空间位置上，对不同 CAV 的特征执行 scaled dot-product attention，并在 PointPillar 主干的多个尺度上进行特征融合。

### 5.2 训练数据和主要配置

- 训练集：`/data/scd/datasets/opv2v_official_data_dumping/train`
- 验证集：`/data/scd/datasets/opv2v_official_data_dumping/validate`
- 数据集：官方 OPV2V Clean
- 模型：PointPillar AttFuse（Intermediate Fusion）
- 优化器：Adam
- 初始学习率：`0.002`
- 实际训练轮数：`15 epochs`
- 训练 batch size：`4`
- 最大协作车辆数：`5`
- 特征压缩：关闭（`compression: 0`）
- 学习率策略：MultiStepLR，`gamma: 0.1`，milestones 为 `[10, 15]`
- GPU/HCU 隔离：`ROCR_VISIBLE_DEVICES=0`
- Python：`/home/cjm/miniconda3/envs/opencood/bin/python`
- 训练采用 `nohup` 后台运行，模型和日志均存放在外部数据盘，没有写入代码目录。

### 5.3 训练输出与最佳权重

- 完整训练输出目录：`/data/cjm/datasets/logs/attfuse_opv2v_20260903_200923`
- 训练配置副本：`/data/cjm/datasets/logs/attfuse_opv2v_20260903_200923/config.yaml`
- 训练日志：`/data/cjm/datasets/logs/attfuse_opv2v_20260903_200923/train.log`
- TensorBoard 日志：`/data/cjm/datasets/logs/attfuse_opv2v_20260903_200923/events.out.tfevents.1788437369.admin1`
- PID 文件：`/data/cjm/datasets/logs/attfuse_opv2v_20260903_200923/train.pid`
- 训练保存了以下 checkpoint：`net_epoch1.pth`、`net_epoch3.pth`、`net_epoch5.pth`、`net_epoch7.pth`、`net_epoch9.pth`、`net_epoch11.pth`、`net_epoch13.pth`、`net_epoch15.pth`。
- 最佳模型的选择严格基于官方 OPV2V Clean 验证集的最低 validation loss，不使用 OPV2V-W 测试结果挑选权重，以避免测试集泄漏。
- **最终选中的最佳 checkpoint：`net_epoch13.pth`**。
- **最佳权重完整路径：`/data/cjm/datasets/logs/attfuse_opv2v_20260903_200923/net_epoch13.pth`**。
- 推理日志已确认：`resuming by loading epoch 13`。

### 5.4 测试方法

- 推理入口：`/home/cjm/OpenCOOD-main/opencood/tools/inference.py`
- 推理模式：`--fusion_method intermediate`
- 推理设备：HCU 0，通过 `ROCR_VISIBLE_DEVICES=0` 隔离。
- 未启用 `--global_sort_detections`，因此采用经典 OPV2V 论文/官方基准的非全局检测置信度排序方式。
- 指标为 3D 检测 Average Precision：AP@IoU 0.3、AP@IoU 0.5 和 AP@IoU 0.7。
- 三种 OPV2V-W 天气测试使用同一个由 Clean 验证集选出的 `net_epoch13.pth`，不存在针对不同天气单独挑选权重的情况。
- 测试时为每个数据集建立独立模型目录，复制配置和最佳权重，再将该目录中 `config.yaml` 的 `validate_dir` 指向相应测试集。

### 5.5 OPV2V-W 恶劣天气测试结果

| 测试条件 | 测试集路径 | AP@0.3 | AP@0.5 | AP@0.7 |
|---|---|---:|---:|---:|
| Fog | `/data/cjm/datasets/opv2v-w/fog/test` | 0.7012558589321348 | 0.6885083870783378 | 0.6103433629425823 |
| Rain | `/data/cjm/datasets/opv2v-w/rain/test` | 0.6812502845321651 | 0.6588940304245862 | 0.5619786781954386 |
| Snow | `/data/cjm/datasets/opv2v-w/snow/test` | 0.5802907290801040 | 0.5577118375943968 | 0.4683919287372450 |

- 恶劣天气结果根目录：`/data/cjm/datasets/logs/attfuse_opv2v_20260903_200923/opv2v_w_best_epoch13`
- Fog 结果：`/data/cjm/datasets/logs/attfuse_opv2v_20260903_200923/opv2v_w_best_epoch13/fog/eval.yaml`
- Rain 结果：`/data/cjm/datasets/logs/attfuse_opv2v_20260903_200923/opv2v_w_best_epoch13/rain/eval.yaml`
- Snow 结果：`/data/cjm/datasets/logs/attfuse_opv2v_20260903_200923/opv2v_w_best_epoch13/snow/eval.yaml`
- 各天气目录下的 `inference.log` 保存对应完整推理日志。

### 5.6 OPV2V Clean 测试结果

- 测试集：`/data/scd/datasets/opv2v_official_data_dumping/test`
- 测试样本数：`2170`
- 加载权重：`net_epoch13.pth`
- 推理耗时：约 `3 分 59 秒`
- 平均推理吞吐：约 `9.06 samples/s`
- AP@IoU 0.3：`0.91`（推理日志显示精度）
- AP@IoU 0.5：`0.90`（推理日志显示精度）
- AP@IoU 0.7：`0.82`（推理日志显示精度）
- Clean 测试结果目录：`/data/cjm/datasets/logs/attfuse_opv2v_20260903_200923/opv2v_clean_best_epoch13`
- Clean 推理日志：`/data/cjm/datasets/logs/attfuse_opv2v_20260903_200923/opv2v_clean_best_epoch13/inference.log`
- Clean 完整精度结果：`/data/cjm/datasets/logs/attfuse_opv2v_20260903_200923/opv2v_clean_best_epoch13/eval.yaml`
- 当前记录的 Clean AP 来自终端日志的两位小数显示；若论文或表格需要更高精度，应从上述 `eval.yaml` 读取 `ap30`、`ap_50`、`ap_70` 原始值。

### 5.7 已观察到的运行信息

- 数据集构建阶段出现过 `too many cavs`，表示部分样本中的车辆数超过配置的 `max_cav: 5`，数据加载器会限制参与协作的车辆数量；训练仍正常完成。
- 推理阶段 Shapely 偶尔报告 `RuntimeWarning: invalid value encountered in intersection`，但 Clean 测试完整处理了 2170 个样本并正常生成 AP 结果。
- `torch.load` 报告了关于未来默认启用 `weights_only=True` 的 FutureWarning。本次权重为自己训练并受信任的权重，因此没有影响本次实验结果。

## 6. 当前研究方向：GSPR-AttFuse（2026-09-04）

### 6.1 研究目标与范围

- 总体研究方向：恶劣天气条件下协同感知的鲁棒性。
- 当前集中研究整个架构的第一阶段，即 LiDAR 点云进入 PointPillar/BEV 编码前的点级可靠性建模。
- 当前方法暂定名：**Geometry-first Sensor-adaptive Pillar Reliability Network（GSPR）**。
- 当前版本只研究单车局部点云的可靠性估计及其与 AttFuse 的连接，**暂不加入时序分支，也暂不加入协同通信或融合可靠性分支**。
- 该阶段不把任务限定为传统的硬去噪，而是输出连续点级可靠性和不确定性，为后续通信选择和质量感知融合保留信息。

### 6.2 研究动机与拟解决的问题

当前论文叙述与核心假设如下：

1. 现有点云去噪方法通常较早地混合几何信息和传感器相关的强度信息；不同 LiDAR 的强度标定、线束和点密度发生变化时，这种耦合可能导致跨传感器泛化不稳定。
2. GSPR 使用几何优先、传感器自适应的晚期证据融合：几何分支负责相对稳定的空间结构，辐射分支单独处理 intensity/ring/sensor statistics，再在决策层融合。
3. TripleMixer 的三平面频域建模需要单独的 KNN 和 3D→2D 投影。GSPR 将其改造成与 PointPillar XY 网格对齐的轻量低频/高频模块，减少重复预处理和密集三平面开销。
4. 模型输出经过证据建模的软 reliability 和 uncertainty，而不是直接不可逆地删除点。
5. reliability/uncertainty 将作为统一接口，在后续研究阶段扩展到通信选择、CAV 质量估计和质量感知融合。

这一路线不是简单地把 TripleMixer 原封不动接到 AttFuse 前面，而是针对协同感知数据流、跨 LiDAR 稳定性和 PointPillar 计算结构进行重新设计。

### 6.3 当前网络结构

当前 GSPR 的结构为：

```text
单车原始 XYZI 点云
        ↓
SpVoxelPreprocessor 仅进行点槽打包（坐标与点属性仍保留）
        ├── 几何分支
        │     xyz / range / elevation(ring proxy) / pillar density
        │     cluster offset / pillar-center offset
        │     + PointPillar XY 网格对齐的轻量 Haar 低频/高频上下文
        │
        ├── 辐射分支
        │     robust normalized intensity / bounded intensity
        │     elevation 或真实 ring / density
        │     + 逐 CAV sensor statistics FiLM 调制
        │
        └── 物理先验分支
              range / robust intensity / density
                         ↓
                 两类非负证据融合
                         ↓
           point reliability + point uncertainty
                         ↓
            Reliability-aware PillarVFE 软加权
                         ↓
              BEV backbone + 原始 AttFuse
```

证据融合把每个点建模为“可靠/噪声”两类 Dirichlet evidence，并由 evidence 同时计算 reliability 和 uncertainty。当前还会输出 pillar 级均值接口，但 AttFuse v1 不利用该接口进行通信或协同融合。

### 6.4 TripleMixer 与 AttFuse 对接问题及当前处理

此前识别出三个主要接口问题：

1. TripleMixer 输入是稀疏原始点云 `B×C×N`，AttFuse 神经网络部分接收体素化数据并进一步生成 `B×C×H×W` BEV。
2. TripleMixer 依赖 KDTree KNN 和单独的三平面量化投影；完成 OpenCOOD 体素化后，直接重建这些关系比较困难且会重复预处理。
3. 如果硬删除点，点数和索引会改变，已有 `cell_ind`、`occupied_cell` 或 OpenCOOD 的 voxel slot 对齐关系需要重新计算。

当前实现选择的边界是：**spconv 完成 voxel slot 打包之后、PillarVFE 特征编码和 BEV 生成之前**。此时 `voxel_features[M,T,4]` 中仍保存逐点 XYZI，因此 GSPR 仍然执行逐点预测；同时可以直接复用 `voxel_coords`、`voxel_num_points` 和 PointPillar 网格。

没有把可训练网络直接放进 `SpVoxelPreprocessor` 之前，是因为 OpenCOOD 的该预处理在 CPU DataLoader/NumPy/spconv 流程中执行，把网络放在那里会切断端到端梯度。

当前主路径也没有采用把噪点坐标改成 `(9999,9999,9999)` 的办法，原因是：

- 哨兵点可能污染 KNN、range、密度和传感器统计；
- 它会把软可靠性重新变成不可逆的硬过滤；
- 训练阶段不能通过 spconv 的 CPU 量化操作反向传播；
- GSPR 保持点槽、坐标和点数不变即可天然避免索引失效。

`9999` 哨兵或基于阈值的硬删除以后只可作为**推理阶段消融实验**，并且必须在可靠性校准完成后进行，不能作为当前主方法。

### 6.5 当前代码完成情况

已新增以下独立实现，没有覆盖原始 AttFuse 基线文件，便于后续进行严格 A/B 对照：

- `attfuse_gspr/reliability.py`：几何分支、pillar 对齐频率模块、辐射分支、传感器调制、物理先验和证据融合。
- `attfuse_gspr/pillar_vfe.py`：Reliability-aware PillarVFE，在 PFN pooling 前对点特征进行软加权。
- `attfuse_gspr/point_pillar_gspr_attfuse.py`：GSPR 与经典 AttFuse 的完整连接模型。
- `attfuse_gspr/loss.py`：原始检测损失、可选逐点监督损失、keep-rate 防塌缩项和 evidence 正则项。
- `gspr_attfuse_config.yaml`：OPV2V 训练配置和 GSPR 超参数。
- `train_gspr_attfuse.py`：训练入口，支持从指定的 AttFuse checkpoint 精确热启动，并强制训练输出位于 `/data/cjm/datasets`。
- `inference_gspr_attfuse.py`：注册本地 GSPR 模型后调用 OpenCOOD 标准推理流程。
- `verify_gspr.py`：服务器端前向、反向和点槽对齐 smoke test。
- `attfuse_gspr/README.md`：服务器部署、检查、训练和评测说明。

实现保留以下统一输出接口：

- `point_reliability`
- `point_uncertainty`
- `point_evidence`
- `pillar_reliability`
- `pillar_uncertainty`

OPV2V 当前点云为四维 XYZI，不提供真实 ring。当前配置使用 elevation 作为 ring proxy；如果以后数据加载器提供第五维 ring，可设置 `ring_index: 4`。

### 6.6 当前验证状态（必须与实验结果区分）

- 新增 Python 文件已经在 Windows 本地通过 `py_compile` 语法检查。
- `gspr_attfuse_config.yaml` 已通过 YAML 解析和关键字段检查。
- 本地 Windows Python 没有安装 PyTorch，因此尚未完成真实 tensor 前向/反向运行。
- `verify_gspr.py` 需要上传到服务器后，在 `opencood` Conda环境执行。
- **截至本记录写入时，GSPR 尚未在服务器正式训练，也没有产生 Clean 或 OPV2V-W AP；不能把网络设计或静态检查描述为实验性能结果。**

服务器 smoke test：

```bash
cd /home/cjm/OpenCOOD-main/cjmnet

ROCR_VISIBLE_DEVICES=0 \
PYTHONPATH=/home/cjm/OpenCOOD-main:/home/cjm/OpenCOOD-main/cjmnet \
/home/cjm/miniconda3/envs/opencood/bin/python verify_gspr.py
```

### 6.7 当前训练入口与热启动策略

GSPR-AttFuse 计划从已经由 Clean validation loss 选出的经典 AttFuse 最佳权重初始化：

```text
/data/cjm/datasets/logs/attfuse_opv2v_20260903_200923/net_epoch13.pth
```

训练入口会按 checkpoint 文件精确加载 `net_epoch13.pth` 中名称和形状匹配的 PillarVFE、BEV backbone、AttFuse 和检测头参数；新增加的 GSPR 参数单独初始化。GSPR 初始 reliability 接近 0.95，使热启动初期尽量接近原 AttFuse，而不是随机大幅抑制点云。

当前 `gspr_attfuse_config.yaml` 使用 Clean OPV2V train/validate。它首先用于确认结构接通、显存、速度和检测训练是否正常。仅使用 Clean 检测损失微调不能证明网络学会识别天气噪声。

训练输出目录规范：

```text
/data/cjm/datasets/logs/gspr_attfuse_opv2v_<timestamp>
```

任何生成的恶劣天气训练数据、逐点标签、模型权重和大型日志都必须放在 `/data/cjm/datasets` 下，严禁写入 `/home/cjm/OpenCOOD-main` 或 `/home/cjm/OpenCOOD-main/cjmnet`。

### 6.8 逐点监督数据的关键缺口

当前 OpenCOOD 内已有的天气增强流程只返回增强点云和总体统计，没有把每个增强点的来源/噪声类别传递到 voxel slot，因此当前训练只能使用检测损失和防塌缩正则，不能冒充逐点有监督去噪训练。

TripleMixer 下载代码中的 fog/rain/snow 模拟器能够生成或保留相应的增强标签，是下一阶段优先适配的代码来源。暂时不需要额外下载其他论文源码。LiSnowNet、3D-OutDet 等代码以后可以作为独立的轻量去噪基线，而不是 GSPR v1 的强制依赖。

计划把模拟器输出统一为至少以下信息：

```text
weather_points:     [N, 4]，增强后的 XYZI
reliability_target: [N]，连续或二值可靠性目标
noise_type:         [N]，clear / attenuation / fog / rain / snow / other
source_index:       [N]，能够对应原始点时记录原始点索引，否则为 -1
weather_metadata:   天气类型、强度、随机种子和模拟器参数
```

随后必须在体素化时让 `reliability_target` 与 `voxel_features[M,T]` 使用完全相同的排序和截断规则，最终传入模型已经预留的 `point_reliability_target`。不能仅按坐标近邻事后猜测标签，因为模拟天气可能改变距离、强度、点数和点序。

### 6.9 计划实验方案

实验必须按以下阶段推进，避免直接进行耗时的完整训练：

#### 阶段 A：接口和数值稳定性检查

1. 在服务器运行 `verify_gspr.py`，确认 forward/backward、有效点 mask、reliability/uncertainty 范围和梯度正常。
2. 用 1 个真实 OPV2V batch 做前向检查，记录 voxel 数、有效点数、reliability 均值/分位数、uncertainty 均值和峰值显存。
3. 从 AttFuse `net_epoch13.pth` 热启动，确认除新 GSPR 参数外不存在意外 missing/unexpected keys。
4. 在 GSPR 近透明初始化下检查检测输出与原始 AttFuse 的偏差，防止接口接通就破坏基线。

#### 阶段 B：小样本短时间筛选

建立固定的小型 development subset，只从 OPV2V train/validate 中抽取，不使用 OPV2V-W test 选模型。建议固定：

- 训练：4–8 个场景；
- 验证：2 个固定场景；
- 天气：fog/rain/snow 均衡采样多个强度；
- 训练预算：先跑约 1,000–3,000 optimizer steps 或 2–3 个短 epoch；
- 所有方案使用相同样本、随机种子、batch size、步数和初始化权重；
- 至少运行 3 个随机种子后再淘汰方案。

快速筛选不能只看总 loss，需要同时比较：

- 点级 AUROC、AUPRC、F1；
- reliability 的 ECE/Brier score；
- Clean 点保留率与天气噪声召回率；
- Clean 与合成天气 detection AP 或 validation loss；
- 参数量、单帧延迟、峰值显存；
- 是否出现 reliability 全 1、全 0 或 uncertainty 塌缩。

只有小样本结果稳定优于对照，才进入完整训练。小样本筛选只能用于方法开发，最终结论必须由完整训练和冻结测试集给出。

#### 阶段 C：逐点监督预训练

1. 用 TripleMixer 提供的天气模拟代码适配 OPV2V train/validate，生成带 provenance 的 fog/rain/snow 点云或按需在线生成。
2. 所有生成数据写入 `/data/cjm/datasets`，建议根目录：

   ```text
   /data/cjm/datasets/gspr_opv2v_weather_train
   ```

3. 先单独训练 GSPR 的 point reliability/uncertainty，避免检测损失掩盖点级学习是否有效。
4. 使用 weather type、severity 和 sensor-style 随机化，检验几何分支是否比强度分支更稳定。
5. 在 Clean validation 与合成天气 validation 上选择 checkpoint，不访问 OPV2V-W test AP。

#### 阶段 D：与 AttFuse 联合微调

建议采用逐步解冻：

1. 冻结 AttFuse，只训练 GSPR 和 Reliability-aware PillarVFE；
2. 解冻 PillarVFE 与 BEV backbone 的后部，用较小学习率联合微调；
3. 最后视验证结果决定是否解冻完整 AttFuse；
4. 保留原始检测损失，并加入点级监督、可靠性校准和适度 evidence 正则；
5. 对 Clean/Weather 混合 batch 训练，防止只提升恶劣天气而明显损伤 Clean 性能。

#### 阶段 E：冻结测试与正式对比

权重选择完成后一次性评测：

- OPV2V Clean test；
- OPV2V-W fog test；
- OPV2V-W rain test；
- OPV2V-W snow test。

主要对照至少包括：

1. 原始 AttFuse；
2. AttFuse + 原版 TripleMixer 或可复现的 TripleMixer 前处理；
3. AttFuse + GSPR；
4. GSPR 的硬过滤版本（仅作为消融）；
5. GSPR 去掉 uncertainty、sensor adapter、物理先验或频率模块的消融版本。

所有方法使用相同 AttFuse 初始化、训练数据、训练预算和 checkpoint 选择规则。OPV2V-W test 只能做最终报告，不能用于挑权重或调超参数。

### 6.10 计划消融与创新有效性验证

为证明各创新点不是简单堆叠，计划执行：

| 实验 | 几何/辐射解耦 | Pillar 频率模块 | Sensor FiLM | 物理先验 | Evidential uncertainty | 软加权 |
|---|---:|---:|---:|---:|---:|---:|
| AttFuse baseline | 否 | 否 | 否 | 否 | 否 | 否 |
| TripleMixer 接入 | 否 | 三平面 | 否 | 否 | 否/依原实现 | 视实现 |
| GSPR-Geometry | 是 | 否 | 否 | 否 | 是 | 是 |
| GSPR-Geometry+Freq | 是 | 是 | 否 | 否 | 是 | 是 |
| GSPR-NoPhysics | 是 | 是 | 是 | 否 | 是 | 是 |
| GSPR-NoUncertainty | 是 | 是 | 是 | 是 | 否 | 是 |
| GSPR-Hard | 是 | 是 | 是 | 是 | 是 | 否 |
| GSPR-Full | 是 | 是 | 是 | 是 | 是 | 是 |

需要重点回答的研究问题：

- 几何/辐射晚期融合是否提升跨 LiDAR 或 intensity shift 下的稳定性；
- Pillar 对齐频率模块是否以更低时间/显存成本保留 TripleMixer 频域建模收益；
- uncertainty 是否有良好校准，而不只是与 reliability 重复；
- 软加权是否比硬删除更好地维持 Clean AP 和远距离弱目标；
- 物理先验是否在未见过的天气强度上带来稳定增益；
- GSPR 的增益是否来自点级可靠性本身，而不是额外参数或更长训练。

### 6.11 后续扩展边界

只有完成上述单车前端验证后，才进入后续两步：

1. 使用 pillar/point reliability 指导通信区域或特征选择；
2. 使用 uncertainty 和 CAV 质量估计指导 AttFuse 的质量感知协同融合。

时序恢复、跨车一致性和通信分支都不属于当前 GSPR v1，现阶段不要提前加入，以免无法判断第一创新点本身是否有效。

## 7. TripleMixer 逐点监督 GSPR：开发实验完成（2026-09-06）

### 7.1 已生成数据

- Smoke 数据：`/data/cjm/datasets/gspr_opv2v_weather_smoke`
  - Train：1 个原始 PCD，Fog/Rain/Snow moderate 共 3 个 NPZ；
  - Val：1 个原始 PCD，Fog/Rain/Snow moderate 共 3 个 NPZ。
- Development 数据：`/data/cjm/datasets/gspr_opv2v_weather_development`
  - Train：24 个原始 PCD，72 个逐点监督样本；
  - Val：12 个原始 PCD，36 个逐点监督样本；
  - 当前只包含 moderate 强度。
- TripleMixer Fog、Rain、Snow 接口、逐点标签长度、来源索引和有限数值预检均已通过。

### 7.2 GSPR 逐点监督预训练

- 运行目录：`/data/cjm/datasets/logs/gspr_point_pretrain_20260905_120258`
- 最佳权重：`/data/cjm/datasets/logs/gspr_point_pretrain_20260905_120258/gspr_best.pth`
- 最佳开发验证结果：
  - AUROC：0.994094；
  - AUPRC：0.999798；
  - 可靠点均值：0.864473；
  - 噪声点均值：0.552586；
  - 最佳平衡阈值：0.588589；
  - 最佳阈值可靠点召回：0.968239；
  - 最佳阈值噪声召回：0.953375。
- `noise_recall_at_0.5=0` 是固定阈值校准偏移，不是无判别能力。当前主路径使用连续软权重，不采用 0.5 硬删除。
- Development Val 噪声点分布严重不平衡：Fog 35117、Rain 27、Snow 4920。Rain 的近满分逐点指标不能作为可靠结论；正式数据必须扩大天气强度和独立噪声点数量。

### 7.3 GSPR 与 AttFuse 联合开发微调

- 运行目录：`/data/cjm/datasets/logs/gspr_point_joint_20260905_193746`
- 使用 Clean OPV2V 检测 batch 与 TripleMixer 逐点监督天气 batch 联合优化。
- 三个 epoch 中，epoch2 的 Clean validation detection loss 最低：0.259608；其点级 AUROC 为 0.996829，最佳阈值可靠点召回为 0.978671，噪声召回为 0.981230。
- 按事先规定的“主任务验证损失优先、点级能力不得退化”规则冻结：

  ```text
  /data/cjm/datasets/logs/gspr_point_joint_20260905_193746/net_epoch2.pth
  ```

### 7.4 冻结 epoch2 的四条件结果

评测目录：

```text
/data/cjm/datasets/logs/gspr_point_joint_20260905_193746/frozen_epoch2_all_weather
```

| 条件 | AP@0.3 | AP@0.5 | AP@0.7 |
|---|---:|---:|---:|
| Clean | 0.9218719674 | 0.9131681576 | 0.8327910755 |
| Fog | 0.7177120438 | 0.7067931916 | 0.6339269293 |
| Rain | 0.7384136184 | 0.7199802952 | 0.6154857005 |
| Snow | 0.6262144193 | 0.6096065764 | 0.5035343244 |

相对 `GSPR-AttFuse-Clean-DetectionOnly` epoch11：

| 条件 | AP@0.3 变化（百分点） | AP@0.5 变化（百分点） | AP@0.7 变化（百分点） |
|---|---:|---:|---:|
| Clean | +0.150 | +0.096 | -0.225 |
| Fog | +2.040 | +2.077 | +1.915 |
| Rain | +2.231 | +2.013 | +1.764 |
| Snow | +3.406 | +3.493 | +2.740 |

- 三种恶劣天气平均 AP@0.7：0.584316；
- Detection-only GSPR 的三天气平均 AP@0.7：0.562919；
- 平均提升：+2.140 个百分点；
- 相对原始 AttFuse 的三天气平均 AP@0.7 提升约 +3.741 个百分点。

开发实验满足进入扩大训练的门槛：Clean AP@0.7 降幅小于 0.5 个百分点，三种天气均提升，且恶劣天气平均 AP@0.7 提升超过 1 个百分点。

### 7.5 完整逐点语料与正式训练记录（当前多种子训练状态见第8节）

- 完整语料根目录：`/data/cjm/datasets/gspr_opv2v_weather_full_v1`。
- Train：516 个唯一源 PCD，4644 个 NPZ；Val：108 个唯一源 PCD，972 个 NPZ；每个源点云均包含 3 种天气 × 3 种强度；manifest 无重复路径，全部 NPZ 校验通过。
- Train Rain 噪声统计：light 138 点且 78.88% 样本零噪声；moderate 1291 点且 18.60% 样本零噪声；heavy 5820 点且 2.91% 样本零噪声。
- Val Rain 噪声统计：light 31 点且 75.00% 样本零噪声；moderate 304 点且 9.26% 样本零噪声；heavy 1344 点且无零噪声样本。
- Fog/Snow 均有稳定噪声监督；Fog heavy 噪声比例约 34%，明显高于其他组合。普通 shuffle 会造成训练梯度偏向 Fog，不能直接使用。

正式点级预训练采用可复现的分层采样：9 个 weather/severity 组合等概率；每组含噪样本占比至少 80%、零噪声样本占比至多 20%，若自然零噪声比例低于 20% 则保持自然比例，避免反向过采样 Fog-light 等极少数零噪声帧；验证集保持原始自然分布。checkpoint 使用 9 个组合的验证宏平均 AUROC 选择，而不是受可靠点占比主导的总体 AUPRC。

后续步骤：

1. full_v1 正式 GSPR 点级预训练 seed 20260906 已完成 5 epoch。epoch5 同时取得最低 val loss 0.059142、最高 validation macro AUROC 0.999871 和最高 macro balanced accuracy 0.998569，因此 `gspr_best.pth` 对应 epoch5。
2. full_v1 正式联合微调 seed 20260906 已完成。epoch1 冻结检测 backbone/head，epoch2–5 全解冻。五个 epoch 的 Clean validation loss 分别为 0.275915、0.274252、0.273731、0.271947、0.273817；epoch4 最低，且其 point macro AUROC 为 0.999861、macro balanced accuracy 为 0.998496，因此正式 seed1 冻结候选是 epoch4/`net_best_validation.pth`。
3. 联合训练的天气点 batch 使用相同的九组分层采样；权重按最低 Clean validation detection loss 选择，同时要求 point validation macro AUROC 不低于 0.995。
4. 下一步顺序执行另外两个独立训练种子。后续脚本补齐 Python、NumPy、CPU/CUDA PyTorch 的统一随机种子；当前 seed1 只显式设置了 PyTorch 与数据集种子，因此可作为完整 pilot replicate，严格三种子汇报时应使用统一种子代码重新运行三个正式种子或明确披露差异。
5. 不得为了提高 Rain 指标任意修改 TripleMixer 物理参数；当前优先使用已发布强度、扩大帧数和分层采样。
6. 当前 OPV2V-W 已在开发阶段多次查看，应在论文记录中视作 pilot benchmark，而不能再宣称为完全未触碰的测试集。正式结论需要冻结协议，并增加未用于开发的真实数据集、自建保留集或其他外部数据集测试。

### 7.6 full_v1 联合训练稳定性与冻结接口（2026-09-08）

三次完整联合运行的验证结果：

| 运行 | 最佳 epoch | Clean det val loss | Point macro AUROC | Macro balanced accuracy | 冻结权重 |
|---|---:|---:|---:|---:|---|
| 20260906 pilot | 4 | 0.271947 | 0.999861 | 0.998496 | `/data/cjm/datasets/logs/gspr_joint_full_v1_seed20260906_*/net_best_validation.pth` |
| 20260907 | 3 | 0.271935 | 0.999783 | 0.998331 | `/data/cjm/datasets/logs/gspr_joint_full_v1_seed20260907_20260907_161155/net_best_validation.pth` |
| 20260908 | 4 | 0.272649 | 0.999855 | 0.998485 | `/data/cjm/datasets/logs/gspr_joint_full_v1_seed20260908_20260907_214827/net_best_validation.pth` |

三次运行统计：

- Clean detection validation loss：`0.272177 ± 0.000409`（样本标准差）；
- Point macro AUROC：`0.999833 ± 0.000043`；
- Macro balanced accuracy：`0.998437 ± 0.000092`；
- 最佳 epoch 为 4/3/4，没有出现训练崩溃或点级能力遗忘。

因此 GSPR-v1 已满足作为后续通信创新可靠性提供器的稳定性要求。自此冻结以下语义接口：

```text
point_reliability
point_uncertainty
point_evidence
pillar_reliability
pillar_uncertainty
```

后续通信实验应新增独立的 `gspr_communication/` 与新模型/训练入口，不覆盖当前 `attfuse_gspr/`、`gspr_supervision/` 或 full_v1 训练入口。通信第一版使用冻结的 GSPR-v1 权重，先证明通信策略的独立增益；允许通信损失反向修改 GSPR 的版本只能作为后续联合优化消融。

### 7.7 full_v1 三随机种子最终检测结果（2026-09-08）

为了避免混淆，先列原始 AttFuse，再列加入 GSPR 后的结果。原始 AttFuse 使用仅在 Clean OPV2V 上训练、由 Clean validation loss 选出的 `net_epoch13.pth`；GSPR 使用 full_v1 联合训练的三个独立运行均值。

#### A. 原始 AttFuse（未加入 GSPR）

权重：`/data/cjm/datasets/logs/attfuse_opv2v_20260903_200923/net_epoch13.pth`

| 条件 | AP@0.3 | AP@0.5 | AP@0.7 |
|---|---:|---:|---:|
| Clean | 0.91 | 0.90 | 0.82 |
| Fog | 0.7012558589 | 0.6885083871 | 0.6103433629 |
| Rain | 0.6812502845 | 0.6588940304 | 0.5619786782 |
| Snow | 0.5802907291 | 0.5577118376 | 0.4683919287 |

说明：原始 AttFuse 的 Clean 推理日志只保存了两位小数，因此这里不能假造更多精度；Fog、Rain、Snow 来自各自 `eval.yaml`，保留了完整精度。

#### B. AttFuse + GSPR full_v1（三随机种子均值）

| 条件 | AP@0.3（均值 ± 样本标准差） | AP@0.5（均值 ± 样本标准差） | AP@0.7（均值 ± 样本标准差） |
|---|---:|---:|---:|
| Clean | 0.920849 ± 0.002051 | 0.912204 ± 0.002133 | 0.831512 ± 0.001838 |
| Fog | 0.718529 ± 0.001066 | 0.708345 ± 0.000560 | 0.639272 ± 0.001119 |
| Rain | 0.760051 ± 0.000276 | 0.743675 ± 0.000511 | 0.645840 ± 0.000853 |
| Snow | 0.642076 ± 0.001294 | 0.622683 ± 0.002086 | 0.514817 ± 0.004604 |

相对原始 AttFuse 的恶劣天气提升：

| 条件 | AP@0.3 变化（百分点） | AP@0.5 变化（百分点） | AP@0.7 变化（百分点） |
|---|---:|---:|---:|
| Fog | +1.727 | +1.984 | +2.893 |
| Rain | +7.880 | +8.478 | +8.386 |
| Snow | +6.179 | +6.497 | +4.643 |
| 三天气宏平均 | +5.262 | +5.653 | +5.307 |

Clean 只能确认加入 GSPR 后约为 `0.921/0.912/0.832`，而原始日志为四舍五入后的 `0.91/0.90/0.82`；由于原始值精度不足，不在正式表中计算 Clean 的精确百分点差值。

#### C. AttFuse + GSPR full_v1 各随机种子明细

| Seed | 条件 | AP@0.3 | AP@0.5 | AP@0.7 |
|---:|---|---:|---:|---:|
| 20260906 | Clean | 0.922690 | 0.914023 | 0.832428 |
| 20260906 | Fog | 0.718184 | 0.707992 | 0.640453 |
| 20260906 | Rain | 0.760344 | 0.744261 | 0.645525 |
| 20260906 | Snow | 0.641564 | 0.624004 | 0.516750 |
| 20260907 | Clean | 0.921220 | 0.912733 | 0.832711 |
| 20260907 | Fog | 0.719725 | 0.708990 | 0.639138 |
| 20260907 | Rain | 0.760012 | 0.743446 | 0.646805 |
| 20260907 | Snow | 0.643548 | 0.623768 | 0.518140 |
| 20260908 | Clean | 0.918638 | 0.909856 | 0.829396 |
| 20260908 | Fog | 0.717678 | 0.708052 | 0.638227 |
| 20260908 | Rain | 0.759796 | 0.743320 | 0.645189 |
| 20260908 | Snow | 0.641117 | 0.620278 | 0.509562 |

下面另外保留相对 `GSPR-AttFuse-Clean-DetectionOnly` epoch11 的对照。这个 Detection-only 是用于隔离“GSPR逐点监督本身贡献”的中间实验，不等于上面的原始 AttFuse：

| 条件 | 指标 | 均值 | 样本标准差 | 相对 Detection-only（百分点） |
|---|---|---:|---:|---:|
| Clean | AP@0.3 | 0.920849 | 0.002051 | +0.048 |
| Clean | AP@0.5 | 0.912204 | 0.002133 | -0.001 |
| Clean | AP@0.7 | 0.831512 | 0.001838 | -0.353 |
| Fog | AP@0.3 | 0.718529 | 0.001066 | +2.122 |
| Fog | AP@0.5 | 0.708345 | 0.000560 | +2.233 |
| Fog | AP@0.7 | 0.639272 | 0.001119 | +2.449 |
| Rain | AP@0.3 | 0.760051 | 0.000276 | +4.395 |
| Rain | AP@0.5 | 0.743675 | 0.000511 | +4.382 |
| Rain | AP@0.7 | 0.645840 | 0.000853 | +4.800 |
| Snow | AP@0.3 | 0.642076 | 0.001294 | +4.992 |
| Snow | AP@0.5 | 0.622683 | 0.002086 | +4.801 |
| Snow | AP@0.7 | 0.514817 | 0.004604 | +3.868 |

三种恶劣天气的宏平均结果：

| 指标 | GSPR full_v1 | 相对 Detection-only | 相对原始 AttFuse |
|---|---:|---:|---:|
| AP@0.3 | 0.706885 | +3.836 个百分点 | +5.262 个百分点 |
| AP@0.5 | 0.691568 | +3.805 个百分点 | +5.653 个百分点 |
| AP@0.7 | 0.599976 | +3.706 个百分点 | +5.307 个百分点 |

结论与冻结决定：

- full_v1 GSPR 已通过第一阶段成功门槛：Fog、Rain、Snow 在全部三个 IoU 阈值上均取得正增益，且三个种子的波动远小于对应增益。
- 相对原始 AttFuse，最严格的 AP@0.7 下，Fog/Rain/Snow 分别提升 `+2.893/+8.386/+4.643` 个百分点，恶劣天气宏平均提升 `+5.307` 个百分点。这是后续论文叙述中最直观的基线收益。
- 相对用于隔离逐点监督贡献的 Detection-only 对照，Fog/Rain/Snow 的 AP@0.7 分别提升 `+2.449/+4.800/+3.868` 个百分点，恶劣天气宏平均提升 `+3.706` 个百分点。
- 相对 Detection-only，Clean AP@0.3 基本持平、AP@0.5 基本无变化、AP@0.7 下降 0.353 个百分点；该代价处于预先允许的小幅 Clean 退化范围。原始 AttFuse 的 Clean 结果只有两位小数，不能据此计算或声称精确的 Clean 增益。
- Snow AP@0.7 的样本标准差最高（0.004604，即 0.460 个百分点），但仍明显小于 3.868 个百分点的平均增益；结果不是由单个随机种子偶然造成。
- 允许进入通信创新阶段。默认把 GSPR-v1 当作已冻结的可靠性/不确定性提供器，不再以 OPV2V-W test 调整其结构、阈值或选择 checkpoint。
- `20260906` 为完整 pilot replicate，但其随机数初始化规范与后两个运行不完全一致。当前三运行足以支持工程决策；若将该表作为严格论文主表，应披露这一差异，或用统一随机种子代码补跑一个替代种子。

## 8. GSPR-v1冻结与下一阶段通信原型计划（2026-09-07）

### 8.1 当前状态与不可修改清单

用户确认正在进行 GSPR 正式多种子训练。此处仅记录用户提供的运行状态，未远程核验进程；不推断所有种子已完成，不新增未经提供的权重路径或指标。第7节的既有结果和种子设置说明继续保留。

以下路径相对于本地 `cjmnet/`，对应服务器 `/home/cjm/OpenCOOD-main/cjmnet/`，均为冻结范围：

```text
attfuse_gspr/reliability.py
attfuse_gspr/pillar_vfe.py
attfuse_gspr/point_pillar_gspr_attfuse.py
gspr_supervision/*                  # 整个目录及全部子目录，不只是顶层Python文件
pretrain_gspr.py
train_gspr_multitask.py
```

- 禁止为通信实验修改、覆盖、删除、重命名或批量格式化上述文件；禁止通过更改其依赖、导入路径或全局 monkey patch 间接改变正式训练行为。
- 代码冻结不等于暂停服务器训练，也不等于把正在训练的参数冻结。不得停止、重启、改种子或改写现有多种子任务的配置、数据、权重及日志。
- 在用户明确解冻之前，不因修复、接口兼容或新实验需求自行突破范围；确需修改时先说明影响并取得用户授权。
- 不修改共享 OpenCOOD 数据加载器、AttFuse backbone 或环境依赖来接入新模块，避免正式训练的后续进程加载到不同实现。采用新增局部适配器、独立类和入口；不能安全隔离时先报告阻塞。
- 本次只更新本文档，不创建通信实现、不下载源码、不启动训练、不修改文件系统只读权限。冻结是项目执行约束，不代表已经创建服务器快照或锁定Git提交。

### 8.2 研究定位与待验证假设

下一阶段目标：去噪后仍然不足的观测，如何在有限通信预算下从邻车获得有效补充。当前 GSPR-AttFuse 已经融合邻车信息，因此新通信模块不能把“第一次获得邻车信息”当成相对现有模型的新增能力。

核心候选假设：**回波可信不等于观测充分；天气条件下仅用检测置信度或点密度进行供需选择，可能错误判断需求和供给。** 先验证这种混淆是否确实导致错误，再研究质量与充分性解耦是否有额外价值。

“需求图＋邻车二次预测”是可替换的原型结构，不是已确认的新颖性。Who2com、How2comm已有请求条件下的学习匹配/筛选，CoSDH已有密度需求与前景供给；不得声称首次按需通信或首次增加发送端预测头。明确区别匹配分数、发送优先级与实际检测增益，不将未经增益监督的输出称为预计AP提升。

### 8.3 原型A/B接口与GSPR边界

- **模块A（需求端）**：仅利用自车通信前的BEV语义、GSPR质量统计和可用的观测支持/覆盖信息，生成补充需求。不得利用尚未通信的邻车完整特征构造推理输入。
- **模块B（响应端）**：利用收到的压缩需求及邻车自身特征、质量、观测支持和相对位姿，生成响应优先级，在预算内选择BEV块。邻车不能访问请求中未传输的自车内部特征。
- A/B接口分开便于替换与分析，不要求独立训练，不预设最后必须形成两个创新点。第一版保持单帧、LiDAR、相同GSPR与融合器，不同时加入时序、雷达、扩散、码本和复杂调度。
- GSPR现有点级/pillar级可靠性与不确定性只描述已有有效点的输出；不等同于覆盖全部BEV的需求图。空pillar需要显式有效性/未知状态，不能把填充值0解释为可信空闲。
- 可信回波支持量是拟新增的外部统计，例如对区域内原始可靠概率求和，不是GSPR现成输出，也不要求另训练网络。应区分软权重下限映射与原始概率，并注明体素最多保留32点带来的截断；不能将保留点统计称为完整原始密度。
- 覆盖/射线统计须使用正确的传感器原点与坐标系；BEV空格不等于自由空间，一条3D射线穿过也不证明整个pillar为空。缺少足够证据时保留未知。
- `D_missing`/`D_verify`仅为后续候选，不强制在第一版同时实现。验证响应若引入，应区分支持存在、可信负证据和无法判断；邻车未检测到不能直接作为删除自车目标的依据。若两个需求无可区分监督或响应行为，优先保留单需求头。

### 8.4 工程隔离计划（尚未实施）

拟在冻结目录之外新增独立通信实验包，例如 `communication_experiments/`，自带适配器、A/B模块、配置、训练和评测入口；名称可在实施前调整。通过组合/局部派生复用冻结代码，保持v1输出语义与权重键不变，不向冻结文件内补接口。

实施前检查：

1. 记录冻结源码校验和及正式训练采用的版本；服务器版本须另行核验，不能假定与本地一致。实验修改后复核冻结源码未变。
2. 选用已经完成且固定的checkpoint，记录来源与哈希，不追读正在写入的 `best/latest` 文件。探索阶段固定前端参数并使用eval模式，防止BN等运行状态漂移；所有对照使用相同前端。
3. 新配置、新结果目录与v1完全分离。所有服务器生成数据、缓存、通信实验日志及权重均放 `/data/cjm/datasets` 下，不放服务器代码目录，不覆盖已有语料/manifest/日志。
4. 不自动启动抢占当前多种子任务资源的计算；运行前确认设备空闲及资源安排，继续遵守只用 `ROCR_VISIBLE_DEVICES` 的隔离规范。
5. 外部代码作为参考保存在独立 `third_party/`，不覆盖现有OpenCOOD；优先CoSDH、How2comm、Where2comm，CodeFilling可选。保留LICENSE、配置和版本信息，下载/复现状态待核实；作者参数只是起点，不保证迁移后最优。

### 8.5 分阶段实验与决策

**阶段1：小样本、假设驱动的原型。** 不以完整复现三个外部模型为前置条件。先检查同步多车数据、位姿、检测标注和局部观测统计，建立需求/响应基线接口；现有单车逐点NPZ语料不自动等于完整同步多车通信数据。按场景隔离train/validation，可缓存固定前端特征以降低试验成本。

**阶段2：四组因子实验。** 固定GSPR checkpoint、融合器、数据划分、训练预算和通信总字节，分别替换A/B：

| 实验 | 需求端A | 响应端B | 要回答的问题 |
|---|---|---|---|
| A0B0 | 现有需求规则 | 现有响应规则 | 共同起点 |
| A1B0 | 候选需求模型 | 现有响应规则 | 需求改进的贡献 |
| A0B1 | 现有需求规则 | 候选响应模型 | 响应改进的贡献 |
| A1B1 | 候选需求模型 | 候选响应模型 | 二者交互与联合贡献 |

同时保留自车无通信、全量通信及简单GSPR质量加权供需对照。A/B单独增益低不代表无效，需检查旧组件是否限制新组件；冻结融合器若不适应稀疏输入，各组须给予相同适配机会并明确记录。

**阶段3：按失败机制决定是否继续，而不是效益低就自动加模块。**

- 请求区域正确但邻车也看不到：属于无可用补充，不强行复杂化A。
- 将可信空背景当缺失区域：针对空闲/未知区分研究A。
- 邻车噪声或无效响应被选中：针对质量与可回应性研究B。
- 已选有效信息但融合无改善：检查对齐/融合，不能直接归因于B。
- 简单规则与预测头相当：考虑删除预测头；A/B联合才有效则研究交互。
- 若假设未获支持，停止该分支；不把更复杂结构本身当创新。

**阶段4：扩大验证。** 只有小样本支持具体机制且验证集增益稳定后，才扩大数据、训练预算、多种子及外部真实数据评测。届时仍不解冻或修改GSPR-v1源码；任何新联合优化方案使用独立入口、模型实例与输出目录，并另行确定训练协议。

### 8.6 监督、评测与结论限制

- 默认通信原型不依赖oracle消息增益标签、不以反事实损失差或教师学生技巧作为主要创新。可采用检测任务监督与预算约束；离散选择的训练近似与推理行为需检查一致性。
- 不要求真实数据具有clean/weather配对；同步多车观测、标定和覆盖适当的检测标注可支持任务训练。天气逐点变化监督、噪声标签监督与通信检测监督是不同要求。未标注区域不能当背景，无标签数据也不能宣称能直接使用监督检测loss。
- 记录clean/fog/rain/snow AP、观测不足目标召回、通信纠正的漏检与新增误检、充分观测区域退化、请求/索引/质量摘要/BEV载荷合计字节和端到端延迟。不能只报告mask比例或总AP。
- 利用诊断解释质量高但支持不足、噪声高密度、邻车共同盲区等失败类别；分析用标注不得进入推理选择器。
- OPV2V-W test已用于pilot，不继续用其挑阈值或选结构。结构/预算/权重选择依赖独立validation；最终结论需要冻结协议和额外独立测试。
- 本节所有通信结构与收益均为待验证计划，没有新增已完成实验或AP提升结论。

## 9. 2026-09-09：协同复核 GSPR 独立原型（尚未服务器运行）

- 用户不希望继续仅调整 A0B1/残差排序，授权先实现明显不同的通信机制：邻车返回存疑区域的观测证据，帮助自车修正 GSPR 点权重，再重新编码并检测。
- 新增独立 `gspr_review/`，保留 GSPR-v1 和 `gspr_communication/` 原实现。原模型参数与 BN 冻结，仅训练 1473 参数的点权重修正器；不是重新训练完整 GSPR，也不使用离散选择的软梯度代理。
- 自车按自己的 u、原始可靠概率和有效支持请求块；邻车回复按 XY pillar/4 个高度格量化的可靠、噪声、未知和支持统计。没有收到对应格可靠支持时不修正。实际消息先序列化再解码，复核与三尺度 BEV 共用 256 KiB 总预算，复核最多 12.5%。普通 BEV 仍为通信前 A0B0 规则。
- 原型采用任务检测监督和小幅修正正则。现有单车 NPZ 点标签没有被伪装成同步多车标签；权重改动不等于点分类正确率，复核的几何格也不是严格点对应证明。
- 默认 Clean/物理混合天气训练，Clean 和天气开发验证；原增强器同帧天气强度共用、各车扰动随机流独立，没有新增独立每车严重度控制。第一轮是小样本机制初筛，不能声称跨数据集泛化。
- 一键服务器入口 `bash gspr_review/run_experiment.sh`：测试→真实模型验证→训练→Clean/Weather 各 a0b0、protocol、review、shuffled 四组→汇总。protocol 使用相同证据消息和预算但关闭点修正；零初始化应匹配 protocol，而不是全预算 A0B0。shuffled 是块内证据空间错位对照，会同时改变匹配支持。
- 固定前端默认为 `/data/cjm/datasets/logs/gspr_joint_full_v1_seed20260907_20260907_161155/config.yaml` 与 `net_best_validation.pth`；支持显式环境变量覆盖。输出新时间戳目录，仅存 reviewer 权重。
- 本地已通过 7 项新旧协议测试、Python 语法检查和原 14 个冻结文件哈希检查。本机无 torch，张量/真实数据验证与训练尚未执行。不得将此记录表述为已成功训练或取得 AP 提升。
- 详细协议、命令与判读见 `gspr_review/README.md`。早先 `research/通信创新实验建议_20260909.md` 中的“换块收益排序”属于备选历史提案，当前实施方向以本节与用户最新授权为准。

### 9.1 第一轮用户回传与诊断入口

- 用户已在服务器完成训练与八组评测。第 3 epoch 为 137 次更新/139 个 batch；回传 AP 为两位小数，Clean 约 0.93/0.92/0.81、Weather 约 0.84/0.82/0.71，不能据此认定精确 AP 完全相同。
- 正常复核每帧约调整 Clean 864 点、Weather 412 点，覆盖率约 1.94%/0.98%，平均权重变化 0.00523/0.00474。protocol→review 的补回漏检数不变（Clean 267，Weather 237）；Weather A0B0 为 241。当前未显示复核检测收益，不等于链路没有实际作用。
- 用户同意下一步固定可修改点，改变邻车证据内容，并按目标附近/外部统计点权重与柱特征变化。新增 `gspr_review/diagnose.py`、`diagnostic_utils.py`、`test_diagnostics.py`、`run_diagnostics.sh`、`DIAGNOSTICS.md`；没有修改复核模型或重训。
- 诊断单次真实通信后重放 normal/permuted/constant/zero 四组，原支持掩码固定。GT 只作事后区域统计；不重新报告 AP，也不输出虚假的点级正确率。入口自动选唯一已完成复核训练目录，多个目录必须用 --run-dir 指定。输出新目录 summary.json。
- 本地 3 项 NumPy 区域/汇总测试及语法检查通过；1 项固定掩码张量测试因无 torch 跳过，服务器入口会运行。真实诊断结果尚未回传。

### 9.2 2026-09-10：源码适配后的参考差异与校正增益

- 用户随后回传诊断：Clean/Weather 可修改点 46679/22240，全部上调；平均调整约 0.005211/0.005037。固定掩码置换消息的输出差约占调整幅度 2.1%，提示内容敏感性弱，不能据此证明完全不使用通信。PFN 特征确有约 0.5% 相对 L1 变化。
- 用户下载了 `related_code/cfvqa-master`、`KalmanNet_TSP-main`、`InterSHAP-main` 并授权移植。已读取其核心源码，独立适配事实/参考评分差、增益乘差异及两来源 Shapley 交互公式；来源与具体差异见 `gspr_review/SOURCE_ADAPTATION.md`、原文件哈希见 `source_adaptation_hashes.json`。并非完整复现三篇论文。
- 新增 `counterfactual.py`：`contrast` 1472 参数、`contrast_gain` 1729 参数，旧 legacy 1473 参数。参考消息保留证据总量、u、support，把 b_r/b_n 均分；共享网络差异再乘学习增益，经过有界更新。评分零初始化，增益初始 .5。未改变冻结 GSPR/AttFuse、点支持准入、消息字段与预算，不比较不同物理点的可靠概率。
- 旧配置缺省 architecture=legacy，保留旧 checkpoint 加载。`run_counterfactual.sh` 在新时间戳目录重新训练三组，Clean/Weather 各 normal/neutral/permuted/protocol/a0b0 共 30 组评测；neutral/permuted 为保持原准入掩码与实际字节的离线内容干预。训练模型初始化后重置数据 RNG，匹配采样顺序。
- diagnose 增加评分差/增益及参考交互分析，每帧至多 2048 可修改点；不是完整数据集 InterSHAP，也不是 AP。原固定掩码/PFN 诊断仍保留。默认训练仍为 3 epoch、小场景机制初筛。
- 本地独立临时虚拟环境 CPU PyTorch 2.5.1 中 28 项新旧测试全部通过，无跳过；Python 编译、Bash 语法、原 14 个冻结文件哈希通过。真实服务器连接、训练和 AP 尚未执行。只需同步更新的 gspr_review，运行 `bash gspr_review/run_counterfactual.sh`；原论文源码和本地临时测试环境不需同步。

## 10. 2026-09-13：证据缺口匹配与实际补充收益监督

- 用户最新授权实现“GSPR 质量/u＋几何支持＋任务风险 → 压缩缺口请求 → 邻车匹配 → BEV 传输”，并明确要求正式实验不要再用极少样本。此授权更新早期 8.5/8.6 的小样本与非收益教师约定。
- 已读取 `related_code/D-Map-main`、`dynamic-selection-main`、`Learning-Loss-for-Active-Learning-master` 的核心源码；新增隔离包 `gspr_evidence/`。源码思想适配、未移植部分和哈希见包内 `SOURCE_ADAPTATION.md` 与 `source_adaptation_hashes.json`。不是三篇论文完整复现，也未修改旧实验模块。
- 新增 11,940 参数稠密风险头与 11,521 参数收益头。风险由 GT 全区域监督及同帧 clean/weather 恢复目标训练；收益由真实序列化邻车 BEV 块在固定规则上下文中等字节替换的检测损失差直接监督，保留有害补充的负标签。先训风险再冻结训收益。
- 请求含 17 个量化字段、索引和位姿，默认 256 块；元数据与三尺度特征共用 256 KiB/场景。响应仅可读解码请求、自身观测与语义。射线采样采用真实传感器原点；没有将 GSPR 可靠/噪声证据改名为占据/空闲。
- 正式入口强制完整 train/validation 场景、stride=1、不设 steps 上限，检查场景名不交叉；教师每帧每邻车至多 8 个候选，不是抽取 8 帧。缓存完整性和配置/前端/关键源码哈希受检查，支持逐帧恢复；训练在 epoch 末保存恢复点。
- `bash gspr_evidence/run_experiment.sh` 运行单元/真实连通检查、完整教师缓存、matching/concat/no_u 三组训练及 clean/fog/rain/snow 完整验证。matching 保留 none/a0b0/protocol/learned/full 对照；concat 为参数量相同的普通拼接；no_u 重新训练但保持冻结前端。正式 AP 使用 global sort，与旧 AP 直接相比前须对齐指标协议。
- 输出 `results.md`、候选收益排序/遗憾诊断、逐帧换块/纠正漏检/损失目标/字节及场景级 AP 原始统计。教师指标是抽样固定上下文下的指标，不是全候选 oracle AP 上限；最终多块组合的冗余仍需通过验证识别。
- 本地 CPU 已通过协议、几何、信息隔离、GT 漏检风险、候选重放和两阶段缓存训练/恢复测试；14 个冻结源码哈希通过。真实服务器连通、全量数据生成、GPU 训练和 AP 尚未执行，不宣称取得性能提升。完整流程、固定前端路径及恢复命令见 `gspr_evidence/README.md`。

## 11. 2026-09-14：正式测试协议——后续 AI 必须遵守

**正式报告及与历史 AttFuse/GSPR 的性能比较，必须使用下表测试集。完整 validation 也不能代替 test。**

| 条件 | 固定正式测试数据 |
|---|---|
| Clean | `/data/scd/datasets/opv2v_official_data_dumping/test` |
| Fog | `/data/cjm/datasets/opv2v-w/fog/test` |
| Rain | `/data/cjm/datasets/opv2v-w/rain/test` |
| Snow | `/data/cjm/datasets/opv2v-w/snow/test` |

- OPV2V-W 的文件已经含天气退化。正式评测必须关闭在线天气增强与随机数据增强，四种条件都读取 `processed_lidar`；不得把晴天 validation 在线生成的 fog/rain/snow 称为 OPV2V-W 正式测试。
- 与原 `evaluate_gspr_all_weather.py` 一致使用 OpenCOOD **非全局排序** AP：`global_sort_detections=False`，输出 `eval.yaml`，指标为 AP30/AP50/AP70。该实现使用平面多边形 IoU，不将其宣称为高度感知的 3D IoU。全局排序结果只能另列，不能与原结果直接相减。
- 全部场景、全部帧、stride=1、不设帧数上限。记录实际数据路径、帧数、天气来源、前端/通信 checkpoint 哈希和 AP 协议。原 Clean test 记录为 2170 帧；本地数据实际帧数由程序打印，异常差异应核查。
- 使用同一份固定前端、相同模型配置/后处理/阈值比较。默认前端为 seed20260907 的 `gspr_joint_full_v1_seed20260907_20260907_161155/net_best_validation.pth`。单种子测试首先对照 7.7C 的相同种子，不把与三种子均值的差解释成通信增益。
- train/validation 用于训练、调参、选择模型；在线天气 validation 只属于开发验证。test 仅用于固定模型的报告，不能据其结果反复挑 epoch 或阈值。必须在叙述中区分开发验证与正式测试。
- 用户本轮回传的 `gspr_evidence_seed20260913_20260913_094932/results.md` 是 **1980 帧 validation＋在线模拟天气＋global-sort AP**，不是历史 OPV2V-W 测试。其 full 高于历史值不能称为模型提升；这一批结果可用于同协议内部开发分析。
- 修正后 `gspr_evidence.evaluate` **默认 benchmark**；`--evaluation-protocol benchmark` 使用固定正式测试路径并关闭增强。开发验证必须显式传 `--evaluation-protocol development`。原 `run_experiment.sh` 仍为训练/开发流程；正式测试改用 **`bash gspr_evidence/run_benchmark.sh`**。
- 新正式入口复用现有 matching/concat/no_u 的 `gain_best.pth`，不重新训练、不生成教师标签缓存、不覆盖原运行目录。每个模型四种天气均比较 none/a0b0/protocol/learned/full，并增加 `original_full`：在同一个实际输入上调用原始 `PointPillarGsprAttfuse.forward`，逐帧检查与通信 full 的检测输出接近，同时独立报告 AP。它核对模型路径的一致性；数据和历史随机点采样不同仍可能产生小幅复现差异。
- 保持训练源码/模型/通信协议与现有 checkpoint 指纹一致；本次只增加正式评测配置、入口和汇总，未取消哈希校验。正式结果输出新目录 `${RUN}_historical_test_时间戳/results.md`。不可混合 development 与 benchmark 结果生成同一对比表。
- 本次本地完成协议与回归测试后，仍需用户在服务器运行正式测试；不得把尚未执行的正式 AP 写成已完成结果。

### 11.1 全通信质量过滤诊断（用户授权，2026-09-14）

- 用户已回传 Clean/Fog 正式测试，full 与 original_full 在两位小数上匹配；Fog 的 A0B0/learned AP70 约 .63/.53，而 full 约 .64。用户希望快速判断质量过滤空间是否有限，以决定保留 A0B0、进入下一阶段，而不是继续盲目叠加通信网络。
- 新增独立 `gspr_evidence/harm_diagnostic.py`、`run_harm_diagnostic.sh`、`test_harm_diagnostic.py`、`HARM_DIAGNOSTIC.md`。没有修改原模型、训练/正式测试入口或 checkpoint 指纹。
- **这里是用户明确授权的开发诊断，不是正式测试替代品**：固定完整 OPV2V validation，分别 Clean/在线物理 fog/rain/snow，global-sort AP；拒绝 test 路径。正式报告仍遵守第11节的 OPV2V/OPV2V-W test、无在线增强、非全局排序协议。
- 默认 STAGE=both：比较全通信与每邻车观察区域中低可靠性/高 u/随机删除 1%、5%、10%；同删除比例每车删块数严格相同。空区域没有点级质量均值，不参与排序，不能据此判定所有无点区域的 BEV 信息是否有害。
- 再逐帧每邻车从低 r、高 u、随机各取2个候选，交集去重；每次只从原全通信删除一个块，记录检测损失差与 GT 匹配/FP 改变。全验证帧都参与，候选非穷举，无 oracle AP/自动最优删块政策。损失改善不等于 AP 改善。
- 首帧验证快速硬掩码重放与实际序列化解码一致；每帧重复全通信损失估计数值噪声。记录完整协议、候选来源、四类天气与分场景 AP，供判断过滤机会/模型目标错位/信息丢失问题。
- 入口 `bash gspr_evidence/run_harm_diagnostic.sh`，等待当前正式测试完成再使用空闲卡；不训练新模型、不需要通信头权重或教师缓存。结果新建 `/data/cjm/datasets/logs/gspr_harm_validation_both_时间戳/{clean,fog,rain,snow}/diagnosis.md`。
- 若过滤未显示相对全通信和等量随机删除的稳定收益、抽样单块也很少改善实际检测，可支持停止当前冻结系统的质量过滤分支；不能外推成所有差信息都无害。真实诊断未在本地执行，不记录虚构结论。

### 11.2 正式 benchmark 完整回传结果（2026-09-14）

**本轮结论：正式测试的 full 与 original_full 一致；A0B0 已有很强的省通信效果，新学习模块在三种天气下仍明显落后于规则。Clean AP70 有小幅正差，不能写成所有条件、所有指标均下降。**

#### 结果来源和使用边界

- 来源是用户本轮贴回的完整 benchmark 表，共 72 行（3 个模型 × 4 种天气 × 6 个对照），每行 2170 帧。原始附件 SHA256：`d421f3d2ea8a06510ac322cfce7c90d024b1189f413f5ccbab06a9687203c697`。
- 对应已有训练运行 `/data/cjm/datasets/logs/gspr_evidence_seed20260913_20260913_094932`；本次附件未提供正式测试输出目录的确切时间戳及服务器 protocol.json 内的 checkpoint 哈希，不虚构这些字段。
- 按附件明确声明的 benchmark 协议：OPV2V clean test＋已生成的 OPV2V-W fog/rain/snow test，不再在线增强；非全局排序 AP。路径以第11节为准。与此前 1980 帧开发结果分开记录。
- 使用既有冻结前端与已训练通信头进行整体测试；本轮未重新训练。默认前端为 seed20260907；匹配历史结果时参照 7.7C 对应种子，权重身份的最终核验仍以运行目录中的哈希为准。
- `Replaced` 是 learned 相对该模型自身 protocol 的换块比例；本轮三个模型均运行 protocol，learned 的换块统计有效。其他对照的 0 不代表绝对没有进行通信选择。
- `Corrected / lost` 指相对同天气 none 的补回/丢失 GT 匹配次数（IoU .7），不是 AP 或误检数，不能只按补回数排序模型优劣。
- full/original_full 不受稀疏通信预算限制；original_full 的字节栏沿用 full 的逻辑载荷核算，不是额外发了一份消息。

#### A. 三个模型完全一致的公共对照

已逐项核对：none、a0b0、full、original_full 在三个模型间一致；full 与 original_full 的 AP、字节和补回/丢失数也一致。为避免重复，下表合并两种全通信行，保留原数值。

| Weather | Control | Frames | AP30 | AP50 | AP70 | Bytes/frame | Replaced | Corrected / lost |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| clean | none | 2170 | 0.755883 | 0.724591 | 0.577708 | 0.0 | 0.0000 | 0 / 0 |
| clean | a0b0 | 2170 | 0.865883 | 0.850472 | 0.751121 | 244003.7 | 0.0000 | 5537 / 158 |
| clean | full = original_full | 2170 | 0.921226 | 0.912739 | 0.832485 | 25052662.3 | 0.0000 | 6161 / 259 |
| fog | none | 2170 | 0.470151 | 0.460173 | 0.401465 | 0.0 | 0.0000 | 0 / 0 |
| fog | a0b0 | 2170 | 0.711587 | 0.698048 | 0.626587 | 244003.7 | 0.0000 | 8105 / 48 |
| fog | full = original_full | 2170 | 0.719748 | 0.709023 | 0.639663 | 25052662.3 | 0.0000 | 8015 / 110 |
| rain | none | 2170 | 0.559361 | 0.518568 | 0.406726 | 0.0 | 0.0000 | 0 / 0 |
| rain | a0b0 | 2170 | 0.745094 | 0.726788 | 0.626826 | 244003.7 | 0.0000 | 7661 / 158 |
| rain | full = original_full | 2170 | 0.759965 | 0.743751 | 0.647507 | 25052662.3 | 0.0000 | 7669 / 264 |
| snow | none | 2170 | 0.410345 | 0.378557 | 0.283999 | 0.0 | 0.0000 | 0 / 0 |
| snow | a0b0 | 2170 | 0.639357 | 0.618630 | 0.504870 | 244003.7 | 0.0000 | 7878 / 67 |
| snow | full = original_full | 2170 | 0.644169 | 0.624993 | 0.518248 | 25052662.3 | 0.0000 | 7707 / 149 |

#### B. 三个模型各自的请求与学习响应结果

matching 为显式证据匹配；concat 为相同参数量的普通拼接；no_u 为去掉通信选择器的 u 通道后重新训练，冻结 GSPR 本身仍保留。每个 protocol 都使用对应风险头生成的请求，所以 no_u 的 protocol 可以与另外两组不同。

| Variant | Weather | Control | Frames | AP30 | AP50 | AP70 | Bytes/frame | Replaced | Corrected / lost |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| matching | clean | protocol | 2170 | 0.861001 | 0.846262 | 0.751891 | 244913.4 | 0.0000 | 5348 / 178 |
| matching | clean | learned | 2170 | 0.842288 | 0.830326 | 0.753614 | 244913.4 | 0.5925 | 3787 / 209 |
| matching | fog | protocol | 2170 | 0.652706 | 0.642419 | 0.584758 | 244913.4 | 0.0000 | 6189 / 56 |
| matching | fog | learned | 2170 | 0.569076 | 0.565248 | 0.529966 | 244913.4 | 0.6782 | 3497 / 102 |
| matching | rain | protocol | 2170 | 0.695727 | 0.680275 | 0.592840 | 244913.4 | 0.0000 | 5979 / 181 |
| matching | rain | learned | 2170 | 0.629614 | 0.620866 | 0.562264 | 244913.4 | 0.6065 | 3814 / 230 |
| matching | snow | protocol | 2170 | 0.568902 | 0.552957 | 0.463232 | 244913.4 | 0.0000 | 5835 / 69 |
| matching | snow | learned | 2170 | 0.491605 | 0.483079 | 0.423336 | 244913.4 | 0.7001 | 3649 / 133 |
| concat | clean | protocol | 2170 | 0.861001 | 0.846262 | 0.751891 | 244913.4 | 0.0000 | 5348 / 178 |
| concat | clean | learned | 2170 | 0.835638 | 0.823565 | 0.747949 | 244913.4 | 0.5856 | 3521 / 200 |
| concat | fog | protocol | 2170 | 0.652706 | 0.642419 | 0.584758 | 244913.4 | 0.0000 | 6189 / 56 |
| concat | fog | learned | 2170 | 0.557111 | 0.553209 | 0.518139 | 244913.4 | 0.6718 | 3096 / 103 |
| concat | rain | protocol | 2170 | 0.695727 | 0.680275 | 0.592840 | 244913.4 | 0.0000 | 5979 / 181 |
| concat | rain | learned | 2170 | 0.623765 | 0.615198 | 0.556416 | 244913.4 | 0.5992 | 3622 / 230 |
| concat | snow | protocol | 2170 | 0.568902 | 0.552957 | 0.463232 | 244913.4 | 0.0000 | 5835 / 69 |
| concat | snow | learned | 2170 | 0.482046 | 0.473504 | 0.414656 | 244913.4 | 0.6947 | 3332 / 130 |
| no_u | clean | protocol | 2170 | 0.861583 | 0.847005 | 0.753196 | 244913.4 | 0.0000 | 5383 / 177 |
| no_u | clean | learned | 2170 | 0.833329 | 0.821260 | 0.744214 | 244913.4 | 0.6009 | 3461 / 209 |
| no_u | fog | protocol | 2170 | 0.651972 | 0.641984 | 0.583754 | 244913.4 | 0.0000 | 6158 / 58 |
| no_u | fog | learned | 2170 | 0.565454 | 0.561657 | 0.525568 | 244913.4 | 0.6867 | 3378 / 106 |
| no_u | rain | protocol | 2170 | 0.696115 | 0.680768 | 0.593711 | 244913.4 | 0.0000 | 5986 / 177 |
| no_u | rain | learned | 2170 | 0.629930 | 0.621024 | 0.561395 | 244913.4 | 0.6179 | 3763 / 231 |
| no_u | snow | protocol | 2170 | 0.570041 | 0.553864 | 0.463495 | 244913.4 | 0.0000 | 5843 / 70 |
| no_u | snow | learned | 2170 | 0.494923 | 0.485988 | 0.423982 | 244913.4 | 0.7095 | 3643 / 135 |

#### C. matching 的性能变化拆解

下表是 AP70 的百分点变化（例如 +0.2493 表示 AP 从 0.751121 变成 0.753614），不是相对百分比。

| Weather | protocol − a0b0 | learned − protocol | learned − a0b0 | a0b0 − full |
|---|---:|---:|---:|---:|
| clean | +0.0770 | +0.1723 | +0.2493 | -8.1364 |
| fog | -4.1829 | -5.4792 | -9.6621 | -1.3076 |
| rain | -3.3986 | -3.0576 | -6.4562 | -2.0681 |
| snow | -4.1638 | -3.9896 | -8.1534 | -1.3378 |

- **全通信路径已对齐。** 两条 full 的 AP 在本表六位小数上完全一致。相对 7.7C 中 seed20260907 的历史 AP70，Clean/Fog/Rain/Snow 分别为 −0.0226/+0.0525/+0.0702/+0.0108 个百分点，差异很小；原因未单独验证，不能直接归因于随机采样。此前十几个百分点的高分不能再算作模型提升。
- **A0B0 是当前更可靠的通信基线。** 平均字节 244003.7 对比 full 的 25052662.3，按表中口径减少约 99.0260%。AP70 在 Fog/Rain/Snow 比 full 低 1.3076/2.0681/1.3378 个百分点；Clean 低 8.1364 个百分点。不能概括成“所有天气几乎无损”。
- **天气下两步都损失性能。** matching 的 a0b0→protocol 同时改变请求范围和元数据/特征分配，不能把全部下降归因于风险头；protocol→learned 使用相同请求与实际总字节，更直接说明本版响应排序没有选出更有效的组合。总字节相近不等于实际 BEV 载荷相同。
- **Clean 需按指标说。** matching learned 的 AP70 比 a0b0 高 0.2493 个百分点，但 AP30/AP50 分别低 2.3595/2.0146 个百分点；单种子小差不能宣布有稳定增益。
- **显式匹配有有限积极迹象，u 贡献不稳定。** matching−concat 的 AP70 在 Clean/Fog/Rain/Snow 为 +0.5665/+1.1827/+0.5848/+0.8680 个百分点；matching−no_u 为 +0.9400/+0.4398/+0.0869/−0.0646 个百分点。Snow 中 no_u 略高，不能写成“不确定性在所有天气都有效”。这些对照差异不足以弥补天气下相对 A0B0 的下降。
- **网络确实改变了消息。** matching 的换块比例为 59.25%/67.82%/60.65%/70.01%；不能再用“网络没有发生任何选择变化”解释本轮结果。
- **保留设计问题与证据边界。** 现有收益教师是在保留大部分规则块的情况下学习单块补位，推理却重新选择大量块，训练/使用方式有偏差；它是已知设计问题，尚未被实验隔离为唯一原因。

#### D. 当前进度与下一步

- 用户已完成并回传这批正式 benchmark 表；旧章节中“正式 AP 尚未执行”是当时的历史状态，当前以本节为准。
- 用户已经授权第11.1节的全通信质量过滤诊断，并请求后台运行命令；**诊断结果尚未回传**。本次附件不是 diagnosis.md，不能当成删除实验已经完成。
- 尚未确定“差邻车块基本无害”“GSPR 质量没找到真正有害块”“删掉有效信息的代价更大”中哪个主导。等待完整开发验证的等数量随机过滤对照和独立删块结果；若没有稳定、可利用的过滤收益，优先保留 A0B0，结束当前过滤分支并转向下一阶段。
- 不根据这批 test AP 挑选删除比例、epoch 或继续调参；开发诊断与正式测试的区分继续遵守第11节。

## 12. 2026-09-15：诊断已完成，保留 A0B0 并结束当前通信创新分支

### 12.1 用户已回传的删除诊断

- 结果目录：`/data/cjm/datasets/logs/gspr_harm_validation_both_20260914_144741/`，包含 `clean/fog/rain/snow/diagnosis.md`。本轮四份结果均已回传；第11.1、11.2节中“诊断尚未回传/等待结果”的表述是历史状态，已由本节更新。
- 这是完整开发 validation 上的诊断：Clean 与在线模拟 Fog/Rain/Snow，使用 global-sort AP。不是 OPV2V-W 正式 test，不能与第11.2节非全局排序的 test AP 直接相减。
- 每帧固定编码结果，比较全通信与低可靠性、高 u、随机过滤。删除比例的分母是每个邻车有保留点的区域；三种策略在相同比例下每车删除同样数量的块。无点区域不参与点质量排序。

下表是相对同条件全通信的 **AP70 变化，单位为百分点**：

| 删除方式 | Clean | Fog | Rain | Snow |
|---|---:|---:|---:|---:|
| 高 u 前 1% | -0.0178 | +0.1034 | -0.0125 | +0.1897 |
| 高 u 前 5% | -0.8297 | -0.5822 | -0.8140 | +2.0853 |
| 高 u 前 10% | -2.4138 | -2.5761 | -2.4561 | +2.6520 |
| 随机删除 10% | -1.1304 | -1.2435 | -1.1272 | -0.8218 |

- **模拟雪天存在过滤收益。** 高 u 前10%过滤使 AP70 从 0.693438 升至 0.719959；相对全通信补回1732次目标匹配、丢失277次，净增加1455次，同时 FP 增加601。不能只报告 AP 上升而隐去误检增加，也不能外推到真实/正式 OPV2V-W Snow test。
- **统一使用该规则会伤害其他条件。** 高 u 前10%过滤在 Clean/Fog/Rain 分别损失约2.41/2.58/2.46个百分点。少量过滤的收益有限；不存在本轮已经验证的跨条件最优比例。
- **低可靠性不等于整块无用。** 低 r 过滤在 Clean/Fog/Rain 的1%、5%、10%设置下全部降低 AP70，且都差于等数量随机删除。Snow 仅10%低 r 过滤有约+0.7325个百分点收益。区域可能同时含噪声和关键目标证据，不能把平均点可靠性直接当作整块任务价值。
- 单块独立删除的去重候选数：Clean 24525、Fog 23917、Rain 24570、Snow 25430；其中检测改善/恶化次数分别为13/136、24/146、15/107、25/137。它们是候选干预次数，不是唯一目标数或帧数；“检测改善”使用固定阈值下的匹配/FP规则，不等于 AP 增益。
- 单块检查每邻车仅取低 r、高 u、随机各2个候选；批量过滤覆盖更多候选，且多个块可共同影响检测。单块改善稀少不能推翻雪天批量过滤的收益，也不能证明所有差信息无害。
- 分场景 `scene_ap70.json` 尚未在本轮对话中回传分析；未额外验证收益的场景分布、多种子稳定性或跨数据集泛化。不能写成已确认普遍稳定收益。

### 12.2 用户已经确定的研究决策

用户明确认为：雪天局部提升不足以抵消其他条件下降，继续增加判断模块不值得优先投入；保留最简单的 A0B0 更合适，并已确认结束当前分支。

1. **保留 A0B0 作为后续实验的通信基础。** A0B0 是规则需求＋规则响应、受字节预算限制的 BEV 块传输，绝不是全通信。
2. **后续默认实验基础为冻结 GSPR＋A0B0 通信＋AttFuse。** 全通信与原始全通信继续作为参考，不要求稀疏通信必须超过全通信。
3. **停止继续训练/调节当前学习通信选择器与质量过滤分支。** 保留代码、权重、正式测试和诊断结果供追溯，不删除既有实验。不自动续训 matching/concat/no_u，不自动扫删除比例，也不为了雪天收益再加天气判断或自适应网络。重启该研究分支需用户明确提出。
4. **下一阶段创新内容尚未确定。** 不擅自写成已经选定了新的模块、方案或完成了新实验；等用户提出下一阶段目标后，在上述固定基础上开展。
5. 这一决策是研究投入与收益的选择，不能改写成“通信研究没有价值”“所有差块无害”或“GSPR 已完全消除天气影响”。雪天的过滤现象作为局部发现保留。

### 12.3 后续引用结果时的准确表述

- 正式 benchmark（第11.2节）：A0B0 平均244003.7字节/帧，全通信25052662.3字节/帧，按本项目载荷口径减少约99.0260%。
- 代价是正式 AP70 相对全通信：Clean -8.1364、Fog -1.3076、Rain -2.0681、Snow -1.3378个百分点。**必须保留 Clean 的明显损失，不能笼统写“全部天气保持 AP”或“近乎无损”。**
- 学习通信在三种恶劣天气下没有超过 A0B0；Clean matching 的 AP70 有小幅提升，但 AP30/AP50 下降，不能据此声称学习模块整体更优。
- 今后正式报告仍严格使用第11节固定 test 数据和非全局排序 AP；validation 上的在线天气诊断只作为开发分析。
- 用户要求：科研解释先用一句话说结论，再用给同事讲代码的方式说明；术语首次出现用白话解释。可能耗时很长的运行指令默认提供后台运行方式（如 nohup），并附日志查看方法。

本次只更新研究进度与决策记录，没有改模型、切换配置、启动训练或运行新测试。

## 13. 2026-09-15：A0B0 规则保留，通信预算重新选择（代码已新增，服务器待运行）

- 用户在确认 Clean AP70 损失8.1364个百分点后，明确表示不能接受晴天及恶劣天气的 AP 下降，愿意增加通信量。因此第12节的“保留 A0B0”指保留规则，不代表接受原256 KiB配置；后续默认预算必须等本轮结果确认。
- 已新增 `gspr_evidence/budget_scan.py`、`budget_report.py`、`run_budget_scan.sh`、`test_budget_scan.py`、`BUDGET_SCAN.md`。没有改 GSPR、AttFuse、A0B0 排序、邻车平均配额、原训练源码或原正式测试入口；不重训，不需通信头checkpoint或教师缓存。
- 扫描每场景总预算256 KiB、1/2/4/8/16/32 MiB及全通信。每帧各档位共用编码，仍实际序列化请求/索引/三尺度特征；报告实际平均及峰值字节，不能把上限当作实际通信量。
- 开发阶段固定完整 OPV2V validation，天气在线模拟；本次预算实验**开发和正式均用非全局排序 AP**，与前几轮 global-sort 开发指标不同，须看协议标注。不是把在线模拟天气冒充 OPV2V-W。
- 通过条件：Clean/Fog/Rain/Snow 的 AP30/AP50/AP70 共12项，未经显示舍入的值全部不低于同条件全通信。`selection.json` 选已测档位中最小通过预算；若没有通过档位，明确回退 full，不自动放宽允许下降幅度。
- 正式阶段只读取开发选定的预算，固定 OPV2V clean test 与已有 OPV2V-W fog/rain/snow test，关闭在线增强，不扫描 test 来挑预算。若正式有指标下降，不宣称无损，也不据 test 自动改选档位。
- 两阶段均包含 full 和 original_full：每帧原模型与通信全量路径核对输出，独立报告 AP。检测源码/权重/配置指纹受检查；不根据平均AP或两位小数宣布通过。
- 后台入口 `nohup bash gspr_evidence/run_budget_scan.sh ... &`。默认 `PHASE=development`；正式需设置 `PHASE=benchmark`、`DEV=已完成开发扫描目录`。完整命令见 `BUDGET_SCAN.md`。结果在 `/data/cjm/datasets/logs/gspr_a0b0_budget_阶段_时间戳/budget_results.md`，含ETA、每帧统计及完整条件。
- 本地预算/实际包/全量饱和一致性、12指标筛选及错误协议拒绝测试通过；正式GPU扫描尚未执行，没有已经满足零下降要求的稀疏预算结论。未来数据也不能由本轮有限测试保证永不下降。

## 14. 2026-09-15：CEIF 融合设想与训练前可行性审查（服务器待运行）

- 用户要求从方法论层面设计融合机制，已提出 CEIF：用可信端点、可信射线穿越和实际通信可用性建立观测约束，区分需要补全的缺失与需要纠正的伪信息。设计见 `研究相关论文/融合相关论文/CEIF_创新融合模块设计_20260915.md`。这是研究设计，不是已训练模块。
- 用户进一步明确要求：实际改模型训练前，先检查该机制若正确发挥作用，当前数据与冻结表示中是否存在 AP 收益。已新增独立 `ceif_audit/`，不修改冻结 GSPR/AttFuse、原通信选择器及原训练入口，不启动训练。
- 主审查：同帧固定编码与收到的特征，在预测框对应局部区域替换为某来源的已接收特征，再经过原检测头重算 AP。候选区域和证据规则不读取 GT，缺失消息不作为自由证据。点可靠性与射线原始回波必须关联，不能借用其他回波可信度；穿越不等于整框/整柱为空。
- 对照包括基线/full、GT 删除全部 FP70 的宽松参照、仅证据标记 FP 的理想清理、同池随机特征干预、无 GT 证据规则干预、证据候选/全部候选中的 GT 事后特征选择。事后选择要求保留全部基线 GT 匹配且不新增 FP70，优先更多补回，其次更少 FP；不是 AP 最大化或 CEIF 理论上限，必须看最终实际 AP。
- 默认 full，因为 A0B0 最终预算尚未回传；支持 `BASELINE=a0b0 BUDGET_BYTES=明确选定值`，不擅自采用 256 KiB。额外几何证据侧信道单独估算字节，未从特征预算扣除，不声称同总带宽收益。
- 默认 `PHASE=development`：完整 validation、在线模拟天气、非全局排序 AP。可显式使用 `PHASE=benchmark-audit` 核实固定 OPV2V/OPV2V-W test，关闭在线增强；含 GT 事后选择的结果必须标为探索性审查，不当成训练后正式性能，不据其调阈值或选模型。
- 后台入口及完整命令见 `ceif_audit/README.md`。默认全帧，可先 `SMOKE=2` 做连通检查。结果含 `feasibility.md`、各天气 `audit.md`、AP 输入、候选干预记录、分场景 AP、源码/模型指纹及通信成本。
- 本地 CPU 的几何/掩码/真实消息重放/匹配/合成流程测试通过，14 个冻结源码哈希通过。合成流程测试不等于真实服务器推理；当前没有 CEIF 可行性审查的实际数据 AP，不得记录成已证实提升。

### 14.1 2026-09-16：用户回传完整 CEIF 审查结果

- 结果目录 `/data/cjm/datasets/logs/ceif_opv2vw_audit_20260915_165200`。本次依据用户贴回的 feasibility.md 和四份 audit.md；尚未读取服务器 protocol.json、frames.jsonl、actions.jsonl，不能声称已核验其源码/权重哈希或完成逐样本归因。
- benchmark-audit，四条件均 2170 帧、full 基线、无在线增强、非全局排序平面 IoU AP。基线 AP70 Clean/Fog/Rain/Snow 为 .832485/.639663/.647507/.518248，与第11.2节公共 full 数值一致。
- 同条件 AP70 变化（百分点）：证据 FP70 理想清理为 +.0000/+.0004/+.0486/+.2624；无 GT 证据特征规则为 -.8215/-.5474/-1.3489/-1.3691；证据候选中 GT 事后特征选择为 +.1558/+.6094/+.8369/+1.0375；完整有限候选池 GT 事后选择为 +.3616/+.7931/+1.2526/+2.0826。
- 被冲突标记的框共 757/394/787/1195，其中 FP70 为 0/2/52/272，其余为 TP70。标记内错误比例为 0%/.51%/6.61%/22.76%；这些是所筛查高分框的结果，缺少该候选群整体 FP 比例，不能直接推断相对随机富集程度。
- 证据合格干预 3022/5799/5402/6211 次，有益 39/134/204/297，有害 575/762/1239/1383；有益率约 1.29%/2.31%/3.78%/4.78%，有害率 19.03%/13.14%/22.94%/22.27%。候选级统计不等于独立帧或独立目标数量。
- 当前结论：存在有限的局部特征干预机会，但现有规则四条件都掉 AP，不能宣布 CEIF 可行性通过。GT 事后选择的正差不能直接归因于射线纠错，因为汇总混合 correct/complete 两类动作；当前补全条件也不检查基线是否已经检出该目标。
- 代码复核确认：局部端点/射线冲突会触发预测框覆盖 block 的整块单来源替换，属于几何证据尺度与干预尺度不一致；这是已知代理设计局限，不是已通过逐样本实验确认的唯一失败原因。局部冲突不等于整框错误，FP70 也可能是定位/重复而非天气假目标。
- 下一步建议仅利用既有动作日志区分纠错和补全贡献、规则选择的误伤及场景集中性，暂不直接进入完整 CEIF 训练，不在 test 上扫阈值。本节是分析建议，不替用户记录已经决定放弃/继续训练。

## 15. 2026-09-16：用户明确授权最小可训练 CEIF 原型

- 用户曾提出至少 +5 个百分点的研究收益目标并倾向停止 CEIF，随后明确要求实际实现最小原型、训练后判断效果。因此当前任务是推进该原型，不能继续拿前面的停止建议阻止已授权工作。
- 用户明确确认：OPV2V train 在线模拟恶劣天气用于训练，validation 用于选择轮次；最终必须在现有 OPV2V-W fog/rain/snow test 上测试，同时保留原 Clean test。测试关闭在线增强。不能把 validation 用于反向传播。
- 新增独立 `ceif_min/`：冻结原 GSPR/骨干/AttFuse/检测头，先训练占用查询头，再冻结该头并训练轻量残差网络与一步局部观测投影。可信端点提供下界，可信射线提供上界；未知不修正；冲突保留并减弱折中修正。投影只作用于查询邻域特征，不再按预测框整块替换来源特征。
- 查询头使用 clean train 实际端点/射线伪标签，不用车辆框填充占用真值。融合训练每帧使用 clean 与随机物理天气两个分支，沿用现有模拟器；不重训 GSPR，不宣称复刻原 GSPR 全部训练超参数。
- 默认 full 通信，1 epoch 查询头预训练 + 3 epochs 融合训练，batch size 1，约1024查询/帧，学习率2e-4。两个同初始化模型同时训练：ceif 启用前向投影；aux_only 使用同网络和相同观测辅助监督但关闭投影。基线为冻结原模型。
- 查询头按 clean validation 观测损失选优，两个融合模型分别按四条件 validation AP70 均值选优。固定最佳权重后测试，不在 test 上选轮次、调阈值或自动续训。全量非全局排序平面 IoU AP，报告各天气 AP30/AP50/AP70，并核查 +5 pp 目标和 Clean 代价。
- `ceif_min/run_dev.sh` 默认完成训练、完整开发评价后自动运行 OPV2V-W 正式测试；`TEST_AFTER_TRAIN=0` 可仅开发。SMOKE 不允许正式测试。支持按 epoch 恢复，另有 `run_test.sh` 独立测试入口；后台命令见 README。
- 额外几何证据单独计费，不宣称同总带宽；观测查询/投影幅度、投影步长、约束冲突和违背量均记录，检查模型是否实际使用核心机制。
- 本地已完成新模块方向/未知/冲突/梯度/真实点射线组包/冻结头、实际优化器与检测损失的合成训练及保存恢复测试，连同既有审查边界共22项测试通过。真实数据训练、HIP执行和 OPV2V-W AP 尚未在本地执行，不得写成已获得提升。

### 15.1 2026-09-17：CEIF 最小原型正式 benchmark 结果

- 结果文件：`/data/cjm/datasets/logs/ceif_min_train_20260916_101742_opv2vw_test_20260917_155212/results.md`。
- 阶段为 `benchmark`，`smoke=False`。原前端、骨干、AttFuse 和检测头冻结；`ceif` 与 `aux_only` 仅训练相同的残差修正网络，`ceif` 额外启用显式观测投影，`aux_only` 使用相同观测辅助损失但不执行前向投影。使用 full communication；额外几何载荷单独统计。

| Weather | Model | AP30 | AP50 | AP70 | ΔAP70 / pp |
|---|---|---:|---:|---:|---:|
| clean | baseline | 0.921149 | 0.912662 | 0.832568 | +0.0000 |
| clean | ceif | 0.910333 | 0.901992 | 0.821628 | -1.0941 |
| clean | aux_only | 0.910735 | 0.902603 | 0.823145 | -0.9423 |
| fog | baseline | 0.719483 | 0.708803 | 0.639395 | +0.0000 |
| fog | ceif | 0.714430 | 0.703079 | 0.631742 | -0.7654 |
| fog | aux_only | 0.715186 | 0.703918 | 0.631884 | -0.7512 |
| rain | baseline | 0.760127 | 0.743775 | 0.646653 | +0.0000 |
| rain | ceif | 0.747349 | 0.731584 | 0.635622 | -1.1031 |
| rain | aux_only | 0.749440 | 0.732994 | 0.637272 | -0.9381 |
| snow | baseline | 0.643945 | 0.624918 | 0.518487 | +0.0000 |
| snow | ceif | 0.633831 | 0.613762 | 0.508052 | -1.0434 |
| snow | aux_only | 0.635088 | 0.614819 | 0.508243 | -1.0244 |

- `ceif` 在 Clean/Fog/Rain/Snow 四个条件下的 AP30、AP50、AP70 均低于对应 baseline；AP70 分别下降 `1.0941/0.7654/1.1031/1.0434` 个百分点。
- `aux_only` 同样在四个条件下低于 baseline；AP70 分别下降 `0.9423/0.7512/0.9381/1.0244` 个百分点。
- `ceif` 在四个条件的 AP30/AP50/AP70 共 12 项指标上均未超过 `aux_only`；显式观测投影在本次固定预算最小原型中没有带来 AP 增益。
- 三种恶劣天气均未达到预设的 `+5 pp` AP70 收益目标，实际均为负变化；Clean AP70 也下降 `1.0941 pp`。
- 本轮固定预算最小原型的正式结果表明：当前 CEIF 训练实现没有提升检测 AP，且相对冻结 baseline 与 `aux_only` 均未显示正向性能收益。本轮测试未根据 test 结果修改阈值、继续训练或重新挑选轮次。

## 16. 2026-09-17：无学习的全覆盖通信压缩与层级重算基线（正式 benchmark 已完成）

### 16.1 当前研究定位

- 本节覆盖旧的“先做 A0B0 budget scan 再决定通信基础”的当前优先级描述。第11–13节的 A0B0、learned communication、质量过滤和 budget scan 仍是有效历史证据，但不再是 `lossless_comm` 的前置门禁。
- `lossless_comm` **不是学习模块**：不新增训练参数、不生成新 checkpoint、不量化、不做区域/通道 top-k，也不使用 GSPR reliability/u 决定哪些特征发送。
- 当前问题从“哪些块值得传”转为：**在保留 full 空间覆盖和任务信息时，现有多尺度 BEV 表示中有多少编码冗余与跨尺度结构冗余可以去掉？**
- 当前目标仍是：**先保住 AP，再看通信量。** development 与正式 benchmark 均已完成，正式结果见 16.10。

### 16.2 四条对照路径

`lossless_comm` 当前比较：

1. `raw_full`：复用现有工程的三尺度 full communication，作为通信参照。
2. `lossless_full`：三尺度全部发送，但对每个 float32 做 bit-exact 无损编码；解码后检查 IEEE-754 bit pattern 完全一致。无特征选择、无量化、无训练。
3. `level0_recompute`：只发送完整 level0；接收端针对**每个 peer 自己的 level0、在融合之前**运行冻结的 `backbone.blocks[1]`、`backbone.blocks[2]` 重算 level1/level2，再走原逐尺度 AttFuse 与检测头。
4. `original_full`：直接调用冻结 GSPR-AttFuse 原始 forward，用于核对通信 full 路径没有改变模型语义。

### 16.3 level0-only 的结构依据

当前冻结 backbone 是串行依赖：

```text
level0 = block0(canvas)
level1 = block1(level0)
level2 = block2(level1)
```

每个 coarse communication cell 的 raw feature values：

- level0：`64 × 4 × 4 = 1024`
- level1：`128 × 2 × 2 = 512`
- level2：`256 × 1 × 1 = 256`
- 合计：`1792`

因此只传 level0 会在熵编码前去掉 `768 / 1792 = 42.86%` 的 raw feature values。

**重要：42.86% 只是 raw-value 理论减少，不等于最终 wire bytes 减少 42.86%。** 最终字节仍由浮点熵、零值比例、包头、控制消息与 codec 决定。

### 16.4 无损 codec 边界

- `lossless_comm/codec.py` 保持 float32 IEEE-754 bit pattern；`+0.0` 可由 bitmap 表示，同时保留 `-0.0` 的位模式。
- 每个 tensor 比较两种可逆表示：完整 uint32 words 的 byte shuffle；或 `+0.0` bitmap + 非零 words byte shuffle，选择压缩后更小者。
- `BACKEND=auto`：安装 `zstandard` 时使用 Zstandard，否则回退 Python 内置 zlib。实验记录必须报告实际 backend。
- `lossless_full` 解码结果若不能逐 bit 等于发送端原始三尺度，先修 codec/transport，禁止继续解释 AP。

### 16.5 level0 重算的数值边界

- level0 解码必须 bit-exact，`max_level0_error = 0`。
- level1/level2 用冻结 block1/2 在 receiver 重新前向，并与 sender-side 原尺度比较。
- sender 原 levels 来自多 CAV batch，receiver 当前按 peer 单独重算；HIP/GPU kernel 与 batch shape 差异可能带来微小浮点误差，因此不能先验承诺 level1/level2 bit-exact。
- 必须同时检查 `max_level1_error/max_level2_error`、最终 logits 和 AP；不能通过放宽 tolerance 或四舍五入 AP 来制造“无损”。

### 16.6 开发与正式评测协议

Development：

- 完整 OPV2V validation；
- Clean + 在线物理 Fog/Rain/Snow；
- 全帧、非 global-sort AP；
- 只用于验证数值路径、压缩率、AP方向与工程开销；不得称为 OPV2V-W 正式测试。

Formal benchmark：

- OPV2V Clean test；
- 现有 OPV2V-W fog/rain/snow test；
- 关闭在线天气增强；
- 全帧、非 global-sort AP；
- 已完成，结果见 16.10。

### 16.7 当前计量边界

1. **字节口径尚需统一。** 当前 `raw_full.total_bytes` 复用旧 full 协议并包含 request packet；`lossless_full/level0_recompute` 当前 `total_bytes` 主要累计新压缩 feature packet。论文若比较完整协议通信量，必须统一 request/metadata/feature 口径；至少另报 feature-payload 对 feature-payload。
2. **当前 codec 时间不是严格端到端延迟。** `encode_ms/decode_ms` 未完整包含 `.cpu().numpy()` 的 GPU→CPU staging 和 decoded tensor `.to(device)` 的 CPU→GPU staging。若报告端到端延迟，应分项统计 D2H、encode、decode、H2D、recompute。
3. **通信—计算交换仍需单独讨论。** `level0_recompute` 增加 receiver 端 block1/2 计算，但本次测得重算仅约 7–8 ms；当前更大的时间开销来自 CPU codec 路径。用户当前研究目标优先确认 AP 与通信量，因此本阶段不把 CPU codec 时延作为否决条件。

### 16.8 设备与运行脚本

- `lossless_comm/run_all.sh` 的 `GPU=<physical_index>` 只用于设置 `ROCR_VISIBLE_DEVICES="${GPU}"`，不再设置 `HIP_VISIBLE_DEVICES` 或 `CUDA_VISIBLE_DEVICES`，符合本项目设备隔离规则。
- 脚本默认 `GPU=0`。若使用物理 HCU3，应保证 `GPU=3` 与 `bash lossless_comm/run_all.sh` 在同一个 shell 命令中，并检查启动日志打印 `GPU/HCU physical index=3`。
- 此前曾出现把 `nohup env \\` 单独执行、后续环境变量分开输入，导致脚本实际使用默认 HCU0；该误启动不得作为正式实验结果。

### 16.9 当前状态与决策门槛

- `lossless_comm/codec.py`、`transport.py`、`benchmark.py`、`summarize.py`、`run_all.sh`、测试和 README 已存在，且不修改冻结 GSPR-v1 源码。
- development 与 formal benchmark 均已完成；截至 2026-09-17，正式结果见 16.10。
- 本轮主要成功标准已满足：**在固定正式测试协议下保持全部 AP 指标不下降，同时显著降低当前实现统计的通信量。**
- `lossless_full` 保留为 bit-exact 无损工程基线；`level0_recompute` 在保持 AP 的同时进一步降低字节，当前优先作为“利用 backbone 层级确定性依赖，避免重复传输可重算尺度”的通信方法候选。
- 暂不回到 learned block selection 主线，也不因当前 CPU codec 延迟较高就否定通信压缩结果。若后续论文需要系统级延迟结论，再单独优化/测量 codec 与 staging。
- near-lossless / learned codec 暂不提前实现；只有在当前无损路线的进一步压缩空间不足时再讨论。

### 16.10 2026-09-17：`lossless_comm` 正式 benchmark 完成

结果目录：

```text
/data/cjm/datasets/logs/lossless_comm_benchmark_20260916_182339
```

结果文件：

```text
/data/cjm/datasets/logs/lossless_comm_benchmark_20260916_182339/results.md
```

正式协议继续遵守第11节：OPV2V Clean test + 既有 OPV2V-W Fog/Rain/Snow test，关闭在线天气增强，全帧、非 global-sort AP。

#### A. 正式结果

| Weather | Mode | AP30 | AP50 | AP70 | Mean MiB/frame | Reduction vs raw |
|---|---|---:|---:|---:|---:|---:|
| clean | raw_full | 0.921226 | 0.912739 | 0.832485 | 23.8921 | 0.00% |
| clean | lossless_full | 0.921226 | 0.912739 | 0.832485 | 3.5508 | 85.14% |
| clean | level0_recompute | 0.921226 | 0.912739 | 0.832485 | 2.8540 | 88.05% |
| fog | raw_full | 0.719748 | 0.709023 | 0.639663 | 23.8921 | 0.00% |
| fog | lossless_full | 0.719748 | 0.709023 | 0.639663 | 1.5016 | 93.72% |
| fog | level0_recompute | 0.719748 | 0.709023 | 0.639663 | 1.1632 | 95.13% |
| rain | raw_full | 0.759965 | 0.743751 | 0.647507 | 23.8921 | 0.00% |
| rain | lossless_full | 0.759965 | 0.743751 | 0.647507 | 1.8139 | 92.41% |
| rain | level0_recompute | 0.759965 | 0.743751 | 0.647507 | 1.4260 | 94.03% |
| snow | raw_full | 0.644169 | 0.624993 | 0.518248 | 23.8921 | 0.00% |
| snow | lossless_full | 0.644169 | 0.624993 | 0.518248 | 1.2960 | 94.58% |
| snow | level0_recompute | 0.644169 | 0.624993 | 0.518248 | 0.9901 | 95.86% |

`original_full` 在四种条件下与 `raw_full` 的 AP 完全一致，且表中通信量同为 23.8921 MiB/frame，用于确认原模型路径与通信 full 路径对齐。

#### B. 当前可直接成立的结论

1. **AP 保住了。** `lossless_full` 和 `level0_recompute` 在 Clean/Fog/Rain/Snow × AP30/AP50/AP70 共12项指标上，都与 `raw_full` 完全一致到 `results.md` 展示的六位小数。
2. **无损熵编码本身已有很大压缩空间。** `lossless_full` 的通信量由 23.8921 MiB/frame 降至 1.2960–3.5508 MiB/frame，对应当前表口径减少 85.14%–94.58%。
3. **层级重算进一步有效。** `level0_recompute` 仅发送完整 level0、接收端冻结重算 level1/2，通信量进一步降至 0.9901–2.8540 MiB/frame，对应减少 88.05%–95.86%，同时 AP 不下降。
4. 当前最有价值的通信发现是：**多尺度 BEV 通信中同时存在显著的可逆熵冗余和可利用的跨尺度确定性结构冗余。**
5. 对当前研究目标而言，`level0_recompute` 优先于旧 A0B0：A0B0 在极低通信量下会损失 AP，而 `level0_recompute` 在本次正式 benchmark 中实现了显著压缩且未观察到 AP 下降。

#### C. 时间结果的使用边界

- 当前 `results.md` 中 `lossless_full` 编码约 221–456 ms、解码约 55–71 ms；`level0_recompute` 编码约 159–360 ms、解码约 42–56 ms，重算仅约 7–8 ms。
- 用户当前的研究判断是：**本阶段首先确认“通信量显著降低且 AP 保持”，CPU codec 耗时不作为否决该路线的条件。**
- 不能把较高编码时间直接归因于“CPU性能差”；服务器 CPU 性能、Python/NumPy 实现、codec backend、单线程/多线程配置等尚未做独立对照。
- 当前计时仍不是严格端到端网络系统延迟；此前记录的 D2H/H2D staging 未完全计入。若论文需要延迟主张，后续再单独做统一硬件与完整链路测量。

#### D. 字节统计口径注释

- `results.md` 中的 `Reduction vs raw` 是当前实现直接输出的正式实验数值，可用于当前工程判断。
- 第16.7节记录的计费差异仍未被单独修正：`raw_full.total_bytes` 复用旧协议并包含 request packet，而新 lossless 模式主要累计压缩 feature packet。
- 由于本轮降幅达到约 85%–96%，该口径差异不会改变“存在显著压缩空间”的方向性结论；但论文若声称严格的 protocol/wire reduction，仍应统一 request/metadata/feature 计费后重新生成最终通信表。
