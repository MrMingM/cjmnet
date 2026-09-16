# GSPR 通信创新原型 v1

新增规则起点小网络 `residual` 的结构、服务器检查与一键命令见 [RESIDUAL.md](RESIDUAL.md)。原有对照模式保持不变。

目标：分别表达已有观测的质量与支持程度，让 A 判断需求、B 判断响应优先级。收益与新颖性均待实验确认。

## 已实现

- 冻结 GSPR、PillarVFE、BEV 编码器、原检测头参数和 BN 状态，只训练 A/B。
- 复用已有三尺度网络；CoSDH/Where2comm 选择逻辑的内部适配规则，来源见 SOURCE.md。
- A 只接收自车通信前语义、质量/支持统计及单车前景置信度；B 只接收发送车自身信息和解码后的请求。标签不进入模型输入。
- 6 个统计通道：原始可靠概率均值、不确定性均值、log(1+可靠概率总和)、log(1+保留点数)、有点 pillar 比例、有效性。空区域有效性为 0，不表示自由空间。不确定性不是需求标签。
- 8×8 原始 pillar 的区域块；当前 0.4m 网格对应 3.2m。每块发送三个尺度的 4×4×64、2×2×128、1×1×256 特征。
- 请求 uint8；默认特征 float32；实际请求/响应序列化、索引与头部计费；固定邻车配额，无未计费的全局排序分数。
- 未发送的邻车特征不参与融合。训练硬前向、软反向；测试真实字节包还原硬 mask。训练反向近似会使用完整缓存特征，推理只使用收到的特征。
- 评测 AP、逐帧字节/所选块/质量/支持统计、模型及包处理耗时，可与自车路径逐目标比较纠正漏检、丢失检测和误检数量差。

## 版本边界

当前是共用 ego 网格的算法原型。保持 GSPR 原训练的 proj_first=True，不更换输入坐标分布、不重复 warp。局部数据适配器保留真实 CAV→ego 矩阵用于审计，但第一版 B **不额外输入位姿向量**；相对位置体现在预对齐网格中。传感器覆盖/射线、异步位姿、跨接收车本地编码与网络部署尚未实现。保留的位姿不作为隐藏的消息输入。

消息计费包括 uint8 请求、区域 uint32 索引、float32/float16 三尺度载荷及每包 16 字节头部；假设网格和坐标对齐信息预先共享，不包括定位广播、IP/MAC/无线重传。故是应用层载荷字节，不是无线链路总流量。每个发送车收到一份请求，按单播逐份计费。请求无法负担时直接零通信。

默认 FP32 每块 7172 字节（含 4 字节索引）。25×88 请求图每车 2216 字节，响应头每车 16 字节；256 KiB 为每个 ego 样本所有邻车合计预算。`full` 是不受预算限制的全量参照，其总载荷也完整计费。与预算组报告分开。

默认 experiment.yaml 使用 Clean 数据，仅用于接线和初筛，不能证明天气鲁棒通信。天气训练应使用同步多车 PCD/YAML 检测样本，在独立 experiment YAML 中更换 root_dir/validate_dir；不能使用 GSPR 单车逐点 NPZ 替代。也可显式使用现有天气增强配置和 `processed_lidar_weather` 分支，必须先检查对应服务器代码/外部资源及同步场景一致性。未自动生成天气数据，也未修改模拟器参数。

当前诊断是所选区域统计、目标级纠正/丢失和误检数量差；没有把“低支持目标”“共同盲区”“充分观测区域”自动判定为真值，也不把误检数量差冒充新增误检实例匹配。后续应在保留验证集按固定定义分组分析。

## 部署与检查（先运行这些）

只上传新增的整个 gspr_communication/ 目录。不要覆盖冻结目录或共享 OpenCOOD。服务器必须具有与 frozen_sources.json 一致的 v1 源码；核对失败时先确认服务器版本，不能直接更新哈希绕过检查。

选择一个已经完成、固定保存的 joint checkpoint 和它的 config.yaml。参数必须完全匹配；不允许自动跳过缺失参数。不要追读正在写入的 best/latest 文件。以下环境变量中的路径由你填入真实绝对路径；没有默认选择某个种子。

```bash
cd /home/cjm/OpenCOOD-main/cjmnet
export PYTHONPATH=/home/cjm/OpenCOOD-main/cjmnet:/home/cjm/OpenCOOD-main
# 确认该卡空闲后再设置，只能 0–6，不能使用 7。
export ROCR_VISIBLE_DEVICES=0
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
export PYTHONHASHSEED=20260908
export FRONTEND_CONFIG=/实际固定运行目录/config.yaml
export FRONTEND_CHECKPOINT=/实际固定运行目录/net_best_validation.pth

/home/cjm/miniconda3/envs/opencood/bin/python -m unittest \
  gspr_communication.test_codec gspr_communication.test_tensors -v

/home/cjm/miniconda3/envs/opencood/bin/python -m gspr_communication.verify \
  --frontend-config "$FRONTEND_CONFIG" \
  --frontend-checkpoint "$FRONTEND_CHECKPOINT"
```

verify 使用一个真实多车验证样本检查：全通信≈原模型、零通信≈单车、A/B 梯度非零、冻结参数和 BN 不漂移、训练硬前向≈序列化推理。任何失败先修接口，不开始正式训练。

## 对照与训练

| variant | 含义 | 是否训练 |
|---|---|---|
| a0b0 | 需求=1−自车置信度；响应=请求×发送车置信度 | 否 |
| a1b0 | 学习 A，规则 B | 是 |
| a0b1 | 规则 A，学习 B | 是 |
| a1b1 | 学习 A 与 B | 是 |
| quality | a0b0 响应再乘发送车质量和有效性 | 否 |
| random | 同预算随机排序 | 否 |
| none | 自车无通信 | 否 |
| full | 全量通信参照 | 否 |

这些是内部对照，不用 CoSDH/Where2comm 的论文名称命名实验结果。

```bash
/home/cjm/miniconda3/envs/opencood/bin/python -m gspr_communication.train \
  --config gspr_communication/experiment.yaml \
  --frontend-config "$FRONTEND_CONFIG" \
  --frontend-checkpoint "$FRONTEND_CHECKPOINT" \
  --variant a1b1 \
  --run-dir /data/cjm/datasets/logs/gspr_comm_a1b1_seed20260908_run01
```

a1b0/a0b1 替换 variant，并使用全新的 run-dir。训练目录已存在会拒绝，避免覆盖。默认 4 个 train 场景、2 个 validate 场景、每 5 帧抽取一帧、每轮最多 300 次更新、3 轮；场景索引对应各自 root 下排序后的文件夹。记录实际样本索引、种子、源码/前端权重哈希；同一数据协议/预算/初始化用于所有组。Python/NumPy/PyTorch/worker 均设置种子；硬件算子不保证跨平台逐位一致。

默认只用检测任务损失训练 A/B，不引入 oracle 增益标签。硬预算已经限制实际开销，不额外惩罚固定 K 的 mask 数。规则组不进行无意义的空优化。学习组根据独立 validation 检测损失选择 selectors_best.pth，仅保存很小的 A/B 权重；不重写前端权重。没有自动断点续训。

## 评测

```bash
# 学习组：使用训练目录保存的实际 experiment.yaml，避免 variant 参数不一致。
/home/cjm/miniconda3/envs/opencood/bin/python -m gspr_communication.evaluate \
  --config /data/cjm/datasets/logs/gspr_comm_a1b1_seed20260908_run01/experiment.yaml \
  --frontend-config "$FRONTEND_CONFIG" \
  --frontend-checkpoint "$FRONTEND_CHECKPOINT" \
  --selectors /data/cjm/datasets/logs/gspr_comm_a1b1_seed20260908_run01/selectors_best.pth \
  --compare-ego \
  --output-dir /data/cjm/datasets/logs/gspr_comm_a1b1_validation_run01

# 规则组不提供 selectors；其他选项、前端和预算保持一致。
/home/cjm/miniconda3/envs/opencood/bin/python -m gspr_communication.evaluate \
  --config gspr_communication/experiment.yaml \
  --frontend-config "$FRONTEND_CONFIG" \
  --frontend-checkpoint "$FRONTEND_CHECKPOINT" \
  --variant a0b0 --compare-ego \
  --output-dir /data/cjm/datasets/logs/gspr_comm_a0b0_validation_run01
```

`--data-root /实际测试目录 --all-scenes` 才是完整指定测试集：关闭开发用场景/抽帧过滤。Clean/fog/rain/snow 分别使用独立输出目录，同一个已选定权重。默认保留现有非全局置信度排序；如改用 --global-sort-detections，所有组必须统一。当前仓库 eval_utils 实际按 BEV 多边形 IoU 匹配，不应将其称为含高度重叠的 3D IoU 指标。

输出 eval.yaml、communication.json、frames.jsonl、protocol.json。延迟是编码/A/B/选择/CPU 序列化还原/融合的时间，排除第一帧 warmup，**不含数据加载、后处理或真实无线传输**；compare-ego 的第二次前向不计入这个值。暂不宣称完整端到端网络延迟。

先完成服务器 smoke test，再运行小规模四组实验。OPV2V-W 已用于 pilot，不能继续用于挑结构/阈值。通过独立 validation 选择方案后，再扩大天气数据、种子及外部数据验证。

## 本地验证状态

Windows 无 PyTorch。已通过 CPU 消息协议/预算单元测试、Python 静态语法检查、YAML 解析与冻结源码哈希复核。Tensor 测试和真实数据 verify 尚待服务器执行；无新增 AP 或训练成功声明。
