# TripleMixer 监督的 GSPR 实验流程

本流程与冻结的 OPV2V-W 测试集完全分离。流程使用干净的 OPV2V
train/validate 数据，通过 TripleMixer 中提供的天气模拟代码生成紧凑的逐点监督数据集，
先预训练 GSPR 点级可靠性网络，再联合微调 GSPR 点级任务和 AttFuse 检测任务。

所有生成的数据、逐点标签、权重和日志必须存放在：

```text
/data/cjm/datasets
```

严禁将大型数据、权重或日志写入 `/home/cjm/OpenCOOD-main` 或
`/home/cjm/OpenCOOD-main/cjmnet`。

## 1. 三种天气模拟后端说明

- Fog：调用 TripleMixer 发布的 Fog 模拟器及其积分查找表。
- Rain：调用 TripleMixer 发布的 LISA Rain 路径。
- Snow：第一轮快速筛选调用 TripleMixer 同一 LISA 模块中实际存在的
  `atm_model="snow"` 分支，并在 manifest 中明确记录为
  `triplemixer_lisa_snow_branch`。

当前 Snow 快速后端不能被描述成完整的粒子场 LiDAR_snow_sim。完整 LiDAR_snow_sim
依赖传感器线束标定和预计算粒子场，应该在逐点监督闭环验证成功后，作为更严格的 Snow
后端消融实验加入。

TripleMixer 的 LISA 源码使用了旧版 `scipy.integrate.trapz`。服务器新版 SciPy
已经移除该名称，GSPR 适配器会在加载 LISA 前将功能等价的
`scipy.integrate.trapezoid` 映射为 `trapz`。因此不需要降级服务器 SciPy，也不需要修改
TripleMixer 原始文件。

LISA 还会无条件导入 `PyMieScatt`，但本实验使用 TripleMixer 随仓库发布的
`tools/rain_sim/mie_q.npz` 预计算 Mie 系数，并不会在线调用 `PyMieScatt`。适配器会先验证
该缓存存在；若服务器没有安装 `PyMieScatt`，则在导入期间提供仅支持缓存路径的受限兼容
占位。因此无需额外安装 `PyMieScatt`。如果缓存不存在，适配器会直接终止，避免静默改用
不同的系数或模拟过程。

## 2. 服务器依赖与标签语义预检

必须先运行下面的预检。它会检查：

- TripleMixer Fog 查找表；
- `mie_q.npz`；
- LISA、Mie 系数缓存和 SciPy 兼容性；
- Fog、Rain、Snow 三种模拟器接口；
- 输出点是否为有限数值；
- 点云、可靠性标签、天气类型和来源索引是否严格等长。

```bash
cd /home/cjm/OpenCOOD-main

GSPR_DIR=/home/cjm/OpenCOOD-main/cjmnet
PYTHON=/home/cjm/miniconda3/envs/opencood/bin/python

PYTHONPATH="/home/cjm/OpenCOOD-main:$GSPR_DIR" \
"$PYTHON" "$GSPR_DIR/verify_gspr_supervision.py" \
  --triplemixer-root "$GSPR_DIR/TripleMixer-main"
```

只有当输出 JSON 同时包含 `fog`、`rain`、`snow` 且没有异常时，才能继续生成数据。
预检只使用 128 个随机点，某种天气暂时出现 `noise: 0` 不一定表示模拟器错误；正式开发集在
训练前还会检查是否同时包含可靠点和噪声点。

## 3. 最小数据生成 Smoke Test

首先只从 train 和 validate 各取一个原始 PCD，并且只生成 moderate 强度。这个数据仅用于
检查格式、依赖、路径和标签对齐，不能用于报告模型性能。

```bash
DATA_ROOT=/data/cjm/datasets/gspr_opv2v_weather_smoke
TRIPLEMIXER_ROOT=/home/cjm/OpenCOOD-main/cjmnet/TripleMixer-main

nohup "$PYTHON" "$GSPR_DIR/generate_gspr_opv2v.py" \
  --source-root /data/scd/datasets/opv2v_official_data_dumping/train \
  --output-root "$DATA_ROOT" \
  --triplemixer-root "$TRIPLEMIXER_ROOT" \
  --split train \
  --max-scenarios 1 \
  --max-files-per-scenario 1 \
  --severities moderate \
  --overwrite \
  > "$DATA_ROOT.train.log" 2>&1 &
echo $! > "$DATA_ROOT.train.pid"

nohup "$PYTHON" "$GSPR_DIR/generate_gspr_opv2v.py" \
  --source-root /data/scd/datasets/opv2v_official_data_dumping/validate \
  --output-root "$DATA_ROOT" \
  --triplemixer-root "$TRIPLEMIXER_ROOT" \
  --split val \
  --max-scenarios 1 \
  --max-files-per-scenario 1 \
  --severities moderate \
  --overwrite \
  > "$DATA_ROOT.val.log" 2>&1 &
echo $! > "$DATA_ROOT.val.pid"
```

数据生成只使用 CPU，train 和 validate 两个进程可以同时运行。查看状态：

```bash
for SPLIT in train val; do
  PID=$(cat "$DATA_ROOT.$SPLIT.pid")
  if kill -0 "$PID" 2>/dev/null; then
    echo "$SPLIT 仍在运行，PID=$PID"
  else
    echo "$SPLIT 已结束，PID=$PID"
  fi
  tail -20 "$DATA_ROOT.$SPLIT.log"
done
```

完成后必须存在：

```text
$DATA_ROOT/train_manifest.jsonl
$DATA_ROOT/val_manifest.jsonl
$DATA_ROOT/train_generation_summary.json
$DATA_ROOT/val_generation_summary.json
```

每个 NPZ 样本包含：

```text
points          增强后的 XYZI 点
reliability     二值可靠性目标，可靠回波为 1，天气散射或替代点为 0
noise_type      TripleMixer 模拟器原始天气状态
source_index    能对应原始点时记录原始索引；天气替代点为 -1
```

## 4. 小型 Development 数据集

第一轮先只生成 moderate 天气，用于快速比较方法。推荐：

- Train：4 个场景，每个场景均匀抽取 6 个 PCD；
- Validate：2 个场景，每个场景均匀抽取 6 个 PCD；
- 每个 PCD 分别生成 Fog、Rain、Snow；
- 共得到 72 个训练样本和 36 个验证样本。

```bash
DATA_ROOT=/data/cjm/datasets/gspr_opv2v_weather_development

nohup "$PYTHON" "$GSPR_DIR/generate_gspr_opv2v.py" \
  --source-root /data/scd/datasets/opv2v_official_data_dumping/train \
  --output-root "$DATA_ROOT" \
  --triplemixer-root "$TRIPLEMIXER_ROOT" \
  --split train \
  --max-scenarios 4 \
  --max-files-per-scenario 6 \
  --severities moderate \
  --overwrite \
  > "$DATA_ROOT.train.log" 2>&1 &
echo $! > "$DATA_ROOT.train.pid"

nohup "$PYTHON" "$GSPR_DIR/generate_gspr_opv2v.py" \
  --source-root /data/scd/datasets/opv2v_official_data_dumping/validate \
  --output-root "$DATA_ROOT" \
  --triplemixer-root "$TRIPLEMIXER_ROOT" \
  --split val \
  --max-scenarios 2 \
  --max-files-per-scenario 6 \
  --severities moderate \
  --overwrite \
  > "$DATA_ROOT.val.log" 2>&1 &
echo $! > "$DATA_ROOT.val.pid"
```

只有 moderate 快速实验能够正常学习后，才生成完整强度：

```bash
--severities light moderate heavy
```

扩大数据生成完成后，必须先按 split、weather、severity 审计，不能仅按文件数判断天气监督
是否均衡：

```bash
"$PYTHON" "$GSPR_DIR/summarize_gspr_corpus.py" \
  --root "$FULL_ROOT" \
  --splits train val \
  --verify-npz \
  --json-output "$FULL_ROOT/corpus_audit.json" \
  > "$FULL_ROOT/corpus_audit.log" 2>&1
```

重点检查 `noise_points`、`zero_noise_fraction`、`noise_per_sample_q50/q95` 和
`lost_points`。如果 Rain 大量样本完全没有噪声，必须先制定分层采样方案，不能直接用普通
shuffle 启动正式训练。

如果 NPZ 已生成但 manifest 因中断或误用 `--overwrite` 变成空文件，使用与原生成任务完全
相同的参数重新运行 `generate_gspr_opv2v.py`，但必须去掉 `--overwrite`。生成器会读取已有
NPZ、校验点与标签并补写缺失记录，不会重新执行耗时的天气仿真。恢复完成后 summary 中
`recovered` 应大于 0。

## 5. 标签与点共同体素化

预训练数据加载器使用与 AttFuse 完全相同的关键参数：

```text
voxel_size             [0.4, 0.4, 4.0]
cav_lidar_range        [-140.8, -40, -3, 140.8, 40, 1]
max_points_per_voxel   32
max_voxels             32000
```

可靠性标签与对应点使用相同的范围裁剪、随机顺序、pillar 分组和每柱最多 32 点截断规则，最终
形成：

```text
voxel_features       [M, T, 4]
point_targets        [M, T]
voxel_coords         [M, 4]
voxel_num_points     [M]
```

因此不会出现先体素化点、再通过坐标近邻猜测标签的问题。

## 6. GSPR 逐点监督预训练

从已经完成的 detection-only epoch11 初始化 GSPR，使本实验主要改变监督方式，而不是随机
改变初始表征：

```bash
POINT_RUN=/data/cjm/datasets/logs/gspr_point_pretrain_$(date +%Y%m%d_%H%M%S)
DETECTION_ONLY=/data/cjm/datasets/logs/gspr_attfuse_opv2v_20260904_154404/net_epoch11.pth

mkdir -p "$POINT_RUN"
cd /home/cjm/OpenCOOD-main

ROCR_VISIBLE_DEVICES=0 \
PYTHONPATH="/home/cjm/OpenCOOD-main:$GSPR_DIR" \
nohup "$PYTHON" "$GSPR_DIR/pretrain_gspr.py" \
  --train-manifest "$DATA_ROOT/train_manifest.jsonl" \
  --val-manifest "$DATA_ROOT/val_manifest.jsonl" \
  --output-dir "$POINT_RUN" \
  --init-attfuse-checkpoint "$DETECTION_ONLY" \
  --epochs 3 \
  --batch-size 1 \
  --num-workers 4 \
  --lr 0.0002 \
  > "$POINT_RUN/train.log" 2>&1 &

echo $! > "$POINT_RUN/train.pid"
echo "POINT_RUN=$POINT_RUN"
echo "PID=$(cat "$POINT_RUN/train.pid")"
```

查看训练：

```bash
tail -f "$POINT_RUN/train.log"
```

最佳逐点权重保存为：

```text
$POINT_RUN/gspr_best.pth
```

完整语料存在显著的跨天气噪声比例差异时，正式训练使用分层采样，而不是普通 shuffle：

```bash
--sampling stratified \
--noise-bearing-fraction 0.8 \
--samples-per-epoch 0
```

`samples-per-epoch=0` 表示每个 epoch 的抽样次数等于 manifest 样本数。9 个
weather/severity 组合具有相同抽样概率。每组含噪样本占比至少 80%、零噪声样本占比至多
20%；如果自然零噪声比例本来低于 20%，则保持其自然比例，避免反向过采样少数零噪声帧。
验证集不重采样。

开发版旧权重由 point-level validation AUPRC 选择；正式分层训练版改用 9 个组合的验证宏平均
AUROC 选择，不能通过 OPV2V-W AP 选择。需要同时检查：

- AUROC；
- AUPRC；
- `clean_recall_at_0.5`；
- `noise_recall_at_0.5`；
- 训练集与验证集 loss；
- 噪声点比例。

如果 AUROC/AUPRC 很高但 `noise_recall_at_0.5` 为 0，说明模型已经形成排序能力，固定
0.5 阈值下的概率校准仍然失败。此时不能只根据 AUPRC 进入联合训练，应先运行校准审计：

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH="/home/cjm/OpenCOOD-main:$GSPR_DIR" \
"$PYTHON" "$GSPR_DIR/evaluate_gspr_point_checkpoint.py" \
  --checkpoint "$POINT_RUN/gspr_best.pth" \
  --manifest "$DATA_ROOT/val_manifest.jsonl" \
  --num-workers 4 \
  --device cuda \
  > "$POINT_RUN/calibration_audit.json"
```

审计同时报告整体及 Fog/Rain/Snow 分天气的可靠点均值、噪声点均值、最佳平衡阈值，
以及保留 95% 可靠点时的噪声召回率。只有确认可靠点与噪声点的分数确实分离后，才能判断
应采用后验校准、继续训练，还是修改监督损失。

## 7. GSPR 与 AttFuse 联合多任务微调

联合训练的每个 optimizer step 同时输入：

1. 一个 Clean OPV2V 检测 batch；
2. 一个带 TripleMixer 逐点标签的天气 batch。

总损失为：

```text
L_total = L_detection + lambda_point × L_point_evidential
```

这样 AttFuse 能适应 GSPR 的软权重，同时 GSPR 不会在检测微调中遗忘逐点天气监督。

```bash
JOINT_RUN=/data/cjm/datasets/logs/gspr_point_joint_$(date +%Y%m%d_%H%M%S)
mkdir -p "$JOINT_RUN"

cd /home/cjm/OpenCOOD-main

ROCR_VISIBLE_DEVICES=0 \
PYTHONPATH="/home/cjm/OpenCOOD-main:$GSPR_DIR" \
nohup "$PYTHON" "$GSPR_DIR/train_gspr_multitask.py" \
  --config "$GSPR_DIR/gspr_attfuse_config.yaml" \
  --run-dir "$JOINT_RUN" \
  --baseline-checkpoint "$DETECTION_ONLY" \
  --gspr-checkpoint "$POINT_RUN/gspr_best.pth" \
  --train-manifest "$DATA_ROOT/train_manifest.jsonl" \
  --val-manifest "$DATA_ROOT/val_manifest.jsonl" \
  --epochs 3 \
  --detection-batch-size 2 \
  --point-batch-size 1 \
  --max-train-steps 500 \
  --num-workers 4 \
  --lr 0.00002 \
  --lambda-point 1.0 \
  > "$JOINT_RUN/train.log" 2>&1 &

echo $! > "$JOINT_RUN/train.pid"
echo "JOINT_RUN=$JOINT_RUN"
echo "PID=$(cat "$JOINT_RUN/train.pid")"
```

联合训练输出：

```text
$JOINT_RUN/net_epoch1.pth
$JOINT_RUN/net_epoch2.pth
$JOINT_RUN/net_epoch3.pth
$JOINT_RUN/history.json
```

正式 full_v1 联合训练应额外使用：

```bash
--point-sampling stratified \
--noise-bearing-fraction 0.8 \
--point-samples-per-epoch 0 \
--freeze-detector-epochs 1 \
--min-point-macro-auroc 0.995
```

第一个 epoch 冻结 BEV backbone 和检测头，但保持 GSPR 与 PillarVFE 可训练；之后解冻全模型。
除逐 epoch 权重外，还会生成 `net_best_validation.pth` 与 `best_selection.json`。最佳权重由
最低 Clean validation detection loss 选择，同时要求点级 validation macro AUROC 达到门槛。

checkpoint 选择必须同时考虑：

- Clean validation detection loss；
- Point validation AUPRC；
- Clean-point recall；
- Noise-point recall。

在 checkpoint 和所有超参数冻结前，不允许查看 OPV2V-W AP 来挑选权重。

## 8. 最终测试顺序

完成小样本筛选并冻结 checkpoint 后，才按下面顺序进行最终评测：

1. OPV2V Clean test；
2. OPV2V-W Fog test；
3. OPV2V-W Rain test；
4. OPV2V-W Snow test；
5. 与原始 AttFuse 和 `GSPR-AttFuse-Clean-DetectionOnly` epoch11 对比；
6. 重新运行 reliability/uncertainty 机制诊断。

必须显式传入冻结的 checkpoint。不要直接把包含多个 `net_epoch*.pth` 的联合训练目录传给
OpenCOOD 通用推理入口，因为它会自动选择编号最大的 epoch，而不一定是由验证集冻结的
epoch。四种条件按顺序在同一个后台进程中评测，并分别保存 `eval.yaml`：

```bash
EVAL_ROOT="$JOINT_RUN/frozen_epoch2_all_weather"

CUDA_VISIBLE_DEVICES=0 \
PYTHONUNBUFFERED=1 \
PYTHONPATH="/home/cjm/OpenCOOD-main:$GSPR_DIR" \
nohup "$PYTHON" "$GSPR_DIR/evaluate_gspr_all_weather.py" \
  --config "$JOINT_RUN/config.yaml" \
  --checkpoint "$JOINT_RUN/net_epoch2.pth" \
  --output-root "$EVAL_ROOT" \
  --conditions clean fog rain snow \
  --num-workers 4 \
  --device cuda \
  > "$EVAL_ROOT.log" 2>&1 &

echo $! > "$EVAL_ROOT.pid"
```

总结果写入 `$EVAL_ROOT/summary.json`，四种条件的完整 AP 文件分别位于
`$EVAL_ROOT/{clean,fog,rain,snow}/eval.yaml`。

小样本实验只用于判断路线是否值得扩展，不能替代完整数据、多个随机种子和正式冻结测试。
