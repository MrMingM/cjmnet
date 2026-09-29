# Adapted LMD-core for frozen F / top256

## 目的

固定当前 F detector、decoded boxes、top256 候选池和原 rotated NMS，只训练 LMD 风格的 meta quality model，检查公开 LiDAR prediction-quality estimation 方法能追回多少 GT-IoU Oracle 空间。

主方法是 **meta regression**：利用候选自身几何/分数和重叠 proposal 的离散度预测真实 BEV IoU；推理时用预测质量替换原 objectness 进行排序和 NMS。

这不是新方法，也不是 exact LMD reproduction。原因见 `FEATURE_COMPATIBILITY.md`。

## 数据协议

主训练协议：

- 完整 OPV2V official train，stride=1；
- 冻结同一个 F，分别生成 Clean + online physics Fog/Rain/Snow；
- 四条件混合训练 **一个** LMD model；
- train 内部按 OPV2V scene 留出 20% 做模型选择，同一 scene 的四天气版本不会跨到两侧；
- development validation 不参与 LMD 模型选择；
- GT 只作 meta model 的训练标签。

开发验证：

- 完整 OPV2V validation；
- Clean + online physics Fog/Rain/Snow；
- 只评价已经冻结的 LMD。

正式 benchmark：

- OPV2V clean test；
- 固定 OPV2V-W fog/rain/snow test；
- 关闭在线天气增强。

## 为什么默认训练只保留每个天气 50000 个候选

完整 train 的每帧 top256 × 四条件会产生数百万 pre-NMS candidates，而原 LMD 处理的 post-NMS outputs 数量远少。

本实现仍会先用冻结 F **完整扫描所有 official-train frame 和四种天气**。训练脚本随后对每个天气做 reservoir sampling，默认各保留 50000 个候选，再训练原 MetaDetect3D 源码支持的 Ridge、Random Forest、Gradient Boosting 三种回归器；只使用 train scene holdout 选择回归器。

如需要强制全部候选训练：

```sh
MAX_ROWS_PER_WEATHER=0 sh local_fusion_detector_adaptation/sota_reproduction/lmd/run_train.sh
```

但第一次正式运行不建议这样做，因为 sklearn GradientBoosting 在数百万 pre-NMS candidates 上开销非常大。

meta classification 是次要诊断，默认关闭。需要时：

```sh
TRAIN_CLASSIFICATION=1 sh local_fusion_detector_adaptation/sota_reproduction/lmd/run_train.sh
```

## GPU 3

冻结 F 前向和候选导出默认固定物理 **HCU 3**：

```sh
export ROCR_VISIBLE_DEVICES=3
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
```

程序内部仍看到 `cuda:0`。LMD 的 sklearn meta model 训练主要运行在 CPU。

## 运行

### 1. 先做 preflight

```sh
cd /home/cjm/OpenCOOD-main/cjmnet
CHECK_ONLY=1 sh local_fusion_detector_adaptation/sota_reproduction/lmd/run_train.sh
```

### 2. 小型 smoke

每天气只走前 4 帧：

```sh
cd /home/cjm/OpenCOOD-main/cjmnet
SMOKE=4 OUT=/data/cjm/datasets/logs/lmd_smoke \
sh local_fusion_detector_adaptation/sota_reproduction/lmd/run_train.sh
```

smoke 会明确记录 `full_split=false`，不能当正式结果。

### 3. 完整 train + weather-mixed LMD 训练

```sh
cd /home/cjm/OpenCOOD-main/cjmnet
nohup sh local_fusion_detector_adaptation/sota_reproduction/lmd/run_train.sh \
  > /data/cjm/datasets/logs/lmd_train_launcher.log 2>&1 &
```

训练结束后日志会打印 `model.pkl` 的实际路径。

### 4. 完整 development validation

```sh
MODEL=/data/cjm/datasets/logs/lmd_YYYYMMDD_HHMMSS/model/model.pkl \
MODE=validation \
nohup sh local_fusion_detector_adaptation/sota_reproduction/lmd/run_eval.sh \
  > /data/cjm/datasets/logs/lmd_validation_launcher.log 2>&1 &
```

### 5. 方法完全冻结后跑正式 benchmark

```sh
MODEL=/data/cjm/datasets/logs/lmd_YYYYMMDD_HHMMSS/model/model.pkl \
MODE=benchmark \
nohup sh local_fusion_detector_adaptation/sota_reproduction/lmd/run_eval.sh \
  > /data/cjm/datasets/logs/lmd_benchmark_launcher.log 2>&1 &
```

## 输出

训练：

- `train_cache/candidate_audit.json`
- `model/model.pkl`
- `model/training_report.json`

评价：

- `results/results.json`
- `results/results.md`
- `results/frames.jsonl`

报告同时包含：

- frame-order AP30/AP50/AP70；
- global-sort AP30/AP50/AP70；
- selection-only AP；
- score-IoU Spearman；
- IoU>=0.7 ROC-AUC / PR-AUC；
- recovered GT / lost GT / new FP；
- Oracle Recovery Ratio。

Oracle Recovery Ratio：

```text
(LMD AP70 - Original AP70) / (GT-IoU Oracle AP70 - Original AP70)
```

如果 top256 不能完整包含某帧全部原 `score>0.2` 候选，结果 JSON 会明确记录该帧数，不能把那一批 `original_f` 叫作严格原检测器复现。
