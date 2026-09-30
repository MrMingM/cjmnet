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

## 2026-09-30：已有 validation cache 的加速离线复评

旧评价在每个方法、每个 AP 阈值上重复调用 Shapely 几何交集，且只在整天气完成时输出一次进度。加速版：

- 同帧 top256 IoU 矩阵在 feature extraction 和各方法 NMS 中只计算一次；
- 与 GT 的 IoU 直接复用项目已审计的 `gspr_evidence.stage3_trace.polygon_ious`；
- AP 的贪心 GT 匹配复用同一矩阵，保留 OpenCOOD 的逐帧排序、跨帧排序、阈值、输出预算和候选集合；
- 每天气前 2 帧逐项对照旧 OpenCOOD AP、NMS 和 matched GT/FP；如果不一致就直接退出，拒绝生成似是而非的 AP；
- 每 25 帧打印进度，每完成一个天气写 `results_partial.json`；四天气全部完成后再写正式 `results.json`/`results.md`。

旧版运行中的 Python 进程不会自动加载仓库新代码，必须结束旧的 `lmd.evaluate`（只停止 evaluator，不停止 detector/train），然后对已经完整生成的 `cache` 运行：

```sh
cd /home/cjm/OpenCOOD-main/cjmnet
git pull --ff-only
export ROCR_VISIBLE_DEVICES=3
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
export PYTHONPATH=/home/cjm/OpenCOOD-main/cjmnet:/home/cjm/OpenCOOD-main
export MODEL=/data/cjm/datasets/logs/lmd_full_20260929_205547/model/model.pkl
export EVAL_CACHE=/data/cjm/datasets/logs/lmd_validation_20260930_154010/cache

SMOKE_OUT=/data/cjm/datasets/logs/lmd_fast_smoke_$(date +%Y%m%d_%H%M%S)
/home/cjm/miniconda3/envs/opencood/bin/python -u -m \
  local_fusion_detector_adaptation.sota_reproduction.lmd.evaluate \
  --eval-root "$EVAL_CACHE" --model "$MODEL" \
  --output "$SMOKE_OUT" --max-frames-per-weather 2
```

smoke 结果目录只是两帧诊断，**不得作为正式论文 AP**。smoke 通过后，用未存在的新输出目录跑完整复评：

```sh
export OUT=/data/cjm/datasets/logs/lmd_validation_fast_$(date +%Y%m%d_%H%M%S)
export MODEL=/data/cjm/datasets/logs/lmd_full_20260929_205547/model/model.pkl
export EVAL_CACHE=/data/cjm/datasets/logs/lmd_validation_20260930_154010/cache

nohup sh local_fusion_detector_adaptation/sota_reproduction/lmd/run_eval.sh \
  > "$OUT.log" 2>&1 &
```

`run_eval.sh` 将复用 `EVAL_CACHE`，不会重新提取 7920 帧。

如果仍出现 `RuntimeWarning: invalid value encountered in intersection`，不要过滤警告后直接采信 AP：先检查前 2 帧新旧结果是否完全一致，再结合天气/帧号检查退化或无效多边形。
