# CIA-SSD-style IoU quality on frozen F/top256

## 问题与边界

这个基准只问一件事：原 F 检测器已经生成的固定 top256 框，如果用 CIA-SSD 式定位质量修正分数，能追回多少 GT-IoU Oracle 的 AP70 空间？它不提出新融合方法。GSPR、PointPillar、Where2comm、融合、原分类/回归头与 decoded box 全冻结。只训练一个 IoU head。GT 只进入 official train 标签与评价；validation 只用于冻结后的评价。OPV2V-W test 完全不读取。

开发验证仍是已反复研究的 9 个 validation 场景，不能当独立泛化证据。IoU-only 分数仅作诊断，`ciassd_score` 才是本基准的主方法。

## 对照论文与源码后确认的计算

阅读：`研究相关论文/候选1paper/CIA-SSD.pdf` 第 3–4 页；原作者 `det3d/models/bbox_heads/mg_head_v4_release.py`、`det3d/models/losses/losses.py`、`det3d/core/anchor/target_ops.py`、`det3d/core/bbox/box_torch_ops.py`、`det3d/ops/nms/nms_cpu.h`、KITTI 配置；仓库 `opencood/models/ciassd.py`、`opencood/models/sub_modules/cia_ssd_utils.py`、`opencood/data_utils/post_processor/ciassd_postprocessor.py`、`opencood/loss/ciassd_loss.py`。

1. 原作者发布的实际 IoU branch 是 `Conv2d(128, 2, 3, padding=1, bias=False)`，输入为多任务检测头的 BEV 特征；2 个输出对应同一 BEV cell 的 2 个 anchor。论文图示是并行 IoU 回归分支，没有写清卷积核尺寸。原作者源码保留了已注释的 1×1 版本；当前生效版本是 **3×3**。OpenCOOD 自带 `Head.conv_iou` 则是 **1×1**。本基准跟随原作者生效代码的 3×3。
2. 原作者只在回归正 anchor 上训练 IoU branch。正 anchor 的 decoded box 与其**被分配的** GT 计算 3D IoU，target 为 `2*IoU3D-1`。负 anchor/忽略 anchor 没有 IoU loss；classification loss 仍各自处理正负。IoU loss 是 `WeightedSmoothL1Loss(sigma=3, loss_weight=1)`，按正 anchor 权重归一化。论文描述“nearest GT”散点图，但没有写 `[-1,1]` target 或该损失的实现细节；这两点以源码为准。论文说训练中将 decoded box 从 IoU loss 反传路径断开；离线冻结 F 自然满足。
3. 推理时原作者先用分类阈值 0.3 筛框，再把 raw IoU output 转为 `i=(raw+1)/2`，计算 `f=c*i^4`，其中 `c=sigmoid(cls_logit)`。源码无 clamp；本基准也不 clamp，并报告映射出 `[0,1]` 的数量。论文式子为 `c*i^β`，示例/源码 `β=4`。
4. CIA-SSD 完整模型还使用 DI-NMS：以 rectified score 排序、根据 anchor/box 中心偏移调整分数、利用 predicted IoU 与旋转框重叠给邻框加权，按距离改变融合强度并压冗余框。论文 Algorithm 1 与发布的 C++ 实现存在细节差别：C++ 使用中心偏移平方的 `exp` 来归一化权重，主框选择读取 `scores_r`，聚合得分另存在 `scores_rw`；`cnt_thresh=2.6`，重叠抑制阈值 0.3，距离分段 sigma 也硬编码。这里**不移植 DI-NMS**，因为问题要求固定原 OpenCOOD rotated NMS、范围过滤和输出预算；因此本实验是 *CIA-SSD-style score* 基准，不等同完整 CIA-SSD detector。

## 缓存适配与等价性

固定候选 ID 完全来自 `candidate_audit.candidate_ids(trace, 0., 256, geometry_ids(trace))`。旧 cache 的 `F_features_*.npz` 只含 candidate cell 的中心向量。若质量头为 1×1：

`Conv1x1(F)[a,y,x] = sum_c W[a,c,0,0]*F[c,y,x] + b[a]`，

所以缓存向量上的按 anchor 线性层与 feature map 位置的 1×1 convolution 数学等价。**3×3 需要周围 8 个 cell；旧 cache 不够。 `extract.py` 只在原候选 ID 上补 3×3 零填充特征块，并让 detector 输出与 patch 来自同一次 frozen fusion。逐帧仍以 candidate ID、原 score、回归量、decoded box/corners 作为硬一致性门；3×3 patch 中心必须与本次同一 fused feature map 的对应 cell 精确一致。旧 `F_features_*.npz` 是历史上由另一独立 fusion 调用保存的中间特征，只作为诊断记录其差异，不再要求跨运行逐元素相等。不会另选候选池或改旧 cache。

训练标签从当前 `VoxelPostprocessor.generate_label` 的 `pos_equal_one` / `pos_gt_index` 取得。当前 OpenCOOD 使用自己的 standup-BEV anchor 分配；原作者 KITTI 代码使用 `nearest_iou_similarity` 与同样的 0.6/0.45 阈值。候选池只含 top256，也只对**其中的正 anchor**训练质量头。这些是数据集/检测器结构导致的适配差别。标签使用匹配 GT 的 **3D IoU**；评测 Oracle 使用已有 `max_gt_iou` 的 **BEV IoU**，两者刻意分开。

训练脚本按 official train 的 scene ID 留出 20% scene 选择 epoch；四天气的同一 scene 始终落在同一侧。没有使用 development validation 选择 epoch、指数、阈值或校准参数。Direct IoU 对照读取已经训练的 S0 `iou_regression_seed20260929.pt`；该旧 MLP 用 BEV max-IoU 目标与候选元数据，容量和监督与 CIA-SSD 不同，报告时保留此区别。本基准不重新训练它。

## 回放与输出

`evaluate.py --method-disabled` 是必须先通过的回归门：原 F 的 AP30/50/70 对照候选审计保存值，且原分数的 fixed-budget top256 AP 必须与原流程一致，容差 1e-6。失败即退出。之后所有方法复用 `candidate_hypothesis_replay._nms_and_budget`（原 rotated NMS → range filter → 原 scorepass 每帧输出框数预算）和仓库 `eval_utils` 的帧顺序及跨帧统一排序 AP。

输出方法：`original_f`、`top256_fused`、已训练 S0 `direct_iou_regression`、主 `ciassd_score`、诊断 `predicted_iou_only`、`gt_iou_oracle`。Oracle 在同一 top256 池用每框 max GT BEV IoU 作分数，并像旧 `oracle_quality.py` 删除 IoU=0 背景候选，再执行相同 NMS/range/budget。各方法提供 AP30/50/70、两种 AP 顺序、候选 score–GT BEV IoU Spearman、IoU≥0.7 ROC-AUC/PR-AUC、与原 F 比的恢复 GT、失去原 GT、新 FP 以及逐帧候选 ID。Oracle Recovery Ratio 分别按两种 AP 口径计算：

`(method AP70 - original_f AP70) / (gt_iou_oracle AP70 - original_f AP70)`。

分母非正时输出 `null`，不能解释为追回比例；负值表示方法降低 AP。比较时只能在同一 AP 口径内使用该比例。

## 服务器命令

先把本目录代码同步到 `/home/cjm/OpenCOOD-main/cjmnet`。以下命令只在服务器运行；本地 Windows 不运行科研训练或数据集评价。脚本是 LF、POSIX `sh`，可用 `ROCR_VISIBLE_DEVICES=0` 指定一张可用卡。

```sh
cd /home/cjm/OpenCOOD-main/cjmnet
PY=/home/cjm/miniconda3/envs/opencood/bin/python
$PY -m unittest local_fusion_detector_adaptation.sota_reproduction.ciassd_iou.test_ciassd_iou -v
```

快速回归 smoke（只读现有 development cache，不训练）：

```sh
RUN=/data/cjm/datasets/logs/fusion_detector_adaptation_20260927_103227
$PY -u -m local_fusion_detector_adaptation.sota_reproduction.ciassd_iou.evaluate \
  --method-disabled \
  --eval-root "$RUN/top256_score_geometry_20260928_155556/extraction" \
  --output "$RUN/ciassd_disabled_smoke"
```

完整 official-train 补充导出及 IoU head 训练可能较久，用后台任务：

```sh
RUN=/data/cjm/datasets/logs/fusion_detector_adaptation_20260927_103227
TRAIN_OUT="$RUN/ciassd_iou_train_$(date +%Y%m%d_%H%M%S)"
nohup env OUT="$TRAIN_OUT" ROCR_VISIBLE_DEVICES=0 \
  sh local_fusion_detector_adaptation/sota_reproduction/ciassd_iou/run_train.sh \
  > "$TRAIN_OUT.log" 2>&1 < /dev/null &
echo "PID=$! OUT=$TRAIN_OUT"
```

开发 Clean/Fog/Rain/Snow 评价（同样可能较久）：

```sh
EVAL_OUT="$RUN/ciassd_iou_eval_$(date +%Y%m%d_%H%M%S)"
CHECKPOINT="$TRAIN_OUT/head/best.pt"
# 脚本会自动寻找唯一的旧 S0 checkpoint；若有多个，请显式指定 DIRECT_IOU_CHECKPOINT。
nohup env OUT="$EVAL_OUT" CHECKPOINT="$CHECKPOINT" ROCR_VISIBLE_DEVICES=0 \
  sh local_fusion_detector_adaptation/sota_reproduction/ciassd_iou/run_eval.sh \
  > "$EVAL_OUT.log" 2>&1 < /dev/null &
echo "PID=$! OUT=$EVAL_OUT"
```

训练结果：`$TRAIN_OUT/regression/results.json`、`train_patches/manifest.json`、`head/best.pt`、`head/train.json`。评价结果：`$EVAL_OUT/regression/results.json`、`validation_patches/manifest.json`、`results/results.json`、`results/results.md`、`results/frames.jsonl`。可设 `PATCH_ROOT` 复用同一冻结评价特征缓存。旧 F、原候选缓存和先前实验不被覆盖。
