# 融合后检测端适应：F 与 F＋D 小型对照

这次实验只检验一个问题：同一份 v3 residual 融合特征交给检测端后，让检测端跟着训练，是否比只继续训练融合有稳定净收益。

- **F**：继续训练 v3 融合；独立复制的融合后 `deblocks`、`cls_head`、`reg_head` 全部冻结。
- **F＋D**：从相同权重起步，继续训练相同融合，同时训练这套融合后检测端。所有 BatchNorm 均处于 eval，运行均值和方差固定；卷积权重与 BatchNorm 的可学习参数可以更新。
- 原基础模型在两组中始终冻结，仍负责 GSPR、各车编码和来源置信度。融合后检测端是独立副本，不会改变来源信号。

两组使用同一官方 train 场景与帧、同一 shuffle 种子、Clean/在线 mixed 天气配对、PointPillarLoss 与 v3 权重修改惩罚、同一融合学习率和固定轮数。单车帧两组都计算损失但跳过更新，使优化步数相等。F＋D 的额外计算和参数量会单独体现在运行耗时；同样步数并不等于同样 FLOPs。

默认选最多 32 个训练场景、24 个 validation 场景，每场景最多 10 帧，训练 2 轮。场景与帧只按固定种子随机抽取，不按目标或失败类型挑选。没有足够场景时使用现有场景，实际索引记入 `protocol.json`。每组只保存最后一轮，不按 validation 反复挑 checkpoint。

验证沿用 BEV 平面 IoU、帧顺序累积的 AP 协议，逐条件报告 AP30/AP50/AP70、相对原 baseline 与相对 F 的恢复、原 TP 损失和新增 FP 身份。`decision_results.json` 的扩大门槛预先固定：F＋D 相对 F 三天气平均 AP70 至少 +0.005，至少两天气为正；Clean AP70 和四条件 AP50 相对 F 与原 baseline 均不能低于 0.001；原 TP 损失、新增 FP 身份相对 F 与 baseline 各不能超过参照 TP 的 1%。这是投入门槛，不是统计显著性检验。

本轮只运行官方 validation 子集和在线模拟天气，不访问 OPV2V-W。validation 与历次方案曾共用，因此也不能称完全未接触的新数据。若小型实验过门槛，再换一个配对种子复验；方向一致后才考虑完整验证和一次正式评价。

## 服务器后台运行

在原服务器项目目录和 opencood 环境：

```bash
cd /home/cjm/OpenCOOD-main/cjmnet
export RUN=/data/cjm/datasets/logs/fusion_detector_adaptation_$(date +%Y%m%d_%H%M%S)
nohup bash local_fusion_detector_adaptation/run_all.sh > "${RUN}.launcher.log" 2>&1 < /dev/null &
echo "PID=$! RUN=$RUN"
tail -f "${RUN}.launcher.log"
```

可覆盖 `PY`、`FRONTEND_ROOT`、`FRONTEND_CONFIG`、`FRONTEND_CHECKPOINT`、`V3_RUN`、`V3_CONFIG`、`PILOT_CONFIG`。`V3_RUN/residual/best.pth` 必须存在；`RUN` 必须是新目录。`protocol.json` 保存起点 checkpoint 哈希、训练索引、源码快照哈希和门槛；`train_history.json` 记录每轮帧数、优化步数和耗时；`decision_results.json` 保存两组原始指标与决定。

## top256 候选区分性审计（离线，不重新推理）

`candidate_discriminability.py` 读取 `candidate_audit.py` 用 `AUDIT_POOL=top_256` 导出的四种天气、F/F+D 候选行。它只保留范围内、至少两个来源的候选。GT BEV IoU 仅定义好框（≥0.7）和坏框（<0.5），不进入推理特征。

审计先给出低分好框、低分坏框、高分坏框和高分好框的特征分布。区分能力在相同分数段内评估：低分好框对低分坏框，高分好框对高分坏框。简单一致性对照使用融合分数、`max(来源分数 × 来源框/融合框 IoU)` 和平均来源框/融合框 IoU；完整来源对照加入各来源分数和几何统计；最后才加入融合分数变化与几何变化的交互项。报告按帧分组五折 AUC 和成组重抽样的增量区间。若有 `--group-map`（JSON：帧 ID → 场景 ID），则按场景分组。另报告同一 anchor 在 Clean 和天气 top256 中都出现时的配对变化；此项存在选池偏差。

此项科研审计在远程服务器运行。`INPUT_ROOT` 应指向包含 `candidate_audit.json` 的 top256 审计目录；可能需要几分钟，建议后台运行：

```bash
cd /home/cjm/OpenCOOD-main/cjmnet
PY="${PY:-/home/cjm/miniconda3/envs/opencood/bin/python}"
INPUT_ROOT=/data/cjm/datasets/logs/fusion_detector_adaptation_20260927_103227/candidate_source_top_256
OUTPUT="$INPUT_ROOT/discriminability_$(date +%Y%m%d_%H%M%S)"
nohup "$PY" -u -m local_fusion_detector_adaptation.candidate_discriminability \
  --input-root "$INPUT_ROOT" --output-dir "$OUTPUT" \
  > "${OUTPUT}.log" 2>&1 < /dev/null &
echo "PID=$! OUTPUT=$OUTPUT"
```

结果在 `top256_discriminability.json` 和 `.md`。旧候选行没有 `fused_bev_corners`，NMS 竞争项会明确写为 `unavailable`；其他审计照常完成。更新后的 `candidate_audit.py` 会保存融合框的四个 BEV 顶点。要补齐 NMS 竞争项，需用原冻结 checkpoint、`STAGE0_ONLY=0 AUDIT_POOL=top_256` 重导出候选行，再运行离线脚本。该竞争项报告真实框重叠与成对排序，仍需完整 NMS/AP 复验才可作最终性能结论。

重导出需要模型推理，放到服务器后台运行；`OUT` 指向全新目录：

```bash
cd /home/cjm/OpenCOOD-main/cjmnet
RUN=/data/cjm/datasets/logs/fusion_detector_adaptation_20260927_103227
OUT="$RUN/candidate_source_top_256_boxes_$(date +%Y%m%d_%H%M%S)"
nohup env STAGE0_ONLY=0 AUDIT_POOL=top_256 RUN="$RUN" OUT="$OUT" \
  bash local_fusion_detector_adaptation/run_candidate_audit.sh \
  > "${OUT}.launcher.log" 2>&1 < /dev/null &
echo "PID=$! OUT=$OUT"
```

## 方案 A：固定候选的逐来源干预审计

`run_scheme_a.sh` 在原 F/F+D pilot 的相同 validation 帧、相同四种条件上执行。它只对 F 的固定 top256 候选依次移除每辆邻车，记录同一 anchor 的分类分数变化和框变化，并缓存候选自身的融合检测特征、分类与回归输出。GT IoU 只用于离线好坏标签以及事后质量变化；GT 不参与候选选择或探针输入。单车独立预测仍只作为来源代理；逐车移除是**条件干预响应**，由于剩余来源权重会重新归一化，不能把各响应相加解释为来源贡献。

离线 `candidate_intervention_probe.py` 在低分（≤0.2）和高分（>0.2）内分别比较好框（IoU≥0.7）和坏框（IoU<0.5）。四级对照依次是候选自身特征、加普通来源代理、加逐车干预量、同时加入两类来源信息。它使用按场景分组的交叉验证；场景 ID 从 validation 数据集边界写入 `candidate_audit.json`。若读取老候选记录缺少场景映射，则必须提供 `SCENE_MAP` 才能作按场景分析。逻辑回归只是离线信息探针，**没有训练或修改 F/F+D 检测网络，也没有重评分后的 AP 结论**。

本流程需要多次模型前向，建议在服务器后台运行；默认对 90 帧/条件的 F 分支做逐邻车移除。`OUT` 必须是新目录，可设 `ABLATION_FRAMES` 先做较小的运行检查：

```bash
cd /home/cjm/OpenCOOD-main/cjmnet
RUN=/data/cjm/datasets/logs/fusion_detector_adaptation_20260927_103227
OUT="$RUN/scheme_a_top256_$(date +%Y%m%d_%H%M%S)"
nohup env RUN="$RUN" OUT="$OUT" ABLATION_FRAMES=90 \
  bash local_fusion_detector_adaptation/run_scheme_a.sh \
  > "${OUT}.launcher.log" 2>&1 < /dev/null &
echo "PID=$! OUT=$OUT"
```

完成后读取 `OUT/intervention_probe/scheme_a_intervention.md` 和 `.json`，以及四张 `distributions_*.svg`。探针会报告两个分数段的 AUC、PR-AUC、单个干预量的排序效应、按场景重抽样区间和事先固定的开发继续门槛。若只使用少量帧导致某折没有足够好坏框，会明确返回 `insufficient`，不能解释为机制失败。当前 validation 帧已多次用于开发；通过门槛后仍需另做固定候选池、相同输出框预算的完整 NMS/AP 验证。

## 完整的 score–geometry 假设检验（服务器）

`run_score_geometry_hypothesis.sh` 将候选导出、来源变化审计、逐邻车干预审计和最终后处理回放连成一次运行。新导出会保存每帧 GT 框、全部 top256 融合框四角、场景 ID 和候选自身检测特征。GT 仅用来定义训练标签和评价结果。

最终回放比较六组固定特征：候选自身；加简单来源一致性；加来源基础量；再分别加入融合前后分数—几何变化、逐邻车移除响应，以及二者一起使用。轻量逻辑回归按**场景**做折外预测：一个场景及其四种天气都只能出现在训练侧或评价侧的一边。每个方法使用相同 top256 候选、原旋转 NMS；每帧最终框数限制为原 `score>0.2` 流程的最终框数。脚本还统计相互重叠的低分好框与高分坏框谁最终留下，并同时输出原帧顺序 AP 与跨帧排序 AP。运行前会逐天气复现已保存的原流程 AP；复现失败时直接停止。

这套科研实验只在远程服务器运行。提取全部帧并逐邻车重算可能耗时很久，请后台运行：

上传本版代码后，可先在服务器做快速预检（只检查依赖、原 pilot 文件和 checkpoint；不进行模型推理）：

```bash
cd /home/cjm/OpenCOOD-main/cjmnet
RUN=/data/cjm/datasets/logs/fusion_detector_adaptation_20260927_103227 \
CHECK_ONLY=1 bash local_fusion_detector_adaptation/run_score_geometry_hypothesis.sh
```

预检通过后再后台运行完整实验：

```bash
cd /home/cjm/OpenCOOD-main/cjmnet
RUN=/data/cjm/datasets/logs/fusion_detector_adaptation_20260927_103227
HYP_OUT="$RUN/top256_score_geometry_$(date +%Y%m%d_%H%M%S)"
nohup env RUN="$RUN" HYP_OUT="$HYP_OUT" \
  bash local_fusion_detector_adaptation/run_score_geometry_hypothesis.sh \
  > "${HYP_OUT}.log" 2>&1 < /dev/null &
echo "PID=$! HYP_OUT=$HYP_OUT"
```

完成后看 `HYP_OUT/passive/top256_discriminability.md`、`HYP_OUT/intervention/scheme_a_intervention.md` 和 `HYP_OUT/replay/candidate_hypothesis_replay.md`，以及对应 JSON。若已用**本版代码**导出全部 90 帧的 `top_256`、`ABLATION_ARM=F`、`CANDIDATE_FEATURES=1`，并包含 `frame_targets.jsonl`，可给上述命令增加 `EXISTING_AUDIT=/绝对路径/到/该候选审计目录`，跳过耗时的重复推理；也可传入包含 `extraction/` 的运行总目录。旧日志缺少完整字段时下游会报错，不会产生貌似完整的结论。

读结果时按三步判断：先看同 anchor 的 Clean/天气配对中，分数下降但几何稳定、分数上升但几何未增强是否真实出现，并查看配对数量；再看低分和高分组中“加入变化量”的折外 AUC 相对候选自身、简单一致性、来源基础量的差值与场景区间；最后看固定框数、原 NMS 后的 AP70 增量和低分好框/高分坏框竞争去留。若只有候选自身特征已达到同等效果，协同变化没有提供额外辨别力。报告里的区间只覆盖固定折外预测的场景重抽样；这批 validation 场景用于探索，论文结论仍需独立场景复验。

## top256 真实 NMS 竞争范围验证（复用已有导出）

`candidate_nms_scope.py` 读取已有 `extraction/` 的候选行和每帧 GT，无需再次运行模型。它逐帧复现原始 `score>0.2` 检测和 top256 旋转 NMS，找到每个被压掉的候选及**第一个实际压掉它的框**。重点数出范围内“低分好框”（分数 ≤0.2、GT IoU ≥0.7）确实被范围内“高分坏框”（分数 >0.2、GT IoU <0.5）压掉的次数，以及这些候选可能覆盖多少原 top256 未命中的 GT。

脚本再做两个只用于判断空间大小的 GT 辅助排序：`targeted_gt_order` 只在上述真正发生的抑制组内换排名；`all_conflicts_gt_order` 在所有原始 NMS 抑制组内换排名，作为更宽的参照。每次都重新执行原旋转 NMS、范围过滤和原流程每帧输出框数预算。框的位置及用于计算 AP 的**原融合分数**保持不变。结果会给出实际新增和丢失的 GT、TP/FP、帧顺序和跨帧排序 AP30/50/70，以及预算填满率。启动时会检验原流程 AP 是否在 1e-6 内复现；失败会停止。

这两个重排使用 GT，属于离线反事实诊断，不能当成可部署方法或严格 AP 上界。它们也可能因重新运行 NMS 影响别的候选。`eligible_missed_gt` 只表示候选覆盖机会，不能直接当成找回的 TP。当前仍是已反复研究的 9 个 validation 场景。

将代码上传到远程服务器后，先做快速预检（不运行实验）：

```bash
cd /home/cjm/OpenCOOD-main/cjmnet
INPUT_ROOT=/data/cjm/datasets/logs/fusion_detector_adaptation_20260927_103227/top256_score_geometry_20260928_155556 \
CHECK_ONLY=1 sh local_fusion_detector_adaptation/run_candidate_nms_scope.sh
```

范围验证可能耗时较长，在服务器后台运行：

```bash
cd /home/cjm/OpenCOOD-main/cjmnet
INPUT_ROOT=/data/cjm/datasets/logs/fusion_detector_adaptation_20260927_103227/top256_score_geometry_20260928_155556
OUT="$INPUT_ROOT/nms_scope_$(date +%Y%m%d_%H%M%S)"
nohup env INPUT_ROOT="$INPUT_ROOT" OUT="$OUT" ARM=F \
  sh local_fusion_detector_adaptation/run_candidate_nms_scope.sh \
  > "${OUT}.log" 2>&1 < /dev/null &
echo "PID=$! OUT=$OUT"
```

完成后查看 `OUT/candidate_nms_scope.md`、`candidate_nms_scope.json` 和逐帧的 `candidate_nms_scope_frames.jsonl`。若真正的坏压好事件很少、覆盖的漏检 GT 很少，或受限重排也难以提高原帧顺序 AP70，这条 NMS 纠错方向在当前候选池内的可挖空间就很小。`ARM=F+D` 可单独复验另一分支，需指定不同的 `OUT`。

## 冻结 F 检测器的候选可靠性排序试验（服务器训练）

`candidate_ranker_extract.py` 从官方 **train** 场景导出 F 分支的 top256 候选、来源到融合框的分数/几何量，以及候选自身检测特征。默认随机选择最多 32 个训练场景、每场景最多 10 帧；Clean/Fog/Rain/Snow 使用同一帧索引。场景和帧只按固定种子选择。它复用原冻结 checkpoint，不训练或改写检测器，也不进行逐邻车移除前向。训练标签是候选与本帧 GT 的 BEV IoU；验证 GT 只用于评价。

`candidate_ranker_pilot.py` 训练四个**相同两层 MLP 结构、相同初始化种子**的排序头，只改变可见输入：候选自身、加简单来源一致性、加来源基础量、再加融合前后 score–geometry distortion。单车帧使用原分数。训练同时使用候选好坏标签和真实 BEV 重叠的好/坏框排序对；分数反转的低分好框/高分坏框对加权，但不只用那 77 个 validation 事件训练。默认 3 个种子、固定 6 轮，保存最后一轮；不按 validation 选 checkpoint。

回放统一使用保存的 top256 几何、原旋转 NMS、范围过滤，以及原 `score>0.2` 流程的**每帧输出框数上限**。预测可靠性只重排原 NMS 抑制组占据的排名位置，其他位置保持原顺序，再完整运行 NMS。主 AP 仍用原融合分数评价，只看排序带来的框去留；另报告用学习分数输出时的 AP。报告含 AP30/50/70 的原帧顺序和跨帧口径、同分数段好坏框 AUC、TP/FP、找回/丢失 GT、真实坏压好边排序正确数、每帧预算填满率，以及逐场景剔除的敏感性。运行前逐天气在 1e-6 内复现保存的原流程 AP。源码、冻结 checkpoint、训练导出和验证导出哈希保存在协议中。

门槛预先固定：distortion 模型平均帧顺序 AP70 相对**原流程和最好的同结构对照**，在至少两个恶劣天气各提高 ≥0.005；剔除任意一个场景后仍高于两种参照；Clean 相对两种参照的下降均不超过 0.001。这只是继续投入门槛。当前 9 个 validation 场景已反复使用，即使通过也要用新的独立场景复验，才能支持论文结论。

上传代码后在远程服务器预检（不导出、不训练）：

```bash
cd /home/cjm/OpenCOOD-main/cjmnet
CHECK_ONLY=1 sh local_fusion_detector_adaptation/run_candidate_ranker_pilot.sh
```

默认需导出最多约 32×10×4 帧，可能很久，请后台运行；训练导出可用 `TRAIN_ROOT` 复用，`OUT` 每次应为新目录：

```bash
cd /home/cjm/OpenCOOD-main/cjmnet
RUN=/data/cjm/datasets/logs/fusion_detector_adaptation_20260927_103227
EVAL_ROOT="$RUN/top256_score_geometry_20260928_155556/extraction"
TRAIN_ROOT="$RUN/candidate_ranker_train_20260929"
OUT="$RUN/candidate_ranker_pilot_$(date +%Y%m%d_%H%M%S)"
nohup env RUN="$RUN" EVAL_ROOT="$EVAL_ROOT" TRAIN_ROOT="$TRAIN_ROOT" OUT="$OUT" \
  sh local_fusion_detector_adaptation/run_candidate_ranker_pilot.sh \
  > "${OUT}.log" 2>&1 < /dev/null &
echo "PID=$! OUT=$OUT"
```

完成后看 `OUT/candidate_ranker_pilot.md`、`candidate_ranker_pilot.json`、`candidate_ranker_frames.jsonl`，模型在 `OUT/checkpoints/`。若之后有**新的、相同格式的独立评价导出**，使用冻结模型只评价，不重新训练：

```bash
NEW_EVAL_ROOT=/绝对路径/到/新的候选审计目录
NEW_OUT="$RUN/candidate_ranker_independent_eval_$(date +%Y%m%d_%H%M%S)"
nohup env MODE=evaluate-only EVAL_ROOT="$NEW_EVAL_ROOT" \
  CHECKPOINT_DIR="$OUT/checkpoints" OUT="$NEW_OUT" \
  sh local_fusion_detector_adaptation/run_candidate_ranker_pilot.sh \
  > "${NEW_OUT}.log" 2>&1 < /dev/null &
echo "PID=$! NEW_OUT=$NEW_OUT"
```

评价导出必须含相同 F checkpoint 的 top256 候选、每帧 GT 和候选特征缓存，并记录原流程 AP；不同模型或缺文件会拒绝运行。目前现成的评价导出仍属于已经用于方法开发的 validation，报告会标明这一限制。

## 倒挂框对的限定范围挽救验证（服务器训练）

`candidate_ranker_rescue.py` 复用上一试验已完成的官方 train 候选导出和冻结 validation 导出，不重新运行检测器。上一试验的 127742 个训练竞争框对中，仅 1253 个满足“低分好框压在高分坏框后”的倒挂条件。本次每个训练批次分别抽取相同数量的倒挂对和普通好坏竞争对；候选好坏分类损失只作辅助。四个同结构模型仍分别看候选自身、简单一致性、来源基础量、完整 distortion，以检查新增信息的贡献。

推理时只考虑**原始 NMS 中直接相互抑制**、高分框已在原 top256 结果中、低分框 `score<=0.2` 且高分框 `score>0.2` 的多来源框对。模型比较两框质量预测。每帧最多选择一对，将两框在原排序中的位置交换，再运行原 NMS；其余候选顺序不变。这个决策不读取 GT。0.70、0.80、0.90、0.95 四个阈值全部输出，**0.90 是固定的主阈值**，不得在这批验证场景上另选最好阈值作正式结论。主 AP 用原融合分数和原帧顺序口径；报告同时给新增/丢失 GT、交换数、真实倒挂交换数、反向质量交换数、可参与决策的真实边数量、逐场景剔除结果。训练固定 3 个种子、6 轮，保存最后一轮，不按验证集挑模型。

继续投入门槛也预先固定：在至少两个恶劣天气，distortion 的平均主 AP70 比原流程及最佳同结构对照都高至少 0.005，净找回 GT 为正，剔除任意一个场景后仍领先；Clean 相对两个参照的下降都不超过 0.001。当前验证场景已参与方法开发，门槛通过仅表示值得用独立新场景复验。

上传这三个新文件及本 README 后，先在远程服务器预检：

```bash
cd /home/cjm/OpenCOOD-main/cjmnet
CHECK_ONLY=1 sh local_fusion_detector_adaptation/run_candidate_ranker_rescue.sh
```

训练及回放可能耗时较长，在远程服务器后台运行：

```bash
cd /home/cjm/OpenCOOD-main/cjmnet
RUN=/data/cjm/datasets/logs/fusion_detector_adaptation_20260927_103227
TRAIN_ROOT="$RUN/candidate_ranker_train_20260929"
EVAL_ROOT="$RUN/top256_score_geometry_20260928_155556/extraction"
OUT="$RUN/candidate_ranker_rescue_$(date +%Y%m%d_%H%M%S)"
nohup env RUN="$RUN" TRAIN_ROOT="$TRAIN_ROOT" EVAL_ROOT="$EVAL_ROOT" OUT="$OUT" \
  sh local_fusion_detector_adaptation/run_candidate_ranker_rescue.sh \
  > "${OUT}.log" 2>&1 < /dev/null &
echo "PID=$! OUT=$OUT"
```

完成后查看 `OUT/candidate_ranker_rescue.md`、`candidate_ranker_rescue.json`、`candidate_ranker_rescue_frames.jsonl` 和逐冲突对的 `candidate_ranker_rescue_edges.jsonl`；模型及输入哈希记录在 `OUT/checkpoints/`。边日志保存全部符合推理范围的候选对及评价用质量标签，能区分漏改的真实倒挂对和误改的普通对。若后来导出了同一冻结检测器的独立新场景，可使用 `MODE=evaluate-only EVAL_ROOT=... CHECKPOINT_DIR="$OUT/checkpoints" OUT=...` 调用同一启动脚本，只评价保存的模型。

## S0 有序质量 / S6 冻结头稳定性验证（服务器运行）

`candidate_s0_s6.py` 复用已导出的官方 train top256 候选和旧 validation top256 候选。S0 将 GT IoU≥0.3/0.5/0.7 作为三个有序训练标签，对照同宽度网络的 `IoU≥0.7` 二分类与直接 IoU 回归。S6 对缓存的融合后检测特征施加固定的 10% 特征 dropout（每候选 8 次），分别计算冻结分类头输出与框几何的变化。程序先要求原分类 logit 和回归 delta 与缓存记录的最大绝对误差均不超过 `1e-4`，不符合就停止。S6 对照包括仅分类变化、仅几何变化、两者合用，以及同天气/同原分数段内打乱稳定性特征；候选自身输入始终保留。

所有方法训练 3 个种子、固定 6 轮、保存最后一轮；只对原 NMS 抑制组重排，再运行同一 NMS 和原每帧输出预算。主 AP 使用原融合分数和帧顺序口径。报告含按原分数≤0.2/>0.2 分层的好坏框 AUC、直接抑制边排序、AP30/50/70、TP/FP、新增/丢失 GT、逐场景剔除结果和预定继续门槛。缓存稳定性是原候选特征的确定性变换，不能称为新增信息。旧 validation 已反复用于开发；即使过门槛，也只决定是否用新的独立场景复验。

上传两个新脚本到服务器后，先做不训练的预检：

```bash
cd /home/cjm/OpenCOOD-main/cjmnet
CHECK_ONLY=1 sh local_fusion_detector_adaptation/run_candidate_s0_s6.sh
```

完整实验在远程服务器后台运行：

```bash
cd /home/cjm/OpenCOOD-main/cjmnet
RUN=/data/cjm/datasets/logs/fusion_detector_adaptation_20260927_103227
TRAIN_ROOT="$RUN/candidate_ranker_train_20260929"
EVAL_ROOT="$RUN/top256_score_geometry_20260928_155556/extraction"
OUT="$RUN/candidate_s0_s6_$(date +%Y%m%d_%H%M%S)"
nohup env RUN="$RUN" TRAIN_ROOT="$TRAIN_ROOT" EVAL_ROOT="$EVAL_ROOT" OUT="$OUT" \
  sh local_fusion_detector_adaptation/run_candidate_s0_s6.sh \
  > "${OUT}.log" 2>&1 < /dev/null &
echo "PID=$! OUT=$OUT"
```

完成后查看 `OUT/candidate_s0_s6.md`、`.json` 和 `OUT/checkpoints/`。启动脚本要求训练候选导出已存在；缺失时应先按本 README 上一节的官方 train 导出流程生成，不能把 validation 当训练集。
