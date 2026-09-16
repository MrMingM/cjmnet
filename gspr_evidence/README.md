# GSPR 证据缺口匹配：第一版研究实验

## 2026-09-14 更正：与原实验比较的正式测试入口

此前 1980 帧结果是开发 validation 和在线模拟天气，不能与历史 OPV2V-W test AP 比较。正式测试现在默认使用 OPV2V clean **test** 和已有 OPV2V-W **fog/rain/snow test**，关闭在线天气增强，按原非全局排序方式输出 `eval.yaml`。

同步整个更新后的 `gspr_evidence/` 后直接执行（复用已训练模型，无需重训/缓存）：

```bash
cd /home/cjm/OpenCOOD-main/cjmnet
export ROCR_VISIBLE_DEVICES=0
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
export RUN=/data/cjm/datasets/logs/gspr_evidence_seed20260913_20260913_094932
bash gspr_evidence/run_benchmark.sh
```

新结果在 `${RUN}_historical_test_时间戳/results.md`；原结果不覆盖。对三组模型均运行完整四天气 none/a0b0/protocol/learned/full，并增加 `original_full`（原始 AttFuse＋GSPR）。两条全通信路径在相同输入上逐帧核对并分别计算 AP。`original_full` 的字节栏沿用 full 的逻辑载荷核算，不是额外发送一份数据。

固定数据路径与协议详见根目录 `AI_CONTEXT.md` 第 11 节。正式入口不需要在线雾模拟的查找表。已有 checkpoint 哈希校验保留，训练配置本身不被改写。

**以下原训练/验证流程属于开发实验。`run_experiment.sh` 不负责正式 test 报告。** `evaluate` 默认已改为 benchmark，旧开发验证命令须增加 `--evaluation-protocol development`。

本包实现“缺口描述 → 退化与实际补充监督 → 邻车匹配 → BEV 通信检测”。GSPR、PointPillar 主干、检测头和缺失块掩码 AttFuse 保持冻结。新增风险头 11,940 参数、收益头 11,521 参数，总计 23,461 参数。

## 实际流程

1. 各车独立编码。对已有点使用 `point_evidence` 恢复原始可靠概率，保留 `point_uncertainty`；不累加经过 floor 映射的权重冒充可靠点数。
2. 区域描述保留可靠概率、u、有效支持、是否有点、4 个高度层的支持以及 4 个高度的射线穿越采样，另加前景置信度。没有回波的区域仍可发起请求。
3. 自车风险头输出漏检风险、背景误检风险、定位风险、天气后相对晴天可恢复的误差。它只读取本车 BEV 和区域描述。GT 和晴天对照仅用于训练标签。
4. 自车选取 256 个请求块，将 17 个字段逐个量化成 uint8，连同块索引与位姿发送。当前默认请求每个邻车 5,460 字节；按每邻车单播计费，未采用免费广播假设。
5. 邻车读取**解码后的请求**与自身 BEV、区域描述、相对位姿。收益头明确计算“邻车支持 × 自车缺少的支持”等匹配特征，输出有符号收益分数。
6. 按总计 256 KiB/场景预算分配邻车配额，选择三尺度 BEV 块。实际序列化/解码后再送入原掩码 AttFuse。头部、请求字段、索引、位姿及三个尺度全部计费。

几何部分借鉴 D-Map 的传感器坐标角度深度投影。BEV 已经投到 ego 坐标；只对射线查询使用真实传感器位姿，不再变换 BEV。覆盖统计来自体素保留点，有每柱点数截断；射线图使用该车裁剪后的点云。射线结果只是固定角分辨率下、区域高度中心的穿越采样，**不能解释成整个区域确定空闲，也不是完整 D-Map 占据建图**。噪声近回波可能阻断射线，交由可靠性、u 和任务监督共同判断。

## 训练标签分别回答两个问题

风险监督覆盖全部标注区域，包括没有预测框的 GT 位置。前景/背景分别归一化；分类、背景误报、定位分别输出。它是 anchor 层面的检测风险代理，不等同于经过 NMS 的目标漏检概率。晴天/天气对照生成额外恢复目标，不直接当作通信收益。

收益监督实际重放邻车 BEV：按新协议的载荷配额构造 A0B0 上下文，每次从某邻车撤去一个块，在**同一上下文**中依次插入不同候选块；用冻结检测器的原任务损失差生成正/负标签。候选包含原规则块、较高规则分块及均匀探索块，每邻车每帧最多 8 个。替换方案的字节数相同，教师不修改点权重。

教师遍历所有训练/验证帧；“8 个”限制的是**每帧候选重放数量**，不是只跑 8 帧。教师不穷举数千个区域。标签缓存保存小尺度语义和区域统计，不保存完整三尺度 BEV。全部候选共享该组的上下文，原始上下文索引和候选标签也被保存。

收益是固定教师上下文分布下的监督目标。推理时邻车不知道其他邻车的内部特征，最终 top-k 同时换多个块；因此预测值应解释为条件期望近似，不能声称精确预测最终组合的边际收益。缓存内排序好但最终 AP 差时，这种上下文差异/冗余是应检查的具体原因。

先训练风险头，再冻结风险头训练收益头。直接监督绕开之前“通过硬 top-k 间接学习却没有换块”的问题；是否真的选得更好仍须完整验证。Learning Loss 的排序思想用于候选排序辅助损失，动态特征选择的任务效用与掩码思想用于实际候选重放。详见 `SOURCE_ADAPTATION.md`。

## 数据量与对照

- `train_scene_indices: null`、`validation_scene_indices: null`、`frame_stride: 1`，不设置每轮步数上限。程序拒绝场景抽样/步长抽样配置，并检查训练和验证目录及场景名不重叠。
- 每阶段默认 8 个完整 epoch，按完整验证缓存选权重。准备缓存时，每个场景帧都使用同帧晴天/物理天气配对；固定一种混合天气随机实现供模型公平共用。
- 正式 AP 评测分别遍历**完整验证集的 clean、fog、rain、snow**。全部模式共用一次编码结果；同请求规则与学习模式共用同一个真实请求包。
- 第一个入口中的最多 8 帧仅作连通检查，不产生 AP 结论。CPU 合成数据单元测试也不属于效果验证。
- 固定 1 个种子做第一轮完整实验。最终论文仍需多种子和独立测试集；完整验证划分不等于跨数据集泛化证据。

| 对照 | 用途 |
|---|---|
| `none` | 自车无通信 |
| `a0b0` | 原始全网格需求规则、原字节协议，固定总预算 |
| `protocol` | 新请求格式、风险选出的相同请求区域，响应按旧规则排序 |
| `matching` / `learned` | 区域证据匹配的有监督收益头 |
| `concat` / `learned` | 相同参数量、相同原始信息和标签，用普通拼接替换显式匹配，重新训练 |
| `no_u` / `learned` | 请求/响应中 u 通道均置零，风险与收益头重新训练；保留冻结 GSPR 本身 |
| `full` | 不受 256 KiB 约束的全通信参考，不参与同预算优劣结论 |

`no_u` 检查通信选择器使用 u 的贡献，不能解释成去掉了冻结前端内部的不确定性机制。`matching` 与 `concat` 使用同样的风险训练流程；`no_u` 允许改变请求区域，因为其风险头也需适应缺少 u 的输入。

## 服务器一键运行

同步整个 `gspr_evidence/` 到服务器 `/home/cjm/OpenCOOD-main/cjmnet/`。依赖服务器已有的 `gspr_communication/`、`gspr_review/resources.py`、`gspr_review/runtime.py` 及冻结 GSPR/OpenCOOD。三个外部源码仓库不需要编译或安装。

确认选用空闲卡后执行：

```bash
cd /home/cjm/OpenCOOD-main/cjmnet
export ROCR_VISIBLE_DEVICES=0
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
bash gspr_evidence/run_experiment.sh
```

脚本默认固定前端：

```text
/data/cjm/datasets/logs/gspr_joint_full_v1_seed20260907_20260907_161155/config.yaml
/data/cjm/datasets/logs/gspr_joint_full_v1_seed20260907_20260907_161155/net_best_validation.pth
```

如需使用另一份已固定前端，在启动前设置 `FRONTEND_CONFIG`、`FRONTEND_CHECKPOINT`。雾表默认使用 `TripleMixer-main/tools/fog_sim/integral_lookup_tables_seg_light_0.008beta/original`，可通过 `FOG_LOOKUP_DIR` 指定实际目录。

顺序是：单元测试 → 真实连通检查 → 完整 train/validation 教师缓存 → 三种小网络训练 → 每种四类天气完整验证 → `results.md` 汇总。输出新目录 `/data/cjm/datasets/logs/gspr_evidence_seed20260913_时间戳/`，保留 `console.log`。

完整教师重放比以前的 54 帧评测耗时明显更长，不能根据旧实验估算。缓存阶段输出实际总帧数、处理速度、剩余小时和估算磁盘量；评测也输出 ETA。三个模型复用同一缓存。第一次看到 ETA 后可以按服务器资源安排执行，代码没有自动缩小样本规模。

## 断点恢复

缓存逐帧原子保存，中断后重用已完成文件。先恢复相同环境变量：

```bash
cd /home/cjm/OpenCOOD-main/cjmnet
export PYTHONPATH=/home/cjm/OpenCOOD-main/cjmnet:/home/cjm/OpenCOOD-main
export ROCR_VISIBLE_DEVICES=0
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
export PYTHONHASHSEED=20260913
export FRONTEND_CONFIG=/data/cjm/datasets/logs/gspr_joint_full_v1_seed20260907_20260907_161155/config.yaml
export FRONTEND_CHECKPOINT=/data/cjm/datasets/logs/gspr_joint_full_v1_seed20260907_20260907_161155/net_best_validation.pth
PY=/home/cjm/miniconda3/envs/opencood/bin/python
# 将这一行改成脚本开头实际打印的 RUN。
RUN=/data/cjm/datasets/logs/gspr_evidence_seed20260913_实际时间戳

"$PY" -m gspr_evidence.prepare --split train --resume \
  --config "$RUN/cache_train/experiment.yaml" \
  --frontend-config "$FRONTEND_CONFIG" --frontend-checkpoint "$FRONTEND_CHECKPOINT" \
  --output-dir "$RUN/cache_train"
# 若中断在 validation 缓存阶段，把上面的三处 train 改为 validation。

"$PY" -m gspr_evidence.train --variant matching --resume \
  --train-cache "$RUN/cache_train" --validation-cache "$RUN/cache_validation" \
  --output-dir "$RUN/matching"
```

训练在完整 epoch 末保存 `last.pth` 和优化器，恢复时重做未完成的 epoch。`risk_best.pth` / `gain_best.pth` 是验证选择的权重，只包含新增头与优化器。配置、前端哈希和关键源码哈希不一致时拒绝复用。

评测中断需用一个新的输出目录重跑该天气/模型，不把部分帧文件冒充完整结果；单独评测命令：

```bash
"$PY" -m gspr_evidence.evaluate \
  --config "$RUN/matching/experiment.yaml" \
  --frontend-config "$FRONTEND_CONFIG" --frontend-checkpoint "$FRONTEND_CHECKPOINT" \
  --heads "$RUN/matching/gain_best.pth" --weather fog \
  --modes none a0b0 protocol learned full \
  --output-dir "$RUN/eval_matching_fog_retry"
```

## 如何判断失败发生在哪里

1. `history.jsonl` 中 `oracle_over_rule` 很小：当前抽样候选/冻结融合器提供的可学习替换收益很少；这不是全候选的理论上限。
2. `oracle_over_rule` 明显为正，但 `predicted_over_rule` 很小、`candidate_regret` 大：收益预测/排序没学好，优先检查响应模块。
3. 缓存排序有效，但正式 `replacement_fraction` 为 0：最终查询域/配额/排序未改变实际内容。
4. 已换块但 AP 无收益：检查完整 `frames.jsonl` 中风险覆盖、补回/丢失目标以及组合冗余；候选损失下降不保证 AP 上升。
5. `protocol` 低于 `a0b0`：请求区域或新增元数据载荷开销已经有损失，需要与响应头效果分开解释。
6. `matching` 不优于 `concat`，或 `no_u` 没有下降：本轮尚不支持“结构匹配/u 是关键贡献”的主张。

AP 按全局置信度排序，以 6 位小数汇总。旧实验若未采用 global sort，应使用本轮重跑的 A0B0 对照，不直接与历史数值相减。每帧记录场景 ID、各 IoU 的 TP/FP/score，可按场景继续分析。计时分开记录共享编码、请求、响应选择/序列化/融合，不包含数据读取和后处理。

## 当前验证状态

本地 CPU 新旧共 20 项测试通过，含协议/张量、合成缓存两阶段训练及恢复；Python 编译、Bash 语法与 14 个冻结源码文件哈希检查通过。真实服务器数据连通、完整缓存生成、GPU 训练与 AP 必须通过上述入口执行后才能确认，目前没有新增 AP 提升结论。
