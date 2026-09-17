# Q-A Stage-2.5 / Stage-3：融合失败诊断

## 简单版

先查清楚：**邻车自己能检出的目标，为什么到全融合的最终输出里不见了？**

- **Stage-2.5**：只读旧日志，判断失败是否集中在某些场景、距离。每组都有分母。
- **Stage-3A**：逐框追踪，定位正确框最后在哪个处理环节消失。
- **Stage-3B**：人工阅读 A 后，另行启动。固定特征和原 AttFuse，只改变参与融合的车辆组合，查是否存在恢复机会。
- **Stage-3C**：仅保留后续计划，本次不实现新网络。

Stage-3A **不能回答根因是什么**。例如融合改变特征导致分数下降，最后表现为分数阈值删框，但阈值不一定是根因。
Stage-3B **不能回答融合的理论上限是什么**，也不产生部署方法或 AP 提升结论。

**Stage-3B 只能在人工阅读完 Stage-3A 后主动启动。** `run_stage3a.sh` 不会执行 B。

## 数据和候选定义

仅使用 `/data/scd/datasets/opv2v_official_data_dumping/validate` 的 development validation，
以及 Stage-1/2 使用的在线 fog/rain/snow。无 OPV2V-W test，无训练，无参数调优。

候选直接来自 Stage-2 的 `peer/<weather>/peer_targets.jsonl`：
`source_valid_full_miss == true`。不重新使用 N_eff、coverage 或 strong/weak 筛选。
读取时重新核对该标志与各来源实际检测状态，检查重复、缺失和汇总计数。
预期全量候选为 Fog 162、Rain 63、Snow 988；代码根据输入日志计算，不把这些数字硬编码成通过条件。

`peer-alone` 定义保持 Stage-2 不变：取已对齐 ego 坐标系的单个 peer 的多尺度融合前特征，
完全绕过 AttFuse，不加入 ego，经过原冻结 deblocks、分类/回归头、原后处理，
按原稳定分数顺序的一对一贪心匹配，对同一个 GT 达到平面多边形 IoU ≥ 0.7。
IoU 表示两个框的重叠程度；此处不是三维体积 IoU。

统计单位始终为 **target-frame occurrence（目标在某一帧中的一次出现）**，不是独立车辆。
Stage-2.5 的分母为 Stage-2 审查子集中 `any_peer_alone_detected == true` 的次数，
该子集本身限定为 clean ego 检出、weather ego 漏检。不能把条件失败率解释成全数据集漏检率。
独立帧数、失败帧数另外统计。距离分桶复用 `qa_observation_diagnostic.metrics.distance_bin`。
JSON 同时包含每个场景内的距离分布。

## Stage-3A 的工作流程

1. 验证冻结文件、旧日志、配置、checkpoint SHA、源码快照和原始完整帧队列。
2. 用原加载器从队列开头重放天气；非候选帧仍经过数据生成，只跳过 GPU 检测。
3. 候选帧编码后，复现 ego、full 和所有 peer 的旧检测结果；核对 GT 中心、来源顺序、
   每来源感知统计、匹配分数/IoU、最佳后处理框指标和帧级 FP 数。
4. 对 full 和曾单独检出目标的 peer 逐框追踪。多个候选共用帧级结果。
5. 每个追踪分支都与原 `ds.post_process()` 核对最终框数、顺序、分数、几何及匹配状态。
   任何不一致立即报错，不生成完成报告。
6. 保存逐目标 JSONL、逐框 NPZ，最后再次验证冻结文件和输入日志未被修改。

真实流程为：

```text
psm → sigmoid score
rm + anchors → decoded boxes
→ score > threshold
→ large-box / abnormal-z filtering
→ 原 NMS 内部的最高分 1000 框截断
→ rotated NMS（框间 IoU > nms_threshold 才抑制）
→ range filtering
→ Stage-2 的最终 GT 贪心匹配（IoU ≥ 0.7）
```

candidate ID 是原 anchor 展平编号。所有阶段复用同一编号，不重新排序编号。
score 和 GT IoU 始终绑定同一个框。每阶段记录最佳 IoU 及该框分数、
达到 IoU70 的框数，以及这些合格框中的最高分和对应编号。

失败分类：

| 分类 | 通俗解释 |
|---|---|
| `no_iou70_after_decode` | 解码后所有框都达不到该目标的 IoU70 |
| `score_filtered` | 原本有正确框，但全部被分数阈值删除 |
| `geometry_filtered` | 剩余正确框全部被大框或异常高度规则删除 |
| `nms_topk_filtered` | 剩余正确框都没进入 NMS 的最高分 1000 框 |
| `nms_suppressed` | 剩余正确框被实际 NMS 选中的框抑制 |
| `range_filtered` | NMS 后的正确框全部超出范围 |
| `matching_competition` | 最终框仍达 IoU70，但被匹配给其他 GT |
| `other` | 为未覆盖情况保留；当前有限正常流程不应产生 |

这些都是**最后可观察到的失败阶段，不是根因标签**。
NMS 记录真实首个 suppressor（实际删掉它的框）、双方编号/分数/GT IoU、框间 IoU，
以及 suppressor 最接近哪个 GT。top-1000 截断不虚构 suppressor。
最终匹配竞争记录实际被分配的 GT。

几何变化同时保存两种对照：

- **同一 anchor**：peer 最终匹配框与 full 同编号框，比较中心、尺寸、方向、IoU、分数变化。
- **不同分支的最佳框**：额外描述 full 最佳 IoU 框；明确它不一定与 peer 框对应。

不能把 `no_iou70_after_decode` 简单命名为“位置错了”，因为尺寸、方向也影响 IoU。

## Stage-3B 的工作流程（独立、手动）

必须提供已完成的 Stage-3A 输出。仅 smoke 的 A 不能用于全量 B。
检查 A/B 的输入逐帧 SHA、实现、配置、环境版本一致。

同一帧编码一次，全部组合复用该组固定特征。另做原模型/来源复现检查，
但绝不为每个目标或每个组合重新编码。始终保留 ego，最多 5 辆车，枚举最多 16 种整车组合。
原 AttFuse、检测头、后处理完全不变；使用完整车辆特征，不做区域选择或新通信策略。

分别报告：

1. **逐目标恢复机会**：哪些组合能补回该目标。不同目标的恢复结果不能直接合并成系统输出。
2. **统一帧级结果**：每帧只选一个车辆组合。预先固定规则：
   补回候选数最多 → 丢失 full 原有正确检出最少 → 新增 FP 最少 → 车辆编号字典序。
   同时保存所有组合和 Pareto 结果，即没有另一组合在这三项上全面更好的选择。

每组合记录补回哪些候选、丢失哪些原 TP、总匹配 GT、帧 FP、新增 FP 和 FP 数净变化。
**新增 FP** 的固定定义：将组合产生的最终误检框，按分数稳定降序，
与 full 的最终误检框做 IoU≥0.7 的一对一贪心匹配；未匹配的组合误检框算新增。
这表示误检框的几何位置未延续，不是物理对象身份识别。保留框坐标、分数和对应关系供复查。
净变化另记为 `subset FP - full FP`，不用于冒充新增数。

如果所有组合都失败，只能说这个有限的整车组合空间无法恢复，不能排除区域选择、
修改融合算子或特征修正的可能。
`peer-alone ✓、ego+peer ✗` 只能说明原融合算子加入 ego 后未保持单源检测能力，
不能据此断言 ego query 压制 peer。
`ego+peer ✓、ego+all ✗` 是额外来源干扰的候选现象，也不能直接给某车贴 harmful source 标签。

Stage-3C 若值得继续，再设计固定 values 仅改变 query/weights，或固定 weights 仅改变 values 的单因素干预。
本次不实现这些干预，不实现学习式融合网络。

## 一致性和复现边界

- 实验前后复用 `verify_frozen()`。原 GSPR、AttFuse、detector、postprocessor 均未修改。
- `stage3_sources.json` 是本次审阅时的历史依赖源码快照，SHA 仅标准化 CRLF→LF，
  适应 Windows/Linux 文本换行；checkpoint 和历史 protocol SHA 检查仍用原始字节。
- **旧 Stage-1/2 没保存逐帧天气输入哈希，也没保存实验 YAML 的 SHA。**
  不能从现有产物证明历史天气输入逐字节相同，也不能独立证明当时所有依赖的版本。
  当前严格使用已审阅配置、完整采样队列，复现旧感知统计与检测结果；发现不同即停止。
  这属于有证据的重放一致性检查，不把它表述为已证明历史输入逐字节一致。
- Stage-3 新增输入 SHA，覆盖处理后点云、原始天气点云、变换、record_len、GT，
  B 与 A 必须一致。protocol 保存 Python/Torch/NumPy/Shapely 版本和有效阈值。
- 原始模型与重放检测头输出容差为 `atol=rtol=2e-4`，继承 Stage-2；
  旧检测/感知浮点指标容差 `2e-5`；整数统计必须相等；
  tracer 最终框容差 `1e-6`，分数 `atol=1e-7, rtol=1e-6`。
  数量、顺序、匹配状态、候选队列、NMS keep 顺序不得变化。
- 如果 SHA、天气重放或 tracer 失败，保留报错日志供定位。不要修改旧日志、删除断言、
  放宽阈值或重生成快照来掩盖差异。修复原因后用新的 RUN_NAME 运行。

所有输出只能在 `/data/cjm/datasets/logs/` 下的新目录，不覆盖旧结果。
每个完整运行会先做独立的三天气 smoke，写到 `preflight/`；smoke 失败则停止全量运行。
独立 `SMOKE=2` 只跑每种天气的前两个候选帧，不是前两个数据帧。
为保持天气随机序列，找到这些候选前仍须从队列起点加载数据，因此 smoke 也可能耗时。
Smoke 只用于检查，不可作科研结论。

## 服务器命令

先将本次新增文件同步到 `/home/cjm/OpenCOOD-main/cjmnet/gspr_evidence/`。
本次本地修改不会自动提交或推送 GitHub。使用原 opencood 环境，不升级服务器依赖。

### Stage-3A smoke（后台，整段复制）

```bash
cd /home/cjm/OpenCOOD-main/cjmnet
RUN_NAME=qa_stage3a_smoke_$(date +%Y%m%d_%H%M%S)
nohup env ROCR_VISIBLE_DEVICES=3 STAGE2_ROOT=/data/cjm/datasets/logs/qa_evidence_validity_20260917_112301 RUN_NAME="$RUN_NAME" SMOKE=2 bash gspr_evidence/run_stage3a.sh > "/data/cjm/datasets/logs/${RUN_NAME}_launcher.log" 2>&1 &
echo "PID=$!  RUN_NAME=$RUN_NAME"
tail -f "/data/cjm/datasets/logs/${RUN_NAME}_launcher.log"
```

`Ctrl+C` 只退出 tail，不停止后台实验。检查末尾 `DONE` 和 `stage3a_report.md` 的 SMOKE 标记。
脚本清除 HIP_VISIBLE_DEVICES/CUDA_VISIBLE_DEVICES，仅使用 ROCR_VISIBLE_DEVICES；模型内部仍为 cuda:0。

### Stage-3A 全量（后台，整段复制）

```bash
cd /home/cjm/OpenCOOD-main/cjmnet
RUN_NAME=qa_stage3a_$(date +%Y%m%d_%H%M%S)
nohup env ROCR_VISIBLE_DEVICES=3 STAGE2_ROOT=/data/cjm/datasets/logs/qa_evidence_validity_20260917_112301 RUN_NAME="$RUN_NAME" SMOKE=0 bash gspr_evidence/run_stage3a.sh > "/data/cjm/datasets/logs/${RUN_NAME}_launcher.log" 2>&1 &
echo "PID=$!  RUN_NAME=$RUN_NAME"
tail -f "/data/cjm/datasets/logs/${RUN_NAME}_launcher.log"
```

也可以看真正输出目录的日志：

```bash
tail -f "/data/cjm/datasets/logs/$RUN_NAME/console.log"
```

### Stage-3B（仅人工阅读 A 后执行）

把下面 Stage-3A 路径替换为实际完整运行目录。可以先把 `SMOKE=0` 改为 `SMOKE=2`。

```bash
cd /home/cjm/OpenCOOD-main/cjmnet
RUN_NAME=qa_stage3b_$(date +%Y%m%d_%H%M%S)
nohup env ROCR_VISIBLE_DEVICES=3 STAGE2_ROOT=/data/cjm/datasets/logs/qa_evidence_validity_20260917_112301 STAGE3A_ROOT=/data/cjm/datasets/logs/qa_stage3a_替换为实际时间戳 RUN_NAME="$RUN_NAME" SMOKE=0 bash gspr_evidence/run_stage3b.sh > "/data/cjm/datasets/logs/${RUN_NAME}_launcher.log" 2>&1 &
echo "PID=$!  RUN_NAME=$RUN_NAME"
tail -f "/data/cjm/datasets/logs/${RUN_NAME}_launcher.log"
```

可覆盖的变量：`PY`、`FRONTEND_CONFIG`、`FRONTEND_CHECKPOINT`、`STAGE2_ROOT`、`RUN_NAME`、`SMOKE`、`ROCR_VISIBLE_DEVICES`。
配置默认 `qa_observation_diagnostic/experiment.yaml`；虽然保留 `CONFIG` 参数，但内容必须匹配本次审阅快照。
默认 checkpoint/config 来自 `gspr_joint_full_v1_seed20260907_20260907_161155`，并与 Stage-1 的 SHA 核对。

## 产物与回传

```text
qa_stage3a_<timestamp>/
  console.log
  protocol.json
  stage3a_results.json
  stage3a_report.md
  stage2_5/scene_distance_report.{json,md}
  preflight/{fog,rain,snow}/...  # 全量启动时自动生成
  {fog,rain,snow}/
    protocol.json
    summary.json
    targets.jsonl
    proposals/<frame>_{full,peer_<index>}.npz
```

NPZ 按帧/分支保存一次：全部 anchor ID、anchor shape、分类 logit、回归 delta、score、
解码框和 ego corners；候选 GT 列的 IoU；所有阶段的保留 ID；suppressor ID 和框间 IoU；
最终框的 GT 分配。`target_indices` 指明 IoU 列对应的 GT；`final_assigned_gt` 对齐 `range_ids`。
`suppressor_id=-1` 表示无 NMS suppressor，不能直接解释为框最终保留。
NPZ 体积可能较大，全部留在外部日志目录，不写代码仓库。

**A 完成后先回传：**

1. `stage3a_report.md`、`stage3a_results.json`、根目录 `protocol.json`；
2. `stage2_5/scene_distance_report.json`；
3. 三种天气各自的 `summary.json`、`protocol.json`、`targets.jsonl`；
4. `console.log`。

不用先回传全部 NPZ。需要核查具体案例时，再按 targets.jsonl 中的路径取对应文件。
B 另外输出 `stage3b_results.json`、`stage3b_report.md` 和每天气 `frames.jsonl`。
未生成完整报告不能当成完成实验。

## 文件与验证

| 文件 | 作用 |
|---|---|
| `stage3_analysis.py` | 分母统计、稳定编号、NMS 追踪、匹配和组合选择的纯函数 |
| `stage3_trace.py` | 原后处理逐框重放、最终一致性断言、NPZ 和几何对照 |
| `stage3_runtime.py` | 候选/配置/旧结果校验、完整天气重放、输入指纹与冻结保护 |
| `stage3_sources.json` | 审阅过的历史代码与配置指纹 |
| `stage3a.py` | Stage-3A 实验入口 |
| `stage3b.py` | 手动 Stage-3B 实验入口 |
| `stage3_report.py` | CPU Stage-2.5 和 A/B 完成报告 |
| `test_stage3.py` | 纯函数测试和可用环境下的原后处理集成测试 |
| `run_stage3a.sh`、`run_stage3b.sh` | 独立启动入口 |
| `stage3_launch.sh` | 公共环境、测试、smoke、天气循环和报告生成 |
| `STAGE3.md` | 实验定义、边界、命令和输出说明 |

单元测试：`python -m unittest gspr_evidence.test_stage3 -v`。
原后处理集成测试缺少 OpenCOOD 依赖时明确显示 skipped，不以替代实现假装通过。
服务器真实模型 smoke 是必要验证；本地单元测试不能证明 GPU/在线天气推理已通过。
