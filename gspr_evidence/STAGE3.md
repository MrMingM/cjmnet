# Stage-3：H-A6 融合失败因果诊断

> 目标：回答“某个邻车单独能够正确检测，但 full fusion 后目标丢失”这一已被 Stage-2 支持的现象，**最后表现在哪个处理环节失效，以及固定原融合器时仅改变参与来源是否能够恢复**。
>
> 本阶段仍是 **development diagnostic**。只使用 OPV2V validation + online fog/rain/snow，不使用 OPV2V-W test 做假设筛选，不训练新网络，不修改冻结 GSPR/AttFuse。

## 1. Stage-3 的边界

Stage-2 已经确认存在：

```text
peer-alone detected + full fusion miss
```

这说明同一 frozen detector 可以从某个单源特征中得到正确框，但 full fusion 的最终输出没有保住该检测能力。

Stage-3 不允许直接把它写成：

- AttFuse attention 已被证明是根因；
- ego query 已被证明压制 peer；
- 这些 target-frame 数量等价于独立车辆数；
- 这些候选数量可以直接换算成 AP 上限。

所有统计中的 target 数均为 **target-frame occurrences**。

---

# 2. Stage-2.5：先补分母

文件：

```text
gspr_evidence/stage3_stratify.py
```

不跑模型，只读取 Stage-1/Stage-2 JSON。

对每个 scene / distance bin 报告：

```text
denominator = any peer-alone detected
numerator   = any peer-alone detected + full miss
failure rate = numerator / denominator
```

同时报告：

- source-valid target-frame occurrences；
- failure target-frame occurrences；
- source-valid unique frames；
- failure unique frames。

这样避免把“某场景候选本来就很多”误读成“该场景 fusion 特别容易失败”。

---

# 3. Stage-3A：Detection-path disappearance audit

文件：

```text
gspr_evidence/stage3a.py
```

默认必须先执行 Stage-3A。

候选严格来自 Stage-2：

```text
any peer-alone detected + full fusion miss
```

每个候选重新使用相同：

- validation frame queue；
- online weather 随机协议；
- frontend checkpoint；
- frozen encoding；
- full AttFuse。

程序会强制检查：

1. 当前 frame queue 与 Stage-2 完全一致；
2. frontend SHA256 与 Stage-2 一致；
3. Stage-1 root 与 Stage-2 一致；
4. full final miss 必须复现；
5. 选中的 peer-alone detection 必须复现；
6. 手工逐阶段重放的最终 post-process 输出必须与原 `ds.post_process()` 一致。

任一不一致直接停止。

## 3.1 追踪顺序

当前 checkout 的 Voxel post-processing 顺序被显式重放：

```text
raw decoded anchors
    ↓
score threshold
    ↓
large-box / abnormal-z geometry sanity filter
    ↓
NMS internal top-1000 truncation
    ↓
rotated NMS suppression
    ↓
range filter
    ↓
final score-ordered greedy GT matching
```

Stage-3A 对每个候选记录：

- 每一步是否仍存在 `IoU >= 0.7` 的预测框；
- 最佳 IoU；
- **与该 IoU 对应的同一个预测框的 score**；
- IoU>=0.7 候选中最高分框的 IoU / score / anchor id；
- box center / size / yaw；
- peer-alone 最终正确框与 full decoded best-IoU 框的几何变化；
- 若死在 NMS：被哪个框压掉、压制框 score、压制框对 GT 的 IoU、两框重叠；
- 若最终仍有 IoU>=0.7 框但 GT 未匹配：记录该框在 greedy matching 中被分配给哪个 GT。

## 3.2 输出标签的含义

可能的：

```text
no_iou70_decoded
score_threshold
geometry_sanity
nms_top1000
nms_suppression
range_filter
final_matching_competition
```

这些标签只表示：

> **IoU>=0.7 的检测支持最后在哪里消失。**

不表示：

> “该步骤就是最终根因”。

例如 `score_threshold` 仍可能由上游 fusion 改变 feature → 降低 cls score 引起。

---

# 4. Stage-3B：有限 whole-agent source-subset Oracle

文件：

```text
gspr_evidence/stage3b.py
```

**不会由 Stage-3A 脚本自动启动。**

必须先读完 Stage-3A 报告，再决定是否运行。

固定：

- 同一帧 encoding；
- 同一原始 AttFuse；
- ego 永远存在；
- 不修改 feature；
- 不修改 attention/query；
- 不训练任何模块。

只枚举参与 full fusion 的 peer 子集。

当前最多 4 peers 时：

```text
2^4 = 16
```

组，包括 ego-only 和 ego+all peers。

它回答：

> **仅做 whole-agent source selection，同时保持原 AttFuse 不变，是否存在恢复这些 target 的空间？**

## 4.1 Target-level 与 Frame-level 必须分开

### Target-wise recoverability

对单个 target：

```text
是否存在任意 ego-containing subset 把它恢复？
```

不同 target 可以对应不同 hindsight subset。

因此它只是 opportunity statistic。

### Frame-level realizable subset

同一帧必须使用**一个统一 subset**。

每个 subset 同时记录：

- candidate targets 补回多少；
- 相比 full 丢掉多少原 TP；
- 相比 full 新增多少 TP；
- FP 数及 FP delta。

报告两种 hindsight frame Oracle：

1. `best_frame_subset`
   - 先最大化 candidate recovery；
   - 再最小化 lost full TP；
   - 再最小化新增 FP。

2. `safe_best_frame_subset`
   - 必须 `lost full TP = 0`；
   - 必须 `FP delta <= 0`；
   - 再最大化 candidate recovery。

Stage-3B **不报告 AP**，因为它只运行候选帧，不是完整数据集评测。

## 4.2 结论边界

如果所有 subset 都失败，只能说明：

> 固定 encoding + 固定原 AttFuse + ego 必须存在 + whole-agent inclusion/exclusion

这一有限操作空间不能恢复。

不能排除：

- spatial source selection；
- region-specific source routing；
- 修改 fusion operator；
- 修改 query/attention weight；
- feature replacement/correction。

同样：

```text
peer-alone ✓
ego + peer ✗
```

只能说明：

> 原融合方式在加入 ego 后没有保住该 peer 的检测能力。

不能直接证明：

> ego query 压制 peer。

若 Stage-3A/B 出现稳定对应模式，再设计 Stage-3C 对 query/value/attention 做受控 intervention。

---

# 5. 文件

```text
gspr_evidence/
├── stage3_common.py
├── stage3_stratify.py
├── stage3a.py
├── stage3b.py
├── stage3_report.py
├── test_stage3.py
├── run_stage3a.sh
├── run_stage3b.sh
└── STAGE3.md
```

---

# 6. Stage-3A 后台运行

默认 Stage-2：

```text
/data/cjm/datasets/logs/qa_evidence_validity_20260917_112301
```

`run_stage3a.sh` 会从 Stage-2 protocol 自动读取**真正使用过的 Stage-1 root**，避免手工写错。

建议先做 2 个候选帧 smoke：

```bash
cd /home/cjm/OpenCOOD-main/cjmnet
git pull

RUN=/data/cjm/datasets/logs/qa_stage3a_smoke_$(date +%Y%m%d_%H%M%S)
nohup env GPU=0 MAX_CANDIDATE_FRAMES=2 OUT="$RUN" \
  bash gspr_evidence/run_stage3a.sh \
  > "${RUN}.launcher.log" 2>&1 &

echo $!
echo "$RUN"
```

查看：

```bash
tail -f "${RUN}.launcher.log"
```

smoke 完成后检查：

```bash
cat "$RUN/stage3a_report.md"
```

注意：`MAX_CANDIDATE_FRAMES>0` 只能用于连通性检查，不能作为 Stage-3 结论。

## 正式完整 Stage-3A

```bash
cd /home/cjm/OpenCOOD-main/cjmnet
git pull

RUN=/data/cjm/datasets/logs/qa_stage3a_$(date +%Y%m%d_%H%M%S)
nohup env GPU=0 OUT="$RUN" \
  bash gspr_evidence/run_stage3a.sh \
  > "${RUN}.launcher.log" 2>&1 &

echo $!
echo "$RUN"
```

查看进度：

```bash
tail -f "${RUN}.launcher.log"
```

或：

```bash
tail -f "$RUN/console.log"
```

结束后优先回传：

```text
$RUN/stage3a_report.md
$RUN/stage3a_report.json
$RUN/stratification/stage2_5_stratification.md
```

---

# 7. Stage-3B 后台运行

**不要在 Stage-3A 完成前运行。**

Stage-3A 结果确认后：

```bash
cd /home/cjm/OpenCOOD-main/cjmnet
git pull

STAGE3A_ROOT=/data/cjm/datasets/logs/你的完整Stage3A目录
RUN=/data/cjm/datasets/logs/qa_stage3b_$(date +%Y%m%d_%H%M%S)

nohup env GPU=0 STAGE3A_ROOT="$STAGE3A_ROOT" OUT="$RUN" \
  bash gspr_evidence/run_stage3b.sh \
  > "${RUN}.launcher.log" 2>&1 &

echo $!
echo "$RUN"
```

查看：

```bash
tail -f "${RUN}.launcher.log"
```

完成后：

```bash
cat "$RUN/stage3b_report.md"
```

优先回传：

```text
$RUN/stage3b_report.md
$RUN/stage3b_report.json
```

---

# 8. GPU 规则

严格遵守本项目服务器规则：

```text
只设置 ROCR_VISIBLE_DEVICES=<physical_index>
```

脚本内部会：

```bash
export ROCR_VISIBLE_DEVICES="$GPU"
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
```

不要再额外设置 `HIP_VISIBLE_DEVICES` 或 `CUDA_VISIBLE_DEVICES`。
