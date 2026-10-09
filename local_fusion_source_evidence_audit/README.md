# S4 Source Evidence Audit

这一轮回答：旧 S1/S2 看不到的来源局部证据，能否让相同线性模型判断分类/回归该选哪个来源。全部科研计算在服务器进行。

## 一次后台启动

把 **整个** `local_fusion_source_evidence_audit/` 目录一起复制到服务器，包括 `source_snapshot.json`。不需要 `.git`，不要在服务器手改源码哈希。

```sh
cd /home/cjm/OpenCOOD-main/cjmnet
conda activate opencood
export ROCR_VISIBLE_DEVICES=2
unset HIP_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES
sh local_fusion_source_evidence_audit/launch.sh
```

终端打印 `RUN= / PID= / LOG=`。默认旧运行是 `/data/cjm/datasets/logs/action_utility_audit_20261007_210243`；启动器会校验其完整性。可用 `BASE_RUN=/data/cjm/datasets/logs/<other_complete_run>` 覆盖。新输出全部位于独立的 `/data/cjm/datasets/logs/source_evidence_audit_<timestamp>`。

```sh
tail -f "$RUN/driver.log"
python -m local_fusion_source_evidence_audit.status --run "$RUN"
```

若使用另一个终端，把 `$RUN` 替换为打印出的完整目录。状态命令先重新检查哈希、所有帧/消融模型和原运行完整性；只有全部通过才返回 0 并打印六个 `complete`。检查大缓存的哈希需要读盘，耗时取决于存储速度。

断点续跑同一目录：

```sh
RUN=/data/cjm/datasets/logs/source_evidence_audit_<timestamp> \
sh local_fusion_source_evidence_audit/launch.sh
```

帧、训练 epoch 和阶段均有原子缓存与源码/标签/模型/schema 哈希。进程锁阻止同一 RUN 并发运行。源码、schema、旧运行或环境发生变化会停止；工程修复需要记录原因并使用新 RUN，不能静默迁移已有结果。

## 比较如何保持公平

- BASE_RUN 必须完成 S0/S1/S2/S3/FINAL；核对全部完整 artifact、checkpoint、source snapshot、候选定义、场景划分和原标签缓存。
- R0 使用旧两种 feature variant 和保存模型重算全部指标、计数及背景/关联分层，差值超过 `1e-6` 立即停止。
- `output_only` 是旧 primary `with_competition_features` 的完整 schema 和原保存模型。
- 复用原 S0 structured outcomes，不枚举新的 counterfactual labels，不改变任何 action。
- 两任务均为单个 `torch.nn.Linear(dim, 1)`，保持原 pairwise logistic、30 epochs、AdamW lr=.01、weight_decay=.0001、pair batch 和顺序。只用 fit 场景归一化，原 calibration 场景校准阈值，固定最后 epoch。
- 主方法预注册为 `all_evidence`。其他配置只做机制消融，包含 no-LOO 和 no-GSPR。
- Geometry 消融中的 LOO 只含回归变化；Semantic 消融中的 LOO 只含分类变化。合并配置才同时使用两者。
- 继续使用旧 learnability、额外 evidence 增量和 S3 回放门槛，不能根据结果调整。

## 证据和坐标

实际流水线 `proj_first=True`：source 点云先投到 ego，再体素化。这里只统计 `voxel_features` 的保留有效点，不把原始点重新投影，也不重建体素化丢弃的点。运行时检查有效点确实位于记录的 voxel cell、ego proposal transform 为单位矩阵、pose 和 source 顺序对应。

预测框扩张沿用旧配置 1.5。点使用 3D oriented ROI；pillar 与 BEV 使用其 XY ROI。BEV ROI 空时保留空和 validity=0，动作 ROI 仍完整沿用旧 S3 的 fallback/mask。几何统计包括支持数量、覆盖、协方差、spread 和相对观察位置；SEM 统计 levels 0/1 的局部激活及与 Shared/ego/其他来源的相似度。GSPR 读取同一次 frozen forward 的可靠性、不确定性和两类 evidence，带 count/coverage/validity。

`point_count` 与 `valid_point_count` 均指框内有效保留点；不包含 padded slots。非零 voxel/pillar 是带框内有效点的 source/grid cell。occupied ratio 的分母是框内网格中心数（KEEP 汇总时乘来源数）；边缘跨 cell 的点支持可能与网格中心覆盖略有差异。GSPR point coverage 是 ROI 点占其附近 bbox 保留点的比例，pillar coverage 是 ROI 非空 pillar 数占对应网格中心数的比例。

LOO：完整融合与删除一个来源后的融合都使用冻结模块重新计算权重。B0 Shared 的 learned router 与 original ego-query AttFuse 分开记录。删除 ego 后原 query 不存在，结果不可定义；query:i 删除 i 后自身 query 也不可定义，显式设置 applicability=0。`loo_shared_*` 始终表示该 underlying CAV 对 **ego-query B0 Shared** 的影响，因此可用于 query:i 的证据；它不冒充删除 i 后的 query:i 输出。每个标量都有 validity，填充零只作存储占位。

query:i 表示 i 作为 query 并消费全部来源。local evidence 来自其 underlying CAV，query-conditioned output 和 single/query 差异另行保留。绝对 CAV 索引只用于张量对齐和报告，不是学习输入。

## 复现边界

Validation 提取/回放使用原运行保存的精确 FP32 action pool，旧特征继续要求完全相同。重新 forward 的中间证据按原 `reproducibility.py` 的 `atol=rtol=2e-5` 核对冻结输出。Train 没有原始 action tensor pool，重新提取时严格核对旧候选/source/output feature，原始数值量仍用原容差。仅两种派生量可额外核验：分类名次必须从旧/新 `roi_max` 精确重建，并证明每个改变的比较位于原score误差界内；`中心距离×重叠数` 必须数量完全相同、原始距离及XYZ变化仍在原容差内、旧/新乘积均可在FP32存储舍入范围内重建，再按原距离误差界传播核对。模型输入始终使用旧缓存特征与旧名次；其他超限仍停止。原 AP 复现仍要求 `1e-6`。具体规则和失败运行见 `REPAIR_20261009.md`。

若复现失败，日志会给出具体frame、字段及 `diagnostics/output_reproduction_<split>_<weather>_<frame>.json` 路径。该JSON分别记录最大绝对差位置和真正超容差的位置（两者可能不同），包括原/新值、超限数和派生量核验证据。修改源码后使用新RUN，保留失败RUN，不把旧manifest移到新目录。

标签与评价可使用 GT。Assisted 仅复用旧关联来限定 proposal 集；Proposal 的 source choice、KEEP/MODIFY、margin 和冲突在 GT 评价前固定。Inference feature GT=0。S4 不访问正式 test，所有 split 和链接先检查 realpath。

## 输出和成本

自动生成 R0、GEO、SEM、ABLATIONS、REPLAY、FINAL 的 MD/JSON、protocol/manifest、feature/coordinate contract、主线性权重和 normalization。FINAL 先给几何/语义是否发现证据，再列排名、消融、整帧 AP 和预注册 Case 1–6；未覆盖的边界情况明确保留。

提取每帧只做冻结 forward、每个 peer 的 B0/AttFuse LOO 和统计，不重复旧 S0 的逐动作 NMS。回放统一计算全部方法，使用精确旧预测池。日志打印每帧/epoch/阶段进度；总时长取决于 GPU、保留点数和旧缓存的读盘速度。代码不提供未经测速的小时数承诺。

## 检查

本地只允许静态与纯 Python/NumPy 检查：

```sh
python -m local_fusion_source_evidence_audit.test_core
sh -n local_fusion_source_evidence_audit/launch.sh
sh -n local_fusion_source_evidence_audit/run_all.sh
```

服务器 driver 强制加 `--require-server`，执行 tiny synthetic Torch/GSPR/AttFuse/OpenCOOD decoder/extractor/replay 检查，缺包或失败不能跳过。它们不访问真实数据，不替代正式实验结果。

源码修改后，只在本地、同步前重新导出本目录的发布 snapshot：

```sh
python -m local_fusion_source_evidence_audit.common
```

该操作只计算源码哈希，不运行模型或读取数据。已开始的 RUN 不接受重新导出的不同源码。
