# Q-A Observation Diagnostic

本目录把 `idea/01_competing_hypotheses.md` 当前六个竞争假设的**第一轮门禁实验**合并为一次开发验证运行。

研究问题 Q-A：

> 恶劣天气下，自车局部区域究竟是 sensing evidence 不足，还是证据其实存在但在本地表征 / 融合 / 后处理阶段没有被正确利用？对于空区域，能否区分 unknown 与具有几何支持的 observed-free？

## 它测试什么

一次采集后统一生成：

| 假设 | 第一轮证据 |
|---|---|
| H-A1 Quantity-dominant | E-A0 quantity proxy Pearson/Spearman/partial correlation；quantity 对 ego detection 的 AUC |
| H-A2 Structure-dominant | quantity/distance 控制后 4×4 目标表面覆盖、entropy、span 的增量关系 |
| H-A3 Visibility/free-space | 已有 endpoint/traversal proxy 在 zero-support background 与 GT target cells 上的区分；**不是 occupancy truth** |
| H-A4 Occlusion × Weather | GT angular-overlap occlusion proxy × weather-induced miss/support loss |
| H-A5 Local downstream bottleneck | clean 可检测、weather ego miss 中，仍保有强 local raw/GSPR/structure evidence 的比例；同时记录 local confidence / semantic norm |
| H-A6 Fusion bottleneck | `ego weak + peer strong + full still misses` 与 `full recovers` 的比例；额外检查 peer local confidence |

旧 H-A4“邻车有证据 ⇒ ego 需要帮助”不作为部署假设；这里只把 neighbor evidence 当离线 recoverability Oracle。旧 H-A6“天气不同”只作为 Fog/Rain/Snow 分层变量。

## 数据纪律

默认且强制使用：

`/data/scd/datasets/opv2v_official_data_dumping/validate`

Fog/Rain/Snow 是 validation 上的在线物理天气，只用于 hypothesis diagnosis。**脚本拒绝 test root，不使用 OPV2V-W test 进行假设筛选。**

GSPR-v1 仍冻结；本目录不修改 `attfuse_gspr/`、`gspr_supervision/` 或训练入口。

## 一键运行

```bash
cd /home/cjm/OpenCOOD-main/cjmnet
bash qa_observation_diagnostic/run_all.sh
```

默认物理卡 0。指定其它卡只设置 `ROCR_VISIBLE_DEVICES`：

```bash
ROCR_VISIBLE_DEVICES=3 bash qa_observation_diagnostic/run_all.sh
```

**不要同时设置 `HIP_VISIBLE_DEVICES` / `CUDA_VISIBLE_DEVICES`。** 脚本会主动 unset 两者。

先做 32 帧 smoke：

```bash
SMOKE=32 ROCR_VISIBLE_DEVICES=3 bash qa_observation_diagnostic/run_all.sh
```

完整运行使用 `SMOKE=0`（默认）。

默认冻结前端：

- config: `/data/cjm/datasets/logs/gspr_joint_full_v1_seed20260907_20260907_161155/config.yaml`
- checkpoint: `/data/cjm/datasets/logs/gspr_joint_full_v1_seed20260907_20260907_161155/net_best_validation.pth`

可通过 `FRONTEND_CONFIG`、`FRONTEND_CHECKPOINT` 显式覆盖，但四种天气必须使用同一冻结前端。

## 输出

所有大文件只写 `/data/cjm/datasets/logs`：

```text
/data/cjm/datasets/logs/qa_observation_YYYYmmdd_HHMMSS/
├── console.log
├── clean/
│   ├── protocol.json
│   ├── frames.jsonl
│   ├── targets.jsonl
│   └── summary.json
├── fog/ ...
├── rain/ ...
├── snow/ ...
├── hypothesis_results.json
└── hypothesis_report.md
```

完成后优先回传：

1. `hypothesis_report.md`
2. `hypothesis_results.json`

`targets.jsonl` 用于需要进一步 case 分析时再检查。

## 关键定义

### Quantity

- `raw_count`：该 CAV 投影到 ego 坐标后的 raw cloud 中落入 GT 3D box 的点数；
- `box_neff`：GSPR **retained voxel slots** 中落入 GT box 的 `Σp(reliable)`；
- `reliable_count`：上述 retained slots 中 `p(reliable) >= 0.5` 的数量；
- `region_neff`：与 GT footprint 相交的通信网格 cell 内，所有 retained slots 的 GSPR support 总和。

因此 E-A0 真正检查“区域 quantity”和“target-box quantity”是否实际上高度冗余。

### Structure

对 GT box 内 retained points 映射到 target-local 4×4 网格，报告：

- `coverage4x4`
- `quadrants`
- `entropy4x4`
- `span`

H-A2 必须在 quantity、distance、box area、occlusion proxy 被控制后仍有增量才算获得支持。

### Occlusion proxy

第一轮不用 CARLA replay。对每个 GT vehicle，计算 ego 视角下与**更近 GT vehicle** 的 2D angular interval overlap。它只是低成本 proxy，不等价于真实 occlusion fraction。

如果该 proxy 显示稳定 interaction，再进入 `idea/00_research_workflow.md` 的后续 Oracle / 更精确 simulator replay，而不是现在直接投入高成本 ray casting。

### Visibility proxy

复用 `gspr_evidence.geometry` 当前实现：actual endpoint 的 angular-depth traversal samples + endpoint support。

源码定义已经明确：这不是 occupancy probability，也不是 CARLA 全 beam reconstruction。因此：

- 可以作为 H-A3 的 Cheap Gate；
- 不能据此宣称真实传感器 free-space 已经解决；
- 如果第一层门禁不过，不进入精确 replay；
- 如果通过，再单独设计 E-A5 precise simulator visibility audit。

## E-A2 failure-stage partition

阈值不是人工调到天气结果最好，而是只从 **Clean ego-detected targets** 按 distance bin 取 Q25，冻结后应用到 Fog/Rain/Snow。

对“Clean ego 能检测、Weather ego 失败”的目标分类：

- `all_source_weak`
- `ego_weak_peer_strong_full_recovers`
- `ego_weak_peer_strong_full_still_misses`
- `ego_strong_full_miss_local_or_downstream`
- `ego_strong_full_recovers`
- `mixed_other`

这个分解用于决定下一步资源投向，不是训练标签，也不是最终方法。

## 不应该过度解释的地方

- `ego strong` 是 clean-derived empirical support label，不是 GT sensing-sufficiency truth。
- `peer strong + full still misses` 支持“融合端值得继续查”，但不能直接证明 attention 是唯一原因。
- `traversal >= 0.5` 只是诊断阈值，不是可部署 free-space rule。
- 任何自动 verdict 都是 Cheap Gate 摘要；最终结论必须回看分层数值和 case。
