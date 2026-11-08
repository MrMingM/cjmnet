# LR-G / LR1: V2X-R 双模态消融实验

## 目标与结构

这是 LR0 的匹配对照，不是新的融合方法。共两组独立从零训练的模型：

- **LRG**：官方 LiDAR+4D Radar AttFuse，**仅 LiDAR** 的第一层 PFN 前接入原始 GSPR 点可靠性乘权。
- **LR1**：在 LRG 基础上，仅 LiDAR 的柱特征输出后接入现有 GGCA A1 残差局部聚合。
- Radar 分支（官方 PointPillars 编码）、两个 64 通道 BEV 拼接、官方 AttFuse 主干和检测头完全保留，不改模态融合、不加入 MDD、不修改后处理。
- GSPR 直接调用冻结的 `attfuse_gspr/reliability.py`，GGCA A1 直接调用 `gspr_ggca/modules.py` 的 `GGCA(..., level=1)`，不复制算法、不开启 A2/A3。
- 两组都保留 GSPR 原始 keep/evidence 正则；**检测损失沿用 LR0 的原生双模态 `point_pillar_loss`**。这与历史 LiDAR-only A1 的检测损失类别不同，属于跨框架迁移，不宣称逐位等价。

## 严格匹配的变量

| 变量 | LR0 / LRG / LR1 |
| --- | --- |
| 数据 | 已验收 V2X-R 8083 Train、829 Validate/天气 |
| 采样 | 每车辆 Clean4/7、Fog3/7；雷达保持同帧原文件 |
| 训练 | 从零训练、seed 20261017、20 轮、batch8、每轮 1010 次更新 |
| 优化 | Adam lr0.002，原 10/15 调度 |
| 选模 | 完整 Clean 验证 O AP70 最高；严格同分选较早轮 |
| 验证 | Clean/Fog/Snow 每条件829帧，同一份选中权重 |
| 计分 | 官方 O、O*、双模态固定 Clean 真值 F；并对照 LR0 每帧官方GT身份 |
| 空模态 | 完整复用 LR0 全零空特征、保留车辆和真值的处理 |
| 风险 | LR1 增加参数和 GSPR 正则；只在匹配 LR0 上比较，不能当作新融合模块的贡献 |

所有新代码仅在 `v2xr_lr1/`，调用现有 `v2xr_lidar_radar_baseline/engine.py`、`metrics.py` 和现有断点恢复/评分机制。新模型共享参数与 LR0 同种子初值逐张量相等，LRG/LR1 的 GSPR 初值一致，GGCA 的残差投影初值严格为零。单真实帧先验证 LRG 与 LR1 初始输出在数值容差内一致，再进行 smoke。

## 执行

在海光服务器同步代码后，运行以下 **唯一入口**，自动按 LRG → LR1 顺序完成：数据/源码审计 → 初始等价预检 → 两批次真实 smoke → 三天气12帧 smoke → 每组20轮训练 → Clean选模 → 829帧三天气 O/O*/F 正式验证 → LR0 真值严格核对 → 汇总。

```sh
cd /home/cjm/OpenCOOD-main/cjmnet
mkdir -p /data/cjm/datasets/logs/v2xr-lr1-v1
nohup sh v2xr_lr1/run_all.sh >> /data/cjm/datasets/logs/v2xr-lr1-v1/launch.log 2>&1 < /dev/null &
```

恢复使用同一命令（**先确认上次进程已退出**）。不会覆盖既有 LR0、GSPR 或 GGCA 权重及记录。新的独立权重/结果：`/data/cjm/datasets/logs/v2xr-lr1-v1/LRG`、`.../LR1`；总报告：`FINAL_COMPARISON.json`，故障：`FAILURE.json`。

`run_all.sh` 严格 POSIX `sh`、LF，禁止修改已冻结官方及历史文件。脚本对实验身份、源码、配置、真值和已有缓存采取 fail-closed 策略。仅允许海光物理卡1运行；该卡必须实际空闲。正式 test 不参与训练或选择。

## 科研验收及后续研究

确认 `PREFLIGHT.json.status=complete`、两组 `SMOKE.json`、20个独立完整 epoch、`MATCHED_GT.json.status=identical`、各组 `FINAL_RESULTS.json.status=complete`、总 `FINAL_COMPARISON.json.status=complete`，并核验原始哈希和逐帧匹配。

LR1 不一定优于 LR0。若有效，再讨论**独立的跨模态融合模块**；若无增益，先分析 GSPR 可靠性分布、A1 残差幅度、雷达空观测与跨模态冲突，不要直接叠加门控模块。对多种子统计稳健性，应在首轮结论成立后增加独立完整种子并复核，不通过天气验证集挑选有利配置。

### 查新定位

- Huang et al. *V2X-R: Cooperative LiDAR-4D Radar Fusion with Denoising Diffusion for 3D Object Detection*, CVPR 2025.
- Yang et al. *Hybrid Robust Collaborative Perception with LiDAR-4D Radar Fusion under Adverse Weather Conditions*, CVPR 2026.
- Yan et al. *PointASNL: Robust Point Clouds Processing Using Nonlocal Neural Networks With Adaptive Sampling*, CVPR 2020.

**注意：无法访问 Scite 的当前账号论文检索（需要有效试用/订阅），本轮基于公开 CVF 论文原文与现有官方代码判断。**
