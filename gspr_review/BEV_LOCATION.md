# 作用位置对照：点权重与区域 BEV 校正

## 2026-09-10 消息哈希不一致后的配对评测修订

用户回传：54 帧 point_normal 与 protocol 的样本、字节、准入点数一致，但消息 SHA256 全部不同。旧记录没有原始包，不能从哈希断定差异大小或根因，也不能把结果直接当作严格同消息对照。

现改为 protocol 保存实际二进制消息（JSON/base64），其他校正组复用这些消息；逐帧校验原始输入 tensor 的形状、类型与内容指纹。请求区域、接收 BEV 掩码由实际重放包确定，配额和字节上限仍严格检查；未发送特征仍为零。各组还保留自行生成的候选包与固定包在 review_query/review_evidence/bev_query/bev_features 四类中的不同次数，供定位来源。

这是固定通信消息条件下的**离线配对评测**，不是证明原独立推理数值一致，也不是对原哈希差异的原因诊断。消息缓存只供评测，不改训练与实际通信协议。旧权重可用于该条件评测；不能用它宣称已证明训练时逐字节相同。原评测文件保留在旧目录，重放结果存入新目录。

复用已完成的训练，仅重跑 12 组评测：

```bash
cd /home/cjm/OpenCOOD-main/cjmnet
bash gspr_review/run_location_experiment.sh \
  --point-run /data/cjm/datasets/logs/gspr_cf_seed20260909_20260910_102406/contrast \
  --reuse-bev /data/cjm/datasets/logs/gspr_cf_seed20260909_20260910_102406/contrast_location_20260910_162019/bev
```

本次检查源训练配置、前端哈希、训练完成 epoch 及 checkpoint；不重训、不重跑训练前验证。完成回传新目录 location_comparison.json，其中包含候选包差异分类次数。如输入指纹也不一致，将停止，不强行重放。

本版只训练一个新 `bev_contrast`（1472 参数），复用已有 `contrast` 点权重 checkpoint。没有新增评分网络层数、输入字段、特征回包或解冻前端。

## 机制

继续用原查询、量化证据与 11 维输入，通过相同 contrast 头得到有界的标量建议。原版在 PFN 前修改点权重；本版保持原点权重和编码结果，把可修改点的标量按其几何位置汇总到三个 BEV 尺度，每个 BEV 位置按支持点数取平均，再执行 `ego_feature * (1 + adjustment)`，最后进入原缺失块掩码 AttFuse。

原支持格没有有效自车点的位置不会新增校正。因此本实验仍不生成缺失目标或扩展请求覆盖。粗尺度一个位置对应多个 pillar，支持范围按原网格下采样；它与逐点改动不是物理上完全等价的扰动，结论只能针对这两种实现比较，不能宣称已隔离所有因素。BEV 是相对特征缩放，点权重是加性变化，两者作用量纲不同；保持同一数值上限不等于能量完全一致。

邻车完整 BEV 不进入校正头，只有已有实际传输 BEV 经原路径参与融合。GSPR 可靠/未知/支持是校正条件，未把未知当作目标缺失概率。所有原参数及 BN 冻结；点权重版本和 BEV 版本不会在同一轮同时生效。

## 对照与运行

```bash
cd /home/cjm/OpenCOOD-main/cjmnet
bash gspr_review/run_location_experiment.sh \
  --point-run /data/cjm/datasets/logs/gspr_cf_seed20260909_20260910_102406/contrast
```

先同步整个更新后的 `gspr_review` 到服务器。默认从传入的旧训练目录读取前端路径、数据/天气配置、种子、epochs 等。新目录 `contrast_location_<时间戳>` 内先进行真实连接验证，再训练新的 BEV 头；不修改旧训练目录。新旧头结构与初始化方式相同，但最终分别学习，不直接把点权重 checkpoint 当作已训练 BEV 模型。

Clean/Weather 每种做 6 组，共 12 组：protocol、a0b0、point_normal、point_permuted、bev_normal、bev_permuted。模型使用各自验证损失最优权重；不因该开发集 AP 选择某个 epoch。点模型复用旧权重，无需重训。

protocol 保留复核消息及字节开销但不校正；A0B0 保留原全 BEV 预算规则。normal/permuted 保持实际消息、原准入点不变，只在接收侧对证据内容做离线置换。本次新增逐帧消息 SHA256，自动核对 point/BEV/protocol 实际发送内容、样本、字节及准入点一致（A0B0 的协议本来不同，不要求同消息）。

评测输出新增 `mean_bev_changed_cells`（三尺度位置总数，非独立物理区域数）、`mean_bev_abs_adjustment_sum`（每帧三尺度缩放系数绝对值总和）；BEV 版 changed_points 应为 0。详细帧记录中有消息哈希。旧 point/PFN diagnose 不适用于该版，已显式拦截，避免输出误导的点修改统计。

每组完成后保存 location_partial.json，完成全部后保存 location_comparison.json。回传最终文件和新 bev/history.jsonl。真实消息 BEV 版同时优于 protocol 与置换版，才支持空间匹配证据有用；进一步超过 A0B0 才支持其补偿了通信开销。若只优于点版而未优于 protocol，不视为成功。

本地 CPU 单元测试覆盖零初始化、无支持不修改、位置汇总、正负边界、未支持位置隔离，以及梯度从 BEV 回到校正头。真实数据验证由脚本在训练前执行；尚未运行服务器训练或声称有 AP 提升。
