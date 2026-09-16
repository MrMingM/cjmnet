# 协同复核 GSPR：第一版创新机制实验

已完成第一轮训练、需要排查“点权重变化但检测无明显收益”时，使用 [固定支持诊断说明](DIAGNOSTICS.md) 和 `bash gspr_review/run_diagnostics.sh`。此入口复用旧权重，不启动训练。

实现日期：2026-09-09。独立原型，未取得服务器训练结果，不能称为已验证的创新或性能提升。

## 这次实际改变了什么

旧实验在已经固定的 BEV 上改变发送优先级。本版加入一条反馈链：

`GSPR 初步点权重 → 存疑区域请求 → 邻车几何证据回复 → 修正自车点权重 → 自车重新编码 → 与收到的 BEV 融合检测`

GSPR-v1、主干和检测器的参数与 BN 均冻结；新增 1473 参数的点权重修正 MLP（11→32→32→1）。最后一层零初始化。新网络通过真实硬掩码融合和重新编码直接获得检测损失梯度，不使用旧 A/B 的软选择梯度代理。原 `attfuse_gspr`、`gspr_supervision`、`gspr_communication` 源码均未修改。

## 消息与作用位置

- 自车仅根据自己的统计，用 `平均 u × (1−平均原始可靠概率) × 有效观测标志` 为通信块排序，选取预算能承担的存疑块。请求是块编号；没有发送自车完整特征，也不需要邻车访问它们。
- 邻车只针对收到的块，按原 XY pillar 和 4 个高度格整理证据。每格传可靠证据份额、噪声证据份额、未知份额及饱和支持量，均 uint8 量化。支持量为保留点数 `n/(n+4)`，不是原始密度、可见性或目标概率。
- 可靠/噪声/未知由 `point_evidence` 计算，不把有 floor 映射的权重当概率。多个邻车对应格取 `可靠份额×支持量` 最大的一份，不累加为独立概率证据。
- 自车修正网络读取自己的点信息与实际解码的邻车证据。仅对被请求、且相同 XY pillar/高度格内收到非零可靠支持的有效点调整权重，幅度最多 ±0.25，范围仍为原 floor 至 1。没有对应支持或只有明确噪声而无可靠支持时保持原权重；邻车没看到不等于自车点是噪声。
- 收到的信息是粗粒度几何支持，不是严格的点对应或物体存在证明。相同格内也可能有不同表面；第一版保留这个可检验的限制。
- 重新计算自车柱特征和整个自车 BEV 主干。邻车编码不重算，普通特征仍使用通信前置信度的 A0B0 规则选择。原先没有观测的区域仍可通过普通 BEV 通信得到补充；本版没有新增自适应“两类请求预算分配器”。
- 保持原 `proj_first=True` 输入约定，不重复坐标变换。几何按既有 ego 对齐坐标核对，未新增位姿误差/时延补偿。

通信总预算 262144 字节（256 KiB），其中最多 12.5% 用于复核；不足或无可查询区域时不发复核消息，剩余预算交给普通 BEV 通信。单条请求 16 字节头部＋每块 4 字节编号；单条回复 16 字节头部＋每块 4 字节编号＋`8×8×4×4=1024` 字节证据。逐邻车单播计费，所有三尺度 BEV 载荷也实际打包计费。radio/IP/MAC 开销未模拟。

训练与评测始终做真实 CPU 消息序列化/解码；只有收到的证据参与修正，只有收到的邻车 BEV 块参与融合。此实现为了协议一致性优先正确性，CPU 打包及自车二次编码可能增加延迟，日志报告全部模型与消息处理耗时。

## 第一轮训练协议

默认 4 个训练场景、2 个开发验证场景、每 5 帧取 1 帧，3 个 epoch，每 epoch 最多 300 个有有效复核支持的更新。沿用已有物理 Fog/Rain/Snow 增强器，训练时 25% 概率使用清晰分支，其余使用天气分支；验证同时计算 Clean 和天气分支损失，以二者平均选择 best。天气分支固定种子，验证 epoch 固定为 0。

现有天气增强器按帧选天气/强度、按车辆使用独立随机扰动；本版没有修改成独立抽取每辆车天气强度的专门异质质量协议。它是机制初筛，不是 OPV2V-W 正式泛化实验。小规模开发结果不代表最终测试性能。

目标为检测损失加轻量权重改动平方惩罚。同步多车 loader 没有点真值，单车 NPZ 不直接混入；不提供虚假的点分类正确率。`changed_points` 只表示发生调整，不代表调整正确。训练日志记录有效证据点数、权重上调/下调和改动幅度，便于定位是否根本没有发生复核。

## 上传与一键运行

2026-09-09 修复：初版缺少 PhysicsFog 的 `lookup_dir`，会错误地在当前工作目录找表。现已在配置中指定 `TripleMixer-main/tools/fog_sim/integral_lookup_tables_seg_light_0.008beta/original`，按 cjmnet 项目根目录解析；旧配置缺省时也使用此路径。它是本版明确选用的现有表资源，不表示已复现 TripleMixer 全部雾强度协议。启动时提前检查文件及 alpha 范围，并把解析后的路径保存到运行配置。

服务器 TripleMixer 放在其他位置时，设置 `export FOG_LOOKUP_DIR=/包含pickle文件的实际目录`；必须指向直接含表的目录，TripleMixer 通常要包含最后的 `/original`。环境变量优先于 YAML。无需重新生成点标签或 GSPR 权重。

将本目录完整上传为 `/home/cjm/OpenCOOD-main/cjmnet/gspr_review`，保留服务器已有 `gspr_communication`。无需替换 GSPR 文件。

```bash
cd /home/cjm/OpenCOOD-main/cjmnet
bash gspr_review/run_experiment.sh
```

脚本默认使用：

```text
/data/cjm/datasets/logs/gspr_joint_full_v1_seed20260907_20260907_161155/config.yaml
/data/cjm/datasets/logs/gspr_joint_full_v1_seed20260907_20260907_161155/net_best_validation.pth
```

脚本优先读取已有 `FRONTEND_CONFIG`、`FRONTEND_CHECKPOINT` 环境变量。要明确使用上述固定版本，可以先执行：

```bash
export FRONTEND_CONFIG=/data/cjm/datasets/logs/gspr_joint_full_v1_seed20260907_20260907_161155/config.yaml
export FRONTEND_CHECKPOINT=/data/cjm/datasets/logs/gspr_joint_full_v1_seed20260907_20260907_161155/net_best_validation.pth
export ROCR_VISIBLE_DEVICES=0
bash gspr_review/run_experiment.sh
```

脚本依次做单元测试、真实多车连接验证、训练、Clean/Weather 共八组评测及汇总；失败会停止，不会覆盖旧运行目录。环境使用现有 opencood Python，取消 CUDA/HIP_VISIBLE_DEVICES 的二次过滤。输出到日志根下带时间戳的新目录，只保存 reviewer 权重，不重复保存冻结前端。

## 只先验证连接

```bash
cd /home/cjm/OpenCOOD-main/cjmnet
export PYTHONPATH=/home/cjm/OpenCOOD-main/cjmnet:/home/cjm/OpenCOOD-main
export ROCR_VISIBLE_DEVICES=0
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
export PYTHONHASHSEED=20260909
export FRONTEND_CONFIG=/data/cjm/datasets/logs/gspr_joint_full_v1_seed20260907_20260907_161155/config.yaml
export FRONTEND_CHECKPOINT=/data/cjm/datasets/logs/gspr_joint_full_v1_seed20260907_20260907_161155/net_best_validation.pth
/home/cjm/miniconda3/envs/opencood/bin/python -m unittest gspr_review.test_codec gspr_review.test_tensors -v
/home/cjm/miniconda3/envs/opencood/bin/python -m gspr_review.verify \
  --frontend-config "$FRONTEND_CONFIG" --frontend-checkpoint "$FRONTEND_CHECKPOINT"
```

真实数据验证包括 full/none 端点、零初始化与同消息对照一致、非零修正实际改变点权重及检测输出、检测梯度进入 reviewer、冻结参数/BN 不变、训练/评测消息路径一致、零预算不能修改点权重。前 20 帧如果没有可复核的跨车支持，会明确报错，避免静默训练一个始终不起作用的模块。

## 对照组与判读

| mode | 行为 | 用途 |
| --- | --- | --- |
| a0b0 | 全部预算供原规则 BEV 通信 | 与当前实用参照比较；使用这次相同数据重新评测 |
| protocol | 发送完全相同的复核消息，但不修正权重 | 分离复核占用带宽导致的影响；零初始化只保证与此组一致 |
| review | 正常证据反馈与训练后的修正器 | 新机制 |
| shuffled | 同一训练权重、相同载荷，将已收到的证据在块内平移后使用 | 检查收益是否依赖证据位置；不是保持有效支持掩码不变的纯信息消融 |

每组评测都带同一输入的 ego-only 配对检测分析。另支持 `--mode none/full`，需要时单独运行。

- review 超过 protocol 但不超过 a0b0：修正可能有用，但还不够补偿占用的特征预算。
- review 与 protocol 相同，且 changed_points≈0：检查收到的支持和修正器训练。
- changed_points 明显但 AP 不涨：目前不能断言点级质量改善；需要进一步用同步点标签或几何一致性核验修正方向。
- review 优于 a0b0，且优于 shuffled：支持正确位置的复核证据有用，仍需多种子与更多场景确认。

最终 `controls_summary.json` 包含所有八组的 AP、总字节、复核字节、有效支持及点权重改动。各评测目录还有 `frames.jsonl`、`protocol.json`、`communication.json`。

## 2026-09-10：参考差异 + 校正增益移植实验

后续作用位置对照入口为 `bash gspr_review/run_location_experiment.sh --point-run <已完成的contrast目录>`；只训练新的 BEV 版本，详见 [BEV_LOCATION.md](BEV_LOCATION.md)。

新增 `contrast`、`contrast_gain` 两种复核头，未指定 architecture 的旧配置继续使用原模型，可加载旧 checkpoint。运行 `bash gspr_review/run_counterfactual.sh` 完成三种架构的训练、同预算评测及诊断。具体来源、参考消息定义、对照和限制见 [SOURCE_ADAPTATION.md](SOURCE_ADAPTATION.md)。

## 初版验证状态与源码来源（历史记录）

本地仅完成 Python 语法编译、7 项新旧协议单元测试，以及原 14 个冻结源文件哈希核验。本机没有 torch/OpenCOOD 完整运行环境；张量测试、真实模型验证、训练和 AP 需要服务器执行，尚未声称通过。

沿用现有 `gspr_communication` 的冻结前端加载、数据适配、规则选择、三尺度 codec 和缺失键 AttFuse；其来源与上游许可保留在原包 `SOURCE.md` / `UPSTREAM_LICENSE.txt`。本版复核协议和点修正模块是项目新实现，未声称从论文成熟源码中直接取得已经验证的相同机制。
