# 局部检测修复 v1

这是一个方法可行性原型，不是已成立的论文创新。它不修改历史 Stage-3 代码，也不修改冻结的 GSPR、AttFuse 或检测头。默认关闭修复时，仍走原模型与原后处理。

## 1. 实际插入位置

当前冻结前端最终输出 psm（每个 anchor 的分类 logit）和 rm（每个 anchor 的 7 维回归量）。

原后处理顺序：

    rm 解码 -> psm sigmoid -> score > 0.2 -> 几何异常过滤
    -> top-k / rotated NMS -> 范围过滤 -> OpenCOOD AP

v1 只在 psm/rm 已产生、最终 score 过滤和 NMS 之前追加少量“修复候选”。原始 dense proposal 永远保留。

## 2. 不依赖 GT 的候选

推理候选只来自模型：

1. 原 full 输出；
2. ego 单独输出；
3. 每个 peer 单独输出。

单车输出复用当前 CommunicationModel 已有的、已经对齐到 ego 坐标系的 pre-fusion 多尺度特征，再通过原冻结 deblock 和 cls/reg head。GT 不参与候选位置、来源选择、区域选择或动作选择。

默认上限：

- 最多 5 个 source；
- 每个 full/source 取分数不低于 0.01 的 top-24 anchor；
- anchor id 去重后每帧最多 32 个候选；
- 每个候选只读取 full 同 anchor 的 3x3 分类上下文；
- 因而每帧局部区域数最多也是 32。

同一 anchor 被多个来源提出时只保留一个候选，并把各来源同 anchor 的预测一起作为特征。不同 anchor 的重复框不手工裁决，统一交给原 NMS。

## 3. 修复器

LocalRepairNet 是一个小型多层感知机，不使用额外 attention。

输入包含 full 与各单车来源同 anchor 的 logit、score、7 维回归量，来源存在 mask，full 的 3x3 局部 logit/score，full decoded box 的归一化描述，以及各来源分数的简单差异统计。

输出两部分：

- 7 维几何残差：中心、尺寸对数比例、周期处理后的 yaw；
- 一个新的质量分数 logit。

几何尺寸通过 exp 保持为正，yaw 用 atan2(sin, cos) 处理周期。

消融不复制模型：

- score：只使用新分数，几何保持 full；
- geometry：只使用新几何，分数保持 full；
- joint：分数和几何都修复。

evaluation_presets.yaml 提供 baseline、score_only、geometry_only、joint_all、joint_selector 五个预设。

## 4. 第一阶段监督

训练和推理使用完全相同的模型候选生成。候选固定后才读取 GT：

- 任一模型来源在该 anchor 的 decoded box 与某 GT 最大 IoU >= 0.5：正样本；
- 所有来源最大 IoU < 0.2：背景负样本；
- 0.2 <= IoU < 0.5：忽略。

因此背景候选和“不需要修复但本来就正确”的候选都会保留，不是只训练失败样本。

几何目标是“full 当前 decoded box 到匹配 GT”的有限残差。分数目标不是 Stage-3C 动作标签，而是该候选由模型来源实际提供的最佳定位 IoU，作为 0 到 1 的软质量监督。

原前端始终 requires_grad=False、eval()，BatchNorm 不更新；优化器只接 LocalRepairNet 参数。训练代码在更新时检查这些条件。

## 5. 第二阶段 KEEP / REPAIR 选择器

默认采用两阶段训练。

官方 OPV2V train 的 43 个 scene 用固定种子做场景级拆分：

- 约 80% scene：训练修复器；
- 剩余约 20% scene：固定修复器后，生成选择器动作标签。

具体 scene id 写入 checkpoint contract，两个集合不得重叠。validation 不参与梯度，只选 checkpoint。test 从不生成训练标签。

对 selector 子集的每个模型候选，固定第一阶段 repairer，真正把“这个候选单独修复”插入原 score/NMS 流程，再与原 full 帧比较。IoU=0.7 下同时满足以下三项才标为 REPAIR=1：

- 至少补回 1 个 baseline miss；
- 不丢任何 baseline TP；
- 没有新增 FP。

造成 TP 损失、增加 FP、或者完全无变化，都标为 KEEP=0。

新增 FP 沿用 Stage-3 的保守定义：action FP 与 baseline FP 做 IoU>=0.7 的一对一几何匹配，未匹配 action FP 才算新增。

标签对应当前固定 repairer checkpoint，不使用 Stage-3C 的“peer 硬替换成功/失败”。

v1 标签按“单候选动作”生成；推理时多个通过 selector 的候选可以同时加入，最终冲突仍由一次共享原 NMS 解决。这是第一版简化，需要靠整帧 AP 和 TP/FP 指标判断是否足够。

## 6. 数据与天气

学习参数只用：

    /data/scd/datasets/opv2v_official_data_dumping/train

开发验证只用：

    /data/scd/datasets/opv2v_official_data_dumping/validate

固定方案后的最终评估：

    clean /data/scd/datasets/opv2v_official_data_dumping/test
    fog   /data/cjm/datasets/opv2v-w/fog/test
    rain  /data/cjm/datasets/opv2v-w/rain/test
    snow  /data/cjm/datasets/opv2v-w/snow/test

训练仍覆盖 clean、physics_fog、physics_rain、physics_snow，四类得到相同的 scene/frame 暴露次数，但昂贵的数据与冻结模型计算不再每个 epoch 重跑。

第一阶段先对 repair train、selector train、validation 三个场景集合分别建立一次性缓存。缓存阶段执行天气模拟、冻结 GSPR/PointPillar、full/source 候选、必要的 GT 几何匹配；随后 3 个 repair epoch 只读取缓存的逐帧候选特征与监督。第二阶段 selector 继续复用第一阶段缓存，不再次执行天气模拟或冻结前端。这样不改变候选、监督和最终在线推理定义，只去掉重复计算。

普通训练/评价 loader 仍保持 workers=0。一次性缓存构建可使用 cache_workers（默认 2）在 CPU 后台预取天气模拟与预处理；随机种子、worker 数量和配置都写入 contract。正式 A/B/C 评价仍在同一个 batch 上顺序计算，使用 workers=0，因此同一帧看到完全相同的在线天气实例。

开发 validation 加在线天气只能叫 development validation，不能叫 OPV2V-W。benchmark 模式移除在线天气模拟，直接读取固定 OPV2V clean test / OPV2V-W 文件。

## 7. 评价

evaluate.py 在一次遍历中同时计算：

- A baseline：原模型；
- B all：所有模型候选都执行指定消融的修复；
- C selector：只加入 KEEP/REPAIR 选择器通过的修复候选。

继续使用项目 eval_utils，输出 AP@0.3、AP@0.5、AP@0.7，重点比较 AP@0.5、AP@0.7。

另外记录：

- 相比 baseline 的补回 GT、原 TP 损失、新增 FP，分别在 IoU 0.5/0.7；
- 候选数与事后候选覆盖率；
- B/C 实际执行修复的比例；
- baseline forward、source/candidate 准备、repair net、selector 耗时；
- baseline 与完整路径峰值显存；
- 每帧明细。

候选覆盖只做流程检查，不用于选择测试动作。

当前 full intermediate-fusion 基线下，接收端已经拥有每个来源的 pre-fusion 特征，因此 source-only heads 增加的是计算，不新增 transmitted tensor，记录 additional_communication_bytes=0。以后如果换成稀疏或压缩通信协议，必须重新统计，不能沿用这个 0。

## 8. 保护和断点

- 第一帧强制检查“0 个 extra repair candidate”的自定义后处理与原 ds.post_process 完全一致；
- 第一帧还检查全来源冻结融合重放与原 full psm/rm 一致；
- 检查 NaN/Inf、source 数量、feature 维度、候选数量、尺寸正值、GT 对齐；
- checkpoint 保存模型、优化器、学习率调度器、Python/NumPy/Torch/CUDA RNG、场景划分、前端 SHA、配置 SHA、关键源码 SHA；
- 第一阶段缓存按 split/weather 原子保存；如果缓存中途停止，--resume 会跳过已完成天气，只重建中断项；
- 缓存完成后，--resume 从 last.pth 的 repair/selector epoch 边界恢复；
- smoke checkpoint 被正式 benchmark 明确拒绝。

历史 Stage-3 清单、日志和源码不修改。

## 9. 服务器命令

默认服务器路径和前端 checkpoint 已写在 run.sh，可通过环境变量覆盖。脚本使用 POSIX sh 语法和 LF 换行，不依赖 bash 的 pipefail 或双中括号。

### 9.1 smoke

先跑纯函数测试，再跑真实 train/validation 前向，并额外用每个 split/weather 1 帧检查一次性缓存路径（包括 cache_workers）：

    cd /home/cjm/OpenCOOD-main/cjmnet
    GPU=0 sh local_detection_repair/run.sh smoke

它只验证代码链路，不代表方法有效。

### 9.2 第一阶段修复器训练

    cd /home/cjm/OpenCOOD-main/cjmnet
    GPU=0 sh local_detection_repair/run.sh repair

日志、一次性 repair_cache 和 checkpoint 进入 /data/cjm/datasets/logs/local_repair_phase1_<timestamp>/。缓存生成是第一阶段最耗时的部分，但每个 split/weather 只生成一次；后续 repair epoch 不再重新模拟天气或跑冻结检测器。

恢复：

    RUN_NAME=原来的目录名 RESUME=1 GPU=0 sh local_detection_repair/run.sh repair

### 9.3 第二阶段选择器训练

默认 joint：

    REPAIR_CHECKPOINT=/data/cjm/datasets/logs/第一阶段目录/repair_best.pth \
    GPU=0 sh local_detection_repair/run.sh selector

如要给 score 或 geometry 消融单独训练选择器：

    ABLATION=score REPAIR_CHECKPOINT=... GPU=0 sh local_detection_repair/run.sh selector

### 9.4 development validation

四种天气分别启动。下面以 snow 为例：

    REPAIR_CHECKPOINT=/data/cjm/datasets/logs/第一阶段目录/repair_best.pth \
    SELECTOR_CHECKPOINT=/data/cjm/datasets/logs/第二阶段目录/selector_best.pth \
    WEATHER=snow PRESET=joint_selector GPU=0 \
    sh local_detection_repair/run.sh validate

将 WEATHER 改为 clean、fog、rain、snow 分别运行。PRESET 可改为 baseline、score_only、geometry_only、joint_all、joint_selector。

score_only / geometry_only 如果使用 selector，需要用对应 ABLATION 重新训练 selector；默认 joint_selector 只接受 joint selector checkpoint。

### 9.5 固定方案后的最终测试

只有在 validation 已经固定候选参数、checkpoint、selector threshold 和 ablation 后再运行。下面以 OPV2V-W snow 为例：

    REPAIR_CHECKPOINT=/data/cjm/datasets/logs/第一阶段目录/repair_best.pth \
    SELECTOR_CHECKPOINT=/data/cjm/datasets/logs/第二阶段目录/selector_best.pth \
    WEATHER=snow PRESET=joint_selector GPU=0 \
    sh local_detection_repair/run.sh test

clean 使用 OPV2V official test；fog/rain/snow 使用现有 OPV2V-W 对应 test 目录。benchmark 不在线生成天气，也不允许 smoke checkpoint。

run.sh 不会自动连跑全部消融、多随机种子或四个正式测试。

## 10. 第一版能回答什么

如果 B/C 在完整 validation，随后在固定方案的 clean/OPV2V-W test 上提高 AP，同时原 TP 损失、新增 FP 和开销可控，才能说明“学习式局部修复”值得继续研究。

如果只有候选覆盖率高、失败样本恢复率高，但整体 AP 不升，不能认为方法成立。

如果 B 有收益、C 没收益，优先说明 KEEP/REPAIR 判别仍不够；如果连 B 都没有整体收益，应先质疑修复器或候选定义，而不是继续堆选择器。
