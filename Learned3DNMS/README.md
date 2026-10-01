# Frozen-F Learned 3D NMS benchmark

本目录为独立论文基准，原 F 与 decoded box 完全冻结。实现论文的 D2D-Rescore（全局候选注意力）和 GossipNet3D（5 米内局部消息传递）两种残差评分模块，不继续训练协同感知模型，也不重新生成天气。

**使用的数据集**：固定数据集 /data/cjm/datasets/opv2v-physics-fixed-v1。读取 clean 的官方 train/validate 以及 fog/rain/snow 各自 train/validate；严格复用 SAQC.offline_weather.make_loader 的 manifest 完整性、模型配置哈希、来源路径校验与禁用 weather_augmentation 逻辑。mixed 目录是固定逐帧混合天气，主实验使用四种各自的物理天气与 clean，共用一个模型。参见 opencood/tools/PHYSICS_WEATHER_DATASET.md。

## 一条命令自动完成：检查 → 提取 → 训练 → 评价

先把此分支代码同步到远程服务器 /home/cjm/OpenCOOD-main/cjmnet，并配置你已经验证的冻结 F 前端、F run 与 v3 checkpoint。切勿使用其他变体或未匹配的前端 YAML。

在 cjmnet 仓库根目录执行：

~~~sh
cd /home/cjm/OpenCOOD-main/cjmnet
conda activate opencood
export PYTHONPATH=/home/cjm/OpenCOOD-main/cjmnet:/home/cjm/OpenCOOD-main
export RUN=/data/cjm/datasets/logs/fusion_detector_adaptation_20260927_103227
export FRONTEND_ROOT=/data/cjm/datasets/logs/gspr_joint_full_v1_seed20260907_20260907_161155
export FRONTEND_CONFIG=$FRONTEND_ROOT/config.yaml
export FRONTEND_CHECKPOINT=$FRONTEND_ROOT/net_best_validation.pth
export V3_RUN=/data/cjm/datasets/logs/local_fusion_v3_20260922_182832
export V3_CONFIG=local_fusion_v3/experiment.yaml
export V3_CHECKPOINT=$V3_RUN/residual/best.pth
export WEATHER_DATASET_ROOT=/data/cjm/datasets/opv2v-physics-fixed-v1
export ROCR_VISIBLE_DEVICES=2
sh Learned3DNMS/run_all.sh
~~~

上方 V3_CHECKPOINT 是一个示例路径，**必须以当前服务器真实存在、且与 RUN/protocol.json 对应的 v3 checkpoint 为准**，不能因为示例路径不存在就更换模型。可以设置 EPOCHS=100、SMOKE=2（只做小样本通路检查）、OUT=独立结果目录。默认 SMOKE=0 处理完整 train 和 validate；EPOCHS=10。D2D 为 6 层 / 64 通道 / 4 头，GossipNet3D 为 4 层 / 64 通道 / 5 米半径。可用 nohup 从后台启动同一个 run_all.sh，训练成功后脚本会**直接运行完整固定天气 validate 评价**，无需再次输入测试命令。sh 使用 LF 换行、POSIX 语法，不使用 Bash-only 特性或 pipefail。


## 中断后继续（包括 extraction 只完成一部分）

更新后的 `RESUME=1` 同时支持两种情况：

- `manifest.status=complete`：直接复用完整 cache；
- `manifest.status=in_progress`：逐个验证已有连续 NPZ 前缀，不覆盖已有文件，然后继续剩余 extraction。重新启动时，已有部分只重走 DataLoader 以核对 sample ID，不再执行冻结 F 的 GPU forward。

例如当前实验只完成 `train/clean` 前 630 帧：

~~~sh
cd /home/cjm/OpenCOOD-main/cjmnet
git pull --ff-only

export PYTHONPATH=/home/cjm/OpenCOOD-main/cjmnet:/home/cjm/OpenCOOD-main
export OUT=/data/cjm/datasets/logs/learned_3d_nms_20260930_203326
export RESUME=1

export RUN=/data/cjm/datasets/logs/fusion_detector_adaptation_20260927_103227
export FRONTEND_ROOT=/data/cjm/datasets/logs/gspr_joint_full_v1_seed20260907_20260907_161155
export FRONTEND_CONFIG=$FRONTEND_ROOT/config.yaml
export FRONTEND_CHECKPOINT=$FRONTEND_ROOT/net_best_validation.pth
export V3_RUN=/data/cjm/datasets/logs/local_fusion_v3_20260922_182832
export V3_CONFIG=local_fusion_v3/experiment.yaml
export V3_CHECKPOINT=$V3_RUN/residual/best.pth
export WEATHER_DATASET_ROOT=/data/cjm/datasets/opv2v-physics-fixed-v1
export ROCR_VISIBLE_DEVICES=2

sh Learned3DNMS/run_all.sh
~~~

如果 extraction 完成后进程再次中断，后续同一条 `RESUME=1` 命令还会复用已有 `train_d2d/last.pt` / `train_gossip/last.pt`，并把优化后的评价写到独立的 `validation_d2d_fast/` 和 `validation_gossip_fast/`，不会覆盖旧的部分结果。

新版 evaluator 直接复用 extraction 已保存的 top256×GT BEV IoU 矩阵。候选方法不再为 AP30/AP50/AP70 和 recovered/lost 统计反复执行 Shapely candidate-to-GT 相交；Original F 因为旧 cache 没有保存其 candidate ID 映射，每帧仍只计算一次 polygon IoU。原 rotated NMS 保持不变，所以方法定义没有改变。

## 输出

所有结果在新时间戳目录：

- cache/manifest.json 记录模型及数据来源哈希，cache/train 和 cache/validate 保存每帧候选；包含 GT 的缓存仅允许用于训练监督与评价。
- train_d2d/last.pt、train_gossip/last.pt，以及各自的 history.json、summary.json。
- validation_d2d/ 和 validation_gossip/ 中各有 results.json、results.md、frames.jsonl；每个模型训练成功后立即自动测试。

结果包括 Original F、top256 原分数+原 NMS、D2D native Top-K、D2D + 原旋转 NMS、GT-IoU Oracle 的 frame-order 与 global-sort AP30/50/70，以及同口径 Oracle recovery ratio。单独输出 recovered/lost GT 和误检数差；误检数差并非逐框 new-FP 身份匹配。

**正式测试边界**：当前一键脚本在固定 OPV2V validate 上直接评测，绝不自动触碰 OPV2V-W test。固定物理天气 validate 与历史 OPV2V-W test 是两套不同数据。需要在固定开发协议上确认代码/超参后，需另行对原始 OPV2V/OPV2V-W test 做一次独立正式评测；不能把开发验证 AP 说成独立测试集 AP。

## 代码清单

- model.py：检测属性编码、D2D self-attention、GossipNet3D 局部消息池化、零初始化残差 score head。
- data.py：固定候选集缓存、按 OPV2V BEV IoU 做一对一匹配。
- extract.py：只在冻结 F 上提取 top256 并预计算候选对 GT IoU。
- train.py：固定 Clean/Fog/Rain/Snow 为每个方法分别训练一个跨天气共享模型。
- evaluate.py：两条 NMS 推理路径、原 F 对照和 Oracle、AP 及指标。
- run_all.sh：成功训练后自动评价，任何阶段出错立即停止。
- test_learned_nms.py：CPU 模型/匹配/掩码单元测试。

注意：由于作者官方仓库当前不可访问，本实现需在服务器先跑 SMOKE=2 和测试确认，再进行完整训练。源码可用性、实际消耗及 AP 结果不得凭推测填写。
