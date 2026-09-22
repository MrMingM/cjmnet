# 方案一代码：预测一次局部融合修改的收益与代价

## 简单版

原模型先照常得到 full 融合结果。新模块把尺度 0 划成固定 8×8 网格；对每个局部网格和每个 peer，它预测三件事：这次修改可能补回几个目标、损失几个原有 TP、增加几个新 FP。预测净收益没有超过验证集门槛时，保持原 full。

动作不是“直接拿 peer 替换 full”。在选中的尺度 0/1 区域里，以该 peer 的特征作为查询，重新计算所有已收到来源的 AttFuse 权重；尺度 2 保持原结果。分类头、回归头和 NMS 都继续使用冻结模型。推理不读取 GT、天气类别、有效 peer 标签或事后动作。

## 文件与数据流

- `fusion.py`：精确重放 full AttFuse、构造修改前描述量、应用局部来源动作。
- `supervision.py`：只用预测活跃度采样 top/低分/背景网格；GT 只在完整后处理后生成三项训练标签。
- `outcomes.py`：按 BEV IoU=0.7 匹配 TP；新增 FP 通过 action FP 与 baseline FP 的空间身份匹配确定，不使用 FP 数量净差。
- `network.py`：所有 peer 和网格共享的轻量 1×1 网络。
- `prepare.py`：在完整 train/validation split 上生成可恢复缓存，每帧默认对 clean/weather 各评估 8 个动作。
- `train.py`：训练主方法 `utility`，或参数量和输入相同的普通检测损失改善对照 `loss_gain`。
- `calibrate.py`：只使用 validation 缓存选择 KEEP_FULL 门槛。
- `evaluate.py`：统一评价 baseline、简单置信度、主方法和 loss-gain 对照；可额外评价仅尺度 0、仅尺度 1、去掉 KEEP_FULL。

通信固定为 raw/full float32 三尺度特征。描述量和来源独立预测都在接收端计算，没有新增 peer 框、质量图或天气标签；报告仍记录全量特征字节数。额外成本主要是逐来源冻结检测头和第二次最终检测头前向。

## UECP 与 ICPB 源码

实现参考了 `related_code/UECP-main/opencood/models/utils/pyramid_fusion_modulev1.py` 中“多尺度、可靠性、残差保留”的实现边界，但没有复制其模块，也没有使用它的密度监督。这样可以避免把 UECP 已覆盖的多尺度不确定性融合直接当作本项目创新。

运行本实现不需要 ICPB 源码。ICPB 全文和作者实现拿不到时，应在论文中明确“未核实实现细节”，不能声称完成了代码级复现；它不会阻塞本方案训练。若以后要把 ICPB 设为正式复现基线，再补作者代码或自行实现并标为 reimplementation。

## 运行顺序

先在服务器设置冻结前端配置、checkpoint 和一个新的日志路径。完整缓存和训练耗时较长，建议后台运行：

```bash
cd /home/cjm/OpenCOOD-main/cjmnet
export FRONTEND_CONFIG=/absolute/path/to/frontend.yaml
export FRONTEND_CHECKPOINT=/absolute/path/to/net_epoch_bestval_at70.pth
export RUN=/data/cjm/datasets/logs/local_fusion_utility_seed20260913
export ROCR_VISIBLE_DEVICES=0
nohup bash local_fusion_utility/run_all.sh > "${RUN}.log" 2>&1 &
echo $!
```

正式测试必须等开发集流程完成并固定 checkpoint、代价系数和门槛后单独执行：

```bash
export RUN=/data/cjm/datasets/logs/local_fusion_utility_seed20260913
export OUT=/data/cjm/datasets/logs/local_fusion_utility_seed20260913_test
nohup bash local_fusion_utility/run_test.sh > "${OUT}.log" 2>&1 &
echo $!
```

缓存中断后可对 `prepare.py` 原命令加 `--resume`。`--smoke N` 仅用于开发检查，正式 benchmark 会拒绝 smoke。

## 当前实现边界

- 主方法、置信度对照、loss-gain 公平对照、KEEP_FULL/尺度消融已经接通。
- 计划中的“普通可训练 attention”与检测端小修复模块是独立基线，不在本目录伪装成主方法。小修复模块尚未有被锁定的训练接口和 checkpoint，因此这里没有猜测性接入。
- 单网格动作标签不能保证多个网格联合修改时收益可加；这是独立验证必须测的失败风险。
- `41/64` 的 Stage-3C 诊断恢复机会不进入训练或阈值选择，也不作为性能上限。
- 本目录不自动启动任何训练或正式评价。

