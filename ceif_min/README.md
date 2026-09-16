# 最小可训练 CEIF

目标：真正训练“观测约束 → 局部特征修正”，再测实际 AP。它不再使用真值选择动作，也不把整块特征替换成某辆车的特征。

## 1. 实现内容

1. 冻结现有 GSPR、PillarVFE、BEV 骨干、AttFuse、上采样层及检测头。使用全通信，与上一轮审查基线一致。
2. 用 train 的 clean 实际端点/可信射线，训练一个小型占用查询头。它从融合特征和查询位置判断该局部位置是否被占用。标签来自传感器观测，不把车辆框当作实心物体，不把未观测空间当空。
3. 冻结查询头。训练轻量残差网络，并在其输出上执行一次观测投影：端点下界推动占用不足的特征，穿越上界修正占用过强的特征。
4. 修正后的特征通过原检测头，以真实检测标签训练新模块。GT 只在训练损失和评价中使用，不进入查询选择、证据构建、前向修正。

“观测投影”就是沿查询头指出的方向，给局部特征一个小幅修正，使它更符合实际观测。本版采用有阻尼的一步线性修正；多查询重叠时平均修正，因此不是声称精确解出了全局最优问题。

未知 [0,1] 严格不产生观测修正；冲突区间保留冲突量并向中间解释作减弱的修正。投影只写入查询周围四个特征位置；小型残差网络可以改变其他位置。二者在日志中分别记录幅度。

## 2. 与审查代码的关系

- 复用 `ceif_audit/core.py` 的实际源原点、retained 点与原始回波关联、首回波检查和接收掩码语义。
- 不调用审查代码中的 GT 挑选、预测框动作生成和整块替换。
- 约束下界来自 5 cm 邻域端点支持，上界来自 20 cm 横向容差内的可信射线，远端留 1 m 裕量。这些是固定原型参数，没有用 test 扫描选择。
- 查询共约 1024 个/帧，覆盖各车端点和真实射线片段；一半偏向不同的高置信块，一半随机探索，查询选择不使用 GT。
- 查询头使用连续位置查询，但空间分辨能力仍受 BEV 特征限制。`query_epoch_*.json` 报告其 clean validation 损失、观测伪标签准确率及正负样本数，便于区分查询头失败和融合训练失败。

## 3. 必需对照

训练两个同初始化、同残差网络、同检测监督、同观测辅助损失的模型：

| 名称 | 可训练残差网络 | 观测辅助损失 | 前向观测投影 |
|---|---|---|---|
| baseline | 无 | 无 | 无 |
| aux_only | 有 | 有 | 关闭 |
| ceif | 有 | 有 | 开启，步长可学习 |

查询头先统一训练并冻结，两个模型共享同一个查询头。原始前端不更新。若 CEIF 只超过 baseline、没有超过 aux_only，不能说显式观测修正起到了作用。

默认训练：查询头 1 个完整 epoch；两个融合修正模型同时训练 3 个完整 epoch；学习率 2e-4；隐藏通道 32；观测损失权重 .1；batch size 1 帧，最多 5 车。共享每帧冻结编码，分别优化两个模型。

## 4. 数据协议（用户已确认）

- train：`/data/scd/datasets/opv2v_official_data_dumping/train`。沿用项目现有 GSPR 数据管线的物理 fog/rain/snow 模拟器，每帧训练 clean 和随机一种天气分支；每个 epoch 更新模拟器 epoch。不是重训 GSPR 的点监督任务。
- validation：`/data/scd/datasets/opv2v_official_data_dumping/validate`，分别完整评估 clean/在线 fog/rain/snow。仅用于查询头和融合模型选择，不做梯度更新。
- 最终 test：clean 为原 OPV2V test；fog/rain/snow 分别读取 `/data/cjm/datasets/opv2v-w/{fog,rain,snow}/test`，关闭在线天气增强和随机数据增强。
- 各模型独立选择四条件 validation AP70 算术平均最高的 epoch；test 不参与选择。报告十二项 AP，不把天气平均增长掩盖成 Clean 无损。
- 全流程使用原非全局排序、平面多边形 IoU AP。正式运行全场景全帧，stride=1；SMOKE 结果不能判断有效性，且拒绝把 smoke 权重用于正式 test。
- 为保持射线与真实位姿一致，沿用现有通信实验配置，关闭随机世界翻转/旋转/缩放；不宣称完全复刻原 GSPR 所有训练超参数。

## 5. 同步代码

将代码包中的 `ceif_min/` 和配套 `ceif_audit/` 放到：

```text
/home/cjm/OpenCOOD-main/cjmnet/
```

必须保留项目原来的 gspr_evidence、gspr_communication、attfuse_gspr 和 OpenCOOD 依赖。本代码包不是独立的完整 OpenCOOD 仓库。

默认前端 config/checkpoint：

```text
/data/cjm/datasets/logs/gspr_joint_full_v1_seed20260907_20260907_161155/config.yaml
/data/cjm/datasets/logs/gspr_joint_full_v1_seed20260907_20260907_161155/net_best_validation.pth
```

可通过 FRONTEND_CONFIG / FRONTEND_CHECKPOINT 环境变量覆盖。模型与源码指纹写入 protocol.json、checkpoint；恢复和测试必须一致。不要在训练途中替换 Python 源码。

## 6. 后台命令

### 6.1 先运行小样本连通检查

这会实际执行查询头训练、两个模型反向传播、保存权重和开发评价，但每个 split 只取前 2 帧。

```bash
cd /home/cjm/OpenCOOD-main/cjmnet
JOB="/data/cjm/datasets/logs/ceif_min_smoke_$(date +%Y%m%d_%H%M%S)"
nohup env ROCR_VISIBLE_DEVICES=0 OUT="$JOB" SMOKE=2 EPOCHS=1 DECODER_EPOCHS=1 QUERIES=128 \
  bash ceif_min/run_dev.sh > "${JOB}.log" 2>&1 < /dev/null &
echo "PID=$!  TRAIN=$JOB  LOG=${JOB}.log"
```

### 6.2 完整训练及开发评价

```bash
cd /home/cjm/OpenCOOD-main/cjmnet
JOB="/data/cjm/datasets/logs/ceif_min_train_$(date +%Y%m%d_%H%M%S)"
nohup env ROCR_VISIBLE_DEVICES=0 OUT="$JOB" \
  bash ceif_min/run_dev.sh > "${JOB}.log" 2>&1 < /dev/null &
echo "PID=$!  TRAIN=$JOB  LOG=${JOB}.log"
```

把卡号 0 换为空闲卡；不要同时在同一张卡运行 smoke 和全量。
默认训练后自动对最优权重做四条件完整 validation，再自动执行固定权重的 OPV2V-W 正式测试。
输出目录分别为 `${JOB}_validation_时间戳` 和 `${JOB}_opv2vw_test_时间戳`。
如只想先完成开发评价，可设置 `TEST_AFTER_TRAIN=0`；SMOKE 始终跳过正式测试。

查看日志：

```bash
tail -f "${JOB}.log"
```

查询预训练、融合训练及 validation 均打印进度和 ETA。总耗时包含完整 train 和每 epoch 四条件 validation，请以服务器实际速度为准。

### 6.3 单独启动 OPV2V-W 测试（自动测试未执行或中断时）

设置 RUN 为完整训练目录，不能设成 validation 输出目录。

```bash
cd /home/cjm/OpenCOOD-main/cjmnet
RUN="/data/cjm/datasets/logs/ceif_min_train_你的实际时间戳"
TEST_JOB="${RUN}_opv2vw_test_$(date +%Y%m%d_%H%M%S)"
nohup env ROCR_VISIBLE_DEVICES=0 RUN="$RUN" OUT="$TEST_JOB" \
  bash ceif_min/run_test.sh > "${TEST_JOB}.log" 2>&1 < /dev/null &
echo "PID=$!  RESULT=$TEST_JOB  LOG=${TEST_JOB}.log"
```

这里没有 GT 事后挑选。输出就是两个已训练模型和原基线的实际检测结果。

### 6.4 中断后恢复

使用原训练目录、完全相同的 EPOCHS/DECODER_EPOCHS/QUERIES/SMOKE 设置：

```bash
JOB="/data/cjm/datasets/logs/ceif_min_train_你的实际时间戳"
nohup env ROCR_VISIBLE_DEVICES=0 OUT="$JOB" RESUME=1 \
  bash ceif_min/run_dev.sh >> "${JOB}.log" 2>&1 < /dev/null &
echo "PID=$!"
```

按完整 epoch 恢复，不是逐 batch 恢复；中断的 epoch 从头执行。已完成训练恢复不会重复更新参数，会重新做开发评价，并按照 TEST_AFTER_TRAIN 设置决定是否测试。

## 7. 回传文件

优先回传最终 OPV2V-W 目录的 `results.md`。另外保留：

- 训练目录 `query_epoch_1.json`：查询头是否学会了基本占用判断。
- 训练目录 `epoch_1.json` 到 `epoch_3.json`：训练损失、四天气 validation 和选模记录。
- 正式输出 `results.json`：每条件 AP、实际帧数、checkpoint 指纹、查询覆盖、冲突比例、额外几何成本和修正幅度。
- `diagnostics.ceif.projection_step` 与 `projection_rms`：网络是否实际使用了观测投影，而不是把它学到接近关闭。

验收看三个问题：CEIF 是否超过原基线；是否超过 aux_only；是否达到你要求的三种天气 +5 个百分点，同时核查 Clean 的代价。结果不足就按真实结果判断，不用 oracle 收益代替实际 AP。

## 8. 当前边界

这是一次固定训练预算的最小原型，只有一轮局部观测投影，不是全部设想的多尺度迭代求解器。全通信之外的几何证据作为额外侧信道，单独报告估算字节，尚未实现带宽压缩或在 A0B0 同总预算下验证。

本地可核验前向、梯度、冻结参数、保存恢复和合成流程；真实训练、HIP 性能及 OPV2V-W AP 需要在服务器运行。本地通过测试不等于已经提升了 AP。
