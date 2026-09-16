# CEIF 训练前可行性审查

这个实验回答：**当前数据里，是否存在能被跨车观测证据覆盖、并且通过改变融合特征实际改善 AP 的机会？**

不训练 CEIF，不改 GSPR/AttFuse 原源码，不续训通信选择器。它在冻结模型上进行局部反事实干预：同一帧编码一次，固定实际收到的特征，局部改用某个来源的特征，再通过原检测头得到新预测。

## 1. 实验层次

| 输出方法 | 做什么 | 使用 GT？ |
|---|---|---|
| baseline | 当前 full 或明确预算的 A0B0 + AttFuse | 仅评价 |
| full_reference | 同一输入上的全通信参照 | 仅评价 |
| generic_fp_oracle70 | 删除基线所有 IoU .7 假阳性，保留原框和分数 | 是 |
| evidence_fp_oracle70 | 只删除被局部观测冲突标记、且 GT 确认错误的框 | 是 |
| pool_random_feature | 同一有限候选池随机选一个特征替换 | 否 |
| evidence_rule_feature | 按固定观测证据优先级选择一个特征替换 | 否 |
| evidence_hindsight_feature | 在证据合格候选中事后挑选有益替换 | 是 |
| pool_hindsight_feature | 在同一完整候选池中事后挑选有益替换 | 是 |

GT 从不用于生成候选框、局部区域、观测证书或规则优先级。候选框来自原模型和对收到的单车特征使用同一个检测头得到的预测；没有插入 GT 框、提高 TP 分数、读取 clean 配对云或访问未发送的邻车特征。

“事后挑选”要求：IoU .7 下保留基线所有已匹配 GT，不增加 FP，并增加匹配或减少 FP。在满足条件的候选中优先更多新增匹配，其次更少 FP。它不是逐帧 AP 最大化，更不是完整 CEIF 的理论上限。重新累计整个 split 的真实 AP 后可能出现与成功次数不同的结论。

FP70 清理是按 .7 的匹配定义进行的，可能删掉 IoU .3/.5 的 TP，因此必须同时看 AP30/AP50/AP70。随机对照与其他方法共享证据优先构建的候选池，用来检查池内选择作用，不是完全脱离证据的空间采样基线。

## 2. 观测规则

- 直接从冻结 GSPR 最终点证据恢复 b_R/b_N/u；不用带 floor 的 reliability 冒充概率。
- 保留位于实际选中 block 内的端点。射线必须与原始实际回波对应，并通过更近首回波检查；不能借用同角度其他点的可靠性。
- 自车/邻车的坐标统一在 ego 坐标系，射线从真实来源传感器原点出发。
- 射线只检查端点附近的局部几何冲突，不把整个预测框内部当作实心或把整个 pillar 当空。
- 未收到、没回波、射线未覆盖都不作为反证。
- 参数预先固定：可靠信念 .6、候选端点信念 .15、射线半径 .2 m、远端裕量 1 m、至少 2 个跨车冲突点且比例 .3。补全候选要求邻车至少 3 个可靠端点、自车无可靠端点，且未发现足量强冲突。
- 这些规则是 CEIF 观测思想的保守代理，不是已训练的占用查询头。需要核查 `source_valid_rays` 和 `eligible_actions`，否则零收益可能只是覆盖不足。

每来源最多取 12 个高分预测框，每来源/框最多查询 32 个端点，每帧最多执行 8 个候选；纠错和补全轮流选取并按来源/区域去重。限定范围是为了控制审查开销，阴性结果不等价于否定所有潜在重建机制。

## 3. 运行前

同步新增的整个 `ceif_audit/` 到：

```text
/home/cjm/OpenCOOD-main/cjmnet/ceif_audit/
```

依赖沿用现有 OpenCOOD 环境（PyTorch、NumPy、SciPy 等）。运行脚本会先做单元测试；每帧核对本地特征重放与实际序列化消息的检测输出，首帧核对 full 与原 GSPR 模型输出。原冻结源码哈希与 checkpoint 哈希继续检查。

默认 checkpoint：

```text
/data/cjm/datasets/logs/gspr_joint_full_v1_seed20260907_20260907_161155/net_best_validation.pth
```

默认 full，是因为项目目前没有回传最终 A0B0 预算，不能擅自沿用已经不被接受的 256 KiB。如果已经确定预算，通过 BASELINE=a0b0 和 BUDGET_BYTES 指定。

## 4. 后台运行

在服务器项目目录执行。将 ROCR_VISIBLE_DEVICES 改成当前空闲卡。

### 4.1 建议先做四条件各 2 帧连通检查

```bash
cd /home/cjm/OpenCOOD-main/cjmnet
JOB="/data/cjm/datasets/logs/ceif_smoke_$(date +%Y%m%d_%H%M%S)"
nohup env ROCR_VISIBLE_DEVICES=0 PHASE=development BASELINE=full SMOKE=2 OUT="$JOB" \
  bash ceif_audit/run_audit.sh > "${JOB}.launch.log" 2>&1 < /dev/null &
echo "PID=$!  LOG=${JOB}.launch.log  RESULT=$JOB"
```

这是连通检查，不能据 2 帧 AP 判断方案是否有效。

### 4.2 完整开发审查（默认推荐）

```bash
cd /home/cjm/OpenCOOD-main/cjmnet
JOB="/data/cjm/datasets/logs/ceif_development_$(date +%Y%m%d_%H%M%S)"
nohup env ROCR_VISIBLE_DEVICES=0 PHASE=development BASELINE=full OUT="$JOB" \
  bash ceif_audit/run_audit.sh > "${JOB}.launch.log" 2>&1 < /dev/null &
echo "PID=$!  LOG=${JOB}.launch.log  RESULT=$JOB"
```

完整 OPV2V validation、所有场景/帧、stride=1；fog/rain/snow 是在线天气模拟。使用非全局排序 AP，与旧 global-sort 删除诊断分开。**这组不是 OPV2V-W 正式天气 test。**

### 4.3 需要核实实际 OPV2V-W 数据时

保持默认规则不变，使用独立的 `benchmark-audit`：

```bash
cd /home/cjm/OpenCOOD-main/cjmnet
JOB="/data/cjm/datasets/logs/ceif_opv2vw_audit_$(date +%Y%m%d_%H%M%S)"
nohup env ROCR_VISIBLE_DEVICES=0 PHASE=benchmark-audit BASELINE=full OUT="$JOB" \
  bash ceif_audit/run_audit.sh > "${JOB}.launch.log" 2>&1 < /dev/null &
echo "PID=$!  LOG=${JOB}.launch.log  RESULT=$JOB"
```

读取固定 clean test 和 OPV2V-W fog/rain/snow test，关闭在线增强，全部帧，非全局排序 AP。由于这里含 GT 事后选择，结果明确标为探索性审查，不作为 CEIF 正式模型性能；不据该表反复调阈值或选模型。

### 4.4 已有选定的 A0B0 预算

在以上任一命令中把 `BASELINE=full` 换成 `BASELINE=a0b0 BUDGET_BYTES=实际选定的字节数`。不要把示例数字当成已选预算。本审查不会自动在 test 中扫预算。

### 日志查看

在上述同一终端中：

```bash
tail -f "${JOB}.launch.log"
```

程序每 20 帧打印进度、平均时间、ETA 和有益证据候选数量。不同任务使用独立 JOB 目录；不要在同一张卡同时启动 smoke 和全量任务。估计耗时以实际前 20 帧为准。

## 5. 输出与判断

回传：

```text
feasibility.md
clean/audit.md
fog/audit.md
rain/audit.md
snow/audit.md
```

每个天气还保存 protocol.json、summary.json、frames.jsonl、actions.jsonl 和各方法 eval.yaml。候选日志包含修正/丢失 GT 数、FP 改变及 AP 输入，便于按场景复查。

判断顺序：

1. 只有 generic_fp_oracle70 上涨：只能说明误检存在，不能支持 CEIF。
2. evidence_fp_oracle70 上涨：实际证据至少覆盖了一些有价值的错误。
3. evidence_hindsight_feature 上涨：证据所指区域内，当前收到的表征确实允许通过融合改变产生收益。
4. evidence_rule_feature 上涨且优于同池随机，并且 clean 和其他天气没有明显代价：可识别性更强，值得进入具体 CEIF 训练。
5. 只有事后选择上涨、规则明显下降：机会存在，但判别机制尚未可靠；先查证据误伤和覆盖，不把 oracle 差值当作预计 AP 增幅。

不要用固定“涨 0.x 就通过”的阈值自动作决定；同时看 TP/FP/FN、分场景一致性、候选覆盖以及十二项 AP。

## 6. 审查范围

本实验检验接收端局部融合纠错/补全机会，不等于实现 CEIF 的全部反演能力。新模型未来可能产生本候选池不存在的结构，因此本实验不是穷尽性否定工具。

几何信息使用额外理想证据侧信道。仅从已选 block 取端点，且跨车几何查询也要求对应源 block 已收到。发送端可用完整本地云计算有效射线标记，但不把完整云交给接收端。按 xyz+三元意见+一字节有效射线标记估算每点 25 字节，加每源 64 字节位姿；不含所有额外协议开销，也没有从原特征预算中扣除。该审查先判断信息是否有用，不声称同总带宽收益。

局部特征替换会完整经过原检测头，可能带来新误检或漏检，程序不会隐去这些副作用。GT 只作为离线评价与理想选择依据，不能在部署中使用。

本地测试覆盖几何语义、接收掩码隔离、真实消息重放、重复框匹配与事后选择；全量服务器 AP 需要上述真实运行后确认。
