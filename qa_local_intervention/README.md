# Stage-3C：局部证据干预诊断

这次回答：**同一种修改，如果只作用于漏检车辆附近，能不能保留补回能力，同时减少对其他车辆的误伤？**
它承接 Stage-3B 的分数、几何与权重干预，使用固定模型、固定来源特征，不训练网络。
它不能保证一次确定唯一根因，也不直接给出可部署方法。

## 实验内容

固定抽取 64 个失败候选帧（Fog 16、Rain 16、Snow 32），另加 24 个不重叠的成功控制帧（各天气 8）。
每帧预先指定一个焦点目标，遍历它的所有有效 peer；peer 有效性沿用 Stage-2 独立检出记录。
75% 候选优先来自主要失败阶段（Fog/Rain NMS，Snow 分数过滤），余下覆盖其他阶段；不足时补足。
在各抽样池内按全局可恢复/不可恢复、场景、距离尽量均衡，固定 seed=20260919。
这是诊断分层抽样，不能解释为总体恢复率。抽样和全部输入 SHA 保存在 plan/sampling_plan.json。

控制帧来自 Stage-2 中 clean ego 检出、weather ego 漏检，但 full 和至少一个 peer 检出的目标；
整帧不得含任何 Stage-3B 失败候选。它们是“融合已成功恢复”的对照，**不代表所有正常帧**。
控制帧会报告所有干预的损伤，不只报告事后挑选出的安全动作。

| 动作 | 改动位置 | 固定内容 |
|---|---|---|
| 局部分数替换 | 目标区域所有 anchor 的 psm | full 的 rm |
| 局部几何替换 | 目标区域所有 anchor 的 rm | full 的 psm |
| 局部权重插值 | 尺度 0、1、0+1 的目标区域 | 来源 features、keys、values |

区域是 GT 朝向矩形的 1.0、1.5 倍；输出用实际 anchor 中心定位，特征层用 BEV 网格中心定位。
如果区域内没有网格中心，使用最近中心，并在 roi_stats 记录 fallback。
所有 anchor 类型在同一个空间位置一起修改，不按 GT IoU 选择某个“好 anchor”。
权重是原 ego-query 权重与同 peer-query 权重的 0.5 或 1.0 插值；不是 query 向量插值。
特征区域外保持原融合特征，但后续反卷积、卷积和 NMS 仍可能影响区域外输出，不能视为严格隔离。

每 peer 共 16 个局部动作，配 8 个同 peer/同尺度/同强度的全局动作。
经版本、输入与 full 输出一致性检查后，复用 Stage-3B 已存的对应全局分支；缺失的全局对照重新计算。
保留所有原 full 检出的 GT、最终框、固定 anchor、真实抑制者的变化。
新增 FP 沿用 Stage-3B 的几何匹配定义，不用 FP 净变化冒充新增 FP。

每帧只能选一个完整动作或 KEEP_FULL，不能拼接多个目标的最优输出。
选择顺序是补回候选最多、全部 GT 净增最多、原 TP 损失最少、新增 FP 最少、平局保持 full。
另报“原 TP 零损失且新增 FP 为零”的约束选择；这不保证原 TP 分数不下降，所以另存逐 GT 分数变化。
所有选择使用 GT，是诊断上限，不能解释成无 GT 的策略效果。

## 启动

把整个 `qa_local_intervention/` 文件夹同步到服务器项目根目录。
**不要用本地 stage3_sources.json 覆盖服务器已修复的历史清单，不修改历史 Stage-3 文件。**
脚本校验完整 Stage-3B、其 Stage-3A 输入文件、源码、配置、checkpoint；版本不一致直接列出错误停止。
每个候选重放检查输入 SHA、目标状态、全部 full 最终框和匹配；沿用历史 sensing/单源重放保护。
只有历史 runtime 中已确认的 Stage-2 FP count warning 保持原样，其他一致性检查不放宽。

服务器 Bash 后台执行（3 号卡需空闲，可换 0–6；不使用 7）：

```bash
cd /home/cjm/OpenCOOD-main/cjmnet
RUN_NAME="qa_stage3c_local_$(date +%Y%m%d_%H%M%S)"
nohup env ROCR_VISIBLE_DEVICES=3 RUN_NAME="$RUN_NAME" SMOKE=0 \
  bash qa_local_intervention/run_all.sh \
  > "/data/cjm/datasets/logs/${RUN_NAME}_launcher.log" 2>&1 &
echo "PID=$! RUN_NAME=$RUN_NAME"
tail -f "/data/cjm/datasets/logs/${RUN_NAME}_launcher.log"
```

`SMOKE=0` 自动依次完成单元测试、固定抽样、每天气 2 候选+1控制预检、预检报告校验，再跑正式 88 帧。
只想预检时设 `SMOKE=2`。预检报告明确标注不能作科研结论。
两者都按历史完整队列生成天气，不能把 loader 改为仅加载抽样帧，否则随机天气可能改变。
因此少量帧也可能运行较久。日志报告每帧进度与粗略 ETA；不同帧 peer 数、NMS 框数不同，ETA 不是承诺。
失败立即停止；已有逐帧文件保留，但不支持断点续跑。修复后使用新的 RUN_NAME，禁止覆盖旧产物。

默认 Stage-2/3B 路径已填入用户已完成实验的位置；若移动目录，通过 STAGE2_ROOT/STAGE3B_ROOT 设置。
历史日志中的绝对路径也需仍可访问，不能只迁移部分文件。

## 结果怎么看

- `stage3c_report.md`：可直接回传的汇总报告。
- `stage3c_results.json`：各天气配对、机会、整帧代价、控制误伤和框变化计数。
- `天气/frames/*.json`：每个动作最终框、目标阶段、固定 anchor、原 TP 分数变化、10 米内外损伤。
- `plan/sampling_plan.json`、`天气/protocol.json`、`天气/summary.json`：抽样、配置、输入和逐帧哈希。

优先看：同 peer 配对的局部恢复是否保留、相同零损伤约束下的机会是否增加、控制帧是否被普遍误伤。
Fog/Rain 再看原抑制框是否因局部几何改变成为最终正确框；Snow 看固定正确 anchor 是否保持定位并跨过分数门槛。
局部输出替换有效但局部权重干预无效，支持继续研究更靠近输出端的改动，不能证明融合完全无关。
局部权重有效且代价下降，才支持下一步研究“不用 GT 如何决定位置、来源和强度”。
若只扩大区域后有效，需继续区分空间定位误差、感受野与区域外竞争，不能直接认定信息不存在。

本地验证命令：`python -m unittest qa_local_intervention.test_local -v`。
无 PyTorch 的机器会明确跳过张量测试；服务器启动脚本要求 Torch、Shapely 和 GPU/HCU 可用，并运行全部测试。
