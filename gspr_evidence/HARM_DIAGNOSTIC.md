# 全通信删除诊断

目的：决定当前冻结 GSPR＋AttFuse 下是否有值得继续研究的质量过滤空间，不训练新模型。先等正在运行的正式测试结束，再安排空闲卡。

## 数据边界

**本轮明确是开发诊断**：固定 `/data/scd/datasets/opv2v_official_data_dumping/validate`，全部场景、全部帧、stride=1；分别 clean 和在线模拟 fog/rain/snow。程序拒绝将 test 路径用于这一入口。指标使用开发协议的全局排序 AP；不能与正式 OPV2V-W test 数值相减。

每帧只编码一次，冻结权重/BN/检测阈值，所有干预复用相同特征。删除三尺度中对应的同一物理区域，并通过原缺失块掩码融合。它是离线干预，不宣称是一个可部署的通信算法，也不把没有计费的筛选信息算成通信节省。第一帧各组额外核对真实包解码与快速掩码重放一致。

## 两部分

1. `filters`：全通信，以及按低可靠概率、高 u、随机删除的各三组。预设删除比例为 1%、5%、10%。**分母为每个邻车有保留点的区域数**，三个策略每车删除完全相同数量；无点区不具有点质量均值，不参与这轮排序。使用原始概率，而非 reliability_floor 后的权重。记录完整 AP30/50/70、相对全通信补回/丢失目标及 FP 改变、各场景 AP70。
2. `single`：每帧每邻车检查最多 6 个候选，来自低可靠性前2、高 u 前2、随机2；交集去重并保留来源。**每一次都从完整通信开始，只移除一个候选块**。同帧其余输入不变。记录 GT 检测损失改变，以及 IoU .7、固定后处理阈值下的目标匹配与误检改变。重复一次全通信前向测损失数值噪声；损失改善判定阈值为 `max(1e-4, 5*重复损失差)`。

6 限制的是每邻车每帧的候选数，不是只抽 6 帧。候选检查并不穷举所有块，无法证明所有差信息无害；逐块删除也不能穷举联合删除的非线性效果。没有构建或报告使用 GT 选块的 oracle AP。

## 推荐执行：一次完成，避免重复编码

同步新增文件后：

```bash
cd /home/cjm/OpenCOOD-main/cjmnet
export ROCR_VISIBLE_DEVICES=0
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
export STAGE=both
bash gspr_evidence/run_harm_diagnostic.sh
```

默认前端与正式测试相同：seed20260907 的固定 `config.yaml` / `net_best_validation.pth`。支持 `FRONTEND_CONFIG`、`FRONTEND_CHECKPOINT` 环境变量覆盖；不需要通信头 checkpoint 或教师缓存。在线 fog 仍需已有雾表，允许 `FOG_LOOKUP_DIR` 覆盖。

新输出目录为 `/data/cjm/datasets/logs/gspr_harm_validation_both_时间戳/`。其中 clean/fog/rain/snow 分别包含：

- `diagnosis.md`：过滤 AP 对照和逐块汇总。
- `summary.json`：完整统计。
- `scene_ap70.json`：各场景的过滤 AP，用于判断收益是否集中于个别场景。
- `frames.jsonl`：逐帧删除数、损失与 AP 原始统计。
- `candidates.jsonl`：每个独立删除候选的实际影响。
- `protocol.json`：数据、全部帧索引、源码/权重哈希及诊断范围。

程序显示处理帧数和 ETA；完整候选重放较耗时。默认一次完成全部诊断，整个流程没有帧数缩减。中断后需新目录重跑对应天气；未生成最终 summary/diagnosis 的目录是未完成结果，不可用于结论。

如果希望先只看过滤效果，可设 `STAGE=filters`；后续 `STAGE=single` 再查单块，两次都会重新读取和编码完整验证集。

## 判读与停止条件

1. 质量/u 过滤没有稳定 AP 收益，且与等量随机删除相近；逐块候选也很少改善实际目标匹配：支持在**当前冻结系统与诊断范围内**停止继续做质量过滤，保留 A0B0，转向下一阶段。
2. 单块检查发现明显有害候选，但低质量/高 u 来源不比随机更容易命中：点质量未能很好表达任务伤害，继续做质量排序可能收益有限。
3. 少量过滤有收益，删除比例增大后下降：有效信息丢失的代价可能超过过滤收益。
4. 损失下降但目标匹配/AP没有改善：不能把监督损失收益当成检测性能收益。

`candidate_groups` 中 `all_unique` 是去重候选；其他来源允许交叉，不能把它们的 count 相加作为总候选数。`loss_improved` 是损失指标；`detection_improved` 要求匹配目标数不减少、FP 不增加且至少一项严格改善，不等于 AP 提升。相反方向为 detection_worsened；其余包括无变化和精度/召回权衡。

这里不自动依据一个比例或平均损失宣判成功/失败。先看四类天气、各场景及随机对照；如决定采用一种过滤方式，应在开发阶段固定规则后再做正式测试，不能用 test 反复挑比例。
