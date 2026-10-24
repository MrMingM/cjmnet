# GSPR REPE

2026-10-24：已新增NVIDIA CUDA入口，默认BACKEND=cuda。租用服务器请按 [CUDA_REMOTE.md](CUDA_REMOTE.md) 上传新包和启动；下文旧海光命令仅用于BACKEND=rocm。CUDA模式独立生成数据/环境/GT记录，不要求历史D−1A/B/C运行产物。

本目录实现独立的可靠性感知特征提取及两阶段串行管线。未修改旧实验、权重、结果，不创建分支或 worktree。

## 结构

`modules.py`：A柱内可靠性聚合、B观测双头、C八邻域残差增强。
`models.py`：官方GSPR子类适配器；A关时保留旧GSPR，B/C位于Scatter之后、官方AttFuse之前。A门控及C增强门控bias初始化−5，初始改动很小。B3/B4模块参数量一致。
`data.py`：保留官方混合天气及审计。新增训练Clean参考读取，保存/恢复Python与NumPy随机状态；重新抽帧或车辆数量不一致时屏蔽配对监督。
`training.py/checkpoint.py/evaluation.py`：从L0–L2独立适配，保留完整状态恢复、每轮提交和官方O/O*/F评测。
`orchestrator.py`：阶段日志、SHA256校验、锁和失败停止。
`server_tests.py`：服务器单元/集成/完整状态恢复及失败恢复测试。
`diagnostics.py`：利用官方匹配轨迹输出逐目标恢复/丢失及按距离、框内点数分层召回，不另造匹配规则。

## 固定实验协议

五组均随机初始化共享主干/GSPR，seed20261017、20轮、batch8、官方Adam及学习率调度；不复用不同配方历史成绩。新增模块初始化不推进CPU随机状态。实际训练路径和随机状态写入sampling.jsonl并在汇总比较，不预先承诺GPU逐位确定。

B0旧GSPR；B1+A；B2普通局部增强；B3+A+B+目标状态门控的普通增强；B4+A+B+C完整约束。主训练Clean/Fog每车4/7与3/7，Snow仅锁定权重后验证。只按829帧Clean O AP70选轮次，平局选更早轮次。正式test不读取。

保留原GSPR正则。Q/S辅助损失权重各.01，分别记录。Q使用停止梯度的GSPR可靠性代理；S以训练Clean参照的有效观测比监督，仅对Clean参考有至少2点且天气有观测的位置监督；未对齐时采用同距离段中位有效数量作弱代理。Clean不作为完整场景真值。完全空区域不强行补全。

B3与B4同参数、同八邻域循环，差别为可信来源限制。当前B3−B2同时改变A/B，不能据此单独排名B的贡献。该限制随最终报告保留。

## 单一入口和服务器命令

整套项目同步到服务器后，在服务器运行下列一条后台命令。GPU=0须按当时空闲卡修改；脚本只设置ROCR_VISIBLE_DEVICES。

```sh
cd /home/cjm/OpenCOOD-main/cjmnet && mkdir -p /data/cjm/datasets/logs && nohup env BACKEND=rocm GPU=0 MODE=all RESUME=1 sh gspr_repe/run_all.sh > /data/cjm/datasets/logs/gspr_repe_v1.launch.log 2>&1 < /dev/null &
```

默认输出 `/data/cjm/datasets/logs/gspr-repe-v1`。需要新实验时设置新的OUTPUT_ROOT，不覆盖历史目录。同一命令可恢复中断；源码、配置、数据或已完成产物校验改变会拒绝复用，须另开输出目录。

MODE=preflight只校验；MODE=smoke执行数据检查→结构测试→五组各16训练帧/一轮及12帧三天气推理→第一阶段验收；MODE=all在同样验收通过后继续五组正式训练与评测。默认串行单GPU，技术错误停止；低AP保留并继续。

首次启动及各训练/评测保留历史数据和环境哈希门禁，不会自动安装依赖或重建旧数据。数据布局沿用L0–L2已验收环境。preflight会扫描大量文件，耗时可能较长。

## 输出

- `logs/*.log`、`current_stage.json`、`stages/*/{complete,failure}.json`：进度和失败原因。
- `self_test/summary.json`、`smoke/B*/`、`stage1/{acceptance.json,STAGE1_REPORT.md}`：第一阶段，成功后才生成验收报告。
- `B*/epochs/`、`resume_states/`、`tensorboard/`、`selected.json`：正式训练和恢复状态。
- `B*/final_evaluation/`：官方AP、逐帧匹配、缓存和repe_inference.json，后者仅描述本次实际未缓存前向的同步耗时/显存/状态统计。恢复时不把少量新前向计时误称全量计时。
- `reports/STAGE2_RESULTS.json`、`STAGE2_REPORT.md`、`ABLATION_TABLE.csv`、`FAILURE_CASES.md`、`FINAL_RESEARCH_ASSESSMENT.md`、`STRATIFIED_DETECTION.json`。

本地只执行静态检查。结构测试、ROCm前后向、真实数据训练和检测评测均未运行；没有AP或成功声明。Q/S与具体目标恢复的因果关系仍需额外干预验证，当前报告不会据AP上涨宣布机制成立。
