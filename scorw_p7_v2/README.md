# SCOR-W P7 v2：连续帧通信检查修复

v1 的选帧代码检查了时间连续性，却未检查选中的两帧是否各有协作邻车。
后续冻结的坐标审计要求 ego 的 70 米 XY 范围内至少有两辆 CAV（包括 ego），因此停止。
用户回传的异常发生在正式运行的 P7-0，单凭 traceback 无法确定具体失败 scene/frame。
本修订不推测失败身份，也不把此次失败当作天气实验结论。

## 修订范围和科学边界

- 新代码及新运行合同全部放在 `scorw_p7_v2/`；v1/P6/reference/模型及旧实验产物不修改。
- 只读复用 `scorw_p7/` 的场池、分配、光路抽样、积分、指标、统计和报告实现。
- 沿用 v1 根据历史分区与初始布局确定的同一组 scene，不重排或替换 scene。
- P6 的四个开发 scene 及历史两帧保持冻结；任一开发帧不合格时硬停止，并记录身份和距离。
- 对原定八个新验证 scene，先检查 v1 的原定连续两帧。两帧都合格则保留。
  否则从原定起始索引开始，按起点递增、尾部回到开头的固定顺序检查相邻两帧块，取第一个合格块。
  每个块内部保持真实时间顺序，最后一帧与第一帧不组成跨尾首的块。
- 合格条件与冻结坐标审计完全相同：每帧 ego 的 XY 欧氏距离 **<=70m** 内至少两辆 CAV。
  两帧可以有不同的 active CAV，原有固定目标/有效配对与缺失值规则仍由冻结数值模块执行。
- 选帧仅用原始 train 的时间戳和 `lidar_pose`。不计算天气、GT 点数、信号差异或 AP 后挑帧。
  原 YAML 读取器会解析并验证元数据，选帧只使用其中的位姿。
- 如果原定八个验证 scene 中任一个没有合格连续帧块，八个全部不执行验证。
  保留开发结果及八个候选 scene 的检查报告，明确返回 `INCONCLUSIVE_VALIDATION_UNAVAILABLE`。
  这表示本次冻结的候选名单不可执行，不声称整个 train 已无其他候选 scene，也不改用 future 保留场景。
- 20 场、seed 规则、场物理参数、0.5m 积分步长、光路预算、70m 半径、数值容差和科学决策门槛不变。
  `load_config` 比较 v1/v2 协议，允许的差别仅为版本/输出路径和本次选帧规则。
- 这是在 v1 P7-0 失败之后登记的协议修订，不追溯声称 v1 已采用此规则。
  v2 实际验证帧在任何 v2 天气积分前冻结；本地未读取远程天气结果。
- 通信条件属于实验适用范围。通过这一检查不保证有足够 GT/真实回波或有效目标配对。
  后续 GT、坐标、平衡和来源检查继续严格执行；样本不足时只能保留不确定结论。

## 文件职责

`select_scenes.py` 补充通信预检、完整候选块诊断、元数据哈希、坐标审计参与车辆复核。
`run.py` 保留 P7-0 至 P7-4 的调度，并接入新预检及独立运行合同。
`decision.py` 只补全验证不可执行的真实原因，所有科学阈值沿用 v1。
`audit_sources.py` 严格核验新目录以及冻结 P7 snapshot，继续调用原来源审计和历史实验身份检查。
`pipeline.py` 顺序执行 checks → SMOKE → full，验证每一步产物哈希以及两次运行的冻结计划一致性。
`run_all.sh` 是唯一常用入口，`run_stage.sh` 提供单阶段入口。
`tests/` 包含本修订回归测试及隔离的启动器测试，服务器 checks 同时运行原 P7 数值测试。
`source_snapshot.json` 是交付时固定的源码清单，运行时不会生成或更新它。

## 同步和一键后台运行

把本地 **整个 `scorw_p7_v2/`** 同步到服务器项目根目录。
服务器仍需保留原交付的 `scorw_p7/`、`shared_weather_benchmark/` 及它们的冻结依赖；不编辑旧 snapshot。
所有命令都在服务器执行，不要求 `.git`，CPU 运行，不会训练、评测检测器或写天气 PCD。

```sh
cd /home/cjm/OpenCOOD-main/cjmnet
mkdir -p /data/cjm/datasets/logs/scorw-p7-v2
nohup env MODE=all RESUME=1 sh scorw_p7_v2/run_all.sh \
  >> /data/cjm/datasets/logs/scorw-p7-v2/pipeline.log 2>&1 &
```

这条命令自动执行检查、独立 SMOKE、正式 P7-0 至 P7-4 和小报告归档。
每步失败即停止。SMOKE 不替代正式统计；数值失败不会自动放宽精度。
首次运行也可用 `RESUME=1`，会创建新合同；已有已提交阶段先核验哈希再复用。
恢复时确认旧进程已退出，再执行同一条命令。并发重复启动会因阶段锁拒绝。
v1 的输出和失败记录全部保留；v2 不加载 v1 的运行缓存。

```sh
tail -n 100 -f /data/cjm/datasets/logs/scorw-p7-v2/pipeline.log
```

可选单步入口：

```sh
cd /home/cjm/OpenCOOD-main/cjmnet
MODE=checks RESUME=1 sh scorw_p7_v2/run_all.sh
```

耗时的单步 SMOKE/full 同样使用后台命令；通常直接运行上面的 `MODE=all`：

```sh
nohup env MODE=smoke RESUME=1 sh scorw_p7_v2/run_all.sh \
  >> /data/cjm/datasets/logs/scorw-p7-v2/smoke.log 2>&1 &
nohup env MODE=full RESUME=1 sh scorw_p7_v2/run_all.sh \
  >> /data/cjm/datasets/logs/scorw-p7-v2/full.log 2>&1 &
```

两个单步命令按需分别使用，不并发运行。
`RESUME=0` 只接受全新目录，禁止覆盖已有结果。
修改源码、协议、任何参与选帧的 YAML、原始输入或运行环境后，旧合同都不可复用。
不要删合同、改哈希或扩大半径来绕过失败。

## 诊断及结果路径

- 正式输出：`/data/cjm/datasets/scorw-p7-v2/`
- SMOKE：`/data/cjm/datasets/scorw-p7-v2-smoke/`
- 日志/调度状态：`/data/cjm/datasets/logs/scorw-p7-v2/` 及其 `pipeline/`
- 临时目录：`/data/cjm/datasets/tmp/scorw_p7_v2/`
- P7-0 内部选帧报告：`stages/P7-0/P7_SCENE_SELECTION.json`
  （即使坐标审计失败，这份已经写出的报告也保留。）
- 成功归档后根目录也发布 `P7_SCENE_SELECTION.json`；`connectivity_preflight` 包含原帧、新帧、
  各被查帧的 CAV 距离/参与身份、全部尝试的连续块及不可执行原因。
  `metadata_inputs` 包括被否决帧的 YAML SHA256，防止恢复时悄悄换帧。
- 其余主报告名沿用 P7：`P7_PROTOCOL_FREEZE.json`、`VALIDATION_GATES.json`、
  `P7_PER_SCENE_RESULTS.json`、`P7_STATISTICAL_ANALYSIS.json`、`RESULTS.md`、
  `NEXT_STAGE_DECISION.md` 等。
- 小结果包：`P7_SMALL_RESULTS.tar.gz`；启动器汇总：日志目录的 `pipeline/RESULT_SUMMARY.txt`。
  只同步小报告到 GitHub；原始数据、数组和大型缓存留在数据盘。

## 本次验证范围

遵照用户要求，本地只做 Python 语法、JSON、shell 文本、LF 和静态源码哈希检查。
本机 Git sh 无法启动，未取得本地 `sh -n` 通过结果；两个脚本在服务器启动时均先执行 `sh -n "$0"`。
没有在本地或远程运行数值实验、真实数据实验、训练或 AP；回归测试的实际通过结果以服务器
`logs/scorw-p7-v2/checks/CHECKS.json` 为准。本地静态记录在 `local_validation/STATIC_REVIEW.json`。
服务器 checks 会检查平衡场池原有数学性质，以及本次边界、双帧、身份、不可执行、恢复和启动顺序。

继续/不确定/停止的科学规则完全沿用 P7：只有控制与样本质量合格、关联和互补结构差异稳定且达到
预登记实际意义门槛时，才给出 `CONTINUE_TO_DETECTOR` 的下一步资格；本脚本仍不执行检测。
控制通过但区间宽/个别 scene 主导/配对不足时保留 `INCONCLUSIVE`。
只有足量样本下相关与互补效应区间均落入预登记小效应范围，才给出 `STOP_OR_REDESIGN`。
统计不显著不能当作无效证明；不追加 seed、换场景或调参数寻找正结果。
