# SCOR-W R3-A → R3-B（args兼容修复）

一个入口串行执行接收器审计、数值/Hahner 回归、Clean 场景筛选、完整预算检查、回波缓存、Ego/Neighbour/Fusion 推理和逐场景分析。只新增 `scorw_r3/`，不改写 R0–R2、P0–P7 或模型权重。服务器自动流程不会下载数据、训练模型或调整天气参数。

空列表配置报错与已完成天气缓存的只读恢复说明见 [ARGS_FIX.md](ARGS_FIX.md)。源码与新输出合同分开保留，已通过的旧v1同模式天气缓存会在严格核验后复用。

物理依据与已发现的限制见 [PHYSICS_AND_SCOPE.md](PHYSICS_AND_SCOPE.md)。运行结果必须由服务器生成；本地合成测试不属于 OPV2V 实验结果。

## 数据、身份与恢复

- 真实数据固定为原始 OPV2V train。只在 `/home/cjm/OpenCOOD-main/cjmnet` 的 Linux 环境执行入口。
- 历史服务器结果固定为 `/data/cjm/datasets/scorw-repair-v2-budgetfix/full`。`inputs/R2_IDENTITY.json` 固定完整报告包、源码及合同身份。沿 `LEGACY_IMPORT.json` 的显式收据解析旧 coordinates/R1，不能只复制预算修复目录后删掉原运行缓存。
- 仍需保留收据所指 `/data/cjm/datasets/scorw-repair-v2/full/stages/{coordinates,R1}`，以及 budgetfix 的 R2 完整 cloud_cache。缺少必需数组或任何已有文件哈希变化立即停止。单独缺失旧 diagnostic_pcd 时，在新合同冻结缺失名单，15条取证报告相应记录不可复原；运行中缺失名单变化也停止。不会补造结果。原结果通过只读共享锁和提交清单保护。
- 新源码、配置、运行依赖版本、线程设置、旧报告、Clean 输入、场池、模型及每阶段产物均记录 SHA256。服务器不依赖 Git。模型路径和哈希从旧合同再次核验。
- 每个阶段、每个回波缓存、每个场景推理分别原子提交及加锁。RESUME 仅接纳完全相同合同；已提交目录新增/丢失/更改文件都拒绝。中断后未提交的阶段重跑，已提交子任务只读核验。
- 根目录报告是阶段结果的派生发布，使用发布日志避免中途断电留下半套报告。不要手工修改 snapshot 或 CONTRACT；源码/配置变化需要独立的新输出根和合同。

## R3-B 冻结的选择与天气对照

历史 P5/P6/P7、R0–R2 与 future 划分已核对并固化到 `inputs/HISTORY_FREEZE.json`。已知43个 train scene 中：24个已使用/审阅、14个future保留，余下只有5个待 Clean 资格检查。未知新目录只报告，不自动招募。最多5个新scene可能进入本轮，低于判断门槛6个，因此本轮独立验证自动决策将为 INCONCLUSIVE。不会解除 future 保留或重用旧验证场景凑数。

五个候选：`2021_08_19_15_07_39`、`2021_08_20_21_00_19`、`2021_08_21_16_08_42`、`2021_08_22_22_30_58`、`2021_08_23_11_22_46`。名单是本地已核验历史结果的差集，真实完整性仍待服务器检查。

scene、每scene最多12个候选帧按固定seed哈希排序。取首个通过 Clean 资格的帧：时间戳完整、70m内≥2CAV、固定ego范围内≥8个GT、ego与邻车≥3个共同目标且每车每目标≥3个正强度原始点。邻车按共同目标数降序、CAV ID打破平局。每scene固定1帧；最多4个scene加入第三辆合格车。新增第三车的视图与两车视图共用同一份车辆身份、GT和场池偏移，避免车辆重新分配天气。

B0/B1/B2 单次；B3/B4 各20次。完整接收结果每CAV每场只计算一次并保存NPZ及兼容PCD，再由两组重组。无新增散射随机数。逐CAV完整场池的回波、来源、强度与光学厚度边缘分布严格相同；单次repeat总体天气强度允许不同并实际报告。全体CAV共同循环移位作为负对照。

SMOKE 使用旧 R2 第一个开发帧、2CAV，完整20场以验证相同协议。优先核验后复制旧接收缓存；原始B0和B1按历史协议生成。SMOKE、FULL、preflight 分属三个输出合同，SMOKE 无新独立scene。

## 预算

所有 Clean 资格检查和坐标核验完成后、R3-B 波形计算前冻结：

- 全部光路×场≤6400万，每scene≤600万；每原始点20个非均匀场+1常数场。
- 缓存估算≤48GiB；每光路×场按512字节保守估算，磁盘额外保留20GiB。
- 单CAV/单场流式读写，主机内存估算≤8GiB。GPU沿用R2模型及前处理配置，首次运行记录设备；内存估算不等于硬件峰值保证。
- 输出完整推理调用上界、每scene点数/光路数/磁盘与内存估算。CPU耗时取决于各条光路的距离，不提供未经测量的小时数估计。
- 超预算即停止，保留R3-A及预算报告；不删scene、seed或自动减少点数。额外旧记录取证最多15条。

## 推理和统计

冻结纯 AttFuse 配置/权重，所有点云投影到相同ego坐标后，用同一套冻结预/后处理和固定GT。Ego-only、每个Neighbour-only、Fusion分别推理。Neighbour-only的含义是“邻车观测在共同ego坐标中的单CAV输入”，与邻车自身坐标运行的模型可能不同。所有范围保持历史R2 project-first协议。

无预测按空结果处理。AP30/50/70基于BEV IoU、冻结逐图分数贪心匹配和逐repeat全部固定场景的全局分数排序。不会用scene AP平均替代全局AP。对应B3/B4的ego输入/输出精确一致，repeat0额外复算验证。报告TP/FP/FN、救回、协同损失和新增误检；后者以协同FP与ego FP的IoU<0.5定义，与历史一致。

潜在融合未利用邻车信息必须是Ego漏检、至少一个Neighbour检测成功、Fusion漏检。框内点数只描述观测。原始候选框诊断枚举全部与GT的BEV外接框相交的有限候选，区分分数阈值、定位、GT匹配竞争和无法唯一定位的后处理/NMS原因，不声称已证明故障因果。无效候选记unknown。Shapely相交警告保留有限有效结果，NaN/无效GT或最终预测几何停止。

旧R2的15条记录对应6个唯一(scene,frame,object)、2scene。用原诊断PCD、模型和GT重跑旧Ego/Fusion，核对旧输出，再补Neighbour-only与候选框诊断；缺缓存则明确缺失。

20次结果均报告，不当作20个独立scene。主独立单位为scene。≥6scene才执行2000次整scene bootstrap，重采样时重算全局AP。报告逐scene方向、leave-one-scene、逐天气重复敏感性、目标事件集中度、两车/三车同scene嵌套比较。

继续门槛预先冻结：≥6新scene、≥48固定GT实例、≥12邻车可检测的唯一目标，|平均ΔAP70|≥0.01、95%整scene区间排除0、≥75%场景方向一致、删任一scene相对变化≤50%、最大单目标事件贡献≤25%，并通过全部公平性/身份/物理门槛。只批准进一步数据集研究，不代表真实物理标定。

停止需要足够样本且AP差异区间位于±0.01、目标事件差异区间同时位于±0.02，或确认不可接受物理错误。其余为INCONCLUSIVE。阈值是透明的项目决策规则，无真实统计校准；未显著不能解释成无效。

## 服务器一键命令

先把整个 `scorw_r3/` 同步到项目目录，旧代码保持原字节。默认Python `/home/cjm/miniconda3/envs/opencood/bin/python`，无需激活conda或安装新依赖。预检查会读取旧提交与原始输入哈希，耗时取决于缓存大小。

```sh
cd /home/cjm/OpenCOOD-main/cjmnet
MODE=preflight RESUME=0 sh scorw_r3/run_all.sh
```

SMOKE仍包含完整20场和冻结检测器，建议后台执行：

```sh
mkdir -p /data/cjm/datasets/logs/scorw-r3-v1-argsfix
nohup env MODE=smoke RESUME=0 sh scorw_r3/run_all.sh > /data/cjm/datasets/logs/scorw-r3-v1-argsfix/smoke.log 2>&1 < /dev/null &
```

正式一条命令自动跑R3-A→R3-B→报告，不用先手动跑各个子阶段：

```sh
nohup env MODE=full RESUME=0 sh scorw_r3/run_all.sh > /data/cjm/datasets/logs/scorw-r3-v1-argsfix/full.log 2>&1 < /dev/null &
```

查看和恢复：

```sh
tail -f /data/cjm/datasets/logs/scorw-r3-v1-argsfix/full.log
nohup env MODE=full RESUME=1 sh scorw_r3/run_all.sh >> /data/cjm/datasets/logs/scorw-r3-v1-argsfix/full.log 2>&1 < /dev/null &
```

恢复必须等上次进程停止；锁拒绝同时运行。任何身份不一致都会停止，不自动删除旧结果或改snapshot。若再次预检查，使用 `MODE=preflight RESUME=1`。

## 输出与小结果同步

结果根 `/data/cjm/datasets/scorw-r3-v1-argsfix/{preflight,smoke,full}`；日志 `/data/cjm/datasets/logs/scorw-r3-v1-argsfix`；临时文件 `/data/cjm/datasets/tmp/scorw_r3`。

FULL根目录包含任务要求的所有 R3A/R3B JSON/Markdown、`R3_PROTOCOL_FREEZE.json`、`R3_PROJECT_DECISION.json`、`VALIDATION_GATES.json`、`EVIDENCE_LEVELS.json`、`FINAL_RESULTS.md`、`NEXT_STAGE_DECISION.md`、`REPORT_MANIFEST.json`、`SMALL_RESULTS.tar.gz`。额外有平衡/负对照、IoU和旧15条取证审计。未执行阶段提供NOT_EXECUTED及原因，科研负结果或样本不足可正常完成。

下载 `full/SMALL_RESULTS.tar.gz` 和 `full/REPORT_MANIFEST.json` 即可审阅小结果；归档只含根层报告，不含PCD、数组、权重或逐场景完整预测缓存。需要完整预测时按 `R3B_SCENE_RESULTS.json` 的路径和SHA256单独取 `stages/detector/scenes/.../SCENE.json`。建议小报告存新的 `logs/newdataset/R3/`，不要覆盖旧包；GitHub不提交大缓存、点云或模型。

合成验收在smoke/full内部自动执行。对应测试命令 `python -B -m unittest scorw_r3.tests.test_r3 scorw_r3.tests.test_argsfix -v` 只读合成数组和本地已核验Hahner资源，不访问OPV2V、不启动检测器。
