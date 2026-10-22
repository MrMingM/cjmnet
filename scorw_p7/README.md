# SCOR-W P7：平衡天气场池统计验证

P7用同一批20个世界天气场计算每辆车的固定Clean光路，然后重排车辆与场的配对。这样每辆车在E2/E3中经历完全相同的单车退化集合，比较重点是多车信号一起变化的方式。

代码、合成测试已交付；P7真实数据实验尚未运行。旧代码、快照、模型与产物全部保留。不要求Git，不创建branch/worktree，不训练、不评测AP、不生成天气PCD。

## 实现和审阅

实际审阅、旧结果来源及P6限制见 `REVIEW_AND_SCOPE.md`。本地P6研究模块存在且可检查其snapshot；P7没有把它作为运行导入依赖。服务器需已有完成的P5/P6产物及原始OPV2V train。

- `audit_sources.py`：v1及本包严格源码哈希；P6源包存在性/完整性诊断；只核对使用到的历史选择、合同、summary元数据。
- `select_scenes.py`：旧P6四个开发scene；排除P5/P6分析scene和全部future保留scene，按CAV数、初始布局及固定哈希选8个新验证scene。开发复用其P6两帧；新验证固定连续索引两帧。
- `balanced_assignment.py`、`field_pool.py`：ego偏移0，其余CAV按稳定字符串ID排序偏移1..N−1；同次E3场索引互异。每scene的20个seed、系数和hash全部在积分前冻结。CAV超过20立即停止。
- `extract_rays.py`：原始RGB红通道/255强度；继承位姿和GT约定；去本车矩形。按目标-CAV、距离、零/正强度分层；先给每个非空层1点，再按Clean多车共同观测优先轮转加点；层内用point-identity哈希排序。原始点重复落入多个框时只积分一次，关联权重分开保存。
- `integrate_pool.py`：只读调用v1世界场和光路积分；每个原始光路/场配对只计算一次，另列数值诊断探针。逐场原子提交，可按field恢复。没有新散射、经验阈值或PCD编码。
- `signal_metrics.py`：逐目标/帧/CAV对沿20场中心化相关性和协方差；C及其分布、跨车R和光学厚度差异；缺失保留null。先在同一object内合并帧/车辆对，再等object形成scene值。
- `statistics.py`：scene等权，5000次整scene探索性bootstrap；场景方向、留一scene/目标、留一场池敏感性。20次重组合未当成20个道路样本。
- `decision.py`：严格按冻结规则进行项目判断，永不调用检测器。
- `core.py`、`run.py`、`run_all.sh`、`report.py`：锁、原子写、严格合同恢复、顺序阶段、报告和小型归档。
- `config/protocol.json`、`tests/`、`source_snapshot.json`：预声明协议、合成单元测试、确切字节身份。

## 冻结预算与近似

正式每个非空层最多8点，每云最多1024个目标关联样本；共用原始光路去重后实际积分数只会更少。每层必须至少保留1点，预算不足立即停止，不能删目标绕过。SMOKE固定为第一个开发scene、两帧、每层最多1点，保留完整20场；没有验证计算。

权重为层全点数/层样本数。R为固定样本的 `sum(weight*I*T)/sum(weight*I)`。这个加权比值处于[0,1]，分母对全部天气固定；它是采样近似，不能叫全量精确或无偏估计。全量Clean点数、强度、样本数、比例和样本分母独立保存。正强度单独成层，防止抽样把本来有信号的目标变成零分母。

20场的算法、mean_alpha=.06、length25m、contrast=.8、32模式、最大步长.5m全部保持v1。世界场在scene的两帧间固定。没有真实束ID、发射时间、接收阈值或波形，不补造。数值探针检查复现、步长减半；P0复用v1全部坐标/GT身份门禁。

E2全车用Fj；E3的CAV c用F[(j+offset[c])%20]。完整循环保证逐CAV及逐固定光路/目标的tau、T、R的分布严格相同。每次重复的总体退化仍可不同；主结果保留全部20次，不筛选matched重复。

共享重排负对照让所有CAV共同循环移位1；完整联合集合与E2相同。单邻车平均ΔC必须在1e-12内为0，同时保留C标准差、分位数及P(C>0)。一般等价检查绝对/相对容差均1e-12，不自动放宽。

留一场敏感性遍历全部20个预声明删除位置，仅作为辅助诊断：从缓存构造19场平衡子池并按同一ID排名偏移重组，E3的配对也会改变。没有新积分、新seed或择优删除；主结果始终完整20场。20个CAV无法放入19个不同场时，此敏感性不可用，不能据此继续检测。

## 服务器准备与检查

完整同步 **`scorw_p7/`，包含source_snapshot.json**。现有 `shared_weather_benchmark/` 必须与冻结快照一致；服务器历史目录必须存在：

- `/data/cjm/datasets/shared-weather-benchmark-v1-smoke`
- `/data/cjm/datasets/shared-weather-benchmark-p6-v1`

P7独立源码策略只把 `AI_CONTEXT.md` 作为不执行的审阅文档，保留原v1完整快照对此文档的差异诊断；旧快照不改写。其余57个旧严格文件、旧snapshot本体和19个P7文件全部逐字节固定。原因及文档的三个哈希口径见REVIEW_AND_SCOPE与P7_SOURCE_AUDIT。服务器不要为通过P7而回滚其他工作更新的上下文文档。

P7只读取其中登记过的元数据，重新建立Clean光路，不复制旧PCD或sidecar。如果需补齐独立P6包，请整目录同步其 `__init__.py`、`core.py`、`run.py`、`selection.py`、`repeat.py`、`legacy.py`、`metrics.py`、`comparison.py`、`export_audit.py`、`run_server.sh`、`config/protocol.json`、`tests/`、README/指标/交付文档及`source_snapshot.json`；不要只同步总控入口，也不要重写旧snapshot。

```sh
cd /home/cjm/OpenCOOD-main/cjmnet
MODE=checks RESUME=0 sh scorw_p7/run_all.sh
```

checks只做源码/AST/JSON/LF及合成测试，不读取真实数据。服务器输出为 `/data/cjm/datasets/logs/scorw-p7-v1/checks/`。

本地允许的同一纯合成检查入口：

```text
python -B -m scorw_p7.run --mode checks --check-output scorw_p7/local_validation
```

## SMOKE后台运行

```sh
cd /home/cjm/OpenCOOD-main/cjmnet
mkdir -p /data/cjm/datasets/logs/scorw-p7-v1
nohup env MODE=smoke RESUME=0 sh scorw_p7/run_all.sh \
  > /data/cjm/datasets/logs/scorw-p7-v1/smoke.log 2>&1 &
echo $! > /data/cjm/datasets/logs/scorw-p7-v1/smoke.pid
```

SMOKE数据/缓存/报告独立位于 `/data/cjm/datasets/scorw-p7-v1-smoke`，状态目录 `/data/cjm/datasets/logs/scorw-p7-v1-smoke`。它不自动触发正式验证，结果不能作为正式统计。

## 完整P7后台运行

```sh
cd /home/cjm/OpenCOOD-main/cjmnet
mkdir -p /data/cjm/datasets/logs/scorw-p7-v1
nohup env MODE=full RESUME=0 sh scorw_p7/run_all.sh \
  > /data/cjm/datasets/logs/scorw-p7-v1/full.log 2>&1 &
echo $! > /data/cjm/datasets/logs/scorw-p7-v1/full.pid
```

P7-0源码/历史/场景/坐标审计 → P7-1合成门禁 → P7-2开发计算 → P7-3已冻结新验证 → P7-4分析决策归档，自动串行。
新验证不足8个时选0个，不调用旧验证或future作替代。开发结果照常完成，P7-3.status=not_executable，整体complete_development_only，决策INCONCLUSIVE_VALIDATION_UNAVAILABLE。

```sh
tail -n 100 -f /data/cjm/datasets/logs/scorw-p7-v1/full.log
cat /data/cjm/datasets/scorw-p7-v1/VALIDATION_GATES.json
```

## 中断恢复

```sh
cd /home/cjm/OpenCOOD-main/cjmnet
nohup env MODE=full RESUME=1 sh scorw_p7/run_all.sh \
  >> /data/cjm/datasets/logs/scorw-p7-v1/full.log 2>&1 &
echo $! > /data/cjm/datasets/logs/scorw-p7-v1/full.pid
```

SMOKE恢复把MODE改为smoke并追加smoke.log。保持源、配置、解释器、线程和路径一致。所有已提交阶段、嵌套commit、数组、原始输入及已发布报告都核验SHA256；未提交阶段确定性重算，逐field缓存复用。运行时不依赖.git，不手改合同/哈希，不因失败重抽天气。并发第二进程被锁拒绝。

源码/配置/输入身份变化、GT/CAV/point身份不一致、无效非有限数、缺场、边缘失衡、单邻车或重排负对照失败、坐标/数值门禁失败时立即停止后续阶段。null有明确缺失原因，不能变成0。发生SIGTERM时记录失败；SIGKILL/断电可能留下running，恢复仍需核验。

## 科学判断规则

这些是运行前冻结的项目启发式，未做传感器标定、统计功效或多终点正式确认性推断：

- 8个新验证scene全部有效；每scene有效目标实例≥10、有效唯一object≥5、有效目标CAV配对≥10；无效相关性配对比例≤.5，单object配对占比≤.25。
- 相关系数差实际意义门槛|ΔCorr|≥.10；C均值和标准差差为.01，P(C>0)差为.05。方向不限，所有指标和正负scene完整报告。
- CONTINUE_TO_DETECTOR要求主要相关性差和至少一个C分布指标超过门槛且scene CI不跨0；场景方向≥.75；去任一scene仍同方向且超过门槛；留一场池整体方向≥.9、最大相对变化≤.5；至少.75的scene自身留一场方向≥.9；任何目标不能贡献超过.25的绝对效应，留一目标不得反转对应scene方向。
- STOP_OR_REDESIGN要求样本质量门禁充分，相关性和三个C指标的完整CI都落在各自小效应范围内，且预声明留一场敏感性最大整体改变量小于该门槛的一半。无显著性不能触发停止结论。
- 其余INCONCLUSIVE；新验证不足为INCONCLUSIVE_VALIDATION_UNAVAILABLE。

CONTINUE只给予未来固定权重AttFuse小样本诊断资格，不证明数据集创新或模型可恢复性。本轮永不运行检测。P7仍未得到稳定且有实际意义的证据时，建议停止把本版本扩展为完整天气数据集；保留不确定和负结果，不自动增加seed/场景或修改参数。

## 结果与小型同步

正式输出根 `/data/cjm/datasets/scorw-p7-v1/`；缓存位于其stages目录，不写仓库。用户要求的14个报告全部在根目录，另含P7_DECISION.json、P7_SCENE_TABLE.csv、P7_REPORT_MANIFEST.json。RESULTS.md包括每scene数值、每帧CAV名单、有效目标/配对、方向、缺失及限制。

`P7_SMALL_RESULTS.tar.gz` 只含报告、协议、源码/输入哈希和最终门禁，排除光路/大数组/PCD；manifest逐文件列SHA256。`logs/scorw-p7-v1/LATEST.json`记录最终归档路径和SHA256。

建议从本地PowerShell只下载该小包：

```powershell
New-Item -ItemType Directory -Force 'D:\Collaborative perception\mymodule\cjmnet\logs\SCOR-W\P7' | Out-Null
scp cjm@admin1:/data/cjm/datasets/scorw-p7-v1/P7_SMALL_RESULTS.tar.gz 'D:\Collaborative perception\mymodule\cjmnet\logs\SCOR-W\P7\'
```

确认包的SHA256与LATEST一致后，在本地解包、审阅，只提交小型JSON/Markdown/CSV和源码至GitHub。不要提交stages缓存或原始点云，不自动执行Git提交、push或服务器长实验。
