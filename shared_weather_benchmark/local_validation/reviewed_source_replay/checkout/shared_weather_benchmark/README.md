# SCOR-W v1：共享雾场与多车观测原型

这版先检查“同一个目标，换一辆车是否还保留更强证据”。世界里先固定一片雾，各车沿自己的激光路径计算衰减，再比较同一个Clean目标的证据。没有启动检测器、训练或正式全量生成。

默认只改变E1–E3的回波强度，保留原始点记录；接收阈值没有标定依据，因此不把弱点自动删除。连续强度和透射率变化是主诊断。E0继承原版均匀Fog的衰减、soft替换与噪声。非均匀soft回波生成暂时禁用。

## 阅读顺序与交付状态

1. `PRIOR_ART_AND_SCOPE.md`：实际核查的6篇论文、完整/部分阅读范围、历史协议与源码清单。
2. `PHYSICAL_MODEL.md`：每一项物理依据及E0/E1的正确比较范围。
3. `PHYSICS_ASSUMPTIONS.md`：数学保证、数值检查、近似、未校准参数和不可恢复信息。
4. `COORDINATE_AUDIT.md`、`RESULTS.md`：根目录文件为交付状态；服务器输出目录才保存真实数据实测。

纯合成软件检查见 `local_validation/reviewed_source_fix/P{0,1,2}_NUMERICAL.json` 和 `local_validation/reviewed_source_fix/VALIDATION_GATES.json`。真实OPV2V坐标、PCD、E0查表与来源检查尚未运行，不能写成已经通过。

## 目录与新增文件

```text
shared_weather_benchmark/
  __init__.py
  README.md
  config/prototype.json
  geometry.py                    # OpenCOOD位姿约定的独立数值实现
  world_field.py                 # 常数/平滑/分层世界alpha场
  ray_integrator.py              # 标量与NumPy批量光路积分
  fog_physics.py                 # hard衰减与未校准soft核接口
  e0_adapter.py                  # 冻结原核+第五特征来源身份
  dataset_io.py                  # 只允许原始train，场景分区与PCD读取
  generator.py                   # E0–E3逐云原子提交与sidecar
  provenance.py                  # 源码/输入/产物哈希，无Git依赖
  diagnose_sources.py            # 门禁失败时只读列出具体不一致的源码
  observability.py               # 固定Clean GT、来源关联和连续观测指标
  audit.py                       # 必须在生成前通过的真实坐标/GT审计
  compare.py                     # 逐scene统计、场一致性、E2/E3匹配残差
  run.py                         # P0–P5总入口与恢复门禁
  run_server.sh                  # POSIX sh，LF，CPU
  check_static.py                # AST/LF/JSON/源码身份检查
  tests/__init__.py
  tests/test_numerics.py          # 标准库合成数值检查
  tests/test_server_arrays.py     # P3前的CPU合成PCD/批量/E0检查
  source_snapshot.json
  PRIOR_ART_AND_SCOPE.md
  COORDINATE_AUDIT.md
  PHYSICAL_MODEL.md
  PHYSICS_ASSUMPTIONS.md
  DATA_MANIFEST.json
  SCENE_WEATHER_FIELD_MANIFEST.json
  OBSERVABILITY_SUMMARY.json
  E0_E1_E2_E3_COMPARISON.md
  VALIDATION_GATES.json
  RESULTS.md
  local_validation/              # 本地纯数值/静态验证产物，无真实数据
```

只新增该目录。既有reference_weather、GSPR、AttFuse、V2X-R、GT、权重和实验输出保持原样。运行不要求Git、branch或commit。源码身份检查采用文件内容哈希；E0资源兼容原资产锁的blob内容哈希计算，也不需要Git。

## 固定的第一轮协议

| 项目 | 默认值 / 规则 |
|---|---|
| 原始数据 | `/data/scd/datasets/opv2v_official_data_dumping/train` |
| 分区 | generator development / physics validation / future model development，严格按scene拆分 |
| 选择 | CAV数（封顶4）和初始最大车距是否≥30m分层；scene哈希排序后循环分区；分区内交替各层选scene |
| 规模 | 每分区2个scene，future分区不读PCD/不生成；其余4个scene各2帧SMOKE或8帧小样本 |
| 帧选择 | seed固定的连续帧块，时间索引规则和完整序列步长记录；不推断实际秒数 |
| ego | 与BaseDataset一致的字符串排序，首个负ID移到末尾；首车为ego |
| CAV | 为所有已选CAV生成PCD，不按detector的max_cav截断；合作诊断固定70m XY通信半径内CAV |
| GT | 同帧Clean YAML对象按ID合并，跨车同ID角点≤1cm；排除ego自身ID，任一活动CAV距目标≤120m |
| 点关联 | 世界3D框内的原始Clean点，固定车体矩形遮罩；没有逐点表面真值 |
| 场 | seed=20261020；alpha均值=.06m^-1，latent length=25m，contrast=.8，32模式 |
| 积分 | 中点法，最大步长.5m；光学厚度步长减半误差门槛.002 |
| 默认失效策略 | attenuation_only，无物理阈值假设；RGB写盘量化另报 |
| E0 | 原reference Fog配置固定；已存在资产目录只读复用，不下载或改写 |
| 匹配 | 无自动参数拟合；先冻同边缘场分布，逐scene报告实际alpha、点数、距离桶和透射率残差 |
| 检测 | 本轮不执行AttFuse/GSPR/AP；不通过预测分数定义类别 |

样本大小、seed、阈值、场和选择规则包含在源码/配置哈希与GENERATION_FREEZE.json中。后续若做物理参数匹配，只能用generator development scenes另开配置/输出；不能根据physics validation或test结果反向选择。

## 阶段与停止规则

| 阶段 | 工作 | 是否真实数据 |
|---|---|---|
| P0 | 源码身份、合成坐标、旋转与GT自检 | 否 |
| P1 | 零/常数/分段层、双程积分、收敛、hard退化及soft核检查 | 否 |
| P2 | 确定性、共享位置、边界、诊断分类与scene统计检查 | 否 |
| P3 | 先做真实坐标/GT/身份/PCD/资产审计并冻结，再小规模E0–E3生成 | 是，服务器 |
| P4 | 全部PCD/sidecar/来源映射验证、目标观测和scene统计/匹配审计 | 是，服务器 |
| P5 | 输出结果、问题清单与扩大/停止建议 | 读取已审计报告 |

P0–P2只需Python标准库，允许本地运行。P3中的真实审计是P0静态检查的必要补充，必须在第一个天气PCD之前通过。P3还执行NumPy/标量一致性、原版E0复现与PCD读回的合成检查。

以下任一失败立即以非零状态停止，记录`FAILURE.txt`和`VALIDATION_GATES.json`，后续阶段不运行：

- 严格固定的源码、配置、原始输入、GT、资产或已提交产物哈希不一致；恢复时实际源码身份发生变化。
- 导入本轮禁用的`opencood.data_utils`（包括BaseDataset），或它已被其他代码导入。
- 正反变换、角度约定、时间戳配对/连续性、CAV轨迹或跨车同ID GT不一致。
- 点云不支持的强度编码、NaN/Inf、空Clean云、零距离原始点、场外查询。
- 积分非有限/负值、合成解析检查失败、步长误差超预注册容差。
- E0原核与带索引调用输出不同；来源/原始点索引/输出映射不守恒；本版出现unknown输出来源。
- PCD读回改变坐标/行映射或强度误差超8bit编码容差。

没有跳过坏帧、Clean补点回退、手改哈希或放宽容差。匹配统计超容差会标记unmatched并完整输出问题；它是需要报告的实验结果，不应伪装成代码失败后删除样本。unmatched结果禁止作已匹配的空间关联因果结论。

## 服务器准备

整目录同步到`/home/cjm/OpenCOOD-main/cjmnet/shared_weather_benchmark`，保留字节/LF和source_snapshot。snapshot的`sources`仍要求精确匹配；只有已确认不执行的BaseDataset单列为`reviewed_only_sources`，记录实际哈希并禁止导入。发现其他差异时先核查，禁止手改snapshot或覆盖旧研究代码。已存在的Fog表路径：

`/data/cjm/datasets/reference-weather-assets-v1/fog/integral_lookup_tables/original/`

P3/P4使用opencood现有环境的NumPy、SciPy、Open3D、PyYAML及tqdm。所有阶段CPU运行，不导入Torch。脚本直接使用该Conda环境的绝对解释器，不要求先激活。

```sh
cd /home/cjm/OpenCOOD-main/cjmnet
mkdir -p /data/cjm/datasets/logs/shared-weather-benchmark-v1
mkdir -p /data/cjm/datasets/tmp/shared_weather_benchmark
sh -n shared_weather_benchmark/run_server.sh
/home/cjm/miniconda3/envs/opencood/bin/python -B shared_weather_benchmark/check_static.py
```

## 服务器P0/P1/P2

这三个阶段很短且完全不读取OPV2V。首次运行：

```sh
MODE=checks SMOKE=1 RESUME=0 sh shared_weather_benchmark/run_server.sh
```

已检查过同一输出目录时，使用`RESUME=1`核验已有门禁。SMOKE、源码和配置必须与后续P3一致。

### P0之前出现Source hash mismatch

这个检查固定本包和既有依赖文件（包括AI_CONTEXT等已审阅资料）。仅凭旧版通用报错无法判断是哪个文件不同；不要先重装环境、重跑天气或覆盖既有源码。

更新后的报错会列出具体路径、期望/实际SHA256，以及缺失或未登记文件。也可以独立运行只读诊断，不启动任何实验：

```sh
/home/cjm/miniconda3/envs/opencood/bin/python -B \
  -m shared_weather_benchmark.diagnose_sources
```

`--json`输出完整机器可读报告。严格固定的文件不一致时返回1，不修改文件、快照或输出目录。CRLF/BOM提示只帮助定位传输格式变化，字节哈希门禁仍然失败；不能通过自动归一化放行。

服务器已报告唯一差异为`opencood/data_utils/datasets/basedataset.py`：审阅哈希`2b60c2e5…`，服务器哈希`62ef160a…`。代码检查确认本轮通过`dataset_io.py`独立读取原始YAML/PCD，使用冻结的`transformation_utils.py`核查坐标，未调用原dataset或检测器。因此本次修复保留旧审阅哈希，将这一个文件归为仅供历史审阅；没有把服务器未知版本称为已审阅或等价版本。

运行入口在任何阶段之前禁止导入整个`opencood.data_utils`，并拒绝已加载的该模块；静态检查也拒绝本包直接导入它。间接或动态导入同样立即报错。其余源码逐字节校验保持。`SOURCE_AUDIT.json`保存审阅/实际哈希与用途，运行合同包含实际哈希及快照哈希；该文件以后改变会使RESUME停止。文件缺失或不可读仍停止。无需同步或覆盖服务器的BaseDataset。

先根据列出的路径确认同步差异，再决定具体处理。不要在服务器重新生成source_snapshot，也不要在原因未明时批量覆盖被冻结的reference_weather/OpenCOOD文件。

若日志停在`execute -> verify_sources`且尚未创建运行合同，本次P0没有开始；解决源码差异后仍使用首次`RESUME=0`。已经有合法合同的旧结果必须保留，源码更新后使用新的OUTPUT_ROOT。

## P3/P4小样本：后台启动

先通过checks，再执行以下命令。默认4个活动scene×2帧，完成P3/P4/P5后退出。可能耗时，使用nohup：

```sh
nohup env MODE=sample SMOKE=1 RESUME=1 FROM_STAGE=P3 \
  sh shared_weather_benchmark/run_server.sh \
  > /data/cjm/datasets/logs/shared-weather-benchmark-v1/smoke.log 2>&1 &
echo $! > /data/cjm/datasets/logs/shared-weather-benchmark-v1/smoke.pid
```

实时日志：

```sh
tail -n 100 -f /data/cjm/datasets/logs/shared-weather-benchmark-v1/smoke.log
```

中断恢复同一目录、同一配置（追加原日志）：

```sh
nohup env MODE=sample SMOKE=1 RESUME=1 FROM_STAGE=P3 \
  sh shared_weather_benchmark/run_server.sh \
  >> /data/cjm/datasets/logs/shared-weather-benchmark-v1/smoke.log 2>&1 &
echo $! > /data/cjm/datasets/logs/shared-weather-benchmark-v1/smoke.pid
```

逐阶段运行也使用同一入口，例如只完成P3后检查坐标报告，再手动跑P4/P5：

```sh
nohup env MODE=phase THROUGH=P3 FROM_STAGE=P3 SMOKE=1 RESUME=1 \
  sh shared_weather_benchmark/run_server.sh \
  > /data/cjm/datasets/logs/shared-weather-benchmark-v1/p3.log 2>&1 &
```

```sh
nohup env MODE=phase THROUGH=P4 FROM_STAGE=P4 SMOKE=1 RESUME=1 \
  sh shared_weather_benchmark/run_server.sh \
  > /data/cjm/datasets/logs/shared-weather-benchmark-v1/p4.log 2>&1 &
```

```sh
MODE=phase THROUGH=P5 FROM_STAGE=P5 SMOKE=1 RESUME=1 \
  sh shared_weather_benchmark/run_server.sh
```

8帧/scene仍是小样本，不是全量生成。只有查看SMOKE结果后再决定是否手动运行；该命令启用独立标准根目录：

```sh
MODE=checks SMOKE=0 RESUME=0 sh shared_weather_benchmark/run_server.sh
nohup env MODE=sample SMOKE=0 RESUME=1 FROM_STAGE=P3 \
  sh shared_weather_benchmark/run_server.sh \
  > /data/cjm/datasets/logs/shared-weather-benchmark-v1/sample.log 2>&1 &
echo $! > /data/cjm/datasets/logs/shared-weather-benchmark-v1/sample.pid
```

恢复8帧版本时保留SMOKE=0并使用相同OUTPUT_ROOT；不能把SMOKE目录改作标准目录。

## 原子提交与恢复

合同包含精确配置、SMOKE和源码身份（包括仅供审阅文件的实际哈希）。`SOURCE_AUDIT.json`也受P0产物哈希保护。每云PCD、sidecar、GT链接全部验证后才原子写record；缺少record的中断云确定性重算，已有PCD不同则拒绝覆盖。record包含三个文件的SHA256，逐阶段结果亦保存哈希。

恢复先校验已提交阶段、原始输入和运行环境/资源，再复用有效record。真实坐标复查写入临时目录，和冻结身份一致后才提交，防止发现变化时先覆盖旧GT。独占锁拒绝并发写同一根目录。断点单位是完整CAV云/完整阶段，尚未提交的单云从头计算。

源码更新必须另开输出根目录。严禁把运行报错当作允许手改source_snapshot的理由。

## 输出路径与读取方式

标准小样本根：`/data/cjm/datasets/shared-weather-benchmark-v1`。

SMOKE根：`/data/cjm/datasets/shared-weather-benchmark-v1-smoke`。

日志根：`/data/cjm/datasets/logs/shared-weather-benchmark-v1`。

临时目录：`/data/cjm/datasets/tmp/shared_weather_benchmark`。

最终报告直接位于数据根：本任务要求的10份报告，以及`SOURCE_AUDIT.json`、`COMPARISON.json`、`INPUT_MANIFEST.json`、`CLEAN_GT.json`、`SELECTION.json`、`GENERATION_FREEZE.json`、`ISSUES.json`。真实结果是否完整以`VALIDATION_GATES.json`的P0–P5全部passed和status=complete为准。

PCD为`E{0,1,2,3}/train/<scene>/<cav>/<stamp>.pcd`，YAML只链接原件；原始数据需保留。额外来源为`sidecars/E*/train/...npz`。`original_index`对应原PCD解码顺序，point ID可由frame/CAV/原始PCD哈希/索引确定性计算；weather与serialized数组按原索引排列，`output_index=-1`表示无写出回波。soft替换时原hard点retained=false、emitted=true、origin=1。

`OBSERVABILITY_ROWS.jsonl`给出每目标/CAV连续指标、量化前/后经验阈值计数、距离和视角。`OBSERVABILITY_TARGETS.json`给出逐目标A/B/C/D/residual、literal zero标志、ego-best-neighbor差及阈值敏感性。阈值仅作诊断，不能称目标一定可检测。

## 何时值得扩大

先检查所有硬门禁，再看每个scene的E2/E3物理统计残差。只有在总体退化接近时，多个scene仍有同方向的连续互补差异，且结论不依赖单一count/强度阈值，才有理由增加独立scene和天气seed。

第一版每分区仅2个scene，没有统计显著性判断资格。增加scene后，用scene配对区间与效应大小评估稳定性，不能把连续帧当独立样本。物理验证分区与开发分区结果都要保留，不能筛选有利scene。

若变化主要由总体衰减不匹配、RGB量化、少数scene或阈值选择解释，应停止扩大或缩小主张。若只见连续强度差异，点数/零回波类别不变，也如实报告。要形成完整论文基准，还需可核验的传感器参数/实雾对照与冻结后的独立检测失败模式实验；本轮工程通过不能代替这两项。
