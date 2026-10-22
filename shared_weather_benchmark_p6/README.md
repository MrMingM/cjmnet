# SCOR-W P6

本包复核首轮结果，补齐信号观测指标，再用新scene与固定三seed判断E2/E3差异是否值得继续研究。所有代码新增于本目录，只读导入v1稳定函数；不修改旧包、源码快照、天气数据、模型或历史结果。运行不要求Git，不创建branch/worktree。

当前只交付代码、合成测试和命令。服务器真实产物尚未在本地完整取得；已知历史结果来自用户粘贴文本，合同为`ad3c9bd3a5c566d0d1c4e65decf3193d311f972f99e35af7dd98526fa4a40083`。不能把本地合成测试写成P6真实实验已完成。

**审阅与复用范围**

完整阅读AI_CONTEXT.md，重点核对§44公开天气、§45–48官方/固定GT口径、§49当前GSPR负结果及§43停止边界。核对v1 README/config、world_field、geometry、ray_integrator、fog_physics、generator、observability、compare、dataset_io、audit、provenance、run与tests；审查reference_weather的README/config/common/adapters/generate与Fog原核。Rain/Snow厂商算法不进入P6；没有声称本轮完整审阅所有雨雪源文件。也核对此前用户回传的SOURCE_AUDIT、门禁、RESULTS与比较表，未假装读取服务器其余完整JSON。

复用v1坐标与GT审计、世界场/积分、来源核验、PCD编码和E1–E3生成。P6-A在独立临时目录重放旧统计公式，另用P6代码重新计算R、缺失与邻车信息。这是保存数据的独立读取/重算，部分公式代码有共同来源，不能称独立实雾物理验证。

**科学边界与指标**

E2共享场，E3每车独立连续场，二者均只衰减强度，主对照的回波机制一致。E1用于常数场/几何参照。E0含soft替换，仅在旧数据审计和辅助指标中保留，不在P6-C重复生成。

v1原始点保留率在E1–E3非空分母下为1；点数A/B主要继承Clean。旧G实际为best邻车−ego，表头方向有误。旧比例本身已有连续强度含义，P6补全绝对信号、有效邻车数量/分层、null原因、量化前后及阈值差异，未改旧物理。

主指标R=同批Clean关联原始点的天气强度和/Clean强度和；C=最佳有效邻车R−ego R。完整定义、权重、空值、关联代理和反例见METRIC_DEFINITIONS.md。没有标定接收阈值或非均匀散射，不能叫完整真实LiDAR雾物理或detector recoverability。

**阶段与冻结**

- P6-A：只读核对旧合同、六阶段登记产物、每个云record/PCD/sidecar/YAML、原始PCD/YAML及选择布局元数据；复核四组笛卡尔积与分母；重新执行坐标审计、选景规则、光路/强度重算及旧比较公式。源文件清单含路径/大小/哈希，不复制大文件。报告全部匹配条件和CAV布局/路径分布/均值偏移的描述关联，不作四scene因果判断。
- P6-B：从A已核验的缓存独立输出新指标，不重新生成旧PCD。全部旧mode可审查；主解释以E1–E3机制为准。
- P6-C：保留旧分区归属，排除四个旧活动scene，以及旧SELECTION中整个all_scene_partition.future_model_development和selected.future保留scene；在原开发分区剩余候选选4个、原physics分区选8个。按CAV数/初始车距分层，哈希排序交替取景；选不够立即报告，不从future补齐。不按天气结果或AP选景。

每scene固定2个连续索引时间戳。全部12scene、24时间戳的完整计划及三seed在任何新天气生成前冻结到P6C_SELECTION_FREEZE.json与P6C_FROZEN_CONFIG.json。根seed20261020，经`seed_for(root,"scorw-p6-repeat-v1",repeat_index)`派生0/1/2三组，不再抽样；各scene和CAV继续按v1派生规则构场。alpha=.06、length25m、contrast.8、32模式、步长.5m与原匹配容差不变。无匹配方法拟合、重抽或自动调参。

SMOKE也先声明完整12scene计划，但只对各分区首个scene、repeat0生成，即2scene×2帧×1seed×E1/E2/E3。正式为12scene×2帧×3seed×E1/E2/E3。SMOKE/正式使用不同目录和合同；正式不根据SMOKE结果重选或调参。SMOKE已暴露的计划内scene须披露，不能称在正式阶段从未查看。原始OPV2V train可能用于历史检测训练，“新scene”仅指本SCOR-W天气研究未使用，不能宣称全项目从未见过。

**源码与运行准备**

只同步本目录的完整交付，包含source_snapshot.json；旧shared_weather_benchmark和reference_weather保留原样。新snapshot固定本包和复用代码，另核验旧snapshot，绝不重写旧哈希。服务器BaseDataset继续仅记录实际哈希、禁止导入。路径、源/输出、配置、SMOKE、父合同、原始输入、运行依赖和字段清单参与身份校验。

```sh
cd /home/cjm/OpenCOOD-main/cjmnet
mkdir -p /data/cjm/datasets/logs/scorw-p6-v1
mkdir -p /data/cjm/datasets/tmp/scorw_p6_v1
sh -n shared_weather_benchmark_p6/run_server.sh
MODE=checks sh shared_weather_benchmark_p6/run_server.sh
```

checks仅AST/JSON/LF/hash与Python/NumPy合成测试，不读取旧服务器数据、Torch或模型。也可本地执行：

```text
python -B -m shared_weather_benchmark_p6.run --mode checks --check-output shared_weather_benchmark_p6/local_validation
```

真实阶段由入口强制Linux，只使用CPU；解释器默认`/home/cjm/miniconda3/envs/opencood/bin/python`。不需新增训练依赖，PCD依赖既有Open3D/NumPy/PyYAML。

**P6-A：后台只读核验**

可能多次扫描旧PCD/sidecar，因此后台运行：

```sh
nohup env MODE=audit SMOKE=0 RESUME=0 \
  sh shared_weather_benchmark_p6/run_server.sh \
  > /data/cjm/datasets/logs/scorw-p6-v1/p6a.log 2>&1 &
echo $! > /data/cjm/datasets/logs/scorw-p6-v1/p6a.pid
```

产物：`/data/cjm/datasets/logs/scorw-p6-v1/audit/`下P6A_AUDIT.json/.md、P6A_SCENE_TABLE.csv、P6A_MATCH_FAILURES.json、P6A_ARTIFACT_MANIFEST.json，以及坐标复查与外部缓存。先确认A成功再调用B。

**P6-B：只分析旧数据**

```sh
nohup env MODE=metrics SMOKE=0 RESUME=1 \
  sh shared_weather_benchmark_p6/run_server.sh \
  > /data/cjm/datasets/logs/scorw-p6-v1/p6b.log 2>&1 &
echo $! > /data/cjm/datasets/logs/scorw-p6-v1/p6b.pid
```

产物：`/data/cjm/datasets/logs/scorw-p6-v1/metrics/`下P6B_METRIC_DEFINITIONS.md、P6B_TARGET_SIGNAL_ROWS.jsonl、P6B_TARGET_COMPLEMENTARITY.jsonl、P6B_SCENE_SUMMARY.json、P6B_QUANTIZATION_AUDIT.json、P6B_THRESHOLD_SENSITIVITY.json、P6B_RESULTS.md。A/B共享可验证的阶段缓存；不受C的SMOKE规模影响。

**P6-C SMOKE**

需先有已提交的A和B，否则repeat模式立即停止：

```sh
nohup env MODE=repeat SMOKE=1 RESUME=0 \
  sh shared_weather_benchmark_p6/run_server.sh \
  > /data/cjm/datasets/logs/scorw-p6-v1/p6c_smoke.log 2>&1 &
echo $! > /data/cjm/datasets/logs/scorw-p6-v1/p6c_smoke.pid
```

产物：`/data/cjm/datasets/shared-weather-benchmark-p6-v1-smoke/`，标记complete_smoke_only，不代表12scene验证。

**正式12scene×3seed**

A/B通过后，SMOKE也完成且没有软件正确性问题时，用户手动启动：

```sh
nohup env MODE=repeat SMOKE=0 RESUME=1 \
  sh shared_weather_benchmark_p6/run_server.sh \
  > /data/cjm/datasets/logs/scorw-p6-v1/p6c_full.log 2>&1 &
echo $! > /data/cjm/datasets/logs/scorw-p6-v1/p6c_full.pid
```

若从未运行任何阶段，也可用MODE=all串行执行A→B→C；此命令会直接做正式规模，应由用户明确手动启动：

```sh
nohup env MODE=all SMOKE=0 RESUME=0 \
  sh shared_weather_benchmark_p6/run_server.sh \
  > /data/cjm/datasets/logs/scorw-p6-v1/p6_all.log 2>&1 &
echo $! > /data/cjm/datasets/logs/scorw-p6-v1/p6_all.pid
```

**日志、状态与恢复**

```sh
tail -n 100 -f /data/cjm/datasets/logs/scorw-p6-v1/p6a.log
tail -n 100 -f /data/cjm/datasets/logs/scorw-p6-v1/p6b.log
tail -n 100 -f /data/cjm/datasets/logs/scorw-p6-v1/p6c_smoke.log
tail -n 100 -f /data/cjm/datasets/logs/scorw-p6-v1/p6c_full.log
```

上面按正在运行的任务选择一行。状态：

```sh
MODE=status SMOKE=0 sh shared_weather_benchmark_p6/run_server.sh
MODE=status SMOKE=1 sh shared_weather_benchmark_p6/run_server.sh
```

正式C中断后保持源码/配置/SMOKE与输出路径，追加日志恢复：

```sh
nohup env MODE=repeat SMOKE=0 RESUME=1 \
  sh shared_weather_benchmark_p6/run_server.sh \
  >> /data/cjm/datasets/logs/scorw-p6-v1/p6c_full.log 2>&1 &
echo $! > /data/cjm/datasets/logs/scorw-p6-v1/p6c_full.pid
```

恢复A用MODE=audit SMOKE=0 RESUME=1，恢复B用MODE=metrics SMOKE=0 RESUME=1；恢复C SMOKE保留MODE=repeat SMOKE=1 RESUME=1并追加对应日志。各阶段/整个输出有锁；阶段登记产物、旧输入、云record/PCD/sidecar和GT均复核。每云生成、每云指标缓存、阶段COMMIT原子提交；未提交的计算确定性重算，已提交不同内容拒绝覆盖。

修改配置或源码需另开完整新运行路径，不能手改旧CONTRACT、COMMIT或snapshot。外部复制配置后设置CONFIG=/data/.../protocol.json，并更改其中data_root和logs_root；新路径限定为`/data/cjm/datasets/shared-weather-benchmark-p6-*`和`/data/cjm/datasets/logs/scorw-p6-*`的直接子目录命名空间，防止误写旧数据。物理参数与匹配容差仍受本轮约束，不能借此事后调门槛或重抽seed。

**最终报告位置与判读**

正式数据根`/data/cjm/datasets/shared-weather-benchmark-p6-v1/`包含P6C_SELECTION_FREEZE.json、P6C_FROZEN_CONFIG.json、P6C_MATCHING.json、P6C_SIGNAL_COMPLEMENTARITY.json、P6C_SCENE_SEED_RESULTS.csv、P6C_ALL_SCENES_RESULTS.md、P6C_UNMATCHED_ANALYSIS.md、SUMMARY.json、RESULTS.md、NEXT_STAGE_DECISION.md和最终VALIDATION_GATES.json。每repeat的PCD/sidecar/配置/字段与P6B指标位于repeats/repeat_{0,1,2}/。

全程状态记录于`logs/scorw-p6-v1/state-full/VALIDATION_GATES.json`或`state-smoke/`。C尚未执行时从这里查看A/B进度；C成功后复制一份最终门禁至数据根。只读核验或运行失败记录FAILURE.txt；失败时原始旧数据不被修复。

硬停止：源/配置/合同/产物哈希变化，输入/GT/CAV分母不一致，丢帧漏mode，NaN/Inf，坐标/GT/积分失败，来源或8bit映射错误，历史/future场景泄露，候选不足，缺失固定seed结果。null是合法缺失并单列；不伪装成0。匹配失败只作实验结果完整输出，不停止生成、不删样本。

**CONTINUE / INCONCLUSIVE / STOP_OR_REDESIGN**

规则在config/protocol.json.decision预声明；这是项目启发式，不是传感器标定、功效计算或已建立的统计等效标准。

- CONTINUE：正式8个验证scene，匹配scene/seed比例≥.75；scene均值的|ΔC|≥.01且探索性scene区间不跨0；scene和seed方向一致比例均≥.75；信号相关性变化|Δcorr|≥.01且区间不跨0；近似平衡的|Δ全局T|≤.02层至少覆盖5个验证scene并保持同方向的C信号。任一方向均可，未要求E2更好。只是建议下一阶段AttFuse诊断，不自动启动。
- INCONCLUSIVE：SMOKE、覆盖或有效配对不足、seed波动、匹配率低、区间不够明确，或总体退化混杂仍缺解释。
- STOP_OR_REDESIGN：匹配充分时，ΔC和Δ相关性的探索区间全部落入预声明的小效应范围内。仅针对当前参数/场景/代理；不泛化否定所有共享天气。

报告A–G每项supported_exploratory或unsupported；B的数学/构造核验不替代C/D目标观测证据。主要结果含全部scene/seed；匹配子集只探索性报告。全局T残差分层及scene级OLS只辅助解释，不能称消除了物理混杂。仅2帧不会增加独立scene数。

**回传小型P6-A证据**

服务器先核验五份报告并导出到仓库指定位置，默认总大小≤25MiB；不复制PCD、权重、完整sidecar或缓存，不执行Git命令：

```sh
/home/cjm/miniconda3/envs/opencood/bin/python -B \
  -m shared_weather_benchmark_p6.export_audit \
  --source /data/cjm/datasets/logs/scorw-p6-v1/audit \
  --destination /home/cjm/OpenCOOD-main/cjmnet/logs/SCOR-W/P6A
```

然后在本地PowerShell执行（用户名/主机沿用已知服务器）：

```powershell
New-Item -ItemType Directory -Force 'D:\Collaborative perception\mymodule\cjmnet\logs\SCOR-W' | Out-Null
scp -r cjm@admin1:/home/cjm/OpenCOOD-main/cjmnet/logs/SCOR-W/P6A 'D:\Collaborative perception\mymodule\cjmnet\logs\SCOR-W\'
```

这只复制已核验的小目录，不会自动提交到GitHub。
