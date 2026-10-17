# D−1C：只换计分规则，AP会变化多少？

本目录已实现，**服务器尚未运行，没有D−1C AP结果**。
同一个官方V2X-R LiDAR AttFuse checkpoint，每个条件每帧生成一份最终预测，再用O/O*/F三套协议计分。
本轮不训练、不改模型、不改天气数据、不调score/NMS、不访问test。
所有新增文件只放本目录；历史 `v2xr_official_repro/`、`v2xr_attfuse/`、`reference_weather/` 和结果文件保持原样。

## 已核对的前置证据

读取AI_CONTEXT.md、D−1A wrapper/协议/结果，官方basedataset/intermediate fusion、model、postprocess、inference与eval_utils_opv2v。
读取 `logs/D/D1/RESULTS.md`、完整解析其protocol.json/contract.json/input_manifest.json，记录原文件SHA256到EVIDENCE.json。
旧v2xr_attfuse仅用于理解无FOV的历史迁移协议，不导入其中GT/data/eval或OPV2V epoch13。
AI_CONTEXT中D−1B“尚未运行”的旧状态已被这次回传全量证据覆盖；不据旧文字否定已完成的审计。

固定官方commit `f3ebbad95561444a53a17358735b0f0dc4ba89f7`。
D−1A full AP30/50/70 = 0.8599360524464185 / 0.8227943385636924 / 0.6662545216705442。
官方checkpoint为 `/data/cjm/datasets/V2X-R/Attfuse.pth`，SHA256：
`4e92bff533cba4d8e2020f4176bd9f684b5a2337d59f853535f612edd6aa3e94`。
D−1A运行config SHA256：`c4a4d135594b577b0108b0f49a81937f95ee80c422bfc559a443f8a771bbd76a`。
本阶段重新核对aliases及完整key/shape/dtype，调用官方loader后逐tensor核对加载值。

全829帧D−1B：

| Condition | Official GT | Removed vs Clean | Added | Changed frames | Geometry changed |
|---|---:|---:|---:|---:|---:|
| Clean | 6224 | 0 | 0 | 0 | 0 |
| Fog | 3673 | 2551 | 0 | 637 | 0 |
| Snow | 5854 | 370 | 0 | 179 | 0 |

所有删除目标在天气下全部候选CAV框内为零点。Fog报告有26个删除目标涉及官方空体素临时点，Snow为0。
本阶段保留官方临时点逻辑，记录这项历史证据；不能把“fallback涉及”解释成“这些目标由临时点造成删除”。

服务器预检要求D−1B的完整summary/per_frame/removed文件仍保留，并校验其完成哈希、已钉住的protocol/contract/input_manifest哈希。
全量重新检查829帧的ID子集、共同几何、2551/370个删除目标的零点证据、637/179变化帧及GT总数。
所有输入源文件与链接重新核验SHA256，不按目录名、文件大小或mtime默认为可信。

## 三套协议

| 协议 | 使用的GT | prediction=None | 非None空预测 |
|---|---|---|---|
| O / OFFICIAL_EXACT | 当前条件官方最终evaluation GT | 整帧跳过，GT不计入分母 | 原样计分，GT进入分母 |
| O* / OFFICIAL_GT_NO_SKIP | 与O完全相同的GT与顺序 | 0预测，GT进入分母 | 原样计分 |
| F / OFFICIAL_CLEAN_FIXED_GT | 同帧官方Clean已选择的目标ID及几何 | 0预测，GT进入分母 | 原样计分 |

F全量每条件固定6224个GT。它保留原版Clean的FOV、Clean object selection、框与范围条件，
天气下不再次用天气点云删除这些目标。身份为scene/timestamp/object ID，几何来自同帧官方Clean的实际最终GT，
并逐ID核对D−1B结果。F不采用历史v2xr_attfuse的无FOV GT。

O*/F使用同一个原版 `eval_utils_opv2v.caluclate_tp_fp` 和 `calculate_ap(...,False)`。
BEV IoU、VOC2010积分、帧顺序、逐帧分数排序完全保持官方实现；没有另一份AP或匹配器。
因此O*−F固定了预测和None处理，主要隔离GT集合变化及其带来的匹配/分母变化。
共同目标按1e-4m角点/中心/尺寸、1e-5rad yaw核对；实际共同角点逐位是否一致和最大数值差也报告。
保留原GT顺序，官方最大IoU并列按原argmax决定，不新增ID排序规则。

## 预测只生成一次

执行字节未改的官方 `inference.main()`，沿用D−1A的batch1、16 worker、shuffle=False、pin_memory=False、drop_last=False，
每条件启动前seed=20261016；score严格大于.20、NMS=.15、COM_RANGE=70m、max_cav=7、fov/proj_first=True。
原版model/postprocess/inference_utils直接执行。独立 `official_runs/<condition>/` 仅复制路径/天气适配config、建立相同权重aliases，
让原版inference输出写入本次目录，避免修改历史model_dir。
原版官方循环同时产生O的原始TP/FP/score/GT与AP，供缓存回放逐项核对。

实际点云沿用D−1B链接视图，该视图中的Clean/YAML/camera/radar源文件全部继承D−1A视图。
Fog/Snow仍由官方eval_sim/sim_weather选择 `_fog_0.060.pcd` / `_snow_2.5_2.0.pcd`。
校验loader全部实际路径与原manifest一致，不重新提取或生成数据。

缓存每帧记录scene/timestamp/ego/CAV context、最终boxes/scores、原dtype、None/empty状态和官方GT ID/几何。
JSON保存完整浮点表示，回放恢复原dtype，不舍入分数或重排序预测。
每个预测文件有内容身份及文件SHA256，所有协议逐帧引用同一文件哈希。
已保存帧恢复时不会再次前向；未原子保存成功的帧才需要重新计算。

D−1B为可恢复GT审计使用逐帧seed和0 worker；预测阶段恢复原D−1A的16 worker设置。
实际生成GT必须与D−1B每帧ID/几何和CAV数量一致，任何差异立即报错，不能偷偷用audit GT替换原版GT。
恢复不跳过dataset前缀，原版loader从第0帧读取，重放同一随机worker序列，只跳过已保存帧的模型前向。
这会重复数据读取，但保护已保存预测不被改写。完整条件缓存直接校验后复用。

## 删除目标专项分析

对Protocol F每次官方 `caluclate_tp_fp` 的实际IoU调用增加返回值不变的观察器。
记录该次score顺序贪心匹配接受的prediction index、object ID和IoU。
removed/retained是同一次全GT匹配的两个ID分组，不分别重跑matcher，也不用最大IoU覆盖率冒充检出率。
校验每个阈值匹配数=官方TP数、prediction和GT均唯一、两组检出数之和=F全部TP。
IoU阈值=.3/.5/.7，报告两组总数、检出数、recall；零分母recall写null。
这些统计描述最终score/NMS后的可检出性，不改变预测，也不能证明具体依赖了哪个协同来源。

## 输出与公式

默认 `/data/cjm/datasets/logs/v2xr-dminus1c-protocol-v1`：

- protocol.json、contract.json：三套精确定义、checkpoint/config/source/native/input/环境/随机设置身份。
- predictions/{clean,fog,snow}/000000.json…：每帧唯一预测及实际官方GT；manifest.json为原子缓存日志。
- prediction_manifest.json：全部预测文件SHA256及缓存身份。
- gt_manifest.json：逐帧official/fixed GT ID与几何身份、来源D−1B文件哈希。
- per_frame_metrics.jsonl：每帧三协议TP/FP/FN、计分GT、跳帧状态、F一对一匹配及相同预测hash。
- removed_gt_metrics.jsonl：每个被删Clean目标的三阈值检出状态及唯一匹配。
- stat_*_O/Ostar/F.json：原版AP积分前完整统计。
- official_runs/：原版inference原始统计、eval YAML及本次独立config。
- summary.json、RESULTS.md：三张主表、删除/保留目标检出表、协议差异和完整性检查。
- progress.json、attempts/、last_failure.json：进度与失败证据。

差值同时报告AP及pp，pp=100×AP差：

```text
NoneSkipEffect = AP(O) - AP(O*)
GTFilterEffect = AP(O*) - AP(F)
OfficialWeatherGap = Clean_O - Weather_O
FixedWeatherGap = Clean_F - Weather_F
GapMasking = FixedWeatherGap - OfficialWeatherGap
           = Weather_GTFilterEffect + Weather_NoneSkipEffect - Clean_NoneSkipEffect
```

GapMasking包含GT集合与两条件None跳帧差，不能全部归因于GT过滤。
纯粹针对GT定义的主对照是O*−F。全部只作描述性协议差异，不称因果统计估计，不预设正负号。

## 本地静态检查与同步

```powershell
python -B v2xr_protocol_audit/check_static.py
```

本地只做AST/LF/哈希检查，不导入Torch/OpenCOOD、不运行AP合成检查或真实数据。
新增包及其source_snapshot整目录同步到服务器，保留LF；历史包和结果不重写、不重生成清单。
服务器无需Git/branch/commit命令，不联网，不编译扩展或安装依赖。
保持main，不建branch/worktree。已钉住的历史源码或输入改变时停止核验，不能为通过检查修改EVIDENCE或历史snapshot。
服务器入口先sh -n，再做完整预检，随后执行7项原版AP合成边界检查，最后生成预测/回放。

## Smoke：前12帧三条件，后台

GPU=0替换成空闲物理HIP卡；解释器默认D−1A的opencood Python，无需切换环境版本。
smoke也重新核验全量输入哈希，首次启动可能较慢。

```sh
cd /home/cjm/OpenCOOD-main/cjmnet
mkdir -p /data/cjm/datasets/logs
nohup env GPU=0 LIMIT_FRAMES=12 \
  OUTPUT_ROOT=/data/cjm/datasets/logs/v2xr-dminus1c-protocol-smoke-v1 \
  sh v2xr_protocol_audit/run_server.sh \
  > /data/cjm/datasets/logs/v2xr_dminus1c_protocol_smoke_v1.log 2>&1 &
echo $!
```

```sh
tail -n 80 -f /data/cjm/datasets/logs/v2xr_dminus1c_protocol_smoke_v1.log
```

smoke应每条件12帧，official GT=230/216/227，fixed GT各230；O回放与原版统计/AP一致，Clean O*=F。
summary.status=complete、scientific_status=SMOKE_ONLY。不会根据smoke解释全829帧协议影响。

## Full：全829帧三条件，nohup

```sh
cd /home/cjm/OpenCOOD-main/cjmnet
nohup env GPU=0 sh v2xr_protocol_audit/run_server.sh \
  > /data/cjm/datasets/logs/v2xr_dminus1c_protocol_v1.log 2>&1 &
echo $!
```

```sh
tail -n 80 -f /data/cjm/datasets/logs/v2xr_dminus1c_protocol_v1.log
```

全量应official GT=6224/3673/5854、fixed各6224；removed=0/2551/370；每条件829条预测，包含所有None/空帧。
对当前Clean O和D−1A已复现AP预先使用1e-6数值容差，额外报告预测总数/None帧数差。
若未对齐，scientific_status会明确提示，保留完整结果和缓存，先审查差异，不调定义/阈值或反复生成预测。

## Resume：同一运行参数，后台

```sh
cd /home/cjm/OpenCOOD-main/cjmnet
nohup env GPU=0 RESUME=1 sh v2xr_protocol_audit/run_server.sh \
  >> /data/cjm/datasets/logs/v2xr_dminus1c_protocol_v1.log 2>&1 &
echo $!
```

恢复smoke时继续保留LIMIT_FRAMES=12和smoke OUTPUT_ROOT。部分完成条件还要求同一记录的物理GPU。
恢复严格校验源码/权重/config/环境/input/GT身份、每帧内容身份和已提交文件SHA256。
文件损坏或契约变化立即停止，不能静默删除/重算旧预测。未完成的一帧可从原子记录恢复或重算。
已全部生成预测、但报告阶段中断时可使用只读缓存模式，不需要GPU前向：

```sh
nohup env STAGE=evaluate RESUME=1 sh v2xr_protocol_audit/run_server.sh \
  >> /data/cjm/datasets/logs/v2xr_dminus1c_protocol_v1.log 2>&1 &
echo $!
```

evaluate模式缺预测时直接失败，永远不会补推理。完整summary已存在时只复核输出及缓存哈希后返回。
每帧缓存、每条件状态和最终summary都原子保存；所有失败记录保留。
支持PYTHON/TMPDIR/GPU/STAGE/RESUME/LIMIT_FRAMES/OUTPUT_ROOT/RUNTIME_ROOT/MODEL_DIR/D1A_ROOT/GT_AUDIT_ROOT/VIEW_ROOT。

## 最终查看与解释

```sh
cat /data/cjm/datasets/logs/v2xr-dminus1c-protocol-v1/RESULTS.md
cat /data/cjm/datasets/logs/v2xr-dminus1c-protocol-v1/summary.json
```

先核对complete、每条件829帧、三个固定GT总数、O原版回放一致、Clean重复对齐、Clean O*=F。
再看三档GTFilterEffect、GapMasking与removed/retained recall，不能只看删除了多少GT。

- GTFilterEffect与GapMasking明显为正、FixedWeatherGap显著大于OfficialWeatherGap，支持官方协议低估固定Clean目标上的天气退化。
  “严重”应结合绝对pp与占FixedWeatherGap的比例判断，本实验不事后设统计显著性或PASS阈值。
- 删除GT很多，但三档GTFilterEffect与GapMasking都接近零，说明当前同一预测下AP协议差异小。
  还需看NoneSkipEffect是否抵消GTFilterEffect，以及匹配/FP变化；仅GapMasking近零不能推出GT过滤影响小。
- removed recall明显大于零，说明部分零框内点目标确实仍被模型检出；removed recall近零说明大多数在当前最终预测下漏检。
  两者都不能单独推断AP影响大小，AP同时受GT分母、TP/FP归属和顺序影响。
