# D−1B：同一帧的官方天气 GT 会不会变化？

代码已实现，**尚未运行服务器统计，没有 Fog/Snow GT 结论**。
D−1A 已完成829帧：AP30/AP50/AP70=0.85993605/0.82279434/0.66625452，全部帧 GT=6224。
本入口接受该结果作为前提；重新读取其机器可读证据，只验证输入/runtime 身份和逐帧 Clean GT，没有重新运行 detector 或 AP。

## 复用方式

- 使用 `/data/cjm/datasets/v2xr-official-runtime-v1` 已编译的原版 native ROI 与官方源码。
- 使用 D−1A `model_dir/config.yaml`，同一官方 YAML parser。原 model/config 不改写，不加载 detector/权重。
- registry=`IntermediateFusionDatasetV2XR`，官方 `dataset[index] → collate_batch_test → post_processor.generate_gt_bbx`。
- `train=False, visualize=True, fov=True, proj_first=True, max_cav=7, COM_RANGE=70`；其余行为配置继承 D−1A。
- AP 不在审计中执行。后续仍固定 `eval_utils_opv2v.py`、BEV IoU、VOC2010、global_sort=False、score=.20、NMS=.15。
- 全829帧都审计，空 GT 帧保留，完全不依赖 prediction，更没有 prediction=None 跳帧。

官方天气选择源码位于：

- `upstream/opencood/data_utils/datasets/opv2v/basedataset.py:68–77`：读取 eval_sim/sim_weather。
- 同文件 `190–193`：仅在 `eval_sim and not train` 时读取 `timestamp + sim_weather + '.pcd'`。
- 原 YAML 的 Clean 为 `eval_sim=False`，其 sim_weather 字段不生效。
- Fog 为 `eval_sim=True, sim_weather='_fog_0.060'`。
- Snow 为 `eval_sim=True, sim_weather='_snow_2.5_2.0'`。

程序校验全部 loader 实际选中的文件路径，不只检查配置。
输出目录的 `data_view/validate` 只建立链接：D−1A 的 Clean/YAML/radar/四相机逐文件复用，天气链接到
原 paired 提取包 `payload/validate/<scene>/<cav>/<timestamp>_<weather>.pcd`。
先核验 paired plan 的 variant、member、timestamp、condition、size 与提取记录 SHA256。
原始数据视图、paired 目录与历史结果均不修改，不重新解压或模拟天气。

## 记录与判定

`per_frame.jsonl` 每行是一帧，包含 Clean/Fog/Snow 三条件：

- scene、timestamp、ego，全部 CAV 的 pose/transform/time delay/实际输入路径，实际参与 GT 的 CAV。
- 所有 CAV 原始 YAML 的 ID 和数量；同时给出去重 union 数量与按 CAV 相加数量，避免把重复观测当多个目标。
- 每 CAV 进入点依赖 GT 过滤前的 object ID、ego 坐标 box；候选范围过滤移除的 ID。
- 官方 native ROI 实际返回的逐框点数、保留 ID、零点删除 ID。
- 官方 FOV 前后点数、空体素临时点标记；同一 native 算子计算的临时点生成前框内点数。
- 最终 evaluation GT 的 object ID、`[x,y,z,h,w,l,yaw]` 与实际角点，以及最终范围筛选移除 ID。
- Clean→Fog、Clean→Snow 的 retained/removed/added、几何变化目标、每帧数量差。
- Fog→Snow 的共同 ID 几何检查，覆盖三组两两比较。

最终结果按 object ID 建表比较，不以数组序号对齐。中心、尺寸、角点绝对容差 `1e-4 m`，
yaw 容差 `1e-5 rad`，按 2π 周期比较。即使 GT 数相同，ID 或任一几何项不同也计为变化帧。

`removed_gt.jsonl` 是 Clean 中存在、天气最终 GT 不存在的逐目标记录。
包含该目标 Clean ego 距离、各 CAV 框内点数、best-CAV、ego/peer evidence、是否涉及临时点与删除原因。
best-CAV 指同目标最多点的单个 CAV，跨 CAV 点数不相加；ego 不含候选框时记录 null。

`summary.json` 包含三条件 GT 总数、净减少和占 Clean 百分比、retained/removed/added、变化/不变帧、
等数量但变化帧、几何差的最大值、GT 数变化分布、删除目标距离/点数分布及简单 CAV 拆分。
`removed` 是 Clean 丢失的目标数；`net_fewer=removed-added` 是总数净少，两个数都报告。
新增可由天气散射点进入框内、官方临时点或去重代表框变化后范围筛选引起，不假设只会减少 GT。

## 保留官方随机边界行为

官方 `get_item_single_car` 在空体素时生成临时随机点，随后 `projected_lidar` 可能含这些点。
审计保留这段源码及行为，同时记录真实 post-FOV 点数与送入过滤的实际点数。
天气间候选 ID/box、CAV context 不一致会报错；删除目标在全部天气 CAV 候选中均为零点，才归因为零点过滤。
有临时点介入的目标单独标记，不能简单写成物理天气删点。

D−1A 使用16个 DataLoader worker；为单帧恢复，本审计在主进程顺序调用同一 dataset。
每帧每条件重设 Python/NumPy/Torch seed=`20261016+index`，同一帧各条件起始 seed 相同，
但点数或 fallback 次数不同会使后续随机序列不同，不承诺临时点逐个配对。
运行参数及这一差别完整写入 protocol。每帧 Clean GT 数、参与 CAV 数必须与 D−1A 对应帧相同；
全量 Clean 总数必须是6224，否则停止，不输出完成的天气结论。
前12帧每条件还从相同 RNG 状态重跑无观察器的原版 dataset，要求最终 IDs/GT 角点与观察结果一致。
以上是运行完整性检查，不替代829帧实测。

## 同步与本地检查

保持项目 main，不创建 branch/worktree。同步整个 `v2xr_official_repro/`，保持 LF 和 upstream 字节。
服务器不要求 `.git`、分支或 commit 检查，不联网、不安装/重编译 runtime。
新增代码全部位于本子目录；原 D−1A 文件逐字节保留，仅更新顶层 `source_snapshot.json` 纳入新增文件。
旧 manifest 原件保存在 `d1a_source_snapshot.json`，静态检查逐项验证旧文件未变。
历史运行契约仍保存在原结果目录，不覆盖。

本地只运行静态检查：

```powershell
python v2xr_official_repro/gt_audit/check_static.py
```

以下 smoke 和全量命令都只在服务器执行。入口先运行六项纯合成报告边界检查，再执行数据审计。
原生 ROI 调用和原版 voxelization 均为 CPU 路径，无 detector，不需要指定 GPU。
仍须使用 D−1A 的 opencood Python/HIP runtime 与已编译扩展。

## Smoke：前12帧，后台

```sh
cd /home/cjm/OpenCOOD-main/cjmnet
mkdir -p /data/cjm/datasets/logs
nohup env LIMIT_FRAMES=12 \
  OUTPUT_ROOT=/data/cjm/datasets/logs/v2xr-official-gt-audit-smoke-v1 \
  sh v2xr_official_repro/gt_audit/run_server.sh \
  > /data/cjm/datasets/logs/v2xr_official_gt_audit_smoke_v1.log 2>&1 &
echo $!
```

smoke 仍先校验全部输入文件哈希，首次启动可能较慢。
成功必须 summary.status=complete、validation_frame_count=12、clean_per_frame_counts_match_d1a=true，
observer_equivalence_checks 共36项，且没有跳帧。不能根据这12帧判断全829帧是否一致。

## 全829帧，后台

```sh
cd /home/cjm/OpenCOOD-main/cjmnet
nohup sh v2xr_official_repro/gt_audit/run_server.sh \
  > /data/cjm/datasets/logs/v2xr_official_gt_audit_v1.log 2>&1 &
echo $!
```

```sh
tail -n 80 -f /data/cjm/datasets/logs/v2xr_official_gt_audit_v1.log
```

```sh
cat /data/cjm/datasets/logs/v2xr-official-gt-audit-v1/RESULTS.md
cat /data/cjm/datasets/logs/v2xr-official-gt-audit-v1/progress.json
```

## 恢复

中断后在原命令增加 `RESUME=1`；保持 OUTPUT_ROOT、LIMIT_FRAMES、源码、环境与输入相同。
保留 `frames/` 中完整三条件帧；未完成帧重算，已完成帧经契约/内容哈希校验后复用。
每次启动重新核对全输入 SHA256，不依赖 mtime/文件大小推断内容相同。
聚合 JSONL 在所有帧完成后由原子帧记录重建；不从可能截断的 JSONL 尾部恢复。
已经完成的输出也可 RESUME=1 复核哈希后直接返回。
不删除失败证据；`attempts/*/failure.json` 与 `last_failure.json` 保留错误详情。
源码/输入/config/runtime改变需要新 OUTPUT_ROOT，不手改契约或源码快照。

如果启动时报 `Package differs from source_snapshot.json`，先只检查代码目录：

```sh
cd /home/cjm/OpenCOOD-main/cjmnet
/home/cjm/miniconda3/envs/opencood/bin/python -B \
  v2xr_official_repro/gt_audit/package_check.py
```

此命令只读取源码、打印哈希差异，不导入Torch、不读数据、不运行实验、不重写清单。
`missing`表示缺失文件，`unexpected`表示目录多出的文件，`changed`列出内容不匹配的文件及预期/实际SHA256。
若只同步gt_audit子目录、保留顶层D−1A旧source_snapshot，新文件会出现在unexpected中。
若传输或编辑引入CRLF/BOM，诊断会在能通过哈希确认时给出转换证据；仍拒绝运行。
完整同步整个实验包及顶层source_snapshot，使用二进制传输保持字节；不要在服务器重生成快照掩盖差异。
对多余文件先核实来源，不自动删除。确认status=complete后再启动原smoke命令。
此错误发生在main创建输出契约和读取数据之前，这次失败没有生成需要恢复的GT帧。

```sh
nohup env RESUME=1 sh v2xr_official_repro/gt_audit/run_server.sh \
  >> /data/cjm/datasets/logs/v2xr_official_gt_audit_v1.log 2>&1 &
echo $!
```

支持 PYTHON、TMPDIR、RUNTIME_ROOT、MODEL_DIR、VIEW_ROOT、PAIRED_ROOT、D1A_ROOT、OUTPUT_ROOT、LIMIT_FRAMES、RESUME。
默认 TMPDIR=`/data/cjm/datasets/tmp/v2xr_official_gt_audit`，输出与临时目录必须在数据盘 `/data`。
校验成功 D−1A 的 summary/protocol/contract/frames、原数据 manifest、paired plan/manifest/records、
源码 SHA256、原生扩展 SHA256、依赖版本和解析配置。所有这些都无需服务器 Git。

## 如何读结论

必须先确认 `status=complete, full_validation=true, validation_frame_count=829, skipped_frames=0`，
并确认 `Clean_GT_total=6224, clean_per_frame_counts_match_d1a=true`。
`all_conditions_GT_identical=true` 才能回答本验证集在约定容差内三天气最终 GT 完全一致。
否则查 comparisons 下的 removed/added、geometry_changed_objects、equal_count_but_changed_frames；
不要只看 GT_total 或净减少。
`point_filter_share_of_removed` 是删除目标中“所有天气 CAV 均为零点”的占比；
`removed_with_fallback_involved` 和原始/实际框内点数用于检查归因是否受临时点影响。
没有删除目标时占比为 null，不写0%或100%。

如果 GT 随天气变化，官方天气 AP 同时反映模型变化和计分目标变化。
后续分别报告官方 AP 与 paired fixed-GT AP；固定权重、预测后处理、AP 实现后再做受控比较。
GT 一致也不能自动消除 prediction=None 跳帧或旧实验权重/协议差异。
本任务不计算 AP，不量化 GT 变化造成多少 AP 偏差，不修改任何官方过滤行为。
