# Local observability premise audit

本实验只判断三个假设：H1 点云约束是否具有方向性；H2 这些方向能否解释真实定位改善并超过点数等标量基线；H3 两个 peer 的方向互补是否解释额外联合收益。没有新网络训练，也没有 AP 增益门槛。

## 已核对的实现与隔离边界

- 阅读 `AI_CONTEXT.md`、S4 的 README/features/extract/replay/coordinates，只作为结构参考；运行时没有任何其他实验模块 import，不读取 S4 RUN、split、标签或缓存。
- 检查了 `attfuse_gspr/{reliability,pillar_vfe,point_pillar_gspr_attfuse}.py`、`attfuse_code/{att_bev_backbone,self_attn}.py`、OpenCOOD dataset/preprocessor/scatter/postprocessor/box/transformation 相关代码。
- 检查了 `opencood/tools/materialize_physics_weather.py` 和 physics fog/rain/snow augmentor。固定天气 PCD 已经物理退化；本实验关闭在线 weather augmentation，避免重复退化。
- 默认冻结合约来自 AI_CONTEXT §3/§6：`/data/cjm/datasets/logs/gspr_joint_full_v1_seed20260907_20260907_161155/config.yaml` 与 `net_best_validation.pth`；checkpoint SHA256 为 `93598c7f6cde8c8caa54155cde98bbe75ff2921481b24928a2172e2d6c5dd4c1`。使用原始 GSPR+AttFuse，不加载 S4 的 B0/v3/ranker。
- 本地没有读取到服务器 artifact。入口在运行时核对实际 config 类型、checkpoint SHA256、严格 state_dict，写入 protocol。不能将本文的默认配置说明误当作已完成服务器验证。
- 数据路径修正：固定物理天气树没有同级 clean。Clean 使用 `/data/scd/datasets/opv2v_official_data_dumping/validate`；fog/rain/snow 使用 `/data/cjm/datasets/opv2v-physics-fixed-v1/<weather>/validate`。
- 只使用 validation 开发数据；所有路径解析 symlink 后检查，拒绝任何 `test` 路径或 S4 artifact 路径。不创建/修改数据、模型或已有实验输出。
- 工作分支 `codex/observability-audit-v1`；已有工作树修改保留。运行不依赖 Git。
- 本地连接 `cjm@admin1` 失败：主机名无法解析。因此没有确认 S4 PID/GPU，也没有运行服务器 smoke/O0。

## 几何定义与需要注意的限制

状态 `[x,y,l,w,yaw]`，点已经通过 OpenCOOD `proj_first=True` 投到 ego；不重复投影。source-to-ego 变换仅用于传感器位置、距离、视角记录。source ID 顺序取当前 dataset 的 `cav_ids`，与逐车处理的 transform 一一对应；GT 按 object ID 稳定排序。

GT oriented 3D box 扩张固定 1.15，框内有效保留 voxel 点参与两种矩阵；四个竖直面的最近面分配只在参考 GT 状态计算一次。x-face 残差 `q_x - sign*l/2`，Jacobian `[-cos(yaw),-sin(yaw),-sign/2,0,q_y]`；y-face 为 `[sin(yaw),-cos(yaw),0,-sign/2,-q_x]`。中心有限差分只做验证，不选择有利扰动步长。

`I_geo=J.T@J`；`I_gspr=J.T@diag(clamp(r,1e-6,1))@J`。它们是几何信息替代量，未假定 GSPR 是逆噪声方差，不能直接宣称为校准 Fisher 信息。

参数化 `theta=theta_ref+S*eta`：四个长度量固定尺度 1 m，yaw 10 degrees（以 rad 存储）；残差保持 m。`I_norm=S.T@I@S`。谱、方向、condition/logdet、互补指标全部用归一化矩阵。另存物理矩阵及其除点数/trace 版本。归一化矩阵同时存 raw、除 N、除 trace 三种版本。

重要数学边界：一面点的五维 Jacobian 秩最多 2；相邻正面+侧面一般最多 3。它们会增加受约束方向，但最小特征值仍可为 0。测试分别验证 rank 2→3、原弱方向增加信息，以及四面覆盖后 full rank/min eigenvalue 增加。绝不为了让测试变绿把两个单面来源伪造为满秩。

无点、单点、奇异矩阵全部保留。`valid` 表示足够点且满秩；`directional_valid` 表示至少两点且非零秩，允许保留真正的弱方向样本用于审计。奇异 condition/unregularized logdet 为 null 并附 validity；regularized logdet 用固定 lambda=0.001 单独命名。重复特征值对应的单一 eigenvector 不唯一，保存 validity；不将任意零空间基向量当作稳定方向。

`point_count` 是体素化前、裁剪后的 ROI 点数；`valid_point_count` 是参与 J 的保留槽位数。GSPR r/u 与同次 full-frame forward 的槽位逐项对应，核对 mask/count/source/grid/finite；不对原始 PCD 做最近邻权重配对。GSPR 的 full-frame 上下文被明确记录，未声称权重与 source subset 无关。

GT 用于 ROI、参考面、状态尺度误差与评价。所有这些属于 **GT-assisted premise audit**；即使通过也不能证明已经得到可部署特征。送给 detector 的字典只有 processed_lidar 与 record_len；GT 不改变推理、NMS、source subset 或模型权重。

## 子集前向与评价

每帧重新计算 ego、ego+每个 peer、ego+每一对 peer（最多 5 CAV 时 11 个组合）。使用相同已体素化点、保留原顺序、重编号 source index，GSPR/PFN/backbone/fusion/head 全部真正执行。原生 score threshold/NMS/range filtering 由 checkpoint config 与原生 postprocessor 决定。

理由：GSPR 的 `bounded_intensity` 使用输入张量整体最大值；其 `clamp_min(1)` 可能使这项在某些数据上恰好不产生 subset 差异，但源码不足以保证普遍等价。smoke 每帧、O0 每天气前 3 帧，缓存真实 backbone fuse 输入，再使用原 fusion/deblock/head 重融合，与真正 subset forward 比较 psm/rm（atol=rtol=2e-5）；另检查真实 forward 重复一致性。

结果始终来自真正 subset forward。缓存比较失败会被记录，缓存结果不会参与 utility；真正重复前向失败、非有限输出、点对应失败会停止，保留 failure.json，不删除帧或降容差。空 CAV 使原 scatter 无法保持合法 source 维度时也显式停止，禁止静默移除来源。

所有 subset 在相同 GT 集上分别按 score 降序做一对一 BEV IoU>=0.5 匹配；tie 按稳定 index。共同匹配 cohort 比较 x/y/l/w 绝对误差、yaw 模 pi 角差、BEV/3D IoU；ego 未匹配而加入来源后匹配的 recovery 单独记录，lost 同样保留。

H3 aggregate gain = `min(error_ego+i,error_ego+j)-error_ego+i+j`，error 为五参数按上述固定尺度归一化后的平均绝对误差。每参数 gain 使用该参数的较好单来源作评价上界。best 使用 GT 仅评价，不形成推理策略。

## 统计与预注册判定

固定 seed=20261012。所有 weather 使用同一独立选帧清单：scene hash 排序、scene 内 frame hash 排序、按 scene 轮转；默认 120 帧/weather。smoke 3 帧/weather；`--full-validation` 使用全部 validation。没有使用 S4 split。

- H1：谱/秩/anisotropy 分天气；留一 scene 的描述性岭回归比较标量控制与增加 `cos(2*view_angle),sin(2*view_angle)` 后对矩阵 shape 对角的解释。
- H2：两种矩阵×三种归一化×五参数×pooled/四天气，Spearman 与 scene-cluster 95% bootstrap CI；标量逐项基线包括两种点数、mean r/u、trace、regularized logdet。固定 alpha=10 的留一 scene 岭回归比较标量 vs 标量+对应方向信息，报告 R²/rho/MAE 增量。
- H3：原定义 C_logdet、C_weak 全部保留，也报告 count-normalized 和 shape 版本、每参数 gain；标量 pair baseline 控制总点数、两来源 mean/difference r、uncertainty、distance、trace/logdet 及天气。
- 额外共同 nuisance controls：GT 参考 l/w、yaw 的 sin/cos，防止参考框几何自身冒充来源信息。这仍是 GT-assisted 描述性分析，无部署宣称。
- 两个回归均按 scene 留出、仅训练 fold 标准化，不调 alpha；bootstrap 抽 scene，保留 scene 内 frame/object/source/weather 依赖。回归增量 CI 对固定 OOF 预测重采样，没有每次重新训练模型，报告中披露这一限制。
- negative controls：同 weather/distance bin 的观测打乱、保谱方向随机置换、固定点数 bin 匹配、固定 reliability bin 匹配、geometry/GSPR 同槽位对照。方向置换在归一化坐标中执行，保留所有 eigenvalues。
- bootstrap=1000，permutations=100；这些是探索性预注册诊断，未宣称经过全部多重检验校正的确认性显著性。

固定 primary 为 **geometry-only、I/N、对应参数 diagonal**；GSPR-weighted、raw、shape 作为报告完整的对照，不能跑完改 primary。H3 primary 为 shape C_logdet；C_weak 的零值饱和必须如实报告。

判定规则在 experiment.yaml 与 protocol 固定：

1. A：pooled median anisotropy>=0.25；H1 view 增量 R²>=0.05 且超过视角打乱 null q95。方向性很大程度由残差结构决定，单独通过 A 无法支持故事。
2. B/C：至少 2/5 参数 pooled rho CI 下界>0，至少 3/4 天气 rho 为正；增量 R²>=0.01 且增量 CI 下界>0、MAE 改善；点数/reliability 分箱内关系为正；原信号超过 view/eigenvector 打乱 q95。
3. D：shape C_logdet 与 pair gain pooled rho CI 下界>0；标量控制后的增量 R²>=0.01、增量 CI 下界>0、MAE 改善，并超过保谱方向打乱 q95。
4. 每个正式 primary 分析至少 30 行、5 个 scene；CI 至少 90% 重采样有效。样本不足/失败 => INCONCLUSIVE，不解释为否定理论。
5. A+BC+D => CONTINUE_RESEARCH；至少有参数通过但三条件不全 => AUXILIARY_ONLY；样本充分仍无参数通过 => STOP_V1_PREMISE。停止结论限于此 fixed-face surrogate+冻结 detector，不是一般不可观测性理论的否定。

## 输出与完整性

输出只允许 `/data/cjm/datasets/logs/observability_audit_<new_id>/`。JSON 原子写；进程 flock 防止同 RUN 并发；每个 source/prediction/diagnostic artifact 都带 identity 并由 manifest 保存 SHA256。

protocol 记录 config、源码、checkpoint、实际 config、选中数据所有 PCD/YAML 内容 hash、完整目录清单 hash、确定性 selection。源码 snapshot 同时覆盖本模块、GSPR、AttFuse 和实际 OpenCOOD Python 源码（排除 logs 和四个明确列出的未导入备份，只规范 CRLF/LF）；环境版本另存并校验。开始/结束/完整 status 验证数据，不采用 mtime 代替内容 hash。没有 .git 要求。

不同 mode 使用不同 RUN。中断恢复只能相同源码、参数、数据、checkpoint 和环境；不要覆盖 source_snapshot.json 以强行继续旧 RUN。异常帧留在 diagnostics 并使任务停止，禁止删除失败样本后继续。

```
RUN/
  protocol.json / manifest.json / environment.json
  source_observability/<weather>/<framehash>.json
  subset_predictions/<weather>/<framehash>.json
  diagnostics/gpu_preflight.json
  diagnostics/frames/<weather>/<framehash>.json
  diagnostics/failure.json          # only on exception
  reports/O0_SUMMARY.md
  reports/H1_DIRECTIONALITY.md
  reports/H2_DIRECTIONAL_UTILITY.md
  reports/H3_COMPLEMENTARITY.md
  reports/NEGATIVE_CONTROLS.md
  final_results.json
```

## 本地检查与同步

本地只执行合成单元测试，不加载 Torch/OpenCOOD，也不读科研数据。几何、统计与缓存测试使用 NumPy/标准库；匹配测试额外需要 Shapely。本机 25 项中 24 项通过、1 项因缺少 Shapely 跳过；服务器入口使用 `--require-server`，缺少该依赖会直接停止，不能跳过匹配测试。两份 shell 脚本语法检查通过。真实 GPU 数值等价性、实际 checkpoint 加载和固定天气数据读取尚未在服务器验证。

```powershell
python -B -m local_observability_audit.test_core
bash -n local_observability_audit/launch.sh
bash -n local_observability_audit/run_all.sh
```

本地实际修改代码后、同步之前导出源码快照（只读源码哈希）：

```powershell
python -B -m local_observability_audit.common --snapshot
```

将整个 `local_observability_audit/` 复制至服务器 `/home/cjm/OpenCOOD-main/cjmnet/`。现有冻结源码必须匹配快照；若服务器不一致，先诊断具体文件差异，不修改 S4，也不手改 hash。

## 服务器运行

先在服务器检查设备、S4 进程和对应进程的可见设备：

```sh
ps -eo pid,args | grep -E 'source_evidence_audit|python' | grep -v grep
rocm-smi --showuse --showmeminfo vram --showpids
# 对 ps 中识别的 S4 Python PID：
tr '\0' '\n' < /proc/<S4_PID>/environ | grep -E '^(ROCR|HIP|CUDA)_VISIBLE_DEVICES='
```

不 kill/pause/restart 任何现有任务。v1 保守预留 GPU 2、7；其余 GPU 也必须通过空闲检查。入口读取 ps/rocm-smi/proc，仅记录可见设备字段；存在冲突、读不出 S4 GPU、设备繁忙或 rocm JSON 不兼容时退出。检查是瞬时状态，无法替代集群调度预约。

Smoke（3 帧×4天气，后台）：

```sh
cd /home/cjm/OpenCOOD-main/cjmnet
conda activate opencood
export GPU=<确认空闲的物理卡号>
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
sh local_observability_audit/launch.sh --smoke
```

保存输出 RUN/PID/LOG。完整 status 成功且 smoke 子集检查报告已确认后，O0（120帧×4天气，新的后台 RUN）：

```sh
unset RUN
sh local_observability_audit/launch.sh
```

只有 O0 信号支持继续研究时才另开全 validation：

```sh
unset RUN
sh local_observability_audit/launch.sh --full-validation
```

状态/日志（RUN 改成启动器实际打印的目录；完整校验会读大量数据）：

```sh
export RUN=/data/cjm/datasets/logs/observability_audit_<实际时间戳>
python -B -m local_observability_audit.status --run "$RUN" --quick
tail -n 80 -f "$RUN/driver.log"
# 完整性检查也使用后台，避免长时间读盘阻塞终端：
nohup python -B -m local_observability_audit.status --run "$RUN" > "$RUN/status_check.log" 2>&1 < /dev/null &
```

同 RUN 恢复（原 smoke 必须保留 --smoke；原 full-validation 必须保留对应参数）：

```sh
nohup sh local_observability_audit/run_all.sh --smoke >> "$RUN/driver.log" 2>&1 < /dev/null &
```

无空闲 GPU 就不运行 smoke/O0。launch 的 PID 只代表后台进程启动，不表示检查通过；以日志及 status 为准。

## 2026-10-12 ROCm 空闲检查兼容修复

服务器实际输出为 `HCU use (%)`、`vram Total Used Memory (MiB)`。解析器兼容该格式和 `GPU use (%)`、`VRAM Total Used Memory (B)`，MiB 显式乘以 1024²。未知单位、缺字段、非有限值和重复字段均停止，错误包含原始 rocm-smi 输出。使用用户提供的 GPU 0/4 字段值做合成回归测试；保留 S4 进程检查和原空闲阈值。同步 device_guard.py、test_core.py、source_snapshot.json、README.md 后，使用新 RUN 重启 smoke；尚未验证修复后的服务器运行。

## 2026-10-12 S4 进程识别修复

旧检查按整行是否含 source_evidence_audit 判断，会将查看 S4 日志的 tail 误判为实验进程。用户服务器进程列表包含 S4 pipeline、ranker 及 tail。新检查按命令首个可执行程序分类：明确的日志查看/查询工具不作为计算进程；Python pipeline/ranker 继续检查设备变量；未知 S4 启动器继续保守检查。设备不可确认时错误输出 PID、命令及仅设备可见性变量，便于定位；不取消占用阈值或保留卡限制。新增合成回归测试，未在本地执行科研实验，也未修改 S4。修复后的服务器 smoke 仍待运行。

## 2026-10-12 源码范围与服务器差异诊断

旧快照递归包含 OpenCOOD 示例 YAML、历史 logs 和备份，范围过宽。现在排除这些无关文件；四个备份排除项在 common.include_source 中逐项列明，本地 import 搜索无引用。实际 checkpoint config 继续单独计算内容哈希。basedataset.py 等正式 Python 源码仍严格校验，不能通过重新生成服务器快照掩盖差异。

只同步本模块后运行 `python -B -m local_observability_audit.common --check-sources`。该命令报告实际 OpenCOOD 根目录、所有差异哈希，并用随模块附带的只读 reference_sources/basedataset.py.txt 输出统一文本 diff。参考文本不导入、不替换服务器源码。诊断不需要 GPU，也不读取数据集。用户所报 basedataset.py 差异尚待服务器 diff 核实；此前停止在源码检查阶段。不要覆盖 S4 依赖的 OpenCOOD 文件，也不要直接在服务器执行 --snapshot。

## 2026-10-12 已审阅 basedataset 服务器差异

用户提供的唯一差异是排除 data_protocol.yaml，避免把协议元数据当作帧。依据完整 diff 重建服务器文件，其 SHA256 精确匹配用户输出 62ef160af933e88db9ab1019e9d44b730e879bb3b7aca9848e89294c537fe381。reviewed_source_variants.json 登记这一个精确版本，并附 reference_sources/basedataset.server.py.txt；二者自身纳入冻结源码快照，任一被改动均不放行。运行 protocol 始终记录实际服务器源码哈希，恢复及结束检查仍要求该实际版本不变。新实验 inventory 同步排除 data_protocol.yaml，合成测试确认帧号连续且不混入元数据。没有修改本地或服务器 OpenCOOD 原文件、S4 文件。

同步整个本实验目录后先运行 --check-sources，预计显示 ACCEPTED reviewed server variant 与 PASS；随后使用新 RUN 启动 smoke。不要覆盖服务器 opencood，也不要在服务器重建快照。
