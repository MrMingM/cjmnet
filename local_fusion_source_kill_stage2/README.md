# Source Utility Kill Protocol — Stage 2

K1 检查“选对 CAV 是否真的重要”；K2 检查“区域稍微变一下，原选择还能否成立”。
严格沿用 AI_CONTEXT §42。科研运行只在服务器；本地只做静态和 synthetic fixture 测试。
不训练、不访问正式 test、不启动 K3，不修改历史模块/结果。运行不需要 `.git`。

## 输入与 provenance

正式 P0 只认 `logs/k/P/` 的 complete/full-cpu manifest 和六份报告。
大型 `p0_oracle_target_records.jsonl` 从服务器原 P0 RUN 读取并逐字节匹配正式 manifest。
默认按 manifest 哈希寻找唯一 RUN；不猜最近路径。必要时设置 `P0_RUN`。
Stage 1 RUN 由 P0 manifest 给出，再核验原 §33/S0/B0、模型、split/indices、source hashes。
检查有效数据根和嵌套链接的 realpath，拒绝正式 test 与任意 test alias。

2026-10-14 经用户确认，Same 基准描述修正为 259 / 9 scenes / 55÷259=21.24%；
Task 保持 64 / 8 scenes / 19÷64=29.69%。所有资格及判定门槛保持原样。
资源截止从正式 P0 manifest 读取：2026-10-27T03:23:56.508866+00:00，不重开时钟。
RESOURCE_STOP 是资源停止，不能解释为科学假设被证伪。

## 复用原 Oracle

adapter 执行原 `run_oracle.__code__`，只创建本次调用的 globals，不修改历史模块。
直接使用原 pool、GT proxy、shortlist、compose、state、key、target order、完整 NMS/AP。
Same shortlist ≤6；Task 保留 overall/separated/same 各 top6 后去重，可超过6。
各 family/seed/mode 独立从 Shared clone 开始。null 先完整生成候选，再过原 GT proxy。

## K1 校准与冻结

S0 `.pt` 未保存可直接抽样的完整 decoded outputs。因此服务器用冻结模型额外做 train-only 前向，
不训练、不反向、不生成 joint utility labels。按排序后的 B0 train indices 等间距固定取64帧，四天气；
从 Shared top256 几何有效 anchors 取真实 source-induced 输出变化，不按 GT outcome 选样本。

- N1 整行采样 `[Δprobability, Δx, Δy, Δz, Δh, Δw, Δl, wrapped Δyaw]`。
  joint source pool 供 cls/reg 两头取值，避免独立打乱维度。分层回退固定为
  weather+family → family → global，至少256行；逐帧保存 fallback/clipping。
  概率限制 [1e-6,1−1e-6]，尺寸至少 .001m。decoded box 按原 xyzhwl/yaw 表示重新编码，校验 roundtrip。
- N2 对当前有效已对齐 CAV levels 作 Dirichlet 加权，一候选的一组权重共享全部空间位置与尺度。
  候选数严格2N、同一个KEEP；冻结 deblocks/heads 输出候选。alpha 固定七点 grid，
  仅用 train 的 score/center/size/yaw/candidate-IoU median/p90 macro abs(log ratio) 选择。
  candidate IoU change=`1−IoU(candidate, Shared)`，不读取 GT。
- score-only 取匹配 train Δprobability，`rm` 直接沿用 Shared，geometry 逐位不变。

preflight 先生成不可变根 `PROTOCOL.json` / `protocol.yaml` / `SOURCE_SNAPSHOT.json`。
train 校准结束、任何 validation 前，生成不可变 `k1/NULL_CONFIG.json` 和 `k1/PROTOCOL.json`，
记录实际 alpha/pool hashes/全部指标。根协议预先规定此 train-only 扩展，根文件不更新。
后续 condition identity 同时包含根协议和 null config 哈希。

四天气 Shared/Same/Task AP30/50/70 全部 ≤1e-6 复现后才跑 null。
恶劣天气三个固定 seeds20261014/15/16；Clean 仅第一 seed sanity。
family 先平均 seeds，再平均三个恶劣天气。Same−Shared / Task−Same 分别判定。
5000次 paired scene bootstrap 从逐帧 TP/FP/GT 重新算 AP，不平均 scene AP；
固定 scene block 顺序，保留原 frame 顺序及重采样 scene multiplicity，不跨帧分数排序。
Source/null 在逐帧层核对模型输入、Shared prediction 和有效 CAV 数的指纹；不配对立即停止。

## K2 状态与并列

primary 使用 P0 A_ONLY validation adverse 的 Same discrete positives / same-state true Task positives。
先自动核算 n/scenes/max share，并逐 scene 对齐正式 P0 专用字段。
重放原 A_ONLY，每个相关 frame 的全部原 records/keys 必须与 Stage 1 原件相同。
每个目标从原 baseline current prediction 独立重算八个扰动，不推进 perturbation trajectory。

center/yaw 只改变 ROI corners；GT proxy 的评价 GT 与 post-NMS GT 保持原值。
ROI expansion 乘 .97/1.03（1.5→1.455/1.545）；每次重新 shortlist、compose、NMS、state/key。
原 ROI helper 只接受倍率≥1；对 .97，Stage 2 adapter 等比例缩小 corners 后调用原 helper 的1.0，
选择同一实际区域，原 helper/GT 均不改。实际 mask 的收缩行为由纯 NumPy 单元测试验证。
保持 P0 A_ONLY 的选择资格，同时真实计算全部 shortlisted A/B/E。

equivalent-best 由 discrete6 定义，忽略 score/IoU/tie-break；按 CAV ID / cls-reg CAV pair 去重。
集合有交集即 retention，另报 exact action/source、winner 在 baseline set、top2/Jaccard、family change。
membership 按同一 expansion/cls/reg 比较 A/B/E；进入/退出 shortlist 均计不一致，ROI scale 归一化配对。
center/yaw/ROI 分别 macro variants 判定，另报最差 variant。Task pair retention 仅在仍 true-positive 中计算。
scene bootstrap 保留目标及其全部 variants 的相关性；逐 variant/scene 报 stable/n 与 flip/n。

## 服务器启动

整目录同步，包含 `source_snapshot.json`；同步修正后的 AI_CONTEXT 与正式 `logs/k/P/`。
保留原 Stage 1/P0/Oracle/B0/S0 RUN 和模型。不要手工修改 snapshot/hash。

```sh
cd /home/cjm/OpenCOOD-main/cjmnet
conda activate opencood
export PYTHONPATH=/home/cjm/OpenCOOD-main/cjmnet:/home/cjm/OpenCOOD-main
export ROCR_VISIBLE_DEVICES=2
unset HIP_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES
```

并行后台启动（2/3 换成实际空闲卡）：

```sh
export K1_GPU=2
export K2_GPU=3
sh local_fusion_source_kill_stage2/launch.sh
```

输出 RUN/PID/LOG；默认 RUN=`/data/cjm/datasets/logs/source_kill_stage2_时间`。
默认先静态/preflight/fixture tests/smoke，再并行两个 GPU worker，内部均使用 cuda:0。
单卡时依次启动，避免两套模型同时占显存。

独立后台启动（同一个RUN）：

```sh
RUN=/data/cjm/datasets/logs/source_kill_stage2_$(date +%Y%m%d_%H%M%S)
export RUN
COMPONENT=k1 K1_GPU=2 sh local_fusion_source_kill_stage2/launch.sh
COMPONENT=k2 K2_GPU=3 sh local_fusion_source_kill_stage2/launch.sh
```

resume：

```sh
RUN=/data/cjm/datasets/logs/source_kill_stage2_已有时间 sh local_fusion_source_kill_stage2/launch.sh
```

complete condition 跳过。K1 weather/family/seed/mode 独立提交；K2 weather/mode/frame 独立提交，
同 frame 的 positive targets×variants 同时完成。tmp→atomic rename，残余 tmp 不当结果。
代码/输入/config hash 改变时拒绝 resume，修 bug 必须新 RUN。

```sh
python -m local_fusion_source_kill_stage2.status --run "$RUN"
tail -f "$RUN/both-driver.log"
tail -f "$RUN/k1.log"
tail -f "$RUN/k2.log"
```

独立启动日志为 `k1-driver.log` / `k2-driver.log`。单组件完成不冒充整个 Stage 2 完成。
GPU Source reproduction 是运行时门禁；静态 preflight 不代表复现通过。
本地 synthetic smoke 使用原 shortlist/greedy 和 fixture NMS 边界，真实 loader/GPU/NMS 在服务器检查。
K1 在 train-only 校准后自动先执行两帧 Source-only server smoke，不计算 AP、不运行 null，独立保存于 `k1/smoke/`。
也可先单独后台检查真实链条（只做 smoke，不启动完整 K1/K2）：

```sh
export RUN=/data/cjm/datasets/logs/source_kill_stage2_待运行
nohup python -m local_fusion_source_kill_stage2.server_smoke --run "$RUN" > /data/cjm/datasets/logs/stage2_smoke.log 2>&1 </dev/null &
```

## 输出与停止逻辑

根目录：PROTOCOL/protocol JSON/YAML、SOURCE_SNAPSHOT，以及
`K1_SOURCE_SPECIFICITY.{json,md}`、`K2_LABEL_STABILITY.{json,md}`、`STAGE2_FINAL.{json,md}`。
`k1/`、`k2/` 分别保存 manifest/condition/result/logs；完整明细为
`k1/k1_scene_bootstrap.json`、`k1/k1_seed_details.json`、
`k2/k2_scene_bootstrap.json`、`k2/k2_flip_records.jsonl`。

H1 任一 FAIL/INCONCLUSIVE→主线 STOP；H1 两项 PASS 才保留 H2。
H2 任一 FAIL/INCONCLUSIVE→Task separation STOP。
K3 data gate 只记录定义，不生成新 train joint labels、不实现/启动 K3。
全部 PASS 也仅说明当前 development/validation 下的 headroom 与小扰动稳定性。
