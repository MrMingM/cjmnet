# D−1D：V2X-R Clean AttFuse 域内 baseline

## 已接受的数据偏差：实际8083，公开参考8084

用户已审阅完整 inventory 并明确回复“可以接受”。本实验使用原始数据中实际存在的8083个协作帧；公开参考8084及差值−1继续记录，不补造、重复或额外删帧。
`TRAIN_DATA_ACCEPTANCE.json` 固定这次授权、原 inventory identity 和四个报告文件 SHA256。
`train_acceptance.py` 先在 `/data/cjm/datasets/v2xr-official-train-v2-inventory` 找到原报告并验证其哈希，随后核对新扫描的全部原始包哈希、逐帧身份和顺序、ego/CAV上下文、空场景与元数据。
相同数量但其他原始包或帧发生变化也会拒绝。请保留原成功 inventory 的整个时间戳目录。
全量目标为33个原始场景、32个有数据场景、1个零帧场景、97个原始包（含5个空目录包），共8083个协作帧。
接受计数偏差后仍必须逐包通过 CRC、文件哈希与内容验证，才允许训练。

同步整个 `v2xr_train_attfuse/`（包括自己的 snapshot）后，恢复当前尚未创建 plan/数据的 v2 失败任务：

```sh
cd /home/cjm/OpenCOOD-main/cjmnet
nohup env MODE=extract SMOKE=0 RESUME=1 \
  TRAIN_ROOT=/data/cjm/datasets/v2xr-official-train-v2 \
  sh v2xr_train_attfuse/run_server.sh \
  >> /data/cjm/datasets/logs/v2xr_train_extract_full_v2.log 2>&1 &
echo $!
```

```sh
tail -n 80 -f /data/cjm/datasets/logs/v2xr_train_extract_full_v2.log
```

成功依据为 `manifest.json.status=complete`、`full_train=true`、`integrity_complete=true`、实际帧数8083；
`data_acceptance.json` 及 manifest 同时保存公开8084/接受8083与原始证据哈希，所有包必须有完成的 CRC 记录。
原始 `inventory_summary.json.status=count_mismatch/training_allowed=false` 继续描述与公开数量的差异及只读扫描状态；
是否可训练由用户接受契约与完整提取 manifest 共同决定。没有确认缺失帧的具体身份，也没有认定官方数字有误。

以下独立清单工具仍可用于只读检查；已经生成并通过哈希校验的原报告无需重跑。
独立 `MODE=inventory` 读取全部 train 压缩包清单和 SHA256，不解压、不导入 Torch/OpenCOOD runtime、不修改旧提取请求。
输出放到 TRAIN_ROOT 后加 `-inventory` 的独立目录，每次运行单独保存。

```sh
cd /home/cjm/OpenCOOD-main/cjmnet
nohup env MODE=inventory SMOKE=0 RESUME=0 \
  TRAIN_ROOT=/data/cjm/datasets/v2xr-official-train-v2 \
  sh v2xr_train_attfuse/run_server.sh \
  > /data/cjm/datasets/logs/v2xr_train_inventory_v2.log 2>&1 &
```

产物：inventory_summary.json、inventory.json、scene_counts.tsv、archive_counts.tsv、frame_keys.jsonl。
summary 对照公开8084、实际物理帧数、逐场景 ego 按官方 YAML 文件名规则的计数，以及重复帧键、忽略的 YAML 名称和候选时间缺口。
inventory.json 保存每包成员名/时间戳/哈希和 scene data_protocol.yaml 的文本预览，便于后续核对原始元数据。
只有官方时间步的参考清单或其他可靠源证据才能确认具体缺帧；候选缺口也可能来自合法的可变采样。
这些报告始终 training_allowed=false；即使数字匹配也尚未完成解压、CRC和内容完整性验证。
官方说明文件真实名为 `upstream/readme.md`（小写）。引用名先按 UPSTREAM.json 大小写严格检查，再开始扫描。
本地静态检查也按冻结清单验证名字，不依赖 Windows 是否允许大小写混用。
扫描结束会先保存 inventory_scan.json，再生成汇总；报告格式或引用出错也保留已读到的包/场景/时间戳证据。
正常提取会在接受契约检查前保存同样报告，计数或身份不符时不会丢失完整清单。
已固定源码版本且尚无 plan/数据的8083计数失败，可在 request_history 记录新旧身份后进入上述已授权检查。

## 压缩包 listing 失败的只读诊断

`Empty/invalid archive listing` 表示冻结解析器没有得到有效普通文件记录；单凭这条消息不能认定压缩包损坏。
新增 `MODE=inspect` 只读取原始 train 的技术清单，不解压、不导入训练 runtime、不修改旧提取 request/plan/records。
只读检查会扫描完所有候选包，将异常统一写入每次运行的 `inspection_summary.json`，不在第一个异常处停止。
`scan_completed=true` 仅表示扫描结束；异常存在时 status=failed，不写 inspection_complete.json，不代表数据通过验证。
失败时保留实际 archive 路径、7-Zip 返回码、stdout/stderr 原始字节和 diagnostic.json。
正常提取入口也会记录同样的诊断。未分类的 listing 失败仍会停止。

用户回传的 `2024_06_26_22_12_38/-1.7z` 为82字节、Blocks=0，只含同名空目录。
目录占位包按内容验证，正负 CAV ID 使用相同条件：数字 CAV ID、仅一个同名目录、Size/Packed Size=0、Blocks=0、无文件/链接。
正 ID 如1084也可以携带空目录；只有运行时技术清单确实满足所有条件才接受，不能根据文件名或82字节大小猜测。
它保留 SHA256、原始清单证据和每包状态，提取阶段必须通过 `7zz t`；不在 `train/<scene>/` 下创建空 CAV 目录。
官方 loader 会登记所有目录，并直接按时间戳取各 CAV 数据，因此空目录会造成缺帧查找错误；官方 loader 源码不修改。
有文件的包仍严格检查 Clean PCD/YAML/雷达/四相机及时间戳，不能以“空包”掩盖不完整数据。
若一个场景所有原始 CAV 包都被逐包证明只有空目录，则它只有零个实际时间戳，记录为 `zero_frame_scene_placeholder`。
这类场景列入 plan/manifest.empty_scenes，所有包仍保留哈希并通过 `7zz t`，但不创建 loader 场景目录。
用户日志已确认 `2024_06_26_22_12_38` 的五个包均满足这一条件；不推断其他场景也为空。
任何有文件但缺数据、损坏、时间戳不齐的场景仍会失败，不允许整体排除。
有效协作帧总数必须符合已授权的8083及确切源/帧清单，max_cav仍为7；全是零帧占位场景的输入直接拒绝。

```sh
cd /home/cjm/OpenCOOD-main/cjmnet
nohup env MODE=inspect SMOKE=1 \
  TRAIN_ROOT=/data/cjm/datasets/v2xr-official-train-smoke-v1 \
  sh v2xr_train_attfuse/run_server.sh \
  > /data/cjm/datasets/logs/v2xr_train_archive_inspect_v1.log 2>&1 &
```

诊断位于 TRAIN_ROOT 后加 `-inspection` 的独立目录，每次运行单独保存。
此命令不需要 RESUME；源码更新后仍可用于检查旧失败目录对应的原始包。
已有提取 request 固定旧源码身份，不能手改 request/snapshot。
本修复在 `RESUME=1` 时允许一个明确记录的升级：仅限已列入哈希白名单、因 listing、已确认的零帧场景或8083计数检查失败且尚无 plan/train/records/manifest 的旧尝试。
旧 request 和失败证据保存在 `request_history/`，记录新旧身份，不复用任何旧 payload/cache。
已经产生 plan、数据、训练或预测缓存的目录拒绝升级源码身份。

需要重新做提取 smoke 时，使用独立新目录；旧已完成 smoke 保留为历史记录：

```sh
nohup env MODE=extract SMOKE=1 \
  TRAIN_ROOT=/data/cjm/datasets/v2xr-official-train-smoke-v2 \
  sh v2xr_train_attfuse/run_server.sh \
  > /data/cjm/datasets/logs/v2xr_train_extract_smoke_v2.log 2>&1 &
```

若原失败任务是默认目录下的 full extraction，使用 `MODE=extract RESUME=1`，不设 SMOKE；始终保持原任务目录和 smoke/full 参数。

本工具先提取、核验完整 train，再从随机初始化训练官方 LiDAR AttFuse。
每个 epoch 只看 Clean validation；20 轮结束、选择结果锁定后，再单独运行 Fog/Snow。
这里交付代码和服务器命令，尚未启动提取、训练或评测。真实数据只允许在 Linux 服务器处理。

## 继承证据和边界

已阅读 AI_CONTEXT.md、D−1A 实现/结果、D−1B 全量结果、D−1C 全量结果及协议、旧 v2xr_attfuse 的历史 GT 设计。
本地 D−1B 证据目前在 `logs/D/D1/no1/`，D−1C 在 `logs/D/D1/`。
AI_CONTEXT 中“D−1B 未运行”的旧记录，由用户确认的 D−1B/D−1C 完成事实覆盖。

- D−1A：官方 commit `f3ebbad95561444a53a17358735b0f0dc4ba89f7`，Clean AP70=0.66625452。
- D−1B：Clean/Fog/Snow 最终 GT=6224/3673/5854；天气删除2551/370，保留目标几何没有变化。
- D−1C：Fog/Snow AP70 GTFilterEffect=10.7819/1.5868 pp，GapMasking=12.0085/2.0919 pp。
- `PINS.json` 固定历史两个代码包的全部文件哈希、已完成 D−1C 证据和 GT manifest 哈希。
  不改历史 snapshot，不通过修改旧 manifest 绕过检查。服务器无 `.git` 也能运行。
- 只新增这个独立目录；不修改旧代码、数据、权重或结果。不创建 branch/worktree。
- 原始数据入口直接限定为 `/data/cjm/datasets/V2X-R/train`；路径及链接目标包含 test/testing 时拒绝。
  不列举、打开或扫描原始 test 目录。

## 核对到的真实官方训练行为

官方 README 的 from-scratch 示例使用 `opencood/tools/train.py`。
`--model_dir` 是可选的继续训练入口；没有要求预训练。首次运行不给它这个参数，也禁止训练内部调用 `torch.load`。
恢复时，只允许本次完整状态中的模型参数，绝不允许作者或 OPV2V checkpoint 初始化。

本工具直接执行冻结 runtime 中的 `train.main()`，保留模型、loss、增强、反向传播、优化器和调度调用。
运行副本 YAML 只改 `root_dir`、`validate_dir`。其余内容，包括未激活的 sim_weather，均保持官方文本。
保留 batch_size=8、20 epochs、Adam lr=.002/eps=1e-10/weight_decay=1e-4、MultiStep milestones10/15/gamma=.1；
voxel=[.4,.4,4]、range=[-140.8,-40,-3,140.8,40,1]、fov/proj_first=true、score>.20、NMS=.15。
YAML 没有 max_cav；实际 dataset 必须验证为默认 **7**，绝不改成5。

官方单 GPU 的两个 loss loader 都是16 workers、batch8、shuffle=True、drop_last=True、pin_memory=True、prefetch_factor4。
validation loss 也使用 train_dataset.collate_batch_train。
因此接受的8083个 train cooperative frames 每轮名义载入8080个、丢弃3个尾项；官方零 GT 重采样行为保留。
829个 validation frame 的 loss 名义载入824个、丢弃5个尾项。
**每轮 Clean AP 单独评测全部829帧**，包括所有 None 帧的 GT；不会拿824帧的 loss loader 计算 AP。

官方每轮在验证前保存 `net_epochN.pth`；best-validation-loss 保存代码被注释，没有启用的 best-model/AP 选择规则。
调度器在每轮末尾调用 `scheduler.step(epoch)`，epoch 从0开始。本工具保留这个实际调用顺序，不根据 YAML 猜一个新时间表。
官方末尾的自动推理 shell 命令被拦截，改为用户显式执行本工具的最终评测。

## 提取和完整性检查

服务器上先实际检查 `train/<scene>/<numeric-CAV>.7z` 清单。遇到不同结构会明确失败，保留诊断，不能猜路径继续。
只允许已识别的 CAV 包和场景 data_protocol.yaml，避免漏掉一个未知格式的车辆包却报告完整。
按官方字典序排序有数据的场景/CAV/时间戳，负 ID RSU 移到末尾；逐有数据 CAV 核对时间戳一致。
目录占位包单独列入 scenes.empty_agents、archives.kind 和 counts.empty_cav_archive_count，不计作有数据 CAV。
normal PCD 与 YAML 一一对应，官方 loader 即使纯 LiDAR 也要读取 radar 和四张相机图，全部作为必需文件验证。

全量提取所有安全的普通文件，包括已有 Fog/Snow 文件；**训练 loader 只允许选择 `<timestamp>.pcd`**。
天气文件只做存在性、归档 CRC、文件哈希/头部完整性记录，不用于训练、epoch 选择或超参数调整。
完整性检查包括每包 `7zz t`、解压后 size/CRC/SHA256、Clean 官方 PCD/YAML 解码、YAML 重复键/ID、有限值、PCD 点数和四张 PNG 校验。
归档中的路径逃逸、链接、同名展开冲突和缺文件都直接失败。

每个包有独立 `records/<scene>/<CAV>.json`，文件在临时目录校验后逐个原子发布并写 journal。
恢复时检查 archive/source/plan/file 哈希，只请求尚未提交的成员；已经完成的文件不重复解压。
原始包的 SHA256 和完整 CRC test 也会记录。最后全体校验成功才原子写 `manifest.json`。
失败记录在 `last_failure.json`、`failure.json` 或 `inspection_failure.json`，失败包不能标为 complete。
原始包保留，后续训练启动也会重核其哈希；不要在实验完成前移动/修改这些包。

`manifest.json.counts` 分开记录 scene_count、cooperative_frames、有数据的 scene_agent_count、agent_frames、clean_pcd、全部pcd/yaml/fog/snow/radar/png。
source_archive_count 包含所有原始包；empty_cav_archive_count 单列已验证的空目录占位包，不能将其写成损坏包或真实 CAV。
source_scene_count 是本次实际检查的原始场景数，等于 scene_count（有数据）加 empty_scene_count（已证明零帧）。
raw_source_scene_count 为原始 train 下全部场景目录数；全量运行时与 source_scene_count 相同。
empty_scenes 列出每个零帧场景的全部包及哈希，验证器会核对所有原始场景/包都有归属，并拒绝空场景进入 loader 目录。
一个协作帧由同一 scene/timestamp 的多辆 CAV 组成，所以单车点云数通常大于协作帧数。
全量必须为已接受的8083个协作帧，并与被接受的原始包/帧清单完全一致；额外文件、缺失、重复或解码错误都不允许训练。
`missing/duplicate/broken_archive` 在成功 manifest 必须为空，失败原因单独保存。

Smoke 取第一个有数据场景的前2个时间戳及该场景有数据的 CAV；此前遇到的零帧场景也逐包记录并校验。
其 manifest.full_train=false，不能用于正式训练。
Smoke 与 full 必须使用不同 TRAIN_ROOT。

## 提前固定的模型选择和恢复

首次构造模型前写入 `selection_policy.json`：随机种子20261017，全部20轮中 **Clean validation F AP70 最大**；
使用未四舍五入的值，完全并列取最早 epoch。Clean F 与 O* 完全相同，避免 None 跳帧改变选择分母。
val loss、AP30、AP50 均记录，不参与选模；Fog/Snow 不进入选模。
推理严格继承 D−1C 的 seed20261016 和 PYTHONHASHSEED20261016；训练 Python/NumPy/Torch 的随机种子独立固定为20261017。

每轮 Clean AP 在独立子进程运行，读取已保存 checkpoint。它不会改变父训练进程的模型、优化器或随机状态。
GPU 上父进程仍保留训练状态，因此显存应同时容纳训练驻留状态和 batch1 推理。显存不足会失败，不自动缩小 batch_size 或改配置。
每次 AP 也严格复核 D−1A/B 的源/输入证据；重复哈希检查可能需要较长时间，这是预期行为。
准备阶段会读取历史天气 GT/输入哈希来验证冻结身份；选择指标和每轮推理仅使用 Clean，不读取天气性能作为决策输入。

每轮保存官方权重以及完整 model/optimizer/scheduler/loss日志累积状态/Python/NumPy/Torch CPU+CUDA RNG。
只有 checkpoint、Clean AP、完整恢复状态都成功才写 `epochs/NN/complete.json`，这是可恢复边界。
中途失败从最近完整 epoch 重跑；已提交 checkpoint 永不覆盖。
未提交 epoch 的权重可以原子更新，旧 AP 尝试按 checkpoint hash 独立保留，不混用缓存。
每次 AP 的输入权重另存不可变硬链接，重新提交同名 epoch 权重也不会改变旧缓存指向的输入。
`history.json` 只汇总成功提交的 epoch；官方原始 train_loss.txt/validation_loss.txt 可能包含中断尝试的重复行。
恢复不声称跨硬件或 HIP 非确定性算子逐位复现；GPU、源码、配置、数据和环境改变时拒绝继续。

20轮全完成后写 `selected.json`，最终评测再次按全部 Clean 记录核对选择结果。
选择规则不能在看过天气结果后改变。`SMOKE=1 MODE=train` 仅在完整 train 准备后做2个训练/验证batch与12帧Clean AP，输出独立目录且不能选模。

## 评测接口：直接复用 D−1C

`evaluation.py` 调用原有 `v2xr_protocol_audit.predict.generate` 和 `evaluate.replay`，没有新写 AP 匹配或 GT 选择器。
官方评测 config 直接使用 D−1A 的配置副本，与 D−1B 的 parsed config 逐字段核对；训练 config 的路径差异不会进入评测协议。
每条件每帧只生成一次最终 boxes/scores/None/empty，O/O*/F 共用同一文件与 SHA256。

| 协议 | GT | prediction=None |
|---|---|---|
| O / OFFICIAL_EXACT | 当前条件官方最终动态 GT | 官方整帧跳过 |
| O* / OFFICIAL_GT_NO_SKIP | 与 O 同一 GT | 0预测，GT进入分母 |
| F / OFFICIAL_CLEAN_FIXED_GT | D−1C 已冻结的同帧官方 Clean 目标ID和几何 | 0预测，GT进入分母 |

F 保留 Clean 的 camera/FOV、原始 object selection 和 box/range 条件；Fog/Snow 不再删除这些目标。
固定 GT 每条件6224，官方 GT 为6224/3673/5854；每帧 ID、顺序、几何 identity 必须与冻结 D−1C gt_manifest 完全一致。
新 manifest 的整体 identity 会包含本次 checkpoint/contract，因此另行保留原始 frozen_d1c_gt_identity，不能用新 manifest 偷换 GT。
任一 GT/hash 改变即停止。原 eval_utils_opv2v、BEV IoU、VOC2010、global_sort=False、score/NMS 均不改。
O 用于官方比较，F 是后续天气鲁棒性主结果，O* 用于 None-frame 审计。
报告包含三协议完整 AP30/50/70、分母/None统计、天气gap、GTFilterEffect、NoneSkipEffect、GapMasking和删除目标的一对一检出统计。

`O*−F` 共用相同预测、None处理和 AP 实现，主要隔离官方天气 GT 集改变的影响；GapMasking 是描述性协议差异，不称因果估计。

## 服务器操作

同步整个新目录，保留服务器上已验证的历史包和 D−1A/B/C 结果目录，特别是 D−1C gt_manifest.json。
无需把服务器大型结果复制进 repo。以下命令均在服务器执行；GPU=0 请换成空闲物理 HIP 卡号。
先进入项目并确保日志父目录存在：

```sh
cd /home/cjm/OpenCOOD-main/cjmnet
mkdir -p /data/cjm/datasets/logs
```

提取 smoke，包含 `7zz t` 和一个场景各 CAV 的前2帧，因此也可能较慢：

```sh
nohup env MODE=extract SMOKE=1 \
  TRAIN_ROOT=/data/cjm/datasets/v2xr-official-train-smoke-v2 \
  sh v2xr_train_attfuse/run_server.sh \
  > /data/cjm/datasets/logs/v2xr_train_extract_smoke_v2.log 2>&1 &
echo $!
```

新目录首次完整提取8083帧（提取及验证一起完成；当前 v2 已有失败 request 应使用本文开头 RESUME=1 命令）：

```sh
nohup env MODE=extract sh v2xr_train_attfuse/run_server.sh \
  > /data/cjm/datasets/logs/v2xr_train_extract_full_v2.log 2>&1 &
echo $!
```

提取中断恢复，用同一 TRAIN_ROOT、SMOKE和其余参数：

```sh
nohup env MODE=extract RESUME=1 sh v2xr_train_attfuse/run_server.sh \
  >> /data/cjm/datasets/logs/v2xr_train_extract_full_v2.log 2>&1 &
echo $!
```

独立完整性复核（SHA/解码，可能较长）：

```sh
nohup env MODE=verify sh v2xr_train_attfuse/run_server.sh \
  > /data/cjm/datasets/logs/v2xr_train_verify_v1.log 2>&1 &
echo $!
```

正式训练前，在服务器跑官方 AP 和断点恢复的合成检查，无真实模型训练：

```sh
nohup env MODE=self-test GPU=0 sh v2xr_train_attfuse/run_server.sh \
  > /data/cjm/datasets/logs/v2xr_d1d_self_test_v1.log 2>&1 &
echo $!
```

训练 smoke（要求完整 train 已完成，输出独立目录）：

```sh
nohup env MODE=train GPU=0 SMOKE=1 \
  OUTPUT_ROOT=/data/cjm/datasets/logs/v2xr-clean-attfuse-train-smoke-v1 \
  sh v2xr_train_attfuse/run_server.sh \
  > /data/cjm/datasets/logs/v2xr_clean_train_smoke_v1.log 2>&1 &
echo $!
```

正式随机初始化20轮训练，包含每轮829帧 Clean AP：

```sh
nohup env MODE=train GPU=0 sh v2xr_train_attfuse/run_server.sh \
  > /data/cjm/datasets/logs/v2xr_clean_train_full_v1.log 2>&1 &
echo $!
```

训练中断恢复（仍用原物理 GPU，源码和配置保持相同）：

```sh
nohup env MODE=train GPU=0 RESUME=1 sh v2xr_train_attfuse/run_server.sh \
  >> /data/cjm/datasets/logs/v2xr_clean_train_full_v1.log 2>&1 &
echo $!
```

训练完成后查看或复核选择，不推理天气：

```sh
env MODE=select sh v2xr_train_attfuse/run_server.sh
```

最终选定 checkpoint 的全829帧 validation：Clean/Fog/Snow，O/O*/F一次输出：

```sh
nohup env MODE=final GPU=0 sh v2xr_train_attfuse/run_server.sh \
  > /data/cjm/datasets/logs/v2xr_clean_selected_validation_v1.log 2>&1 &
echo $!
```

可先做最终评测的前12帧 smoke：`MODE=final GPU=0 LIMIT_FRAMES=12`；输出自动分离到 final_evaluation_smoke12，不覆盖全量目录。
恢复最终评测：相同命令加 `RESUME=1`，严格验证全部预测缓存哈希，只推理尚未缓存的帧。
通用 evaluate-checkpoint 接口也会强制先完成 Clean-only 选模，只允许选定权重或已固定的 OPV2V 对照权重进入天气评测。

OPV2V epoch13 的同协议对照，必须在新模型最终全量评测完成以后运行：

```sh
nohup env MODE=compare GPU=0 RESUME=1 sh v2xr_train_attfuse/run_server.sh \
  > /data/cjm/datasets/logs/v2xr_d1d_domain_comparison_v1.log 2>&1 &
echo $!
```

比较前检查 OPV2V 权重哈希及全部 key/shape/dtype，随后官方加载后逐tensor核对。
结构不兼容会写 comparison.json.status=unavailable；不改模型/max_cav，不用随机缺参填补，也不把旧历史 AP 塞入同协议表。
历史 v2xr_attfuse 是不同 GT/FOV/max_cav 设计，不能直接与本次 F 相减解释域偏移。
只有协议匹配结果才能判断域内训练后的天气 gap 是否缩小；同时查看绝对 Clean/天气 AP，避免 Clean 崩溃导致 gap 变小。
这里单次种子结果只描述差异，不自动声称统计显著。

监控：

```sh
tail -n 80 -f /data/cjm/datasets/logs/v2xr_train_extract_full_v2.log
tail -n 80 -f /data/cjm/datasets/logs/v2xr_clean_train_full_v1.log
tail -n 80 -f /data/cjm/datasets/logs/v2xr_clean_selected_validation_v1.log
```

每轮 AP 细节另见训练根目录 clean_ap_epochNN.log；train主日志会输出 `D-1D EPOCH COMMITTED N/20`。
如果看似长时间停在验证阶段，先看对应 Clean AP 日志及输入 hash 进度。

## 最终产物

提取：`/data/cjm/datasets/v2xr-official-train-v2/{data_acceptance.json,plan.json,manifest.json,records/,train/}`。

训练：`/data/cjm/datasets/logs/v2xr-clean-attfuse-train-v1/`：

- config.yaml、selection_policy.json、contract.json、protocol.json：配置、选择预注册、源码/输入/native/环境身份。
- net_epoch1.pth … net_epoch20.pth；resume_states/：完整恢复状态，都在数据盘。
- history.json、官方 loss 文本/TensorBoard、每轮 Clean AP 缓存/日志。
- selected.json、training_summary.json：选定 epoch、checkpoint SHA256、20轮指标及完成链。
- final_evaluation/RESULTS.md、final_evaluation/summary.json：主结果，三条件×三协议。
- final_evaluation/{protocol.json,prediction_manifest.json,gt_manifest.json,per_frame_metrics.jsonl,removed_gt_metrics.jsonl,predictions/}。
- COMPARISON.md、comparison.json、opv2v_matched_evaluation/：完成可比对照时写出。

训练完成不等于最终天气评测完成；分别以 training_summary.json 和 final_evaluation/summary.json 的 complete 状态及哈希为准。
实际结果必须由服务器运行产生，不在本地预填。F 的大幅天气下降在域内训练后仍存在时，支持继续 D0/D1/D2 机制分析；本工具不启动这些实验。

本地只执行 `python -B v2xr_train_attfuse/check_static.py`，检查 AST/LF/源码哈希。
POSIX shell 语法由服务器 launcher 的 `sh -n` 检查；合成数值检查也只在服务器运行。
