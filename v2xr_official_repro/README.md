# D−1A：V2X-R 官方 LiDAR AttFuse Clean validation

本目录实现官方Clean验证的独立运行路径。已经核对官方源码；**服务器实验尚未运行，当前没有复现AP结果**。
仅读取validate，不训练、不实现天气分析，不读取train/test样本。
`PROTOCOL_REVIEW.md`记录实际协议和官方边界行为；`UPSTREAM.json`记录固定commit与SHA256。
本地只允许运行check_static.py；Torch/OpenCOOD、数据准备、合成检查及模型推理全部在服务器。

## 新增文件

- upstream/、upstream_archive.zip：固定版本官方完整源码及原始归档，逐字节保留。
- common.py：源码哈希、锁、原子记录、链接与数据盘runtime隔离。
- prepare.py：核对原始/paired目录，补提取真实相机/雷达，构建官方目录视图。
- run.py：唯一checkpoint发现、官方YAML核对、独立model_dir、运行/编译调度。
- build_preflight.py：服务器C++编译器与PyTorch头文件预检，记录CC/CXX、头文件哈希及错误。
- audit_native_dependencies.py、NATIVE_DEPENDENCIES.json：官方AttFuse的静态导入依赖检查。
- build_official_ops.py：执行原版setup，选择本次所需的原版ROI扩展，排除未使用的PointNet2等扩展。
- observe.py：直接执行官方inference.main，增加完整参数加载检查和报告。
- run_server.sh：POSIX sh启动器，设置HIP GPU及数据盘TMPDIR。
- test_contract.py：仅服务器运行的7项输入边界合成检查。
- check_static.py、source_snapshot.json、frozen_baseline.json：本地静态/归档/冻结文件完整性检查。
- README.md、PROTOCOL_REVIEW.md、UPSTREAM.json：交付与溯源。

用户提供的本地V2XR_attfuse.yaml已与官方模板逐字节核对一致，哈希为
`7e1ce936d779138b5b1ca6462ebc79d9e604920d9a29b7ef1c3a64e50d96e346`。
记录见USER_CONFIG_REVIEW.json。服务器仍检查原始文件和运行副本的哈希。

已有v2xr_subset/、v2xr_attfuse/、reference_weather/、attfuse_code/和GSPR不修改。
旧实验是OPV2V权重+本项目fixed-GT迁移评测；本次使用用户放置的V2X-R权重、
官方fork/config/dataset/model/postprocess及实际eval_utils_opv2v，不混入旧fixed-GT检查。

## 同步与目录

把整个v2xr_official_repro/同步到服务器项目根目录，必须包括upstream/和所有哈希文件。
无需服务器Git、branch、commit检查，也无需联网拉源码。不要用文本模式转换upstream内容。

| 用途 | 默认路径 |
|---|---|
| 用户原始数据/模型文件 | /data/cjm/datasets/V2X-R |
| 已提取paired输入 | /data/cjm/datasets/v2xr-lidar-weather-validate-v1 |
| 新官方视图 | /data/cjm/datasets/v2xr-official-validate-v1/validate |
| 补提取文件 | /data/cjm/datasets/v2xr-official-validate-v1/auxiliary |
| 官方源码runtime/扩展编译 | /data/cjm/datasets/v2xr-official-runtime-v1 |
| 独立model_dir | /data/cjm/datasets/logs/v2xr-official-attfuse-model-v1 |
| 完整结果 | /data/cjm/datasets/logs/v2xr-official-attfuse-clean-repro-v1 |
| TMPDIR | /data/cjm/datasets/tmp/v2xr_official_repro |

先检查原validate是否已有官方逐agent布局；否则使用paired Clean/YAML作为链接来源。
官方loader无条件读取雷达/4张图片，需要从20个原始validate车辆包选择性补提取缺失文件。
只提取Clean推理依赖，不提取天气PCD。每个包CRC/size/SHA256检查、记录后可续跑；
Clean点云通过链接复用，不重复复制约9GB。原始数据和paired目录都保持原样。
view核对6场景、20个scene-agent、829协作帧、3041单agent帧及全部时间戳顺序；
运行时再次与官方loader实际scenario_database逐项比对。真实参与协作agent还受官方70m范围筛选。

原始根目录只自动查找顶层唯一.pth/.pt/.ckpt及V2XR_attfuse.yaml，不假设checkpoint名称。
多候选时停止，日志列出文件名。可在下列命令加
`CHECKPOINT='/实际权重路径' YAML='/实际yaml路径'`明确指定，不按mtime/文件名猜。
model_dir保留路径适配后的config.yaml，建立0.pth与net_epoch0.pth链接，记录原始/实际YAML及权重哈希。

## 首次准备官方扩展

新建的独立runtime中没有编译扩展。第一次先执行下面的官方编译任务；确认日志成功后再启动smoke。
该步骤不会安装/覆盖共享OpenCOOD。如果官方旧CUDA源码无法在当前HIP工具链编译，
停在此处回传编译日志，再处理环境。

```sh
cd /home/cjm/OpenCOOD-main/cjmnet
conda activate opencood
mkdir -p /data/cjm/datasets/logs
nohup env STAGE=build GPU=0 \
  OUTPUT_ROOT=/data/cjm/datasets/logs/v2xr-official-build-v1 \
  sh v2xr_official_repro/run_server.sh \
  > /data/cjm/datasets/logs/v2xr_official_build_v1.log 2>&1 &
echo $!
```

```sh
tail -n 80 -f /data/cjm/datasets/logs/v2xr_official_build_v1.log
```

## Smoke（前12帧，后台）

GPU=0只是示例，替换成空闲物理HIP卡。只设置ROCR_VISIBLE_DEVICES，清除另两个GPU变量。
内部设备cuda:0。入口先运行服务器合成边界检查，然后检查官方imports与权重完整加载，
通过后准备全部验证集文件视图，最后仅推理前12帧。首次补解压可能耗时；smoke AP不参与829帧复现判断。

```sh
cd /home/cjm/OpenCOOD-main/cjmnet
conda activate opencood
mkdir -p /data/cjm/datasets/logs
nohup env GPU=0 LIMIT_FRAMES=12 \
  OUTPUT_ROOT=/data/cjm/datasets/logs/v2xr-official-attfuse-clean-smoke-v1 \
  sh v2xr_official_repro/run_server.sh \
  > /data/cjm/datasets/logs/v2xr_official_clean_smoke_v1.log 2>&1 &
echo $!
```

```sh
tail -n 80 -f /data/cjm/datasets/logs/v2xr_official_clean_smoke_v1.log
```

smoke必须完成12帧且checkpoint_all_tensors_loaded_exactly=True；frame/order、dataset/model/eval
必须与protocol一致。需要原始依赖文件的准备已在全validate完成，不代表全829帧推理已经完成。

## 全829帧Clean（smoke完成后，由用户单独启动）

```sh
cd /home/cjm/OpenCOOD-main/cjmnet
conda activate opencood
nohup env GPU=0 sh v2xr_official_repro/run_server.sh \
  > /data/cjm/datasets/logs/v2xr_official_clean_full_v1.log 2>&1 &
echo $!
```

```sh
tail -n 80 -f /data/cjm/datasets/logs/v2xr_official_clean_full_v1.log
cat /data/cjm/datasets/logs/v2xr-official-attfuse-clean-repro-v1/RESULTS.md
```

## 依赖不兼容/缺少官方扩展时

probe在补解压前检查依赖；失败会保留official_failure.json和完整stderr。
该fork有自己的原生扩展，拒绝复用当前项目的同名.so，也不会改GT算子或改为Python近似实现。
若缺少扩展、且当前HIP开发工具链完整，可显式编译到独立runtime：

```sh
nohup env STAGE=build GPU=0 \
  OUTPUT_ROOT=/data/cjm/datasets/logs/v2xr-official-build-v1 \
  sh v2xr_official_repro/run_server.sh \
  > /data/cjm/datasets/logs/v2xr_official_build_v1.log 2>&1 &
echo $!
```

build运行固定官方utils/setup.py构建box_overlaps，再执行原版pcdet_utils/setup.py，
在setuptools.setup入口选择roiaware_pool3d扩展，不安装/覆盖共享OpenCOOD。
官方CUDAExtension对象、源码、编译参数、BuildExtension均保留；不编译本次不使用的
PointNet2和pcdet iou3d_nms扩展。选择范围和实际HIP转换源码哈希写入runtime/native_build_selection.json。
上游安装指南采用旧CUDA环境，当前HIP能否成功编译需服务器确认。本地没有执行编译或模型实验。
如果是Python依赖缺失、ABI或编译失败，回传日志后处理环境；不自动pip安装/降级。
官方requirements有旧版本固定，**不要直接将全部旧requirements安装到当前共享opencood环境**。
确认环境能导入官方代码后，使用相同参数重启失败的smoke，增加RESUME=1。

只准备数据/模型（不推理、不要求GPU）可用STAGE=prepare和单独OUTPUT_ROOT，仍是后台任务。

```sh
nohup env STAGE=prepare \
  OUTPUT_ROOT=/data/cjm/datasets/logs/v2xr-official-prepare-v1 \
  sh v2xr_official_repro/run_server.sh \
  > /data/cjm/datasets/logs/v2xr_official_prepare_v1.log 2>&1 &
echo $!
```

## 重启与参数

2026-10-16服务器build反馈：gflags补齐后，继续发现glog头文件缺失和默认GCC<9。
新入口在正式编译前检查实际CXX版本，并对PyTorch的C++17/Flags/Logging头文件做语法编译检查。
结果为build_environment.json；该检查只覆盖host编译，不代表HIP kernel或全部ABI已经通过。
仅修复编译环境，不修改upstream源码、检测逻辑或评测配置。

在已建立的独立编译依赖目录中补齐glog和GCC/G++11：

```sh
nohup /home/cjm/miniconda3/bin/conda install -y \
  -p /data/cjm/tools/v2xr-build-deps-v1 --override-channels -c conda-forge \
  gflags=2.2.2 glog=0.6.0 gcc_linux-64=11 gxx_linux-64=11 \
  > /data/cjm/datasets/logs/v2xr_build_deps_v2.log 2>&1 &
echo $!
```

安装成功并同步新版整个实验目录后，用原opencood Python、显式的新CC/CXX重建：

```sh
nohup env STAGE=build GPU=0 \
  CC=/data/cjm/tools/v2xr-build-deps-v1/bin/x86_64-conda-linux-gnu-gcc \
  CXX=/data/cjm/tools/v2xr-build-deps-v1/bin/x86_64-conda-linux-gnu-g++ \
  CPATH="/data/cjm/tools/v2xr-build-deps-v1/include${CPATH:+:$CPATH}" \
  OUTPUT_ROOT=/data/cjm/datasets/logs/v2xr-official-build-v3 \
  sh v2xr_official_repro/run_server.sh \
  > /data/cjm/datasets/logs/v2xr_official_build_v3.log 2>&1 &
echo $!
```

保留build-v1/v2失败记录，不改系统gcc软链接，不切换opencood Python，不升级Python/Torch。
Conda编译器包是否能完成求解以及HIP与该host工具链是否兼容，仍由服务器结果确认。

2026-10-16 build-v3新反馈：host预检成功，GCC11.4、gflags/glog头文件均已通过；
官方roiaware扩展完成链接后，整套setup在PointNet2的THH/THH.h缺失处失败。
对固定官方源码进行保守AST导入依赖检查，覆盖63个Python模块；当前AttFuse链只到达
box_overlaps及roiaware_pool3d两个原生扩展，没有到达PointNet2/pcdet iou3d_nms。
动态模型分发显式纳入point_pillar_intermediate，实际服务器导入也继续检查，不能把静态检查当运行成功。
因此只缩小构建扩展列表，不移植PointNet2、不补假THH头文件、不改官方GT/NMS/AP。
上游328个文件保持原样。同步新版整个目录后，保留现有依赖与缓存，使用新build-v4输出：

```sh
nohup env STAGE=build GPU=0 \
  CC=/data/cjm/tools/v2xr-build-deps-v1/bin/x86_64-conda-linux-gnu-gcc \
  CXX=/data/cjm/tools/v2xr-build-deps-v1/bin/x86_64-conda-linux-gnu-g++ \
  CPATH="/data/cjm/tools/v2xr-build-deps-v1/include${CPATH:+:$CPATH}" \
  OUTPUT_ROOT=/data/cjm/datasets/logs/v2xr-official-build-v4 \
  sh v2xr_official_repro/run_server.sh \
  > /data/cjm/datasets/logs/v2xr_official_build_v4.log 2>&1 &
echo $!
```

build-v4完成后再按上文运行Clean smoke。无需重装依赖或清空runtime。

支持DATA_ROOT、PAIRED_ROOT、VIEW_ROOT、RUNTIME_ROOT、MODEL_DIR、OUTPUT_ROOT、CHECKPOINT、YAML、
SEVEN_ZIP、PYTHON、LOADER_WORKERS、LIMIT_FRAMES、RESUME。默认worker16沿用官方；
显式调整会写入协议，不改变GT/AP逻辑。固定seed/PYTHONHASHSEED=20261016；OMP/BLAS/MKL默认1线程。

中断后保留数据视图和auxiliary，在原命令加RESUME=1，保持其它参数相同。
数据准备按archive记录续跑；Clean推理没有逐帧恢复，未完成推理从头重跑并覆盖该次未完成frames.jsonl。
失败记录保存在attempts/。已完成结果在参数/源码一致时直接返回。
源码、参数或checkpoint改变必须新建OUTPUT_ROOT；模型改变还需新MODEL_DIR。
锁禁止并发写同一模型/runtime/view。不要删除已完成数据或手改source_snapshot。

## 输出与判断

- RESULTS.md：AP30/50/70、绝对pp差、复现提示、GT/预测/跳帧数量。
- summary.json：完整结果与计数、checkpoint/config哈希、实际类和是否完整829帧。
- protocol.json：实际解析配置、关键阈值/范围、源码逐文件SHA256、导入模块/扩展哈希与路径、版本。
- preflight.json：官方依赖/完整权重加载检查。
- data_verification.json：全部视图文件、SHA256、源路径、场景/CAV/时间戳清单。
- frames.jsonl：逐帧GT/预测数、参与agent数与官方跳帧状态。
- official_result_stat.json：调用官方最终AP之前的原始TP/FP/score统计。
- official_eval.yaml：官方eval_final_results输出的原文件副本。
- contract.json：本实验精确输入与源码身份。
- last_failure.json、official_failure.json、attempts/：失败信息；完成后以summary.status为准。

GT_total统计所有读取帧；official_metric_GT_total按官方跳过None预测帧行为累计。
复现参考为AP30=.8669、AP50=.8258、AP70=.6656。
完整829帧AP70绝对差>2pp显著标记 **OFFICIAL REPRODUCTION NOT YET ALIGNED**。
≤2pp只报告参考距离，不写成统计PASS；smoke只标记SMOKE ONLY。

成功：核对829帧、类/阈值/哈希、AP差值和跳帧计数，保留日志/JSON，提交人工判断。
失败或偏差>2pp：停止，检查权重/YAML、数据顺序、依赖版本、源码差异和跳帧统计。
所有情况都停在D−1A；本工具没有D−1B/D−1C/D0/D1/D2或训练入口。
