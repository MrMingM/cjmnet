# V2X-R Clean：随机初始化训练GSPR-AttFuse，按官方O评测

## v2：修复官方损失选择，旧失败目录保留

首次服务器尝试已完成输入检查/数据集构建和模型创建，但在create_loss处报Unexpected official detection loss，尚未创建optimizer或进行训练更新。
原因是新适配器误用了point_pillar_loss名称和父类；官方YAML实际为point_pillar_loss_opencood / PointPillarLossOpenCOOD。
修复同时更新工厂检查、GSPR损失父类/forward接口、官方运行平均日志与可恢复正则累积状态，不修改官方YAML或旧基线损失。
服务器自检现在通过真正的train_utils.create_loss入口，并检查criterion严格重载；本地静态检查核对官方YAML名称、工厂常量、父类import与settings一致。
INITIAL_LOSS_FACTORY_FAILURE.json记录旧源码快照/适配器哈希及失败阶段。
请整目录同步修复后的新包，旧v1失败目录不删不改；下面所有新任务使用v2输出和日志，RESUME=0首次启动。
不要对旧v1加RESUME=1强行跨源码版本恢复，也不要手改request/contract/snapshot。原提取数据继续复用，无需重新解压。

用户已明确选择：只用V2X-R Clean训练、随机初始化、20轮、只按Clean选择权重。
本次不加载OPV2V GSPR、作者checkpoint或已有AttFuse epoch20来初始化。
已有epoch20仅作为完成后同协议比较的固定参照。
继承的协议审计仍核对作者权重的文件哈希作为历史证据，不将其参数加载到新训练模型。

## 模型和协议

复用冻结的`attfuse_gspr/reliability.py`，不修改其源码，不直接搬用旧OPV2V版GSPR模型或骨干。
新`integration.PointPillarGsprOfficial`继承V2X-R官方PointPillarIntermediate。
在官方VFE进行点特征装饰和padding mask之后、第一层PFN之前乘GSPR连续可靠性；
原始体素、点槽位、坐标、点数不变，官方PFN/scatter/AttBEVBackbone/检测头保持原实现。
GSPR权重不进入dataset，不参与GT过滤，不删除点或目标。
新增GSPR模块在官方骨干初始化之后建立，并恢复额外初始化前的CPU RNG，以保持共享参数初始化及后续全局随机状态对齐。

本次是**检测损失驱动的GSPR训练**，没有天气噪声逐点标签，不属于旧天气多任务监督训练。
总损失=官方检测损失 + 0.02×可靠性均值约束 + 0.0001×evidence平方均值；目标均值0.9。
检测损失父类准确对应官方配置：opencood.loss.point_pillar_loss_opencood.PointPillarLossOpenCOOD，保留(epoch=1, train=True)调用接口。
这两个正则项沿用旧GSPR检测训练设置；不能声称预测可靠性/不确定性已经得到天气标签校准。
设置保存在settings.json和训练config.yaml的gspr_experiment字段，身份受SHA256约束。

仅改模型工厂与损失工厂，明确记录actual_model_class及gspr_adapter；不改官方源文件。
训练config保留官方model.core_method名称以复用原配置解析器，实际实例通过显式工厂生成GSPR子类。
评测继承被冻结的D−1C推理入口，实际执行原版inference.main()，仅模型被上述子类替换。
官方输入处理/GT/后处理/AP保留：IntermediateFusionDatasetV2XR、fov=true、proj_first=true、
score>0.20、NMS=.15、COM_RANGE=70、max_cav=7、BEV IoU、VOC2010、global_sort=False。
**主协议O**按当前天气点云过滤GT，prediction=None时整帧跳过；O*/F仅留作审计。
每条件的O replay必须与相同GSPR预测下原版官方inference统计完全相同。

上游：https://github.com/ylwhxht/V2X-R ，commit `f3ebbad95561444a53a17358735b0f0dc4ba89f7`。
所有历史源码按哈希复用，服务器不依赖Git；不要修改旧snapshot或已有结果来绕过检查。

## 数据、训练与选模

- 原始train包：`/data/cjm/datasets/V2X-R/train`，只用于继承完整性校验。
- 已提取train：`/data/cjm/datasets/v2xr-official-train-v2/train`，8083协作帧，32有数据场景。
- 公开8084与实际8083的接受契约仍读取原D−1D的TRAIN_DATA_ACCEPTANCE.json，不创建新授权或补造帧。
- 只选Clean `<timestamp>.pcd`；已提取Fog/Snow不用于训练、loss验证或选epoch。
- 每轮Clean loss保留官方shuffle/drop_last行为，完整Clean AP另对829帧推理。
- batch8、Adam lr=.002/eps1e-10/weight_decay1e-4、20轮、MultiStep10/15 gamma.1、16 workers、官方增强和scheduler.step(epoch)顺序保持不变。
- 训练seed20261017，推理seed/PYTHONHASHSEED20261016。
- 20轮中Clean **O AP70**最大者入选，精确并列选最早epoch；天气不参与选择。
- 纯AttFuse原选模使用F。已核对其20轮history，按O排序也选epoch20；比较时程序再次确认，无需为改变选模名字重训基线。
- 最终评测Clean/Fog/Snow各829帧；不提取或读取正式test，不包含Rain。
- 此处“测试”指当前已准备的validation基准评测。

完整epoch提交、模型/优化器/调度器/RNG恢复沿用原D−1D实现。
中断从最近完整提交轮次重跑，不能仅加载.pth假装完整恢复。
源码、数据、GPU及配置身份变化拒绝恢复；不自动减小batch或换结构解决显存不足。
每轮Clean AP在独立进程中运行，不消耗父训练进程随机状态；显存须能容纳父训练状态与子进程推理。
共享训练引擎的部分日志仍带D-1D前缀；新输出路径、contract.stage=V2XR_GSPR_CLEAN_O、actual_model_class和settings标明本实验身份。

## 同步与环境

只新增整个`v2xr_gspr_clean/`目录。服务器保留现有成功的`v2xr_train_attfuse/`、`v2xr_protocol_audit/`、`v2xr_official_repro/`、
官方runtime及D−1A/B/C和纯AttFuse结果。
GSPR原文件attfuse_gspr/reliability.py、pillar_vfe.py、loss.py和根目录gspr_attfuse_config.yaml也须与本地冻结哈希相同；
前者为实际网络依赖，后几项用于记录原插入位置/正则配置来源，不导入OPV2V骨干。
解释器默认`/home/cjm/miniconda3/envs/opencood/bin/python`，环境无需重新安装GSPR专属包。
TMPDIR默认`/data/cjm/datasets/tmp/v2xr_gspr_clean`；POSIX sh、LF，无pipefail或[[。
仅ROCR_VISIBLE_DEVICES选择物理GPU，unset HIP_VISIBLE_DEVICES、CUDA_VISIBLE_DEVICES；内部cuda:0。
训练根OUTPUT_ROOT默认`/data/cjm/datasets/logs/v2xr-gspr-clean-train-v2`，基线BASELINE_ROOT默认`/data/cjm/datasets/logs/v2xr-clean-attfuse-train-v1`。
其余历史数据/模型路径默认与已成功D−1D一致，可通过launcher所列环境变量指定。

## 先在服务器做模型自检和训练smoke

GPU=0换为空闲物理卡号。以下串行运行，自检失败则不会开始smoke：

```sh
cd /home/cjm/OpenCOOD-main/cjmnet
mkdir -p /data/cjm/datasets/logs
nohup sh -c '
  env MODE=self-test GPU=0 sh v2xr_gspr_clean/run_server.sh &&
  env MODE=train GPU=0 SMOKE=1 \
    OUTPUT_ROOT=/data/cjm/datasets/logs/v2xr-gspr-clean-train-smoke-v2 \
    sh v2xr_gspr_clean/run_server.sh
' > /data/cjm/datasets/logs/v2xr_gspr_clean_smoke_v2.log 2>&1 &
echo $!
```

自检仅合成体素，检查共享权重初始化/RNG一致、单位权重时官方输出一致、检测损失到GSPR的梯度、padding权重、严格权重和optimizer重载。
自检文件在正式OUTPUT_ROOT后加`-self-test`的独立目录。
smoke继承2个训练batch、2个验证batch、12帧Clean AP，只用于运行检查，不作性能/选模结论。
smoke目录不能恢复到正式20轮。

```sh
tail -n 80 -f /data/cjm/datasets/logs/v2xr_gspr_clean_smoke_v2.log
```

## 完整训练后自动进行最终天气评测

自检和smoke通过后：

```sh
cd /home/cjm/OpenCOOD-main/cjmnet
nohup env MODE=all GPU=0 SMOKE=0 RESUME=0 \
  sh v2xr_gspr_clean/run_server.sh \
  > /data/cjm/datasets/logs/v2xr_gspr_clean_full_v2.log 2>&1 &
echo $!
```

MODE=all以两个独立进程先train再final；训练失败不会进入天气评测。
最终推理前重新验证20轮Clean选模并锁定权重；之后生成O/O*/F，RESULTS.md首先展示O，
并自动与固定纯AttFuse基线生成COMPARISON_O.md。
没有使用天气结果挑epoch，不自动追加训练轮次。

```sh
tail -n 80 -f /data/cjm/datasets/logs/v2xr_gspr_clean_full_v2.log
```

中断恢复，保持同GPU、源码、配置与输出目录：

```sh
nohup env MODE=all GPU=0 SMOKE=0 RESUME=1 \
  sh v2xr_gspr_clean/run_server.sh \
  >> /data/cjm/datasets/logs/v2xr_gspr_clean_full_v2.log 2>&1 &
echo $!
```

需要分开运行可用MODE=train、MODE=final；只查看锁定选择用MODE=select。
最终评测12帧检查用MODE=final LIMIT_FRAMES=12，输出独立final_evaluation_smoke12，不作全量比较。
每轮Clean AP细节在OUTPUT_ROOT/clean_ap_epochNN.log。

## 查看结果

新训练根目录下：

- selection_policy.json、config.yaml、contract.json：预定设置、显式模型适配与源/数据/环境身份。
- history.json、net_epochN.pth、resume_states/、epochs/NN/complete.json：逐轮与完整恢复记录。
- selected.json、training_summary.json：20轮完成与只按Clean O选出的权重/hash。
- final_evaluation/RESULTS.md、summary.json：完整天气评测，O为主，保留O*/F审计。
- COMPARISON_O.md、comparison_O.json：新模型与纯AttFuse的三天气AP30/50/70及差值。

固定纯AttFuse官方O AP70：Clean .7053976613825419、Fog .4421260304638024、Snow .2889221792378152。
最终比较要求冻结GT身份相同，基线文件哈希相同；不同模型可有不同None帧，这是O协议本身的行为。
任何GSPR提升/下降均由真实服务器结果填写；不预设收益，不将单种子validation差异称为统计显著。

本地仅运行`python -B v2xr_gspr_clean/check_static.py`做AST/LF/JSON/源码哈希检查；
不在本地导入或执行训练、模型、服务器自检与真实数据。POSIX语法由服务器入口sh -n自检。
