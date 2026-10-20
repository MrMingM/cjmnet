# V2X-R L0/L1/L2 预注册协议与源码核对

状态：代码交付，真实模型自检、smoke、训练和结果待服务器运行。本文件没有新增实验分数。

## 原始证据

- 已阅读根目录 AI_CONTEXT.md，重点 §44–§49；旧 Clean-only AttFuse 和 GSPR 的训练/恢复/选模/评测实现及已有结果继续冻结。
- 官方版本：`f3ebbad95561444a53a17358735b0f0dc4ba89f7`。服务器用源码 SHA256 检查版本，不需要 `.git`。
- [论文 §5.4](https://arxiv.org/html/2411.08402v4#S5.SS4) 描述从 Normal/Fog 点云随机采样进行训练，没有规定概率或明确协作帧内所有 CAV 必须同天气。
- 对应源码：`v2xr_official_repro/upstream/opencood/data_utils/datasets/opv2v/basedataset.py` 的 `BaseDataset.__init__`、`retrieve_base_data`；数据工厂将 `IntermediateFusionDatasetV2XR` 映射到 `opv2v/intermediate_fusion_dataset.py`。
- 模型配置：`opencood/hypes_yaml/V2X-R/LiDAR/V2XR_attfuse.yaml`，Clean/Fog/Snow 的真实后缀分别是 `.pcd`、`_fog_0.060.pcd`、`_snow_2.5_2.0.pcd`。服务器核对已接受提取 manifest 中的全部配对，绝不以 README 示例猜测名称。

## 作者代码与“每帧各半”的区别

实际 `retrieve_base_data` 在遍历 CAV 的循环内执行：

```python
if self.train_sim and self.train:
    aug = random.randint(0, 6)
    if aug <= 2:
        lidar_file = lidar_file.replace('.pcd', self.sim_weather + '.pcd')
```

因此按作者实现，单次 CAV 读取选择 Fog 的概率是 3/7，Clean 是 4/7；同一协作帧可以出现 Clean 和 Fog 两种输入。概率写在代码里，配置只启用该分支及指定后缀。本实验直接执行冻结的原方法，适配层只观察实际读取和 GT，不接管天气抽样。

`train=False` 时该抽样分支关闭。`eval_sim=True` 且 `train=False` 时，在索引构建阶段把点云路径换成指定天气；本轮每轮选模和 validation loss 都读取 Clean。最终天气评测采用冻结评测器已有的 `eval_sim` 处理。

训练协作帧集合保持 8083；官方 batch8/drop_last 每轮请求 8080 个索引，3 个索引当轮丢弃。每轮并未扩成 16166 帧。原官方训练遇到没有有效 GT 的帧会通过 NumPy 再抽训练索引，这也会再次抽天气，因而实际读取次数可能超过初始索引对应的 CAV 数。日志区分请求索引、重抽样尝试、实际读取和最终 GT。

官方使用当前点云及 camera FOV、native points-in-boxes 过滤 GT，因此 Fog 可以影响训练目标集合。适配器记录每次 native 过滤的框内点数、零点框及最终 GT，保持原始 GT 逻辑。原版空体素的随机临时点和雷达/激光极少点辅助路径也保留；文件缺失必须报错，禁止缺失 Fog 时回退 Clean。原代码即使 LiDAR-only 也读取雷达/相机，故验收数据中的这些文件仍是必需输入。

## 固定训练与选模

C/D 都随机初始化，seed=20261017，20 epoch、batch8、Adam lr=.002、eps=1e-10、weight_decay=1e-4、MultiStep[10,15]/gamma=.1。复用官方数据集、增强、collate、模型、检测损失、优化器及调度器工厂。训练循环逐项保留官方单 GPU 的操作顺序，额外加入读取审计、完整状态提交和每轮独立进程 Clean AP；不直接调用旧 Clean-only 包的拒绝天气训练分支。

保持官方训练和 validation loss 的 16 workers、shuffle=True、pin_memory=True、drop_last=True、prefetch_factor=4；每轮最终执行 `scheduler.step(epoch)`，参数从 0 开始。AP 仍使用原推理器 batch1/16 workers，完整 829 Clean 帧。

增强调用也按真实源码核对：该版本 `IntermediateFusionDataset` 构造了 DataAugmentor，YAML 配置了 world flip/rotation/scaling，但其 `__getitem__`、`get_item_single_car`、collate 路径没有调用 `augment` 或 DataAugmentor.forward。C/D 保留作者这条实际路径，不自行增加配置中未执行的变换。每个样本和 epoch 都记录实际 world augmentation 调用数，服务器等价测试要求与冻结原版一致。点顺序 shuffle、天气采样、空体素临时点及空 GT 重抽样等原有随机操作仍存在。

每轮保存权重后计算完整 Clean O AP70。所有 20 个 epoch 提交后，选择最高 O AP70，精确并列取最早轮。Fog/Snow 只在选择锁定后评测。FiLM 同规则在固定 3 轮中选模。Smoke 的 12 帧 AP 没有正式选模资格。

D 直接按哈希加载 `v2xr_gspr_clean/integration.py` 的 `PointPillarGsprOfficial` 和 `GsprOfficialLoss`。输入 XYZI，官方 VFE 装饰后第一层 PFN 前连续加权，点槽位/点数/GT 不变；有效点掩码和梯度完全沿用旧实现。总损失为官方检测损失加 `.02*(mean(r)-.9)^2 + .0001*mean(evidence^2)`。没有逐点天气监督。

C/D 的共享初始化及额外 GSPR 初始化不推进共享 CPU RNG 的性质由服务器合成测试验证。多 worker 非 persistent，epoch 结束后没有需要保留的工作进程；保存主进程 RNG 后，可重建相同 worker seed。未完成 epoch 重放，已提交 epoch 不重训。日志逐 epoch 比较真实读序列、worker RNG、增强后输入及 GT 的哈希；只有完全一致时才记录该层面的配对。不保证 GPU 内核逐位确定性。

## BN 与 FiLM

BN：对 C/D 各自独立重置全部真实 tracked BatchNorm 层的 running mean=0、variance=1、counter=0，momentum=None；累计平均每次 forward 的 batch 均值和无偏 batch 方差。该策略按 forward 等权，PFN 中不同点数的 batch 也等权，不能描述为全数据池化方差。真实层来自 `isinstance(_BatchNorm)`，包含 PFN 的 BatchNorm1d 和 BEV backbone 中的 BatchNorm2d；以服务器实际清单为准。

仅这些 BN 层处于 train 模式，其余层 eval，所有参数冻结，no_grad，无优化器。按官方 inference 数据集配置，固定顺序遍历 train 根目录全部 8083 Fog 帧、batch1，无增强、固定种子，使用官方 test 预处理上限 max_voxel_test=70000。这里的 inference/test 预处理参数名称不意味着读取正式 test。保留官方 shuffle_points 等内部操作，但其随机序列固定。保存 buffer 差值、所有其他 state 精确相等检查及同一探针输入的校准前后输出差值。

FiLM：两组独立适配，只加载对应 C/D 原权重。Scatter 后、AttBEVBackbone 前，要求实际特征 `[N,64,H,W]`，N 为参与的 CAV 数。每个 CAV 当前特征的每通道空间均值及总体标准差进入 `128 -> 32 -> 128` MLP；最后一层权重和偏置置零，`gamma=1+delta_gamma`，初始精确恒等。没有天气标签、框或 validation 信息作为输入。

FiLM 固定 3 epoch，Adam lr=.001、eps=1e-10、weight_decay=1e-4，恒定 lr，无额外损失正则，只有官方检测损失。冻结全部原模型参数和 buffer，主干始终 eval，只训练新增 8352 个 FiLM 参数。D_FiLM 不加入被冻结 GSPR 的旧正则；原 GSPR 参数和 BN buffer 必须逐项等于 D。FiLM 从 C/D 原模型开始，与 BN-only 相互独立。训练及推理记录 gamma/beta 统计；若一直为恒等或常量，要如实报告，不能写成模块已改善检测。

## 冻结评测与自动门槛

直接复用 `v2xr_train_attfuse/evaluation.py:evaluate`，底层为 `v2xr_protocol_audit/predict.py:generate` 和 `evaluate.py:replay`。实际执行冻结的原版 inference.main，只替换显式模型工厂；严格验证 checkpoint keys/shape/dtype 和全部加载值。三协议共用预测缓存，O 统计与同次原版输出完全一致，GT 与冻结审计逐帧一致。无新 AP/GT/NMS/匹配实现。

主指标 O AP70；所有结果同时保留 O/O*/F、AP30/50/70、GT/None/removed/retained 分母。L0 Fog 相对旧 Clean-only `.4421260304638024` 至少 +5 pp，否则退出码20、写 STOP 和诊断/部分报告，不启动 L1/L2。该负结果不自动判定为程序错误。

L1 比 D-C：Fog 至少 +1.5 pp，三条件至少两项非负，Clean 至少 -1 pp。技术完整有效但性能门槛失败仍继续四个 L2 适配；技术异常统一失败退出。所有门槛已在 settings.json 固定，Decimal 处理精确边界。

## 恢复安全点

每轮依次：训练与读审计 → 保存独立权重 → validation loss → Clean AP → 原调度器 step → 保存完整 model/optimizer/scheduler/criterion/累积日志/Python/NumPy/Torch CPU+CUDA RNG → 最后发布 epoch complete.json。

每轮 AP 使用按权重内容哈希命名的 immutable 硬链接，避免未提交 epoch 重跑时覆盖已有缓存的输入。评测可继续已完成并且哈希匹配的逐帧缓存。BN 未完成则独立重新校准。初始随机状态也有原子提交标记；只有权重没有状态的文件不能恢复。

总流程用文件锁防止重复启动，子工作进程另有独立锁。TERM/INT 时终止该阶段进程组；SIGKILL 无法被捕获，若旧工作进程仍活着，新工作进程会被锁阻止并报技术失败，避免并发写入。服务器重启后文件锁自动释放。

阶段完成要核验 request/源码/设置/上游依赖/全部登记产物 SHA256。恢复时重新核对 preflight 登记的实际外部输入字节。源码、配置、环境版本、物理 GPU 或输入身份变动会明确失败。新版本请换独立 OUTPUT_ROOT；不要手改 snapshot、request 或 complete 标记。
