# E0 实现前的源码与记录核对

本文件记录开发时的代码事实，不是 E0 实验结论。服务器仍会重新校验实际文件、权重和完整缓存。

已核对 `AI_CONTEXT.md`，重点 §6–7、§44–50；指定的 GSPR、V2X-R 集成、天气实验的协议/模型/训练/评测，以及 `logs/train/FINAL_RESULTS.json`。官方执行路径来自被冻结的 `v2xr_official_repro/upstream/`，重点包括 `point_pillar_intermediate.py`、`common_modules/pillar_vfe.py`、`att_bev_backbone.py`、`self_attn.py`、`opv2v/basedataset.py`、`intermediate_fusion_dataset.py`、`sp_voxel_preprocessor.py`、`pcd_utils.py`、`base_postprocessor.py`、`voxel_postprocessor.py`、`inference.py` 和 `eval_utils_opv2v.py`。

源码身份由 `PINS.json`、历史包各自的 snapshot 和官方运行时的既有 provenance 校验。没有运行 Git 检查的前提，不修改任何旧包或哈希。

## 实际执行机制

1. 当前 D 通过 `v2xr_gspr_clean/integration.py:PointPillarGsprOfficial` 继承官方模型，调用原始 GSPR 类；在官方 VFE 完成装饰和 padding mask 后、第一层 PFN 前乘 r。`attfuse_gspr/pillar_vfe.py` 是已冻结的另一条接口，不是这次 V2X-R 执行类。
2. GSPR 读取原始 XYZI 体素槽位；几何、反射强度、物理证据先相加，再统一 softplus 得到两类证据。`*_support` 是分支 logits 各自的正变换，它们的和不等于最终 fused evidence。
3. pillar r/u 由点均值产生，输出字典保留五个主要变量；官方 AttFusion 只接收 BEV 特征和 `record_len`，质量量没有显式作为额外 attention 输入。它们仍通过加权特征间接影响后续网络。
4. 当前损失仅为官方检测损失加固定 keep/evidence 正则，没有逐点天气标签；源码中“calibrated”的 docstring 不能证明当前 V2X-R 权重经过物理概率校准。
5. 原天气混合训练每辆CAV独立抽 `random.randint(0,6)<=2`，Fog概率3/7；当前D使用旧冻结集成的相同模型/正则。这里不改变训练配置，也不训练。
6. 官方GT依赖当前点云、相机FOV和原生框内点数过滤；O跳过None整帧，O*计入None，F固定Clean目标；AP是原版BEV多边形/VOC2010、逐帧分数排序。新增代码不另写 AP。
7. 数据处理按传感器范围、车身遮挡、投影、FOV、原生体素化的实际顺序执行。空体素路径可能补随机临时点；因此记录真实post-FOV/pre-voxel点与原生体素占用两个范围，并单列人工补点。不能把预体素点云覆盖混称为实际PFN槽位覆盖。

## 额外发现：作者绿色通道筛选掩码

官方 `pcd_utils.pcd_to_np(..., get_weather=True)` 返回RGB绿色通道；`BaseDataset.retrieve_base_data` 随后用 `label==0` 构造 `de_lidar_np`。当前 C/D 的 `use_DeLidar=False`，该分支不进入其检测和 GSPR 点级损失。

这纠正了“没有逐点监督就一定没有任何逐点标记线索”的潜在误读。已有代码证明筛选掩码的使用方式，尚不能单独证明每个绿色值的模拟器生成语义。E0 会观察所有实际读取的绿色通道，保存原始掩码与点坐标，另行报告其与 r/u 的关系；不同CAV不混配，重复坐标造成的标签歧义不强制分类。空间近邻代理同时保留，两种信息不混称真实物理噪声标签。

[官方 README 的 MDD 接入说明](https://github.com/ylwhxht/V2X-R/blob/master/readme.md#train-model-with-multi-modal-diffusion-denoising-mdd-module) 也要求 `use_DeLidar=True`。此网络页面用于辅助核对角色；执行仍使用本地冻结源码与服务器哈希，不自动下载最新代码。

## 原实验记录身份

`original_FINAL_RESULTS.json` 是已有 `logs/train/FINAL_RESULTS.json` 的原字节副本，包含原六组结果。它提供历史参照，不能当成本轮 E0 的新结果。C/D的checkpoint SHA256由该记录读取，具体epoch、checkpoint路径和20轮Clean最优选择则从服务器 `L0_C/selected.json`、`L1_D/selected.json` 及训练契约/候选记录核验，代码没有按文件名猜epoch。

原C/D的Fog固定GT收益、Snow AP70与匹配数量方向不同均只构成待诊断现象。新模块不预设可靠性塌缩、有效点被误压或信息缺失中的任何一个已被证明。

## 解释边界

- R1同时改变幅值；R2保留车内平均值、R3保留车内多重集合。仅按固定干预判断空间对应关系在当前模型的作用。
- 分布/条件相关性和配对占用是描述性证据；来源因果、完整候选排序/NMS因果、真实噪声校准仍有限制。
- 原推理重复误差先在同一输入测量；预设容差不可按结果放宽。没有宣称ROCm逐位确定性或把所有差异归因于某一个内核。
- 不继续自动实现E1–E4，不读取正式test。最终建议只能由服务器产物生成。
