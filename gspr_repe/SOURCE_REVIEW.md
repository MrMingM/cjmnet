# 源码和论文审查

审查日期：2026-10-23。原实验目录保持原样；新增代码集中于 gspr_repe。

## 现有接口与执行顺序

- `attfuse_gspr/reliability.py::GeometryFirstPillarReliability.forward`：输入原始 XYZI、有效点数、坐标，输出 point_reliability、point_uncertainty、point_evidence、point_valid_mask 及柱体统计。
- `attfuse_gspr/pillar_vfe.py::ReliabilityPillarVFE.forward`：装饰特征→掩码→可靠性软加权→PFN。旧实现不得再叠一次可靠性加权。
- `attfuse_gspr/point_pillar_gspr_attfuse.py::PointPillarGsprAttfuse.forward`：GSPR→VFE→Scatter→AttBEVBackbone→检测头。
- `v2xr_gspr_clean/integration.py::PointPillarGsprOfficial`：真实官方模型子类，在 VFE 前算 GSPR，在首层 PFN 前对装饰特征软加权。原官方损失是 PointPillarLossOpenCOOD。
- `v2xr_weather_l012/models.py::create/install`、`training.py::run/config/prepare`、`data.py::AuditedDataset`：混合天气、每 CAV Fog 概率3/7；20轮、seed20261017、batch8，Adam .002、milestones10/15。官方 Scatter 后尚未开始多尺度 AttFuse，可在这里接 B/C。保留官方主干、注意力和检测头。
- `gspr_e01/adapters.py::retrieve_tree/training_tree`：旧实验通过受审查的语法树适配固定函数；本次使用独立副本，避免扩大旧适配器的职责。
- `v2xr_train_attfuse/evaluation.py::evaluate` 委托 `v2xr_protocol_audit/predict.py::generate` 和 `evaluate.py::replay/official_match_trace`。本次直接复用，不重写 AP/NMS/匹配。O 跳过 None；O*计入None；F固定Clean GT。
- 版本依据为 SHA256 与 source_snapshot.json，服务器无需 .git。

## 本地资料扫描结果

扫描 `研究相关论文/1`，实际包含 `PointASNL.pdf`、`TransBridge.pdf`、`SAFDNet.pdf`、`PointASNL-master.zip`、`HEDNet & SAFDNet.zip`，另有 TipleMixer.pdf。没有找到本地 TransBridge 源码。PDF正文用 PyMuPDF 读取，源码直接从压缩包读取，未执行其中代码。

### PointASNL

论文 §3.1 公式1–4：先更新邻域关系，再以归一化权重同时更新坐标和特征；§3.2还有局部与全局关系聚合。

实际源码：`PointASNL-master.zip::PointASNL-master/utils/pointasnl_util.py`，关键函数 `SampleWeights`、`AdaptiveSampling`、`PointNonLocalCell`、`PointASNLSetAbstraction`。

`SampleWeights` 使用相对位置及查询/键关系更新组内特征，再对邻居归一化。`AdaptiveSampling` 分别加权坐标和特征，`PointASNLSetAbstraction` 合并局部、全局与残差。

本次 A 只借鉴相对几何、归一化加权和残差。柱内线性复杂度，保持坐标/点槽位，不使用远点采样、不移动点、不移植全局关系模块。原参考为 TensorFlow，使用自定义 .cu 采样/分组算子；这些依赖没有引入本任务。

### TransBridge

论文 §III、Figure 2：检测编码器与补全解码器通过 Up-sampling Bridge 和 Interpreting Bridge 连接；用多帧前景/背景重建得到密集体素存在标签。

本次 B 的 Q/S 双头、距离分层和监督掩码是任务适配，不能归为论文原设计。V2X-R Clean 单帧只提供相对参考，没有多帧完整场景标签。未发现本地官方源码；没有据此宣称不存在公开代码，更不声称完整复现。

推断部分：以轻量二维卷积读取柱体特征及统计，将可信程度和有效支撑分开。Q 为 GSPR 弱代理；S 优先对齐训练 Clean 点数，无法对齐时采用距离分层代理。空单元保持未知，不作为缺失目标真值。当前不监督空单元的物体存在性。

### SAFDNet

论文 §3：通过自适应特征扩散解决目标中心处无特征；由分组前景预测决定扩散范围，再执行稀疏特征提取和检测。

实际源码：`HEDNet & SAFDNet.zip::HEDNet-main/pcdet/models/backbones_3d/hednet.py::SparseHEDNet`，关键函数 `assign_target`、`get_loss`、`feature_diffusion`、`to_bev`；配置在 `tools/cfgs/safdnet_models/`。`assign_target` 用目标框确定前景；`feature_diffusion` 用分组分类及各组池化范围扩展稀疏坐标。

本次 C 只借鉴受条件约束的特征扩散。固定八邻域，预测特征相似性，以来源Q/S/未知度约束权重，并对目标支撑不足作残差门控。不生成新点、不扩展稀疏坐标、不移植检测头。保守版本仅增强已有观测单元，完全空区域的补全留作后续研究。原参考依赖 spconv 与 GPU 框内点算子，新模块没有引入它们。

## 兼容性和解释边界

新增模块只用 PyTorch Linear、BatchNorm、Conv2d、切片、掩码和张量归约，无新增 CUDA 专用扩展。仍依赖服务器既有官方 ROI/IoU 扩展，通过原 D−1A 字节和环境校验。实际 ROCm 前后向尚未本地执行，不能据算子列表宣称已经兼容通过。

A 启用时：原始装饰特征→有效点线性映射/归一化/最大池化，与可靠性几何加权分支门控融合。填充点排除于归一化、最大池化和加权归约。A关闭时严格走历史GSPR路径，因此历史PFN的填充归一化行为仍保留；这项差异应在分析A收益时披露。

B3/B4 的参数和邻域循环相同；B3只用观测状态控制目标增强，来源按占用掩码；B4再用来源质量和支撑约束。B2是同一局部增强器的无状态对照。B3−B2同时包含A/B，不能单独归因B。当前五组支持基本对照，不能独立识别三个模块的全部因果贡献。


## 2026-10-24 NVIDIA迁移补充

新增cuda_runtime.py/cuda_evaluation.py，沿用AI_REMOTE.md已有CUDA12.4运行环境和独立官方源码副本。新环境的.so必须实测CPU/CUDA框内点及框重叠，通过后记录本次哈希；不比较海光.so字节。

rsync -L可将历史软链接复制成普通文件；迁移检查依提取清单和实际内容哈希，不要求历史链接realpath一致。原.7z不作为已提取数据训练的运行前提，原归档和CRC记录仍作为历史提取来源，明确没有在NVIDIA重新校验原归档。

调用冻结 `gt_audit/observe_gt.py::observe` 建立本次真实GT审计；新推理包装直接调用官方 `inference_intermediate_fusion`、原后处理和 `eval_utils_opv2v`，并复用 `official_match_trace`。包装与旧inference.main外层不同，应披露；没有重新定义AP/NMS/匹配，也不声称CUDA与ROCm逐位等价。旧历史GT/环境/权重记录未修改。

另将训练验收辅助代码独立放入gspr_repe/train_acceptance.py，固定指向原v2xr_train_attfuse/TRAIN_DATA_ACCEPTANCE.json，修正独立包导入时HERE指向当前目录造成策略文件位置错误的问题；历史策略内容和来源未改。
