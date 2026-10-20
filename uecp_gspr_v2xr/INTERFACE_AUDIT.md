# UECP / GSPR / V2X-R 接口审计

本目录只新增实现。UECP 原源码、旧 GSPR/AttFuse、旧 V2X-R 流水线和结果均作为只读依赖。没有创建分支或 worktree；运行不依赖 `.git`。

旧扩散实验目录存在其他任务的并行更新。本轮仅固定实际导入的底层common/runtime/data/geometry/training/preflight/config及包入口，不绑定其控制器、报告或文档；这些文件的原始字节不被本轮修改。已验证的原生V2X-R历史包仍按其原身份链完整核对。

启动修复：初版直接依赖清单误含attfuse_gspr/README.md。直接GSPR依赖收紧为实际加载的reliability.py，私有UECP依赖由两个入口的AST导入闭包确定（12个Python文件）及实际YAML；文档、图片、未执行的模型/加载器不作为本轮运行前提。此调整不改模型、损失、超参、数据或训练/评测行为。已有原生历史包的完整身份链保持原样，实际代码不同仍停止。

## 原版 UECP 的实际行为

依据 `related_code/UECP-main/opencood/models/bev_text_v1.py`、`models/utils/pyramid_fusion_modulev1.py`、`loss/point_pillar_depth_loss.py`、`loss/point_pillar_loss.py` 和 `tools/train.py`：

- 使用 LiDAR PointPillars、384→256 缩减层、不确定性预测头、4/2/1 多尺度融合、融合与单车 cls/reg/dir 检测头。复用 V2V4Real UAPF 的实际 LiDAR-only 模型配置，把空间范围改为原生 V2X-R 范围。
- 密度目标为 `1-min(count,10)/10`。200×704 的原始目标通过原模型 bilinear/align_corners=False 变到 100×352；训练和推理均保持相同的特征分辨率。最终推理不传 `un_maps`，且强制 `use_gt_unmap=False`。
- 原训练分别调用融合和单车 criterion。每次 `PointPillarDepthLoss` 都加入 `5×HybridUncertaintyLoss`，本实现原样累计两次。LiDAR 路径没有 `depth_items`，depth loss 为0。所有实验一致。
- 原生NumPy标签进入训练损失前统一转FP32，损失公式保持原函数；最终GT评测保持原生dtype。
- `UGPrecisionFusion` 只在单尺度 precision 权重处 detach U。多尺度置信度加权池化、跨尺度 ego confidence 仍可传梯度。E0 保持原版；E2 通过匹配固定源码的 AST 只移除这一处 detach；额外重训 E2_original_detach。
- 原版 warp 的全1图只表示网格边界，不能代表真实观测；原版平滑可能把权重扩散到缺失处。E0保留。E3用实际有效点区域掩码并在平滑后重新掩码、归一化；无来源处输出0。

## 兼容层

正式原生 runtime 独占顶层 `opencood`。`upstream.py` 用私有 `uecp_gspr_v2xr.uecp_vendor` 命名空间读取 SHA256 固定的 UECP 模型与损失，只改 import 名称，未把 UECP 路径加入 `sys.path`。包壳不执行 UECP 的其他数据工厂。独立 GSPR reliability 文件也显式按固定源码加载，避免 native bootstrap 移除项目路径后导入失败。

原生加载器把点投影到 ego 后作特征编码；UECP 需要每车本地 BEV 再 warp。本适配器保留原生 GT 和评测对象，额外生成每车本地模型输入：

1. 当前天气原始 XYZI → 原生本车范围过滤 → 自车点过滤。
2. 临时投影到 ego，通过原生 camera FOV 条件选出对应点的原始索引，再取回源车坐标。
3. 使用相同原生 voxelizer 在源车网格编码。点顺序使用独立固定随机流，不改变原生 GT 路径的全局随机状态。
4. 以源车到 ego 的 `T_i` 生成 `pairwise[i,j]=inv(T_j)@T_i`。原版 `normalize_pairwise_tfm` 和 `[ego,source]` grid_sample 对齐源特征；没有第二次投影 ego 特征。

训练/开发保留背景帧，并复用已审计的空云适配。空车只用零 sparse 坐标占位，VFE 输出再次清零，GSPR 聚合排除占位；占位不进入 GT 点集合。VFE 原版 `.squeeze()` 的单 pillar 输出由同一掩码操作恢复为二维。最终829评测仍走完整原版数据路径，逐帧核对冻结 D-1B/D-1C 的 GT ID、角点和CAV。相机、雷达仍是原生加载器的文件依赖，模型只接收 LiDAR。

模型输出 `cls_preds/reg_preds/dir_preds` 通过别名 `psm/rm/dm` 交给原生后处理。原生 `VoxelPostprocessor` 已有 dm 方向修正分支；仅补充 UECP 的 dir_args，不另写NMS、IoU或AP。原生 D-1A YAML 只管理数据/GT，实际 UECP 配置单独完整记录在模型配置与评测 contract，避免暗换模型工厂。

原生最终评测在模型构造之后才启动DataLoader随机流。UECP参数个数不同，会改变随机数推进量。评测工厂先在CPU构造后丢弃原生AttFuse参考对象，仅保留其构造后Python/NumPy/Torch/HIP随机状态，再构造UECP并恢复这一状态。参考对象不参与推理、不提供权重；这样原生worker种子与历史GT路径保持一致，不让模型初始化改变随机空云填充等旧行为。额外本地点顺序使用私有RandomState，所有最终GT仍须逐帧通过冻结审计。

## GSPR 和区域质量

冻结 OPV2V 天气逐点监督联合训练 checkpoint，SHA256 为 `93598c7f6cde8c8caa54155cde98bbe75ff2921481b24928a2172e2d6c5dd4c1`。严格加载 `gspr.*` 键、形状和 dtype；不使用 V2X-R Clean-only 随机/无逐点监督的负结果模型。

GSPR reliability 为 `.05+.95*alpha_reliable/sum(alpha)`，uncertainty 为 `2/sum(alpha)`，两者不互为补数。证据双通道非负、上限30。先按有效槽位求点和，再作空间聚合，最后除有效点数；保留独立 validity。原生 RGB 红通道强度0–1，预检逐文件解码，并在E1记录每天气/密度区间的分布、强度范围和饱和情况。

没有可追溯逐点天气来源标签时，E1只报告实际输出分布及覆盖情况，ABCD填空并标记 INCONCLUSIVE。PCD绿色通道没有足够来源说明，不自动充当真值。可选 simulator-origin 标签须绑定当前 PCD SHA256 和逐行标签；.001m内匹配、冲突剔除；未知点不作负例，区域需至少4个已知槽位且标签覆盖≥90%。

## E2/E3 与公平对照

质量适配器为 `Conv1×1(5→16) / SiLU / Conv1×1(16→1)`，113参数，末层零初始化。输入为C、均值可靠性、低可靠比例、独立uncertainty和validity；用有界logit修正，E2初始在有观测处为恒等，空处U=1。GSPR参数冻结且始终eval。

E3的C来自 `1-predicted_U`：控制单车区域内加权池化、跨尺度ego先验。Q控制跨车辆precision；Q按有效区域聚合，warp后按实际有效质量归一化，平滑后再次清除无观测位置。除原版融合块参数外，仅使用同一个113参数适配器。coverage-only、quality-only、joint、shuffled、no-separation、constant-quality均独立训练。

constant-quality保持相同的活跃适配器容量并计算后丢弃GSPR值，是参数量/计算对照。coverage-only停用适配器输出，不能将其登记的闲置参数误称为活跃容量。no-separation结构与E2相同，明确作为单图重训对照，不宣称是额外不同架构。

所有组共享UECP初始化哈希、帧顺序、帧级Mix3天气、有效batch8、优化更新数、20轮、选模规则。GSPR额外历史监督成本单列。ROCm grid_sample backward可能存在舍入差异；恢复重放逐项检查模型/AdamW/scheduler/RNG，浮点预设atol1e-5、rtol1e-4，整数和RNG精确匹配，同时记录是否逐位一致，不静默放宽。

## 已知科学限制

历史 §49 的 V2X-R GSPR Clean-only 结果为负，旧 OPV2V 收益不能推断本轮有效。E1不充分仍可做探索性E2/E3，不能报告E1通过。829 validation已反复使用，非独立盲测；没有正式test访问。20轮未必收敛，不追加；单种子AP差值不代表统计显著。
