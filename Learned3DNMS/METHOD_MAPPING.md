# Learned 3D NMS 方法映射与不可等同项

依据：Osterburg、Schütte、Bertram，Learned Non-Maximum Suppression for 3D Object Detection，IEEE IV 2026，arXiv:2606.03568，Section IV–V。

这是 paper-based **D2D-Rescore 的 OPV2V 适配实现**，不是官方源码的精确复现。论文所列 GitHub 地址在实现时通过 GitHub API 返回 404，无法核对作者具体代码及其未公开参数。GossipNet3D 不包含在本版。

| 论文组件 | 本项目实现 | 边界 |
|---|---|---|
| Detection-set 输入 | 每帧 geometry-valid 原分数 top256；原 F 冻结 | 论文 CenterPoint 在 pre-NMS score>0.1 上提取，非同一池 |
| Detection 属性 | OpenCOOD decoded xyz-h-w-l-yaw 显式转换为 [x,y,z,l,w,h,yaw] + score；类别恒为 vehicle | OPV2V 无 velocity，绝不补造速度或虚构多类别 |
| 特征编码 | 尺寸 log1p、航向 sin/cos、xyz 的 10 频 Fourier | 数值尺度由项目配置给定；LayerNorm 代替论文所述 BatchNorm |
| D2D-Rescore | 6 层、64 通道、4 注意力头、可训练注意力温度 | 符合论文主干描述；无法比对未公开源码 |
| Score refinement | delta logit + frozen original logit，末层零初始化 | 对应论文公式 (8)–(9) |
| 主论文式推理 | learned scores → 范围过滤 → Top-K | 论文 K=300；本项目池至多 256，并以原 F 每帧输出数作为 K，以保持输出预算公平；因此标记 adapted topk |
| 控制性推理 | learned scores → 原 rotated NMS → range → 同一预算 | 不是论文默认推理，仅用于同一 Oracle 口径对照 |
| Supervision | learned-score 顺序的一对一贪心匹配，OPV2V BEV IoU≥0.7 | 论文是 nuScenes 多个 center-distance 阈值 + 误分类补救，不能直接迁移；OPV2V 只有 vehicle |
| BCE | 在真实候选上计算二分类交叉熵，padding 不参与 | 论文同类型损失 |
| 训练数据 | Clean 官方 train + 固定物理 Fog/Rain/Snow train | 完全不调用天气模拟 |
| 开发评测 | 官方 validate Clean + 固定物理 Fog/Rain/Snow validate | 不是独立 OPV2V-W test |
| Oracle | 同一 top256 池 max GT BEV IoU、零 IoU 删除、原 NMS、相同预算 | GT 只用于训练标签或事后评价，从不传入模型 |
| 可学习范围 | 只训练 D2D score 模块 | 原 F、GSPR、PointPillar、Where2comm、cls/reg、box geometry 保持冻结 |

论文网络训练报告 100 个 epoch；本项目默认 10 个以允许较快完成完整数据的首轮基准，可通过 EPOCHS=100 改为论文量级。未执行真实服务器完整训练时，不得报告指标或宣称论文复现达到原文性能。
