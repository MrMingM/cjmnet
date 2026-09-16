# 源码适配记录

本轮读取用户下载的三个仓库，抽取与当前冻结 BEV 框架相容的核心计算，在独立包中实现。没有把它们作为三个现成模块直接串接，也没有声明完成原论文复现。文件校验值见 `source_adaptation_hashes.json`。

| 来源 | 已阅读核心 | 本包中的适配 | 未移植部分 |
|---|---|---|---|
| `related_code/D-Map-main` | `include/D-Map/D-Map.cpp` 的 `GenerateDepthMap`，传感器逆位姿投影、角度格最近深度 | `geometry.py`：真实原点的角度深度图与 4 层查询；另用 GSPR 点证据统计高度支持 | ROS、C++ 八叉树、滑动地图、完整占据/未知状态更新 |
| `related_code/Learning-Loss-for-Active-Learning-master` | `models/lossnet.py` 的中间特征损失预测；`main.py` 的 `LossPredLoss` | `heads.py`：稠密风险预测、候选对排序辅助监督；使用独立实现的 softplus 排序，忽略等值标签 | 原始整图 LossNet、标注采样循环、多任务原复现训练 |
| `related_code/dynamic-selection-main` | `dynamic_selection/greedy.py` 中掩码选择、任务损失评估与禁止重复选取；`utils.py` 的选择工具 | `supervision.py`：固定上下文的实际特征插入收益；`model.py`：有预算的无重复块选择 | Concrete/Gumbel 端到端软选择、原分类预测器、逐轮交互协议 |

这些是基于已读源码的结构/计算思想适配，Python 核心为本项目独立实现，不是逐行复制。尤其动态特征选择的软掩码训练没有移植：本轮使用明确的实际检测收益标签，便于检验排序是否学会。

D-Map 源仓库附 GPL-2.0 许可；dynamic-selection 附 MIT（Copyright 2023 Ian Covert）；所下载 Learning Loss 复现仓库根目录未见 LICENSE。源仓库及已有许可证均保留原位；本包没有复制其文件内容。已存在 OpenCOOD/GSPR 的许可与来源不因本次适配而改变。

GSPR 可靠/噪声证据不转换成占据/空闲质量函数。EvOcc 是研究阶段的状态表达参考，本轮没有获取或声称移植 EvOcc 源码。

本轮机制的待验证贡献是：显式保留天气证据缺口，再用真实补充重放监督匹配；组合三个论文名称本身不构成创新性证明。`concat` 与 `no_u` 的重新训练对照用于检验具体机制是否必要。
