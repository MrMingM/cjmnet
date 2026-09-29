# LMD / MetaDetect3D 特征兼容性

本目录实现的是 **adapted LMD-core**，不是把原仓库依赖 MMDetection3D 的数据爬取器原样搬进 OpenCOOD。

判断依据来自仓库中已提交的 `related_code/MetaDetect3D-main`，重点是：

- `src/metadetect3d/meta_detect3d_metrics.py`
- `src/metadetect3d/training_metrics.py`
- `configs/data_configs/kitti_train_val_test_reduced_pointpillars_2.py`
- `configs/meta_models_configs/best_models.py`

| 原 LMD 特征 | 原源码含义 | 当前中融合 F 是否可无歧义得到 | 本实现 |
|---|---|---|---|
| output score | 检测器输出置信度 | 是 | 保留 |
| output box x/y/z/l/w/h/yaw | 检测框几何 | 是 | 保留 |
| volume / surface / ratio | 输出框几何派生量 | 是 | 保留 |
| Number of Candidate Boxes | 与当前框 BEV IoU 超阈值的 proposal 数 | 是；固定 top256 作为 proposal pool | 保留，属于 top256 adaptation |
| proposal x/y/z/l/w/h/yaw min/max/mean/std | 邻近 proposal 几何离散度 | 是 | 保留 |
| proposal volume/surface/ratio stats | 邻近 proposal 几何离散度 | 是 | 保留 |
| BEV IoU min/max/mean/std | output 周围 proposal 的 BEV 重叠离散度 | 是 | 保留，精确 polygon IoU |
| max-score / score-sum stats | proposal 类别分数统计 | 单类别 OpenCOOD 只有一个融合 vehicle score；无多类 score-sum 等价物 | 保留单一 score 统计；不伪造 score-sum |
| IoU3D min/max/mean/std | output 与 proposal 的 3D IoU 离散度 | 当前项目 Oracle/主评测固定为平面 polygon IoU；高度坐标语义没有必要另造 | 不保留 |
| points-in-box | 原始单车 LiDAR 框内点数 | **无语义等价物**。当前 detector 是多车中间特征融合，没有一份等价的 fused raw point cloud | 不保留 |
| points / total points | 框内点比例 | 同上 | 不保留 |
| reflectance max/mean/std | 框内原始点反射强度 | 同上 | 不保留 |
| category / multi-class scores | 多类 detector 输出 | 当前任务单类别 vehicle | 不保留 |
| dir_score | 独立方向分类分数 | 当前 F 接口没有等价独立方向分支 | 不保留 |

## 另一处必须明确的 adaptation

原 MetaDetect3D pipeline 主要对 detector **post-NMS outputs** 建立 meta quality。

本项目现在要回答的是：

> 固定 decoded geometry 时，现实质量估计方法能否把 top256 pre-NMS 候选中的好框重新排到正确位置，并兑现 GT-IoU Oracle 空间。

因此本实现把 LMD 的核心结构：

> output/candidate 自身属性 + 邻近 proposal 离散度 -> prediction quality

应用到固定 top256。

论文/实验中应写成：

> adapted LMD-core on the fixed top-256 cooperative candidate pool

不能写成“完整复现原 LMD”。

## 训练目标

原提交源码的 meta regression 标签为 `True_BEV_IoU`；本实现对应为当前候选与同帧 GT 的最大 BEV polygon IoU。

GT 只允许用于 official train 标签和事后评价，推理时不进入任何 feature、选池或阈值。
