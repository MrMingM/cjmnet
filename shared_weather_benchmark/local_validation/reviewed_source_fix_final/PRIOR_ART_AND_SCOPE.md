# 已有工作与本轮范围

核查日期：2026-10-20。本轮问题是：总体衰减相近时，空间结构是否改变多车对同一目标的证据差异？这是待验证的问题，尚未证明研究空缺。共享空间场、非均匀雾、共享粒子和不同车辆干扰均已有先例。

## 实际查阅的论文证据

| 工作 | 本次可核实内容 | 本次阅读范围 / 实现关系 |
|---|---|---|
| [Hahner et al., ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Hahner_Fog_Simulation_on_Real_LiDAR_Point_Clouds_for_3D_Object_ICCV_2021_paper.pdf) | 已有晴天点云上的雾衰减与散射模拟 | 查阅CVF论文检索文本和摘要，PDF直接抓取失败；完整阅读仓库冻结原核。E0直接调用该原核；E1–E3仅扩展衰减 |
| [Alam et al., 2026](https://arxiv.org/html/2607.25612v1) | ReDAM将雾强度分布对齐到参考数据；Unified-weather-edit用世界坐标共享雨雪粒子 | 阅读HTML方法II-A/II-B及验证结构。雾强度对齐与雨雪位置对齐需区分。本轮未复现其跨模态模拟或校准 |
| [Qian et al., WeatherEdit, AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/37802) / [方法全文](https://arxiv.org/html/2505.20471v2) | 多视角、多帧天气编辑和4D Gaussian粒子场；局部场有随渲染相机移动的设计 | 阅读HTML §3及AAAI出版页面。本轮世界固定消光场为独立设计，不声称复现其渲染场或LiDAR物理 |
| [Adversarial Fog, AsiaCCS 2025](https://doi.org/10.1145/3708821.3733904) / [作者机构记录](https://waseda.elsevierpure.com/ja/publications/adversarial-fog-exploiting-the-vulnerabilities-of-lidar-point-clo/) | 已有分层雾及预处理失败研究 | 核查出版记录/摘要；完整论文抓取失败。层状数值对照不称该攻击复现，也不进行攻击优化 |
| [RCP-Bench, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Du_RCP-Bench_Benchmarking_Robustness_for_Collaborative_Perception_Under_Diverse_Corruptions_CVPR_2025_paper.html) | 摄像头协同感知的14类干扰、六种协同情况；存在ego/CAV干扰分析 | 核查CVF摘要与检索到的论文/补充材料相关段落；完整PDF抓取失败。已有协作补偿/干扰评价，不宣称首次研究 |
| [MSC-Bench, ICME 2025](https://arxiv.org/html/2501.01037v1) | 16类多传感器干扰；LiDAR/相机雾参数一致性 | 阅读HTML §III及实验，会议身份通过ICME目录核查。本轮不复现其nuScenes基准 |

上表不代表所有论文均完整阅读全文。无法核实的物理细节不进入实现依据。代码中的世界场和非均匀散射核接口统一称“独立设计的近似原型”。

## 实际源码审阅

- 完整阅读根目录 `AI_CONTEXT.md`，重点核对§44–49的新旧状态及冻结边界。
- `reference_weather/` 的Fog原核、adapters、common、generate、config、evaluate、README、UPSTREAM和assets_lock；资源校验入口也已检查。Rain/Snow不进入本原型。
- `opencood/tools/materialize_physics_weather.py`：旧生成器有shuffle、车体过滤与空输入回退，不能拿其点顺序作为原始身份。本实现读取原始PCD后立即分配索引。
- `opencood/utils/{transformation_utils,pcd_utils,box_utils,eval_utils}.py`。
- `opencood/data_utils/datasets/{basedataset,intermediate_fusion_dataset,late_fusion_dataset}.py`：场景/CAV字典、模拟时间戳、位姿读取、投影与GT合并路径。
- `opencood/data_utils/post_processor/base_postprocessor.py`、`opencood/hypes_yaml/yaml_utils.py`、`opencood/tools/inference_utils.py`，以及reference_weather评测入口。

本轮不调用detector和原dataset的数据增强。PCD沿用OPV2V格式，观测GT另行冻结；不能把此GT口径直接当原官方AP口径。

BaseDataset用途修正：服务器报告其SHA256为`62ef160af933e88db9ab1019e9d44b730e879bb3b7aca9848e89294c537fe381`，与本次审阅的`2b60c2e5160a985cf0197e471216292be26685aec45643d6ec711412dd66a36b`不同。尚未阅读服务器变体，不判断两版本是否等价。SCOR-W的`dataset_io.py`独立读取YAML/PCD，`audit.py`只调用固定的`transformation_utils.py`与自身GT审计；没有调用BaseDataset。快照保留其审阅哈希，运行时记录实际哈希，禁止导入整个`opencood.data_utils`（含间接/动态导入及缓存模块）。其他固定源码的精确哈希要求保持，真实坐标/GT/身份门禁保持。此项仅修正参考源码的运行依赖分类，不改变天气或观测协议。

## 从历史研究继承的约束

§44的旧AP天气筛选结果与新协议完全分开。§45早期迁移与§46–48官方GT/AP不是同一口径。§48已确认天气GT筛选会改变分母，因此本轮固定Clean目标集。§49当前Clean-only GSPR为负结果；本轮不优化GSPR、不恢复已停止的source utility主线。

只用原OPV2V train的一小部分scene；不读取validate/test。OPV2V-W继续封存，V2X-R validation已反复研究，不能称独立test。场景规则、seed、诊断阈值和匹配容差先固定，再生成；本轮无AP选择或参数拟合。
