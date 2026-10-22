# P7 实际审阅与边界

本轮完整读取仓库根 `AI_CONTEXT.md`，包括§43停止边界、§44原版物理天气、§45–48协议/固定GT审计和§49当前GSPR负结果。本轮只新增 `scorw_p7/`，没有解冻任何检测器、旧实验或天气实现。

交付末期共享工作区的AI_CONTEXT被其他工作更新，新增§50天气混合训练结果；已补读该追加记录。它不授权本轮训练/检测，也不改变P7范围。原v1快照将AI_CONTEXT也严格固定，因此完整旧快照会报告这份文档不同。旧快照、旧检查源码和文档内容均不回滚/改写。P7独立策略将这一个不执行的上下文文档列为review-only，同时保留旧检查的原始失败诊断；57个原v1代码/配置/资源文件和原snapshot本体仍严格固定。当前文档哈希、审阅时哈希和旧期望哈希分开记录，文档更新不充当可执行源码变化或天气结果。

逐项阅读/核对：

- v1：`world_field.py`、`ray_integrator.py`、`geometry.py`、`audit.py`、`dataset_io.py`、`generator.py`、`observability.py`、`compare.py`、`provenance.py`、`run.py`、`config/prototype.json`；检查了PCD读取器和位姿源码的实际接口。
- P6：`core.py`、`run.py`、`selection.py`、`repeat.py`、`legacy.py`、`metrics.py`、`comparison.py`、`export_audit.py`、配置与README，以及 `scorw_p6_pipeline/` 的总控逻辑。实际本地P6包存在，源码完整性另由其snapshot核对，不能根据总控入口推断服务器/远端仓库也完整。
- 本地回传材料：`logs/sota/SUMMARY.json`、`NEXT_STAGE_DECISION.md`、`P6C_ALL_SCENES_RESULTS.md`、`P6C_UNMATCHED_ANALYSIS.md`。实际只有这四份P6结果文件，不声称本地读到服务器完整JSON、PCD或sidecar。审阅时的字节哈希存于新snapshot的 `reviewed_reports`，这是审阅记录；服务器历史审计另核对实际使用的原始产物。

确认P6为12 scene、2帧/scene、3 seed、36 scene-seed组合，总体匹配4/36，结论INCONCLUSIVE。验证组8 scene的ΔC均值方向3正5负，仅1个scene三seed同方向。这些结论只来自回传表与summary，P7没有补造旧置信区间或服务器原始数据。

发现的关键设计限制：

1. P6 E2/E3分别抽取天气场。相同总体分布不能保证有限实现的光路退化相等。32/36未匹配结果完整保留，旧差异不足以隔离天气关联。
2. P6的 `metrics.pair_analysis` 在固定天气实现内跨目标聚合相关性。目标距离和Clean可见性会进入这个相关系数。P7改成固定目标/帧/CAV对，在20场池内中心化后计算天气相关性。
3. 衰减型E1–E3原点数保留率为1，无法用原始点数变化替代连续信号变化。P7只计算量化前目标回波衰减，不写RGB量化天气PCD。
4. 原代码通过GT框关联Clean点，这只有几何代理含义，没有逐点车辆表面语义真值。P7沿用并披露；原始PCD行索引参与point identity。
5. P5/P6的future保留集合与已分析场景都必须排除在新验证之外。旧8个P6 physics scene不作为P7新验证；开发仅复用P6的4个开发scene。

P7的最重要区别是固定同一20场池、同一光路和同一Clean分母。每辆车在两个协议中都遍历这20个场；E3仅改变车辆之间的配对。这精确控制有限循环上的逐车边缘分布，保留跨车联合关联的差异。

E3是有限场池错位对照。它没有保证每次重复同强度，也没有保证有限池相关系数恰好为0。共同循环依赖不能当成20个独立scene。单邻车平均ΔC理论上为0，分布宽度和正值概率仍可改变；多个邻车的max算子也可改变平均C。

P7无需导入P6研究模块。P6源包缺失或不完整时会在source audit列出，真实运行仍要求服务器已完成P5/P6的选择、合同和汇总元数据。没有复用P6大数组/sidecar，因而没有把未经哈希检查的缓存当事实。重新读取原始Clean PCD并执行v1坐标/GT审计；只核验使用到的历史元数据，不谎称再次完整重放P6。

物理依据限于既有非负世界消光场与双程透射率 `exp(-2*tau)`，沿用固定0.5m最大步长的中点积分。数值复现、平衡和负对照通过不能证明实雾物理真实性。非均匀soft散射、标定阈值、真实扫描束、逐束时间与缺失晴天回波仍不具备。本轮没有新增论文复现或新颖性主张。
