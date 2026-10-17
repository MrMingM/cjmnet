# D−1B：官方天气 evaluation GT 审计

范围：829/829 帧；完整 validation。
GT 独立于 detector；没有 prediction=None 跳帧。

A. 在已审计帧和约定数值容差内，Clean/Fog/Snow 最终 GT **不一致**。

| 条件 | 最终 GT 总数 |
|---|---:|
| Clean | 6224 |
| Fog | 3673 |
| Snow | 5854 |

B. 以下目标按同一物理帧的 object ID 配对；removed 与净减少分别报告。

| Clean→天气 | retained | removed | added | 净少 | 净少/Clean | 变化帧 | 几何变化目标 |
|---|---:|---:|---:|---:|---:|---:|---:|
| fog | 3673 | 2551 | 0 | 2551 | 40.9865% | 637 | 0 |
| snow | 5854 | 370 | 0 | 370 | 5.9447% | 179 | 0 |

C. 每个候选框的点数直接取自官方 native ROI 调用；同时核对天气间候选 ID、几何和 CAV context 一致。
- fog：2551/2551 个删除目标在天气下所有候选 CAV 框内均为零点；占 100.00%，主要由点依赖过滤导致：True。fallback 涉及 26 个删除目标。
  新增原因计数：{}。新增可来自天气点进入框内、临时点，或去重代表框改变后的范围筛选。
- snow：370/370 个删除目标在天气下所有候选 CAV 框内均为零点；占 100.00%，主要由点依赖过滤导致：True。fallback 涉及 0 个删除目标。
  新增原因计数：{}。新增可来自天气点进入框内、临时点，或去重代表框改变后的范围筛选。

D. 官方天气 AP 与 paired fixed-GT AP 不能直接相减解释为纯模型天气鲁棒性。
GT 有变化时，计分目标集合也随天气改变，难目标可能退出计分，新增点也可能使目标进入计分。
后续应分开报告官方协议与固定 GT 协议，并在相同权重、输入、预测后处理和 AP 实现下做配对控制。
即使本次 GT 完全一致，也只排除 GT 集合差异；官方 prediction=None 跳帧与其他评测差异仍须对齐。
本审计没有计算 AP，也没有修改过滤行为；无法据此量化 AP 偏差。

几何容差：center/dimensions/corners=1e-4 m；yaw=1e-5 rad，角度按 2π 周期比较。
点数是各 CAV 经官方裁点/FOV/空体素处理后、送入 GT 过滤的点数。best-CAV 为最大单 CAV 点数；跨 CAV 不相加。
physical_post_fov 点数在临时点生成之前，由相同 native 算子旁路计算；fallback 单独列出。
完整输入、源码、native 扩展、运行配置、逐帧恢复契约见 protocol.json / input_manifest.json / contract.json。
