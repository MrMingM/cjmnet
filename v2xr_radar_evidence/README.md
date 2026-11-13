# V2X-R 冻结雷达证据诊断

唯一入口 run_all.sh。入口先检查设备，再并行启动独立采集，全部成功后按 D0、B、A、P1 的依赖顺序汇总。D0 用 CPU，训练开发采集、正式验证采集、F3 敏感性和 F0 开关配对依次用卡 1、2、3、4。F0 的 A 和 P1 共用候选缓存。输出目录 /data/cjm/datasets/logs/v2xr-radar-evidence-v1/。入口支持复用已完成任务，候选和开关缓存逐帧写入。

D0 复用全包字段清单，并对各车的首、中、末雷达 PCD 解包比对。物理含义只有在有发布编码证据时才确认。B 使用 F3 第 10 轮，重跑同一批 Fog 输入的四个干预，同时核验历史缓存。现有柱打乱保留占用坐标并重新编码柱内点。跨帧雷达选自同场景，保留供体坐标。A 与 P1 使用冻结 F0 第 20 轮，原后处理固定不变。另采集实际降低分数阈值到 0.05 后的候选，记录原阈值、各阶段覆盖和官方匹配。正常预测与干预的 AP 统一使用固定 Clean 真值的 F 协议。

A 使用训练场景的五折统计和冻结模型正式验证。逻辑回归为离线统计模型。P1 的逻辑回归拟合与阈值校准采用不相交的训练场景。验证集等误检预算前沿只用于评估排序。F0 的同帧开关配对使用历史 Fog 120 帧，按同帧鸟瞰重叠 0.5 贪心配对。候选在官方重叠 0.7 阈值下匹配真值。

在海光后台启动或恢复。

    r=/data/cjm/datasets/logs/v2xr-radar-evidence-v1
    mkdir -p "$r" && nohup sh /home/cjm/OpenCOOD-main/cjmnet/v2xr_radar_evidence/run_all.sh >> "$r/launch.log" 2>&1 < /dev/null &

日志为 launch.log、D0.log、A_P1_dev.log、A_P1_val.log、A_pairs.log、B.log 和 reports.log（部分早期运行的汇总输出直接进入 launch.log）。

