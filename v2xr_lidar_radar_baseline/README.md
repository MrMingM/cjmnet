# LR0 · V2X-R 双模态 AttFuse

LR0 直接构建官方 PointPillarIntermediateLRF。激光雷达和雷达各用一个 PointPillars 编码器，把每个点柱压成 64 个数。两路分别铺成每车 64 × 200 × 704 的鸟瞰特征图，按激光雷达在前、雷达在后的顺序拼成 128 通道。官方三层主干和 AttFuse 根据 record_len 对每组车辆做注意力融合，再用原分类和边框回归头。检测阈值和去重方式沿用官方配置。

## 数据和协议

- 海光已逐个校验 120,440 个训练文件和 18,246 个验证文件，覆盖 8,083/829 个协作帧及 24,088/3,041 个单车帧。训练只核验所需的 Clean、Fog、雷达、标注及 camera0。验证另外核验 Snow。
- 同一场景、车辆和时间戳配对。官方数据发布时已将雷达投到对应车的激光雷达坐标系，两路再用同一个车到主车变换。官方加载器关闭模拟时延。天气分支只切换激光雷达路径。
- 雷达文件的字段实测为 x y z rgb。官方加载器取前三维位置和 rgb 的红色通道。该通道按官方原值输入，没有显式多普勒速度输入。
- 沿用 A1 的种子 20261017、20 轮、每批 8 帧、Adam 初始学习率 0.002、权重衰减 0.0001、eps 1e-10。每轮末传入从 0 开始的轮次，调度里程碑为 10/15，倍率 0.1。逐车读取时调用官方 randint(0, 6)，其中 0–2 读 Fog，其余读 Clean。每轮 1,010 次更新，末批 3 帧按旧配方丢弃。官方训练集遇到无有效真值仍会重抽训练样本。
- 沿用官方双模态检测损失 point_pillar_loss。A1 的检测损失入口为 point_pillar_loss_opencood，且 A1 还有自研分支。LR0 按同一遍 829 帧 Clean 推理记录验证损失与 AP，逐帧计算损失后平均。A1 的损失来自独立的批量验证，因此验证损失数值不能直接横比。
- F 协议固定同一模型对应的双模态 Clean 真值。原官方双模态过滤在任一模态有点落入框时可保留框。正式报告同时对照冻结的纯激光雷达真值，单列每天气的目标增减。
- 官方双模态原配置为 30 轮，本实验用 20 轮，按完整 Clean 验证 O 协议 AP70 选最高值，精确同分取早轮。

## 空观测

海光已观察到场景 2024_06_24_19_56_04、帧 000068、车辆 182 的雷达过滤后为空。原官方实现会在这种情况下产生随机点。当前协议保留真实点用于真值过滤，保留车辆位置，空模态的鸟瞰特征置零并记录次数。设置 LR0_EMPTY_POLICY=reject 可在首个空模态停止。退化预测框在官方去重前排除，统计记录在各天气评测报告中。

## 启动和续跑

在海光执行。

    r=/data/cjm/datasets/logs/v2xr-lr0-v3; mkdir -p "$r" && nohup sh /home/cjm/OpenCOOD-main/cjmnet/v2xr_lidar_radar_baseline/run_all.sh >> "$r/launch.log" 2>&1 < /dev/null &

正式阶段在真实数据自测通过后启动。重跑相同命令从最后已提交的完整轮次恢复。已完成的天气评测按哈希复用。最新错误见 FAILURE.json，详细过程见 launch.log。结束后生成 RESULTS.md、FINAL_RESULTS.json 和 selected.json。严格拒绝模式的输出目录为 /data/cjm/datasets/logs/v2xr-lr0-v3-reject。

## 后续 LiDAR 分支接入

官方 forward 将 ego.processed_lidar 的 voxel_features、voxel_coords、voxel_num_points 和 record_len 交给 lidar_pillar_vfe。这是接入 GSPR 和柱内邻居增强的分支入口。可以在独立实验中扩展这个编码器或在它前面增加输入适配层。保留 64 通道的柱特征和 scatter 接口，即可继续使用同一雷达编码器、拼接、主干及检测头。

