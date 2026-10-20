# 坐标审计：交付状态

本地仅有合成数值检查；真实OPV2V检查待服务器P3最前面的强制审计，尚未放行真实数据生成。运行输出目录中的本文件会由真实审计结果替换。

源码确认：点云是发送CAV的LiDAR local坐标；世界变换由lidar_pose提供，位姿顺序[x,y,z,roll,yaw,pitch]，角度为度。旋转矩阵等于Rz(yaw) Ry(-pitch) Rx(-roll)。yaw=+90时local +X映射至world +Y，保留CARLA数值世界约定。

PCD强度由Open3D读取RGB红通道，归一化0–1。GT以object ID合并，extent为半尺寸；OpenCOOD把location+center后构造物体位姿。本轮保留完整3D角点，避免仅转yaw丢弃roll/pitch。

P0验证反变换误差、混合旋转顺序、两车对共同世界点一致性及GT角点数值回转。P3逐文件检查同帧各CAV时间戳、恒定索引步长、相邻位姿跳变、同ID跨车GT角点、全部点往返误差、原始编码与输入哈希。

时间戳为loader使用的模拟帧标识；物理秒数、真实扫描线、逐束时间没有可验证依据。重要检查失败立即停止，保留FAILURE.txt和VALIDATION_GATES.json；没有跳过坏帧或补入Clean回退。
