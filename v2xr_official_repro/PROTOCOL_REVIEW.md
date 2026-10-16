# D−1A 官方源码核对记录（2026-10-16）

仓库：https://github.com/ylwhxht/V2X-R

固定 commit：`f3ebbad95561444a53a17358735b0f0dc4ba89f7`。
通过 GitHub API 核实 master 指向该版本，再下载该 commit 的完整源码归档。
`upstream/` 是归档逐字节展开结果；`UPSTREAM.json` 保存逐文件 SHA256、归档 SHA256
与关键代码定位。保留上游原始作者/许可证说明，未编辑上游代码。服务器不需要 Git 或网络下载源码。

用户追加提供的本地文件 `C:\Users\25313\Downloads\V2XR_attfuse.yaml` 已读取并比较：
与此commit模板逐字节一致，SHA256均为
`7e1ce936d779138b5b1ca6462ebc79d9e604920d9a29b7ef1c3a64e50d96e346`。
无需改变网络或评测字段；服务器仍核对实际传入文件，避免同步错文件。

## 实际调用链

1. `opencood/tools/inference.py`：`yaml_utils.load_yaml(None,opt)` 从 model_dir/config.yaml
   加载配置；`build_dataset(..., visualize=True, train=False)`；batch=1、shuffle=False、
   drop_last=False，默认16个worker。
2. `data_utils/datasets/__init__.py:7`：注册名 IntermediateFusionDatasetV2XR 实际是
   `opencood.data_utils.datasets.opv2v.intermediate_fusion_dataset.IntermediateFusionDataset`
   的别名。路径带opv2v不意味着它与本项目现有实现相同。
3. `train_utils.create_model()` 从官方 `opencood.models.point_pillar_intermediate`
   创建 `PointPillarIntermediate`，使用官方 common_modules 的 VFE、scatter、AttBEVBackbone。
   model.args.use_modality=`processed_lidar`。
4. `inference_utils.inference_intermediate_fusion()` 执行模型和官方 dataset.post_process，
   后者调用官方 VoxelPostprocessor 和 generate_gt_bbx。
5. `inference.py:61–62` 的 `if 'opv2v' or 'V2XR' in opt.model_dir` 因非空字符串总为真，
   实际选择 `opencood.utils.eval_utils_opv2v`。`utils/eval_utils.py` 也已检查，
   但它不在这条官方推理路径上。未修复此分支。

## 固定配置与GT

| 项目 | 本版本实际值/行为 |
|---|---|
| 模板 | opencood/hypes_yaml/V2X-R/LiDAR/V2XR_attfuse.yaml |
| fov / proj_first | True / True |
| eval_sim / train_sim | False / False，仅读取Clean |
| score threshold | postprocess.target_args.score_threshold = 0.20；原代码使用严格大于 |
| NMS threshold | postprocess.nms_thresh = 0.15，官方 nms_rotated |
| voxel | [0.4,0.4,4]，每voxel最多32点，eval最多70000 voxels |
| 输入/anchor范围 | [-140.8,-40,-3,140.8,40,1] |
| GT_RANGE_OPV2V常量 | [-140,-40,-3,140,40,1]；不能直接称为本次实际GT范围 |
| 实际GT范围 | base_postprocessor.py:223、:109 使用anchor_args.cav_lidar_range，即±140.8 |
| COM_RANGE | 70m |
| max_cav | YAML未指定，官方BaseDataset默认7；本验证集每场景2–5个agent，不改成项目默认5 |
| ego选择 | 场景与agent按字典序；负数RSU从首位移到末尾；第一个剩余agent为ego |
| GT可见性筛选 | 每个agent的投影点云先裁到ego camera0 FOV，再用points_in_boxes_cpu保留至少1点的框；跨agent按ID去重 |
| 数据过滤顺序 | 单车LiDAR先按本车范围过滤、去自身点，再投影和FOV裁剪；不替换为本项目投影后范围过滤 |

纯LiDAR分支依然加载真实radar.pcd和4张camera图，处理radar特征；模型只消费processed_lidar。
`mask_ego_fov_flag` 使用ego camera0标定与当前agent camera0实际图片尺寸。
不得用空雷达、假图片、固定尺寸或手写FOV替代。旧paired视图缺少这些依赖，因此只复用
其原始Clean PCD/YAML，额外文件来自原validate包，放新目录。

## AP及必须保留的边界行为

`eval_utils_opv2v` 使用Shapely BEV平面多边形IoU。逐帧按分数从高到低贪心一对一匹配，
按官方帧顺序追加TP/FP；`eval_final_results` 默认global_sort_detections=False，
使用VOC2010精度包络积分。没有跨帧统一按分数排序。本工具直接调用官方函数，不重写AP。

`inference.py:139` 在pred_box_tensor为None时continue，连该帧GT都不计入AP分母。
本工具原样执行，并分别记录所有帧GT总数、官方AP累计GT、跳过帧数。
官方数据处理里的空voxel随机点处理也保持原样；不把旧实验“没有补点”的声明挪到本实验。

## 官方checkpoint加载

README要求文件名0.pth，但当前 `train_utils.load_model(...,epoch=0)` 实际打开
`net_epoch0.pth`。独立model_dir建立0.pth与net_epoch0.pth两个链接，均指向用户的同一原件。

官方loader会去除DataParallel的module前缀、映射cdd前缀，然后strict=False，
甚至可能丢弃多余参数、用随机初始化补缺失/形状不匹配参数。
本工具在调用官方loader之前验证完整key、shape、dtype吻合，之后逐tensor验证加载值完全相同。
任何补缺失/丢参数需求都停止，不接受“部分加载也能跑”。这只是失败断言，不改变加载成功时的模型行为。

用户YAML必须与该固定版本模板的行为配置一致；仅允许原始数据路径不同。
运行副本只重写root_dir、validate_dir，保留增强、阈值、网络、GT等全部字段。
原checkpoint/YAML永不改写，checkpoint自动发现要求唯一，否则必须明确指定。

## 隔离方式及可解释范围

完整官方snapshot复制到数据盘runtime。子进程cwd/PYTHONPATH指向runtime；
所有已导入opencood模块的路径与哈希逐项记录/核验，拒绝从cjmnet或父OpenCOOD拼接Python或扩展模块。
wrapper执行原始 `inference.main()`，仅增加路径/完整性/有限性断言、计数、报告。
smoke用前N帧Subset；全量不替换数据集、不修改采样顺序。worker默认16，可显式调整并记录。
随机种子固定20261016，保留官方Torch backend设置；版本/HIP/GPU均记录。

上游安装文档推荐Python3.7/Torch1.10/CUDA；服务器是Python3.10/Torch2.5.1/HIP。
当前任务复用官方源码和协议，数值复现仍须实测。缺依赖/扩展/ABI不兼容立即失败，
不注入stub，不调用本项目替代实现，不自动pip降级或改写共享环境。
可显式执行build阶段，在独立runtime中运行官方build_ext；Torch如何HIP编译由服务器工具链决定。
若编译失败，提交完整日志再处理环境，不擅自修改GT算子或换成近似实现。

2026-10-16构建范围修正：初版全量pcdet扩展构建在PointNet2旧THC头文件经HIP转换后的
THH/THH.h处失败。host预检已经成功，roiaware已完成链接；全量build没有成功结束，
不能据此宣称roiaware已可导入。新wrapper在执行官方setup时，仅缩小传入setuptools的ext_modules列表。
保守AST检查覆盖63个模块，只到达box_overlaps和roiaware_pool3d；PointNet2与pcdet iou3d_nms未到达。
只构建这两个原版依赖，官方源码/编译对象/检测路径不修改。静态证据在NATIVE_DEPENDENCIES.json；
实际选择/转换后编译源码哈希在runtime/native_build_selection.json，并进入结果protocol。
运行时若出现意外PointNet2依赖会停止，禁止stub替代。上游源码快照继续逐字节校验。

完整829帧结果对照参考0.8669/0.8258/0.6656，输出绝对pp差。
AP70差>2pp显著标记OFFICIAL REPRODUCTION NOT YET ALIGNED。
差≤2pp只表示参考距离，不产生统计意义PASS。所有情况都停止在D−1A，没有后续阶段入口。
