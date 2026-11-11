# POT-CF 双模态多车融合

POT-CF 在最低分辨率的鸟瞰特征图上，先找邻车和主车可能对应的格子，再把有可信对应和没有对应的邻车观测分开送入检测模型。

- 入口为 `run_all.sh`，只在海光服务器的物理卡 4 执行。
- `model.py` 在官方 `AttBEVBackbone.forward` 调用 `fuse_modules[2](x, record_len)` 时接入新算子。主干前两层、原双模态编码器、检测头、数据和评测均直接复用 LR0。
- `fusion.py` 提供局部候选代价、同窗口普通注意力、已匹配和未匹配残差、传感器来源图。
- `transport.py` 提供非平衡最优传输和独立的平衡拒配槽对照。细节见 [方法设计](METHOD_DESIGN.md) 和 [实验计划](EXPERIMENT_PLAN.md)。
- `settings.json` 记录初始窗口、代价权重、求解和残差限制。正式开始后用源码和配置哈希保护续跑身份。

## 海光服务器后台启动

```sh
cd /home/cjm/OpenCOOD-main/cjmnet
r=/data/cjm/datasets/logs/v2xr-fusion-pot-v1
mkdir -p "$r"
/home/cjm/miniconda3/envs/opencood/bin/python -m pip install --no-deps --target "$r/deps" POT==0.9.5
nohup sh v2xr_fusion_pot/run_all.sh >> "$r/launch.log" 2>&1 < /dev/null &
```

POT 只用于数值对照，安装在实验数据盘。验收与训练同一入口，验收全部通过后自动开始 P1、P2、P3。参数 `--check-only` 只运行验收。续跑前确认旧进程已退出，再运行相同的后台命令。流程使用实验锁、空闲检查以及逻辑卡与物理总线地址核对。

海光实验目录保存 `STATUS.json`、`launch.log`、失败详情 `FAILURE.json`、实测 `SMOKE_RESULTS.json` 及完成后的 `FINAL_RESULTS.json`、`FINAL_COMPARISON.md`。在轮次训练结束后中断，可从已保存的验证前状态继续。在未保存的轮次内部中断，会重跑该轮。

## 参考

- 拒配槽和平衡传输。[SuperGlue 论文](https://arxiv.org/abs/1911.11763) · [实现](https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/models/superglue.py)
- 固定匹配质量和点云异常观测处理。[部分最优传输配准论文](https://doi.org/10.1111/cgf.14614)
- 局部几何与匹配掩码。[GeoTransformer 论文](https://arxiv.org/abs/2202.06688) · [实现](https://github.com/qinzheng93/GeoTransformer/blob/main/geotransformer/modules/sinkhorn/learnable_sinkhorn.py)
- 两侧边缘质量惩罚及数值对照。[POT 文档](https://pythonot.github.io/quickstart.html) · [求解器](https://github.com/PythonOT/POT/blob/master/ot/unbalanced/_sinkhorn.py)
- 原数据、投影及模型。[V2X-R 官方源码](https://github.com/ylwhxht/V2X-R)
