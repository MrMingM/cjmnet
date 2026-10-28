# GGCA：GSPR引导的几何共识聚合

本文件定义待服务器验证的方法，不预设性能提升。A0复用指定旧B0；A1/A2/A3分别独立训练，只改变下述因素。没有B/C、点移动、硬删点、配准、额外点标签或辅助损失。

## 原路径与点特征

原始点x经过冻结实现的GSPR得到r、u。官方装饰特征d仍在原钩子中乘r，然后完整执行原PFN：Linear、原BatchNorm、ReLU、包括填充位置的原Max。其输出记h0。

新增钩子读取同一次原BatchNorm的输出，转置并执行无状态ReLU，得到f_i。它与该次PFN Max前的点特征值相同，保留梯度。没有重复执行Linear或BatchNorm，没有修改pfn.forward。所有有效点掩码仅作用于新分支；原路径的填充行为保留。

f_i的值已受GSPR影响。新分支不再次逐元素乘r；r和u只用作归一化关系权重的先验p_i=r_i(1-u_i)。两级归一化分别决定邻居贡献和查询点汇总，不将值直接乘r²。不能据此声称与一次加权完全等价，服务器需记录可靠性分布及梯度。

## A1：局部关系聚合

每点只在本柱有效点中找最多4个最近邻，排除自身，不跨车、跨柱。关系输入包含相对XYZ、平滑距离、两点r/u；另有f的8维query/key。可学习关系分数l_ij=.5*tanh(q_i·k_j/sqrt(8)+MLP(relative_i_j))。

a_ij=masked_softmax_j(l_ij+log(p_j))；m_i=sum_j a_ij*f_j。没有邻居时m_i=f_i。查询汇总权重b_i=masked_softmax_i(log(p_i))；v=sum_i b_i*m_i。关系残差有界，可靠性先验保持主导，但不假定先验已完成天气校准。

每柱8项统计：n/32、平均r、平均u、r方差、覆盖跨度、平均传感器距离、有效观测数/32、平均p。通过共享结构的逐柱MLP产生eta=.5*sigmoid(MLP(stats))，不使用全局可学习标量。

修正delta=tanh(W_out LayerNorm(v-h0))；h=h0+eta*sqrt(mean(h0²)+1e-6)*delta。W_out零初始化，初始残差严格为零；之后可学习。幅值因子停止梯度，避免通过放大主干尺度绕过幅值约束。不添加损失。

## A2：有界几何支持

PointDSC的原约束是两份点云对应边长度差：abs(||x_i-x_j||-||y_i-y_j||)。它依赖跨点云对应关系及共同刚体变换。本任务没有这些对应关系，因此不使用该公式，不做谱匹配，也不将本方法称PointDSC复现。

新定义是柱内局部支持代理。以每点邻域平均距离s_i估计局部间隔，e_ij=s_i+s_j（下界.02米）。

g_ij=exp(-||x_i-x_j||²/e_ij²)*exp(-abs(s_i-s_j)/e_ij)*(.25+.75*clip(cos(f_i,f_j),0,1))。

c_i=sum_j(g_ij*p_j)/sum_j(g_ij)。没有支持时令c_i=p_i，表示未知。g及用于几何判断的p停止梯度，避免从该分支直接通过改变“支持真值”自我迎合。它们仍由当前共享特征及GSPR产生，属于弱代理。

几何修正系数t_i=clip(1+.5*(c_i-p_i),.5,1.5)，用p'_i=p_i*t_i替代A1的权重先验；f_i值保持不变。四种情况单独计数：高可靠/高共识、高可靠/低共识、低可靠/高共识、低可靠/低共识，阈值固定.5，只用于诊断。高低指p=r(1-u)，原r/u分布另存。

这是相对支持均值，没有按邻居数量或到全柱中心距离判噪声。每个点自建局部支持，不拟合单一平面，可允许多个表面和边缘；近邻跨越真实表面、稀疏随机点偶然相互支持仍可能失败。A2不承诺几何真值或去噪正确性。

## A3：证据不足时减弱整体修正

有效观测数N_eff=(sum p)²/sum p²。E=clip((N_eff-2)/4,0,1)*(1-mean(u))*coverage/(coverage+.05+.002*range)*mean(c)。A3采用eta_A3=eta_A2*E。

因此一两点、有效证据极少或几何支持弱时，残差变小；极少点时严格回到原GSPR-PFN结果。远处点间隔本来可能更大，跨度只调节证据充分性，不判定点为噪声。即便r高也不靠孤立点推断异常。E的常数是事先固定的设计值，没有用Snow验证成绩调参。

A3减弱整个新残差，包括A1聚合和A2几何重分配，优先保护h0。A3−A2验证的是这一整体保护策略，不能解释为仅某一个点的噪声判定效果。保护触发定义为E<.25，记录柱数、点数、修正强度及按有效点数分组的分布。

传感器距离由官方collate提供的逐车lidar_pose及ego_flag，使用官方x1_to_x2将各传感器原点变换到ego坐标，再计算点距。缺失/顺序不匹配时失败，不用ego距离冒充逐车距离。评测中的目标距离分层则明确用ego距离。

## 计算和数值

最多32点，分256柱处理。距离搜索O(M*T²)，聚合O(M*T*K*D)，K=4、D=8；新分支训练用PyTorch非重入checkpoint减少保存的中间张量。没有新的CUDA专用算子，ROCm实际执行待验收。额外重算及分布记录会增加耗时，不预先声称轻量高效。

空掩码softmax显式归零；坐标开平方加入1e-6；无点/无支持归约有分母下界；无有效点的范围先移除inf哨兵。无效点只从新计算中屏蔽，不能声称修复了原PFN的全部填充敏感性。

## 参考来源和适配边界

- 本地PointASNL原文第3–5页，公式1–8；源码`研究相关论文/1/PointASNL-master.zip::PointASNL-master/utils/pointasnl_util.py`的SampleWeights、AdaptiveSampling、PointNonLocalCell。原方法有自适应坐标更新和局部/非局部结构；本实现只借鉴点间关系和归一化聚合，未引入其TensorFlow或自定义采样算子。
- PointDSC，CVPR 2021，§3.2公式2、§3.4：[CVF官方原文](https://openaccess.thecvf.com/content/CVPR2021/papers/Bai_PointDSC_Robust_Point_Cloud_Registration_Using_Deep_Spatial_Consistency_CVPR_2021_paper.pdf)。已核对其输入为两份点云的候选对应关系。
- 稳健统计启发体现为有界修正、局部自适应间隔和有效观测数保护。这里没有鲁棒估计的破坏点或一致性证明，不借名称宣称理论保证。
