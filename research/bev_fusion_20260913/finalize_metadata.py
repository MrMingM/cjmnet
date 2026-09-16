import json,re
from pathlib import Path
BASE=Path(__file__).resolve().parent
p=BASE/'manifest.json';papers=json.loads(p.read_text(encoding='utf8'))
authorfix={
'How2comm':['Dingkang Yang','Kun Yang','Yuzheng Wang','Jing Liu','Zhi Xu','Rongbin Yin','Peng Zhai','Lihua Zhang'],
'TransIFF':['Ziming Chen','Yifeng Shi','Jinrang Jia'],
'NEAT':['Kun Yang','Dingkang Yang','Ke Li','Dongling Xiao','Zedian Shao','Peng Sun','Liang Song']}
for x in papers:
    if x['alias'] in authorfix:x['authors']=authorfix[x['alias']]
    if x['alias']=='Co-Denoising':
        x['mechanism']='距离-强度采样和 PFN 特征细化先粗去噪；接收端以 VAE 潜变量表示和可学习协作边权实现概率融合。'
        x['fit']='A｜直接天气：LISA 模拟 OPV2V/DAIR-V2X 雨；与 GSPR 前端及通信后融合最接近，但文中简化了位姿通信延迟，不能据此断言异步鲁棒。'
        x['sources'].append({'label':'作者附件页','url':'https://faculty.csu.edu.cn/dengxiaoheng/en/lwcg/10453/content/60083.htm'})
    if x['alias']=='CoDS':x['code_status']='GitHub 仅 README；另一 OpenI 镜像可用性[待确认]'
    if x['alias']=='Reliable-Heterogeneous':
        x['mechanism']='稀疏度感知主动请求、前景传输；中间协作与单车检测并行，以双流多层融合补偿稀疏接收信息。'
        x['fit']='A｜OPV2V-H/DAIR-V2X 低带宽异构协作；保留本地 proposal 很契合，未核实雨雾雪专门实验。'
    if x['alias']=='ICPB':
        x['fit']='A｜针对通信退化、参与者退出和异步，平衡单车/协作能力；天气专门实验[待确认]。'
    if x['alias']=='UECP':
        x['publication']='ECCV 2026, LNCS 17061:111–129；2026-09-09 online'
        x['fit']='A｜DAIR-V2X/V2V4Real 接收融合；密度衍生不确定性不等于天气回波可靠性，建议分别测试密度和 GSPR q/u；天气专门验证未确认。'
    if x['alias']=='CoInfra':
        x['fit']='D｜真实雨、暴雪、冻雨主要在 I2I 子集；V2I 子集目前晴天。不能把 I2I 天气结果直接视为 V2I 天气评测。'
    if x['alias']=='DSRC':
        x['title']='DSRC: Learning Density-Insensitive and Semantic-Aware Collaborative Representation Against Corruptions'
        x['publication']='AAAI 2025, 39(9):9942–9950';x['doi']='10.1609/aaai.v39i9.33078'
        x['sources'].insert(0,{'label':'正式论文页','url':'https://ojs.aaai.org/index.php/AAAI/article/view/33078'})
    if x['alias']=='V2X-ViT':x['publication']='ECCV 2022:107–124'
    if x['alias']=='SyncNet':x['publication']='ECCV 2022:316–332'
    if x['alias']=='TransIFF':x['sources'].insert(0,{'label':'正式论文页','url':x['pdf'].replace('/papers/','/html/').replace('.pdf','.html')})
    if x['alias']=='Intermediate-Fusion-Survey':x['code_status']='综述；算法代码不适用'
    if x['alias']=='V2X-Survey-PIEEE':x['code_status']='综述；算法代码不适用'
p.write_text(json.dumps(papers,ensure_ascii=False,indent=2),encoding='utf8')
print('metadata finalized',len(papers))
