import json,re,csv,collections
from pathlib import Path
BASE=Path(__file__).resolve().parent
papers=json.loads((BASE/'manifest.json').read_text(encoding='utf8'))
outcomes=json.loads((BASE/'download_results.json').read_text(encoding='utf8')) if (BASE/'download_results.json').exists() else []
results={x['id']:x for x in outcomes}
CATS=[
('一、天气感知与跨模态抗退化融合类','共同特点是直接讨论恶劣天气造成的观测退化，通过去噪、域泛化或互补传感器改善协作感知。这里同时区分接收端融合与通信前去噪，避免把前端收益算成融合收益。'),
('二、可靠性、不确定性与自适应信任融合类','共同特点是显式估计质量、不确定性或可信度，再决定合作特征的权重或保留程度。需要特别检查不确定性的监督来源：密度代理、检测后验、贝叶斯模型和天气回波质量并不等价。'),
('三、供需互补、稀疏消息与实例交互融合类','共同特点是围绕接收方所缺的信息，选择并组织局部、实例或稀疏协作特征，减少无关与重复信息。许多方法联合改变发送策略和融合，与你比较时需固定通信条件或拆开消融。'),
('四、空间注意力、图网络与高效基础融合类','共同特点是对齐到共同坐标后，通过最大值、卷积、图消息、注意力或状态空间模型聚合跨车 BEV 表征。这些方法适合作为结构基线，但通常没有显式区分天气噪声、观测不足和任务需求。'),
('五、空间与域对齐后融合类','共同特点是在融合前或融合内部校正跨车坐标、特征域或时空对应关系，避免把不同位置的证据直接相加。对齐质量与观测质量应分别估计，否则门控可能错误拒绝原本有用的邻居信息。'),
('六、时序、异步与通信中断融合类','共同特点是使用历史特征、运动流或时空交互，将不同时间到达的信息变成可融合表征。它们解决消息新鲜度与缺失问题，不能直接推断为雨、雾、雪下的传感可靠性方法。'),
('七、特征重建、去噪扩散与多尺度恢复融合类','共同特点是先恢复或重建被压缩、稀疏化、丢失或污染的协作信息，再送入检测融合。迁移时应区分真实观测证据和模型生成的补全，并计入解码与多步推理代价。'),
('八、异构模态、公共表征与专家路由融合类','共同特点是处理不同模态、模型或智能体之间的表征差异，通过共享空间、适配器或动态专家实现交互。对你的同构 LiDAR 管线，优先迁移融合组件，再考虑完整异构系统。'),
('九、单车多模态质量融合的可迁移参考类','共同特点是在单车或单路侧系统中，把多传感器投影到公共表示，并通过质量或互补性进行融合。这些论文提供机制启发，但不是 V2X 通信后融合工作的直接证据。'),
('十、证据冲突与全体不可信条件下的迁移参考类','共同特点是显式面对不可信或冲突的合作信息，研究证据组合或稳健决策。本类仅保留与当前问题联系最紧的 2 篇：一个是决策级融合、一个是攻击防御；数量少源于本报告的适用性筛选和与第二类避免重复，并不表示整个可信协作方向空白。'),
('十一、恶劣天气评测、鲁棒性基准与综述资源类','共同特点是提供可验证的退化条件、公开数据或文献分类，帮助检验融合方法是否超出单一数据集和理想通信设置。这里明确区分 LiDAR 天气退化、相机 corruption 和真实路侧天气数据。')]
def esc(s):return str(s).replace('|','\\|').replace('\n',' ')
def link(label,url):return f'[{label}]({url})'
def source_kind(r):
    u=r.get('source_url',''); k=r.get('source_kind','')
    if k.startswith('arxiv'):return 'arXiv 主文'
    if k=='author_manuscript':return '作者公开稿'
    if any(x in u for x in ['openaccess.thecvf','ecva.net','proceedings.','papers.nips','openreview.net','ojs.aaai']):return '会议公开主文'
    if any(x in u for x in ['faculty.csu','par.nsf','ris.utwente','bibliothek.kit']):return '机构存档主文'
    if 'ieeexplore' in u:return 'IEEE 主文'
    return '作者/机构公开主文'
ok=sum(x['status']=='ok' for x in outcomes); failed=[x for x in outcomes if x['status']!='ok']
intro=(BASE/'analysis.md').read_text(encoding='utf8')
summary=f'''\n### 文献与下载概况\n\n共 **{len(papers)} 篇去重论文/资源，11 类**；已验证下载 **{ok} 篇**，未成功 **{len(failed)} 篇**。文件名以 `大类编号_类内序号` 开头，例如 `02_01_UECP_2026.pdf` 对应“二”类第 1 篇；同一篇不重复保存正式版和 arXiv 版。\n\n- [下载失败汇总](下载失败汇总.md)：逐篇原因、尝试链接与后续获取入口。\n- [论文元数据与下载清单](论文元数据与下载清单.json)：完整作者、arXiv 版本、来源与候选下载地址。\n- [下载校验记录](下载校验记录.json)：实际地址、文件大小、页数、SHA-256 和失败记录。\n\n| 大类 | 篇数 | 下载成功 |\n|---|---:|---:|\n'''
for i,(name,_) in enumerate(CATS,1):
    ps=[p for p in papers if int(p['cat'])==i]; n=sum(results.get(p['id'],{}).get('status')=='ok' for p in ps)
    summary+=f'| {name} | {len(ps)} | {n} |\n'
parts=[intro,summary]
for i,(name,desc) in enumerate(CATS,1):
    parts.append(f'\n## {name}\n\n{desc}\n\n| 编号、论文（英文原题）与作者 | 发表/版本信息 | 融合机制 | 与当前 GSPR/BEV 通信的关系和证据边界 | 是否开源 | 主文下载与本地文件 |\n|---|---|---|---|---|---|\n')
    for p in papers:
        if int(p['cat'])!=i:continue
        r=results.get(p['id'],{}); sources=p['sources'];url=sources[0]['url'] if sources else p['candidates'][0]['url']
        authors=p['authors']; author=authors[0]+(' et al.' if len(authors)>1 else '') if authors else '作者[待确认]'
        title=f"**{p['id']} · {p['alias']}**<br>{link(p['title'],url)}<br>{author}"
        pub=p['publication']+'<br>'+' / '.join(link(x['label'],x['url']) for x in sources if x['label']!='arXiv')
        if p['arxiv']:
            av=p['arxiv']+(('v'+str(p['arxiv_version'])) if p.get('arxiv_version') else '')
            pub+='<br>'+link('arXiv:'+av,'https://arxiv.org/abs/'+av)
        else:pub+='<br>对应 arXiv：未定位[待确认]'
        code=p['code_status']+('<br>'+link('作者仓库/项目',p['code']) if p['code'] else '')
        if r.get('status')=='ok':
            dl=link('下载主文',r['source_url'])+' · '+source_kind(r)+'<br>'+link(p['filename'],p['filename'])+f"<br>{r['pages']} 页，{r['bytes']/1024/1024:.2f} MiB"
        else:
            dl=link('尝试下载',p['candidates'][0]['url'])+'<br>**未成功**；见失败汇总<br>预定：`'+p['filename']+'`'
        parts.append('| '+' | '.join(esc(x) for x in [title,pub,p['mechanism'],p['fit'],code,dl])+' |\n')
parts.append('\n### 检索使用的具体关键词组合\n\n下面记录本次实际提交的英文检索式（去除完全重复项），包括广域发现、方法定向、发表核对和全文/代码查找。含 `site:` 的是外部搜索引擎站点限定，不表示已穷尽站内数据库；逐篇 DOI、arXiv 和作者仓库直接访问另见清单。\n\n| 序号 | 实际关键词组合 |\n|---:|---|\n')
log=json.loads((BASE/'search_keywords.json').read_text(encoding='utf8'));queries=list(dict.fromkeys(q['q'] if isinstance(q,dict) else q for q in log['queries']))
for i,q in enumerate(queries,1):parts.append(f'| {i} | `{esc(q)}` |\n')
parts.append('\n检索索引入口：[协作感知系统综述配套索引](https://github.com/leiwanrobotics/awesome-collaborative-perception)。表内每篇的题名链接、正式来源、arXiv 和作者代码均为进一步核对入口；检索结果并非该领域全部英文论文的穷尽清单。\n')
(BASE/'BEV通信后融合_恶劣天气鲁棒性论文检索_20260913.md').write_text(''.join(parts),encoding='utf8')
fail=['# 论文下载失败汇总\n\n','检查日期：2026-09-13。仅将通过 PDF 格式、页数和首页题名检查的文件记为成功。主文未取得的条目保留原编号；没有创建伪 PDF，也未用其他论文或旧版替代。\n\n',f'共 {len(papers)} 篇，成功 {ok} 篇，未成功 {len(failed)} 篇。\n\n','| 编号 | 论文 | 失败原因 | 尝试过的链接 | 建议获取方式 |\n|---|---|---|---|---|\n']
byid={p['id']:p for p in papers}
for r in failed:
    p=byid[r['id']];errors=list(dict.fromkeys(a.get('error','未知') for a in r['attempts'] if a.get('outcome')=='failed'))
    links=list(dict.fromkeys(a['url'] for a in r['attempts']))
    reason='<br>'.join(e.split(' for url:')[0] for e in errors)
    source=p['sources'][0] if p['sources'] else {'label':'论文页','url':p['candidates'][0]['url']}
    guidance='通过机构订阅或作者公开稿获取；'+link(source['label'],source['url'])
    if p['alias']=='SCOPE-plus':guidance+='。需要 SCOPE++ 最新期刊主文，不能用 ICCV 2023 SCOPE 替代。'
    fail.append('| '+' | '.join(esc(x) for x in [r['id'],p['title'],reason,'<br>'.join(link(str(i+1),u) for i,u in enumerate(links)),guidance])+' |\n')
fail.append('\n### 正式排版版未取得、但同一工作公开主文已取得\n\n以下不计为“无全文失败”，因为已保存同一工作的 arXiv 主文；如引用，请使用主报告列出的正式发表信息。\n\n| 编号 | 方法 | 保存版本 | 正式发表信息 |\n|---|---|---|---|\n')
for p in papers:
    r=results.get(p['id'],{})
    if r.get('status')=='ok' and r.get('source_kind','').startswith('arxiv') and not p['publication'].startswith('arXiv'):
        fail.append(f"| {p['id']} | {p['alias']} | {link(p['filename'],p['filename'])} | {p['publication']} |\n")
fail.append('\n完整 HTTP 返回、重试记录、实际最终 URL 和文件校验值见 [下载校验记录](下载校验记录.json)。部分失败只代表当前环境无法公开获取全文，并不表示论文没有正式发表。\n')
(BASE/'下载失败汇总.md').write_text(''.join(fail),encoding='utf8')
print('REPORT',len(papers),'OK',ok,'FAILED',len(failed),'QUERIES',len(queries))
