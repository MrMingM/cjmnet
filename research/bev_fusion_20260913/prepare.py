from harvest import *
from collections import Counter
def main():
    counts=Counter(); manifest=[]
    overrides={
      'UECP':{'doi':'10.1007/978-3-032-37095-2_7'},
      'RSTA':{'publication':'IEEE ICCC 2026（arXiv 注明已接收；卷页[待确认]）','title':'Domain-Generalized Adaptive Semantic Communication for Collaborative Perception','mechanism':'部署前跨域原型对齐和跨信道梯度一致性；部署时仅用高语义相关且高信道保真的 token 做熵最小化适配。'},
      'CoDS':{'publication':'ACM MM 2026（arXiv journal-ref 标注；正式出版[待确认]）；截至检索日使用预印本'},
      'CoBEVMoE':{'publication':'ICRA 2026（arXiv 已确认接收；正式卷页[待确认]）'},
      'BM2CP':{'publication':'CoRL 2023；PMLR 229:1022–1035, 2023'},
      'Systematic-Review':{'doi':'10.1109/TITS.2025.3631141'},
      'CoMamba':{'code':'https://github.com/taco-group/CoMamba'},
    }
    for p in PAPERS:
        p.update(overrides.get(p['alias'],{})); path=CACHE/(p['alias']+'.json'); m=json.loads(path.read_text(encoding='utf8'))
        if p['alias']=='CoInfra':
            p['title']=m['arxiv_meta']['citation_title'][0]; p['publication']='arXiv 2025，最新 v3 更新于 2026-03-20；TR-C 投稿中，非已接收'
        cvf=m.get('cvf',{})
        if cvf:
            p['title']=cvf['title']
            if not p['arxiv'] and cvf.get('arxiv'):
                p['arxiv']=cvf['arxiv'][0].split('/abs/')[-1]
                try:
                    r=get('https://arxiv.org/abs/'+p['arxiv']); s=BeautifulSoup(r.text,'html.parser')
                    metas={}
                    for x in s.select('meta[name^="citation_"]'): metas.setdefault(x['name'],[]).append(x.get('content',''))
                    m['arxiv_meta']=metas; h=s.select_one('.submission-history'); m['history']=h.get_text(' ',strip=True) if h else ''
                    c=s.select_one('.comments'); m['comments']=c.get_text(' ',strip=True) if c else ''
                    m['pdfs'].append({'url':'https://arxiv.org/pdf/'+p['arxiv'],'kind':'arxiv_latest'})
                except Exception as e: m['errors'].append('extra_arxiv:'+str(e))
            if not p['code'] and cvf.get('code'): p['code']=cvf['code'][0]
        if p['doi'] and not m.get('crossref'):
            try:
                cr=get('https://api.crossref.org/works/'+quote(p['doi'],safe='/')).json()['message']; m['crossref']=cr
            except Exception as e: m['errors'].append('extra_doi:'+str(e))
        cr=m.get('crossref',{})
        if cr and p['alias']=='Systematic-Review':
            p['publication']='IEEE T-ITS '+str(cr.get('volume','[待确认]'))+'('+str(cr.get('issue','[待确认]'))+'):'+str(cr.get('page','[待确认]'))+', 2026；online 2025'
        authors=m.get('arxiv_meta',{}).get('citation_author',[])
        if not authors and cr: authors=[(x.get('given','')+' '+x.get('family','')).strip() for x in cr.get('author',[])]
        if cvf:
            a=re.search(r'author\s*=\s*\{([^}]+)',cvf.get('text',''))
            if a: authors=a.group(1).split(' and ')
        p['authors']=authors
        p['history']=m.get('history',''); p['comments']=m.get('comments','')
        versions=re.findall(r'\[v(\d+)\]',p['history']); latest=max(map(int,versions)) if versions else None
        p['arxiv_version']=latest
        candidates=[]
        if p['alias']=='UECP': candidates.append({'url':'https://link.springer.com/content/pdf/10.1007/978-3-032-37095-2_7.pdf','kind':'publisher'})
        if p['alias']=='TransIFF': candidates.append({'url':p['pdf'],'kind':'formal_open_access'})
        for x in m['pdfs']:
            u=x['url']
            if 'arxiv.org/pdf/' in u and latest: u=re.sub(r'v\d+(?:\.pdf)?$','',u).removesuffix('.pdf')+'v'+str(latest)
            candidates.append(dict(x,url=u))
        if p['alias']=='Systematic-Review' and cr:
            u=cr.get('resource',{}).get('primary',{}).get('URL',''); doc=re.search(r'document/(\d+)',u)
            if doc:candidates.append({'url':'https://ieeexplore.ieee.org/stampPDF/getPDF.jsp?tp=&arnumber='+doc.group(1),'kind':'publisher'})
        seen=set(); p['candidates']=[]
        for x in candidates:
            if x['url'] not in seen: p['candidates'].append(x); seen.add(x['url'])
        sources=[]
        if cvf: sources.append({'label':'正式论文页','url':cvf['url']})
        if p['doi']: sources.append({'label':'DOI','url':'https://doi.org/'+p['doi']})
        if p['arxiv']: sources.append({'label':'arXiv','url':'https://arxiv.org/abs/'+p['arxiv']})
        if not sources and p['pdf']: sources.append({'label':'原文','url':p['pdf']})
        p['sources']=sources
        if p['code']:
            if p['code']!=m.get('code_final_url'):
                try:
                    r=get(p['code']); s=BeautifulSoup(r.text,'html.parser'); article=s.select_one('article')
                    m['code_final_url']=r.url; m['code_tree_links']=list(dict.fromkeys(urljoin(r.url,a['href']) for a in s.select('a[href]') if '/tree/' in a['href'] or '/blob/' in a['href']))
                    m['readme']=article.get_text(' ',strip=True) if article else s.get_text(' ',strip=True)
                except Exception as e:m['errors'].append('extra_code:'+str(e))
            paths=m.get('code_tree_links',[])
            implementation=[u for u in paths if any(t in u.lower() for t in ['/opencood','/coperception','/models','/src','/tools','/main','/projects','/belt_fusion','.py']) and not u.lower().endswith(('/readme.md','/license'))]
            if p['alias']=='Systematic-Review': state='公开文献索引（非模型代码）'
            elif p['alias'] in ['ER-CoPe','CoDS']: state='未公开实现：当前仅 README'
            elif 'repository is empty' in m.get('readme','').lower():state='未公开实现：空仓库'
            elif implementation: state='已公开代码（未运行验证）'
            elif paths: state='公开仓库；实现完整性[待确认]'
            else:state='代码状态[待确认]'
            p['code_status']=state
        else: p['code_status']='未找到官方公开实现[待确认]'
        p['code_evidence']=m.get('code_tree_links',[])
        counts[p['cat']]+=1; p['id']=f"{int(p['cat']):02d}_{counts[p['cat']]:02d}"
        year=re.search(r'\b20(?:19|2[0-6])\b',p['publication']); year=year.group(0) if year else 'year-pending'
        p['filename']=p['id']+'_'+p['alias']+'_'+year+'.pdf'
        path.write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf8')
        manifest.append(p);print(p['id'],p['alias'],len(p['candidates']),flush=True)
    (BASE/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
    print('PREPARED',len(manifest),flush=True)
if __name__=='__main__':main()
