import json,re,concurrent.futures
from harvest import BASE,CACHE,get,BeautifulSoup,urljoin,quote
def main():
    path=BASE/'manifest.json';ps=json.loads(path.read_text(encoding='utf8'))
    codes={'AFFormer':'https://github.com/zoeyzhouxi/AFFormer','FocalComm':'https://github.com/scdrand23/FocalComm','V2VAM-LCRN':'https://github.com/jinlong17/V2VLC','UncertainBEV':'https://github.com/NUST-Machine-Intelligence-Laboratory/UncertainBEV'}
    for p in ps:
        if p['alias'] in codes:
            p['code']=codes[p['alias']]
            try:
                r=get(p['code']);s=BeautifulSoup(r.text,'html.parser');links=list(dict.fromkeys(urljoin(r.url,a['href']) for a in s.select('a[href]') if '/tree/' in a['href'] or '/blob/' in a['href']))
                p['code_evidence']=links
                impl=[u for u in links if '/tree/' in u or u.endswith('.py')]
                p['code_status']='已公开代码（未运行验证）' if impl else '未公开实现：当前仅 README'
                print(p['alias'],p['code_status'],links[:8],flush=True)
            except Exception as e:p['code_status']='论文给出代码链接；当前可访问性[待确认]';print(str(e),flush=True)
        if p['alias']=='ICPB':
            p['doi']='10.1016/j.eswa.2026.131311';p['sources'].insert(0,{'label':'DOI','url':'https://doi.org/'+p['doi']})
            try:
                cr=get('https://api.crossref.org/works/'+p['doi']).json()['message'];p['authors']=[(a.get('given','')+' '+a.get('family','')).strip() for a in cr['author']]
                (CACHE/'ICPB_crossref.json').write_text(json.dumps(cr,ensure_ascii=False,indent=2),encoding='utf8')
            except Exception as e:print('ICPB Crossref',str(e),flush=True)
    failed={'ER-CoPe','ICPB','Reliable-Heterogeneous','What2comm','CRCNet','CoGMoE','Fusion2comm','CoRange','SCOPE-plus','UncertainBEV','BELT-Fusion'}
    def oa(p):
        if p['alias'] not in failed or not p['doi']: return p
        try:
            r=get('https://api.openalex.org/works/https://doi.org/'+p['doi']);j=r.json()
            (CACHE/(p['alias']+'_oa.json')).write_text(json.dumps(j,ensure_ascii=False,indent=2),encoding='utf8')
            found=[]
            for loc in j.get('locations',[]):
                if loc.get('pdf_url') and loc.get('is_oa'):
                    u=loc['pdf_url'];found.append(u)
                    if not any(x['url']==u for x in p['candidates']):p['candidates'].append({'url':u,'kind':'oa_repository'})
            print(p['alias'],'OA',found,flush=True)
        except Exception as e:print(p['alias'],'OA check',str(e)[:100],flush=True)
        return p
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool: ps=list(pool.map(oa,ps))
    path.write_text(json.dumps(ps,ensure_ascii=False,indent=2),encoding='utf8')
if __name__=='__main__':main()
