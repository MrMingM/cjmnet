from harvest import *
def readmeta(p): return json.loads((CACHE/(p['alias']+'.json')).read_text(encoding='utf8'))
def save(p,m): (CACHE/(p['alias']+'.json')).write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf8')
if __name__=='__main__':
    indices={}
    for conf in ['CVPR2023','CVPR2024','CVPR2025','CVPR2026','ICCV2023','ICCV2025','WACV2026']:
        try:
            url='https://openaccess.thecvf.com/'+conf+'?day=all'; r=get(url); soup=BeautifulSoup(r.text,'html.parser')
            indices[conf]=[(a.get_text(' ',strip=True),urljoin(url,a['href'])) for a in soup.select('dt.ptitle a[href]')]
            print(conf,len(indices[conf]),flush=True)
        except Exception as e: print(conf,str(e),flush=True)
    for p in PAPERS:
        m=readmeta(p)
        matches=[]
        for conf,entries in indices.items():
            if conf.replace('20',' 20') not in p['publication']: continue
            for title,url in entries:
                ratio=difflib.SequenceMatcher(None,norm(title),norm(p['title'])).ratio()
                matches.append((ratio,title,url))
        if matches:
            ratio,title,url=max(matches)
            if ratio>.77:
                try:
                    r=get(url); soup=BeautifulSoup(r.text,'html.parser')
                    m['cvf']={'title':title,'url':url,'similarity':ratio,'text':soup.get_text(' ',strip=True)}
                    links=[urljoin(url,a['href']) for a in soup.select('a[href]')]
                    pdfs=[x for x in links if '/papers/' in x and x.endswith('.pdf')]
                    m['pdfs']=[{'url':x,'kind':'formal_open_access'} for x in pdfs]+m['pdfs']
                    m['cvf']['arxiv']=[x for x in links if 'arxiv.org/abs/' in x]
                    m['cvf']['code']=[x for x in links if 'github.com/' in x]
                    print(p['alias'],'CVF',round(ratio,3),title,flush=True)
                except Exception as e: m['errors'].append('CVF: '+str(e))
        cr=m.get('crossref',{})
        primary=cr.get('resource') or {}; primary=(primary.get('primary') or {}).get('URL','')
        if 'ieeexplore.ieee.org/document/' in primary:
            doc=re.search(r'document/(\d+)',primary).group(1)
            m['pdfs'].append({'url':'https://ieeexplore.ieee.org/stampPDF/getPDF.jsp?tp=&arnumber='+doc,'kind':'publisher'})
        if p['doi'].startswith('10.1145/'):
            m['pdfs'].append({'url':'https://dl.acm.org/doi/pdf/'+p['doi'],'kind':'publisher'})
        # Only follow author repository PDF links for this work, excluding supplements.
        for u in m.get('code_pdf_links',[]):
            if 'supplement' not in u.lower() and norm(p['alias']).replace('plus','') in norm(u):
                if 'github.com/' in u and '/blob/' in u: u=u.replace('github.com/','raw.githubusercontent.com/').replace('/blob/','/')
                m['pdfs'].append({'url':u,'kind':'author_repository'})
        if p['alias']=='Adaptive-Weighting':
            try:
                r=get(p['pdf']); soup=BeautifulSoup(r.text,'html.parser')
                m['author_page']=soup.get_text(' ',strip=True)[:24000]
                links=list(dict.fromkeys(urljoin(r.url,a['href']) for a in soup.select('a[href]') if '.pdf' in a['href'].lower()))
                m['pdfs']=[{'url':u,'kind':'author_manuscript'} for u in links]+m['pdfs']
            except Exception as e: m['errors'].append('author: '+str(e))
        save(p,m)
    (BASE/'metadata.json').write_text(json.dumps([readmeta(p) for p in PAPERS],ensure_ascii=False,indent=2),encoding='utf8')
