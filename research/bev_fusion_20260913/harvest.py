import csv,json,re,time,concurrent.futures,difflib
from pathlib import Path
from urllib.parse import urljoin,quote
import requests
from bs4 import BeautifulSoup

BASE=Path(__file__).resolve().parent
PAPERS=list(csv.DictReader((BASE/'papers.tsv').open(encoding='utf-8'),delimiter='\t'))
CACHE=BASE/'metadata'; CACHE.mkdir(exist_ok=True)
def get(url):
    r=requests.get(url,timeout=(12,35),headers={'User-Agent':'Mozilla/5.0 (compatible; literature-research/1.0)'})
    r.raise_for_status(); return r
def norm(s): return re.sub(r'[^a-z0-9]','',s.lower())
def run(p):
    path=CACHE/(p['alias']+'.json')
    if path.exists(): return json.loads(path.read_text(encoding='utf8'))
    out={'alias':p['alias'],'errors':[],'pdfs':[]}
    if p['pdf']: out['pdfs'].append({'url':p['pdf'],'kind':'publisher_or_author'})
    if p['arxiv']:
        url='https://arxiv.org/abs/'+p['arxiv']
        try:
            r=get(url); soup=BeautifulSoup(r.text,'html.parser')
            (CACHE/(p['alias']+'_arxiv.html')).write_text(r.text,encoding='utf8')
            metas={}
            for m in soup.find_all('meta'):
                key=m.get('name',''); val=m.get('content','')
                if key.startswith('citation_'): metas.setdefault(key,[]).append(val)
            out['arxiv_meta']=metas
            history=soup.select_one('.submission-history'); out['history']=history.get_text(' ',strip=True) if history else ''
            comments=soup.select_one('.comments'); out['comments']=comments.get_text(' ',strip=True) if comments else ''
            abstract=soup.select_one('.abstract'); out['abstract']=abstract.get_text(' ',strip=True) if abstract else ''
            j=soup.select_one('.jref'); out['journal_ref']=j.get_text(' ',strip=True) if j else ''
            for x in metas.get('citation_pdf_url',[]): out['pdfs'].append({'url':x,'kind':'arxiv_latest'})
            if not metas.get('citation_pdf_url'): out['pdfs'].append({'url':'https://arxiv.org/pdf/'+p['arxiv'],'kind':'arxiv_latest'})
            out['arxiv_links']=[a.get('href') for a in soup.select('a[href]') if any(x in a.get('href','') for x in ['github.com','doi.org/10.1109','openaccess.thecvf','openreview.net','link.springer'])]
        except Exception as e:
            out['errors'].append('arxiv: '+str(e)); out['pdfs'].append({'url':'https://arxiv.org/pdf/'+p['arxiv'],'kind':'arxiv_latest_unpinned'})
    if p['doi']:
        try:
            msg=get('https://api.crossref.org/works/'+quote(p['doi'],safe='/')).json()['message']
            out['crossref']={k:msg.get(k) for k in ['title','author','container-title','published','published-online','published-print','volume','issue','page','article-number','link','resource','URL']}
            for l in msg.get('link',[]):
                if l.get('content-type')=='application/pdf': out['pdfs'].append({'url':l['URL'],'kind':'publisher'})
        except Exception as e: out['errors'].append('crossref: '+str(e))
    if p['code']:
        try:
            r=get(p['code']); soup=BeautifulSoup(r.text,'html.parser')
            out['code_final_url']=r.url
            out['code_title']=soup.title.get_text() if soup.title else ''
            links=list(dict.fromkeys(urljoin(r.url,a['href']) for a in soup.select('a[href]') if '/blob/' in a['href'] or '/tree/' in a['href']))
            out['code_tree_links']=links[:120]
            article=soup.select_one('article'); out['readme']=(article.get_text(' ',strip=True) if article else soup.get_text(' ',strip=True))[:16000]
            out['code_pdf_links']=[urljoin(r.url,a['href']) for a in soup.select('a[href]') if '.pdf' in a['href'].lower()]
            out['code_outlinks']=[urljoin(r.url,a['href']) for a in soup.select('a[href]') if any(t in a['href'] for t in ['github.com/','arxiv.org/','openaccess.thecvf.com/'])][-60:]
        except Exception as e: out['errors'].append('code: '+str(e))
    path.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8')
    print(p['alias'], 'done',len(out['errors']),flush=True); return out

if __name__=='__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool: results=list(pool.map(run,PAPERS))
    (BASE/'metadata.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf8')
    print('TOTAL',len(results),flush=True)
