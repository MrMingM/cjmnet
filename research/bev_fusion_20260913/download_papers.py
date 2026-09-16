"""Download the reviewed manifest sequentially; never execute downloaded content."""
import argparse,hashlib,io,json,re,sys,time,logging
from pathlib import Path
from datetime import datetime
import requests
from pypdf import PdfReader
logging.getLogger('pypdf').setLevel(logging.ERROR)
BASE=Path(__file__).resolve().parent
def tokens(s):
    return set(re.findall(r'[a-z0-9]{3,}',s.lower()))-{'the','and','for','with','via','based','from','under','into','using','multi','agent'}
def validate(data,p):
    if not data[:1024].lstrip().startswith(b'%PDF-'): raise ValueError('响应不是 PDF（可能是登录页、反爬页面或失效链接）')
    doc=PdfReader(io.BytesIO(data)); n=len(doc.pages)
    if n<3: raise ValueError('页数少于3，需人工确认是否为主文')
    text=doc.pages[0].extract_text() or ''
    expected=tokens(p['title']); actual=tokens(text[:6500]); overlap=len(expected&actual)/max(1,len(expected))
    # Compare title tokens on the first page, not references on later pages.
    if overlap<.65: raise ValueError('首页题名不匹配，词覆盖率 '+str(round(overlap,3))+'；'+text[:160].replace('\n',' '))
    if re.search(r'supplementary\s+material|supplemental\s+material',text[:600],re.I): raise ValueError('下载到补充材料，未将其作为主文')
    # Traverse the page tree and parse every page object.
    for page in doc.pages: _=page.mediabox
    return {'pages':n,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'title_token_coverage':round(overlap,4),'first_page':text[:14000]}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--target',required=True);ap.add_argument('--retry-failed',action='store_true');ap.add_argument('--prefer-formal',action='store_true');args=ap.parse_args()
    target=Path(args.target).resolve(); target.mkdir(parents=True,exist_ok=True)
    papers=json.loads((BASE/'manifest.json').read_text(encoding='utf8'))
    logpath=BASE/'download_results.json'; prior=json.loads(logpath.read_text(encoding='utf8')) if logpath.exists() else []
    old={x['id']:x for x in prior}; results=[]; session=requests.Session();session.headers['User-Agent']='Mozilla/5.0 (compatible; AcademicLiteratureDownload/1.0)'
    for p in papers:
        dest=target/p['filename']; previous=old.get(p['id'])
        upgrade=False
        if previous and previous['status']=='ok' and dest.exists():
            if hashlib.sha256(dest.read_bytes()).hexdigest()==previous['sha256']:
                upgrade=args.prefer_formal and previous.get('source_kind','').startswith('arxiv') and p['candidates'][0]['kind']=='formal_open_access'
                if not upgrade:
                    results.append(previous); print(p['id'],'SKIP verified',flush=True);continue
        out={'id':p['id'],'alias':p['alias'],'title':p['title'],'filename':p['filename'],'status':'failed','attempts':list(previous.get('attempts',[])) if previous else [],'checked_at':datetime.now().isoformat(timespec='seconds')}
        for candidate in (p['candidates'][:1] if upgrade else p['candidates']):
            u=candidate['url'];attempt={'url':u,'kind':candidate['kind']}
            try:
                headers=candidate.get('headers',{})
                r=session.get(u,timeout=(15,45),headers=headers); attempt['http_status']=r.status_code;attempt['final_url']=r.url;r.raise_for_status()
                data=r.content; checked=validate(data,p)
                if dest.exists():
                    existing_hash=hashlib.sha256(dest.read_bytes()).hexdigest()
                    if existing_hash!=checked['sha256']:
                        if upgrade and existing_hash==previous['sha256']: dest.write_bytes(data)
                        else: raise FileExistsError('目标文件已存在且内容不同；保留原文件')
                else:
                    with dest.open('xb') as f:f.write(data)
                first=checked.pop('first_page'); (BASE/'metadata'/(p['alias']+'_firstpage.txt')).write_text(first,encoding='utf8')
                out.update(checked,status='ok',source_url=r.url,source_kind=candidate['kind']); attempt['outcome']='ok';out['attempts'].append(attempt);break
            except Exception as e:
                attempt['outcome']='failed';attempt['error']=str(e);out['attempts'].append(attempt)
                print(p['id'],'attempt failed',str(e)[:160],flush=True)
        if upgrade and out['status']!='ok':
            previous['attempts']=out['attempts'];out=previous
        results.append(out)
        # Save all older outcomes too, so an interrupted retry remains resumable.
        updated={x['id']:x for x in prior};updated.update({x['id']:x for x in results})
        logpath.write_text(json.dumps([updated[x['id']] for x in papers if x['id'] in updated],ensure_ascii=False,indent=2),encoding='utf8')
        print(p['id'],p['alias'],out['status'],out.get('pages',''),out.get('bytes',''),flush=True)
        time.sleep(.3)
    print('FINISHED',sum(x['status']=='ok' for x in results),'/',len(results),flush=True)
if __name__=='__main__':main()
