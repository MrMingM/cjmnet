import json,hashlib
from pathlib import Path
import requests
from download_papers import validate,BASE
target=Path(r'D:\Collaborative perception\mymodule\OpenCOOD-main\论文\融合相关论文')
mp=BASE/'manifest.json';rp=BASE/'download_results.json';ps=json.loads(mp.read_text(encoding='utf8'));rs=json.loads(rp.read_text(encoding='utf8'))
p=next(x for x in ps if x['alias']=='V2X-Survey-PIEEE');o=next(x for x in rs if x['id']==p['id']);dest=target/p['filename']
assert hashlib.sha256(dest.read_bytes()).hexdigest()==o['sha256'], 'Existing file changed; do not overwrite'
url='https://arxiv.org/pdf/2310.03525v5';r=requests.get(url,timeout=(15,60));r.raise_for_status();checked=validate(r.content,p)
first=checked.pop('first_page');assert 'Huang' in first[:2500], 'Author mismatch'
old=o['source_url'];o['attempts'].append({'url':old,'outcome':'rejected_on_manual_review','error':'不同作者的另一篇综述；人工题名和作者复核后拒绝，已替换为正确论文'})
dest.write_bytes(r.content)
o.update(checked,status='ok',source_url=r.url,source_kind='arxiv_latest');o['attempts'].append({'url':url,'final_url':r.url,'kind':'arxiv_latest','outcome':'ok','http_status':r.status_code})
p['arxiv']='2310.03525';p['arxiv_version']=5;p['history']='Latest v5: 2025-09-02';p['pdf']='';p['sources'].append({'label':'arXiv','url':'https://arxiv.org/abs/2310.03525v5'})
p['candidates']=[x for x in p['candidates'] if x['url']!=old]+[{'url':url,'kind':'arxiv_latest'}]
(BASE/'metadata'/(p['alias']+'_firstpage.txt')).write_text(first,encoding='utf8')
mp.write_text(json.dumps(ps,ensure_ascii=False,indent=2),encoding='utf8');rp.write_text(json.dumps(rs,ensure_ascii=False,indent=2),encoding='utf8')
print('Correct survey verified',o['pages'],o['bytes'])
