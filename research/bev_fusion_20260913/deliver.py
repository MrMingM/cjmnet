import argparse,hashlib,json,re,shutil,runpy
from pathlib import Path
from pypdf import PdfReader
BASE=Path(__file__).resolve().parent
TARGET=Path(r'D:\Collaborative perception\mymodule\OpenCOOD-main\论文\融合相关论文').resolve()
def main():
    a=argparse.ArgumentParser();a.add_argument('--deliver',action='store_true');args=a.parse_args()
    qf=BASE/'search_keywords.json';q=json.loads(qf.read_text(encoding='utf8'))
    for s in ['"Vehicle-to-Everything Cooperative Perception for Autonomous Driving" pdf','"Cooperative Perception for Automated Driving" "Survey of Algorithms"']:
        if not any(x['q']==s for x in q['queries']):q['queries'].append({'q':s})
    qf.write_text(json.dumps(q,ensure_ascii=False,indent=2),encoding='utf8')
    runpy.run_path(str(BASE/'build_report.py'))
    ps=json.loads((BASE/'manifest.json').read_text(encoding='utf8'));rs=json.loads((BASE/'download_results.json').read_text(encoding='utf8'));by={p['id']:p for p in ps}
    assert len(ps)==len(by)==85 and len(rs)==85
    expected=[]; hashes=[];size=0
    for r in rs:
        p=by[r['id']];assert r['filename']==p['filename'] and p['filename'].startswith(p['id']+'_')
        f=TARGET/p['filename']
        assert f.parent==TARGET
        if r['status']=='ok':
            assert f.exists();data=f.read_bytes();assert hashlib.sha256(data).hexdigest()==r['sha256'];assert len(PdfReader(f).pages)==r['pages']
            expected.append(f.name);hashes.append(r['sha256']);size+=len(data)
        else:assert not f.exists(),'Failed paper unexpectedly has a PDF'
    assert len(hashes)==len(set(hashes)), 'Duplicate PDFs'
    actual={f.name for f in TARGET.glob('*.pdf')};assert actual==set(expected),(actual-set(expected),set(expected)-actual)
    report=BASE/'BEV通信后融合_恶劣天气鲁棒性论文检索_20260913.md';text=report.read_text(encoding='utf8')
    assert len(re.findall(r'^## ',text,re.M))==11
    ids=re.findall(r'^\| \*\*(\d{2}_\d{2}) ·',text,re.M);assert ids==[p['id'] for p in ps]
    for filename in expected:assert f']({filename})' in text
    assert '\ufffd' not in text
    local_assets={
        report:TARGET/report.name,
        BASE/'下载失败汇总.md':TARGET/'下载失败汇总.md',
        BASE/'manifest.json':TARGET/'论文元数据与下载清单.json',
        BASE/'download_results.json':TARGET/'下载校验记录.json',
        BASE/'search_keywords.json':TARGET/'检索关键词记录.json',
    }
    if args.deliver:
        for src,dst in local_assets.items():
            if dst.exists() and dst.read_bytes()!=src.read_bytes():raise RuntimeError('Refusing to overwrite a different existing report: '+str(dst))
            shutil.copy2(src,dst)
        for src,dst in local_assets.items():assert hashlib.sha256(src.read_bytes()).digest()==hashlib.sha256(dst.read_bytes()).digest()
    print(json.dumps({'mode':'delivered' if args.deliver else 'verified','papers':len(ps),'pdfs':len(expected),'failed':len(ps)-len(expected),'pdf_MiB':round(size/1024/1024,2),'headings':11,'rows':len(ids),'keywords':len(set(x['q'] for x in q['queries'])),'target':str(TARGET),'report':str(TARGET/report.name)},ensure_ascii=False))
if __name__=='__main__':main()
