"""Read completed SCOR-W reports through the existing forced read-only SSH reader.

No simulator, dataset generation, model inference or server file mutation occurs.
Only whitelisted report names are requested; downloaded exact bytes must match
the existing server publication manifest. Private key contents are never read.
"""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import hashlib
import json
import re
import subprocess

HERE=Path(__file__).resolve().parent
PRIOR="C:/Users/25313/.codex/visualizations/2026/10/26/01a1760c-afc0-7891-bd7e-0809b7e56784"
ROOT="/data/cjm/datasets/scorw-fog-v1-coveragefix/"
SSH=["C:/Windows/System32/OpenSSH/ssh.exe","-T","-o","BatchMode=yes","-o","IdentitiesOnly=yes",
     "-o","StrictHostKeyChecking=yes","-o","UserKnownHostsFile="+PRIOR+"/server_assessment_20261026.known_hosts",
     "-o","ConnectTimeout=10","-o","ConnectionAttempts=1","-o","ServerAliveInterval=10","-o","ServerAliveCountMax=3",
     "-o","ClearAllForwardings=yes","-o","ControlMaster=no","-o","ControlPath=none",
     "-i",PRIOR+"/admin1_readonly_ssh/id_ed25519","-p","22","root@172.19.165.3"]
REPORTS=("REPORT_MANIFEST.json","VALIDATION_GATES.json","DATASET_QA.json","PHYSICS_AUDIT.json",
         "COMPATIBILITY_AUDIT.json","GENERATION_BUDGET.json","PERFORMANCE_BENCHMARK.json",
         "SOURCE_INVENTORY.json","ENGINEERING_ACCEPTANCE.json","VALIDATION_ATTFUSE_RESULTS.json",
         "R1_HAHNER_REGRESSION.json","R1_HAHNER_REGRESSION.md","R2_IOU_AUDIT.json","MODEL_AUDIT.json",
         "DEPENDENCY_AUDIT.json","FINAL_RESULTS.md","FINAL_COMMIT.json","DATASET_PROTOCOL.json",
         "PERFORMANCE_SAMPLE_FREEZE.json","CONTRACT.json","SOURCE_LAYOUT.json","README.md")


def page(name,first):
    if name not in REPORTS:raise ValueError("Unregistered report name")
    result=subprocess.run(SSH+["read "+ROOT+name+" "+str(first)+" 260"],capture_output=True,timeout=45)
    if result.returncode:raise RuntimeError(name+": "+(result.stderr or result.stdout[-2000:]).decode("utf-8","replace"))
    lines=result.stdout.decode("utf-8").splitlines()
    header=json.loads(lines[0])
    if header.get("action")!="read" or header.get("path")!=ROOT+name:raise ValueError("Unexpected read-only SSH response: "+str(header))
    rows=[]
    for s in lines[1:]:
        m=re.fullmatch(r"(\d+): (.*)",s)
        if not m:raise ValueError("Unnumbered report line")
        rows.append((int(m[1]),m[2]))
    return header,rows


def fetch(pool,name):
    header,first=page(name,1);rows=dict(first)
    cursor=261
    while len(first)==260:
        # At most four concurrent bounded read-only connections. No shell command
        # or server-side Python execution is requested through SSH.
        futures=[pool.submit(page,name,cursor+260*k) for k in range(4)]
        pages=[f.result() for f in futures]
        for h,r in pages:
            if h["bytes"]!=header["bytes"]:raise ValueError("Server report changed during copy")
            rows.update(r)
        cursor+=1040
        first=pages[-1][1]
    if list(sorted(rows))!=list(range(1,len(rows)+1)):raise ValueError("Report line gap")
    return ("\n".join(rows[k] for k in sorted(rows))+"\n").encode("utf-8"),header


def main():
    dest=HERE/"verified_reports";dest.mkdir(exist_ok=True)
    records={};manifest=None
    with ThreadPoolExecutor(max_workers=4) as pool:
        for name in REPORTS:
            expected=manifest.get(name) if manifest is not None else None
            saved=dest/name
            if expected and saved.exists() and hashlib.sha256(saved.read_bytes()).hexdigest()==expected["sha256"]:
                records[name]=dict(status="SHA256_VERIFIED",bytes=saved.stat().st_size,sha256=expected["sha256"])
                print(name+": SHA256_VERIFIED (existing local copy)",flush=True)
                continue
            try:data,head=fetch(pool,name)
            except Exception as error:
                records[name]=dict(status="TRANSPORT_UNAVAILABLE",reason=str(error))
                print(name+": TRANSPORT_UNAVAILABLE: "+str(error)[-400:],flush=True)
                continue
            actual=hashlib.sha256(data).hexdigest()
            if len(data)!=head["bytes"] or expected and (actual!=expected["sha256"] or len(data)!=expected["bytes"]):
                # Never label a redacted/truncated transport copy as the original.
                records[name]=dict(status="COPY_MISMATCH",bytes=len(data),expected_bytes=head["bytes"],sha256=actual)
                print(name+": COPY_MISMATCH",flush=True)
                continue
            (dest/name).write_bytes(data)
            if name=="REPORT_MANIFEST.json":manifest=json.loads(data)
            records[name]=dict(status="SHA256_VERIFIED" if expected else "PUBLICATION_MANIFEST_COPIED",bytes=len(data),sha256=actual)
            print(name+": "+records[name]["status"],flush=True)
    audit=dict(server="root@172.19.165.3:22",root=ROOT,transport="existing bounded read-only forced-command SSH reader",
        host_fingerprint="SHA256:eKT+SiVJqKJZvaCmeXUl0kIc0++uBsJt6G2yA85hEVg",records=records,
        server_files_modified=False,experiments_or_inference_run=False,
        scope="published report copies only; full PCD and provenance arrays were not rehashed by this review")
    (HERE/"REPORT_DOWNLOAD_AUDIT.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")


if __name__=="__main__":main()
