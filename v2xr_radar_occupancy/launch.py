"""Launch independent jobs with per-run logs, wait and report failures."""
import fcntl
import os
import re
import subprocess
import sys

from paths import OUT,read

JOBS = (
    ('pcd',None,['-c','from pcd_audit import run; run()'],'PCD_FIELDS.json'),
    ('D3',None,['fp_audit.py'],'D3.json'),
    ('D1','1',['driver.py','D1'],'D1.json'),
    ('D2_fog','2',['driver.py','fog'],'D2_fog.json'),
    ('D2_snow','3',['driver.py','snow'],'D2_snow.json'),
)


def done(name,result):
    path=OUT/result
    if not path.exists():
        return False
    return name!='D1' or set(read(path)['weather'])=={'clean','fog','snow'}


def main():
    usage=subprocess.check_output(['rocm-smi','--showmeminfo','vram'],text=True)
    processes=[]
    for name,card,cmd,result in JOBS:
        if done(name,result):
            print('REUSE',name,flush=True)
            continue
        env=dict(os.environ,ROCR_VISIBLE_DEVICES=card or '')
        if card:
            match=re.search(rf'HCU\[{card}\].*?vram Total Used Memory \(MiB\):\s*(\d+)',usage)
            if not match or int(match[1])>256:
                print(f'SKIP {name}: card {card} occupied',flush=True)
                processes.append((name,None))
                continue
        with (OUT/f'{name}.log').open('ab') as stream:
            p=subprocess.Popen([sys.executable,'-uB',*cmd],env=env,stdout=stream,stderr=subprocess.STDOUT)
        processes.append((name,p))
        print('START',name,'CPU' if card is None else 'GPU '+card,p.pid,flush=True)
    failures=[]
    for name,p in processes:
        rc=p.wait() if p else 1
        print('EXIT',name,rc,flush=True)
        if rc:
            failures.append(name)
    if failures:
        raise RuntimeError(f'Incomplete jobs: {failures}')
    print('COMPLETE ALL',flush=True)


if __name__=='__main__':
    with (OUT/'launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        main()

