import argparse
import fcntl
import os
import re
import subprocess
import sys

from context import OUT

JOBS=(('D0',None,['field_audit.py']),('A_P1_dev','1',['collect.py','train_dev']),
      ('A_P1_val','2',['collect.py','validate']),('B','3',['sensitivity.py']),('A_pairs','4',['a_pairs.py']))


def launch(selected):
    from context import read
    def completed(name):
        if name=='D0': return (OUT/'D0.json').exists()
        if name=='B': return (OUT/'B_COMPLETE.json').exists()
        if name=='A_pairs': return (OUT/'A_PAIRS_COMPLETE.json').exists()
        split='train_dev' if name=='A_P1_dev' else 'validate'
        return (OUT/split/'COMPLETE.json').exists()
    jobs=[j for j in JOBS if (not selected or j[0] in selected) and not completed(j[0])]
    cards=subprocess.check_output(['rocm-smi','--showmeminfo','vram'],text=True)
    for name,card,_ in jobs:
        if card:
            m=re.search(rf'HCU\[{card}\].*?vram Total Used Memory \(MiB\):\s*(\d+)',cards)
            if not m or int(m[1])>256:
                raise ValueError(f'Card {card} occupied for {name}')
    processes=[]
    for name,card,args in jobs:
        with (OUT/f'{name}.log').open('ab') as log:
            proc=subprocess.Popen([sys.executable,'-uB',*args],stdout=log,stderr=subprocess.STDOUT,
                                  env=dict(os.environ,ROCR_VISIBLE_DEVICES=card or ''))
        processes.append((name,proc))
        print('START',name,card or 'CPU',proc.pid,flush=True)
    errors={}
    for name,proc in processes:
        status=proc.wait()
        print('EXIT',name,status,flush=True)
        if status:
            errors[name]=status
    if errors:
        raise RuntimeError(f'Jobs failed: {errors}')
    print('COMPLETE COLLECTION',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--collect',nargs='*',choices=[x[0] for x in JOBS])
    args=p.parse_args()
    with (OUT/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        launch(args.collect)
        if args.collect is None:
            with (OUT/'reports.log').open('ab') as log:
                subprocess.run([sys.executable,'-uB','analyze.py'],check=True,stdout=log,stderr=subprocess.STDOUT,
                               env=dict(os.environ,ROCR_VISIBLE_DEVICES=''))
            print('COMPLETE ALL',flush=True)

