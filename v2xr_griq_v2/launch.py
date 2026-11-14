"""One shared stage, four physical GPUs, and fail-closed joins."""
import fcntl
import os
import re
import subprocess
import sys
import time
from v2xr_griq_v2.dataset import HERE,OUT,root_for,save

DEVICES={'E1':'1','E2':'2','E3':'3','E4':'4'}


def gpu_audit(cards):
    status=subprocess.check_output(['rocm-smi','--showmeminfo','vram'],text=True)
    for card in cards:
        used=re.search(rf'HCU\[{card}\].*?Total Used Memory \(MiB\):\s*(\d+)',status)
        total=re.search(rf'HCU\[{card}\].*?Total Memory \(MiB\):\s*(\d+)',status)
        if not used or not total or int(used[1])>256 or int(total[1])<6144:
            raise RuntimeError(f'Physical GPU {card} is occupied or unavailable')


def start(name,module,args,card=''):
    logs=OUT/'logs'
    logs.mkdir(exist_ok=True)
    with (logs/f'{name}.log').open('ab') as f:
        p=subprocess.Popen([sys.executable,'-uB','-m','v2xr_griq_v2.'+module,*args],
              env=dict(os.environ,ROCR_VISIBLE_DEVICES=card),stdout=f,stderr=subprocess.STDOUT)
    save(logs/f'{name}.pid.json',dict(pid=p.pid,physical_gpu=card or None))
    print('START',name,p.pid,card or 'CPU',flush=True)
    return name,p


def join(tasks):
    bad={}
    for name,p in tasks:
        code=p.wait()
        save(OUT/'logs'/f'{name}.exit.json',dict(pid=p.pid,exit_code=code))
        print('EXIT',name,p.pid,code,flush=True)
        if code:
            bad[name]=code
    if bad:
        raise RuntimeError(f'See individual logs: {bad}')


def wait_for_cards(cards):
    last = None
    while True:
        try:
            gpu_audit(cards)
            return
        except RuntimeError as exc:
            if '--wait-for-gpus' not in sys.argv:
                raise
            if str(exc)!=last:
                print('WAITING FOR GPUS',exc,flush=True)
                last=str(exc)
            time.sleep(45)


def main():
    if b'\r' in (HERE/'run_all.sh').read_bytes():
        raise ValueError('Shell script must be LF')
    subprocess.run(['sh','-n',str(HERE/'run_all.sh')],check=True)
    join([start('preflight','preflight',[])])
    if '--preflight-only' in sys.argv:
        return
    join([start('tests','tests.test_design',[])])
    for smoke in (True,False):
        root=root_for(smoke)
        root.mkdir(exist_ok=True)
        args=['--smoke'] if smoke else []
        tag='smoke_' if smoke else ''
        if not (root/'DATA_AUDIT.json').exists():
            wait_for_cards(('3',))
        join([start(tag+'shared','prepare_shared',['shared',*args],'3')])
        if '--shared-smoke' in sys.argv:
            return
        pending=[(g,card) for g,card in DEVICES.items() if not (root/g/'COMPLETE.json').exists()]
        if pending:
            wait_for_cards([card for _,card in pending])
        join([start(tag+g,'worker',[g,*args],card) for g,card in DEVICES.items()])
        if '--smoke' in sys.argv:
            return
    join([start('compare','compare',[])])
    print('COMPLETE GRIQ V2',flush=True)


if __name__=='__main__':
    with (OUT/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        main()
