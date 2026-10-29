"""Adoptable original-code A1 worker. No optimization or signal handling here."""
from pathlib import Path
import os
import subprocess
import time
from auditlib import atomic_json, complete, read_json, sha256
from lineage import plan, selected


def process(pid):
    p=Path('/proc')/str(pid)
    raw=(p/'stat').read_text().rsplit(')',1)[1].split()
    command=(p/'cmdline').read_bytes().split(b'\0')
    return dict(start=raw[19],state=raw[0],command=[v.decode() for v in command if v])


class A1Lane:
    def __init__(self,root,command):
        self.root=Path(root);self.command=command
        self.folder=self.root/'A1_lane';self.folder.mkdir(exist_ok=True)
        p=plan(root);self.training=Path(p['old_root'])/'A1'
        self.boundary_record=Path(p['boundary_record'])

    def alive(self,row):
        try:
            p=process(row['pid'])
            return p['start']==row['start'] and p['state']!='Z' and p['command']==self.command
        except FileNotFoundError:return False

    def finished(self):
        p=self.training/'training_summary.json'
        if not p.exists() or read_json(p).get('status')!='complete':return False
        selected(self.root,'A1')
        return True

    def ensure(self):
        if self.finished():return
        status=self.folder/'state.json'
        if status.exists() and self.alive(read_json(status)):
            print('ADOPT preserved A1 PID='+str(read_json(status)['pid']),flush=True)
            return
        from checkpoint import committed
        c=read_json(self.training/'contract.json')
        rows=committed(self.training,20,c['identity'])
        stop=read_json(self.boundary_record)
        if len(rows)<stop['completed_epoch']:raise ValueError('Preserved A1 commit missing')
        env=os.environ.copy()
        # Device 2 and the old environment/settings are retained exactly.
        env['ROCR_VISIBLE_DEVICES']=os.environ['ROCR_VISIBLE_DEVICES']
        import re
        check=subprocess.run(['rocm-smi','--showmeminfo','vram','-d',env['ROCR_VISIBLE_DEVICES']],
            capture_output=True,text=True,check=True)
        usage=re.findall(r'Total Used Memory \(MiB\):\s*(\d+)',check.stdout)
        if len(usage)!=1 or int(usage[0])>16:
            raise RuntimeError('Original A1 GPU occupied; no unrelated process was stopped')
        with (self.folder/'train.log').open('ab') as log:
            p=subprocess.Popen(self.command,env=env,stdin=subprocess.DEVNULL,stdout=log,
                stderr=subprocess.STDOUT,start_new_session=True)
        proc=process(p.pid)
        row=dict(status='running',pid=p.pid,start=proc['start'],command=self.command,
            original_contract_sha256=sha256(self.training/'contract.json'),
            completed_at_launch=len(rows),planned_next_epoch=len(rows)+1,
            resume_state=rows[-1]['resume_state'],resume_state_sha256=rows[-1]['resume_state_sha256'],
            physical_gpu=env['ROCR_VISIBLE_DEVICES'],adoption_supported=True)
        count=len(list(self.folder.glob('launch_*.json')))+1
        atomic_json(self.folder/f'launch_{count:03d}.json',row)
        atomic_json(status,row)
        print('PRESERVED A1 START PID='+str(p.pid)+' next_epoch='+str(len(rows)+1),flush=True)

    def wait(self):
        self.ensure()
        while not self.finished():
            row=read_json(self.folder/'state.json')
            if not self.alive(row):raise RuntimeError('Original A1 exited before complete; inspect A1_lane/train.log')
            time.sleep(2)
        summary=self.training/'training_summary.json'
        atomic_json(self.folder/'complete.json',dict(status='complete',training_summary=str(summary),
            sha256=sha256(summary),original_code_used=True,completed_epochs=20))
        return [self.folder/'complete.json']
