"""Sequential CPU-only diagnostics. Never train, generate weather or alter old outputs."""
import argparse
import os
from pathlib import Path
import traceback
from .core import HERE, REPO, server_only, read, write, text, identity, digest, sources, lock, stage


def execute(mode, resume):
    server_only()
    os.environ['ROCR_VISIBLE_DEVICES']='';os.environ.pop('HIP_VISIBLE_DEVICES',None);os.environ.pop('CUDA_VISIBLE_DEVICES',None)
    c=read(HERE/'config.json'); src=sources(); out=Path(c['output'])/mode
    request=dict(mode=mode,config=c,sources=src,GPU_used=False)
    with lock(out):
        contract=dict(identity=identity(request),request=request)
        p=out/'CONTRACT.json'
        if p.exists():
            if not resume:raise ValueError('Existing run requires RESUME=1')
            if read(p)!=contract:raise ValueError('Source/config/runtime changed; use a new versioned output')
        else:
            if any(x.name!='.lock' for x in out.iterdir()):raise ValueError('Nonempty unregistered output')
            write(p,contract)
        active='freeze';gates=dict(status='RUNNING',mode=mode,stages={},GPU_used=False)
        try:
            from .data import freeze
            # Recheck all frozen inputs on every RESUME, even when all stages are committed.
            temp=out/'input_check';temp.mkdir(exist_ok=True)
            f=freeze(c,temp)
            if (out/'INPUT_FREEZE.json').exists() and read(out/'INPUT_FREEZE.json')!=f:
                raise ValueError('Original inputs/cache identities changed since first invocation')
            write(out/'INPUT_FREEZE.json',f);write(out/'SOURCE_AUDIT.json',src)
            parent=identity(dict(contract=contract['identity'],inputs=f['inputs_identity']))
            def phase(name,callback):
                nonlocal active
                active=name;print('START '+name,flush=True)
                folder=out/name
                stage(folder,dict(parent=parent,stage=name),lambda:callback(folder))
                gates['stages'][name]='passed';write(out/'VALIDATION_GATES.json',gates)
                print('PASS '+name,flush=True)
            from . import tests, ap_audit, point_flow, report
            phase('tests',tests.run)
            phase('A1',lambda p:ap_audit.replay(c,p))
            # Preflight verifies sources, all input bytes, AP replay and actual spconv synthetic tests.
            if mode=='preflight':
                gates['status']='PREFLIGHT_COMPLETE'
            else:
                phase('A',lambda p:ap_audit.run(c,p,parent,mode=='smoke'))
                # Always validate the fixed first two real frames before full tracing.
                phase('B_smoke',lambda p:point_flow.run(c,p,parent,True))
                b='B_smoke'
                if mode=='full':
                    phase('B',lambda p:point_flow.run(c,p,parent,False));b='B'
                phase('reports',lambda p:report.run(c,p,out/b,out/'A',mode=='smoke'))
                gates['status']='SMOKE_COMPLETE' if mode=='smoke' else 'DIAGNOSTICS_COMPLETE'
            write(out/'VALIDATION_GATES.json',gates)
            names=['AP_PROTOCOL_AUDIT.json','AP_PROTOCOL_COMPARISON.md','GT_DIFFERENCE.json','MATCHING_DIFFERENCE.json',
                   'FACTOR_CONTRIBUTIONS.json']
            # Publish exact copies with hashes; original committed stage artifacts remain immutable.
            for name in names:
                source=out/('A1' if name=='AP_PROTOCOL_AUDIT.json' else 'A')/name
                if source.exists():
                    if (out/name).exists() and digest(out/name)!=digest(source):raise ValueError('Published report changed: '+name)
                    text(out/name,source.read_text('utf-8'))
            for source in (out/'reports').glob('*'):
                if source.is_file() and source.suffix in ('.json','.md') and source.name not in ('CONTRACT.json','COMMIT.json'):
                    target=out/source.name
                    if target.exists() and digest(target)!=digest(source):raise ValueError('Published report changed: '+str(target))
                    text(target,source.read_text('utf-8'))
            write(out/'REPORT_MANIFEST.json',{p.relative_to(out).as_posix():digest(p) for p in out.iterdir()
                  if p.is_file() and p.suffix in ('.json','.md') and p.name!='REPORT_MANIFEST.json'})
        except BaseException as error:
            gates.update(status='FAILED',stage=active,error_type=type(error).__name__,reason=str(error))
            write(out/'VALIDATION_GATES.json',gates)
            text(out/'FAILURE_TRACEBACK.txt',traceback.format_exc());raise
        finally:
            print('Reports: %s; status=%s'%(out,gates['status']),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['preflight','smoke','full'],required=True)
    p.add_argument('--resume',action='store_true');a=p.parse_args();execute(a.mode,a.resume)


if __name__=='__main__':main()
