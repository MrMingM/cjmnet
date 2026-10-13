"""Read-only Linux ROCm preflight. Fail closed; no process signalling or GPU defaults."""
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess

def command(args):
    p=subprocess.run(args,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=30)
    if p.returncode:
        raise RuntimeError('Preflight command failed: '+' '.join(args)+'\n'+p.stderr)
    return p.stdout

def parse_stats(smi,gpu):
    """ROCm GPU/B and server HCU/MiB formats; unknown units fail closed."""
    try:
        stats=json.loads(smi[smi.index('{'):smi.rindex('}')+1])
        card=stats['card'+str(gpu)]
        uses=[]; memories=[]
        for key,value in card.items():
            label=' '.join(key.lower().split())
            if label in ('gpu use (%)','hcu use (%)'):
                uses.append(float(str(value).strip().removesuffix('%').strip()))
            memory=re.fullmatch(r'vram (?:total )?used memory \((b|mib)\)',label)
            if memory:
                memories.append(float(value)*(1024**2 if memory.group(1)=='mib' else 1))
        if len(uses)!=1 or len(memories)!=1:
            raise ValueError('Expected one GPU/HCU utilization and one VRAM used-memory field')
        use,used=uses[0],memories[0]
        if not math.isfinite(use) or not 0<=use<=100 or not math.isfinite(used) or used<0:
            raise ValueError('Invalid utilization or memory value')
        return stats,use,used
    except (ValueError,KeyError,TypeError,AttributeError) as e:
        raise RuntimeError(f'Cannot establish idle GPU {gpu}: {e}. rocm-smi output: {smi}') from e


def process_kind(args):
    """Classify the executable, not text inside a log path or grep expression."""
    try:
        tokens=shlex.split(args)
    except ValueError:
        return 'unknown_s4' if 'source_evidence_audit' in args else 'other'
    if not tokens:
        return 'other'
    executable=Path(tokens[0]).name
    s4='source_evidence_audit' in args
    if executable in ('tail','head','cat','less','more','grep','rg','pgrep','ps'):
        return 'observer'
    if re.fullmatch(r'python(?:[0-9.]*)?',executable):
        return 's4_python' if s4 else 'python'
    return 'unknown_s4' if s4 else 'other'


def check(gpu,cfg):
    if os.name!='posix' or not Path('/proc').is_dir():
        raise RuntimeError('Research forward is server/Linux only')
    if gpu in cfg['gpu']['reserved']:
        raise ValueError('GPU is reserved by protocol: '+str(gpu))
    if os.environ.get('ROCR_VISIBLE_DEVICES')!=str(gpu):
        raise ValueError('Set exactly ROCR_VISIBLE_DEVICES='+str(gpu))
    if os.environ.get('HIP_VISIBLE_DEVICES') or os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise ValueError('Unset HIP_VISIBLE_DEVICES and CUDA_VISIBLE_DEVICES')
    ps=command(['ps','-eo','pid,args'])
    smi=command(['rocm-smi','--showuse','--showmeminfo','vram','--json'])
    pids=command(['rocm-smi','--showpids'])
    stats,use,used=parse_stats(smi,gpu)
    running=[]
    for line in ps.splitlines()[1:]:
        parts=line.strip().split(maxsplit=1)
        if len(parts)!=2 or int(parts[0])==os.getpid():
            continue
        pid,args=parts
        kind=process_kind(args)
        if kind in ('other','observer'):
            continue
        try:
            env=Path('/proc',pid,'environ').read_bytes().split(b'\0')
        except FileNotFoundError:
            continue
        except PermissionError:
            if kind in ('s4_python','unknown_s4'):
                raise RuntimeError(f'Cannot read S4 device visibility: PID={pid}, command={args}')
            continue
        visibility={}
        for field in env:
            if field.startswith((b'ROCR_VISIBLE_DEVICES=',b'HIP_VISIBLE_DEVICES=',b'CUDA_VISIBLE_DEVICES=')):
                k,v=field.decode(errors='replace').split('=',1); visibility[k]=v
        if kind in ('s4_python','unknown_s4'):
            if not visibility.get('ROCR_VISIBLE_DEVICES'):
                raise RuntimeError(f'S4 visibility unknown; refusing allocation: PID={pid}, kind={kind}, command={args}, visibility={visibility}')
            running.append({'pid':int(pid),'args':args,'visibility':visibility})
        if str(gpu) in visibility.get('ROCR_VISIBLE_DEVICES','').split(','):
            raise RuntimeError('Selected GPU referenced by existing Python/S4 process '+pid)
    if use>cfg['gpu']['maximum_idle_use_percent'] or used>cfg['gpu']['maximum_idle_vram_mb']*1024**2:
        raise RuntimeError(f'GPU {gpu} not idle: use={use}%, VRAM={used/1024**2:.1f} MiB')
    return {'gpu':gpu,'ps':ps,'rocm_smi':stats,'rocm_pids':pids,'s4_processes':running,
            'use_percent':use,'vram_bytes':used,'ROCR_VISIBLE_DEVICES':str(gpu),
            'limitation':'Read-only point-in-time check; no scheduler reservation of external workloads.'}
