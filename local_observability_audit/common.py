"""Immutable identities, safe paths and atomic JSON. No torch or experiment imports."""
from __future__ import annotations
import argparse
import contextlib
import difflib
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import numpy as np

PACKAGE = Path(__file__).resolve().parent
PROJECT = PACKAGE.parent
WEATHERS = ('clean', 'fog', 'rain', 'snow')

def code_root():
    candidates = (PROJECT, PROJECT.parent, PROJECT.parent / 'OpenCOOD-main')
    for p in candidates:
        if (p / 'opencood/data_utils/datasets/intermediate_fusion_dataset.py').is_file():
            return p
    raise FileNotFoundError('Cannot locate actual OpenCOOD source tree')

def install_paths():
    for p in (PROJECT, code_root()):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))

def digest(path, text=False):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        if text:
            h.update(f.read().replace(b'\r\n', b'\n'))
        else:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
                h.update(chunk)
    return h.hexdigest()

def plain(x):
    if isinstance(x, np.ndarray):
        return plain(x.tolist())
    if isinstance(x, np.generic):
        return plain(x.item())
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return [plain(v) for v in x]
    if isinstance(x, float) and not np.isfinite(x):
        raise ValueError('Non-finite JSON value; use explicit null + validity')
    if isinstance(x, Path):
        return str(x)
    return x

def object_hash(value):
    return hashlib.sha256(json.dumps(plain(value), sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='\n') as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)

def atomic_json(path, value):
    atomic_text(path, json.dumps(plain(value), ensure_ascii=False, indent=2,
                               sort_keys=True, allow_nan=False) + '\n')

def read_json(path):
    with Path(path).open(encoding='utf-8') as f:
        return json.load(f)

def verify_protocol(protocol):
    payload={k:protocol[k] for k in ('config','mode','sources','frontend','dataset','python')}
    if object_hash(payload)!=protocol['identity']:
        raise ValueError('Protocol contents/identity mismatch')
    if object_hash(protocol['config'])!=protocol['config_hash'] or object_hash(protocol['dataset'])!=protocol['dataset_hash']:
        raise ValueError('Protocol component hash mismatch')
    if object_hash(protocol['sources'])!=protocol['source_hash'] or protocol['selection']!=protocol['dataset']['selection']:
        raise ValueError('Protocol source/selection mismatch')

def safe_path(path):
    p = Path(path).expanduser()
    for candidate in (p.absolute(), p.resolve()):
        if any(x.lower() == 'test' or 'source_evidence_audit' in x.lower()
               for x in candidate.parts):
            raise ValueError('Prohibited test/S4 artifact path: ' + str(p))
    return p.resolve()

def run_path(path):
    p = safe_path(path)
    if p.parent != Path('/data/cjm/datasets/logs').resolve() or not p.name.startswith('observability_audit_'):
        raise ValueError('RUN must be /data/cjm/datasets/logs/observability_audit_<new_id>')
    return p

def include_source(label, relative):
    """Exclude artifacts and four reviewed, unimported OpenCOOD backup files."""
    relative=Path(relative)
    if '__pycache__' in relative.parts:
        return False
    if label=='opencood':
        excluded={
            'data_utils/datasets/intermediate_fusion_dataset_back.py',
            'models/point_pillar_where2comm_back.py',
            'utils/weather_reliability_back.py',
            'models/fuse_modules/where2comm_fuse(6.27第一次实验）.py',
        }
        return (relative.suffix=='.py' and 'logs' not in relative.parts
                and relative.as_posix() not in excluded)
    return (label=='audit' and relative.as_posix()=='reviewed_source_variants.json') or relative.suffix in ('.py','.sh','.yaml') or (
        label=='audit' and relative.parts[0]=='reference_sources' and relative.suffix=='.txt')


def source_identity():
    roots = {'audit': PACKAGE, 'gspr': PROJECT / 'attfuse_gspr',
             'attfuse': PROJECT / 'attfuse_code', 'opencood': code_root() / 'opencood'}
    output = {}
    for label, root in roots.items():
        for p in sorted(root.rglob('*')):
            if p.is_file() and include_source(label,p.relative_to(root)):
                output[label + '/' + p.relative_to(root).as_posix()] = digest(p, text=True)
    return output

def snapshot_changes(expected,actual):
    """Accept only exact, reviewed server bytes; record actual hashes in protocol."""
    changed=sorted(k for k in expected.keys() | actual.keys() if expected.get(k)!=actual.get(k))
    policy_key='audit/reviewed_source_variants.json'
    policy=PACKAGE/'reviewed_source_variants.json'
    if policy_key not in expected or actual.get(policy_key)!=expected[policy_key]:
        return changed,[]
    accepted=[]
    for key,variant in read_json(policy).items():
        if (key in changed and expected.get(key)==variant['local_sha256']
                and actual.get(key)==variant['server_sha256']):
            reference_key='audit/'+variant['reference']
            # The reviewed reference and allowlist are both inside the frozen snapshot.
            if (expected.get(reference_key)==variant['server_sha256']
                    and actual.get(reference_key)==variant['server_sha256']):
                changed.remove(key); accepted.append(key)
    return changed,accepted


def verify_snapshot():
    expected = read_json(PACKAGE / 'source_snapshot.json')
    actual = source_identity()
    changed,_ = snapshot_changes(expected,actual)
    if changed:
        raise ValueError('Source snapshot mismatch; inspect with python -B -m local_observability_audit.common --check-sources (do not overwrite server source or regenerate hashes): ' + ', '.join(changed))
    return actual


def check_sources():
    """Read-only diagnosis; no model import, GPU, dataset access or hash changes."""
    expected=read_json(PACKAGE/'source_snapshot.json'); actual=source_identity()
    changed,accepted=snapshot_changes(expected,actual)
    print('OpenCOOD source root:',code_root())
    for key in accepted:
        print('ACCEPTED reviewed server variant:',key,actual[key])
    for key in changed:
        print(key,'expected='+str(expected.get(key)),'actual='+str(actual.get(key)))
    key='opencood/data_utils/datasets/basedataset.py'
    if key in changed:
        reference=PACKAGE/'reference_sources'/'basedataset.py.txt'
        reference_key='audit/reference_sources/basedataset.py.txt'
        if digest(reference,text=True)!=expected.get(reference_key):
            raise ValueError('Diagnostic reference hash mismatch')
        current=code_root()/'opencood/data_utils/datasets/basedataset.py'
        if current.is_file():
            print(''.join(difflib.unified_diff(
                reference.read_text(encoding='utf-8').splitlines(keepends=True),
                current.read_text(encoding='utf-8').splitlines(keepends=True),
                fromfile='reviewed_local/basedataset.py',tofile='server/basedataset.py')))
    print('PASS' if not changed else 'MISMATCH: '+str(len(changed))+' files; inspection only, nothing changed.')
    return not changed

def config(path):
    import yaml
    with safe_path(path).open(encoding='utf-8') as f:
        c = yaml.safe_load(f)
    if c['state'] != ['x', 'y', 'l', 'w', 'yaw'] or tuple(c['weathers']) != WEATHERS:
        raise ValueError('Unsupported schema/state/weather')
    if c['subset_policy'] != 'true_forward':
        raise ValueError('Only true subset forward is authorized in v1')
    if len(c['parameter_scales']) != 5 or min(c['parameter_scales']) <= 0:
        raise ValueError('Positive five-dimensional scales required')
    if c['matrix_ridge']<=0 or not 0<c['weight_eps']<=1 or not 1<=c['roi_expansion']<=2:
        raise ValueError('Invalid geometric protocol')
    if any(c[k]<1 for k in ('frames_per_weather','smoke_frames_per_weather','bootstrap_replicates','permutation_replicates')):
        raise ValueError('Counts must be positive')
    if c['frames_per_weather']>150:
        raise ValueError('O0 capped at 150 frames/weather; use --full-validation separately')
    return c

def hash_order(values, seed, namespace=''):
    return sorted(values, key=lambda x: (object_hash([seed, namespace, x]), str(x)))

def select_frames(records, count, seed):
    """Round-robin scene stratification, hashed scenes and hashed frames; paired across weather."""
    scenes = hash_order(sorted({r['scene_id'] for r in records}), seed, 'scene')
    groups = {s: sorted((r for r in records if r['scene_id'] == s),
                        key=lambda r: object_hash([seed, s, r['frame_id']])) for s in scenes}
    chosen = []
    for offset in range(max((len(v) for v in groups.values()), default=0)):
        for s in scenes:
            if offset < len(groups[s]) and (count is None or len(chosen) < count):
                chosen.append(groups[s][offset])
    return chosen

@contextlib.contextmanager
def run_lock(run):
    import fcntl
    with (Path(run) / '.lock').open('a+') as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield

class Manifest:
    def __init__(self, run, identity):
        self.run, self.identity = Path(run), identity
        p = self.run / 'manifest.json'
        self.value = read_json(p) if p.exists() else {'identity': identity, 'artifacts': {}, 'stage': 'initialized'}
        if self.value['identity'] != identity:
            raise ValueError('Cache identity mismatch')

    def save(self):
        self.value['updated_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        atomic_json(self.run / 'manifest.json', self.value)

    def put(self, relative, value):
        p = self.run / relative
        if p.exists() and relative not in self.value['artifacts']:
            # Interrupted write before manifest publication: recompute, never adopt blindly.
            pass
        atomic_json(p, {'identity': self.identity, 'payload': value})
        self.value['artifacts'][relative] = digest(p)
        self.save()

    def get(self, relative):
        expected = self.value['artifacts'].get(relative)
        if expected is None:
            return None
        p = self.run / relative
        if not p.is_file() or digest(p) != expected:
            raise ValueError('Cache corruption: ' + relative)
        obj = read_json(p)
        if obj['identity'] != self.identity:
            raise ValueError('Artifact identity mismatch')
        return obj['payload']

    def stage(self, name):
        self.value['stage'] = name
        self.save()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', action='store_true')
    parser.add_argument('--check-sources', action='store_true')
    args = parser.parse_args()
    if args.check_sources:
        raise SystemExit(0 if check_sources() else 2)
    if args.snapshot:
        atomic_json(PACKAGE / 'source_snapshot.json', source_identity())
        print('Exported source snapshot; hashes only, no experiments executed.')
