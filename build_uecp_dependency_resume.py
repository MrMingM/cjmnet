"""Package byte-verified historical sources; never import Torch/OpenCOOD or model weights."""
import ast
import hashlib
import json
from pathlib import Path
import zipfile
from uecp_gspr_dependency_resume import CURRENT, LEGACY_SHA, MODULES, ORIGINAL, SNAPSHOT, digest

ROOT = Path(__file__).resolve().parent


def payloads():
    legacy = (ROOT / 'uecp_gspr_resume_fix_v2.zip').read_bytes()
    if digest(legacy) != LEGACY_SHA:
        raise ValueError('Original v2 runner archive differs')
    sources = {}
    with zipfile.ZipFile(ROOT / 'uecp_gspr_v2xr_sync_v2.zip') as archive:
        original = {n: archive.read('v2xr_raygen_e0e3/'+n+'.py') for n in MODULES}
    with zipfile.ZipFile(ROOT / 'v2xr_raygen_e0e3_v3_20261020.zip') as archive:
        sources['training'] = archive.read('v2xr_raygen_e0e3/training.py')
    sources.update({n: original[n] for n in ('runtime', 'preflight')})
    for name in MODULES:
        if digest(original[name]) != ORIGINAL[name] or digest(sources[name]) != SNAPSHOT[name]:
            raise ValueError('Historical snapshot hash differs: '+name)
        if digest((ROOT / 'v2xr_raygen_e0e3' / (name+'.py')).read_bytes()) != CURRENT[name]:
            raise ValueError('Current source no longer matches the reported server version: '+name)
    def parts(raw):
        tree = ast.parse(raw.decode('utf-8'))
        return ({n.name: ast.dump(n, include_attributes=False) for n in tree.body if isinstance(n, ast.FunctionDef)},
            [ast.dump(n, include_attributes=False) for n in tree.body if not isinstance(n, ast.FunctionDef)])
    old, old_top = parts(original['training'])
    prior, prior_top = parts(sources['training'])
    helpers = ('tensor_hash', 'atomic_torch', 'rng_state', 'restore_rng')
    if old_top != prior_top or any(old[n] != prior[n] for n in helpers):
        raise ValueError('Original enrolled helper computations differ')
    before, _ = parts(original['runtime'])
    current, _ = parts((ROOT / 'v2xr_raygen_e0e3/runtime.py').read_bytes())
    if before['bootstrap'] == current['bootstrap']:
        raise ValueError('Expected runtime compute-policy transition was not found')
    review = dict(scope='UECP process-private snapshots; no filesystem rollback of the shared Raygen project',
        original=ORIGINAL, executed=SNAPSHOT, physical_reported_server_version=CURRENT,
        reused_training_helpers=list(helpers), helper_bodies_and_module_level_statements_equal=True,
        runtime_bootstrap='Execute exact original de6db... snapshot; do not call the shared stable_crop.configure',
        preflight='Original preflight module; only its original content verifier is called by UECP',
        training='Previously enrolled cc5f... snapshot; UECP imports the same four original helpers, not its Raygen training loop',
        model_data_loss_optimizer_selection_AP='Unchanged sealed UECP implementation and original v2 policy',
        legacy_runner_sha256=LEGACY_SHA)
    members = {
        '__main__.py': b'from uecp_gspr_dependency_resume import main\nmain()\n',
        'uecp_gspr_dependency_resume.py': (ROOT / 'uecp_gspr_dependency_resume.py').read_bytes(),
        'legacy/uecp_gspr_resume_fix_v2.zip': legacy,
        'SOURCE_REVIEW.json': (json.dumps(review, sort_keys=True, indent=2)+'\n').encode(),
        'README.md': (ROOT / 'UECP_DEPENDENCY_SNAPSHOT_RESUME.md').read_bytes()}
    members.update({'snapshots/'+n+'.py': raw for n, raw in sources.items()})
    return members


def build(destination):
    members = payloads()
    members['manifest.json'] = (json.dumps(dict(files={n: digest(raw) for n, raw in members.items()}),
        sort_keys=True, indent=2)+'\n').encode()
    with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, raw in sorted(members.items()):
            archive.writestr(name, raw)
    print('ZIP:', destination)
    print('SHA256:', hashlib.sha256(Path(destination).read_bytes()).hexdigest())


if __name__ == '__main__':
    build(ROOT / 'uecp_gspr_dependency_snapshot_v1.zip')
