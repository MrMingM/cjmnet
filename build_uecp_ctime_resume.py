"""Build a self-contained recovery ZIP; original archives are never rewritten."""
import ast
import hashlib
import json
from pathlib import Path
import zipfile

from uecp_gspr_ctime_resume import PRIOR_SHA

ROOT = Path(__file__).resolve().parent


def build(destination):
    candidates = [ROOT / 'uecp_gspr_dependency_snapshot_v1.zip',
                  ROOT / 'project_backup_20261023/files/uecp_gspr_dependency_snapshot_v1.zip']
    prior = next((p for p in candidates if p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest() == PRIOR_SHA), None)
    if prior is None:
        raise ValueError('Exact original dependency snapshot ZIP not found')
    members = {'__main__.py': b'from uecp_gspr_ctime_resume import main\nmain()\n',
        'uecp_gspr_ctime_resume.py': (ROOT / 'uecp_gspr_ctime_resume.py').read_bytes(),
        'prior/dependency_snapshot_v1.zip': prior.read_bytes(),
        'README.md': (ROOT / 'UECP_CTIME_RESUME.md').read_bytes()}
    for name, raw in members.items():
        if name.endswith('.py'):
            ast.parse(raw.decode(), feature_version=(3, 10))
    members['manifest.json'] = (json.dumps(dict(files={n: hashlib.sha256(raw).hexdigest() for n, raw in members.items()}),
        sort_keys=True, indent=2)+'\n').encode()
    with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name, raw in sorted(members.items()):
            z.writestr(name, raw)
    print('ZIP:', destination)
    print('SHA256:', hashlib.sha256(Path(destination).read_bytes()).hexdigest())


if __name__ == '__main__':
    build(ROOT / 'uecp_gspr_ctime_resume_v1.zip')
