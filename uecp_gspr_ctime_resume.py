"""Server-only additive recovery for byte-identical training files with changed ctime.

The exact prior dependency-snapshot ZIP remains the executed source lineage.
Only its input stamp guard is adapted in memory, with immutable SHA-backed receipts.
"""
import argparse
import ast
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time
import zipfile

PRIOR_SHA = '2d40d997bb8228fd78c2639041c78a37f3b8fe6af4e8489a9f50aac19a956377'
PREFIX = 'source_repair/uecp_ctime_resume_v1'
AUDIT_ENV = 'UECP_CTIME_AUDIT'
HASH_ENV = 'UECP_CTIME_AUDIT_SHA256'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(part)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def stamp(path):
    p = Path(path)
    s = p.stat()
    return dict(path=str(p), resolved=str(p.resolve(strict=True)), size=s.st_size,
                mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns)


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()
    # Independent audit namespace: never overwrite a previous receipt or manifest.
    with path.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def baseline(root):
    return {n: sha(root / n) for n in ('request.json', 'contract.json',
        'preflight/input_stamps.json', 'preflight/train_manifest.json',
        'source_repair/uecp_dependency_snapshot_v1.json')}


def train_files(root):
    request = read(root / 'request.json')
    train = Path(request['paths'].get('TRAIN_ROOT') or '/data/cjm/datasets/v2xr-official-train-v2')
    if train.name == 'train' and (train.parent / 'manifest.json').exists():
        train = train.parent
    folder = (train / 'train').absolute()
    rows = {}
    for row in read(root / 'preflight/train_manifest.json')['files']:
        p = folder / row['relative']
        if not p.resolve().is_relative_to(folder.resolve()) or str(p) in rows:
            raise ValueError('Invalid/duplicate training manifest path: '+str(p))
        rows[str(p)] = row
    return rows


def changed_file(root, old, live, manifest, hash_file=sha):
    changed = {k for k in set(old) | set(live) if old.get(k) != live.get(k)}
    if changed != {'ctime_ns'}:
        raise ValueError('Only ctime-only training changes may be audited: '+old['path']+' '+str(sorted(changed)))
    row = manifest.get(old['path'])
    if row is None or row['size'] != old['size'] or not isinstance(row.get('sha256'), str) or len(row['sha256']) != 64:
        raise ValueError('Changed input lacks an original training SHA: '+old['path'])
    p = Path(old['path'])
    # Non-radar weather PCDs must retain their original successful decode receipt.
    receipt = None
    if p.suffix == '.pcd' and not p.name.endswith('_radar.pcd'):
        q = root / 'preflight/content' / (row['relative']+'.json')
        r = read(q)
        if r.get('status') != 'complete' or r.get('source') != old or r.get('sha256') != row['sha256']:
            raise ValueError('Original content receipt differs: '+str(q))
        receipt = dict(path=str(q), sha256=sha(q))
    before = stamp(p)
    actual = hash_file(p)
    after = stamp(p)
    if before != live or after != live:
        raise ValueError('Training file changed during hashing: '+str(p))
    if actual != row['sha256']:
        raise ValueError('Training content SHA differs; no recovery: '+str(p)+' actual='+actual)
    return dict(original=old, accepted=live, sha256=actual, relative=row['relative'],
                original_content_receipt=receipt)


def audit_inputs(root, archive):
    before = baseline(root)
    rows = read(root / 'preflight/input_stamps.json')
    manifest = train_files(root)
    accepted, seen = [], set()
    record = dict(status='RUNNING', launcher_sha256=sha(archive), baseline=before,
        output=str(root), scanned=0, changed=accepted,
        scope='Only ctime-only accepted training files; original SHA required; all other guards unchanged',
        limitation='Current byte identity cannot prove that no transient historical modification ever occurred.')
    path = root / PREFIX / 'audits' / (str(time.time_ns())+'_'+str(os.getpid())+'.json')
    try:
        for i, old in enumerate(rows):
            if old['path'] in seen:
                raise ValueError('Duplicate original input stamp: '+old['path'])
            seen.add(old['path'])
            record['checking'] = old['path']
            live = stamp(old['path'])
            if live != old:
                accepted.append(changed_file(root, old, live, manifest))
                print('SHA_VERIFIED ctime-only '+old['path'], flush=True)
            record['scanned'] = i+1
            if (i+1) % 10000 == 0:
                print('INPUT_AUDIT scanned='+str(i+1)+'/'+str(len(rows))+' changed='+str(len(accepted)), flush=True)
        if baseline(root) != before:
            raise ValueError('Original audit anchors changed while scanning')
        for row in accepted:
            if stamp(row['original']['path']) != row['accepted']:
                raise ValueError('Accepted file metadata changed before audit commit: '+row['original']['path'])
        record.update(status='COMPLETE', total=len(rows))
        record.pop('checking', None)
    except BaseException as exc:
        record.update(status='FAILED', error=str(exc))
        write_new(path, record)
        raise
    write_new(path, record)
    print('INPUT_AUDIT_COMPLETE files='+str(len(rows))+' byte_verified_changes='+str(len(accepted))+' receipt='+str(path), flush=True)
    return path


def verify_audit(root, archive, path, expected_sha):
    path = Path(path).resolve(strict=True)
    if path.parent != (root / PREFIX / 'audits').resolve() or sha(path) != expected_sha:
        raise ValueError('Wrong/tampered input audit receipt')
    audit = read(path)
    if (audit.get('status') != 'COMPLETE' or audit['output'] != str(root)
            or audit['launcher_sha256'] != sha(archive) or audit['baseline'] != baseline(root)):
        raise ValueError('Input audit lineage differs')
    old_rows = read(root / 'preflight/input_stamps.json')
    old = {r['path']: r for r in old_rows}
    if len(old) != len(old_rows) or audit['scanned'] != len(old) or audit['total'] != len(old):
        raise ValueError('Incomplete original input coverage')
    manifest, accepted = train_files(root), {}
    for row in audit['changed']:
        key = row['original']['path']
        changes = {k for k in set(row['original']) | set(row['accepted']) if row['original'].get(k) != row['accepted'].get(k)}
        if (key in accepted or old.get(key) != row['original'] or changes != {'ctime_ns'}
                or key not in manifest or row['sha256'] != manifest[key]['sha256']
                or row['relative'] != manifest[key]['relative']):
            raise ValueError('Invalid accepted training stamp: '+key)
        p = Path(key)
        receipt = row['original_content_receipt']
        if p.suffix == '.pcd' and not p.name.endswith('_radar.pcd'):
            q = root / 'preflight/content' / (row['relative']+'.json')
            if receipt != dict(path=str(q), sha256=sha(q)):
                raise ValueError('Original content receipt changed: '+key)
        accepted[key] = row
    return accepted


def adapt_guard(runtime, accepted):
    original = runtime.input_stamps
    tree = ast.parse(inspect.getsource(original))
    hits = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'stamp']
    if len(hits) != 1 or ast.unparse(hits[0]) != "stamp(row['path'])":
        raise ValueError('Unexpected original input stamp guard')
    hits[0].func.id = 'audited_stamp'
    def audited_stamp(path):
        live = runtime.stamp(path)
        row = accepted.get(str(path))
        if row is not None:
            if live != row['accepted']:
                raise ValueError('Input changed again after SHA audit; stop and re-audit: '+str(path))
            return row['original']
        return live
    namespace = dict(original.__globals__, audited_stamp=audited_stamp)
    exec(compile(ast.fix_missing_locations(tree), '<uecp-byte-verified-input-stamps-v1>', 'exec'), namespace)
    runtime.input_stamps = namespace['input_stamps']
    alias = sys.modules.get('uecp_gspr_v2xr.runtime')
    if alias is not None:
        if alias.input_stamps is not original:
            raise ValueError('Unexpected preexisting UECP input guard binding')
        alias.input_stamps = runtime.input_stamps


def install(bridge, archive, root, command):
    original_install = bridge.install_runner
    def installed(repair, diagnose, probe, prior_archive, legacy, actual_root, project, snapshots, bindings):
        if actual_root != root:
            raise ValueError('Unexpected output for input audit installation')
        base_compile = repair.compile_worker_command
        original_install(repair, diagnose, probe, prior_archive, legacy, root, project, snapshots, bindings)
        # This executes all previous source/contract/evidence checks before file approval.
        repair.lineage(root, project, legacy)
        if command == 'run':
            with bridge.stopped(root):
                from uecp_gspr_v2xr.common import complete
                complete(root / 'E0/baseline/epochs/03/train_complete.json')
                path = audit_inputs(root, archive)
                os.environ[AUDIT_ENV], os.environ[HASH_ENV] = str(path), sha(path)
        elif not os.environ.get(AUDIT_ENV) or not os.environ.get(HASH_ENV):
            raise ValueError('Worker requires its controller input audit')
        path, digest = os.environ[AUDIT_ENV], os.environ[HASH_ENV]
        accepted = verify_audit(root, archive, path, digest)
        runtime = sys.modules['v2xr_raygen_e0e3.runtime']
        adapt_guard(runtime, accepted)
        runtime.input_stamps(root)  # complete coverage + unchanged upstream source checks
        repair.compile_worker_command = lambda source, namespace, ignored, project: base_compile(source, namespace, archive, project)
        provenance = dict(audit=str(path), audit_sha256=digest, launcher_sha256=sha(archive),
                          changed_training_bytes_verified=True, historical_transient_changes_not_observable=True)
        old_bind = repair.bind_commit
        def bind(*a):
            old_bind(*a)
            from uecp_gspr_v2xr import common
            original_commit = common.commit
            def commit(path, outputs, **fields):
                return original_commit(path, outputs, **dict(fields, input_ctime_audit=provenance))
            common.commit = commit
        repair.bind_commit = bind
        old_report = repair.amended_report
        def report(output, marker, smoke=False, partial=False):
            target = Path(output) / ('smoke' if smoke else '') / ('PARTIAL_RESULTS.json' if partial else 'FINAL_RESULTS.json')
            done = target.exists() and read(target).get('status') == 'COMPLETE'
            result = old_report(output, marker, smoke, partial)
            if not done:
                from uecp_gspr_v2xr.common import write
                result['input_ctime_audit'] = provenance
                write(target, result)
                with target.with_suffix('.md').open('a', encoding='utf-8') as stream:
                    stream.write('\n\n训练文件状态时间恢复审计：'+str(path)+'。仅放行原始内容SHA一致的ctime变化；'
                                 '未修改原始清单和模型计算。当前内容一致不证明历史上从未发生短暂更改。\n')
            return result
        repair.amended_report = report
        print('VERIFIED SHA-backed training ctime audit; original source/model/AP rules retained', flush=True)
    bridge.install_runner = installed


def verify_zip(path):
    path = Path(path).resolve(strict=True)
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        files = json.loads(z.read('manifest.json'))['files']
        if len(names) != len(set(names)) or set(names) != set(files) | {'manifest.json'}:
            raise ValueError('Input recovery archive member set differs')
        for name, expected in files.items():
            if hashlib.sha256(z.read(name)).hexdigest() != expected:
                raise ValueError('Input recovery archive member differs: '+name)
        if hashlib.sha256(z.read('prior/dependency_snapshot_v1.zip')).hexdigest() != PRIOR_SHA:
            raise ValueError('Original dependency snapshot ZIP differs')
    return path


def main():
    if sys.platform != 'linux':
        raise RuntimeError('Server execution only; local checks must use synthetic fixtures')
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('command', choices=('run', 'worker'))
    p.add_argument('action', nargs='?')
    p.add_argument('--output', required=True)
    args, _ = p.parse_known_args()
    archive = verify_zip(sys.argv[0])
    root = Path(args.output).resolve(strict=True)
    if Path('/data/cjm/datasets/logs') not in root.parents:
        raise ValueError('Require the original UECP output under /data/cjm/datasets/logs')
    if not (root / 'source_repair/uecp_dependency_snapshot_v1.json').is_file():
        raise ValueError('Requires the previously enrolled dependency-snapshot recovery')
    config = read(root / 'config.json')
    seed = '20261016' if args.command == 'worker' and args.action in ('dev', 'final') else str(config['seed'])
    if os.environ.get('PYTHONHASHSEED') != seed:
        os.execve(sys.executable, [sys.executable, '-B', str(archive), *sys.argv[1:]], dict(os.environ, PYTHONHASHSEED=seed))
    # Prevent both new- and old-entry controllers from overlapping the audit.
    if args.command == 'run':
        import fcntl
        lease = (root / '.ctime_resume_controller.lock').open('a+')
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    prior = root / PREFIX / 'dependency_snapshot_v1.zip'
    if not prior.resolve().is_relative_to(root):
        raise ValueError('Private prior ZIP escaped output')
    # The exact old bridge is embedded; no reliance on copies elsewhere on the server.
    with zipfile.ZipFile(archive) as z:
        prior.parent.mkdir(parents=True, exist_ok=True)
        if not prior.exists():
            with prior.open('xb') as stream:
                stream.write(z.read('prior/dependency_snapshot_v1.zip'))
    if sha(prior) != PRIOR_SHA:
        raise ValueError('Private prior snapshot archive changed')
    sys.path.insert(0, str(prior))
    import uecp_gspr_dependency_resume as bridge
    if not str(bridge.__file__).startswith(str(prior)+os.sep):
        raise ValueError('Prior bridge was not imported from the exact embedded ZIP')
    install(bridge, archive, root, args.command)
    with bridge.argv0(prior):
        bridge.main()


if __name__ == '__main__':
    main()
