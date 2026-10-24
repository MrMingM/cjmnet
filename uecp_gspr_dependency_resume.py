"""Additive source-snapshot launcher for the enrolled UECP resume-v2 experiment.

The original v2 runner and its policy are embedded byte-for-byte. Only three shared
Python modules are loaded from private, reviewed snapshots in each fresh process.
No project source, dependency pins, request, contract or old ZIP is rewritten.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import zipfile

PREFIX = 'source_repair/uecp_dependency_snapshot_v1'
MARKER = 'source_repair/uecp_dependency_snapshot_v1.json'
LEGACY_SHA = 'eef5dbb83f03b3112c582a458040761d708495df1fb309f302383dc0e46e72ca'
MODULES = ('runtime', 'preflight', 'training')
ORIGINAL = {
    'runtime': 'de6db71df8fbcdc9fd1fb176f344ecf8ce887203b362781365bc94ff78a2d24f',
    'preflight': '82be740472c4cf2e2c1fc65b341f25b8fc1f53e796bd9f4a4957718c6a8717f4',
    'training': 'b6577e83734086bc760cde237eb99c9086ac2a6d514e3fa0bf1ae7a44ba6c129'}
SNAPSHOT = dict(ORIGINAL, training='cc5f523e64cb34814208a5901ef30be45712701ef0f00a0e4892accf9a683793')
CURRENT = {
    'runtime': '0e48bcf2e17c11e492951c9cdc5b9ae453b2cf4884f05072aec31d3b7e41079e',
    'preflight': '27489af2a85dd147746d6025a72d17e5acba9400d8b16c85a7a88b8bc5d1f5b7',
    'training': '1952a740a40804d36eb186ba1121e77cac219d593eb9193939648ab4a636d4ad'}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def identity(value):
    return digest(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(part)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def immutable_bytes(path, data):
    path = Path(path)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError('Existing private snapshot differs: '+str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.'+str(os.getpid())+'.'+str(time.time_ns())+'.tmp')
    with temporary.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def immutable_json(path, value):
    immutable_bytes(path, (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+'\n').encode())


def verify_zip(path):
    path = Path(path).resolve(strict=True)
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read('manifest.json'))
        if len(archive.namelist()) != len(set(archive.namelist())) or set(archive.namelist()) != set(manifest['files']) | {'manifest.json'}:
            raise ValueError('Snapshot launcher ZIP member set differs')
        for name, expected in manifest['files'].items():
            if digest(archive.read(name)) != expected:
                raise ValueError('Snapshot launcher member changed: '+name)
        if digest(archive.read('legacy/uecp_gspr_resume_fix_v2.zip')) != LEGACY_SHA:
            raise ValueError('Embedded original v2 runner differs')
        for name in MODULES:
            if digest(archive.read('snapshots/'+name+'.py')) != SNAPSHOT[name]:
                raise ValueError('Embedded source snapshot differs: '+name)
    return path


def prepare_files(archive, root):
    destination = root / PREFIX
    if destination.resolve().is_relative_to(root.resolve()) is False:
        raise ValueError('Private source directory escaped output')
    with zipfile.ZipFile(archive) as z:
        legacy = destination / 'original_resume_v2.zip'
        targets = [legacy, destination / 'SOURCE_REVIEW.json']+[
            destination / 'modules' / (n+'.py') for n in MODULES]
        if any(not p.resolve().is_relative_to(root.resolve()) for p in targets):
            raise ValueError('Private snapshot file resolves outside this experiment output')
        immutable_bytes(legacy, z.read('legacy/uecp_gspr_resume_fix_v2.zip'))
        for name in MODULES:
            immutable_bytes(destination / 'modules' / (name+'.py'), z.read('snapshots/'+name+'.py'))
        immutable_bytes(destination / 'SOURCE_REVIEW.json', z.read('SOURCE_REVIEW.json'))
    return legacy, destination / 'modules'


class SnapshotProject:
    """The original v2 checker reads the exact modules this process will execute.

    All other dependencies and the entire sealed UECP package resolve to the real
    project. Physical shared-file drift is separately checked and recorded.
    """
    def __init__(self, project, snapshots):
        self.project, self.snapshots = Path(project), Path(snapshots)

    def __truediv__(self, relative):
        routes = {'v2xr_raygen_e0e3/'+n+'.py': self.snapshots / (n+'.py') for n in MODULES}
        return routes.get(str(relative), self.project / relative)


def physical_sources(project, recorded):
    rows = {}
    for name in MODULES:
        relative = 'v2xr_raygen_e0e3/'+name+'.py'
        if recorded['dependencies'].get(relative) != ORIGINAL[name]:
            raise ValueError('Original dependency pin differs: '+relative)
        actual = sha(Path(project) / relative)
        if actual not in {ORIGINAL[name], SNAPSHOT[name], CURRENT[name]}:
            raise ValueError('Unknown shared source transition; no execution: '+relative+' sha256='+actual)
        rows[relative] = dict(physical_sha256=actual, executed_snapshot_sha256=SNAPSHOT[name],
            scope='physical shared file is not imported by this UECP process')
    return rows


def load_snapshots(project, snapshots):
    sys.path.insert(0, str(project))
    import importlib.util
    package = importlib.import_module('v2xr_raygen_e0e3')
    expected_package = (Path(project) / 'v2xr_raygen_e0e3/__init__.py').resolve()
    if Path(package.__file__).resolve() != expected_package:
        raise ValueError('Unexpected shared-package namespace')
    bindings = {}
    for name in MODULES:
        key = 'v2xr_raygen_e0e3.'+name
        if key in sys.modules:
            raise ValueError('Shared helper imported before private source isolation: '+key)
        path = Path(snapshots) / (name+'.py')
        if sha(path) != SNAPSHOT[name]:
            raise ValueError('Private source snapshot changed: '+name)
        spec = importlib.util.spec_from_file_location(key, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        setattr(package, name, module)
        spec.loader.exec_module(module)
        bindings[key] = dict(path=str(path), sha256=sha(path))
    return bindings


def verify_bindings(bindings):
    for key, row in bindings.items():
        module = sys.modules.get(key)
        if module is None or Path(module.__file__).resolve() != Path(row['path']).resolve() or sha(row['path']) != row['sha256']:
            raise ValueError('Executed private module binding changed: '+key)


@contextmanager
def argv0(path):
    previous = sys.argv[0]
    sys.argv[0] = str(path)
    try:
        yield
    finally:
        sys.argv[0] = previous


def install_runner(repair, diagnose, probe, archive, legacy, root, project, snapshots, bindings):
    original_check = diagnose.diagnostic_sources
    def checked_sources(actual_project, recorded):
        if Path(actual_project).resolve() != project:
            raise ValueError('Unexpected project for source verification')
        physical_sources(project, recorded)
        verify_bindings(bindings)
        return original_check(SnapshotProject(project, snapshots), recorded)
    # The unchanged v2 lineage now verifies its actual frozen source view. It still
    # compares the complete source review with the original immutable enrollment.
    repair.diagnostic_sources = checked_sources
    diagnose.diagnostic_sources = checked_sources
    probe.diagnostic_sources = checked_sources
    repair.verify_archive = lambda: legacy
    original_command = repair.compile_worker_command
    repair.compile_worker_command = lambda source, namespace, ignored_legacy, actual_project: original_command(
        source, namespace, archive, actual_project)
    original_bind = repair.bind_commit
    def bind(output, marker):
        original_bind(output, marker)
        from uecp_gspr_v2xr import common
        original_commit = common.commit
        def commit(path, outputs, **fields):
            fields['execution_dependency_snapshot'] = dict(marker=str(root / MARKER),
                marker_sha256=sha(root / MARKER), launcher_sha256=sha(archive), executed_modules=bindings)
            return original_commit(path, outputs, **fields)
        common.commit = commit
    repair.bind_commit = bind
    original_report = repair.amended_report
    def report(output, marker, smoke=False, partial=False):
        target = Path(output) / ('smoke' if smoke else '')
        path = target / ('PARTIAL_RESULTS.json' if partial else 'FINAL_RESULTS.json')
        already_complete = path.exists() and read(path).get('status') == 'COMPLETE'
        result = original_report(output, marker, smoke, partial)
        if not already_complete:
            from uecp_gspr_v2xr.common import write
            result['execution_dependency_snapshot'] = dict(marker=str(root / MARKER),
                marker_sha256=sha(root / MARKER), launcher_sha256=sha(archive),
                original_resume_v2_sha256=LEGACY_SHA, executed_modules=bindings,
                original_model_loss_data_AP_policy_unchanged=True)
            write(path, result)
            md = path.with_suffix('.md')
            with md.open('a', encoding='utf-8') as stream:
                stream.write('\n\n## 共享源码隔离恢复\n\n本次UECP进程加载原先登记的辅助模块副本；'
                    '项目当前Raygen文件未修改，未启用其新增计算设置。模型、损失、数据、选模、AP和v2恢复检查规则不变。'
                    '隔离记录：'+str(root / MARKER)+'。\n')
        return result
    repair.amended_report = report


def enroll_record(archive, root, project, repair, diagnose, snapshots):
    recorded = read(root / 'request.json')['sources']
    physical = physical_sources(project, recorded)
    review = diagnose.diagnostic_sources(SnapshotProject(project, snapshots), recorded)
    prior = read(root / repair.MARKER)
    if prior['archive_sha256'] != LEGACY_SHA or prior['source_review'] != review:
        raise ValueError('Private source view does not match original v2 enrollment')
    # Validate every available epoch receipt before the controller can reuse training.
    from uecp_gspr_v2xr.common import complete, variant_path
    from uecp_gspr_v2xr.state_logic import valid_stage
    if not valid_stage(read(root / 'pipeline_status.json')['preflight']):
        raise ValueError('Original preflight receipt is not valid')
    complete(root / repair.MID_GATE)
    train = variant_path(root, 'E0') / 'epochs/01/train_complete.json'
    if not train.exists():
        raise ValueError('This recovery requires the already committed formal E0 epoch1')
    complete(train)
    marker = dict(revision=1, family='UECP_dependency_snapshot_resume',
        output=str(root), project=str(project), launcher_sha256=sha(archive),
        original_resume_v2_sha256=LEGACY_SHA, original_v2_marker_sha256=sha(root / repair.MARKER),
        request_sha256=sha(root / 'request.json'), contract_sha256=sha(root / 'contract.json'),
        source_view_review=review, executed_snapshot_sha256=SNAPSHOT,
        original_dependency_sha256=ORIGINAL, known_physical_versions=CURRENT,
        scientific_policy='Unchanged original model/data/loss/optimizer/selection/AP and v2 resume policy',
        private_scope='runtime/bootstrap, preflight/content helper, training/RNG/atomic helpers',
        original_files_unchanged=True)
    marker['identity'] = identity(marker)
    immutable_json(root / MARKER, marker)
    observation = dict(physical_sources=physical, preserved_E0_epoch1_commit_sha256=sha(train),
        private_source_marker_sha256=sha(root / MARKER),
        time_ns=time.time_ns(), pid=os.getpid())
    immutable_json(root / PREFIX / 'observations' / (str(observation['time_ns'])+'_'+str(os.getpid())+'.json'), observation)
    return marker


@contextmanager
def stopped(root):
    """Check both original kernel locks before creating any private source files."""
    import fcntl
    from contextlib import ExitStack
    with ExitStack() as stack:
        for name in ('.controller.lock', '.worker.lock'):
            stream = stack.enter_context((Path(root) / name).open('r'))
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError('Original controller/worker is active; no recovery started') from error
        yield


def main():
    if sys.platform != 'linux':
        raise RuntimeError('UECP recovery execution is server-only; local verification is synthetic/static only')
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('command', choices=('run', 'worker'))
    parser.add_argument('action', nargs='?')
    parser.add_argument('--output', required=True)
    parser.add_argument('--project', default='/home/cjm/OpenCOOD-main/cjmnet')
    parser.add_argument('--check-only', action='store_true')
    args, _ = parser.parse_known_args()
    if args.check_only and args.command != 'run':
        raise ValueError('--check-only is for controller source verification')
    archive = verify_zip(sys.argv[0])
    root, project = Path(args.output).resolve(strict=True), Path(args.project).resolve(strict=True)
    if Path('/data') not in root.parents or any(p.lower() in ('test', 'testing') for p in root.parts):
        raise ValueError('Output must be the original non-test /data experiment')
    if not (root / 'source_repair/uecp_resume_v2.json').is_file():
        raise ValueError('Original resume-v2 enrollment is required; no new experiment is created')
    # Only the controller may create the private snapshot. Workers verify existing files.
    if args.command == 'run':
        # This lock is separate from the original controller/worker locks.
        import fcntl
        lease = (root / '.dependency_snapshot_controller.lock').open('a+')
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    legacy = root / PREFIX / 'original_resume_v2.zip'
    snapshots = root / PREFIX / 'modules'
    if args.command == 'run':
        with stopped(root):
            prepare_files(archive, root)
    if not legacy.is_file() or sha(legacy) != LEGACY_SHA:
        raise ValueError('Private original v2 ZIP is missing or changed')
    sys.path.insert(0, str(legacy))
    import uecp_gspr_resume_repair as repair
    import uecp_gspr_resume_diagnose as diagnose
    import uecp_gspr_resume_probe as probe
    for module in (repair, diagnose, probe):
        if not str(module.__file__).startswith(str(legacy)+os.sep):
            raise ValueError('Original v2 module was not loaded from the verified private ZIP: '+module.__name__)
    with argv0(legacy):
        probe.verify_archive()
    env = probe.environment(root, project)
    env['PYTHONHASHSEED'] = repair.worker_hash_seed(args.command, args.action, env['PYTHONHASHSEED'])
    if os.environ.get('PYTHONHASHSEED') != env['PYTHONHASHSEED']:
        os.execve(sys.executable, [sys.executable, '-B', str(archive), *sys.argv[1:]], env)
    os.environ.clear()
    os.environ.update(env)
    sys.path.insert(0, str(project))
    from uecp_gspr_v2xr.paths import check
    check(root)
    if args.command == 'run':
        with diagnose.stopped_pipeline(root):
            enroll_record(archive, root, project, repair, diagnose, snapshots)
    elif not (root / MARKER).is_file():
        raise ValueError('Controller snapshot enrollment is missing')
    marker = read(root / MARKER)
    if marker['identity'] != identity({k: v for k, v in marker.items() if k != 'identity'}):
        raise ValueError('Private snapshot enrollment content hash changed')
    for name, expected in (('launcher_sha256', sha(archive)), ('original_v2_marker_sha256', sha(root / repair.MARKER)),
            ('request_sha256', sha(root / 'request.json')), ('contract_sha256', sha(root / 'contract.json'))):
        if marker[name] != expected:
            raise ValueError('Snapshot lineage changed: '+name)
    recorded = read(root / 'request.json')['sources']
    physical_sources(project, recorded)
    bindings = load_snapshots(project, snapshots)
    observation = dict(pid=os.getpid(), time_ns=time.time_ns(), command=args.command, action=args.action,
        physical_sources=physical_sources(project, recorded), executed_modules=bindings,
        marker_identity=marker['identity'], hash_seed=os.environ['PYTHONHASHSEED'])
    immutable_json(root / PREFIX / 'observations' / ('process_'+str(observation['time_ns'])+'_'+str(os.getpid())+'.json'), observation)
    install_runner(repair, diagnose, probe, archive, legacy, root, project, snapshots, bindings)
    repair.lineage(root, project, legacy)  # original immutable lineage checks remain in force
    print('VERIFIED private UECP helpers; original E0 epoch1 training retained; no Raygen compute-policy update', flush=True)
    if args.check_only:
        print('CHECK_ONLY_COMPLETE; no model load, forward, backward or training update', flush=True)
        return
    repair.main()


if __name__ == '__main__':
    main()
