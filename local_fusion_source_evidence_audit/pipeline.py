"""Single resumable server driver: exact old-result replay, evidence, probes, AP, reports."""
import argparse
from pathlib import Path
import os
import signal
import subprocess
import sys
import time
import traceback
from .common import (ROOT, PACKAGE, DEFAULT_BASE, Manifest, WEATHERS, atomic_json,
                     contract, digest, external_run, object_hash, read_json, safe_path,
                     settings, source_hashes, verify_base)


def prepare(run, base):
    import yaml
    from .coordinates import specification
    from .features import schema_document
    from .reproduction import POLICY as output_feature_policy
    if read_json(PACKAGE/'source_snapshot.json')['sources'] != source_hashes():
        raise ValueError('Server S4 files differ from source_snapshot.json; sync the entire S4 directory')
    base_protocol, base_identity = verify_base(base)
    # Read plain YAML and resolve split aliases before any dataset enumeration.
    front = yaml.safe_load(Path(base_protocol['inputs']['frontend_config']).read_text(encoding='utf-8'))
    options = yaml.safe_load(Path(base_protocol['inputs']['v3_config']).read_text(encoding='utf-8'))
    roots = {k: safe_path(options.get(k) or front.get(k), dataset=True) for k in ('root_dir', 'validate_dir')}
    if roots['root_dir'] == roots['validate_dir'] or roots['root_dir'] in roots['validate_dir'].parents or roots['validate_dir'] in roots['root_dir'].parents:
        raise ValueError('Train and validation roots overlap')
    for key, path in roots.items():
        if str(path) != str(Path(base_protocol['dataset_paths'][key]).resolve()):
            raise ValueError('Dataset root differs from original run')
    from .common import guard_dataset_tree
    for root in roots.values():
        guard_dataset_tree(root)
    # Scene numbers are local to each split. Compare actual scene names, not
    # integer scene IDs which legitimately overlap across separate roots.
    scene_names = [{p.name for p in root.iterdir() if p.is_dir()} for root in roots.values()]
    if scene_names[0] & scene_names[1]:
        raise ValueError('Train/validation scenario names overlap')
    schema, coordinates = schema_document(), specification()
    paths = {'feature_schema.json': schema, 'coordinate_contract.json': coordinates}
    for name, value in paths.items():
        path = Path(run)/name
        if path.exists() and read_json(path) != value:
            raise ValueError('Resume schema changed; use new RUN with repair record: '+name)
        if not path.exists():
            atomic_json(path, value)
    dependencies = dict(base_protocol['source_hashes'])
    dependencies.update(coordinates['source_hashes'])
    for name, expected in dependencies.items():
        if digest(safe_path(ROOT/name)) != expected:
            raise ValueError('Base/frozen dependency changed: '+name)
    identity = {'source_hash': object_hash(source_hashes()), 'dependency_source_hash': object_hash(dependencies),
                'base_hash': object_hash(base_identity), 'config_hash': object_hash(settings()),
                'checkpoint_hashes': base_identity['checkpoint_hashes'],
                'feature_schema_hash': digest(Path(run)/'feature_schema.json'),
                'coordinate_contract_hash': digest(Path(run)/'coordinate_contract.json')}
    protocol = {'schema': 1, 'purpose': 'Evidence -> Utility -> Action diagnostic, fixed linear comparison',
                'base': base_identity, 'base_protocol': base_protocol, 'sources': source_hashes(),
                'dependency_sources': dependencies, 's4_config': settings(), 'identity': identity,
                'dataset_roots': {k: str(v) for k, v in roots.items()}, 'test_data_used': False,
                'inference_gt_fields': 0, 'primary_group': 'all_evidence', 'version_check': 'source_sha256_only',
                'reproducibility': base_protocol['shared_reproducibility'],
                'output_feature_reproduction': output_feature_policy,
                'repair_policy': 'No automatic migration. Source/schema/base drift requires a new RUN and an explicit repair record.',
                'hardware': {'ROCR_VISIBLE_DEVICES': os.environ.get('ROCR_VISIBLE_DEVICES'), 'python': sys.executable}}
    path = Path(run)/'protocol.json'
    if path.exists() and read_json(path) != protocol:
        raise ValueError('Resume protocol/base/environment differs; use a new RUN')
    atomic_json(path, protocol)
    manifest = Manifest(run)
    if manifest.value['identity'] and manifest.value['identity'] != identity:
        raise ValueError('Resume identity changed')
    manifest.value['identity'] = identity
    manifest.save()
    contract(run)


def stages(run):
    prefix = [sys.executable, '-u', '-m']
    mod = 'local_fusion_source_evidence_audit.'
    suffix = ['--run', str(run)]
    return [
        ('server-tests', prefix+[mod+'test_core', '--require-server']),
        ('R0', prefix+[mod+'r0']+suffix),
        ('evidence-train', prefix+[mod+'extract']+suffix+['--split', 'train']),
        ('evidence-validation', prefix+[mod+'extract']+suffix+['--split', 'validation']),
        ('GEO-fit', prefix+[mod+'ranker']+suffix+['--task', 'reg', '--mode', 'fit']),
        ('GEO', prefix+[mod+'ranker']+suffix+['--task', 'reg', '--mode', 'evaluate']),
        ('SEM-fit', prefix+[mod+'ranker']+suffix+['--task', 'cls', '--mode', 'fit']),
        ('SEM', prefix+[mod+'ranker']+suffix+['--task', 'cls', '--mode', 'evaluate']),
        ('ABLATIONS', prefix+[mod+'summarize']+suffix+['--mode', 'ablations']),
        ('REPLAY', prefix+[mod+'replay']+suffix),
        ('FINAL', prefix+[mod+'summarize']+suffix+['--mode', 'final'])]


def artifacts(run, stage):
    files = {'R0': ['R0_REPRODUCTION.json', 'R0_REPRODUCTION.md'],
             'GEO': ['S4_GEO_RESULTS.json', 'S4_GEO_RESULTS.md'],
             'SEM': ['S4_SEM_RESULTS.json', 'S4_SEM_RESULTS.md'],
             'GEO-fit': ['linear_geo.pt', 'normalization_geo.json'],
             'SEM-fit': ['linear_sem.pt', 'normalization_sem.json'],
             'ABLATIONS': ['S4_ABLATIONS.json', 'S4_ABLATIONS.md'],
             'REPLAY': ['S4_REPLAY_RESULTS.json', 'S4_REPLAY_RESULTS.md'],
             'FINAL': ['final_results.json', 'FINAL_RESULTS.md']}
    return [Path(run)/p for p in files.get(stage, [])]


def integrity(run, require_integrity=False):
    from .features import GROUP_BLOCKS
    from gspr_communication.runtime import verify_frozen
    protocol = contract(run)
    manifest = Manifest(run)
    for stage, _ in stages(run):
        if not manifest.complete(stage):
            raise ValueError('Incomplete stage: '+stage)
    for split in ('train', 'validation'):
        for weather in WEATHERS:
            for frame in protocol['base_protocol'][split+'_indices']:
                if not manifest.complete('evidence-frame', split, weather, frame):
                    raise ValueError('Incomplete evidence frame')
    for task in ('cls', 'reg'):
        for group in GROUP_BLOCKS:
            if group != 'output_only' and not manifest.complete('probe-fit', task, group):
                raise ValueError('Incomplete ablation model')
    for weather in WEATHERS:
        if not manifest.complete('replay-weather', 'validation', weather):
            raise ValueError('Incomplete replay weather')
        for frame in protocol['base_protocol']['validation_indices']:
            if not manifest.complete('replay-frame', 'validation', weather, frame):
                raise ValueError('Incomplete replay frame')
    for name in ('driver.log', 'feature_schema.json', 'coordinate_contract.json'):
        if not (Path(run)/name).is_file() or not (Path(run)/name).stat().st_size:
            raise ValueError('Missing final file: '+name)
    # Original labels, source hashes, checkpoints and frame artifacts must still
    # match, even after all S4 stages have run. BASE_RUN is never written.
    _, actual = verify_base(protocol['base']['base_run'])
    if actual != protocol['base']:
        raise ValueError('BASE_RUN changed while S4 was running')
    verify_frozen()
    if require_integrity and not manifest.complete('integrity'):
        raise ValueError('Final integrity stage incomplete')


def main():
    import fcntl
    signal.signal(signal.SIGTERM, lambda signum, _frame: sys.exit(128+signum))
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--run', required=True)
    cli.add_argument('--base-run', default=DEFAULT_BASE)
    args = cli.parse_args()
    run, base = external_run(args.run, True), external_run(args.base_run)
    if run == base or run in base.parents or base in run.parents:
        raise ValueError('RUN must be separate from read-only BASE_RUN')
    with (run/'.driver.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another driver is using this RUN')
        stage = 'preflight'
        try:
            print('S4 preflight: verify complete BASE_RUN artifacts, unchanged labels/checkpoints and preregistered schemas', flush=True)
            prepare(run, base)
            manifest = Manifest(run)
            manifest.mark('preflight', 'complete', artifacts=[run/'protocol.json', run/'feature_schema.json', run/'coordinate_contract.json'])
            for stage, command in stages(run):
                contract(run)
                manifest = Manifest(run)
                if manifest.complete(stage):
                    print(stage+' reuse complete', flush=True)
                    continue
                print('STAGE START '+stage, flush=True)
                started = time.monotonic()
                with manifest.work(stage):
                    subprocess.run(command, cwd=ROOT, check=True)
                manifest.mark(stage, 'complete', artifacts=artifacts(run, stage))
                print(f'{stage} complete elapsed_seconds={time.monotonic()-started:.1f}', flush=True)
            stage = 'integrity'
            with manifest.work(stage):
                integrity(run)
            manifest.mark(stage, 'complete')
            atomic_json(run/'exit_status.json', {'exit_code': 0, 'integrity': 'complete'})
            for name in ('R0', 'GEO', 'SEM', 'ABLATIONS', 'REPLAY', 'FINAL'):
                print(name+' complete', flush=True)
            print('FINAL_RESULTS='+str(run/'FINAL_RESULTS.md'), flush=True)
        except BaseException as exc:
            Manifest(run).mark(stage, 'failed', error=repr(exc))
            atomic_json(run/'exit_status.json', {'exit_code': 1, 'failed_stage': stage, 'error': repr(exc)})
            print(f'STAGE FAILED {stage}: {exc}', file=sys.stderr, flush=True)
            traceback.print_exc()
            raise


if __name__ == '__main__':
    main()
