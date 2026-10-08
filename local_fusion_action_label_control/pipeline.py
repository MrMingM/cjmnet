"""Offline control or full replay, always reading SOURCE_RUN and writing RUN."""
from pathlib import Path
import argparse
import signal
import subprocess
import sys
import traceback
from .common import (ROOT, HERE, WEATHERS, Manifest, atomic_json, code_hashes, digest,
                     isolated_paths, object_hash, optional_commit, read_json, settings)
from .inputs import Source, control_source

OFFLINE_STAGES = ('test_core', 'LABEL_AUDIT', 'TRAIN', 'CALIBRATION', 'PROBE', 'OFFLINE')
FULL_STAGES = OFFLINE_STAGES + ('REPLAY', 'SOURCE_S3', 'FINAL', 'integrity')


def prepare(source_run, run, config):
    source_run, run = isolated_paths(source_run, run, create=True)
    source = Source(source_run)
    source.verify_dependencies()
    missing = source.readiness()
    atomic_json(run / 'PREFLIGHT.json', {'source_run': str(source_run), 'missing': missing,
        'ready_for_offline': not missing, 'baseline_complete': source.complete('baseline'),
        's3_complete': source.complete('S3'), 'test_data_used': False})
    if missing:
        raise RuntimeError('SOURCE_RUN incomplete; S0 is never regenerated:\n  ' + '\n  '.join(missing))
    spec = settings(config)
    source_files = {'protocol.json': digest(source_run / 'protocol.json')}
    for name, stage in (('S0_RESULTS.json', 'S0'), ('S1_RESULTS.json', 'S1'), ('S2_RESULTS.json', 'S2'),
                        ('feature_schema.json', 'S0'), ('feature_normalization.json', 'S2-fit'),
                        ('linear_cls.pt', 'S1-fit'), ('linear_reg.pt', 'S2-fit')):
        source_files[name] = digest(source.artifact(name, stage))
    from local_fusion_action_utility_audit.features import schema
    schema_file = read_json(source_run / 'feature_schema.json')
    for task in ('cls', 'reg'):
        if schema_file[task] != schema(task) or schema_file.get('inference_gt_fields') != 0:
            raise ValueError('Original feature schema/leakage contract differs')
        source.original_state(task)
    cache_index = {}
    for split in ('train', 'validation'):
        for weather in WEATHERS:
            for row in source.frames(split, weather):
                name = f"cache/{split}/{weather}/{row['frame']:08d}.pt"
                cache_index[name] = source.entry('S0-frame', split, weather, row['frame'])['artifacts'][name]
            print(f'PREFLIGHT {split}/{weather} cache hashes/schema/scenes checked', flush=True)
    hashes = code_hashes()
    identity = {'control_source_hash': object_hash(hashes), 'config_hash': object_hash(spec),
        'source_protocol_hash': source_files['protocol.json'], 'source_input_hash': object_hash(source_files),
        'source_cache_index_hash': object_hash(cache_index),
        'checkpoint_hashes': source.protocol['identity']['checkpoint_hashes']}
    manifest = Manifest(run)
    if manifest.value['identity'] and manifest.value['identity'] != identity:
        raise ValueError('Control resume identity changed; source caches/code/config must remain identical')
    manifest.value['identity'] = identity
    manifest.save()
    p = {'purpose': 'Action ranking label control; two groups only', 'source_run': str(source_run),
         'source_protocol': source.protocol, 'source_identity': source.protocol['identity'],
         'source_files': source_files, 'control_source_hashes': hashes, 'identity': identity,
         'control_commit_metadata': optional_commit(), 'commit_is_runtime_requirement': False,
         'config': spec, 'test_data_used': False, 'inference_gt_fields': 0,
         'training_settings': source.spec, 'features': 'with_competition_features',
         'original_weights': 'loaded unchanged; original historical threshold retained as reference only',
         'shared_runtime': 'source loader/predict methods; read-only content-verified constructor adapter',
         'source_protection': 'no original Manifest or original Runtime constructor; no input writes',
         'normalization': 'original probe-fit parameters; no new normalization fit',
         'sample_weighting': 'original candidate-target groups retained without deduplication'}
    if (run / 'protocol.json').exists():
        previous = read_json(run / 'protocol.json')
        # Commits are provenance only. Adding this module may change HEAD without
        # changing the original dependency files. Content checks remain mandatory.
        p['control_commit_metadata'] = previous['control_commit_metadata']
        if previous != p:
            raise ValueError('Control protocol changed')
    atomic_json(run / 'protocol.json', p)
    atomic_json(run / 'source_cache_index.json', cache_index)
    atomic_json(run / 'feature_schema.json', schema_file)
    atomic_json(run / 'feature_normalization.json', read_json(source_run / 'feature_normalization.json'))
    atomic_json(run / 'SOURCE_S3_CHECK.json', {'status': 'pending'}) if not (run / 'SOURCE_S3_CHECK.json').exists() else None
    return manifest


def verify_inputs(run, baseline=False):
    from .common import protocol
    p = protocol(run)
    source = control_source(run, p)
    source.verify_dependencies()
    for name, expected in p['source_files'].items():
        if digest(source.root / name) != expected:
            raise ValueError('Source input changed after control preflight: ' + name)
    for name, expected in source.cache_index.items():
        _, split, weather, frame = Path(name).parts
        recorded = source.entry('S0-frame', split, weather, int(Path(frame).stem)).get('artifacts', {}).get(name)
        if recorded != expected:
            raise ValueError('Source cache manifest changed after control preflight: ' + name)
    if baseline:
        missing = source.readiness(True)
        if missing:
            raise RuntimeError('Replay requires completed baseline: ' + ', '.join(missing))
        source.artifact('baseline_reproduction.json', 'baseline')
    return source


def stage_artifacts(run, stage):
    if stage in ('LABEL_AUDIT', 'PROBE', 'REPLAY'):
        name = {'PROBE': 'PROBE_RESULTS', 'REPLAY': 'REPLAY_RESULTS'}.get(stage, stage)
        return [run / (name + '.' + ext) for ext in ('json', 'md')]
    if stage == 'TRAIN':
        from .common import GROUPS
        from .probes import state_path
        return [state_path(run, g, t) for g in GROUPS for t in ('cls', 'reg')]
    if stage == 'CALIBRATION':
        return [run / 'common_calibration.json']
    if stage == 'SOURCE_S3':
        return [run / 'SOURCE_S3_CHECK.json']
    if stage == 'FINAL':
        return [run / 'final_results.json', run / 'FINAL_RESULTS.md']
    return []


def integrity(run):
    source = verify_inputs(run, True)
    manifest = Manifest(run)
    for stage in FULL_STAGES[:-1]:
        if not manifest.complete(stage):
            raise RuntimeError('Incomplete control stage: ' + stage)
    for name in ('protocol.json', 'feature_normalization.json', 'feature_schema.json', 'source_cache_index.json', 'driver.log'):
        if not (Path(run) / name).is_file():
            raise RuntimeError('Missing final artifact: ' + name)
    for split in ('train', 'validation'):
        for weather in WEATHERS:
            for index in source.protocol[split + '_indices']:
                if not manifest.complete('audit-frame', split, weather, index):
                    raise RuntimeError('Incomplete label audit frame')
                if split == 'validation' and not manifest.complete('replay-frame', split, weather, index):
                    raise RuntimeError('Incomplete replay frame')


def run_core_tests():
    result = subprocess.run([sys.executable, '-B', '-m',
        'local_fusion_action_label_control.test_core', '--require-server'],
        cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(result.stdout, end='', flush=True)
    result.check_returncode()


def run(args):
    source, output = isolated_paths(args.source_run, args.run, True)
    stage = 'preflight'
    manifest = Manifest(output)
    try:
        with manifest.work(stage):
            manifest = prepare(source, output, args.config)
        manifest.mark(stage, 'complete', artifacts=[output / name for name in (
            'protocol.json', 'source_cache_index.json', 'feature_schema.json', 'feature_normalization.json')])
        if args.mode == 'preflight':
            print('PREFLIGHT ready; no training or model forward executed', flush=True)
            return
        from .audit import run_audit
        from .probes import fit, calibrate
        from .evaluate import evaluate
        jobs = [('test_core', run_core_tests),
                ('LABEL_AUDIT', lambda: run_audit(output)), ('TRAIN', lambda: fit(output)),
                ('CALIBRATION', lambda: calibrate(output)), ('PROBE', lambda: evaluate(output)),
                ('OFFLINE', lambda: print('OFFLINE complete; full replay and FINAL remain pending', flush=True))]
        if args.mode == 'all':
            from .replay import replay, compare_original_s3
            from .summarize import summarize
            jobs += [('REPLAY', lambda: replay(output)),
                     ('SOURCE_S3', lambda: compare_original_s3(output)),
                     ('FINAL', lambda: summarize(output)), ('integrity', lambda: integrity(output))]
        for stage, job in jobs:
            verify_inputs(output, baseline=stage in ('REPLAY', 'SOURCE_S3', 'FINAL', 'integrity'))
            manifest = Manifest(output)
            if manifest.complete(stage):
                print(stage + ' reuse complete', flush=True)
                continue
            print('STAGE START ' + stage, flush=True)
            with manifest.work(stage):
                job()
            manifest.mark(stage, 'complete', artifacts=stage_artifacts(output, stage))
            print(stage + ' complete', flush=True)
        print('CONTROL COMPLETE' if args.mode == 'all' else 'OFFLINE ONLY; experiment completion pending', flush=True)
    except BaseException as exc:
        Manifest(output).mark(stage, 'failed', error=repr(exc))
        print(f'STAGE FAILED {stage}: {exc}', file=sys.stderr, flush=True)
        raise


def main():
    import os
    import fcntl
    signal.signal(signal.SIGTERM, lambda n, f: sys.exit(128 + n))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', required=True)
    parser.add_argument('--run', required=True)
    parser.add_argument('--config', default=str(HERE / 'experiment.yaml'))
    parser.add_argument('--mode', choices=('preflight', 'offline', 'all'), default='offline')
    args = parser.parse_args()
    _, output = isolated_paths(args.source_run, args.run, True)
    with (output / '.control.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another control process is using this RUN')
        class Tee:
            def __init__(self, stream, log):
                self.stream, self.log = stream, log
            def write(self, text):
                self.log.write(text)
                if not os.environ.get('LABEL_CONTROL_REDIRECT'):
                    self.stream.write(text)
            def flush(self):
                self.log.flush()
                self.stream.flush()
        with (output / 'driver.log').open('a', encoding='utf-8', buffering=1) as log:
            stdout, stderr = sys.stdout, sys.stderr
            sys.stdout, sys.stderr = Tee(stdout, log), Tee(stderr, log)
            try:
                run(args)
            except BaseException:
                traceback.print_exc()
                raise SystemExit(1)
            finally:
                sys.stdout, sys.stderr = stdout, stderr


if __name__ == '__main__':
    main()
