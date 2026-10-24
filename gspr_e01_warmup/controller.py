"""One server launch: verified existing E0, four formal 2-epoch gates, same models to epoch 20."""
import argparse
import os
from pathlib import Path
import sys
import traceback
from contextlib import ExitStack

from warm_common import HERE, POLICY, base_sources, bridge, frozen_worker, read, sha, short_temp, signal_handlers


def existing_e0(cli, base):
    legacy = base.safe(cli.legacy_run_root).resolve(strict=True)
    old = read(legacy / 'request.json')
    if old['identity'] != base.identity({k: v for k, v in old.items() if k != 'identity'}) or old['source'] != base_sources():
        raise ValueError('Legacy E01 request/source identity differs')
    if Path(old['output']).resolve() != legacy or Path(old['original']).resolve() != cli.original_root.resolve():
        raise ValueError('Legacy output / original C-D identity differs')
    if Path(old['train']).resolve() != cli.train_root.resolve():
        raise ValueError('Original training data root differs')
    candidate = cli.e0_existing_root or (Path(old['E0_existing']) if old.get('E0_existing') else legacy / 'E0')
    candidate, _ = frozen_worker().verified_e0_source(candidate, cli.original_root)
    return candidate, dict(legacy_root=str(legacy), request_sha256=sha(legacy / 'request.json'))


class Controller:
    def __init__(self, cli, base, request):
        self.cli, self.base, self.root, self.request = cli, base, cli.output_root, request

    def command(self, stage, model, *flags):
        return [sys.executable, '-B', HERE / 'worker.py', stage, '--model-root', model,
                '--train-root', self.cli.train_root, '--original-root', self.cli.original_root, *flags]

    def verify(self, name):
        folder = self.root / '.stages' / name
        wanted = self.base.identity(dict(parent=self.request['identity'], stage=name))
        if self.base.receipt(folder, wanted) is None:
            return False
        for path, digest in read(folder / 'artifacts.json')['files'].items():
            if sha(path) != digest:
                raise ValueError('Committed stage artifact changed: ' + path)
        return True

    def stage(self, name, command, artifacts):
        if self.verify(name):
            print('REUSE verified ' + name, flush=True)
            return
        folder = self.root / '.stages' / name
        self.base.atomic(self.root / 'current_stage.json', dict(status='running', stage=name,
            request=self.request['identity']))
        environment = self.base.gpu_env(self.cli.gpu)
        environment.pop('E01_CHILD', None)
        self.base.run(command, self.root / 'logs' / (name + '.log'), env=environment)
        files = artifacts()
        if not files:
            raise ValueError('No verified artifacts for ' + name)
        self.base.atomic(folder / 'artifacts.json', dict(status='complete', files=files), immutable=True)
        self.base.finish(folder, self.base.identity(dict(parent=self.request['identity'], stage=name)))

    def tree(self, root):
        return {str(path): sha(path) for path in Path(root).rglob('*') if path.is_file()
                and not path.name.endswith(('.tmp', '.lock')) and path.name != 'last_failure.json'
                and 'logs' not in path.relative_to(root).parts}

    def execute(self):
        root, base = self.root, self.base
        e0 = root / 'E0'
        self.stage('E0_import', self.command('E0-import', e0, '--existing', self.cli.e0_existing_root),
                   lambda: self.tree(e0))
        self.stage('E1_self_test', self.command('self-test', root / 'E1/self_test'),
                   lambda: self.tree(root / 'E1/self_test'))
        gates = {}
        for recipe in ('T1', 'T2'):
            for variant in ('C', 'D'):
                model = root / 'E1' / recipe / variant
                flags = ['--recipe', recipe, '--variant', variant]
                receipt = model / 'formal_warmup/receipt.json'
                def gate_files(receipt=receipt):
                    row = read(receipt)
                    if row['status'] != 'formal_warmup_complete':
                        raise ValueError('Missing full-data gate receipt')
                    return dict(row['files'], **{str(receipt): sha(receipt)})
                self.stage('gate_' + recipe + '_' + variant, self.command('warmup', model, *flags), gate_files)
                gates[recipe + '-' + variant] = dict(path=str(receipt), sha256=sha(receipt))
        base.atomic(root / 'formal_warmup_gate.json', dict(status='complete', policy=POLICY,
            models=gates, formal_epochs_already_counted=[1, 2], remaining_epochs=list(range(3, 21)),
            no_extra_training_epochs=True, score_thresholds_changed=False), immutable=True)
        for recipe in ('T1', 'T2'):
            for variant in ('C', 'D'):
                model = root / 'E1' / recipe / variant
                flags = ['--recipe', recipe, '--variant', variant]
                prefix = recipe + '_' + variant
                self.stage(prefix + '_train', self.command('train', model, *flags), lambda model=model: self.tree(model))
                selected = read(model / 'selected.json')
                destination = model / 'final_evaluation'
                self.stage(prefix + '_evaluate', self.command('evaluate', model, *flags,
                    '--checkpoint', selected['checkpoint'], '--destination', destination),
                    lambda destination=destination: self.tree(destination))
                self.stage(prefix + '_targets', self.command('targets', model, *flags, '--destination', destination),
                    lambda model=model: {str(model / 'target_analysis.json'): sha(model / 'target_analysis.json')})
        self.stage('E1_report', self.command('E1-report', root / 'E1'),
                   lambda: {str(p): sha(p) for p in (root / 'E1').glob('E1_*') if p.is_file()})
        self.stage('synthesis', self.command('synthesis', root), lambda: self.tree(root / 'reports'))
        for marker in (root / '.stages').glob('*/complete.json'):
            if not self.verify(marker.parent.name):
                raise ValueError('Missing final stage receipt')
        base.atomic(root / 'completion.json', dict(status='complete', request=self.request['identity'],
            scientific_completion=True, formal_epochs_per_model=20,
            startup_gate_sha256=sha(root / 'formal_warmup_gate.json'),
            input_identity_sha256=sha(root / 'input_identity.json'),
            execution_identity_sha256=sha(root / 'execution_identity.json'),
            final_report_sha256=sha(root / 'reports/E01_FINAL_CONCLUSIONS.md'),
            stage_hashes={str(p.relative_to(root)): sha(p) for p in (root / '.stages').glob('*/complete.json')}))
        base.atomic(root / 'current_stage.json', dict(status='complete', request=self.request['identity']))
        print('COMPLETE output=' + str(root), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--legacy-run-root', type=Path, required=True)
    parser.add_argument('--original-root', type=Path, required=True)
    parser.add_argument('--train-root', type=Path, required=True)
    parser.add_argument('--e0-existing-root', type=Path)
    parser.add_argument('--gpu', required=True)
    parser.add_argument('--resume', choices=('0', '1'), required=True)
    cli = parser.parse_args()
    base = bridge()
    base.server_only()
    from preflight import validate_paths, resources
    cli.output_root, cli.original_root, cli.train_root = validate_paths(cli.output_root, cli.original_root, cli.train_root)
    legacy = base.safe(cli.legacy_run_root).resolve(strict=True)
    if cli.output_root == legacy or legacy in cli.output_root.parents or cli.output_root in legacy.parents:
        raise ValueError('New output must be independent of the complete old E01 run')
    cli.e0_existing_root, legacy_identity = existing_e0(cli, base)
    validate_paths(cli.output_root, cli.original_root, cli.train_root, cli.e0_existing_root)
    base.gpu_env(cli.gpu)  # Validate before forming a shared lease filename.
    cli.output_root.mkdir(parents=True, exist_ok=True)
    root = cli.output_root
    with ExitStack() as stack:
        fds = [stack.enter_context(base.lock(root / '.controller.lock')),
               stack.enter_context(base.lock(Path('/data/cjm/datasets/logs/.gspr-e01-leases') / ('gpu' + cli.gpu + '.lock')))]
        os.environ['E01_LEASE_FDS'] = ','.join(map(str, fds))
        request = dict(family='GSPR_E01_FORMAL_WARMUP', source=base.sources(), policy=POLICY,
            settings=base.SETTINGS, output=str(root), original=str(cli.original_root), train=str(cli.train_root),
            legacy=legacy_identity, E0_existing=str(cli.e0_existing_root), gpu=cli.gpu, executable=sys.executable,
            python_version=sys.version, PYTHONHASHSEED=os.environ.get('PYTHONHASHSEED'),
            threads={key: os.environ.get(key) for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')})
        request['identity'] = base.identity(request)
        marker = root / 'request.json'
        if marker.exists():
            if cli.resume != '1' or read(marker) != request:
                raise ValueError('RESUME=1 and identical source/data/config/GPU/environment required')
        elif any(p.name not in ('logs', '.controller.lock') for p in root.iterdir()):
            raise ValueError('Nonempty unowned output; nothing overwritten')
        base.atomic(marker, request, immutable=True)
        control = Controller(cli, base, request)
        with short_temp(), signal_handlers():
            try:
                print('OUTPUT_ROOT=' + str(root) + '\nE0_REUSE=' + str(cli.e0_existing_root), flush=True)
                probe_path = root / 'preflight/E1_gpu.json'
                base.run(control.command('probe', root / 'preflight', '--destination', probe_path),
                         root / 'logs/E1_probe.log', env=base.gpu_env(cli.gpu))
                resource = resources(root, 'serial', {'E1': read(probe_path)})
                probe = read(probe_path)
                base.atomic(root / 'execution_identity.json',
                    {k: probe[k] for k in ('physical_gpu', 'name', 'hip')}, immutable=True)
                base.run(control.command('preflight', root / 'preflight'), root / 'logs/preflight.log', env=base.gpu_env(cli.gpu))
                bind_inputs(base, root)
                base.atomic(root / 'preflight/resources.json', resource)
                for path in (root / '.stages').glob('*/complete.json'):
                    control.verify(path.parent.name)
                control.execute()
            except BaseException:
                base.atomic(root / 'failure.json', dict(status='technical_failure', traceback=traceback.format_exc(),
                    committed_epochs_preserved=True, original_experiments_unchanged=True))
                raise


def bind_inputs(base, root):
    preflight = read(root / 'preflight/preflight.json')
    if preflight['status'] != 'complete' or preflight['train_frames'] != 8083 or preflight['validation_frames'] != 829:
        raise ValueError('Original full-data preflight is incomplete')
    keys = ('sources', 'train_manifest_sha256', 'train_manifest_identity', 'train_data_acceptance',
            'three_weather_training_identity', 'input_identity', 'frozen_GT_identity', 'T0_reference')
    base.atomic(root / 'input_identity.json', {k: preflight[k] for k in keys}, immutable=True)


if __name__ == '__main__':
    main()
