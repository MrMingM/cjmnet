"""D-1A orchestration only: prepare, compile upstream extensions, or run Clean."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import traceback

from common import (COMMIT, PACKAGE, atomic_json, identity, install_runtime,
                    link, lock, no_test, read_json, server_only, sha256, verify_package)
from prepare import prepare_data


def select_file(root, explicit, kind):
    if explicit:
        result = no_test(explicit).resolve(strict=True)
        if not result.is_file():
            raise ValueError(f'Not a regular {kind} file: {result}')
        return result
    if kind == 'checkpoint':
        choices = sorted(p for p in root.iterdir() if p.is_file()
                         and p.suffix.lower() in ('.pth', '.pt', '.ckpt'))
    else:
        choices = sorted(p for p in root.iterdir() if p.is_file()
                         and p.name.lower() == 'v2xr_attfuse.yaml')
    if len(choices) != 1:
        raise ValueError(f'Expected one {kind} in {root}; found {[p.name for p in choices]}. '
                         f'Specify --{kind} / {kind.upper()} explicitly; never select newest.')
    return no_test(choices[0]).resolve(strict=True)


def config_difference(left, right, prefix=''):
    if isinstance(left, dict) and isinstance(right, dict):
        differences = []
        for key in sorted(set(left) | set(right)):
            name = f'{prefix}.{key}' if prefix else str(key)
            if key not in left or key not in right:
                differences.append(name)
            else:
                differences += config_difference(left[key], right[key], name)
        return differences
    return [] if left == right else [prefix]


def prepare_model(args):
    import yaml
    raw = Path(args.data_root).resolve(strict=True)
    checkpoint = select_file(raw, args.checkpoint, 'checkpoint')
    config = select_file(raw, args.yaml, 'yaml')
    reference_path = PACKAGE / 'upstream/opencood/hypes_yaml/V2X-R/LiDAR/V2XR_attfuse.yaml'
    source_text = config.read_text(encoding='utf-8')
    supplied, reference = yaml.safe_load(source_text), yaml.safe_load(reference_path.read_text(encoding='utf-8'))
    differences = config_difference(supplied, reference)
    invalid = [k for k in differences if k not in ('root_dir', 'validate_dir')]
    if invalid:
        raise ValueError(f'Supplied YAML differs from reviewed official configuration: {invalid}. '
                         'No silent compatibility overrides; review the supplied file first.')
    if sha256(checkpoint) == '06e953cc15391e5b7569e825d33112f79b1e8c15a70b8914a489bb437cb81aba':
        raise ValueError('This is the frozen OPV2V epoch13 checkpoint, not the user-supplied V2X-R weight')
    validate = str(Path(args.view_root).resolve() / 'validate')
    # Preserve YAML anchors and every behavioral setting; rewrite only two path lines.
    adapted = source_text
    changes = {}
    for key, value in (('root_dir', str(raw / 'train')), ('validate_dir', validate)):
        adapted, n = re.subn(r'^' + key + r':[^\n]*$',
            lambda match: key + ': ' + json.dumps(value), adapted, flags=re.MULTILINE)
        if n != 1:
            raise ValueError(f'Expected one top-level {key} path in original YAML')
        changes[key] = dict(original=supplied[key], actual=value)
    parsed = yaml.safe_load(adapted)
    if any(k not in ('root_dir', 'validate_dir') for k in config_difference(supplied, parsed)):
        raise ValueError('Path adaptation changed behavioral YAML settings')
    metadata = dict(upstream_commit=COMMIT, checkpoint=str(checkpoint),
        checkpoint_sha256=sha256(checkpoint), original_yaml=str(config),
        original_yaml_sha256=sha256(config), official_template_sha256=sha256(reference_path),
        config_path_changes=changes,
        checkpoint_aliases=['0.pth', 'net_epoch0.pth'],
        alias_reason='README requests 0.pth; actual load_model(epoch=0) opens net_epoch0.pth')
    directory = Path(args.model_dir)
    directory.mkdir(parents=True, exist_ok=True)
    config_path = directory / 'config.yaml'
    if config_path.exists() and config_path.read_bytes() != adapted.encode():
        raise FileExistsError(f'Existing model config differs: {config_path}; use a new MODEL_DIR')
    if not config_path.exists():
        config_path.write_text(adapted, encoding='utf-8', newline='\n')
    metadata['runtime_yaml_sha256'] = sha256(config_path)
    meta_path = directory / 'provenance.json'
    if meta_path.exists() and read_json(meta_path) != metadata:
        raise FileExistsError('Existing model provenance differs; use a new MODEL_DIR')
    link(checkpoint, directory / '0.pth')
    link(checkpoint, directory / 'net_epoch0.pth')
    atomic_json(meta_path, metadata)
    return metadata


def child(args, probe=False):
    command = [sys.executable, '-u', str(PACKAGE / 'observe.py'),
        '--runtime', args.runtime_root, '--model-dir', args.model_dir,
        '--output-root', args.output_root, '--view-root', args.view_root,
        '--limit-frames', str(args.limit_frames), '--loader-workers', str(args.loader_workers)]
    if probe:
        command.append('--probe')
    environment = os.environ.copy()
    environment['PYTHONPATH'] = args.runtime_root
    subprocess.run(command, cwd=args.runtime_root, env=environment, check=True)


def main():
    server_only()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('prepare', 'build', 'evaluate'), default='evaluate')
    parser.add_argument('--data-root', default='/data/cjm/datasets/V2X-R')
    parser.add_argument('--paired-root', default='/data/cjm/datasets/v2xr-lidar-weather-validate-v1')
    parser.add_argument('--view-root', default='/data/cjm/datasets/v2xr-official-validate-v1')
    parser.add_argument('--runtime-root', default='/data/cjm/datasets/v2xr-official-runtime-v1')
    parser.add_argument('--model-dir', default='/data/cjm/datasets/logs/v2xr-official-attfuse-model-v1')
    parser.add_argument('--output-root', default='/data/cjm/datasets/logs/v2xr-official-attfuse-clean-repro-v1')
    parser.add_argument('--checkpoint')
    parser.add_argument('--yaml')
    parser.add_argument('--seven-zip')
    parser.add_argument('--limit-frames', type=int, default=0)
    parser.add_argument('--loader-workers', type=int, default=16)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if not 0 <= args.limit_frames <= 829 or args.loader_workers < 0:
        raise ValueError('Invalid frame limit or loader worker count')
    package = verify_package()
    for name in ('data_root', 'paired_root', 'view_root', 'runtime_root', 'model_dir', 'output_root'):
        setattr(args, name, str(no_test(getattr(args, name)).resolve()))
    writable = [Path(getattr(args, k)) for k in ('view_root', 'runtime_root', 'model_dir', 'output_root')]
    sources = [Path(args.data_root), Path(args.paired_root), PACKAGE.parent]
    for target in writable:
        if any(target == s or s in target.parents or target in s.parents for s in sources):
            raise ValueError('Use independent output/runtime/model/view directories on the data disk')
    if any(a == b or a in b.parents or b in a.parents for i, a in enumerate(writable) for b in writable[i+1:]):
        raise ValueError('Output/runtime/model/view directories must be separate')
    output = Path(args.output_root)
    with lock(output.with_suffix('.lock')), lock(Path(args.model_dir).with_suffix('.lock')), \
            lock(Path(args.runtime_root).with_suffix('.run.lock')):
        if output.exists() and any(output.iterdir()) and not (output / 'contract.json').exists():
            if not (args.stage == 'build' and (output / 'build.json').exists()):
                raise FileExistsError('Refusing unrelated/non-contract output; use a new OUTPUT_ROOT')
        output.mkdir(parents=True, exist_ok=True)
        attempts = output / 'attempts'
        attempt = attempts / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        attempt.mkdir(parents=True)
        try:
            source_hashes = install_runtime(args.runtime_root)
            if args.stage == 'build':
                env = os.environ.copy()
                env['PYTHONPATH'] = args.runtime_root
                env['MAX_JOBS'] = env.get('MAX_JOBS', '2')
                subprocess.run([sys.executable, '-u', str(PACKAGE / 'build_preflight.py'),
                    '--output', str(output / 'build_environment.json')],
                    cwd=args.runtime_root, env=env, check=True)
                subprocess.run([sys.executable, 'opencood/utils/setup.py', 'build_ext', '--inplace'],
                    cwd=args.runtime_root, env=env, check=True)
                subprocess.run([sys.executable, '-u', str(PACKAGE / 'build_official_ops.py'),
                    '--runtime', args.runtime_root], cwd=args.runtime_root, env=env, check=True)
                install_runtime(args.runtime_root)  # source must remain byte-identical
                atomic_json(output / 'build.json', dict(status='complete', source_hashes=source_hashes,
                    environment=read_json(output / 'build_environment.json'),
                    native_selection=read_json(Path(args.runtime_root) / 'native_build_selection.json')))
                return
            metadata = prepare_model(args)
            contract = dict(stage='D-1A', upstream_commit=COMMIT, package=package,
                metadata=metadata, parameters=vars(args) | {'resume': False, 'stage': 'evaluate'})
            contract_id = identity(contract)
            contract_path = output / 'contract.json'
            if contract_path.exists():
                old = read_json(contract_path)
                if old['identity'] != contract_id or not args.resume:
                    raise ValueError('Existing run needs identical arguments plus RESUME=1; changes need new OUTPUT_ROOT')
                if (output / 'summary.json').exists() and read_json(output / 'summary.json')['status'] == 'complete':
                    print(f'Already complete: {output / "RESULTS.md"}', flush=True)
                    return
            elif any(p.name not in ('attempts',) for p in output.iterdir()):
                raise FileExistsError('Refusing unrelated nonempty output directory')
            atomic_json(contract_path, dict(identity=contract_id, contract=contract))
            if args.stage == 'evaluate':
                child(args, probe=True)  # Fail on missing official dependencies before extracting GBs.
            manifest = prepare_data(args)
            atomic_json(output / 'data_verification.json', manifest)
            if args.stage == 'prepare':
                print(f'Preparation complete: {Path(args.view_root) / "manifest.json"}', flush=True)
                return
            child(args)
        except Exception as error:
            failure = dict(status='failed', stage=args.stage, arguments=vars(args),
                error=repr(error), traceback=traceback.format_exc(),
                next_stages_implemented=False)
            atomic_json(attempt / 'failure.json', failure)
            atomic_json(output / 'last_failure.json', failure)
            completed = ((output / 'summary.json').exists()
                         and read_json(output / 'summary.json').get('status') == 'complete')
            if not completed:
                (output / 'RESULTS.md').write_text('# D−1A Clean validation\n\n'
                    '**OFFICIAL REPRODUCTION NOT YET ALIGNED**\n\n'
                    '运行未完成。检查 last_failure.json 和服务器日志；没有 AP 复现结论。\n'
                    '不得进入后续天气分析或训练。\n', encoding='utf-8', newline='\n')
            raise


if __name__ == '__main__':
    main()
