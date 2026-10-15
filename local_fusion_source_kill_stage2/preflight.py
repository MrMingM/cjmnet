"""Read-only server checks and one immutable root protocol. No Git required."""
import argparse
import ast
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from .common import (ROOT, PACKAGE, atomic_json, atomic_text, digest, external_run,
                     jsonlines, load_contract, object_hash, read_json, resource_check,
                     safe_path, settings, sources)
from .verdicts import sample_gate

P0_REPORTS = tuple(name + ext for name in ('P0_SOURCE_REDUNDANCY', 'P0_DATA_ADEQUACY',
    'P0_STAGE2_PRECHECK_FINAL') for ext in ('.json', '.md'))


def section42():
    context = (ROOT / 'AI_CONTEXT.md').read_text(encoding='utf-8')
    match = re.search(r'^## 42\. Source Utility Kill Protocol：Stage 2 最终冻结协议.*?(?=^## |\Z)',
                      context, re.M | re.S)
    if match is None:
        raise ValueError('Final frozen section 42 missing')
    return match.group()


def static_check(shell=True, snapshot=True):
    for path in PACKAGE.glob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    for path in PACKAGE.glob('*.sh'):
        raw = path.read_bytes()
        if b'\r' in raw or not raw.endswith(b'\n'):
            raise ValueError('Shell must use LF: ' + path.name)
        if any(x in raw for x in (b'pipefail', b'[[', b'<(', b'>(', b'source ')):
            raise ValueError('Non-POSIX shell token: ' + path.name)
        if shell:
            subprocess.run(['sh', '-n', str(path)], check=True)
    spec = settings()
    if spec['seeds'] != [20261014, 20261015, 20261016] or spec['bootstrap'] != {
            'seed': 20261014, 'repetitions': 5000, 'unit': 'paired validation scene'}:
        raise ValueError('Frozen seed/bootstrap specification differs')
    section42()
    from gspr_communication.runtime import verify_frozen
    verify_frozen()
    if snapshot and read_json(PACKAGE / 'source_snapshot.json')['source_hashes'] != sources():
        raise ValueError('Source snapshot mismatch: sync complete code directory')
    return {'status': 'complete', 'scientific_experiment': False}


def find_p0(explicit=None):
    reference = ROOT / 'logs/k/P/P0_MANIFEST.json'
    expected = digest(reference)
    candidates = [safe_path(explicit)] if explicit else [ROOT / 'logs/k/P'] + list(
        Path('/data/cjm/datasets/logs').glob('source_kill_stage2_precheck_*'))
    found = []
    for path in candidates:
        path = safe_path(path)
        manifest_path = path / 'P0_MANIFEST.json'
        if manifest_path.is_file() and digest(manifest_path) == expected and (
                path / 'p0_oracle_target_records.jsonl').is_file():
            if path not in found:
                found.append(path)
    if len(found) != 1:
        raise ValueError(f'Need exactly one full P0 run matching formal manifest; found {len(found)}; set P0_RUN')
    return found[0]


def baseline_rows(path):
    rows = list(jsonlines(path))
    relevant = [r for r in rows if r['split'] == 'validation' and r['weather'] in ('fog', 'rain', 'snow')
                and r['variant'] == settings()['k2']['baseline_variant']]
    keys = [(r['weather'], r['frame'], r['mode'], r['target_index']) for r in relevant]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate P0 current-state target records')
    same = [r for r in relevant if r['mode'] == 'same' and r['selected_discrete_positive']]
    task = [r for r in relevant if r['mode'] == 'task' and r['true_task_improvement']]
    return {'same': same, 'task': task}


def server_preflight(run, p0_run=None):
    static_check()
    resource_check(settings())
    run = external_run(run)
    if shutil.disk_usage(run).free < settings()['minimum_free_disk_gib'] * 1024**3:
        raise ValueError('Insufficient free disk: require 20 GiB')
    import fcntl
    with (run / '.preflight.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (run / 'PROTOCOL.json').exists():
            saved=load_contract(run)
            from local_fusion_source_kill_stage1.common import guard_tree
            for root in saved['dataset_roots'].values():
                guard_tree(root)
            return saved
        if any((run/name/'manifest.json').exists() for name in ('k1','k2')) or (run/'STAGE2_FINAL.json').exists():
            raise ValueError('Scientific state exists without immutable root protocol; use a new RUN')
        p0 = find_p0(p0_run or os.environ.get('P0_RUN'))
        manifest = read_json(p0 / 'P0_MANIFEST.json')
        if manifest['status'] != 'complete' or manifest['identity']['mode'] != 'full-cpu':
            raise ValueError('Formal full-cpu P0 is not complete')
        hashes = {}
        def checked(path, expected=None):
            value = digest(path)
            if expected is not None and value != expected:
                raise ValueError('Historical input hash mismatch: ' + str(path))
            hashes[str(safe_path(path))] = value
        for name in P0_REPORTS + ('p0_oracle_target_records.jsonl',):
            checked(p0 / name, manifest['outputs'][name])
        final_p0=read_json(p0/'P0_STAGE2_PRECHECK_FINAL.json')
        if final_p0['status']!='complete_existing_evidence' or not final_p0['k3_frozen'] or final_p0['test_data_used']:
            raise ValueError('P0 evidence incomplete or scope changed')
        checked(p0 / 'P0_MANIFEST.json')
        for name in P0_REPORTS:
            checked(ROOT / 'logs/k/P' / name, manifest['outputs'][name])
        stage1 = safe_path(manifest['identity']['stage1_run'])
        checked(stage1 / 'protocol.json', manifest['identity']['stage1_protocol_sha256'])
        one = read_json(stage1 / 'protocol.json')
        if read_json(stage1 / 'manifest.json')['status'] != 'complete':
            raise ValueError('Stage 1 incomplete')
        checked(stage1 / 'manifest.json')
        for name, expected in manifest['identity']['historical_hashes'].items():
            checked(stage1 / name, expected)
            checked(ROOT / 'logs/k' / name, expected)
        from local_fusion_source_kill_stage1.common import check_hashes, guard_tree
        check_hashes(one['source_hashes'])
        references = one['references']
        s0 = safe_path(references['s0_run'])
        checked(s0 / 'protocol.json', references['s0_protocol_hash'])
        checked(s0 / 'manifest.json', references['s0_manifest_hash'])
        zero = read_json(s0 / 'protocol.json')
        oracle_run = safe_path(references['oracle_run'])
        checked(oracle_run / 'protocol.json', references['oracle_protocol_hash'])
        checked(oracle_run / 'death_test_results.json', references['oracle_results_hash'])
        old = read_json(oracle_run / 'protocol.json')
        b0_run = safe_path(one['inputs']['b0_run'])
        b0 = read_json(b0_run / 'protocol.json')
        checked(b0_run / 'protocol.json', old['b0_protocol_sha256'])
        check_hashes(old['source_hashes'])
        check_hashes(b0['source_hashes'])
        checked(ROOT / 'local_fusion_task_source_oracle/experiment.yaml', references['oracle_config_hash'])
        import yaml
        actual_spec = yaml.safe_load((ROOT / 'local_fusion_task_source_oracle/experiment.yaml').read_text(encoding='utf-8'))
        if actual_spec != old['config'] or any(actual_spec[k] != v for k, v in settings()['oracle'].items()):
            raise ValueError('Original v2 Oracle settings drift')
        for split in ('train', 'validation'):
            for field in ('indices', 'scenes'):
                name = split + '_' + field
                if zero[name] != b0[name] or not b0[name] or len(b0[name]) != len(set(b0[name])):
                    raise ValueError('B0/S0 split mismatch: ' + name)
        if one['validation_indices'] != old['validation_indices'] or old['validation_indices'] != b0['validation_indices']:
            raise ValueError('Validation indices drift')
        for field in ('frontend_config', 'v3_config', 'frontend_checkpoint', 'v3_checkpoint'):
            checked(one['inputs'][field])
        checked(b0_run / 'Shared.pth', references['b0_checkpoint_hash'])
        if digest(b0_run / 'Shared.pth') != old['b0_shared_checkpoint_sha256']:
            raise ValueError('Oracle B0 checkpoint differs')
        if digest(one['inputs']['frontend_checkpoint']) != old['frontend_sha256'] or digest(
                one['inputs']['v3_checkpoint']) != old['v3_checkpoint_sha256']:
            raise ValueError('Model checkpoint hashes differ')
        front = yaml.safe_load(safe_path(one['inputs']['frontend_config']).read_text(encoding='utf-8'))
        options = yaml.safe_load(safe_path(one['inputs']['v3_config']).read_text(encoding='utf-8'))
        roots = {}
        for name in ('root_dir', 'validate_dir'):
            root = safe_path(options.get(name) or front[name], dataset=True)
            if str(root) != zero['dataset_paths'][name]:
                raise ValueError('Dataset root differs from S0')
            guard_tree(root)
            roots[name] = str(root)
        gates = {layer: sample_gate(rows, layer) for layer, rows in baseline_rows(
            p0 / 'p0_oracle_target_records.jsonl').items()}
        p0_summary = read_json(p0 / 'P0_DATA_ADEQUACY.json')['H2']['A_ONLY']['validation/adverse pooled']
        for layer, field in (('same','same_scene_coverage'),('task','scene_coverage')):
            recorded = p0_summary[field]
            if (gates[layer]['n'] != recorded['n'] or gates[layer]['per_scene'] != recorded['per_scene']):
                raise ValueError('P0 raw records do not reproduce formal scene coverage: '+layer)
        if any(r['verdict'] != 'PASS' for r in gates.values()):
            raise ValueError('K2 sample eligibility gate failed')
        clock = manifest['project_clock']
        if clock['resource_stop_at'] != settings()['resource_stop_at'] or clock != one['project_clock']:
            raise ValueError('Resource clock differs; cannot restart project clock')
        if shutil.disk_usage(run).free < settings()['minimum_free_disk_gib'] * 1024**3:
            raise ValueError('Insufficient free disk: require 20 GiB')
        atomic_text(run / '.write_probe', 'writable\n')
        (run / '.write_probe').unlink()
        canonical_inputs = {name:str(safe_path(value)) for name,value in one['inputs'].items()}
        protocol = dict(stage=settings()['stage'], settings=settings(), inputs=canonical_inputs,
            stage1_run=str(stage1), stage1_protocol=one, p0_run=str(p0), s0_run=str(s0),
            oracle_run=str(oracle_run), validation_indices=old['validation_indices'],
            train_indices=b0['train_indices'], dataset_roots=roots, sample_gates=gates,
            input_hashes=hashes, source_hashes=sources(), section42=section42(),
            resource_stop_at=clock['resource_stop_at'], project_clock=clock,
            null_config_contract='k1/NULL_CONFIG.json is an immutable train-only extension; frozen before validation',
            source_commit=None, test_data_used=False, k3_frozen=True)
        protocol['identity'] = object_hash(protocol)
        atomic_json(run / 'SOURCE_SNAPSHOT.json', protocol['source_hashes'])
        atomic_json(run / 'source_snapshot.json', protocol['source_hashes'])
        atomic_text(run / 'protocol.yaml', json.dumps(protocol, ensure_ascii=False, indent=2) + '\n')
        atomic_json(run / 'protocol.json', protocol)
        atomic_json(run / 'PROTOCOL.json', protocol)  # final commit marker
        return protocol


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--static', action='store_true')
    cli.add_argument('--no-shell', action='store_true')
    cli.add_argument('--run')
    cli.add_argument('--p0-run')
    args = cli.parse_args()
    if args.static:
        print(static_check(shell=not args.no_shell))
    else:
        protocol = server_preflight(args.run, args.p0_run)
        print('PREFLIGHT INPUT CHECKS COMPLETE; GPU reproduction gate pending:', protocol['identity'], flush=True)


if __name__ == '__main__':
    main()
