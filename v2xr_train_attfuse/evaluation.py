"""D-1D adapter around the frozen D-1C generator and O/O*/F replay."""
import argparse
from pathlib import Path
import shutil

from auditlib import (DEFAULTS, HERE, SELECTION, atomic_json, bound_json, complete,
    evaluation_path, frozen_c, identity, jsonable, no_test, read_json, request, runtime, sha256, sources)


def selection(args):
    root = Path(args.output_root)
    policy = read_json(root / 'selection_policy.json')
    if policy != SELECTION:
        raise ValueError('Preregistered selection policy changed')
    contract = read_json(root / 'contract.json')
    if contract['smoke_only'] or contract['sources'] != sources():
        raise ValueError('Selection requires unchanged full training code/contract')
    rows = []
    for epoch in range(1, 21):
        row = complete(root / 'epochs' / f'{epoch:02d}' / 'complete.json')
        if (row.get('smoke_only') or row['contract_identity'] != contract['identity']
                or row['epoch'] != epoch or row['checkpoint_sha256'] != sha256(root / f'net_epoch{epoch}.pth')):
            raise ValueError('Epoch/checkpoint identity differs')
        ap = complete(Path(row['clean_ap_summary']))
        if (sha256(row['clean_ap_summary']) != row['clean_ap_summary_sha256']
                or ap['conditions'] != ['clean'] or ap['frame_count'] != 829
                or ap['checkpoint_sha256'] != row['checkpoint_sha256']):
            raise ValueError('Model selection requires complete Clean-only AP and matching checkpoint')
        row['selection_AP70'] = ap['results']['clean']['AP']['F']['ap70']
        rows.append(row)
    best = min(rows, key=lambda r: (-r['selection_AP70'], r['epoch']))
    chosen = dict(status='complete', policy=SELECTION, epoch=best['epoch'],
        checkpoint=str(root / f'net_epoch{best["epoch"]}.pth'), checkpoint_sha256=best['checkpoint_sha256'],
        clean_F_AP70=best['selection_AP70'], candidate_count=20,
        candidate_metrics=[dict(epoch=r['epoch'], AP70=r['selection_AP70'],
            checkpoint_sha256=r['checkpoint_sha256'], clean_ap_summary_sha256=r['clean_ap_summary_sha256']) for r in rows],
        training_contract_sha256=sha256(root / 'contract.json'),
        policy_sha256=sha256(root / 'selection_policy.json'), weather_used=False)
    chosen['identity'] = identity(chosen)
    bound_json(root / 'selected.json', chosen)
    return chosen


def assert_frozen_gt(actual, frozen, conditions, count):
    expected = {(r['condition'], r['index']): r for r in frozen['gt']['frames']}
    fields = ('frame_key', 'official_gt_identity', 'fixed_gt_identity', 'official_ids',
              'fixed_ids', 'official_count', 'fixed_count', 'removed_ids')
    if len(actual['frames']) != len(conditions) * count:
        raise ValueError('GT manifest frame count differs')
    for row in actual['frames']:
        old = expected[(row['condition'], row['index'])]
        if any(row[key] != old[key] for key in fields):
            raise ValueError('Frozen D-1C GT identity/order/geometry changed: ' + str(row['frame_key']))


def evaluate(args, own, checkpoint, destination, conditions=('clean', 'fog', 'snow'), limit=0):
    from provenance import verify_inputs
    from predict import generate
    import evaluate as d1c_evaluate
    from support import DEFINITIONS, patched
    checkpoint = no_test(checkpoint).resolve(strict=True)
    output = evaluation_path(args, destination)
    if checkpoint == output or output in checkpoint.parents:
        raise ValueError('Evaluation output cannot contain its input checkpoint')
    # verify_inputs intentionally uses the frozen AUTHOR model directory for evidence only.
    # It hashes this weight; it does not load it into the new detector.
    evidence = verify_inputs(args)
    frozen = frozen_c(args)
    runtime(args, evidence['a'])
    import torch
    from observe import module_inventory
    selected = evidence['b']['selected_frame_keys'][:limit or 829]
    training_contract_path = Path(args.output_root) / 'contract.json'
    associated_training = None
    if training_contract_path.exists():
        training_contract = read_json(training_contract_path)
        associated_training = dict(training_contract_sha256=sha256(training_contract_path),
            train_data_acceptance=training_contract['train_data_acceptance'],
            interpretation='Associated D-1D training run; this does not describe the training data of an OPV2V comparison checkpoint.')
    contract = jsonable(dict(stage='D-1D', source_hashes=own,
        associated_d1d_training=associated_training,
        official_source_commit=evidence['a']['upstream_commit'],
        official_source_hashes=evidence['a']['upstream_source_hashes'],
        native_modules={k: v for k, v in module_inventory(Path(args.runtime_root)).items() if v['path'].endswith('.so')},
        checkpoint_sha256=sha256(checkpoint), official_config_sha256=sha256(Path(args.model_dir) / 'config.yaml'),
        d1a=evidence['d1a_artifact_sha256'], d1b=evidence['d1b_artifact_sha256'],
        frozen_d1c_hashes=frozen['hashes'], frozen_d1c_gt_identity=frozen['gt']['identity'],
        input_manifest_identity=evidence['input_manifest']['identity'], frame_keys=selected,
        conditions=list(conditions), protocols=DEFINITIONS, inference_seed=20261016,
        PYTHONHASHSEED=20261016, loader_workers=16, device='cuda:0',
        physical_gpu=os_gpu(), versions=evidence['a']['versions'], HIP=evidence['a']['hip_version'],
        input_roots={key: getattr(args, key) for key in ('model_dir', 'runtime_root', 'gt_audit_root', 'd1a_root', 'd1c_root', 'view_root')},
        torch_backend=dict(cudnn_benchmark=torch.backends.cudnn.benchmark,
            cudnn_deterministic=torch.backends.cudnn.deterministic,
            cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
            matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32)))
    contract['identity'] = identity(contract)
    request(output, contract, args.resume)
    bound_json(output / 'contract.json', contract)
    if (output / 'summary.json').exists():
        done = complete(output / 'summary.json')
        if done['contract_identity'] != contract['identity']:
            raise ValueError('Evaluation completion contract differs')
        prediction = read_json(output / 'prediction_manifest.json')
        for row in prediction['files']:
            if sha256(no_test(row['path'])) != row['sha256']:
                raise ValueError('Completed prediction cache bytes changed')
        assert_frozen_gt(read_json(output / 'gt_manifest.json'), frozen, conditions, len(selected))
        return done
    alias = output / 'evaluation_model'
    alias.mkdir(exist_ok=True)
    source_config = Path(args.model_dir) / 'config.yaml'
    if not (alias / 'config.yaml').exists():
        shutil.copyfile(source_config, alias / 'config.yaml')
    if sha256(alias / 'config.yaml') != sha256(source_config):
        raise ValueError('Evaluation config changed')
    from common import link
    for name in ('net_epoch0.pth', '0.pth'):
        link(checkpoint, alias / name)
    run_args = argparse.Namespace(**vars(args))
    run_args.model_dir = str(alias)
    run_args.stage = 'all'
    # All thresholds/NMS/dataset/model/eval functions are the byte-pinned D-1C originals.
    manifests = {c: generate(c, output, run_args, evidence, contract) for c in conditions}
    with patched(d1c_evaluate, 'CONDITIONS', tuple(conditions)):
        results, prediction, gt = d1c_evaluate.replay(output, evidence, contract, manifests)
    assert_frozen_gt(gt, frozen, conditions, len(selected))
    if len(selected) == 829:
        for c in conditions:
            if results[c]['counts']['fixed_gt'] != 6224 or results[c]['counts']['official_gt'] != evidence['totals'][c]:
                raise ValueError('D-1D full validation GT totals changed')
    protocol = dict(**contract, reused_functions=['v2xr_protocol_audit.predict.generate',
        'v2xr_protocol_audit.evaluate.replay'], imported_modules=module_inventory(Path(args.runtime_root)),
        frozen_gt_check='all per-frame IDs, order and official/fixed geometry identities equal completed D-1C',
        gt_manifest_sha256=sha256(output / 'gt_manifest.json'),
        gt_identity=gt['identity'], prediction_cache_identity=prediction['identity'],
        no_training=True, no_test=True, one_prediction_cache_per_condition=True)
    atomic_json(output / 'protocol.json', protocol)
    summary = dict(status='complete', stage='D-1D', contract_identity=contract['identity'],
        associated_d1d_training=associated_training,
        checkpoint_sha256=contract['checkpoint_sha256'], conditions=list(conditions),
        frame_count=len(selected), full_validation=len(selected) == 829, results=results,
        frozen_d1c_gt_identity=frozen['gt']['identity'], gt_identity=gt['identity'],
        prediction_cache_identity=prediction['identity'])
    from results import write_evaluation
    write_evaluation(output, summary)
    names = ['contract.json', 'protocol.json', 'prediction_manifest.json', 'gt_manifest.json',
        'per_frame_metrics.jsonl', 'removed_gt_metrics.jsonl', 'RESULTS.md']
    names += [f'stat_{c}_{p}.json' for c in conditions for p in ('O', 'Ostar', 'F')]
    names += [f'predictions/{c}/manifest.json' for c in conditions]
    summary['output_hashes'] = {n: sha256(output / n) for n in names}
    atomic_json(output / 'summary.json', summary)
    return summary


def os_gpu():
    import os
    return os.environ.get('ROCR_VISIBLE_DEVICES')


def final(args, own):
    chosen = selection(args)  # Locks selection using only the 20 Clean AP records before any weather inference.
    output = Path(args.output_root)
    name = f'final_evaluation_smoke{args.limit_frames}' if args.limit_frames else 'final_evaluation'
    return evaluate(args, own, chosen['checkpoint'], output / name, limit=args.limit_frames)


def weather_checkpoint_gate(args, checkpoint):
    """No all-condition evaluation of arbitrary epochs before Clean-only selection is locked."""
    chosen = selection(args)
    digest = sha256(no_test(checkpoint))
    if digest == chosen['checkpoint_sha256']:
        return
    if digest != read_json(HERE / 'PINS.json')['opv2v']['checkpoint_sha256']:
        raise ValueError('Weather evaluation permits only the Clean-selected model or the pinned OPV2V reference')
    baseline = complete(Path(args.output_root) / 'final_evaluation' / 'summary.json')
    if not baseline['full_validation'] or baseline['checkpoint_sha256'] != chosen['checkpoint_sha256']:
        raise ValueError('OPV2V weather comparison must follow the selected-model full final evaluation')


def compare(args, own):
    """Protocol-matched OPV2V reference; never used as initialization or model selection."""
    chosen = selection(args)
    baseline = complete(Path(args.output_root) / 'final_evaluation' / 'summary.json')
    if baseline['checkpoint_sha256'] != chosen['checkpoint_sha256'] or not baseline['full_validation']:
        raise ValueError('Comparison requires the completed selected-model full evaluation')
    pins = read_json(HERE / 'PINS.json')['opv2v']
    checkpoint = no_test(args.opv2v_checkpoint).resolve(strict=True)
    if sha256(checkpoint) != pins['checkpoint_sha256']:
        raise ValueError('Unexpected OPV2V epoch13 checkpoint')
    # Check the actual official architecture before running any predictions. A failure is a recorded
    # unavailable controlled comparison, never a reason to reduce max_cav or replace the model.
    from provenance import verify_inputs
    evidence = verify_inputs(args)
    runtime(args, evidence['a'])
    import torch
    from observe import normalize_state
    from opencood.hypes_yaml import yaml_utils
    from opencood.tools import train_utils
    model = train_utils.create_model(yaml_utils.load_yaml(str(Path(args.model_dir) / 'config.yaml')))
    state = normalize_state(torch.load(checkpoint, map_location='cpu', weights_only=False))
    own_state = model.state_dict()
    differences = dict(missing=sorted(own_state.keys() - state.keys()),
        unexpected=sorted(state.keys() - own_state.keys()),
        shape_dtype=[k for k in own_state.keys() & state.keys()
                     if own_state[k].shape != state[k].shape or own_state[k].dtype != state[k].dtype])
    del model, state, own_state
    if any(differences.values()):
        atomic_json(Path(args.output_root) / 'comparison.json', dict(status='unavailable',
            reason='OPV2V checkpoint is incompatible with the frozen official architecture', differences=differences,
            checkpoint_sha256=pins['checkpoint_sha256'], historical_AP_comparable=False))
        (Path(args.output_root) / 'COMPARISON.md').write_text(
            '# D-1D matched domain comparison unavailable\n\n'
            'The pinned OPV2V checkpoint does not match the frozen official architecture.\n'
            'No model/max_cav/GT change or partial weight loading was allowed.\n'
            'See comparison.json for key/shape/dtype differences. Historical AP uses a different protocol,\n'
            'so this run cannot quantify a comparable cross-domain weather-gap reduction.\n',
            encoding='utf-8', newline='\n')
        print('Controlled OPV2V comparison unavailable:', differences, flush=True)
        return
    # evaluate() calls bootstrap once; it must run in a fresh process after this compatibility probe.
    import subprocess, sys
    command = [sys.executable, '-B', str(HERE / 'main.py'), 'evaluate-checkpoint',
        '--checkpoint', str(checkpoint), '--evaluation-root', str(Path(args.output_root) / 'opv2v_matched_evaluation'), '--resume']
    for key in DEFAULTS:
        command.extend(['--' + key.replace('_', '-'), getattr(args, key)])
    subprocess.run(command, check=True)
    from results import write_comparison
    write_comparison(Path(args.output_root))
