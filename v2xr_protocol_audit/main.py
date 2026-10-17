"""D-1C: one official prediction cache, three evaluation protocols, no training."""
import argparse
from datetime import datetime, timezone
from importlib import metadata
import os
from pathlib import Path
import sys
import traceback

from support import (CONDITIONS, DEFINITIONS, HERE, REPRO, atomic_json, identity,
                     jsonable, lock, no_test, read_json, server_only, sha256, verify_sources)
from provenance import verify_inputs
from predict import generate
from evaluate import replay
from report import summarize, write_results


def execute(args, output, own_sources):
    evidence = verify_inputs(args)
    # Same source-isolation and imported-module validation used by successful D-1A.
    from observe import bootstrap, module_inventory
    runtime = bootstrap(args.runtime_root)
    import torch
    import opencood.models.point_pillar_intermediate
    from opencood.tools import inference_utils
    from opencood.utils import eval_utils_opv2v
    versions = {name: metadata.version(name) for name in evidence['a']['versions']}
    if versions != evidence['a']['versions'] or torch.version.hip != evidence['a']['hip_version']:
        raise ValueError('Dependency versions/HIP differ from successful D-1A')
    if os.environ.get('PYTHONHASHSEED') != '20261016':
        raise ValueError('Use run_server.sh to set PYTHONHASHSEED before Python starts')
    modules = module_inventory(runtime)
    for name, row in modules.items():
        if row['path'].endswith('.so') and evidence['a']['imported_modules'].get(name) != row:
            raise ValueError('Imported native extension differs from successful D-1A')
    keys = evidence['b']['selected_frame_keys']
    selected = keys[:args.limit_frames] if args.limit_frames else keys
    contract = dict(stage='D-1C', source_hashes=own_sources,
        inherited_source_hashes=evidence['b']['source_hashes'],
        d1b_artifact_sha256=evidence['d1b_artifact_sha256'], d1a_artifact_sha256=evidence['d1a_artifact_sha256'],
        checkpoint_sha256=evidence['d1a_model']['checkpoint_sha256'],
        original_yaml_sha256=evidence['d1a_model']['original_yaml_sha256'],
        runtime_yaml_sha256=evidence['d1a_model']['runtime_yaml_sha256'],
        official_source_commit=evidence['a']['upstream_commit'],
        official_source_hashes=evidence['a']['upstream_source_hashes'],
        native_extension_hashes={k: v for k, v in modules.items() if v['path'].endswith('.so')},
        native_build_selection=evidence['a']['native_build_selection'],
        d1a_view_manifest_sha256=evidence['d1a_view_manifest_sha256'],
        input_manifest_identity=evidence['input_manifest']['identity'],
        evaluated_configs=evidence['b']['evaluated_configs'], frame_keys=selected,
        protocols=DEFINITIONS, eval_module='opencood.utils.eval_utils_opv2v',
        AP='original official calculate_ap; BEV IoU/VOC2010/global_sort=False',
        prediction_generation='original inference.main with additive cache observers',
        seed=20261016, loader_workers=16, PYTHONHASHSEED=20261016, device='cuda:0',
        operational_environment={k: os.environ.get(k) for k in
            ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
        torch_backend=dict(cudnn_benchmark=torch.backends.cudnn.benchmark,
            cudnn_deterministic=torch.backends.cudnn.deterministic,
            cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
            matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32),
        versions=versions, hip_version=torch.version.hip,
        clean_repeat_AP_tolerance=1e-6, paired_corner_tolerance_m=1e-4,
        input_roots={k: getattr(args, k) for k in ('runtime_root', 'model_dir', 'd1a_root', 'gt_audit_root', 'view_root')})
    contract = jsonable(contract)
    contract['identity'] = identity(contract)
    path = output / 'contract.json'
    if path.exists() and read_json(path) != contract:
        raise ValueError('Prediction contract source/checkpoint/config/input/environment changed; use a new OUTPUT_ROOT')
    atomic_json(path, contract)
    if (output / 'summary.json').exists():
        summary = read_json(output / 'summary.json')
        if summary['status'] != 'complete' or summary['contract_identity'] != contract['identity']:
            raise ValueError('Existing completion identity differs')
        for name, digest in summary['output_hashes'].items():
            if sha256(output / name) != digest:
                raise ValueError(f'Completed output changed: {name}')
        prediction = read_json(output / 'prediction_manifest.json')
        for row in prediction['files']:
            if sha256(row['path']) != row['sha256']:
                raise ValueError('Completed prediction cache changed')
        print('Completed D-1C output/cache hashes verified; no inference or AP rerun.', flush=True)
        return
    # Synthetic validation runs only here on the server, using the exact official evaluator.
    from test_contract import run_tests
    run_tests()
    protocol = dict(**contract, imported_modules=modules,
        d1a_model_provenance=evidence['d1a_model'], d1b_expected_GT_totals=evidence['totals'],
        gt_definition='Official Clean-selected paired fixed GT, from live Clean pipeline checked against D-1B',
        none_empty_distinction='O skips None only; empty non-None is counted in all protocols',
        paired_comparison='Each condition has one cache file per frame; evaluation never forwards a model',
        removed_recall_matching='Observe actual one-to-one F AP matches, partition matched IDs into removed and retained',
        resume='Validate all inputs and cached frame hashes; replay original loader prefix, forward only missing frames',
        historical_rng_note='D-1B used per-frame seeds/0 workers for audit. Prediction uses original D-1A 16-worker inference. '
            'Live GT must match D-1B IDs and geometry; any mismatch fails rather than replacing official GT.',
        interpretation='GapMasking is descriptive and includes differential None-frame skip effects; GTFilterEffect isolates O* minus F',
        no_training=True, no_test=True, no_threshold_adjustment=True)
    atomic_json(output / 'protocol.json', protocol)
    manifests = {}
    for condition in CONDITIONS:
        manifests[condition] = generate(condition, output, args, evidence, contract)
        atomic_json(output / 'progress.json', dict(status='running', stage='prediction',
            completed_conditions=list(manifests), frame_count_per_condition=len(selected)))
    results, prediction_manifest, gt_manifest = replay(output, evidence, contract, manifests)
    if len(selected) == 829:
        if any(results[c]['counts']['official_gt'] != evidence['totals'][c]
               or results[c]['counts']['fixed_gt'] != 6224 for c in CONDITIONS):
            raise ValueError('Full D-1C GT totals differ from accepted audit')
        if results['fog']['counts']['removed_gt'] != 2551 or results['snow']['counts']['removed_gt'] != 370:
            raise ValueError('Full removed-target partition differs from D-1B')
    summary = summarize(results, contract, evidence)
    summary.update(checkpoint_sha256=contract['checkpoint_sha256'], config_sha256=contract['runtime_yaml_sha256'],
        official_source_commit=contract['official_source_commit'],
        prediction_cache_identity=prediction_manifest['identity'], gt_identity=gt_manifest['identity'])
    protocol.update(imported_modules_final=module_inventory(runtime),
        condition_config_sha256={c: sha256(output / 'official_runs' / c / 'config.yaml') for c in CONDITIONS},
        prediction_manifest_sha256=sha256(output / 'prediction_manifest.json'),
        prediction_cache_identity=prediction_manifest['identity'], gt_manifest_sha256=sha256(output / 'gt_manifest.json'),
        gt_identity=gt_manifest['identity'], cache_manifest_sha256={c: sha256(output / 'predictions' / c / 'manifest.json') for c in CONDITIONS})
    atomic_json(output / 'protocol.json', protocol)
    write_results(output, summary)
    artifacts = ['protocol.json', 'prediction_manifest.json', 'gt_manifest.json',
                 'per_frame_metrics.jsonl', 'removed_gt_metrics.jsonl', 'RESULTS.md']
    artifacts += [f'stat_{c}_{p}.json' for c in CONDITIONS for p in ('O', 'Ostar', 'F')]
    artifacts += [f'predictions/{c}/manifest.json' for c in CONDITIONS]
    summary['output_hashes'] = {n: sha256(output / n) for n in artifacts}
    atomic_json(output / 'progress.json', dict(status='complete', frame_count_per_condition=len(selected)))
    atomic_json(output / 'summary.json', summary)  # Final completion marker.
    print((output / 'RESULTS.md').read_text(encoding='utf-8'), flush=True)


def main():
    server_only()
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = dict(output_root='/data/cjm/datasets/logs/v2xr-dminus1c-protocol-v1',
        runtime_root='/data/cjm/datasets/v2xr-official-runtime-v1',
        model_dir='/data/cjm/datasets/logs/v2xr-official-attfuse-model-v1',
        d1a_root='/data/cjm/datasets/logs/v2xr-official-attfuse-clean-repro-v1',
        gt_audit_root='/data/cjm/datasets/logs/v2xr-official-gt-audit-v1',
        view_root='/data/cjm/datasets/v2xr-official-validate-v1')
    for name, default in defaults.items():
        parser.add_argument('--' + name.replace('_', '-'), default=default)
    parser.add_argument('--limit-frames', type=int, default=0)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stage', choices=('all', 'evaluate'), default='all')
    args = parser.parse_args()
    if not 0 <= args.limit_frames <= 829:
        raise ValueError('LIMIT_FRAMES must be 0 (all) or 1..829')
    sources = verify_sources()
    for name in defaults:
        setattr(args, name, str(no_test(getattr(args, name)).resolve()))
    output = Path(args.output_root)
    for source in [HERE.parent] + [Path(getattr(args, k)) for k in defaults if k != 'output_root']:
        if output == source or source in output.parents or output in source.parents:
            raise ValueError('Output must be independent of code, runtime, model and historical results')
    if Path('/data') not in output.parents or Path('/data') not in Path(os.environ.get('TMPDIR', '')).resolve().parents:
        raise ValueError('OUTPUT_ROOT and TMPDIR must be on /data')
    request = dict(arguments={k: v for k, v in vars(args).items() if k not in ('resume', 'stage')}, source_hashes=sources)
    with lock(output.with_suffix('.lock')), lock(Path(args.runtime_root).with_suffix('.run.lock')):
        marker = output / 'request.json'
        if output.exists() and any(output.iterdir()):
            if not args.resume or not marker.exists() or read_json(marker) != request:
                raise ValueError('Existing output requires identical parameters/source and RESUME=1')
        elif args.stage == 'evaluate':
            raise ValueError('Cache-only evaluation requires an existing prediction run')
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(marker, request)
        attempt = output / 'attempts' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        attempt.mkdir(parents=True)
        try:
            execute(args, output, sources)
        except Exception as error:
            failure = dict(status='failed', error=repr(error), traceback=traceback.format_exc(),
                advice='Keep prediction cache/attempt evidence. Never alter historical snapshots or regenerate predictions to choose a different GT definition.')
            atomic_json(attempt / 'failure.json', failure)
            atomic_json(output / 'last_failure.json', failure)
            raise


if __name__ == '__main__':
    main()
