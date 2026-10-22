"""Server-only adapter around the unchanged complete frozen evaluation function."""
import argparse
from contextlib import ExitStack
import os
from pathlib import Path

from v2xr_author_fog_eval.core import (CHECKPOINT_SHA, CHECKPOINT_URL, CONDITIONS, FROZEN_GT,
    HERE, ROOT, bind, checkpoint_file, completed, frozen_modules, identity, read,
    reference_source, sha256, text_file, verify_sources, write)
from v2xr_author_fog_eval.checkpoint import safe_read, strict_load


def arguments(request):
    args = argparse.Namespace(**request['arguments'])
    # These legacy path fields are checked for overlap only. No train data is opened.
    args.train_root = '/data/cjm/datasets/v2xr-official-train-v2'
    args.data_root = '/data/cjm/datasets/V2X-R/train'
    args.resume = True  # The controller alone authorizes resume and binds run identity.
    return args


def evidence(args):
    legacy, evaluator = frozen_modules()
    legacy.gpu()
    from provenance import verify_inputs
    original = verify_inputs(args)
    frozen = legacy.frozen_c(args)
    if frozen['gt']['identity'] != FROZEN_GT:
        raise ValueError('Unexpected frozen GT identity')
    return legacy, evaluator, original, frozen


def check_config(args, original):
    import yaml
    from run import config_difference
    template = ROOT / 'v2xr_official_repro/upstream/opencood/hypes_yaml/V2X-R/LiDAR/V2XR_attfuse.yaml'
    config = Path(args.model_dir) / 'config.yaml'
    expected = yaml.safe_load(template.read_text(encoding='utf-8'))
    actual = yaml.safe_load(config.read_text(encoding='utf-8'))
    changed = config_difference(expected, actual)
    if any(k not in ('root_dir', 'validate_dir') for k in changed):
        raise ValueError('Frozen config differs from official pure LiDAR template beyond root paths: ' + repr(changed))
    required = (actual['model']['core_method'] == 'point_pillar_intermediate'
        and actual['model']['args']['use_modality'] == 'processed_lidar'
        and actual['fusion']['core_method'] == 'IntermediateFusionDatasetV2XR'
        and actual['fusion']['args']['proj_first'] is True and actual['fov'] is True
        and actual['postprocess']['target_args']['score_threshold'] == .20
        and actual['postprocess']['nms_thresh'] == .15
        and actual['eval_sim'] is False and actual['train_sim'] is False)
    if not required or original['b']['d1a_summary']['actual']['max_cav'] != 7:
        raise ValueError('Pure LiDAR/configuration invariant differs')
    return dict(path=str(config), sha256=sha256(config), template_sha256=sha256(template),
                differences=changed, permitted_differences=['root_dir', 'validate_dir'])


def preflight(root, request):
    args = arguments(request)
    checkpoint_file(args.checkpoint)
    legacy, _, original, frozen = evidence(args)
    config = check_config(args, original)
    source = reference_source(args.cd_results)
    from v2xr_author_fog_eval.comparison import validate_summary
    reference = read(source['path'])
    for name in ('C', 'D'):
        validate_summary(reference['results'][name], name)
    record = dict(status='complete', run_identity=request['identity'],
        checkpoint=dict(path=args.checkpoint, sha256=CHECKPOINT_SHA, official_source=CHECKPOINT_URL),
        evaluation_kind='external_fixed_checkpoint; no local training selection',
        configuration=config, data_identity=original['input_manifest']['identity'],
        input_manifest_sha256=sha256(Path(args.gt_audit_root) / 'input_manifest.json'),
        validation_frames=829, selected_frame_keys=original['b']['selected_frame_keys'],
        official_gt=original['totals'], fixed_gt=6224, frozen_gt_identity=frozen['gt']['identity'],
        protocols=legacy.DEFINITIONS, d1a=original['d1a_artifact_sha256'],
        d1b=original['d1b_artifact_sha256'], d1c=frozen['hashes'],
        cd_reference=source, source_identity=identity(verify_sources()),
        context_note='AI_CONTEXT.md in this checkout ends at section 49.6; section 50 was not present. C/D values come from the archived FINAL_RESULTS.json.',
        no_training=True, no_test=True, data_integrity='all frozen input bytes, links and manifests revalidated')
    bind(root / 'preflight.json', record)
    return record


def prepare_runtime(args, original, legacy):
    runtime = legacy.runtime(args, original['a'])
    import torch
    if not torch.cuda.is_available() or not torch.version.hip or torch.cuda.device_count() != 1:
        raise RuntimeError('Expected the frozen HIP runtime and exactly one visible GPU')
    from opencood.data_utils import datasets
    if datasets.COM_RANGE != 70:
        raise ValueError('Official communication range must remain 70m')
    return runtime, torch


def model_name(model):
    actual = type(model).__module__ + '.' + type(model).__name__
    if actual != 'opencood.models.point_pillar_intermediate.PointPillarIntermediate':
        raise ValueError('Author checkpoint requires the official pure LiDAR PointPillarIntermediate, got ' + actual)
    return actual


def load_audit(root, request):
    args = arguments(request)
    legacy, _, original, _ = evidence(args)
    config = check_config(args, original)
    record = dict(status='failed', run_identity=request['identity'],
        checkpoint_path=str(checkpoint_file(args.checkpoint)), checkpoint_sha256=CHECKPOINT_SHA,
        official_source=CHECKPOINT_URL, safe_loading='torch.load(map_location=cpu, weights_only=True); no unsafe fallback',
        configuration=config, external_fixed_checkpoint=True, no_training=True)
    try:
        _, torch = prepare_runtime(args, original, legacy)
        from opencood.hypes_yaml import yaml_utils
        from opencood.tools import train_utils
        from observe import normalize_state, module_inventory
        state, parsing = safe_read(torch, args.checkpoint)
        record['parsing'] = parsing
        if list(normalize_state(state)) != list(state):
            raise ValueError('Legacy name conversion would alter already normalized keys; no nonstandard remapping allowed')
        model = train_utils.create_model(yaml_utils.load_yaml(str(Path(args.model_dir) / 'config.yaml')))
        record['model_class'] = model_name(model)
        strict_load(model, state, torch, record)
        record.update(status='complete', gpu_name=torch.cuda.get_device_name(0),
            physical_gpu=os.environ['ROCR_VISIBLE_DEVICES'], hip=torch.version.hip,
            imported_modules=module_inventory(Path(args.runtime_root)))
        # Run the seven existing synthetic AP/None/one-to-one contract tests on the server only.
        from test_contract import run_tests
        run_tests()
        record['frozen_protocol_synthetic_tests'] = 'passed (7 original tests)'
    except Exception as error:
        record['error'] = repr(error)
        write(root / 'checkpoint_audit.json', record)
        raise
    previous = root / 'checkpoint_audit.json'
    if previous.exists() and read(previous).get('status') == 'complete':
        bind(previous, record)
    else:
        write(previous, record)
    return record


def evaluate(root, request, smoke):
    args = arguments(request)
    destination = root / ('smoke' if smoke else 'author_evaluation')
    legacy, evaluator = frozen_modules()
    legacy.gpu()
    audit = read(root / 'checkpoint_audit.json')
    if audit['status'] != 'complete' or not audit['all_tensors_loaded_exactly'] or audit['run_identity'] != request['identity']:
        raise ValueError('Successful author model loading audit is required')
    checkpoint_file(args.checkpoint)
    calls = []
    # Loading adaptation is narrowly scoped. GT, AP, matching, thresholds and data paths stay frozen.
    with ExitStack() as stack:
        original_runtime = evaluator.runtime

        def author_runtime(runtime_args, a):
            result = original_runtime(runtime_args, a)
            import torch
            from opencood.tools import train_utils
            state, parsing = safe_read(torch, args.checkpoint)
            if parsing != audit['parsing']:
                raise ValueError('Checkpoint parsing differs from completed loading audit')
            native_load = torch.load

            def safe_author_load(path, *positional, **keywords):
                # The frozen generator requests the same author aliases with weights_only=False.
                # Replace that request with the safely parsed and audited tensor dictionary.
                # No pickle is read through the permissive legacy request.
                if positional or checkpoint_file(path) != Path(args.checkpoint).resolve():
                    raise ValueError('Unexpected checkpoint load in author inference')
                if keywords.get('map_location', 'cpu') != 'cpu':
                    raise ValueError('Checkpoint must first be audited on CPU')
                return state

            def strict_author_load(saved_path, model, epoch=None, start_from_best=True):
                directory = Path(saved_path).resolve(strict=True)
                if destination.resolve() not in directory.parents or epoch != 0 or start_from_best:
                    raise ValueError('Unexpected author checkpoint alias/epoch selection')
                for alias in ('0.pth', 'net_epoch0.pth'):
                    if not (directory / alias).is_symlink() or checkpoint_file(directory / alias) != Path(args.checkpoint).resolve():
                        raise ValueError('Model alias must point to the original author checkpoint')
                model_name(model)
                item = dict(model_dir=str(directory), epoch=0)
                strict_load(model, state, torch, item)
                def inference_guard(module, inputs):
                    if module.training or torch.is_grad_enabled():
                        raise ValueError('Author forward requires eval mode and torch.no_grad()')

                def finite_detection(module, inputs, output):
                    if any(not bool(torch.isfinite(output[k]).all()) for k in ('psm', 'rm')):
                        raise ValueError('Nonfinite detection logits/regression')

                model.register_forward_pre_hook(inference_guard)
                model.register_forward_hook(finite_detection)
                calls.append(dict(model_dir=item['model_dir'], strict=True,
                                  tensors_loaded_exactly=item['tensor_count'], eval_no_grad_guard=True,
                                  finite_detection_observer=True))
                return 0, model

            stack.enter_context(legacy.patched(torch, 'load', safe_author_load))
            stack.enter_context(legacy.patched(train_utils, 'load_model', strict_author_load))
            if native_load is safe_author_load:
                raise RuntimeError('Unexpected nested checkpoint adapter')
            return result

        stack.enter_context(legacy.patched(evaluator, 'runtime', author_runtime))
        old_jsonable = evaluator.jsonable

        def external_contract(value):
            value = old_jsonable(value)
            if isinstance(value, dict) and value.get('stage') == 'D-1D':
                value.update(stage='V2XR_AUTHOR_FOG_EVAL', model_variant='Author_Fog_AttFuse',
                    model_origin='external_fixed_checkpoint', local_training_selection_required=False,
                    loading_adapter='safe weights_only read, standard wrapper/uniform module. only, strict=True and exact equality')
            return value

        stack.enter_context(legacy.patched(evaluator, 'jsonable', external_contract))
        own = dict(run_identity=request['identity'], sources=verify_sources(),
                   preflight_sha256=sha256(root / 'preflight.json'),
                   checkpoint_audit_sha256=sha256(root / 'checkpoint_audit.json'))
        summary = evaluator.evaluate(args, own, args.checkpoint, destination,
                                     conditions=CONDITIONS, limit=12 if smoke else 0)
    checkpoint_file(args.checkpoint)
    # Add provenance to this new output only; refresh hashes so completed-stage resume remains verifiable.
    summary.update(stage='V2XR_AUTHOR_FOG_EVAL', model_variant='Author_Fog_AttFuse',
                   model_origin='external_fixed_checkpoint', official_checkpoint_source=CHECKPOINT_URL,
                   no_training=True, no_test=True)
    protocol = read(destination / 'protocol.json')
    if 'author_loading' not in protocol:
        protocol['author_loading'] = dict(checkpoint_audit_sha256=sha256(root / 'checkpoint_audit.json'),
            load_calls=calls, strict=True, safe_read=True, original_checkpoint=CHECKPOINT_URL)
        write(destination / 'protocol.json', protocol)
    from v2xr_author_fog_eval.comparison import evaluation_markdown
    text_file(destination / 'RESULTS.md', evaluation_markdown(summary))
    summary['output_hashes'].update({name: sha256(destination / name) for name in ('protocol.json', 'RESULTS.md')})
    write(destination / 'summary.json', summary)
    completed(destination / 'summary.json')
    return summary
