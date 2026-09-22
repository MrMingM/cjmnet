"""Standalone runtime and immutable contracts for local fusion utility."""
import json
from pathlib import Path

from gspr_evidence import runtime as evidence_runtime
from gspr_communication.runtime import ROOT, sha256, write_json


def settings(options):
    value = dict(options.get('utility') or {})
    defaults = {
        'tile_size': 8,
        'changed_scales': [0, 1],
        'regions_per_view': 3,
        'target_iou': .7,
        'fp_identity_iou': .7,
        'sampling_low_floor': .02,
        'sampling_low_ceiling': .2,
        'hidden': 64,
        'epochs': 8,
        'learning_rate': 3e-4,
        'weight_decay': 1e-4,
        'positive_weight': 4.0,
        'loss_gain_scale': 1000.0,
        'lost_weight': 1.0,
        'fp_weight': 1.0,
        'utility_thresholds': [0., .02, .05, .1, .2, .4, .8],
        'confidence_thresholds': [0., .02, .05, .1, .2],
        'loss_gain_thresholds': [0., .02, .05, .1, .2, .5, 1.0],
    }
    for key, default in defaults.items():
        value.setdefault(key, default)
    integer_positive = ('tile_size', 'regions_per_view', 'hidden', 'epochs')
    if any(not isinstance(value[key], int) or value[key] < 1 for key in integer_positive):
        raise ValueError('tile_size/regions_per_view/hidden/epochs must be positive integers')
    scales = tuple(int(item) for item in value['changed_scales'])
    if not scales or any(item not in (0, 1, 2) for item in scales) or len(set(scales)) != len(scales):
        raise ValueError('changed_scales must be unique values from 0,1,2')
    value['changed_scales'] = scales
    if not 0 < value['target_iou'] <= 1 or not 0 < value['fp_identity_iou'] <= 1:
        raise ValueError('IoU thresholds must be in (0,1]')
    if not 0 <= value['sampling_low_floor'] < value['sampling_low_ceiling'] <= 1:
        raise ValueError('Invalid non-GT sampling confidence range')
    for key in ('learning_rate', 'positive_weight', 'loss_gain_scale', 'lost_weight', 'fp_weight'):
        if float(value[key]) <= 0:
            raise ValueError(key+' must be positive')
    return value


def load_config(path, frontend):
    options, hypes = evidence_runtime.load_config(path, frontend)
    if int(options['communication'].get('value_bytes', 4)) != 4:
        raise ValueError('Scheme 1 first experiment fixes raw/full float32 features')
    settings(options)
    return options, hypes


def contract(options, frontend, digest):
    specification = evidence_runtime.contract(options, frontend, digest)
    sources = sorted((ROOT/'local_fusion_utility_v2').glob('*.py'))
    specification = dict(specification)
    specification['schema'] = 2
    specification['method'] = 'local_fusion_post_nms_utility_v2'
    specification['method_sources'] = {
        str(path.relative_to(ROOT)).replace('\\', '/'): sha256(path) for path in sources
    }
    return specification


def validate_contract(specification):
    evidence_runtime.validate_contract(specification)
    if specification.get('method') != 'local_fusion_post_nms_utility_v2':
        raise ValueError('Wrong cache/checkpoint method')
    if not specification.get('method_sources'):
        raise ValueError('Method source fingerprints are missing')
    for path, expected in specification['method_sources'].items():
        if not (ROOT/path).is_file() or sha256(ROOT/path) != expected:
            raise ValueError('Method source differs from cache/checkpoint: '+path)


def load_manifest(path, require_complete=True):
    manifest = json.loads((Path(path)/'manifest.json').read_text(encoding='utf-8'))
    if require_complete and not manifest.get('complete'):
        raise ValueError('Cache is incomplete: '+str(path))
    validate_contract(manifest['contract'])
    return manifest


def prepare_context(model, ego, branch, utility_settings, verify=True):
    import torch
    from .fusion import build_context
    encoded = model.encode(evidence_runtime.input_branch(ego, branch))
    context = build_context(model.engine.base, encoded['levels'], utility_settings['tile_size'])
    if verify:
        masks = model.empty_masks(encoded)
        masks[:] = 1
        reference, _ = model.detect(encoded, masks, serialize=False)
        for key in ('psm', 'rm'):
            if not torch.allclose(reference[key], context['baseline_prediction'][key], atol=2e-5, rtol=2e-5):
                error = float((reference[key]-context['baseline_prediction'][key]).abs().max())
                raise RuntimeError(f'Full-fusion reimplementation mismatch for {key}: {error}')
    context['encoded'] = encoded
    return context


def raw_feature_bytes(context):
    return sum(value[1:].numel()*value.element_size() for value in context['levels'])


def save_checkpoint(path, predictor, optimizer, epoch, validation, variant, specification):
    import torch
    target = Path(path)
    temporary = target.with_suffix('.partial')
    torch.save({
        'schema': 1,
        'variant': variant,
        'epoch': int(epoch),
        'validation': validation,
        'predictor': predictor.state_dict(),
        'optimizer': optimizer.state_dict(),
        'input_channels': predictor.input_channels,
        'contract': specification,
    }, temporary)
    temporary.replace(target)


def load_predictor(path, expected_contract, device, expected_variant=None):
    import torch
    from .network import UtilityPredictor
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state.get('contract') != expected_contract:
        raise ValueError('Selector checkpoint and current pipeline contract differ')
    if expected_variant and state.get('variant') != expected_variant:
        raise ValueError('Unexpected selector variant')
    model = UtilityPredictor(state['input_channels'], settings(expected_contract['options'])['hidden'])
    model.load_state_dict(state['predictor'], strict=True)
    return model.to(device).eval(), state


def cache_files(cache):
    manifest = load_manifest(cache)
    paths = [Path(cache)/f'{int(index):08d}.pt' for index in manifest['indices']]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError('Missing cache files: '+', '.join(missing[:3]))
    return manifest, paths
