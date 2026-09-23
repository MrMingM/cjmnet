from pathlib import Path
from gspr_evidence import runtime as er
from gspr_communication.runtime import ROOT, sha256


def settings(options):
    s = dict(options['spatial_fusion'])
    if s['epochs'] < 1 or s['hidden'] < 1 or s['learning_rate'] <= 0:
        raise ValueError('Invalid training settings')
    if not 0 < s['max_gate'] <= 1 or s['change_penalty'] < 0:
        raise ValueError('Invalid modification constraints')
    if not s['scales'] or len(set(s['scales'])) != len(s['scales']) or any(x not in (0,1,2) for x in s['scales']):
        raise ValueError('Invalid scales')
    if options['communication']['value_bytes'] != 4:
        raise ValueError('v3 uses raw/full float32 features')
    for key in ('clean_ap70_tolerance','ap50_tolerance','minimum_mean_ap70_gain',
                'max_lost_tp_fraction','max_new_fp_per_baseline_tp'):
        if not 0 <= options['calibration'][key] <= 1:
            raise ValueError('Invalid calibration gate: '+key)
    return s


def contract(options, frontend, digest):
    value = er.contract(options, frontend, digest)
    value['method'] = 'spatial_source_fusion_v3'
    paths = list((ROOT/'local_fusion_v3').glob('*.py'))
    paths += [ROOT/'local_fusion_utility_v2'/name for name in ('fusion.py', 'outcomes.py', 'runtime.py')]
    paths += [ROOT/name for name in ('gspr_evidence/benchmark.py','gspr_communication/runtime.py',
              'gspr_review/runtime.py','gspr_review/resources.py','ceif_audit/scoring.py',
              'opencood/tools/train_utils.py','opencood/utils/eval_utils.py')]
    value['method_sources'] = {p.relative_to(ROOT).as_posix(): sha256(p) for p in paths}
    return value


def train_loader(hypes, options):
    import torch
    from torch.utils.data import DataLoader
    from gspr_communication.runtime import seed_worker
    ds, original, indices = er.make_loader(hypes, options, train=True)
    loader = DataLoader(original.dataset, batch_size=1, shuffle=True, drop_last=False,
                        num_workers=options['workers'], collate_fn=original.collate_fn,
                        worker_init_fn=seed_worker,
                        generator=torch.Generator().manual_seed(options['seed']))
    return ds, loader, indices


def context(model, ego, branch, verify=False):
    from local_fusion_utility_v2.runtime import prepare_context
    # Reuse the checked ego-query baseline and source alignment path.
    return prepare_context(model, ego, branch, {'tile_size': 8}, verify=verify)


def create(channels, options, variant, device):
    from .network import SpatialSourceFusion
    s = settings(options)
    return SpatialSourceFusion(channels, variant, s['hidden'], s['scales'], s['max_gate']).to(device)


def load(path, expected, device):
    import torch
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state['contract'] != expected:
        raise ValueError('Checkpoint/source/config/frontend mismatch')
    module = create(state['channels'], expected['options'], state['variant'], device)
    module.load_state_dict(state['module'])
    return module.eval(), state
