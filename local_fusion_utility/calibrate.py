"""Choose all action thresholds on cached validation labels, never on test data."""
import argparse
from pathlib import Path


def candidate_rows(record, branch):
    return record[branch]['rows']


def true_utility(row, settings):
    value = row['target'].float()
    return float(value[0]-settings['lost_weight']*value[1]-settings['fp_weight']*value[2])


def predicted_scores(rows, method, predictor, settings, device):
    import torch
    from .fusion import DESCRIPTOR_NAMES
    if not rows:
        return []
    if method == 'confidence':
        peer = DESCRIPTOR_NAMES.index('peer_cls_max')
        full = DESCRIPTOR_NAMES.index('full_cls_max')
        return [float(row['descriptor'][peer]-row['descriptor'][full]) for row in rows]
    values = torch.stack([row['descriptor'].float() for row in rows]).to(device)[:, :, None, None]
    with torch.no_grad():
        output = predictor(values)
    if method == 'utility':
        value = output['outcome'][:, :, 0, 0]
        score = value[:, 0]-settings['lost_weight']*value[:, 1]-settings['fp_weight']*value[:, 2]
    elif method == 'loss_gain':
        score = output['loss_gain'][:, 0, 0, 0]
    else:
        raise ValueError(method)
    return score.detach().cpu().tolist()


def evaluate_threshold(paths, method, predictor, threshold, settings, device):
    import torch
    total, selected, harm, frames = 0., 0, 0., 0
    for path in paths:
        record = torch.load(path, map_location='cpu', weights_only=True)
        for branch in ('clean', 'weather'):
            rows = candidate_rows(record, branch)
            scores = predicted_scores(rows, method, predictor, settings, device)
            # Cached candidates may repeat a tile for different peers.  Inference
            # keeps only the highest-scored source at each tile, so calibration does too.
            best = {}
            for row, score in zip(rows, scores):
                tile = int(row['tile'])
                if tile not in best or score > best[tile][0]:
                    best[tile] = (score, row)
            for score, row in best.values():
                if score > threshold:
                    value = true_utility(row, settings)
                    total += value
                    selected += 1
                    harm += value < 0
            frames += 1
    return {'mean_true_utility': total/max(frames, 1),
            'selected_per_view': selected/max(frames, 1),
            'harmful_per_view': harm/max(frames, 1), 'views': frames}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--validation-cache', required=True)
    parser.add_argument('--utility-checkpoint', required=True)
    parser.add_argument('--loss-gain-checkpoint')
    parser.add_argument('--output-file', required=True)
    args = parser.parse_args()

    from gspr_communication.runtime import device, write_json, sha256
    from . import runtime as rt

    manifest, paths = rt.cache_files(args.validation_cache)
    if manifest['split'] != 'validation':
        raise ValueError('Thresholds must be calibrated on validation, not train/test')
    specification = manifest['contract']
    settings = rt.settings(specification['options'])
    target = device()
    utility, _ = rt.load_predictor(args.utility_checkpoint, specification, target, 'utility')
    methods = {'confidence': None, 'utility': utility}
    if args.loss_gain_checkpoint:
        loss_gain, _ = rt.load_predictor(args.loss_gain_checkpoint, specification, target, 'loss_gain')
        methods['loss_gain'] = loss_gain
    choices = {
        'confidence': settings['confidence_thresholds'],
        'utility': settings['utility_thresholds'],
        'loss_gain': settings['loss_gain_thresholds'],
    }
    result = {
        'schema': 1,
        'selection_data': 'complete validation cache; post-NMS action labels',
        'test_data_used': False,
        'cache': str(Path(args.validation_cache).resolve()),
        'cache_contract': specification,
        'checkpoints': {'utility': {'path': str(Path(args.utility_checkpoint).resolve()),
                                    'sha256': sha256(args.utility_checkpoint)}},
        'methods': {},
    }
    if args.loss_gain_checkpoint:
        result['checkpoints']['loss_gain'] = {
            'path': str(Path(args.loss_gain_checkpoint).resolve()),
            'sha256': sha256(args.loss_gain_checkpoint)}
    for method, predictor in methods.items():
        rows = []
        for threshold in choices[method]:
            metrics = evaluate_threshold(paths, method, predictor, float(threshold), settings, target)
            rows.append(dict(threshold=float(threshold), **metrics))
        # Prefer fewer modifications when validation utility ties.
        best = max(rows, key=lambda row: (row['mean_true_utility'], -row['selected_per_view'], row['threshold']))
        result['methods'][method] = {'threshold': best['threshold'], 'candidates': rows}
        print(method, best, flush=True)
    output = Path(args.output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.partial')
    write_json(temporary, result)
    temporary.replace(output)


if __name__ == '__main__':
    main()

