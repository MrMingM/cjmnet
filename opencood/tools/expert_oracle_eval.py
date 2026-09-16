"""Offline frame-level Oracle for complementary weather specialists.

This diagnostic loads two or more fixed detector checkpoints, evaluates every
expert on exactly the same frames, and uses ground truth to select the best
expert per frame.  The resulting Oracle AP is an upper bound for any future
deployable weather router; it must never be reported as an online method.
"""

import argparse
import copy
import json
import os
import random
import sys
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils
from opencood.tools.test_weather_where2comm import (
    load_model_for_eval,
    parse_weather_items,
    resolve_device,
    resolve_validate_dir,
    set_max_cav_override,
)
from opencood.utils import eval_utils


IOU_THRESHOLDS = (0.3, 0.5, 0.7)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Measure frame-level AP headroom between fixed experts.')
    parser.add_argument('--hypes_yaml', required=True)
    parser.add_argument(
        '--expert', action='append', required=True,
        help='Expert in name=/absolute/or/relative/checkpoint.pth form. '
             'Repeat for each expert.')
    parser.add_argument(
        '--weather', action='append', required=True,
        help='Evaluation split in name=/path form. Repeat as needed.')
    parser.add_argument('--selection_iou', type=float, default=0.5,
                        choices=IOU_THRESHOLDS)
    parser.add_argument('--max_cav', type=int, default=0)
    parser.add_argument('--start_index', type=int, default=0)
    parser.add_argument('--max_frames', type=int, default=0)
    parser.add_argument('--frame_stride', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=20260726)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output_dir', required=True)
    return parser.parse_args()


def parse_named_paths(items, kind):
    parsed = []
    names = set()
    for item in items:
        if '=' not in item:
            raise ValueError('%s must use name=/path: %s' % (kind, item))
        name, path = item.split('=', 1)
        name = name.strip()
        path = os.path.abspath(os.path.expanduser(path.strip()))
        if not name or not path:
            raise ValueError('invalid %s: %s' % (kind, item))
        if name in names:
            raise ValueError('duplicate %s name: %s' % (kind, name))
        if not os.path.isfile(path):
            raise FileNotFoundError('%s checkpoint not found: %s' %
                                    (kind, path))
        names.add(name)
        parsed.append((name, path))
    if len(parsed) < 2:
        raise ValueError('at least two experts are required')
    return parsed


def empty_result_stat():
    return {
        threshold: {'tp': [], 'fp': [], 'gt': 0, 'score': []}
        for threshold in IOU_THRESHOLDS
    }


def update_stats(stats, pred_boxes, pred_scores, gt_boxes):
    for threshold in IOU_THRESHOLDS:
        eval_utils.caluclate_tp_fp(
            pred_boxes, pred_scores, gt_boxes, stats, threshold)


def calculate_aps(stats):
    result = {}
    for threshold in IOU_THRESHOLDS:
        ap, _, _ = eval_utils.calculate_ap(
            copy.deepcopy(stats), threshold, True)
        result['ap%d' % int(threshold * 100)] = float(ap)
    return result


def frame_metrics(pred_boxes, pred_scores, gt_boxes, selection_iou):
    stats = {
        selection_iou: {'tp': [], 'fp': [], 'gt': 0, 'score': []}
    }
    eval_utils.caluclate_tp_fp(
        pred_boxes, pred_scores, gt_boxes, stats, selection_iou)
    frame_stat = stats[selection_iou]
    if frame_stat['gt'] > 0:
        ap, _, _ = eval_utils.calculate_ap(
            copy.deepcopy(stats), selection_iou, True)
        frame_ap = float(ap)
    else:
        frame_ap = 0.0
    scores = np.asarray(frame_stat['score'], dtype=np.float64)
    return {
        'frame_ap': frame_ap,
        'true_positives': int(np.sum(frame_stat['tp'])),
        'false_positives': int(np.sum(frame_stat['fp'])),
        'prediction_count': int(scores.size),
        'mean_score': float(scores.mean()) if scores.size else 0.0,
        'max_score': float(scores.max()) if scores.size else 0.0,
    }


def oracle_key(record):
    metrics = record['metrics']
    return (
        metrics['frame_ap'],
        metrics['true_positives'],
        -metrics['false_positives'],
        metrics['mean_score'],
        metrics['max_score'],
    )


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def build_models(hypes, experts, device):
    models = OrderedDict()
    for name, checkpoint in experts:
        model = train_utils.create_model(copy.deepcopy(hypes))
        model = load_model_for_eval(
            os.path.dirname(checkpoint), model, 0, checkpoint)
        model.to(device)
        model.eval()
        models[name] = model
        print('Loaded expert %s: %s' % (name, checkpoint))
    return models


def evaluate_weather(models, hypes, weather_name, validate_dir, opt):
    weather_hypes = copy.deepcopy(hypes)
    weather_hypes['validate_dir'] = resolve_validate_dir(validate_dir)
    dataset = build_dataset(weather_hypes, visualize=True, train=False)

    if opt.start_index < 0 or opt.frame_stride <= 0:
        raise ValueError('invalid start_index/frame_stride')
    indices = list(range(
        opt.start_index, len(dataset), opt.frame_stride))
    if opt.max_frames > 0:
        indices = indices[:opt.max_frames]
    evaluation_dataset = Subset(dataset, indices)

    generator = torch.Generator()
    generator.manual_seed(opt.seed)
    loader = DataLoader(
        evaluation_dataset,
        batch_size=1,
        num_workers=opt.num_workers,
        collate_fn=dataset.collate_batch_test,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=generator)

    expert_stats = {
        name: empty_result_stat() for name in models
    }
    oracle_stats = empty_result_stat()
    selection_counts = {name: 0 for name in models}
    selection_records = []

    description = 'Expert Oracle %s' % weather_name
    for local_index, batch_data in tqdm(
            enumerate(loader), total=len(loader), desc=description):
        dataset_index = indices[local_index]
        batch_data['ego'].pop('weather_augmentation_stats', None)
        batch_data['ego'].pop('processed_lidar_weather', None)
        batch_data = train_utils.to_device(batch_data, opt.device_object)

        frame_results = OrderedDict()
        with torch.no_grad():
            for name, model in models.items():
                output_dict = OrderedDict(
                    ego=model(batch_data['ego']))
                pred_boxes, pred_scores, gt_boxes = dataset.post_process(
                    batch_data, output_dict)
                update_stats(
                    expert_stats[name], pred_boxes, pred_scores, gt_boxes)
                frame_results[name] = {
                    'pred_boxes': pred_boxes,
                    'pred_scores': pred_scores,
                    'gt_boxes': gt_boxes,
                    'metrics': frame_metrics(
                        pred_boxes, pred_scores, gt_boxes,
                        opt.selection_iou),
                }

        selected_name, selected = max(
            frame_results.items(),
            key=lambda item: oracle_key(item[1]))
        selection_counts[selected_name] += 1
        update_stats(
            oracle_stats,
            selected['pred_boxes'],
            selected['pred_scores'],
            selected['gt_boxes'])
        selection_records.append({
            'weather': weather_name,
            'frame_index': local_index,
            'dataset_index': dataset_index,
            'selected_expert': selected_name,
            'experts': {
                name: result['metrics']
                for name, result in frame_results.items()
            },
        })

    expert_aps = {
        name: calculate_aps(stats)
        for name, stats in expert_stats.items()
    }
    oracle_aps = calculate_aps(oracle_stats)
    best_fixed = max(
        expert_aps,
        key=lambda name: expert_aps[name][
            'ap%d' % int(opt.selection_iou * 100)])
    selection_key = 'ap%d' % int(opt.selection_iou * 100)
    return {
        'weather': weather_name,
        'validate_dir': weather_hypes['validate_dir'],
        'frames': len(indices),
        'selection_iou': opt.selection_iou,
        'expert_ap': expert_aps,
        'oracle_ap': oracle_aps,
        'best_fixed_expert': best_fixed,
        'oracle_gain_vs_best_fixed': (
            oracle_aps[selection_key] -
            expert_aps[best_fixed][selection_key]),
        'selection_counts': selection_counts,
    }, selection_records


def main():
    opt = parse_args()
    seed_everything(opt.seed)
    opt.device_object = resolve_device(opt.device)
    experts = parse_named_paths(opt.expert, 'expert')
    weather_items = parse_weather_items(opt.weather)

    hypes = yaml_utils.load_yaml(opt.hypes_yaml)
    set_max_cav_override(hypes, opt.max_cav)
    # Target weather folders already contain corrupted LiDAR. Never apply a
    # second synthetic augmentation during this complementarity diagnostic.
    hypes.setdefault('weather_augmentation', {})[
        'apply_to_validation'] = False

    print('Creating %d fixed experts on %s' %
          (len(experts), opt.device_object))
    models = build_models(hypes, experts, opt.device_object)

    os.makedirs(opt.output_dir, exist_ok=True)
    summaries = []
    records = []
    for weather_name, validate_dir in weather_items:
        summary, weather_records = evaluate_weather(
            models, hypes, weather_name, validate_dir, opt)
        summaries.append(summary)
        records.extend(weather_records)
        print(
            '%s: experts=%s oracle=%s gain@%.2f=%+.4f counts=%s' %
            (weather_name, summary['expert_ap'], summary['oracle_ap'],
             opt.selection_iou,
             summary['oracle_gain_vs_best_fixed'],
             summary['selection_counts']))

    summary_path = os.path.join(
        opt.output_dir, 'expert_oracle_summary.json')
    records_path = os.path.join(
        opt.output_dir, 'expert_oracle_selections.jsonl')
    with open(summary_path, 'w', encoding='utf-8') as file:
        json.dump({
            'experts': dict(experts),
            'seed': opt.seed,
            'weather_results': summaries,
        }, file, indent=2, ensure_ascii=False)
    with open(records_path, 'w', encoding='utf-8') as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + '\n')

    print('Expert Oracle finished')
    print('Summary: %s' % summary_path)
    print('Selections: %s' % records_path)


if __name__ == '__main__':
    main()
