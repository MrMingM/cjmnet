# -*- coding: utf-8 -*-
"""Offline counterfactual Oracle evaluation for PointPillar-Where2comm.

The script encodes all CAVs once per frame and repeats only feature fusion for
four coalition families: ego, all, ego+neighbor_i, and all-neighbor_i.  It then
tracks every GT object across those passes and measures the neighbor's isolated
and coalition-conditional marginal utility.

Example
-------
HIP_VISIBLE_DEVICES=1 CUDA_VISIBLE_DEVICES=1 python -u \
  opencood/tools/oracle_eval.py \
  --model_dir opencood/logs/point_pillar_where2comm_2026_06_17_10_29_43 \
  --hypes_yaml opencood/hypes_yaml/point_pillar_where2comm_opv2v.yaml \
  --validate_dir /home/cjm/datasets/opv2v_synthetic_validation/locked \
  --eval_epoch 50 --max_frames 300 --device cuda
"""

import argparse
import copy
import csv
import json
import math
import os
import random
import sys
from collections import OrderedDict, defaultdict

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils
from opencood.tools.train_oracle_selector import (
    CoalitionRanker, encode_candidate, encode_frame_base)
from opencood.models.sub_modules.spatial_utility_net import (
    SpatialUtilityNet, build_neighbor_input, utility_to_gate)
from opencood.utils import box_utils, common_utils, eval_utils


SCHEMA_VERSION = 'oracle-utility-v4'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Counterfactual marginal-utility evaluation for Where2comm.')
    parser.add_argument('--model_dir', required=True)
    parser.add_argument('--hypes_yaml', default='',
                        help='Evaluation YAML. Defaults to model_dir/config.yaml.')
    parser.add_argument('--validate_dir', required=True,
                        help='Synthetic locked scenario root.')
    parser.add_argument('--eval_epoch', type=int, default=50)
    parser.add_argument('--start_index', type=int, default=0)
    parser.add_argument('--max_frames', type=int, default=300)
    parser.add_argument('--frame_stride', type=int, default=1)
    parser.add_argument('--iou_thresh', type=float, default=0.5)
    parser.add_argument('--score_thresh', type=float, default=None)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--max_cav', type=int, default=5,
                        help='Force the multi-agent dataset/model CAV limit.')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--input_branch', choices=['clean', 'weather_augmented'],
                        default='clean')
    parser.add_argument('--weather_augmentation_seed', type=int, default=None)
    parser.add_argument('--seed', type=int, default=20260715,
                        help='Seed dataset workers and inference preprocessing.')
    parser.add_argument('--keep_weather_reliability', action='store_true',
                        help='Keep YAML weather reliability. Default is off.')
    parser.add_argument('--output_dir', default='',
                        help='Defaults to model_dir/oracle_eval_locked.')
    parser.add_argument('--output_prefix', default='oracle_utility_results')
    parser.add_argument('--selector_checkpoint', default='',
                        help='Optional trained coalition ranker checkpoint.')
    parser.add_argument('--selector_safety_margin', type=float, default=None,
                        help='Override a v2 selector all-fallback margin.')
    parser.add_argument('--domain_name', default='',
                        help='Metadata label for selector training exports.')
    parser.add_argument('--spatial_cache_dir', default='',
                        help='Optional directory for spatial utility training NPZs.')
    parser.add_argument('--spatial_selector_checkpoint', default='',
                        help='Optional spatial utility CNN checkpoint.')
    parser.add_argument(
        '--spatial_gate_mode', choices=['soft', 'hard'], default='soft',
        help='Hard mode preserves exact all-fusion features except at '
             'high-confidence harmful cells.')
    parser.add_argument(
        '--spatial_utility_threshold', type=float, default=None,
        help='Optional runtime override for the utility threshold.')
    parser.add_argument(
        '--spatial_harm_probability_threshold', type=float, default=0.9,
        help='Minimum harmful probability required by hard gating.')
    parser.add_argument(
        '--spatial_gate_floor', type=float, default=None,
        help='Optional runtime override for the suppressed-cell weight.')
    return parser.parse_args()


def resolve_device(device_arg):
    if device_arg == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_arg)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def seed_worker(_worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def load_model(model_dir, model, epoch):
    if epoch <= 0:
        _, model = train_utils.load_saved_model(model_dir, model)
        return model
    path = os.path.join(model_dir, 'net_epoch%d.pth' % epoch)
    if not os.path.exists(path):
        raise FileNotFoundError('Checkpoint not found: %s' % path)
    print('Loading checkpoint: %s' % path)
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint, strict=False)
    del checkpoint
    return model


def load_selector(path, device, safety_margin_override=None):
    if not path:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError('Selector checkpoint not found: %s' % path)
    checkpoint = torch.load(path, map_location=device)
    selector = CoalitionRanker(
        int(checkpoint['input_dim']), checkpoint['hidden_dims'],
        float(checkpoint['dropout'])).to(device)
    selector.load_state_dict(checkpoint['model_state_dict'])
    selector.eval()
    strategy = checkpoint.get('selection_strategy', 'argmax')
    safety_margin = float(
        checkpoint.get('safety_margin', 0.0)
        if safety_margin_override is None else safety_margin_override)
    if strategy == 'regret_with_all_fallback' and safety_margin < 0.0:
        raise ValueError(
            'A conservative selector requires a non-negative safety margin; '
            'retrain with the fixed calibrator or explicitly pass '
            '--selector_safety_margin 0.0 for diagnostic evaluation.')
    return {
        'model': selector,
        'schema': checkpoint['feature_schema'],
        'mean': np.asarray(checkpoint['normalizer_mean'], dtype=np.float32),
        'std': np.asarray(checkpoint['normalizer_std'], dtype=np.float32),
        'strategy': strategy,
        'safety_margin': safety_margin,
        'path': path
    }


def predict_candidate(selector_bundle, selection_record, device):
    schema = selector_bundle['schema']
    base = encode_frame_base(selection_record, schema)
    candidates = selection_record['candidates']
    matrix = np.stack([
        encode_candidate(base, item, schema) for item in candidates
    ])
    matrix = (matrix - selector_bundle['mean']) / selector_bundle['std']
    with torch.no_grad():
        scores = selector_bundle['model'](
            torch.from_numpy(matrix).to(device)).detach().cpu().numpy()
    if selector_bundle['strategy'] == 'regret_with_all_fallback':
        all_index = next(index for index, item in enumerate(candidates)
                         if item['name'] == 'all')
        alternatives = [index for index in range(len(candidates))
                        if index != all_index]
        best_alternative = max(alternatives, key=lambda index: scores[index])
        score_margin = float(
            scores[best_alternative] - scores[all_index])
        selected_index = best_alternative \
            if score_margin > selector_bundle['safety_margin'] else all_index
    else:
        selected_index = int(np.argmax(scores))
    return selected_index, [float(x) for x in scores]


def load_spatial_selector(path, device):
    if not path:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(
            'Spatial selector checkpoint not found: %s' % path)
    checkpoint = torch.load(path, map_location=device)
    model = SpatialUtilityNet(
        int(checkpoint.get('hidden_channels', 32))).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return {
        'model': model,
        'utility_threshold': float(
            checkpoint.get('utility_threshold', 0.0)),
        'temperature': float(checkpoint.get('temperature', 0.02)),
        'gate_floor': float(checkpoint.get('gate_floor', 0.05)),
        'foreground_threshold': float(
            checkpoint.get('foreground_threshold', 0.05)),
        'path': path
    }


def predict_spatial_agent_weights(bundle, encoded, cav_count,
                                  lidar_range, voxel_size, gate_mode='soft',
                                  utility_threshold=None,
                                  harm_probability_threshold=0.9,
                                  gate_floor=None):
    confidence = encoded['psm_single'][:cav_count].sigmoid().amax(dim=1)
    height, width = confidence.shape[-2:]
    density = build_point_density_map(
        encoded, cav_count, height, width, lidar_range, voxel_size)
    weights = torch.ones(
        cav_count, 1, height, width, device=confidence.device,
        dtype=confidence.dtype)
    if cav_count <= 1:
        return weights, {
            'mean': 1.0,
            'low_fraction': 0.0,
            'support_fraction': 0.0,
            'support_mean': 1.0,
            'suppressed_fraction': 0.0
        }
    inputs = torch.stack([
        build_neighbor_input(confidence, density, neighbor_index)
        for neighbor_index in range(1, cav_count)
    ])
    with torch.no_grad():
        output = bundle['model'](inputs)
        support = torch.stack([
            torch.maximum(confidence[0], confidence[neighbor_index]) >=
            bundle['foreground_threshold']
            for neighbor_index in range(1, cav_count)
        ]).unsqueeze(1)
        effective_utility_threshold = (
            bundle['utility_threshold']
            if utility_threshold is None else float(utility_threshold))
        effective_gate_floor = (
            bundle['gate_floor']
            if gate_floor is None else float(gate_floor))
        if gate_mode == 'hard':
            suppress = (
                support &
                (output['utility'] < effective_utility_threshold) &
                (torch.sigmoid(output['harm_logit']) >=
                 float(harm_probability_threshold))
            )
            neighbor_weights = torch.where(
                suppress,
                torch.full_like(output['utility'], effective_gate_floor),
                torch.ones_like(output['utility']))
        else:
            suppress = torch.zeros_like(support)
            neighbor_weights = utility_to_gate(
                output['utility'], output['harm_logit'], support,
                effective_utility_threshold, bundle['temperature'],
                effective_gate_floor)
    weights[1:] = neighbor_weights
    support_float = support.float()
    support_count = support_float.sum().clamp_min(1.0)
    return weights, {
        'mean': float(neighbor_weights.mean().item()),
        'low_fraction': float((neighbor_weights < 0.5).float().mean().item()),
        'support_fraction': float(support_float.mean().item()),
        'support_mean': float(
            (neighbor_weights * support_float).sum().div(
                support_count).item()),
        'suppressed_fraction': float(suppress.float().mean().item())
    }


def configure_hypes(opt):
    config_path = opt.hypes_yaml or os.path.join(opt.model_dir, 'config.yaml')
    hypes = yaml_utils.load_yaml(config_path)
    hypes['validate_dir'] = os.path.abspath(os.path.expanduser(
        opt.validate_dir))
    if opt.max_cav < 2:
        raise ValueError('--max_cav must be at least 2 for Oracle evaluation')
    hypes.setdefault('train_params', {})['max_cav'] = int(opt.max_cav)
    hypes['model']['args']['max_cav'] = int(opt.max_cav)

    if opt.score_thresh is not None:
        if not 0.0 <= opt.score_thresh <= 1.0:
            raise ValueError('--score_thresh must be in [0, 1]')
        hypes['postprocess']['target_args']['score_threshold'] = \
            float(opt.score_thresh)

    if not opt.keep_weather_reliability:
        fusion = hypes['model']['args']['where2comm_fusion']
        fusion.setdefault('weather_reliability', {})['enable'] = False

    if opt.input_branch == 'weather_augmented':
        augmentation = hypes.get('weather_augmentation', {})
        if not augmentation:
            raise ValueError(
                'weather_augmented requires weather_augmentation in YAML')
        augmentation['apply_to_validation'] = True
        if opt.weather_augmentation_seed is not None:
            augmentation['seed'] = int(opt.weather_augmentation_seed)
    elif opt.weather_augmentation_seed is not None:
        raise ValueError(
            '--weather_augmentation_seed requires --input_branch weather_augmented')
    return hypes, config_path


def select_indices(dataset_len, start, count, stride):
    if start < 0 or start >= dataset_len:
        raise ValueError('--start_index is outside the dataset')
    if count <= 0:
        raise ValueError('--max_frames must be positive')
    if stride <= 0:
        raise ValueError('--frame_stride must be positive')
    stop = min(dataset_len, start + count * stride)
    return list(range(start, stop, stride))[:count]


def scenario_index_for_sample(dataset, sample_index):
    for scenario_index, end_index in enumerate(dataset.len_record):
        if sample_index < int(end_index):
            return scenario_index
    raise IndexError('Cannot map dataset index %d to a scenario' % sample_index)


def empty_result_stat():
    return {'tp': [], 'fp': [], 'gt': 0, 'score': []}


def update_result_stat(stats, name, pred_boxes, pred_scores, gt_boxes,
                       iou_thresh):
    if name not in stats:
        stats[name] = empty_result_stat()
    wrapper = {iou_thresh: stats[name]}
    eval_utils.caluclate_tp_fp(
        pred_boxes, pred_scores, gt_boxes, wrapper, iou_thresh)


def calculate_ap_safe(stat, iou_thresh):
    if stat['gt'] <= 0:
        return 0.0
    wrapper = {iou_thresh: copy.deepcopy(stat)}
    ap, _, _ = eval_utils.calculate_ap(wrapper, iou_thresh, True)
    return float(ap)


def box_dimensions(center, order):
    if order == 'hwl':
        height, width, length = center[3], center[4], center[5]
    elif order in ('lhw', 'lwh'):
        length, height, width = center[3], center[4], center[5]
    else:
        raise ValueError('Unsupported box order: %s' % order)
    return float(length), float(width), float(height)


def oriented_inside(x, y, center, order):
    length, width, _ = box_dimensions(center, order)
    dx = x - float(center[0])
    dy = y - float(center[1])
    yaw = float(center[6])
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    local_x = cosine * dx + sine * dy
    local_y = -sine * dx + cosine * dy
    return (np.abs(local_x) <= length * 0.5) & \
        (np.abs(local_y) <= width * 0.5)


def map_box_stats(prob_map, center, lidar_range, order):
    """Sample center/mean/max confidence inside an oriented GT footprint."""
    height, width = prob_map.shape
    x_min, y_min, _, x_max, y_max, _ = [float(x) for x in lidar_range]
    x_center = float(center[0])
    y_center = float(center[1])
    ix = int(math.floor((x_center - x_min) / (x_max - x_min) * width))
    iy = int(math.floor((y_center - y_min) / (y_max - y_min) * height))
    center_value = 0.0
    if 0 <= ix < width and 0 <= iy < height:
        center_value = float(prob_map[iy, ix])

    length, box_width, _ = box_dimensions(center, order)
    radius = 0.5 * math.sqrt(length ** 2 + box_width ** 2)
    ix0 = max(0, int(math.floor((x_center - radius - x_min) /
                                (x_max - x_min) * width)))
    ix1 = min(width, int(math.ceil((x_center + radius - x_min) /
                                   (x_max - x_min) * width)) + 1)
    iy0 = max(0, int(math.floor((y_center - radius - y_min) /
                                (y_max - y_min) * height)))
    iy1 = min(height, int(math.ceil((y_center + radius - y_min) /
                                    (y_max - y_min) * height)) + 1)
    if ix0 >= ix1 or iy0 >= iy1:
        return {'center': center_value, 'mean': 0.0, 'max': 0.0}

    xs = x_min + (np.arange(ix0, ix1) + 0.5) * \
        (x_max - x_min) / width
    ys = y_min + (np.arange(iy0, iy1) + 0.5) * \
        (y_max - y_min) / height
    grid_x, grid_y = np.meshgrid(xs, ys)
    inside = oriented_inside(grid_x, grid_y, center, order)
    values = prob_map[iy0:iy1, ix0:ix1][inside]
    if values.size == 0:
        return {'center': center_value, 'mean': center_value,
                'max': center_value}
    return {'center': center_value, 'mean': float(values.mean()),
            'max': float(values.max())}


def voxel_density_stats(voxel_coords, voxel_num_points, agent_index, center,
                        voxel_size, lidar_range, order):
    agent = voxel_coords[:, 0] == int(agent_index)
    coords = voxel_coords[agent]
    counts = voxel_num_points[agent]
    length, width, _ = box_dimensions(center, order)
    area = max(length * width, 1.0e-6)
    if coords.shape[0] == 0:
        return {'points': 0, 'voxels': 0, 'points_per_m2': 0.0,
                'voxels_per_m2': 0.0}
    x = float(lidar_range[0]) + (coords[:, 3] + 0.5) * float(voxel_size[0])
    y = float(lidar_range[1]) + (coords[:, 2] + 0.5) * float(voxel_size[1])
    inside = oriented_inside(x, y, center, order)
    point_count = int(counts[inside].sum())
    voxel_count = int(inside.sum())
    return {
        'points': point_count,
        'voxels': voxel_count,
        'points_per_m2': float(point_count / area),
        'voxels_per_m2': float(voxel_count / area)
    }


def aligned_gt(ego_dict, order):
    centers = ego_dict['object_bbx_center'][0]
    valid = ego_dict['object_bbx_mask'][0].bool()
    centers = centers[valid]
    object_ids = list(ego_dict.get('object_ids', []))[:centers.shape[0]]
    corners = box_utils.boxes_to_corners_3d(centers, order=order)
    in_range = box_utils.get_mask_for_boxes_within_range_torch(corners)
    keep = torch.nonzero(in_range, as_tuple=False).reshape(-1).tolist()
    centers = centers[in_range]
    corners = corners[in_range]
    ids = [str(object_ids[i]) if i < len(object_ids) else 'gt_%d' % i
           for i in keep]
    return centers, corners, ids


def object_states(gt_boxes, pred_boxes, pred_scores, iou_thresh):
    gt_count = int(gt_boxes.shape[0])
    empty = [{'is_recalled': False, 'iou_with_pred': 0.0,
              'classification_score': 0.0, 'matched_pred_index': -1}
             for _ in range(gt_count)]
    if pred_boxes is None or pred_scores is None or pred_boxes.shape[0] == 0:
        return empty

    gt_np = common_utils.torch_tensor_to_numpy(gt_boxes)
    pred_np = common_utils.torch_tensor_to_numpy(pred_boxes)
    score_np = common_utils.torch_tensor_to_numpy(pred_scores).reshape(-1)
    gt_polygons = list(common_utils.convert_format(gt_np))
    pred_polygons = list(common_utils.convert_format(pred_np))
    iou_matrix = np.zeros((gt_count, len(pred_polygons)), dtype=np.float32)
    for pred_index, polygon in enumerate(pred_polygons):
        iou_matrix[:, pred_index] = common_utils.compute_iou(
            polygon, gt_polygons)

    # OpenCOOD-style one-to-one recall assignment in descending score order.
    unmatched = set(range(gt_count))
    assigned = {}
    for pred_index in np.argsort(-score_np):
        if not unmatched:
            break
        candidates = np.array(sorted(unmatched), dtype=np.int64)
        local = iou_matrix[candidates, pred_index]
        best_local = int(np.argmax(local))
        if float(local[best_local]) >= iou_thresh:
            gt_index = int(candidates[best_local])
            assigned[gt_index] = int(pred_index)
            unmatched.remove(gt_index)

    states = []
    for gt_index in range(gt_count):
        best_pred = int(np.argmax(iou_matrix[gt_index]))
        best_iou = float(iou_matrix[gt_index, best_pred])
        best_score = float(score_np[best_pred])
        matched = assigned.get(gt_index, -1)
        states.append({
            'is_recalled': matched >= 0,
            'iou_with_pred': best_iou,
            'classification_score': best_score,
            'matched_pred_index': matched
        })
    return states


def run_coalition(model, encoded, batch_data, dataset, agent_mask, gt_centers,
                  gt_boxes, hypes, spatial_agent_weights=None):
    output = model.forward_from_encoded(
        encoded, agent_mask=agent_mask, return_aux=False,
        spatial_agent_weights=spatial_agent_weights)
    output_dict = OrderedDict([('ego', output)])
    pred_boxes, pred_scores = dataset.post_processor.post_process(
        batch_data, output_dict)
    states = object_states(
        gt_boxes, pred_boxes, pred_scores,
        float(hypes['_oracle_iou_thresh']))

    fused_prob = output['psm'].sigmoid().amax(dim=1)[0].detach().cpu().numpy()
    lidar_range = hypes['preprocess']['cav_lidar_range']
    order = hypes['postprocess']['order']
    for state, center in zip(states, gt_centers.detach().cpu().numpy()):
        map_stats = map_box_stats(fused_prob, center, lidar_range, order)
        probability = max(map_stats['max'], 1.0e-7)
        state['target_confidence'] = map_stats['max']
        state['target_level_loss'] = float(-math.log(probability))
    return {
        'output': output,
        'pred_boxes': pred_boxes,
        'pred_scores': pred_scores,
        'states': states
    }


def error_pattern(baseline, candidate, iou_thresh):
    if baseline['is_recalled'] and not candidate['is_recalled']:
        if candidate['iou_with_pred'] <= 1.0e-8:
            return 'LCME'
        if candidate['iou_with_pred'] < iou_thresh:
            return 'LCLE'
        return 'MISLEADING_ASSIGNMENT'
    if not baseline['is_recalled'] and candidate['is_recalled']:
        if baseline['iou_with_pred'] <= 1.0e-8:
            return 'CORRECTED_MISSING'
        return 'CORRECTED_LOCALIZATION'
    return 'STABLE'


def flatten_state(row, prefix, state):
    row[prefix + '_is_recalled'] = int(state['is_recalled'])
    row[prefix + '_iou_with_pred'] = state['iou_with_pred']
    row[prefix + '_classification_score'] = state['classification_score']
    row[prefix + '_target_confidence'] = state['target_confidence']
    row[prefix + '_target_level_loss'] = state['target_level_loss']


def build_event(frame_index, dataset_index, neighbor_index, neighbor_id,
                gt_index, gt_id, center, states, ego_density,
                neighbor_density, ego_confidence, neighbor_confidence,
                iou_thresh):
    center = np.asarray(center)
    length, width, height = box_dimensions(center, 'hwl')
    ego = states['ego']
    all_state = states['all']
    plus = states['plus']
    minus = states['minus']

    row = OrderedDict()
    row['schema_version'] = SCHEMA_VERSION
    row['frame_index'] = frame_index
    row['dataset_index'] = dataset_index
    row['cav_count'] = states['cav_count']
    row['neighbor_index'] = neighbor_index
    row['neighbor_id'] = neighbor_id
    row['gt_index'] = gt_index
    row['gt_id'] = gt_id
    row['gt_x'] = float(center[0])
    row['gt_y'] = float(center[1])
    row['gt_z'] = float(center[2])
    row['gt_length'] = length
    row['gt_width'] = width
    row['gt_height'] = height
    row['gt_yaw'] = float(center[6])
    row['gt_distance'] = float(math.hypot(center[0], center[1]))

    row['point_density'] = neighbor_density['points_per_m2']
    row['density_neighbor_points'] = neighbor_density['points']
    row['density_neighbor_voxels'] = neighbor_density['voxels']
    row['density_neighbor_points_per_m2'] = neighbor_density['points_per_m2']
    row['density_neighbor_voxels_per_m2'] = neighbor_density['voxels_per_m2']
    row['density_ego_points_per_m2'] = ego_density['points_per_m2']
    row['where2comm_confidence'] = neighbor_confidence['center']
    row['confidence_neighbor_center'] = neighbor_confidence['center']
    row['confidence_neighbor_box_mean'] = neighbor_confidence['mean']
    row['confidence_neighbor_box_max'] = neighbor_confidence['max']
    row['confidence_ego_center'] = ego_confidence['center']
    row['confidence_ego_box_mean'] = ego_confidence['mean']

    flatten_state(row, 'ego', ego)
    flatten_state(row, 'all', all_state)
    flatten_state(row, 'ego_plus_i', plus)
    flatten_state(row, 'all_minus_i', minus)

    row['addition_recall_delta'] = \
        int(plus['is_recalled']) - int(ego['is_recalled'])
    row['addition_iou_change'] = \
        plus['iou_with_pred'] - ego['iou_with_pred']
    row['addition_score_change'] = \
        plus['classification_score'] - ego['classification_score']
    row['addition_loss_improvement'] = \
        ego['target_level_loss'] - plus['target_level_loss']
    row['addition_soft_utility'] = \
        plus['iou_with_pred'] * plus['classification_score'] - \
        ego['iou_with_pred'] * ego['classification_score']

    # all - all_minus_i is the neighbor's conditional marginal contribution.
    row['coalition_recall_delta'] = \
        int(all_state['is_recalled']) - int(minus['is_recalled'])
    row['coalition_iou_change'] = \
        all_state['iou_with_pred'] - minus['iou_with_pred']
    row['coalition_score_change'] = \
        all_state['classification_score'] - minus['classification_score']
    row['coalition_loss_improvement'] = \
        minus['target_level_loss'] - all_state['target_level_loss']
    row['coalition_soft_utility'] = \
        all_state['iou_with_pred'] * all_state['classification_score'] - \
        minus['iou_with_pred'] * minus['classification_score']
    # A target has no mathematically valid AP. This is an explicitly named
    # target-level proxy; actual paired AP gains are written to summary.json.
    row['target_ap_gain_proxy'] = row['coalition_recall_delta']
    row['addition_error_pattern'] = error_pattern(ego, plus, iou_thresh)
    row['coalition_error_pattern'] = error_pattern(minus, all_state, iou_thresh)
    row['coalition_hard_harmful'] = int(row['coalition_recall_delta'] < 0)
    row['coalition_soft_harmful'] = int(row['coalition_soft_utility'] < 0.0)
    return row


def correlation(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size < 3 or np.std(x) < 1.0e-12 or np.std(y) < 1.0e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def register_candidate(candidates, name, mask, result):
    """Register each distinct active-agent subset once."""
    active = tuple(torch.nonzero(mask, as_tuple=False).reshape(-1).tolist())
    signature = ','.join(str(x) for x in active)
    if signature not in candidates:
        candidates[signature] = {
            'name': name,
            'active_agents': list(active),
            'result': result
        }


def candidate_frame_metrics(candidate, gt_boxes, iou_thresh):
    """GT-aware metrics used only for the offline Oracle upper-bound study."""
    result = candidate['result']
    stat = empty_result_stat()
    eval_utils.caluclate_tp_fp(
        result['pred_boxes'], result['pred_scores'], gt_boxes,
        {iou_thresh: stat}, iou_thresh)
    states = result['states']
    return {
        'frame_ap': calculate_ap_safe(stat, iou_thresh),
        'recalled_targets': int(sum(x['is_recalled'] for x in states)),
        'soft_target_quality': float(sum(
            x['iou_with_pred'] * x['classification_score'] for x in states)),
        'target_loss': float(sum(x['target_level_loss'] for x in states)),
        'true_positives': int(sum(stat['tp'])),
        'false_positives': int(sum(stat['fp'])),
        'prediction_count': int(len(stat['score'])),
        'active_agent_count': int(len(candidate['active_agents']))
    }


def choose_oracle_candidate(candidates, selector):
    """Choose a frame coalition with GT; this is not an inference policy."""
    def key(item):
        metrics = item['metrics']
        efficiency = -metrics['active_agent_count']
        false_positive = -metrics['false_positives']
        if selector == 'frame_ap':
            return (metrics['frame_ap'], metrics['recalled_targets'],
                    metrics['soft_target_quality'], false_positive, efficiency)
        if selector == 'recall':
            return (metrics['recalled_targets'], false_positive,
                    metrics['soft_target_quality'], metrics['frame_ap'],
                    efficiency)
        if selector == 'soft_quality':
            return (metrics['soft_target_quality'],
                    metrics['recalled_targets'], metrics['frame_ap'],
                    false_positive, efficiency)
        raise ValueError('Unknown Oracle selector: %s' % selector)
    return max(candidates.values(), key=key)


def tensor_cosine(first, second):
    first = first.reshape(-1).float()
    second = second.reshape(-1).float()
    denominator = first.norm() * second.norm()
    if float(denominator.item()) < 1.0e-12:
        return 0.0
    return float(torch.dot(first, second).div(denominator).item())


def binary_overlap(first, second, threshold):
    first_mask = first >= threshold
    second_mask = second >= threshold
    union = torch.logical_or(first_mask, second_mask).sum()
    if int(union.item()) == 0:
        return 1.0
    intersection = torch.logical_and(first_mask, second_mask).sum()
    return float(intersection.float().div(union.float()).item())


def confidence_statistics(confidence, communication_threshold):
    flat = confidence.reshape(-1).float()
    clipped = flat.clamp(1.0e-7, 1.0 - 1.0e-7)
    entropy = -(clipped * clipped.log() +
                (1.0 - clipped) * (1.0 - clipped).log()).mean()
    thresholds = sorted(set([
        float(communication_threshold), 0.05, 0.1, 0.2, 0.5]))
    output = {
        'mean': float(flat.mean().item()),
        'std': float(flat.std(unbiased=False).item()),
        'max': float(flat.max().item()),
        'q90': float(torch.quantile(flat, 0.90).item()),
        'q99': float(torch.quantile(flat, 0.99).item()),
        'entropy_mean': float(entropy.item())
    }
    for threshold in thresholds:
        key = ('active_fraction_%.3f' % threshold).replace('.', 'p')
        output[key] = float((flat >= threshold).float().mean().item())
    return output


def build_deployable_features(encoded, cav_count, communication_threshold):
    """Build frame/agent features available without annotations at inference."""
    confidence = encoded['psm_single'][:cav_count].sigmoid().amax(dim=1)
    spatial = encoded['spatial_features_2d'][:cav_count]
    channel_signature = spatial.float().mean(dim=(2, 3))
    voxel_coords = encoded['voxel_coords']
    voxel_counts = encoded['voxel_num_points']

    agents = []
    for agent_index in range(cav_count):
        occupied = voxel_coords[:, 0] == agent_index
        agent_counts = voxel_counts[occupied].float()
        point_count = float(agent_counts.sum().item()) \
            if agent_counts.numel() else 0.0
        voxel_count = int(agent_counts.numel())
        agents.append({
            'agent_index': agent_index,
            'confidence': confidence_statistics(
                confidence[agent_index], communication_threshold),
            'point_count': point_count,
            'voxel_count': voxel_count,
            'points_per_voxel': point_count / max(voxel_count, 1),
            'feature_channel_norm': float(
                channel_signature[agent_index].norm().item())
        })

    pairwise = []
    for first_index in range(cav_count):
        for second_index in range(first_index + 1, cav_count):
            first_conf = confidence[first_index]
            second_conf = confidence[second_index]
            pairwise.append({
                'first_agent': first_index,
                'second_agent': second_index,
                'confidence_cosine': tensor_cosine(first_conf, second_conf),
                'confidence_l1_mean': float(
                    (first_conf - second_conf).abs().mean().item()),
                'first_novelty_over_second': float(
                    torch.relu(first_conf - second_conf).mean().item()),
                'second_novelty_over_first': float(
                    torch.relu(second_conf - first_conf).mean().item()),
                'high_conf_iou_0p05': binary_overlap(
                    first_conf, second_conf, 0.05),
                'high_conf_iou_0p10': binary_overlap(
                    first_conf, second_conf, 0.10),
                'high_conf_iou_0p20': binary_overlap(
                    first_conf, second_conf, 0.20),
                'feature_channel_cosine': tensor_cosine(
                    channel_signature[first_index],
                    channel_signature[second_index])
            })
    return {
        'uses_ground_truth': False,
        'agents': agents,
        'pairwise': pairwise
    }


def build_point_density_map(encoded, cav_count, height, width,
                            lidar_range, voxel_size):
    coords = encoded['voxel_coords']
    counts = encoded['voxel_num_points'].float()
    grid_width = int(round(
        (float(lidar_range[3]) - float(lidar_range[0])) /
        float(voxel_size[0])))
    grid_height = int(round(
        (float(lidar_range[4]) - float(lidar_range[1])) /
        float(voxel_size[1])))
    agent = coords[:, 0].long()
    y_index = torch.clamp(
        torch.div(coords[:, 2].long() * height, grid_height,
                  rounding_mode='floor'), 0, height - 1)
    x_index = torch.clamp(
        torch.div(coords[:, 3].long() * width, grid_width,
                  rounding_mode='floor'), 0, width - 1)
    flat_index = agent * (height * width) + y_index * width + x_index
    density = torch.zeros(
        cav_count * height * width, device=counts.device,
        dtype=torch.float32)
    density.scatter_add_(0, flat_index, counts)
    density = density.view(cav_count, height, width)
    # Log normalization keeps the fixed input range stable across weather.
    return torch.log1p(density).div(math.log(33.0)).clamp(0.0, 1.0)


def oriented_box_map_mask(center, height, width, lidar_range, order):
    x_min, y_min, _, x_max, y_max, _ = [float(x) for x in lidar_range]
    length, box_width, _ = box_dimensions(center, order)
    radius = 0.5 * math.sqrt(length ** 2 + box_width ** 2)
    x0 = max(0, int(math.floor(
        (float(center[0]) - radius - x_min) / (x_max - x_min) * width)))
    x1 = min(width, int(math.ceil(
        (float(center[0]) + radius - x_min) / (x_max - x_min) * width)) + 1)
    y0 = max(0, int(math.floor(
        (float(center[1]) - radius - y_min) / (y_max - y_min) * height)))
    y1 = min(height, int(math.ceil(
        (float(center[1]) + radius - y_min) / (y_max - y_min) * height)) + 1)
    if x0 >= x1 or y0 >= y1:
        return None
    xs = x_min + (np.arange(x0, x1) + 0.5) * (x_max - x_min) / width
    ys = y_min + (np.arange(y0, y1) + 0.5) * (y_max - y_min) / height
    grid_x, grid_y = np.meshgrid(xs, ys)
    inside = oriented_inside(grid_x, grid_y, center, order)
    return y0, y1, x0, x1, inside


def save_spatial_cache(cache_dir, local_frame, dataset_index,
                       scenario_index, domain_name, encoded, cav_count,
                       frame_events, lidar_range, voxel_size, order):
    os.makedirs(cache_dir, exist_ok=True)
    confidence = encoded['psm_single'][:cav_count].sigmoid().amax(dim=1)
    height, width = confidence.shape[-2:]
    density = build_point_density_map(
        encoded, cav_count, height, width, lidar_range, voxel_size)
    neighbor_count = cav_count - 1
    utility_sum = np.zeros((neighbor_count, height, width), dtype=np.float32)
    utility_count = np.zeros_like(utility_sum, dtype=np.float32)
    for event in frame_events:
        neighbor_slot = int(event['neighbor_index']) - 1
        center = np.asarray([
            event['gt_x'], event['gt_y'], event['gt_z'],
            event['gt_height'], event['gt_width'], event['gt_length'],
            event['gt_yaw']], dtype=np.float32)
        region = oriented_box_map_mask(
            center, height, width, lidar_range, order)
        if region is None:
            continue
        y0, y1, x0, x1, inside = region
        patch_sum = utility_sum[neighbor_slot, y0:y1, x0:x1]
        patch_count = utility_count[neighbor_slot, y0:y1, x0:x1]
        patch_sum[inside] += float(event['coalition_soft_utility'])
        patch_count[inside] += 1.0
    valid = utility_count > 0.0
    utility = np.zeros_like(utility_sum)
    utility[valid] = utility_sum[valid] / utility_count[valid]
    path = os.path.join(
        cache_dir, 'frame_%06d_dataset_%06d.npz' %
        (local_frame, dataset_index))
    np.savez_compressed(
        path,
        confidence=confidence.detach().cpu().numpy().astype(np.float16),
        density=density.detach().cpu().numpy().astype(np.float16),
        utility=utility.astype(np.float16),
        valid=valid.astype(np.uint8),
        cav_count=np.asarray(cav_count, dtype=np.int16),
        dataset_index=np.asarray(dataset_index, dtype=np.int32),
        scenario_index=np.asarray(scenario_index, dtype=np.int16),
        domain_name=np.asarray(domain_name))
    return path


def main():
    opt = parse_args()
    seed_everything(opt.seed)
    if not 0.0 < opt.iou_thresh <= 1.0:
        raise ValueError('--iou_thresh must be in (0, 1]')
    hypes, config_path = configure_hypes(opt)
    hypes['_oracle_iou_thresh'] = float(opt.iou_thresh)
    score_thresh = float(
        hypes['postprocess']['target_args']['score_threshold'])
    output_dir = opt.output_dir or os.path.join(
        opt.model_dir, 'oracle_eval_locked')
    os.makedirs(output_dir, exist_ok=True)

    print('Building locked evaluation dataset: %s' % hypes['validate_dir'])
    dataset = build_dataset(hypes, visualize=False, train=False)
    dataset_max_cav = int(getattr(dataset, 'max_cav', opt.max_cav))
    print('Oracle dataset max_cav: %d' % dataset_max_cav)
    if dataset_max_cav < 2:
        raise RuntimeError(
            'Oracle evaluation requires at least two CAVs; check YAML/max_cav')
    indices = select_indices(
        len(dataset), opt.start_index, opt.max_frames, opt.frame_stride)
    loader_generator = torch.Generator()
    loader_generator.manual_seed(opt.seed)
    loader = DataLoader(
        Subset(dataset, indices), batch_size=1, shuffle=False,
        num_workers=opt.num_workers, collate_fn=dataset.collate_batch_test,
        pin_memory=False, drop_last=False, worker_init_fn=seed_worker,
        generator=loader_generator)

    print('Creating PointPillar-Where2comm model')
    model = train_utils.create_model(hypes)
    if not hasattr(model, 'encode_features') or \
            not hasattr(model, 'forward_from_encoded'):
        raise TypeError(
            'oracle_eval requires PointPillarWhere2comm with cached fusion API')
    model = load_model(opt.model_dir, model, opt.eval_epoch)
    device = resolve_device(opt.device)
    if device.type == 'cuda' and device.index is not None:
        torch.cuda.set_device(device)
    model.to(device).eval()
    selector_bundle = load_selector(
        opt.selector_checkpoint, device, opt.selector_safety_margin)
    if selector_bundle is not None:
        print('Loaded coalition selector: %s' % opt.selector_checkpoint)
    spatial_selector_bundle = load_spatial_selector(
        opt.spatial_selector_checkpoint, device)
    if spatial_selector_bundle is not None:
        print('Loaded spatial utility selector: %s' %
              opt.spatial_selector_checkpoint)

    order = hypes['postprocess']['order']
    if order != 'hwl':
        raise ValueError('oracle event schema currently expects hwl box order')
    voxel_size = hypes['preprocess']['args']['voxel_size']
    lidar_range = hypes['preprocess']['cav_lidar_range']
    communication_threshold = float(
        hypes['model']['args']['where2comm_fusion']['communication'].get(
            'threshold', 0.0))

    result_stats = {}
    pass_frame_counts = defaultdict(int)
    events = []
    agent_frame_utilities = []
    oracle_selection_records = []
    oracle_selectors = ('frame_ap', 'recall', 'soft_quality')
    oracle_selection_counts = {
        selector: defaultdict(int) for selector in oracle_selectors
    }
    oracle_active_agent_sums = defaultdict(int)
    learned_selection_counts = defaultdict(int)
    learned_active_agent_sum = 0
    skipped_no_neighbor = 0
    skipped_empty_gt = 0
    total_targets = 0
    cache_equivalence_checked = False
    spatial_cache_files = 0
    spatial_gate_diagnostics = []

    for local_frame, batch_data in tqdm(
            enumerate(loader), total=len(loader), desc='Oracle frames'):
        dataset_index = indices[local_frame]
        scenario_index = scenario_index_for_sample(dataset, dataset_index)
        batch_data['ego'].pop('weather_augmentation_stats', None)
        batch_data = train_utils.to_device(batch_data, device)
        weather_processed = batch_data['ego'].pop(
            'processed_lidar_weather', None)
        if opt.input_branch == 'weather_augmented':
            if weather_processed is None:
                raise KeyError(
                    'weather_augmented branch missing from validation batch')
            batch_data['ego']['processed_lidar'] = weather_processed

        ego_dict = batch_data['ego']
        cav_count = int(ego_dict['record_len'][0].item())
        if cav_count <= 1:
            skipped_no_neighbor += 1
            continue

        gt_centers, gt_boxes, gt_ids = aligned_gt(ego_dict, order)
        if gt_boxes.shape[0] == 0:
            skipped_empty_gt += 1
            continue
        total_targets += int(gt_boxes.shape[0])

        with torch.no_grad():
            encoded = model.encode_features(ego_dict)
            all_mask = torch.ones(cav_count, dtype=torch.bool, device=device)
            ego_mask = torch.zeros(cav_count, dtype=torch.bool, device=device)
            ego_mask[0] = True

            pass_ego = run_coalition(
                model, encoded, batch_data, dataset, ego_mask,
                gt_centers, gt_boxes, hypes)
            pass_all = run_coalition(
                model, encoded, batch_data, dataset, all_mask,
                gt_centers, gt_boxes, hypes)
            pass_spatial = None
            if spatial_selector_bundle is not None:
                spatial_weights, gate_diagnostics = \
                    predict_spatial_agent_weights(
                        spatial_selector_bundle, encoded, cav_count,
                        lidar_range, voxel_size,
                        gate_mode=opt.spatial_gate_mode,
                        utility_threshold=opt.spatial_utility_threshold,
                        harm_probability_threshold=(
                            opt.spatial_harm_probability_threshold),
                        gate_floor=opt.spatial_gate_floor)
                pass_spatial = run_coalition(
                    model, encoded, batch_data, dataset, all_mask,
                    gt_centers, gt_boxes, hypes,
                    spatial_agent_weights=spatial_weights)
                update_result_stat(
                    result_stats, 'spatial_selector',
                    pass_spatial['pred_boxes'], pass_spatial['pred_scores'],
                    gt_boxes, opt.iou_thresh)
                pass_frame_counts['spatial_selector'] += 1
                spatial_gate_diagnostics.append(gate_diagnostics)
            coalition_candidates = OrderedDict()
            register_candidate(
                coalition_candidates, 'ego', ego_mask, pass_ego)
            register_candidate(
                coalition_candidates, 'all', all_mask, pass_all)
            if not cache_equivalence_checked:
                reference_output = model(ego_dict)
                psm_error = float((
                    reference_output['psm'] - pass_all['output']['psm']
                ).abs().max().item())
                rm_error = float((
                    reference_output['rm'] - pass_all['output']['rm']
                ).abs().max().item())
                if psm_error > 1.0e-5 or rm_error > 1.0e-5:
                    raise AssertionError(
                        'cached all-agent pass differs from normal forward: '
                        'psm %.8f rm %.8f' % (psm_error, rm_error))
                print('\nCached fusion equivalence passed: psm %.3e rm %.3e' %
                      (psm_error, rm_error))
                cache_equivalence_checked = True
            update_result_stat(
                result_stats, 'ego', pass_ego['pred_boxes'],
                pass_ego['pred_scores'], gt_boxes, opt.iou_thresh)
            update_result_stat(
                result_stats, 'all', pass_all['pred_boxes'],
                pass_all['pred_scores'], gt_boxes, opt.iou_thresh)
            pass_frame_counts['ego'] += 1
            pass_frame_counts['all'] += 1

            single_prob = encoded['psm_single'].sigmoid().amax(
                dim=1).detach().cpu().numpy()
            voxel_coords = encoded['voxel_coords'].detach().cpu().numpy()
            voxel_counts = encoded['voxel_num_points'].detach().cpu().numpy()
            centers_np = gt_centers.detach().cpu().numpy()
            cav_ids = list(ego_dict.get('cav_ids', []))
            frame_neighbor_labels = []
            frame_events = []

            for neighbor_index in range(1, cav_count):
                plus_mask = ego_mask.clone()
                plus_mask[neighbor_index] = True
                minus_mask = all_mask.clone()
                minus_mask[neighbor_index] = False
                pass_plus = run_coalition(
                    model, encoded, batch_data, dataset, plus_mask,
                    gt_centers, gt_boxes, hypes)
                pass_minus = run_coalition(
                    model, encoded, batch_data, dataset, minus_mask,
                    gt_centers, gt_boxes, hypes)
                register_candidate(
                    coalition_candidates, 'ego_plus_%d' % neighbor_index,
                    plus_mask, pass_plus)
                register_candidate(
                    coalition_candidates, 'all_minus_%d' % neighbor_index,
                    minus_mask, pass_minus)

                slot = str(neighbor_index)
                paired = {
                    'ego_paired_' + slot: pass_ego,
                    'ego_plus_' + slot: pass_plus,
                    'all_paired_' + slot: pass_all,
                    'all_minus_' + slot: pass_minus
                }
                for name, result in paired.items():
                    update_result_stat(
                        result_stats, name, result['pred_boxes'],
                        result['pred_scores'], gt_boxes, opt.iou_thresh)
                    pass_frame_counts[name] += 1

                neighbor_id = cav_ids[neighbor_index] \
                    if neighbor_index < len(cav_ids) else str(neighbor_index)
                frame_recall_utility = 0
                frame_soft_utility = 0.0
                for gt_index, (gt_id, center) in enumerate(
                        zip(gt_ids, centers_np)):
                    ego_density = voxel_density_stats(
                        voxel_coords, voxel_counts, 0, center,
                        voxel_size, lidar_range, order)
                    neighbor_density = voxel_density_stats(
                        voxel_coords, voxel_counts, neighbor_index, center,
                        voxel_size, lidar_range, order)
                    ego_conf = map_box_stats(
                        single_prob[0], center, lidar_range, order)
                    neighbor_conf = map_box_stats(
                        single_prob[neighbor_index], center,
                        lidar_range, order)
                    tracked = {
                        'ego': pass_ego['states'][gt_index],
                        'all': pass_all['states'][gt_index],
                        'plus': pass_plus['states'][gt_index],
                        'minus': pass_minus['states'][gt_index],
                        'cav_count': cav_count
                    }
                    event = build_event(
                        local_frame, dataset_index, neighbor_index,
                        str(neighbor_id), gt_index, gt_id, center, tracked,
                        ego_density, neighbor_density, ego_conf,
                        neighbor_conf, opt.iou_thresh)
                    events.append(event)
                    frame_events.append(event)
                    frame_recall_utility += event['coalition_recall_delta']
                    frame_soft_utility += event['coalition_soft_utility']
                neighbor_label = {
                    'dataset_index': dataset_index,
                    'neighbor_index': neighbor_index,
                    'neighbor_id': str(neighbor_id),
                    'recall_utility': frame_recall_utility,
                    'soft_utility': frame_soft_utility,
                    'hard_harmful': int(frame_recall_utility < 0),
                    'soft_harmful': int(frame_soft_utility < 0.0)
                }
                agent_frame_utilities.append(neighbor_label)
                frame_neighbor_labels.append(dict(neighbor_label))

            if opt.spatial_cache_dir:
                save_spatial_cache(
                    opt.spatial_cache_dir, local_frame, dataset_index,
                    scenario_index, opt.domain_name or opt.input_branch,
                    encoded, cav_count, frame_events, lidar_range,
                    voxel_size, order)
                spatial_cache_files += 1

            candidate_records = []
            for candidate in coalition_candidates.values():
                candidate['metrics'] = candidate_frame_metrics(
                    candidate, gt_boxes, opt.iou_thresh)
                candidate_records.append({
                    'name': candidate['name'],
                    'active_agents': candidate['active_agents'],
                    'metrics': candidate['metrics']
                })
            selection_record = {
                'frame_index': local_frame,
                'dataset_index': dataset_index,
                'scenario_index': scenario_index,
                'domain_name': opt.domain_name or opt.input_branch,
                'cav_count': cav_count,
                'deployable_features': build_deployable_features(
                    encoded, cav_count, communication_threshold),
                'neighbor_labels': frame_neighbor_labels,
                'candidates': candidate_records,
                'selected': {}
            }
            for selector in oracle_selectors:
                selected = choose_oracle_candidate(
                    coalition_candidates, selector)
                result_name = 'oracle_' + selector
                update_result_stat(
                    result_stats, result_name,
                    selected['result']['pred_boxes'],
                    selected['result']['pred_scores'], gt_boxes,
                    opt.iou_thresh)
                pass_frame_counts[result_name] += 1
                oracle_selection_counts[selector][selected['name']] += 1
                oracle_active_agent_sums[selector] += len(
                    selected['active_agents'])
                selection_record['selected'][selector] = {
                    'name': selected['name'],
                    'active_agents': selected['active_agents'],
                    'metrics': selected['metrics']
                }
            if selector_bundle is not None:
                selected_index, learned_scores = predict_candidate(
                    selector_bundle, selection_record, device)
                selected_record = candidate_records[selected_index]
                selected_candidate = list(
                    coalition_candidates.values())[selected_index]
                update_result_stat(
                    result_stats, 'learned_selector',
                    selected_candidate['result']['pred_boxes'],
                    selected_candidate['result']['pred_scores'], gt_boxes,
                    opt.iou_thresh)
                pass_frame_counts['learned_selector'] += 1
                learned_selection_counts[selected_record['name']] += 1
                learned_active_agent_sum += len(
                    selected_record['active_agents'])
                selection_record['learned_selector'] = {
                    'name': selected_record['name'],
                    'active_agents': selected_record['active_agents'],
                    'strategy': selector_bundle['strategy'],
                    'safety_margin': selector_bundle['safety_margin'],
                    'scores': {
                        item['name']: learned_scores[index]
                        for index, item in enumerate(candidate_records)
                    }
                }
            oracle_selection_records.append(selection_record)

    if not events:
        raise RuntimeError(
            'Oracle produced zero events: skipped_no_neighbor=%d, '
            'skipped_empty_gt=%d. Refusing to write an empty success result.' %
            (skipped_no_neighbor, skipped_empty_gt))

    csv_path = os.path.join(output_dir, opt.output_prefix + '.csv')
    jsonl_path = os.path.join(output_dir, opt.output_prefix + '.jsonl')
    if events:
        fields = list(events[0].keys())
        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(events)
        with open(jsonl_path, 'w', encoding='utf-8') as stream:
            for event in events:
                stream.write(json.dumps(event, ensure_ascii=False) + '\n')
    else:
        open(csv_path, 'w', encoding='utf-8').close()
        open(jsonl_path, 'w', encoding='utf-8').close()
        fields = []

    ap_by_pass = {}
    for name, stat in sorted(result_stats.items()):
        ap_by_pass[name] = {
            'ap': calculate_ap_safe(stat, opt.iou_thresh),
            'frames': int(pass_frame_counts[name]),
            'gt': int(stat['gt'])
        }

    paired_ap_gains = {}
    neighbor_slots = sorted({
        int(name.rsplit('_', 1)[-1]) for name in result_stats
        if name.startswith('ego_plus_')
    })
    for slot in neighbor_slots:
        plus = ap_by_pass['ego_plus_%d' % slot]['ap']
        ego = ap_by_pass['ego_paired_%d' % slot]['ap']
        all_ap = ap_by_pass['all_paired_%d' % slot]['ap']
        minus = ap_by_pass['all_minus_%d' % slot]['ap']
        paired_ap_gains[str(slot)] = {
            'isolated_ap_gain': plus - ego,
            'coalition_marginal_ap_gain': all_ap - minus,
            'paired_frames': ap_by_pass['ego_plus_%d' % slot]['frames']
        }

    all_global_ap = ap_by_pass.get('all', {}).get('ap', 0.0)
    oracle_selection_summary = {}
    for selector in oracle_selectors:
        name = 'oracle_' + selector
        selector_ap = ap_by_pass.get(name, {}).get('ap', 0.0)
        selected_frames = int(pass_frame_counts.get(name, 0))
        oracle_selection_summary[selector] = {
            'ap': selector_ap,
            'ap_gain_vs_all': selector_ap - all_global_ap,
            'selected_frames': selected_frames,
            'mean_active_agents': (
                oracle_active_agent_sums[selector] / selected_frames
                if selected_frames else 0.0),
            'selection_counts': dict(oracle_selection_counts[selector])
        }
    learned_selection_summary = None
    if selector_bundle is not None:
        learned_frames = int(pass_frame_counts.get('learned_selector', 0))
        learned_ap = ap_by_pass.get(
            'learned_selector', {}).get('ap', 0.0)
        learned_selection_summary = {
            'checkpoint': opt.selector_checkpoint,
            'strategy': selector_bundle['strategy'],
            'safety_margin': selector_bundle['safety_margin'],
            'ap': learned_ap,
            'ap_gain_vs_all': learned_ap - all_global_ap,
            'selected_frames': learned_frames,
            'mean_active_agents': (
                learned_active_agent_sum / learned_frames
                if learned_frames else 0.0),
            'selection_counts': dict(learned_selection_counts),
            'oracle_frame_ap_headroom_recovery': (
                (learned_ap - all_global_ap) /
                (oracle_selection_summary['frame_ap']['ap'] - all_global_ap)
                if abs(oracle_selection_summary['frame_ap']['ap'] -
                       all_global_ap) > 1.0e-12 else 0.0)
        }
    spatial_selection_summary = None
    if spatial_selector_bundle is not None:
        spatial_ap = ap_by_pass.get(
            'spatial_selector', {}).get('ap', 0.0)
        spatial_selection_summary = {
            'checkpoint': opt.spatial_selector_checkpoint,
            'gate_mode': opt.spatial_gate_mode,
            'utility_threshold': (
                spatial_selector_bundle['utility_threshold']
                if opt.spatial_utility_threshold is None
                else opt.spatial_utility_threshold),
            'harm_probability_threshold':
                opt.spatial_harm_probability_threshold,
            'gate_floor': (
                spatial_selector_bundle['gate_floor']
                if opt.spatial_gate_floor is None
                else opt.spatial_gate_floor),
            'ap': spatial_ap,
            'ap_gain_vs_all': spatial_ap - all_global_ap,
            'evaluated_frames': int(
                pass_frame_counts.get('spatial_selector', 0)),
            'mean_neighbor_gate': float(np.mean([
                item['mean'] for item in spatial_gate_diagnostics]))
            if spatial_gate_diagnostics else 1.0,
            'mean_low_gate_fraction': float(np.mean([
                item['low_fraction'] for item in spatial_gate_diagnostics]))
            if spatial_gate_diagnostics else 0.0,
            'mean_support_fraction': float(np.mean([
                item['support_fraction']
                for item in spatial_gate_diagnostics]))
            if spatial_gate_diagnostics else 0.0,
            'mean_support_gate': float(np.mean([
                item['support_mean'] for item in spatial_gate_diagnostics]))
            if spatial_gate_diagnostics else 1.0,
            'mean_suppressed_fraction': float(np.mean([
                item['suppressed_fraction']
                for item in spatial_gate_diagnostics]))
            if spatial_gate_diagnostics else 0.0
        }

    densities = [x['point_density'] for x in events]
    confidences = [x['where2comm_confidence'] for x in events]
    utilities = [x['coalition_soft_utility'] for x in events]
    event_hard = sum(x['coalition_hard_harmful'] for x in events)
    event_soft = sum(x['coalition_soft_harmful'] for x in events)
    frame_hard = sum(x['hard_harmful'] for x in agent_frame_utilities)
    frame_soft = sum(x['soft_harmful'] for x in agent_frame_utilities)
    event_count = len(events)
    agent_frame_count = len(agent_frame_utilities)

    summary = {
        'schema_version': SCHEMA_VERSION,
        'config_path': config_path,
        'model_dir': opt.model_dir,
        'eval_epoch': opt.eval_epoch,
        'validate_dir': hypes['validate_dir'],
        'input_branch': opt.input_branch,
        'domain_name': opt.domain_name or opt.input_branch,
        'seed': opt.seed,
        'dataset_frames': len(dataset),
        'requested_frames': len(indices),
        'evaluated_frames': int(pass_frame_counts.get('all', 0)),
        'skipped_no_neighbor': skipped_no_neighbor,
        'skipped_empty_gt': skipped_empty_gt,
        'tracked_targets': total_targets,
        'interaction_events': event_count,
        'spatial_cache_files': spatial_cache_files,
        'iou_thresh': opt.iou_thresh,
        'score_thresh': score_thresh,
        'ap_by_pass': ap_by_pass,
        'global_all_vs_ego_ap_gain': (
            ap_by_pass.get('all', {}).get('ap', 0.0) -
            ap_by_pass.get('ego', {}).get('ap', 0.0)),
        'paired_ap_gains_by_neighbor_slot': paired_ap_gains,
        'oracle_frame_selection': oracle_selection_summary,
        'learned_selector': learned_selection_summary,
        'spatial_selector': spatial_selection_summary,
        'harmful_ratios': {
            'target_hard': event_hard / event_count if event_count else 0.0,
            'target_soft': event_soft / event_count if event_count else 0.0,
            'agent_frame_hard': frame_hard / agent_frame_count
            if agent_frame_count else 0.0,
            'agent_frame_soft': frame_soft / agent_frame_count
            if agent_frame_count else 0.0
        },
        'misalignment': {
            'pearson_density_vs_soft_utility': correlation(
                densities, utilities),
            'pearson_confidence_vs_soft_utility': correlation(
                confidences, utilities)
        },
        'event_fields': fields,
        'notes': {
            'target_ap_gain_proxy':
                'A per-target recall delta, not mathematical AP.',
            'coalition_marginal':
                'pass_all minus pass_all_minus_i; positive means neighbor helps.',
            'point_density':
                'Post-voxel retained point count per GT BEV square meter.',
            'where2comm_confidence':
                'Raw neighbor sigmoid classification confidence at GT center.',
            'oracle_frame_selection':
                'GT-aware selection among evaluated coalition candidates. '
                'It estimates achievable headroom but is not a deployable '
                'policy or a mathematical global-AP upper bound.',
            'deployable_features':
                'Agent confidence, voxel and pairwise redundancy features '
                'computed without ground truth for selector training.',
            'learned_selector':
                'Coalition selected without ground truth by the optional '
                'trained ranker checkpoint.',
            'spatial_selector':
                'Per-cell neighbor feature weighting predicted without GT.'
        }
    }
    summary_path = os.path.join(
        output_dir, opt.output_prefix + '_summary.json')
    agent_frame_path = os.path.join(
        output_dir, opt.output_prefix + '_agent_frame.jsonl')
    oracle_selection_path = os.path.join(
        output_dir, opt.output_prefix + '_oracle_selection.jsonl')
    with open(summary_path, 'w', encoding='utf-8') as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
    with open(agent_frame_path, 'w', encoding='utf-8') as stream:
        for item in agent_frame_utilities:
            stream.write(json.dumps(item, ensure_ascii=False) + '\n')
    with open(oracle_selection_path, 'w', encoding='utf-8') as stream:
        for item in oracle_selection_records:
            stream.write(json.dumps(item, ensure_ascii=False) + '\n')

    print('\nOracle evaluation finished')
    print('Events: %d' % event_count)
    print('CSV: %s' % csv_path)
    print('JSONL: %s' % jsonl_path)
    print('Summary: %s' % summary_path)
    print('Oracle selections: %s' % oracle_selection_path)
    if 'ego' in ap_by_pass and 'all' in ap_by_pass:
        print('AP@%.2f ego %.4f all %.4f gain %+.4f' %
              (opt.iou_thresh, ap_by_pass['ego']['ap'],
               ap_by_pass['all']['ap'],
               summary['global_all_vs_ego_ap_gain']))
    for selector, item in oracle_selection_summary.items():
        print('Oracle %-12s AP %.4f gain_vs_all %+.4f mean_agents %.2f' %
              (selector, item['ap'], item['ap_gain_vs_all'],
               item['mean_active_agents']))
    if learned_selection_summary is not None:
        item = learned_selection_summary
        print('Learned selector  AP %.4f gain_vs_all %+.4f mean_agents %.2f '
              'headroom %.3f' %
              (item['ap'], item['ap_gain_vs_all'],
               item['mean_active_agents'],
               item['oracle_frame_ap_headroom_recovery']))
    if spatial_selection_summary is not None:
        item = spatial_selection_summary
        print('Spatial selector  AP %.4f gain_vs_all %+.4f gate_mean %.3f '
              'support_gate %.3f suppressed %.4f' %
              (item['ap'], item['ap_gain_vs_all'],
               item['mean_neighbor_gate'],
               item['mean_support_gate'],
               item['mean_suppressed_fraction']))


if __name__ == '__main__':
    main()
