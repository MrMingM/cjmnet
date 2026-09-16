# -*- coding: utf-8 -*-
"""
Batch inference for Where2comm on OPV2V-w weather splits.

Example:
python opencood/tools/test_weather_where2comm.py ^
  --model_dir opencood/logs/point_pillar_where2comm_xxx ^
  --max_cav 5
"""

import argparse
import copy
import csv
import os
import sys
from collections import OrderedDict

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils
from opencood.tools.experiment_manifest import (
    checkpoint_path, git_revision, scenario_names, sha256_file,
    write_manifest)
from opencood.utils import eval_utils


DEFAULT_WEATHER_DIRS = [
    ('clean', '/data/scd/datasets/opv2v_official_data_dumping/'),
    ('fog', '/data/cjm/datasets/opv2v-w/fog/test'),
    ('rain', '/data/cjm/datasets/opv2v-w/rain/test'),
    ('snow', '/data/cjm/datasets/opv2v-w/snow/test'),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Batch test Where2comm on multiple weather validation sets.')
    parser.add_argument('--model_dir', required=True, type=str,
                        help='Directory containing config.yaml and checkpoint.')
    parser.add_argument('--hypes_yaml', default='', type=str,
                        help='Optional evaluation YAML instead of model_dir/config.yaml.')
    parser.add_argument('--weather', action='append', default=None,
                        help='Weather split in the form name=/path/to/validate_dir. '
                             'Use this argument multiple times. Defaults to the '
                             'current OPV2V/OPV2V-W clean/fog/rain/snow test paths.')
    parser.add_argument('--eval_epoch', default=0, type=int,
                        help='Checkpoint epoch to evaluate, e.g. 50 loads net_epoch50.pth. '
                             'Use 0 to load latest/max epoch.')
    parser.add_argument('--checkpoint_path', default='', type=str,
                        help='Optional explicit checkpoint, including net_stepN.pth. '
                             'Overrides --eval_epoch.')
    parser.add_argument('--output_csv', default='', type=str,
                        help='Where to save summary CSV. Defaults to model_dir/weather_eval.csv.')
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--start_index', default=0, type=int,
                        help='First dataset index included in evaluation.')
    parser.add_argument('--max_frames', default=0, type=int,
                        help='Maximum evaluated frames; zero uses all remaining.')
    parser.add_argument('--frame_stride', default=1, type=int,
                        help='Evaluate every Nth frame from start_index.')
    parser.add_argument('--max_cav', default=0, type=int,
                        help='Override train_params.max_cav and model.args.max_cav. '
                             'Use 0 to keep config.yaml.')
    parser.add_argument('--device', default='auto', type=str,
                        help='Device for inference: auto, cpu, cuda, cuda:0, cuda:1, etc.')
    parser.add_argument('--fusion_mode', choices=['intermediate', 'no'],
                        default='intermediate',
                        help='intermediate uses normal Where2comm. no keeps only ego input '
                             'for a strict no-fusion baseline under the same test script.')
    parser.add_argument('--input_branch',
                        choices=['clean', 'weather_augmented'],
                        default='clean',
                        help='Use clean processed_lidar or deterministic '
                             'pre-voxel augmented input.')
    parser.add_argument('--rain_rate', default=None, type=float,
                        help='Override fixed Physics-Rain rate in mm/h. Only '
                             'valid with --input_branch weather_augmented and '
                             'weather_augmentation.mode=physics_rain.')
    parser.add_argument('--weather_augmentation_seed', default=None, type=int,
                        help='Override deterministic online weather seed.')
    parser.add_argument(
        '--weather_augmentation_mode', default='', type=str,
        choices=['', 'v2x_dgw_awa', 'pre_voxel_matched_dropout',
                 'physics_rain', 'physics_fog', 'physics_snow',
                 'mixed_weather'],
        help='Optional evaluation override of weather augmentation mode.')
    parser.add_argument('--fog_lookup_dir', default='', type=str,
                        help='Physics-Fog integral lookup-table directory.')
    parser.add_argument('--fog_alpha', default=None, type=float,
                        help='Fixed Physics-Fog extinction coefficient.')
    parser.add_argument('--snowfall_rate', default=None, type=float,
                        help='Fixed Physics-Snow precipitation rate in mm/h.')
    parser.add_argument(
        '--snow_voxel_activation_threshold', default=None, type=float,
        help='Override the trained snow-denoiser probability threshold.')
    parser.add_argument(
        '--snow_voxel_gate_floor', default=None, type=float,
        help='Override the minimum retained snow-pillar feature weight.')
    parser.add_argument('--global_sort_detections', action='store_true')
    parser.add_argument('--use_uncertainty', choices=['keep', 'true', 'false'],
                        default='keep',
                        help='Override where2comm communication.use_uncertainty at test time.')
    parser.add_argument('--uncertainty_alpha', default=None, type=float,
                        help='Soft uncertainty penalty alpha. If set, enables uncertainty and uses '
                             'communication = confidence * (1 - alpha * entropy).')
    parser.add_argument('--comm_threshold', default=None, type=float,
                        help='Override where2comm communication threshold at test time.')
    parser.add_argument('--weather_reliability',
                        choices=['keep', 'off', 'density', 'density_isolation'],
                        default='keep',
                        help='Override non-learned weather reliability at test time.')
    parser.add_argument('--weather_apply_to',
                        choices=['keep', 'comm', 'feature', 'both'],
                        default='keep',
                        help='Apply weather reliability to communication, feature modulation, or both.')
    parser.add_argument('--weather_stat_path', default=None, type=str,
                        help='Clean statistics npz path for weather reliability.')
    parser.add_argument('--weather_debug_save', action='store_true',
                        help='Save weather reliability debug npz files.')
    parser.add_argument('--weather_alpha_density', default=None, type=float,
                        help='Override weather_reliability.alpha_density.')
    parser.add_argument('--weather_beta_isolation', default=None, type=float,
                        help='Override weather_reliability.beta_isolation.')
    parser.add_argument('--weather_isolation',
                        choices=['keep', 'true', 'false'],
                        default='keep',
                        help='Override weather_reliability.isolation_enable.')
    parser.add_argument('--weather_reliability_floor', default=None, type=float,
                        help='Soft gate floor. R_comm = floor + (1-floor) * R.')
    parser.add_argument('--weather_empty_cell_reliability', default=None, type=float,
                        help='Reliability assigned to empty BEV cells in R_density.')
    parser.add_argument('--weather_isolation_kernel_size', default=None, type=int,
                        help='Override weather_reliability.isolation_kernel_size.')
    parser.add_argument('--weather_isolation_neighbor_threshold', default=None, type=float,
                        help='Override weather_reliability.isolation_neighbor_threshold.')
    parser.add_argument('--weather_isolation_penalty_value', default=None, type=float,
                        help='Override weather_reliability.isolation_penalty_value.')
    parser.add_argument('--weather_isolation_min_range', default=None, type=float,
                        help='Override weather_reliability.isolation_min_range.')
    parser.add_argument('--weather_isolation_max_range', default=None, type=float,
                        help='Override weather_reliability.isolation_max_range.')
    return parser.parse_args()


def parse_weather_items(items):
    if not items:
        return [(name, resolve_validate_dir(path))
                for name, path in DEFAULT_WEATHER_DIRS]

    weather_dirs = []
    for item in items:
        if '=' not in item:
            raise ValueError('--weather must be formatted as name=/path/to/dir')
        name, path = item.split('=', 1)
        name = name.strip()
        path = path.strip().strip('"')
        if not name or not path:
            raise ValueError('--weather must contain non-empty name and path')
        weather_dirs.append((name, resolve_validate_dir(path)))
    return weather_dirs




def _is_opencood_scenario_root(path):
    if not os.path.isdir(path):
        return False
    scenario_names = sorted([
        x for x in os.listdir(path)
        if os.path.isdir(os.path.join(path, x))
    ])
    if not scenario_names:
        return False

    first_scenario = os.path.join(path, scenario_names[0])
    cav_names = sorted([
        x for x in os.listdir(first_scenario)
        if os.path.isdir(os.path.join(first_scenario, x))
    ])
    if not cav_names:
        return False

    try:
        int(cav_names[0])
        return True
    except ValueError:
        return False


def resolve_validate_dir(path):
    path = os.path.abspath(os.path.expanduser(path))
    if _is_opencood_scenario_root(path):
        return path

    for split_name in ['test_clean', 'test', 'validate', 'val']:
        split_path = os.path.join(path, split_name)
        if _is_opencood_scenario_root(split_path):
            print('Resolved validate_dir: %s -> %s' % (path, split_path))
            return split_path

    return path

def set_uncertainty_override(hypes, opt):
    pass

def set_weather_reliability_override(hypes, opt):
        weather_cfg['empty_cell_reliability'] = 1.0

    if opt.weather_stat_path is not None:
        weather_cfg['stat_path'] = opt.weather_stat_path
        weather_cfg['use_clean_stats'] = True
    if opt.weather_apply_to != 'keep':
        weather_cfg['apply_to'] = opt.weather_apply_to
    if opt.weather_debug_save:
        weather_cfg['debug_save'] = True
    if opt.weather_alpha_density is not None:
        weather_cfg['alpha_density'] = opt.weather_alpha_density
    if opt.weather_beta_isolation is not None:
        weather_cfg['beta_isolation'] = opt.weather_beta_isolation
    if opt.weather_isolation != 'keep':
        weather_cfg['isolation_enable'] = opt.weather_isolation == 'true'
    if opt.weather_reliability_floor is not None:
        weather_cfg['reliability_floor'] = opt.weather_reliability_floor
    if opt.weather_empty_cell_reliability is not None:
        weather_cfg['empty_cell_reliability'] = \
            opt.weather_empty_cell_reliability
    if opt.weather_isolation_kernel_size is not None:
        weather_cfg['isolation_kernel_size'] = opt.weather_isolation_kernel_size
    if opt.weather_isolation_neighbor_threshold is not None:
        weather_cfg['isolation_neighbor_threshold'] = \
            opt.weather_isolation_neighbor_threshold
    if opt.weather_isolation_penalty_value is not None:
        weather_cfg['isolation_penalty_value'] = \
            opt.weather_isolation_penalty_value
    if opt.weather_isolation_min_range is not None:
        weather_cfg['isolation_min_range'] = opt.weather_isolation_min_range
    if opt.weather_isolation_max_range is not None:
        weather_cfg['isolation_max_range'] = opt.weather_isolation_max_range

    if opt.weather_reliability != 'keep':
        print('Override weather_reliability to %s' % opt.weather_reliability)
        print('Weather reliability effective config: enable=%s, density=%s, '
              'isolation=%s, apply_to=%s, alpha=%.3f, beta=%.3f, floor=%.3f, empty=%.3f, '
              'iso_k=%d, iso_thr=%.3f, iso_penalty=%.3f, iso_range=[%.1f, %.1f]' %
              (weather_cfg.get('enable', False),
               weather_cfg.get('density_enable', False),
               weather_cfg.get('isolation_enable', False),
               weather_cfg.get('apply_to', 'comm'),
               float(weather_cfg.get('alpha_density', 0.0)),
               float(weather_cfg.get('beta_isolation', 0.0)),
               float(weather_cfg.get('reliability_floor', 0.0)),
               float(weather_cfg.get('empty_cell_reliability', 0.0)),
               int(weather_cfg.get('isolation_kernel_size', 3)),
               float(weather_cfg.get('isolation_neighbor_threshold', 2.0)),
               float(weather_cfg.get('isolation_penalty_value', 0.5)),
               float(weather_cfg.get('isolation_min_range', 0.0)),
               float(weather_cfg.get('isolation_max_range', 1.0e8))))


def keep_ego_only_for_no_fusion(ego_dict):
    """
    Keep only ego-agent tensors while preserving the original GT/postprocess data.

    The weather test dataloader uses batch size 1, so agent index 0 corresponds
    to ego after IntermediateFusionDataset collation.
    """
    record_len = ego_dict.get('record_len', None)
    if record_len is None:
        return
    if int(record_len.numel()) != 1:
        raise ValueError('--fusion_mode no currently expects test batch size 1.')

    processed = ego_dict['processed_lidar']
    voxel_coords = processed['voxel_coords']
    ego_mask = voxel_coords[:, 0] == 0
    processed['voxel_coords'] = voxel_coords[ego_mask]
    processed['voxel_features'] = processed['voxel_features'][ego_mask]
    processed['voxel_num_points'] = processed['voxel_num_points'][ego_mask]

    ego_dict['record_len'] = torch.ones_like(record_len)
    if 'pairwise_t_matrix' in ego_dict:
        ego_dict['pairwise_t_matrix'] = ego_dict['pairwise_t_matrix'][:, :1, :1]
    if 'prior_encoding' in ego_dict:
        ego_dict['prior_encoding'] = ego_dict['prior_encoding'][:, :1]
    if 'spatial_correction_matrix' in ego_dict:
        ego_dict['spatial_correction_matrix'] = \
            ego_dict['spatial_correction_matrix'][:, :1]


def set_max_cav_override(hypes, max_cav):
    if max_cav <= 0:
        return
    hypes.setdefault('train_params', {})['max_cav'] = max_cav
    if 'model' in hypes and 'args' in hypes['model']:
        hypes['model']['args']['max_cav'] = max_cav
    print('Override max_cav to %d' % max_cav)


def set_snow_voxel_denoiser_override(hypes, opt):
    if opt.snow_voxel_activation_threshold is None and \
            opt.snow_voxel_gate_floor is None:
        return
    denoiser_cfg = hypes.get('model', {}).get('args', {}).get(
        'snow_voxel_denoiser', {})
    if not denoiser_cfg.get('enable', False):
        raise ValueError(
            'Snow voxel overrides require an enabled snow_voxel_denoiser')
    if opt.snow_voxel_activation_threshold is not None:
        threshold = float(opt.snow_voxel_activation_threshold)
        if not 0.5 <= threshold < 1.0:
            raise ValueError(
                '--snow_voxel_activation_threshold must be in [0.5, 1.0)')
        denoiser_cfg['activation_threshold'] = threshold
    if opt.snow_voxel_gate_floor is not None:
        gate_floor = float(opt.snow_voxel_gate_floor)
        if not 0.0 <= gate_floor <= 1.0:
            raise ValueError('--snow_voxel_gate_floor must be in [0, 1]')
        denoiser_cfg['gate_floor'] = gate_floor
    print(
        'Snow voxel effective config: threshold=%.3f gate_floor=%.3f' %
        (float(denoiser_cfg.get('activation_threshold', 0.5)),
         float(denoiser_cfg.get('gate_floor', 0.05))))



def resolve_device(device_arg):
    if device_arg == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_arg)


def load_model_for_eval(model_dir, model, eval_epoch, explicit_path=''):
    if explicit_path:
        path = os.path.abspath(os.path.expanduser(explicit_path))
        if not os.path.isfile(path):
            raise FileNotFoundError('Checkpoint not found: %s' % path)
        print('loading explicit checkpoint %s' % path)
        checkpoint = torch.load(path, map_location='cpu')
        model.load_state_dict(checkpoint, strict=False)
        del checkpoint
        return model
    if eval_epoch <= 0:
        _, model = train_utils.load_saved_model(model_dir, model)
        return model

    checkpoint_path = os.path.join(model_dir, 'net_epoch%d.pth' % eval_epoch)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError('Checkpoint not found: %s' % checkpoint_path)

    print('loading epoch %d from %s' % (eval_epoch, checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint, strict=False)
    del checkpoint
    return model

def evaluate_one_weather(model, hypes, weather_name, validate_dir, device,
                         num_workers, global_sort_detections, fusion_mode,
                         input_branch='clean', start_index=0, max_frames=0,
                         frame_stride=1):
    hypes['validate_dir'] = validate_dir

    print('\n' + '=' * 80)
    print('Weather: %s' % weather_name)
    print('validate_dir: %s' % validate_dir)
    print('=' * 80)

    dataset = build_dataset(hypes, visualize=True, train=False)
    if start_index < 0:
        raise ValueError('--start_index must be non-negative')
    if frame_stride <= 0:
        raise ValueError('--frame_stride must be positive')
    indices = list(range(start_index, len(dataset), frame_stride))
    if max_frames > 0:
        indices = indices[:max_frames]
    evaluation_dataset = Subset(dataset, indices)
    data_loader = DataLoader(evaluation_dataset,
                             batch_size=1,
                             num_workers=num_workers,
                             collate_fn=dataset.collate_batch_test,
                             shuffle=False,
                             pin_memory=False,
                             drop_last=False)

    result_stat = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},
                   0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},
                   0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}
    communication_rates = []
    communication_rates_before_weather = []
    communication_rates_weather = []
    weather_R_means = []
    weather_R_density_means = []
    weather_R_isolation_means = []
    weather_R_isolation_low_fracs = []
    cav_counts = []

    model.eval()
    for _, batch_data in tqdm(enumerate(data_loader), total=len(data_loader)):
        with torch.no_grad():
            batch_data['ego'].pop('weather_augmentation_stats', None)
            batch_data = train_utils.to_device(batch_data, device)
            weather_processed = batch_data['ego'].pop(
                'processed_lidar_weather', None)
            if input_branch == 'weather_augmented':
                if weather_processed is None:
                    raise KeyError(
                        'weather_augmented input requires processed_lidar_weather')
                batch_data['ego']['processed_lidar'] = weather_processed
            elif weather_processed is not None:
                # Do not retain an unused second voxel branch during clean AP.
                del weather_processed
            record_len = batch_data['ego'].get('record_len', None)
            if record_len is not None:
                cav_counts.extend([int(x) for x in record_len.detach().cpu().view(-1).tolist()])
            if fusion_mode == 'no':
                keep_ego_only_for_no_fusion(batch_data['ego'])
            output_dict = OrderedDict()
            output_dict['ego'] = model(batch_data['ego'])

            com = output_dict['ego'].get('com', None)
            if com is not None:
                communication_rates.append(float(com.detach().cpu()))
            com_before_weather = output_dict['ego'].get('com_before_weather', None)
            if com_before_weather is not None:
                communication_rates_before_weather.append(
                    float(com_before_weather.detach().cpu()))
            com_weather = output_dict['ego'].get('com_weather', None)
            if com_weather is not None:
                communication_rates_weather.append(float(com_weather.detach().cpu()))
            weather_R_mean = output_dict['ego'].get('weather_R_mean', None)
            if weather_R_mean is not None:
                weather_R_means.append(float(weather_R_mean.detach().cpu()))
            weather_R_density_mean = output_dict['ego'].get(
                'weather_R_density_mean', None)
            if weather_R_density_mean is not None:
                weather_R_density_means.append(
                    float(weather_R_density_mean.detach().cpu()))
            weather_R_isolation_mean = output_dict['ego'].get(
                'weather_R_isolation_mean', None)
            if weather_R_isolation_mean is not None:
                weather_R_isolation_means.append(
                    float(weather_R_isolation_mean.detach().cpu()))
            weather_R_isolation_low_frac = output_dict['ego'].get(
                'weather_R_isolation_low_frac', None)
            if weather_R_isolation_low_frac is not None:
                weather_R_isolation_low_fracs.append(
                    float(weather_R_isolation_low_frac.detach().cpu()))

            pred_box_tensor, pred_score, gt_box_tensor = dataset.post_process(
                batch_data, output_dict)

            eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score,
                                       gt_box_tensor, result_stat, 0.3)
            eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score,
                                       gt_box_tensor, result_stat, 0.5)
            eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score,
                                       gt_box_tensor, result_stat, 0.7)

    ap30, _, _ = eval_utils.calculate_ap(result_stat, 0.3,
                                         global_sort_detections)
    ap50, _, _ = eval_utils.calculate_ap(result_stat, 0.5,
                                         global_sort_detections)
    ap70, _, _ = eval_utils.calculate_ap(result_stat, 0.7,
                                         global_sort_detections)
    avg_com = sum(communication_rates) / len(communication_rates) \
        if communication_rates else -1.0
    avg_com_before_weather = sum(communication_rates_before_weather) / \
        len(communication_rates_before_weather) \
        if communication_rates_before_weather else -1.0
    avg_com_weather = sum(communication_rates_weather) / \
        len(communication_rates_weather) \
        if communication_rates_weather else -1.0
    avg_weather_R_mean = sum(weather_R_means) / len(weather_R_means) \
        if weather_R_means else -1.0
    avg_weather_R_density_mean = sum(weather_R_density_means) / \
        len(weather_R_density_means) if weather_R_density_means else -1.0
    avg_weather_R_isolation_mean = sum(weather_R_isolation_means) / \
        len(weather_R_isolation_means) if weather_R_isolation_means else -1.0
    avg_weather_R_isolation_low_frac = sum(weather_R_isolation_low_fracs) / \
        len(weather_R_isolation_low_fracs) if weather_R_isolation_low_fracs else -1.0
    avg_cav = sum(cav_counts) / len(cav_counts) if cav_counts else -1.0
    min_cav = min(cav_counts) if cav_counts else -1
    max_cav = max(cav_counts) if cav_counts else -1

    print('Weather %s: AP@0.3 %.4f, AP@0.5 %.4f, AP@0.7 %.4f, Com %.4f, '
          'ComBeforeWeather %.4f, ComWeather %.4f, CAV %.2f [%d, %d]' %
          (weather_name, ap30, ap50, ap70, avg_com, avg_com_before_weather,
           avg_com_weather, avg_cav, min_cav, max_cav))
    if avg_weather_R_mean >= 0:
        print('Weather %s reliability: R %.4f, R_density %.4f, '
              'R_isolation %.4f, isolation_low_frac %.4f' %
              (weather_name, avg_weather_R_mean,
               avg_weather_R_density_mean,
               avg_weather_R_isolation_mean,
               avg_weather_R_isolation_low_frac))

    return {
        'weather': weather_name,
        'fusion_mode': fusion_mode,
        'input_branch': input_branch,
        'validate_dir': validate_dir,
        'samples': len(indices),
        'start_index': start_index,
        'frame_stride': frame_stride,
        'dataset_indices': indices,
        'ap30': ap30,
        'ap50': ap50,
        'ap70': ap70,
        'communication_rate': avg_com,
        'communication_rate_before_weather': avg_com_before_weather,
        'communication_rate_weather': avg_com_weather,
        'weather_R_mean': avg_weather_R_mean,
        'weather_R_density_mean': avg_weather_R_density_mean,
        'weather_R_isolation_mean': avg_weather_R_isolation_mean,
        'weather_R_isolation_low_frac': avg_weather_R_isolation_low_frac,
        'avg_cav': avg_cav,
        'min_cav': min_cav,
        'max_cav': max_cav,
    }


def main():
    opt = parse_args()
    weather_dirs = parse_weather_items(opt.weather)
    output_csv = opt.output_csv or os.path.join(opt.model_dir, 'weather_eval.csv')

    config_path = opt.hypes_yaml or os.path.join(opt.model_dir, 'config.yaml')
    hypes = yaml_utils.load_yaml(config_path)
    if opt.weather_augmentation_mode:
        hypes.setdefault('weather_augmentation', {})['mode'] = \
            opt.weather_augmentation_mode
    if opt.input_branch == 'weather_augmented':
        augmentation_cfg = hypes.get('weather_augmentation', {})
        point_level_modes = [
            'v2x_dgw_awa', 'pre_voxel_matched_dropout', 'physics_rain',
            'physics_fog', 'physics_snow', 'mixed_weather']
        if augmentation_cfg.get('mode') not in point_level_modes:
            raise ValueError(
                'weather_augmented evaluation requires a pre-voxel mode')
        augmentation_cfg['apply_to_validation'] = True
        if opt.weather_augmentation_seed is not None:
            augmentation_cfg['seed'] = opt.weather_augmentation_seed
        if opt.rain_rate is not None:
            if augmentation_cfg.get('mode') != 'physics_rain':
                raise ValueError('--rain_rate requires physics_rain mode')
            if opt.rain_rate <= 0:
                raise ValueError('--rain_rate must be positive')
            augmentation_cfg.setdefault('physics_rain', {})[
                'fixed_rain_rate'] = opt.rain_rate
        if augmentation_cfg.get('mode') == 'physics_fog':
            fog_cfg = augmentation_cfg.setdefault('physics_fog', {})
            if opt.fog_lookup_dir:
                fog_cfg['lookup_dir'] = opt.fog_lookup_dir
            if opt.fog_alpha is not None:
                if opt.fog_alpha <= 0:
                    raise ValueError('--fog_alpha must be positive')
                fog_cfg['fixed_alpha'] = opt.fog_alpha
            if not fog_cfg.get('lookup_dir'):
                raise ValueError(
                    'physics_fog requires --fog_lookup_dir or YAML lookup_dir')
        if opt.snowfall_rate is not None:
            if augmentation_cfg.get('mode') != 'physics_snow':
                raise ValueError(
                    '--snowfall_rate requires physics_snow mode')
            if opt.snowfall_rate <= 0:
                raise ValueError('--snowfall_rate must be positive')
            augmentation_cfg.setdefault('physics_snow', {})[
                'fixed_snowfall_rate'] = opt.snowfall_rate
    elif opt.rain_rate is not None or opt.fog_alpha is not None or \
            opt.snowfall_rate is not None or \
            opt.fog_lookup_dir or \
            opt.weather_augmentation_seed is not None:
        raise ValueError(
            'weather augmentation overrides require '
            '--input_branch weather_augmented')
    set_max_cav_override(hypes, opt.max_cav)
    set_uncertainty_override(hypes, opt)
    set_weather_reliability_override(hypes, opt)
    set_snow_voxel_denoiser_override(hypes, opt)

    print('Creating Model')
    model = train_utils.create_model(hypes)
    device = resolve_device(opt.device)
    if device.type == 'cuda':
        if device.index is not None:
            torch.cuda.set_device(device)
        torch.cuda.empty_cache()

    print('Loading Model from checkpoint')
    model = load_model_for_eval(
        opt.model_dir, model, opt.eval_epoch, opt.checkpoint_path)
    model.to(device)

    rows = []
    for weather_name, validate_dir in weather_dirs:
        rows.append(evaluate_one_weather(model, copy.deepcopy(hypes), weather_name,
                                         validate_dir, device,
                                         opt.num_workers,
                                         opt.global_sort_detections,
                                         opt.fusion_mode,
                                         opt.input_branch,
                                         opt.start_index,
                                         opt.max_frames,
                                         opt.frame_stride))

    os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=['weather', 'fusion_mode', 'input_branch',
                        'validate_dir', 'samples', 'start_index',
                        'frame_stride',
                        'ap30', 'ap50', 'ap70', 'communication_rate',
                        'communication_rate_before_weather',
                        'communication_rate_weather',
                        'weather_R_mean', 'weather_R_density_mean',
                        'weather_R_isolation_mean',
                        'weather_R_isolation_low_frac',
                        'avg_cav', 'min_cav', 'max_cav'],
            extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)

    checkpoint = (
        os.path.abspath(os.path.expanduser(opt.checkpoint_path))
        if opt.checkpoint_path else
        checkpoint_path(opt.model_dir, opt.eval_epoch))
    manifest_path = os.path.splitext(output_csv)[0] + '_manifest.json'
    write_manifest(manifest_path, {
        'evaluation_script': os.path.abspath(__file__),
        'repo_revision': git_revision(REPO_ROOT),
        'model_dir': os.path.abspath(opt.model_dir),
        'checkpoint': os.path.abspath(checkpoint) if checkpoint else None,
        'checkpoint_sha256': sha256_file(checkpoint),
        'config': os.path.abspath(config_path),
        'config_sha256': sha256_file(config_path),
        'arguments': vars(opt),
        'datasets': [{
            'weather': name,
            'validate_dir': path,
            'scenarios': scenario_names(path)
        } for name, path in weather_dirs],
        'effective_protocol': {
            'model_max_cav': hypes.get('model', {}).get(
                'args', {}).get('max_cav'),
            'train_max_cav': hypes.get('train_params', {}).get('max_cav'),
            'communication': hypes.get('model', {}).get(
                'args', {}).get('where2comm_fusion', {}).get(
                    'communication', {}),
            'postprocess': hypes.get('postprocess', {}),
            'weather_augmentation': hypes.get('weather_augmentation', {})
        },
        'results': rows
    })
    print('\nSaved summary to %s' % output_csv)
    print('Saved evaluation manifest to %s' % manifest_path)


if __name__ == '__main__':
    main()
