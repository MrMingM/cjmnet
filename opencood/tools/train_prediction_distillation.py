# -*- coding: utf-8 -*-
"""Foreground prediction distillation for weather-robust Where2comm.

The clean frozen teacher and weather student share the same anchors.  This
baseline distils only positive-anchor classification logits and box regression
outputs.  It deliberately does not constrain the fused BEV feature or add any
inference-time module.
"""

import argparse
import copy
import math
import os
import random
import statistics

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils
from opencood.tools.experiment_manifest import (
    checkpoint_path, git_revision, sha256_file, write_manifest)
from opencood.tools.train_weather_consistency import (
    build_teacher_student_inputs, configure_weather_augmentation,
    freeze_batchnorm, keep_batchnorm_eval, load_model_from_dir,
    print_weather_augmentation_stats)
from opencood.models.sub_modules.snow_voxel_denoiser import \
    snow_voxel_novelty_labels


def parse_args():
    parser = argparse.ArgumentParser(
        description='Positive-anchor clean-to-weather prediction distillation.')
    parser.add_argument('--model_dir', required=True)
    parser.add_argument('--hypes_yaml', default='')
    parser.add_argument('--teacher_model_dir', default='')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1.0e-6)
    parser.add_argument('--warmup_epochs', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=20260720,
                        help='Training, shuffle, NumPy and PyTorch seed.')
    parser.add_argument('--weather_augmentation_seed', type=int, default=None,
                        help='Optional override for deterministic weather augmentation.')
    parser.add_argument(
        '--weather_augmentation_mode', default='',
        choices=['', 'v2x_dgw_awa', 'pre_voxel_matched_dropout',
                 'physics_rain', 'physics_fog', 'physics_snow',
                 'mixed_weather'],
        help='Optional override of weather_augmentation.mode from YAML.')
    parser.add_argument('--fog_lookup_dir', default='')
    parser.add_argument('--fog_alpha_low', type=float, default=0.005)
    parser.add_argument('--fog_alpha_high', type=float, default=0.03)
    parser.add_argument('--fog_fixed_alpha', type=float, default=None)
    parser.add_argument(
        '--fog_alpha_sampling',
        choices=['lookup_discrete', 'uniform'],
        default='lookup_discrete',
        help='Sample official lookup-table alphas by default. Uniform uses '
             'continuous hard attenuation with the nearest soft-fog table.')
    parser.add_argument('--snowfall_rate_low', type=float, default=0.5)
    parser.add_argument('--snowfall_rate_high', type=float, default=2.5)
    parser.add_argument('--snow_fixed_rate', type=float, default=None)
    parser.add_argument(
        '--snow_terminal_velocity', type=float, default=1.0)
    parser.add_argument('--snow_density', type=float, default=0.1)
    parser.add_argument('--snow_intercept_scale', type=float, default=1.0)
    parser.add_argument(
        '--mixed_physics_rain_probability', type=float, default=0.7,
        help='Scene probability of Physics-Rain in mixed_weather mode.')
    parser.add_argument(
        '--max_train_steps', type=int, default=0,
        help='Optional smoke-test cap per epoch; zero uses the full loader.')
    parser.add_argument(
        '--max_val_steps', type=int, default=0,
        help='Optional validation cap; zero uses the full loader.')
    parser.add_argument(
        '--save_every_steps', type=int, default=0,
        help='Save net_stepN.pth every N optimizer steps; zero disables it.')
    parser.add_argument('--train_batchnorm', action='store_true')
    parser.add_argument('--lambda_weather_det', type=float, default=1.0)
    parser.add_argument('--lambda_clean_det', type=float, default=0.5)
    parser.add_argument('--lambda_pred_cls', type=float, default=0.5)
    parser.add_argument('--lambda_pred_reg', type=float, default=0.5)
    parser.add_argument('--lambda_clean_comm', type=float, default=0.0,
                        help='Preserve clean single-agent communication probabilities.')
    parser.add_argument('--lambda_weather_comm', type=float, default=0.0,
                        help='Match weather single-agent communication probabilities '
                             'to the frozen clean teacher.')
    parser.add_argument(
        '--comm_mask_temperature', type=float, default=0.002,
        help='Temperature of the differentiable communication-threshold mask.')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument(
        '--teacher_positive_threshold', type=float, default=0.0,
        help='Optional teacher-confidence filter inside GT-positive anchors.')
    parser.add_argument('--reg_beta', type=float, default=0.1)
    parser.add_argument(
        '--lambda_recoverability_kd', type=float, default=0.0,
        help='Extra KD weight assigned to targets that are weakly observed '
             'by Ego but well observed by at least one neighbor.')
    parser.add_argument(
        '--recoverability_ego_scale', type=float, default=10.0,
        help='Point-count scale controlling how quickly Ego need decays.')
    parser.add_argument(
        '--recoverability_min_neighbor_points', type=float, default=5.0,
        help='Point-count scale for usable neighbor support.')
    parser.add_argument(
        '--recoverability_margin', type=float, default=0.0,
        help='Required log point-count advantage of the best neighbor.')
    parser.add_argument(
        '--recoverability_temperature', type=float, default=0.5,
        help='Temperature of the neighbor-vs-Ego visibility comparison.')
    parser.add_argument(
        '--lambda_counterfactual_kd', type=float, default=0.0,
        help='Extra KD weight for positive anchors whose clean all-agent '
             'teacher confidence exceeds its Ego-only confidence.')
    parser.add_argument(
        '--counterfactual_gain_scale', type=float, default=0.1,
        help='Teacher probability gain mapped to a unit counterfactual '
             'weight; gains at or above this value saturate at one.')
    parser.add_argument(
        '--counterfactual_min_gain', type=float, default=0.0,
        help='Ignore all-vs-Ego teacher probability gains below this value.')
    parser.add_argument(
        '--lambda_counterfactual_gain', type=float, default=0.0,
        help='Distil the clean teacher all-vs-Ego probability gain into the '
             'weather student.')
    parser.add_argument(
        '--counterfactual_gain_beta', type=float, default=0.1,
        help='Smooth-L1 beta for normalized counterfactual-gain distillation.')
    parser.add_argument(
        '--lambda_weather_feature', type=float, default=0.0,
        help='Clean-teacher to weather-student per-agent BEV feature loss.')
    parser.add_argument(
        '--lambda_clean_feature', type=float, default=0.0,
        help='Identity-preservation loss on clean adapted BEV features.')
    parser.add_argument(
        '--feature_background_floor', type=float, default=0.05,
        help='Minimum spatial weight outside teacher-confident BEV cells.')
    parser.add_argument(
        '--lambda_dg_agent_align', type=float, default=0.0,
        help='Masked clean-teacher/weather-student alignment before the '
             'per-agent BEV backbone.')
    parser.add_argument(
        '--lambda_dg_fusion_align', type=float, default=0.0,
        help='Clean-teacher/weather-student alignment after cooperative '
             'feature fusion.')
    parser.add_argument(
        '--lambda_dg_agent_contrast', type=float, default=0.0,
        help='Paired-agent contrastive alignment between clean and weather '
             'per-agent features.')
    parser.add_argument(
        '--dg_contrast_temperature', type=float, default=0.2)
    parser.add_argument(
        '--dg_pool_size', type=int, default=4,
        help='Adaptive pooling size used by agent contrastive alignment.')
    parser.add_argument(
        '--dg_background_floor', type=float, default=0.05,
        help='Background weight for fused semantic alignment.')
    parser.add_argument(
        '--adapter_hidden_channels', type=int, default=64)
    parser.add_argument(
        '--adapter_only', action='store_true',
        help='Freeze the baseline detector and optimize only the weather '
             'feature adapter or CURE restorer.')
    parser.add_argument(
        '--lambda_snow_voxel_denoise', type=float, default=0.0,
        help='Weather-pillar BCE weight for classifying occupied pillars '
             'that do not exist in the paired clean frame.')
    parser.add_argument(
        '--lambda_snow_voxel_clean', type=float, default=0.0,
        help='Clean identity BCE weight that teaches every clean occupied '
             'pillar to remain unsuppressed.')
    parser.add_argument(
        '--snow_voxel_positive_weight', type=float, default=5.0,
        help='Minimum positive-class weight for newly occupied weather '
             'pillars; the effective value is batch-balanced automatically.')
    parser.add_argument(
        '--snow_voxel_max_positive_weight', type=float, default=80.0,
        help='Upper bound for automatic snow-voxel positive reweighting.')
    parser.add_argument(
        '--snow_voxel_hidden_channels', type=int, default=64)
    parser.add_argument(
        '--snow_voxel_gate_floor', type=float, default=0.05,
        help='Minimum retained feature weight for a predicted snow pillar.')
    parser.add_argument(
        '--snow_voxel_activation_threshold', type=float, default=0.5,
        help='Noise probability above which pillar suppression starts.')
    parser.add_argument(
        '--snow_denoiser_only', action='store_true',
        help='Freeze the detector and optimize only the pillar-level snow '
             'voxel denoiser.')
    parser.add_argument(
        '--lambda_cure_recovery', type=float, default=0.0,
        help='CURE recoverability-map supervision weight.')
    parser.add_argument(
        '--lambda_cure_harm', type=float, default=0.0,
        help='CURE harmful-collaboration classification weight.')
    parser.add_argument(
        '--lambda_cure_utility', type=float, default=0.0,
        help='CURE signed weather marginal-utility regression weight.')
    parser.add_argument(
        '--lambda_cure_feature', type=float, default=0.0,
        help='Feature restoration weight focused by recoverable utility.')
    parser.add_argument(
        '--lambda_cure_identity', type=float, default=0.0,
        help='Clean-feature identity preservation for the CURE restorer.')
    parser.add_argument(
        '--cure_utility_scale', type=float, default=0.10,
        help='Absolute teacher probability delta mapped to unit utility.')
    parser.add_argument(
        '--cure_harm_eps', type=float, default=0.005,
        help='Weather marginal utility below -eps is harmful.')
    parser.add_argument(
        '--cure_min_recovery_gap', type=float, default=0.0,
        help='Minimum clean-minus-weather utility loss considered recoverable.')
    parser.add_argument(
        '--cure_positive_weight', type=float, default=3.0,
        help='Extra weight for positive recovery and harm labels.')
    parser.add_argument(
        '--cure_background_weight', type=float, default=0.005,
        help='Low-weight zero-recovery supervision outside GT-positive '
             'neighbor cells.')
    parser.add_argument(
        '--cure_hidden_channels', type=int, default=64)
    parser.add_argument(
        '--cure_location', choices=['pre_fusion', 'post_fusion'],
        default='pre_fusion',
        help='Apply CURE before the BEV backbone or after Where2comm fusion.')
    parser.add_argument(
        '--cure_apply_fusion_gate', action='store_true',
        help='Use the learned harm map to softly gate neighbor fusion.')
    parser.add_argument(
        '--cure_fusion_gate_floor', type=float, default=0.5,
        help='Minimum neighbor weight when CURE fusion gating is enabled.')
    parser.add_argument('--dropout_base', type=float, default=0.05)
    parser.add_argument('--dropout_range', type=float, default=0.35)
    parser.add_argument('--dropout_max', type=float, default=0.65)
    parser.add_argument('--save_name', default='prediction_kd')
    return parser.parse_args()


def setup_hypes(opt):
    config_path = opt.hypes_yaml or os.path.join(
        opt.model_dir, 'config.yaml')
    hypes = yaml_utils.load_yaml(config_path)
    hypes['name'] = '%s_%s' % (
        hypes.get('name', 'where2comm'), opt.save_name)
    hypes['train_params']['epoches'] = opt.epochs
    hypes['train_params']['batch_size'] = opt.batch_size
    hypes['optimizer']['lr'] = opt.lr
    if 'lr_scheduler' in hypes:
        hypes['lr_scheduler']['epoches'] = opt.epochs
        if hypes['lr_scheduler'].get('core_method') == 'cosineannealwarm':
            hypes['lr_scheduler']['warmup_epoches'] = max(
                0, min(opt.warmup_epochs, opt.epochs))
            hypes['lr_scheduler']['warmup_lr'] = opt.lr
            hypes['lr_scheduler']['lr_min'] = min(
                float(hypes['lr_scheduler'].get('lr_min', opt.lr)),
                opt.lr)
    if opt.weather_augmentation_mode:
        hypes.setdefault('weather_augmentation', {})['mode'] = \
            opt.weather_augmentation_mode
    augmentation_cfg = configure_weather_augmentation(hypes, opt)
    if opt.weather_augmentation_seed is not None:
        augmentation_cfg['seed'] = int(opt.weather_augmentation_seed)
    if augmentation_cfg['mode'] == 'mixed_weather':
        if not 0.0 <= opt.mixed_physics_rain_probability <= 1.0:
            raise ValueError(
                '--mixed_physics_rain_probability must be in [0, 1]')
        augmentation_cfg.setdefault('mixed_weather', {})[
            'physics_rain_probability'] = float(
                opt.mixed_physics_rain_probability)
        # Use a deliberately mild AWA branch. The original 0.5--0.8 range
        # produced a large synthetic-domain shift in the earlier experiments.
        awa_cfg = augmentation_cfg.setdefault('v2x_dgw_awa', {})
        awa_cfg.setdefault('perturbation_frame', 'local_sensor')
        awa_cfg.setdefault('dynamic_range', True)
        awa_cfg.setdefault('range_proportion_low', 0.75)
        awa_cfg.setdefault('range_proportion_high', 0.95)
        awa_cfg.setdefault('random_drop_out', True)
        awa_cfg.setdefault('keep_ratio_low', 0.9)
        awa_cfg.setdefault('keep_ratio_high', 1.0)
        awa_cfg.setdefault('random_gaussian_noise', True)
        awa_cfg.setdefault('max_noise_points', 500)
        awa_cfg.setdefault('noise_resolution', 0.01)
        awa_cfg.setdefault('noise_intensity_std', 0.25)
        awa_cfg.setdefault('random_jittering', True)
        awa_cfg.setdefault('jitter_std', 0.005)
        awa_cfg.setdefault('jitter_clip', 0.02)
    if augmentation_cfg['mode'] == 'physics_fog':
        fog_cfg = augmentation_cfg.setdefault('physics_fog', {})
        if opt.fog_lookup_dir:
            fog_cfg['lookup_dir'] = opt.fog_lookup_dir
        if not fog_cfg.get('lookup_dir'):
            raise ValueError(
                'physics_fog requires --fog_lookup_dir or YAML lookup_dir')
        fog_cfg['alpha_low'] = float(opt.fog_alpha_low)
        fog_cfg['alpha_high'] = float(opt.fog_alpha_high)
        fog_cfg['fixed_alpha'] = opt.fog_fixed_alpha
        fog_cfg['alpha_sampling'] = opt.fog_alpha_sampling
        fog_cfg.setdefault('gamma', 1.0e-6)
        fog_cfg.setdefault('beta_scale', 1.0)
        fog_cfg.setdefault('intensity_threshold', 0.01)
        fog_cfg.setdefault('distance_noise_scale', 2.0)
        fog_cfg.setdefault('max_soft_response', 1.0)
    if augmentation_cfg['mode'] == 'physics_snow':
        snow_cfg = augmentation_cfg.setdefault('physics_snow', {})
        snow_cfg['snowfall_rate_low'] = float(opt.snowfall_rate_low)
        snow_cfg['snowfall_rate_high'] = float(opt.snowfall_rate_high)
        snow_cfg['fixed_snowfall_rate'] = opt.snow_fixed_rate
        snow_cfg['terminal_velocity'] = float(
            opt.snow_terminal_velocity)
        snow_cfg['snow_density'] = float(opt.snow_density)
        snow_cfg['intercept_scale'] = float(
            opt.snow_intercept_scale)
        snow_cfg.setdefault('beam_divergence', 0.003)
        snow_cfg.setdefault('snow_reflectivity', 0.9)
        snow_cfg.setdefault('intensity_decay_range', 15.0)
        snow_cfg.setdefault('intensity_threshold', 0.01)
        snow_cfg.setdefault('max_particle_diameter', 0.02)
    augmentation_cfg['apply_to_validation'] = True
    adapter_enabled = (
        opt.lambda_weather_feature > 0 or opt.lambda_clean_feature > 0)
    cure_enabled = any(value > 0 for value in [
        opt.lambda_cure_recovery,
        opt.lambda_cure_harm,
        opt.lambda_cure_utility,
        opt.lambda_cure_feature,
        opt.lambda_cure_identity,
    ])
    snow_denoiser_enabled = (
        opt.lambda_snow_voxel_denoise > 0 or
        opt.lambda_snow_voxel_clean > 0)
    if sum([adapter_enabled, cure_enabled, snow_denoiser_enabled]) > 1:
        raise ValueError(
            'Weather feature adapter, CURE and snow voxel denoiser are '
            'mutually exclusive in one run.')
    if snow_denoiser_enabled and \
            augmentation_cfg['mode'] != 'physics_snow':
        raise ValueError(
            'Snow voxel supervision currently requires physics_snow paired '
            'augmentation.')
    if snow_denoiser_enabled and opt.lambda_snow_voxel_denoise <= 0:
        raise ValueError(
            'Snow voxel denoising requires '
            '--lambda_snow_voxel_denoise > 0.')
    hypes['model']['args']['weather_feature_adapter'] = {
        'enable': adapter_enabled,
        'hidden_channels': int(opt.adapter_hidden_channels),
        'groups': 8
    }
    hypes['model']['args']['counterfactual_utility_restorer'] = {
        'enable': cure_enabled and opt.cure_location == 'pre_fusion',
        'hidden_channels': int(opt.cure_hidden_channels),
        'groups': 8,
        'apply_fusion_gate': bool(opt.cure_apply_fusion_gate),
        'fusion_gate_floor': float(opt.cure_fusion_gate_floor),
    }
    hypes['model']['args']['counterfactual_fusion_restorer'] = {
        'enable': cure_enabled and opt.cure_location == 'post_fusion',
        'hidden_channels': int(opt.cure_hidden_channels),
        'groups': 8,
    }
    hypes['model']['args']['snow_voxel_denoiser'] = {
        'enable': snow_denoiser_enabled,
        'hidden_channels': int(opt.snow_voxel_hidden_channels),
        'max_points': int(
            hypes.get('preprocess', {}).get('args', {}).get(
                'max_points_per_voxel', 32)),
        'gate_floor': float(opt.snow_voxel_gate_floor),
        'activation_threshold':
            float(opt.snow_voxel_activation_threshold),
    }
    hypes['prediction_distillation'] = {
        'teacher_model_dir': opt.teacher_model_dir or opt.model_dir,
        'student_init_model_dir': opt.model_dir,
        'lambda_weather_det': opt.lambda_weather_det,
        'lambda_clean_det': opt.lambda_clean_det,
        'lambda_pred_cls': opt.lambda_pred_cls,
        'lambda_pred_reg': opt.lambda_pred_reg,
        'lambda_clean_comm': opt.lambda_clean_comm,
        'lambda_weather_comm': opt.lambda_weather_comm,
        'comm_mask_temperature': opt.comm_mask_temperature,
        'temperature': opt.temperature,
        'teacher_positive_threshold': opt.teacher_positive_threshold,
        'reg_beta': opt.reg_beta,
        'lambda_recoverability_kd': opt.lambda_recoverability_kd,
        'recoverability_ego_scale': opt.recoverability_ego_scale,
        'recoverability_min_neighbor_points':
            opt.recoverability_min_neighbor_points,
        'recoverability_margin': opt.recoverability_margin,
        'recoverability_temperature': opt.recoverability_temperature,
        'lambda_counterfactual_kd': opt.lambda_counterfactual_kd,
        'counterfactual_gain_scale': opt.counterfactual_gain_scale,
        'counterfactual_min_gain': opt.counterfactual_min_gain,
        'lambda_counterfactual_gain': opt.lambda_counterfactual_gain,
        'counterfactual_gain_beta': opt.counterfactual_gain_beta,
        'lambda_weather_feature': opt.lambda_weather_feature,
        'lambda_clean_feature': opt.lambda_clean_feature,
        'feature_background_floor': opt.feature_background_floor,
        'lambda_dg_agent_align': opt.lambda_dg_agent_align,
        'lambda_dg_fusion_align': opt.lambda_dg_fusion_align,
        'lambda_dg_agent_contrast': opt.lambda_dg_agent_contrast,
        'dg_contrast_temperature': opt.dg_contrast_temperature,
        'dg_pool_size': opt.dg_pool_size,
        'dg_background_floor': opt.dg_background_floor,
        'lambda_cure_recovery': opt.lambda_cure_recovery,
        'lambda_cure_harm': opt.lambda_cure_harm,
        'lambda_cure_utility': opt.lambda_cure_utility,
        'lambda_cure_feature': opt.lambda_cure_feature,
        'lambda_cure_identity': opt.lambda_cure_identity,
        'cure_utility_scale': opt.cure_utility_scale,
        'cure_harm_eps': opt.cure_harm_eps,
        'cure_min_recovery_gap': opt.cure_min_recovery_gap,
        'cure_positive_weight': opt.cure_positive_weight,
        'cure_background_weight': opt.cure_background_weight,
        'cure_location': opt.cure_location,
        'cure_apply_fusion_gate': bool(opt.cure_apply_fusion_gate),
        'cure_fusion_gate_floor': opt.cure_fusion_gate_floor,
        'adapter_only': bool(opt.adapter_only),
        'lambda_snow_voxel_denoise':
            opt.lambda_snow_voxel_denoise,
        'lambda_snow_voxel_clean': opt.lambda_snow_voxel_clean,
        'snow_voxel_positive_weight':
            opt.snow_voxel_positive_weight,
        'snow_voxel_max_positive_weight':
            opt.snow_voxel_max_positive_weight,
        'snow_voxel_gate_floor': opt.snow_voxel_gate_floor,
        'snow_voxel_activation_threshold':
            opt.snow_voxel_activation_threshold,
        'snow_denoiser_only': bool(opt.snow_denoiser_only),
        'augmentation_mode': augmentation_cfg['mode'],
        'weather_augmentation': copy.deepcopy(augmentation_cfg)
    }
    return hypes, config_path


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def positive_anchor_mask(label_dict, psm):
    mask = label_dict['pos_equal_one'].to(
        device=psm.device, dtype=psm.dtype)
    if mask.shape == psm.shape:
        return mask
    if mask.dim() == 4 and \
            mask.shape[0] == psm.shape[0] and \
            mask.shape[1] == psm.shape[2] and \
            mask.shape[2] == psm.shape[3] and \
            mask.shape[3] == psm.shape[1]:
        return mask.permute(0, 3, 1, 2).contiguous()
    raise ValueError(
        'Cannot align pos_equal_one %s with psm %s' %
        (tuple(mask.shape), tuple(psm.shape)))


def recoverability_anchor_map(batch_ego, label_dict, psm, opt):
    """Map GT-level complementary visibility to positive anchors.

    A high value means that Ego contains few clean points inside the GT box,
    while at least one neighbor contains useful support and observes the box
    better than Ego.  GT information is used only to construct a training
    weight and is not required by the inference model.
    """
    required = ['object_point_counts', 'record_len']
    missing = [name for name in required if name not in batch_ego]
    if missing:
        raise KeyError(
            'Recoverability KD requires dataset fields: %s' %
            ', '.join(missing))
    if 'pos_gt_index' not in label_dict:
        raise KeyError(
            'Recoverability KD requires label_dict[pos_gt_index]')

    counts = batch_ego['object_point_counts'].to(
        device=psm.device, dtype=psm.dtype)
    record_len = batch_ego['record_len'].to(device=psm.device)
    if counts.dim() != 3 or counts.shape[0] != psm.shape[0]:
        raise ValueError(
            'object_point_counts must have shape [B, max_cav, max_gt], '
            'got %s' % (tuple(counts.shape),))

    ego_counts = counts[:, 0]
    best_neighbor_counts = torch.zeros_like(ego_counts)
    for batch_index in range(counts.shape[0]):
        cav_count = int(record_len[batch_index].item())
        if cav_count > 1:
            best_neighbor_counts[batch_index] = counts[
                batch_index, 1:cav_count].amax(dim=0)

    ego_scale = max(float(opt.recoverability_ego_scale), 1.0e-4)
    neighbor_scale = max(
        float(opt.recoverability_min_neighbor_points), 1.0e-4)
    comparison_temperature = max(
        float(opt.recoverability_temperature), 1.0e-4)
    ego_need = torch.exp(-ego_counts / ego_scale)
    neighbor_support = 1.0 - torch.exp(
        -best_neighbor_counts / neighbor_scale)
    log_advantage = (
        torch.log1p(best_neighbor_counts) - torch.log1p(ego_counts) -
        float(opt.recoverability_margin))
    complementary = torch.sigmoid(
        log_advantage / comparison_temperature)
    gt_recoverability = ego_need * neighbor_support * complementary

    gt_index = label_dict['pos_gt_index'].to(device=psm.device).long()
    if gt_index.dim() == 4 and \
            gt_index.shape[0] == psm.shape[0] and \
            gt_index.shape[1] == psm.shape[2] and \
            gt_index.shape[2] == psm.shape[3] and \
            gt_index.shape[3] == psm.shape[1]:
        gt_index = gt_index.permute(0, 3, 1, 2).contiguous()
    elif gt_index.shape != psm.shape:
        raise ValueError(
            'Cannot align pos_gt_index %s with psm %s' %
            (tuple(gt_index.shape), tuple(psm.shape)))

    valid = gt_index >= 0
    safe_index = gt_index.clamp(
        min=0, max=gt_recoverability.shape[1] - 1)
    anchor_map = torch.gather(
        gt_recoverability, 1, safe_index.view(psm.shape[0], -1)
    ).view_as(psm)
    return anchor_map * valid.to(dtype=psm.dtype)


def ego_only_agent_mask(record_len, device):
    """Return a flattened mask that keeps the first CAV of every batch."""
    lengths = record_len.detach().cpu().tolist()
    mask = torch.zeros(
        int(sum(lengths)), dtype=torch.bool, device=device)
    offset = 0
    for length in lengths:
        if int(length) <= 0:
            raise ValueError('record_len entries must be positive')
        mask[offset] = True
        offset += int(length)
    return mask


def counterfactual_utility_targets(
        teacher, clean_encoded, weather_encoded, clean_all, weather_all,
        label_dict, record_len, opt):
    """Build dense signed utility targets from clean/weather leave-one-out.

    For every neighbor and positive GT anchor, the clean and weather marginal
    utilities are measured as ``all - all_without_neighbor``.  The returned
    maps are aligned with flattened per-agent features.  Ego rows are invalid
    because their collaboration value is undefined.
    """
    clean_probability = clean_all['psm'].detach().sigmoid()
    weather_probability = weather_all['psm'].detach().sigmoid()
    if clean_probability.shape != weather_probability.shape:
        raise ValueError(
            'Clean/weather teacher PSM shapes differ: %s vs %s' %
            (tuple(clean_probability.shape),
             tuple(weather_probability.shape)))
    positive = positive_anchor_mask(
        label_dict, clean_all['psm']).detach()
    batch_size, _, height, width = clean_probability.shape
    total_agents = int(record_len.sum().item())
    device = clean_probability.device
    dtype = clean_probability.dtype
    utility = torch.zeros(
        total_agents, 1, height, width, device=device, dtype=dtype)
    clean_utility = torch.zeros_like(utility)
    weather_utility = torch.zeros_like(utility)
    recovery = torch.zeros_like(utility)
    harm = torch.zeros_like(utility)
    valid = torch.zeros_like(utility, dtype=torch.bool)
    supervision_weight = torch.zeros_like(utility)

    scale = max(float(opt.cure_utility_scale), 1.0e-4)
    harm_eps = max(float(opt.cure_harm_eps), 0.0)
    min_gap = max(float(opt.cure_min_recovery_gap), 0.0)
    offset = 0
    neighbor_count = 0
    for batch_index, length_value in enumerate(
            record_len.detach().cpu().tolist()):
        length = int(length_value)
        if length <= 0:
            raise ValueError('record_len entries must be positive')
        positive_item = positive[batch_index]
        positive_count = positive_item.sum(
            dim=0, keepdim=True)
        valid_item = positive_count > 0
        safe_count = positive_count.clamp_min(1.0)
        for local_agent in range(1, length):
            flat_agent = offset + local_agent
            supervision_weight[flat_agent].fill_(
                max(float(opt.cure_background_weight), 0.0))
            keep_mask = torch.ones(
                total_agents, dtype=torch.bool, device=device)
            keep_mask[flat_agent] = False
            clean_minus = teacher.forward_from_encoded(
                clean_encoded, agent_mask=keep_mask, return_aux=False)
            weather_minus = teacher.forward_from_encoded(
                weather_encoded, agent_mask=keep_mask, return_aux=False)
            clean_delta = (
                clean_probability[batch_index] -
                clean_minus['psm'][batch_index].detach().sigmoid())
            weather_delta = (
                weather_probability[batch_index] -
                weather_minus['psm'][batch_index].detach().sigmoid())
            clean_cell = (
                clean_delta * positive_item).sum(
                    dim=0, keepdim=True) / safe_count
            weather_cell = (
                weather_delta * positive_item).sum(
                    dim=0, keepdim=True) / safe_count
            lost_cell = (
                clean_cell - weather_cell - min_gap).clamp_min(0.0)

            clean_utility[flat_agent] = clean_cell
            weather_utility[flat_agent] = weather_cell
            utility[flat_agent] = (
                weather_cell / scale).clamp(-1.0, 1.0)
            recovery[flat_agent] = (
                lost_cell / scale).clamp(0.0, 1.0)
            harm[flat_agent] = (
                weather_cell < -harm_eps).to(dtype)
            valid[flat_agent] = valid_item
            supervision_weight[flat_agent][valid_item] = 1.0
            neighbor_count += 1
        offset += length
    if offset != total_agents or batch_size != record_len.shape[0]:
        raise ValueError('record_len is inconsistent with teacher outputs')
    valid_count = int(valid.sum().detach().item())
    return {
        'utility': utility,
        'clean_utility': clean_utility,
        'weather_utility': weather_utility,
        'recovery': recovery,
        'harm': harm,
        'valid': valid,
        'supervision_weight': supervision_weight,
        'neighbor_count': neighbor_count,
        'valid_count': valid_count,
    }


def _resize_cure_prediction(value, size):
    if value.shape[-2:] == size:
        return value
    return F.interpolate(
        value, size=size, mode='bilinear', align_corners=False)


def aggregate_cure_targets(targets, record_len):
    """Aggregate per-neighbor counterfactual targets for fused BEV recovery."""
    scene_targets = {}
    for name in [
            'recovery', 'harm', 'valid', 'supervision_weight',
            'utility', 'clean_utility', 'weather_utility']:
        values = []
        offset = 0
        source = targets[name]
        for length_value in record_len.detach().cpu().tolist():
            length = int(length_value)
            neighbors = source[offset + 1:offset + length]
            if neighbors.shape[0] == 0:
                values.append(torch.zeros_like(source[0]))
            elif name == 'valid':
                values.append(neighbors.any(dim=0))
            elif name in ['recovery', 'harm', 'supervision_weight']:
                values.append(neighbors.amax(dim=0))
            else:
                positive = neighbors.amax(dim=0)
                negative = neighbors.amin(dim=0)
                values.append(torch.where(
                    positive.abs() >= negative.abs(),
                    positive, negative))
            offset += length
        scene_targets[name] = torch.stack(values, dim=0)
    scene_targets['neighbor_count'] = targets['neighbor_count']
    scene_targets['valid_count'] = int(
        scene_targets['valid'].sum().detach().item())
    return scene_targets


def cure_supervision_losses(student_encoded, targets, opt):
    """Supervise CURE recovery, harm and signed utility maps."""
    aux = student_encoded.get('cure_aux', None)
    if aux is None:
        raise ValueError(
            'CURE losses require model counterfactual_utility_restorer')
    size = targets['utility'].shape[-2:]
    recovery_logit = _resize_cure_prediction(
        aux['recoverability_logit'], size)
    harm_logit = _resize_cure_prediction(aux['harm_logit'], size)
    predicted_utility = _resize_cure_prediction(aux['utility'], size)
    valid = targets['valid'].to(dtype=predicted_utility.dtype)
    supervision_weight = targets['supervision_weight'].to(
        dtype=predicted_utility.dtype)
    denominator = valid.sum().clamp_min(1.0)
    positive_weight = max(float(opt.cure_positive_weight), 1.0)

    recovery_weight = supervision_weight * (
        1.0 + (positive_weight - 1.0) * targets['recovery'])
    recovery_map = F.binary_cross_entropy_with_logits(
        recovery_logit, targets['recovery'], reduction='none')
    recovery_loss = (
        recovery_map * recovery_weight).sum() / \
        recovery_weight.sum().clamp_min(1.0)

    harm_weight = supervision_weight * (
        1.0 + (positive_weight - 1.0) * targets['harm'])
    harm_map = F.binary_cross_entropy_with_logits(
        harm_logit, targets['harm'], reduction='none')
    harm_loss = (
        harm_map * harm_weight).sum() / \
        harm_weight.sum().clamp_min(1.0)

    utility_importance = 1.0 + targets['utility'].abs() * (
        positive_weight - 1.0)
    utility_map = F.smooth_l1_loss(
        predicted_utility, targets['utility'],
        reduction='none', beta=0.1)
    utility_loss = (
        utility_map * utility_importance * valid).sum() / denominator
    return recovery_loss, harm_loss, utility_loss


def cure_fused_feature_restoration_loss(
        student_out, teacher_out, targets, background_floor=0.05):
    """Align fused weather semantics only where utility was recoverable."""
    student_feature = student_out['fused_feature']
    teacher_feature = teacher_out['fused_feature'].detach()
    if student_feature.shape != teacher_feature.shape:
        raise ValueError(
            'CURE fused feature shapes differ: %s vs %s' %
            (tuple(student_feature.shape), tuple(teacher_feature.shape)))
    valid = targets['valid'].to(dtype=student_feature.dtype)
    recovery = targets['recovery'].to(dtype=student_feature.dtype)
    if valid.shape[-2:] != student_feature.shape[-2:]:
        valid = F.interpolate(
            valid, size=student_feature.shape[-2:], mode='nearest')
        recovery = F.interpolate(
            recovery, size=student_feature.shape[-2:],
            mode='bilinear', align_corners=False)
    floor = min(max(float(background_floor), 0.0), 1.0)
    weight = valid * (floor + (1.0 - floor) * recovery)
    difference = F.smooth_l1_loss(
        student_feature, teacher_feature,
        reduction='none', beta=0.1).mean(dim=1, keepdim=True)
    return (difference * weight).sum() / weight.sum().clamp_min(1.0)


def cure_fused_identity_loss(student_out, teacher_out, background_floor=0.05):
    """Keep clean fused features equal to the frozen clean teacher."""
    student_feature = student_out['fused_feature']
    teacher_feature = teacher_out['fused_feature'].detach()
    confidence = teacher_out['psm'].detach().sigmoid().amax(
        dim=1, keepdim=True)
    if confidence.shape[-2:] != student_feature.shape[-2:]:
        confidence = F.interpolate(
            confidence, size=student_feature.shape[-2:],
            mode='bilinear', align_corners=False)
    floor = min(max(float(background_floor), 0.0), 1.0)
    weight = floor + (1.0 - floor) * confidence
    difference = F.smooth_l1_loss(
        student_feature, teacher_feature,
        reduction='none', beta=0.1).mean(dim=1, keepdim=True)
    return (difference * weight).sum() / weight.sum().clamp_min(1.0)


def cure_feature_restoration_loss(
        student_encoded, teacher_encoded, targets, background_floor=0.05):
    """Restore only semantic regions with weather-lost clean utility."""
    student_feature = student_encoded['spatial_features_2d']
    teacher_feature = teacher_encoded[
        'spatial_features_2d'].detach()
    if student_feature.shape != teacher_feature.shape:
        raise ValueError(
            'CURE student/teacher feature shapes differ: %s vs %s' %
            (tuple(student_feature.shape), tuple(teacher_feature.shape)))
    valid = targets['valid'].to(dtype=student_feature.dtype)
    recovery = targets['recovery'].to(dtype=student_feature.dtype)
    if valid.shape[-2:] != student_feature.shape[-2:]:
        valid = F.interpolate(valid, size=student_feature.shape[-2:],
                              mode='nearest')
        recovery = F.interpolate(
            recovery, size=student_feature.shape[-2:],
            mode='bilinear', align_corners=False)
    floor = min(max(float(background_floor), 0.0), 1.0)
    weight = valid * (floor + (1.0 - floor) * recovery)
    difference = F.smooth_l1_loss(
        student_feature, teacher_feature, reduction='none', beta=0.1)
    difference = difference.mean(dim=1, keepdim=True)
    return (difference * weight).sum() / weight.sum().clamp_min(1.0)


def dg_agent_alignment_loss(weather_encoded, teacher_encoded):
    """V2X-DGW-style sparse per-agent alignment with a frozen clean anchor."""
    weather_feature = weather_encoded['spatial_features']
    clean_feature = teacher_encoded['spatial_features'].detach()
    if weather_feature.shape != clean_feature.shape:
        raise ValueError(
            'DG agent feature shapes differ: %s vs %s' %
            (tuple(weather_feature.shape), tuple(clean_feature.shape)))
    occupied = weather_feature.detach().abs().amax(
        dim=1, keepdim=True).gt(0).to(weather_feature.dtype)
    difference = F.smooth_l1_loss(
        weather_feature, clean_feature,
        reduction='none', beta=0.1).mean(dim=1, keepdim=True)
    return (difference * occupied).sum() / occupied.sum().clamp_min(1.0)


def dg_fusion_alignment_loss(
        weather_out, teacher_out, background_floor=0.05):
    """Align fused weather semantics to the frozen clean teacher."""
    weather_feature = weather_out['fused_feature']
    clean_feature = teacher_out['fused_feature'].detach()
    if weather_feature.shape != clean_feature.shape:
        raise ValueError(
            'DG fused feature shapes differ: %s vs %s' %
            (tuple(weather_feature.shape), tuple(clean_feature.shape)))
    confidence = teacher_out['psm'].detach().sigmoid().amax(
        dim=1, keepdim=True)
    if confidence.shape[-2:] != weather_feature.shape[-2:]:
        confidence = F.interpolate(
            confidence, size=weather_feature.shape[-2:],
            mode='bilinear', align_corners=False)
    floor = min(max(float(background_floor), 0.0), 1.0)
    weight = floor + (1.0 - floor) * confidence
    difference = F.smooth_l1_loss(
        weather_feature, clean_feature,
        reduction='none', beta=0.1).mean(dim=1, keepdim=True)
    return (difference * weight).sum() / weight.sum().clamp_min(1.0)


def dg_paired_agent_contrastive_loss(
        weather_encoded, teacher_encoded, temperature=0.2, pool_size=4):
    """Use each clean/weather CAV pair as positives and other CAVs as negatives."""
    weather_feature = weather_encoded[
        'spatial_features_2d_before_compression']
    clean_feature = teacher_encoded[
        'spatial_features_2d_before_compression'].detach()
    if weather_feature.shape != clean_feature.shape:
        raise ValueError(
            'DG contrastive feature shapes differ: %s vs %s' %
            (tuple(weather_feature.shape), tuple(clean_feature.shape)))
    pool_size = max(int(pool_size), 1)
    weather_embedding = F.adaptive_avg_pool2d(
        weather_feature, (pool_size, pool_size)).flatten(1)
    clean_embedding = F.adaptive_avg_pool2d(
        clean_feature, (pool_size, pool_size)).flatten(1)
    weather_embedding = F.normalize(weather_embedding, dim=1)
    clean_embedding = F.normalize(clean_embedding, dim=1)
    if weather_embedding.shape[0] <= 1:
        return (
            1.0 - (weather_embedding * clean_embedding).sum(dim=1)
        ).mean()
    temperature = max(float(temperature), 1.0e-4)
    logits = torch.matmul(
        weather_embedding, clean_embedding.t()) / temperature
    labels = torch.arange(
        logits.shape[0], device=logits.device, dtype=torch.long)
    return 0.5 * (
        F.cross_entropy(logits, labels) +
        F.cross_entropy(logits.t(), labels))


def counterfactual_anchor_map(teacher_out, teacher_ego_out, opt):
    """Measure positive-anchor benefit from clean collaborative fusion."""
    if teacher_ego_out is None:
        raise ValueError(
            'teacher_ego_out is required for counterfactual KD')
    temperature = max(float(opt.temperature), 1.0e-4)
    all_probability = torch.sigmoid(
        teacher_out['psm'].detach() / temperature)
    ego_probability = torch.sigmoid(
        teacher_ego_out['psm'].detach() / temperature)
    if all_probability.shape != ego_probability.shape:
        raise ValueError(
            'All/Ego teacher PSM shapes differ: %s vs %s' %
            (tuple(all_probability.shape), tuple(ego_probability.shape)))
    gain = (
        all_probability - ego_probability -
        float(opt.counterfactual_min_gain)).clamp_min(0.0)
    scale = max(float(opt.counterfactual_gain_scale), 1.0e-4)
    return (gain / scale).clamp_max(1.0)


def counterfactual_gain_distillation_loss(
        student_out, student_ego_out, teacher_out, teacher_ego_out,
        label_dict, opt):
    """Match the collaboration-induced probability improvement itself."""
    if student_ego_out is None or teacher_ego_out is None:
        raise ValueError(
            'Counterfactual gain loss requires student and teacher '
            'Ego-only outputs')
    positive = positive_anchor_mask(label_dict, student_out['psm'])
    temperature = max(float(opt.temperature), 1.0e-4)
    teacher_all_probability = torch.sigmoid(
        teacher_out['psm'].detach() / temperature)
    teacher_ego_probability = torch.sigmoid(
        teacher_ego_out['psm'].detach() / temperature)
    student_all_probability = torch.sigmoid(
        student_out['psm'] / temperature)
    student_ego_probability = torch.sigmoid(
        student_ego_out['psm'] / temperature)
    if opt.teacher_positive_threshold > 0:
        positive = positive * (
            teacher_all_probability >= opt.teacher_positive_threshold
        ).to(positive.dtype)

    scale = max(float(opt.counterfactual_gain_scale), 1.0e-4)
    teacher_gain = (
        teacher_all_probability - teacher_ego_probability -
        float(opt.counterfactual_min_gain)).clamp_min(0.0)
    teacher_gain = (teacher_gain / scale).clamp_max(1.0)
    student_gain = (
        student_all_probability - student_ego_probability) / scale
    gain_weight = positive * teacher_gain
    weight_sum = gain_weight.sum().clamp_min(1.0)
    loss_map = F.smooth_l1_loss(
        student_gain, teacher_gain, reduction='none',
        beta=max(float(opt.counterfactual_gain_beta), 1.0e-4))
    loss = (loss_map * gain_weight).sum() / weight_sum
    active_count = int(
        (gain_weight > 0).sum().detach().item())
    return loss, active_count


def feature_restoration_loss(student_encoded, teacher_encoded, opt):
    """Match per-agent semantic BEV features with foreground emphasis."""
    student_feature = student_encoded['spatial_features_2d']
    teacher_feature = teacher_encoded['spatial_features_2d'].detach()
    if student_feature.shape != teacher_feature.shape:
        raise ValueError(
            'Student/teacher BEV feature shapes differ: %s vs %s' %
            (tuple(student_feature.shape), tuple(teacher_feature.shape)))
    teacher_confidence = teacher_encoded['psm_single'].detach().sigmoid()
    teacher_confidence = teacher_confidence.amax(dim=1, keepdim=True)
    if teacher_confidence.shape[-2:] != student_feature.shape[-2:]:
        teacher_confidence = F.interpolate(
            teacher_confidence, size=student_feature.shape[-2:],
            mode='bilinear', align_corners=False)
    floor = min(max(float(opt.feature_background_floor), 0.0), 1.0)
    spatial_weight = floor + (1.0 - floor) * teacher_confidence
    loss_map = F.smooth_l1_loss(
        student_feature, teacher_feature, reduction='none', beta=0.1)
    loss_map = loss_map.mean(dim=1, keepdim=True)
    return (loss_map * spatial_weight).sum() / \
        spatial_weight.sum().clamp_min(1.0)


def prediction_distillation_losses(student_out, teacher_out, label_dict,
                                   opt, batch_ego=None,
                                   teacher_ego_out=None):
    student_psm = student_out['psm']
    teacher_psm = teacher_out['psm'].detach()
    positive = positive_anchor_mask(label_dict, student_psm)
    temperature = max(float(opt.temperature), 1.0e-4)
    teacher_probability = torch.sigmoid(teacher_psm / temperature)
    if opt.teacher_positive_threshold > 0:
        positive = positive * (
            teacher_probability >= opt.teacher_positive_threshold
        ).to(positive.dtype)
    positive_count = positive.sum().clamp_min(1.0)
    recoverability = torch.zeros_like(positive)
    if opt.lambda_recoverability_kd > 0:
        if batch_ego is None:
            raise ValueError(
                'batch_ego is required when recoverability KD is enabled')
        recoverability = recoverability_anchor_map(
            batch_ego, label_dict, student_psm, opt)
    counterfactual = torch.zeros_like(positive)
    if opt.lambda_counterfactual_kd > 0:
        counterfactual = counterfactual_anchor_map(
            teacher_out, teacher_ego_out, opt)
    kd_weight = positive * (
        1.0 + float(opt.lambda_recoverability_kd) * recoverability +
        float(opt.lambda_counterfactual_kd) * counterfactual)
    kd_weight_sum = kd_weight.sum().clamp_min(1.0)

    cls_map = F.binary_cross_entropy_with_logits(
        student_psm / temperature,
        teacher_probability,
        reduction='none') * (temperature ** 2)
    cls_loss = (cls_map * kd_weight).sum() / kd_weight_sum

    batch, anchor_count, height, width = student_psm.shape
    student_rm = student_out['rm'].view(
        batch, anchor_count, 7, height, width)
    teacher_rm = teacher_out['rm'].detach().view(
        batch, anchor_count, 7, height, width)
    reg_map = F.smooth_l1_loss(
        student_rm, teacher_rm, reduction='none', beta=opt.reg_beta)
    reg_mask = kd_weight.unsqueeze(2)
    reg_loss = (reg_map * reg_mask).sum() / (
        kd_weight_sum * student_rm.shape[2])
    active_recoverability = recoverability[positive > 0]
    mean_recoverability = (
        float(active_recoverability.mean().detach().item())
        if active_recoverability.numel() > 0 else 0.0)
    focused_count = int(
        (active_recoverability >= 0.5).sum().detach().item())
    active_counterfactual = counterfactual[positive > 0]
    mean_counterfactual = (
        float(active_counterfactual.mean().detach().item())
        if active_counterfactual.numel() > 0 else 0.0)
    counterfactual_count = int(
        (active_counterfactual > 0).sum().detach().item())
    return (cls_loss, reg_loss,
            int(positive.sum().detach().item()),
            mean_recoverability, focused_count,
            mean_counterfactual, counterfactual_count)


def _gaussian_smooth(confidence, kernel_size, sigma):
    center = kernel_size // 2
    coordinates = torch.arange(
        kernel_size, device=confidence.device,
        dtype=confidence.dtype) - center
    yy, xx = torch.meshgrid(coordinates, coordinates, indexing='ij')
    kernel = (1.0 / (2.0 * math.pi * sigma)) * torch.exp(
        -(xx.square() + yy.square()) / (2.0 * sigma * sigma))
    kernel = kernel.view(1, 1, kernel_size, kernel_size)
    return F.conv2d(
        confidence, kernel, padding=(kernel_size - 1) // 2)


def communication_preservation_loss(student_out, teacher_out, hypes, opt):
    """Match the threshold-sensitive mask used by Where2comm at inference."""
    student_logits = student_out['single_confidence']
    teacher_logits = teacher_out['single_confidence'].detach()
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            'Communication confidence shape mismatch: %s vs %s' %
            (tuple(student_logits.shape), tuple(teacher_logits.shape)))

    student_confidence = student_logits.sigmoid().amax(
        dim=1, keepdim=True)
    teacher_confidence = teacher_logits.sigmoid().amax(
        dim=1, keepdim=True)
    communication_cfg = hypes['model']['args'][
        'where2comm_fusion']['communication']
    smooth_cfg = communication_cfg.get('gaussian_smooth', None)
    if smooth_cfg:
        kernel_size = int(smooth_cfg['k_size'])
        sigma = float(smooth_cfg['c_sigma'])
        student_confidence = _gaussian_smooth(
            student_confidence, kernel_size, sigma)
        teacher_confidence = _gaussian_smooth(
            teacher_confidence, kernel_size, sigma)

    threshold = float(communication_cfg.get('threshold', 0.0))
    temperature = max(float(opt.comm_mask_temperature), 1.0e-5)
    if threshold > 0:
        student_soft_mask = torch.sigmoid(
            (student_confidence - threshold) / temperature)
        teacher_soft_mask = torch.sigmoid(
            (teacher_confidence - threshold) / temperature)
    else:
        student_soft_mask = student_confidence
        teacher_soft_mask = teacher_confidence
    return F.mse_loss(student_soft_mask, teacher_soft_mask)


def snow_voxel_denoising_losses(clean_encoded, weather_encoded, opt):
    """Supervise snow filtering from paired clean/weather occupancy."""
    clean_aux = clean_encoded.get('snow_voxel_aux', None)
    weather_aux = weather_encoded.get('snow_voxel_aux', None)
    if clean_aux is None or weather_aux is None:
        raise ValueError(
            'snow voxel loss requires snow_voxel_aux from both branches')

    clean_logits = clean_aux['noise_logit']
    weather_logits = weather_aux['noise_logit']
    novelty = snow_voxel_novelty_labels(
        clean_encoded['voxel_coords'],
        weather_encoded['voxel_coords']).to(
            device=weather_logits.device, dtype=weather_logits.dtype)
    if novelty.shape != weather_logits.shape:
        raise ValueError(
            'weather novelty/logit shapes differ: %s vs %s' %
            (tuple(novelty.shape), tuple(weather_logits.shape)))

    positive_fraction = novelty.mean().detach().clamp_min(1.0e-6)
    # Balance both weather negatives and the separately weighted clean
    # identity branch. This prevents the trivial all-normal solution when
    # snow-created voxels are only a few percent of occupied pillars.
    clean_to_weather = (
        max(float(opt.lambda_snow_voxel_clean), 0.0) /
        max(float(opt.lambda_snow_voxel_denoise), 1.0e-6))
    balanced_weight = (
        (1.0 - positive_fraction) + clean_to_weather) / positive_fraction
    minimum_weight = weather_logits.new_tensor(
        max(float(opt.snow_voxel_positive_weight), 1.0))
    maximum_weight = weather_logits.new_tensor(
        max(float(opt.snow_voxel_max_positive_weight),
            float(minimum_weight.item())))
    positive_weight = torch.maximum(
        balanced_weight, minimum_weight).clamp_max(maximum_weight)
    weather_loss = F.binary_cross_entropy_with_logits(
        weather_logits, novelty, pos_weight=positive_weight)
    clean_loss = F.binary_cross_entropy_with_logits(
        clean_logits, torch.zeros_like(clean_logits))

    with torch.no_grad():
        novelty_ratio = float(novelty.mean().item()) \
            if novelty.numel() else 0.0
        keep_weight = weather_aux['keep_weight']
        mean_keep = float(keep_weight.mean().item()) \
            if keep_weight.numel() else 1.0
        suppressed_ratio = float(
            (keep_weight < 0.999).float().mean().item()) \
            if keep_weight.numel() else 0.0
        probability = weather_aux['noise_probability']
        positive_mask = novelty > 0.5
        negative_mask = ~positive_mask
        positive_probability = float(
            probability[positive_mask].mean().item()) \
            if positive_mask.any() else 0.0
        negative_probability = float(
            probability[negative_mask].mean().item()) \
            if negative_mask.any() else 0.0
    return (
        weather_loss, clean_loss, novelty_ratio,
        mean_keep, suppressed_ratio, float(positive_weight.item()),
        positive_probability, negative_probability)


def load_inputs(batch_data, hypes, opt, device):
    augmentation_stats = batch_data['ego'].pop(
        'weather_augmentation_stats', None)
    batch_data = train_utils.to_device(batch_data, device)
    inputs = build_teacher_student_inputs(batch_data['ego'], hypes, opt)
    for model_input in inputs:
        model_input['return_aux'] = True
    return batch_data, inputs, augmentation_stats


def evaluate_student(student, loader, criterion, hypes, opt, device):
    clean_losses = []
    weather_losses = []
    clean_communication = []
    weather_communication = []
    student.eval()
    with torch.no_grad():
        for index, batch_data in enumerate(loader):
            if opt.max_val_steps > 0 and index >= opt.max_val_steps:
                break
            batch_data, inputs, _ = load_inputs(
                batch_data, hypes, opt, device)
            _, clean_ego, weather_ego = inputs
            labels = batch_data['ego']['label_dict']
            clean_output = student(clean_ego)
            weather_output = student(weather_ego)
            clean_losses.append(criterion(clean_output, labels).item())
            weather_losses.append(criterion(weather_output, labels).item())
            if clean_output.get('com', None) is not None:
                clean_communication.append(
                    float(clean_output['com'].detach().cpu()))
            if weather_output.get('com', None) is not None:
                weather_communication.append(
                    float(weather_output['com'].detach().cpu()))
    mean = lambda values: (
        statistics.mean(values) if values else float('nan'))
    return {
        'clean_det': mean(clean_losses),
        'weather_det': mean(weather_losses),
        'clean_com': mean(clean_communication),
        'weather_com': mean(weather_communication)
    }


def main():
    opt = parse_args()
    seed_everything(opt.seed)
    hypes, config_path = setup_hypes(opt)
    print('-----------------Dataset Building------------------')
    train_dataset = build_dataset(hypes, visualize=False, train=True)
    val_dataset = build_dataset(hypes, visualize=False, train=False)
    shuffle_generator = torch.Generator()
    shuffle_generator.manual_seed(opt.seed)
    train_loader = DataLoader(
        train_dataset, batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        collate_fn=train_dataset.collate_batch_train,
        shuffle=True, pin_memory=False, drop_last=True,
        generator=shuffle_generator)
    val_loader = DataLoader(
        val_dataset, batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        collate_fn=val_dataset.collate_batch_train,
        shuffle=False, pin_memory=False, drop_last=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    teacher_dir = opt.teacher_model_dir or opt.model_dir
    print('---------------Creating Teacher/Student------------------')
    teacher = load_model_from_dir(
        hypes, teacher_dir, device, trainable=False)
    student = load_model_from_dir(
        hypes, opt.model_dir, device, trainable=True)
    cure_enabled = bool(getattr(student, 'cure_enabled', False))
    snow_denoiser_enabled = bool(
        getattr(student, 'snow_voxel_denoiser_enabled', False))
    dg_enabled = any(value > 0 for value in [
        opt.lambda_dg_agent_align,
        opt.lambda_dg_fusion_align,
        opt.lambda_dg_agent_contrast,
    ])
    # The frozen teacher supplies counterfactual targets and must remain the
    # exact baseline even when the student enables an inference-time module.
    if hasattr(teacher, 'cure_apply_fusion_gate'):
        teacher.cure_apply_fusion_gate = False
    if hasattr(teacher, 'snow_voxel_denoiser_enabled'):
        teacher.snow_voxel_denoiser_enabled = False
    if opt.adapter_only and opt.snow_denoiser_only:
        raise ValueError(
            '--adapter_only and --snow_denoiser_only are mutually exclusive')
    if snow_denoiser_enabled and not opt.snow_denoiser_only:
        raise ValueError(
            'Snow voxel denoising must use --snow_denoiser_only so the '
            'pretrained detector remains frozen.')
    if opt.snow_denoiser_only:
        if not snow_denoiser_enabled:
            raise ValueError(
                '--snow_denoiser_only requires a non-zero snow voxel loss')
        for parameter in student.parameters():
            parameter.requires_grad = False
        for parameter in student.snow_voxel_denoiser.parameters():
            parameter.requires_grad = True
        trainable_count = sum(
            parameter.numel() for parameter in student.parameters()
            if parameter.requires_grad)
        print('Snow-denoiser-only trainable parameters: %d' %
              trainable_count)
    if opt.adapter_only:
        if not getattr(student, 'weather_adapter_enabled', False) and \
                not cure_enabled:
            raise ValueError(
                '--adapter_only requires a weather adapter or CURE loss')
        for parameter in student.parameters():
            parameter.requires_grad = False
        if cure_enabled:
            cure_module = (
                student.counterfactual_fusion_restorer
                if opt.cure_location == 'post_fusion'
                else student.counterfactual_utility_restorer)
            for parameter in cure_module.parameters():
                parameter.requires_grad = True
        else:
            for parameter in student.weather_feature_adapter.parameters():
                parameter.requires_grad = True
        trainable_count = sum(
            parameter.numel() for parameter in student.parameters()
            if parameter.requires_grad)
        module_name = 'CURE-only' if cure_enabled else 'Adapter-only'
        print('%s trainable parameters: %d' % (
            module_name, trainable_count))
    if not opt.train_batchnorm:
        print('Frozen BatchNorm layers: %d' % freeze_batchnorm(student))

    criterion = train_utils.create_loss(hypes)
    optimizer = train_utils.setup_optimizer(hypes, student)
    scheduler = train_utils.setup_lr_schedular(
        hypes, optimizer, len(train_loader))
    saved_path = train_utils.setup_train(hypes)
    writer = SummaryWriter(saved_path)
    repo_root = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', '..'))
    teacher_checkpoint = checkpoint_path(teacher_dir, 0)
    write_manifest(os.path.join(saved_path, 'training_manifest.json'), {
        'method': 'positive_anchor_prediction_distillation',
        'repo_revision': git_revision(repo_root),
        'source_config': os.path.abspath(config_path),
        'source_config_sha256': sha256_file(config_path),
        'teacher_model_dir': os.path.abspath(teacher_dir),
        'teacher_checkpoint': (
            os.path.abspath(teacher_checkpoint)
            if teacher_checkpoint else None),
        'teacher_checkpoint_sha256': sha256_file(teacher_checkpoint),
        'student_init_model_dir': os.path.abspath(opt.model_dir),
        'root_dir': hypes.get('root_dir'),
        'validate_dir': hypes.get('validate_dir'),
        'weather_augmentation': hypes.get('weather_augmentation', {}),
        'arguments': vars(opt)
    })

    augmentation_mode = hypes['weather_augmentation']['mode']
    print('Prediction distillation start')
    print('Weather augmentation mode: %s' % augmentation_mode)
    for epoch in range(opt.epochs):
        if hasattr(train_dataset, 'set_weather_augmentation_epoch'):
            train_dataset.set_weather_augmentation_epoch(epoch)
        if hypes['lr_scheduler']['core_method'] != 'cosineannealwarm':
            scheduler.step(epoch)
        else:
            scheduler.step_update(epoch * len(train_loader))
        for group in optimizer.param_groups:
            print('learning rate %.7f' % group['lr'])

        mixed_weather_counts = {
            'physics_rain': 0,
            'v2x_dgw_awa': 0,
        }
        fog_alpha_counts = {}
        snowfall_rates = []
        progress = tqdm.tqdm(total=len(train_loader), leave=True)
        for index, batch_data in enumerate(train_loader):
            if opt.max_train_steps > 0 and index >= opt.max_train_steps:
                break
            student.train()
            if not opt.train_batchnorm:
                keep_batchnorm_eval(student)
            optimizer.zero_grad(set_to_none=True)
            batch_data, inputs, augmentation_stats = load_inputs(
                batch_data, hypes, opt, device)
            if augmentation_mode == 'mixed_weather' and augmentation_stats:
                for scene_stats in augmentation_stats:
                    if not scene_stats:
                        continue
                    scene_mode = scene_stats[0].get('augmentation_mode')
                    if scene_mode not in mixed_weather_counts:
                        raise AssertionError(
                            'unexpected mixed weather mode: %s' % scene_mode)
                    if any(
                            stats.get('augmentation_mode') != scene_mode
                            for stats in scene_stats):
                        raise AssertionError(
                            'agents in one scene received different weather')
                    mixed_weather_counts[scene_mode] += 1
            if augmentation_mode == 'physics_fog' and augmentation_stats:
                for scene_stats in augmentation_stats:
                    if not scene_stats:
                        continue
                    scene_alphas = {
                        float(stats['fog_alpha'])
                        for stats in scene_stats
                    }
                    if len(scene_alphas) != 1:
                        raise AssertionError(
                            'agents in one scene received different fog alpha')
                    scene_alpha = scene_alphas.pop()
                    fog_alpha_counts[scene_alpha] = \
                        fog_alpha_counts.get(scene_alpha, 0) + 1
            if augmentation_mode == 'physics_snow' and augmentation_stats:
                for scene_stats in augmentation_stats:
                    if not scene_stats:
                        continue
                    scene_rates = {
                        float(stats['snowfall_rate'])
                        for stats in scene_stats
                    }
                    if len(scene_rates) != 1:
                        raise AssertionError(
                            'agents in one scene received different '
                            'snowfall rates')
                    snowfall_rates.append(scene_rates.pop())
            if index == 0 and augmentation_mode in [
                    'v2x_dgw_awa', 'pre_voxel_matched_dropout',
                    'physics_rain', 'physics_fog', 'physics_snow',
                    'mixed_weather']:
                print_weather_augmentation_stats(augmentation_stats)
            teacher_ego, clean_ego, weather_ego = inputs
            ego_mask = None
            with torch.no_grad():
                if opt.lambda_counterfactual_kd > 0 or \
                        opt.lambda_counterfactual_gain > 0 or \
                        opt.lambda_weather_feature > 0 or \
                        opt.lambda_clean_feature > 0 or cure_enabled or \
                        dg_enabled:
                    teacher_encoded = teacher.encode_features(teacher_ego)
                    teacher_out = teacher.forward_from_encoded(
                        teacher_encoded, return_aux=True)
                    if opt.lambda_counterfactual_kd > 0 or \
                            opt.lambda_counterfactual_gain > 0:
                        ego_mask = ego_only_agent_mask(
                            teacher_ego['record_len'], device)
                        teacher_ego_out = teacher.forward_from_encoded(
                            teacher_encoded, agent_mask=ego_mask,
                            return_aux=False)
                    else:
                        teacher_ego_out = None
                    if cure_enabled:
                        teacher_weather_encoded = teacher.encode_features(
                            weather_ego)
                        teacher_weather_out = teacher.forward_from_encoded(
                            teacher_weather_encoded, return_aux=True)
                    else:
                        teacher_weather_encoded = None
                        teacher_weather_out = None
                else:
                    teacher_out = teacher(teacher_ego)
                    teacher_ego_out = None
                    teacher_encoded = None
                    teacher_weather_encoded = None
                    teacher_weather_out = None
            if opt.lambda_clean_feature > 0 or snow_denoiser_enabled or \
                    (cure_enabled and opt.lambda_cure_identity > 0):
                clean_encoded = student.encode_features(clean_ego)
                clean_out = student.forward_from_encoded(
                    clean_encoded, return_aux=True)
            else:
                clean_out = student(clean_ego)
                clean_encoded = None
            if opt.lambda_counterfactual_gain > 0 or \
                    opt.lambda_weather_feature > 0 or cure_enabled or \
                    snow_denoiser_enabled or dg_enabled:
                weather_encoded = student.encode_features(weather_ego)
                weather_out = student.forward_from_encoded(
                    weather_encoded, return_aux=True)
                if opt.lambda_counterfactual_gain > 0:
                    student_ego_out = student.forward_from_encoded(
                        weather_encoded, agent_mask=ego_mask,
                        return_aux=False)
                else:
                    student_ego_out = None
            else:
                weather_out = student(weather_ego)
                student_ego_out = None
                weather_encoded = None
            labels = batch_data['ego']['label_dict']

            weather_det = criterion(weather_out, labels)
            clean_det = criterion(clean_out, labels)
            pred_cls, pred_reg, positive_count, recoverability_mean, \
                recoverability_count, counterfactual_mean, \
                counterfactual_count = \
                prediction_distillation_losses(
                    weather_out, teacher_out, labels, opt,
                    batch_data['ego'], teacher_ego_out)
            clean_comm = communication_preservation_loss(
                clean_out, teacher_out, hypes, opt)
            weather_comm = communication_preservation_loss(
                weather_out, teacher_out, hypes, opt)
            if opt.lambda_counterfactual_gain > 0:
                counterfactual_gain, counterfactual_gain_count = \
                    counterfactual_gain_distillation_loss(
                        weather_out, student_ego_out,
                        teacher_out, teacher_ego_out,
                        labels, opt)
            else:
                counterfactual_gain = weather_det.new_zeros(())
                counterfactual_gain_count = 0
            if opt.lambda_weather_feature > 0:
                weather_feature = feature_restoration_loss(
                    weather_encoded, teacher_encoded, opt)
            else:
                weather_feature = weather_det.new_zeros(())
            if opt.lambda_clean_feature > 0:
                clean_feature = feature_restoration_loss(
                    clean_encoded, teacher_encoded, opt)
            else:
                clean_feature = weather_det.new_zeros(())
            if opt.lambda_dg_agent_align > 0:
                dg_agent_align = dg_agent_alignment_loss(
                    weather_encoded, teacher_encoded)
            else:
                dg_agent_align = weather_det.new_zeros(())
            if opt.lambda_dg_fusion_align > 0:
                dg_fusion_align = dg_fusion_alignment_loss(
                    weather_out, teacher_out, opt.dg_background_floor)
            else:
                dg_fusion_align = weather_det.new_zeros(())
            if opt.lambda_dg_agent_contrast > 0:
                dg_agent_contrast = dg_paired_agent_contrastive_loss(
                    weather_encoded, teacher_encoded,
                    opt.dg_contrast_temperature, opt.dg_pool_size)
            else:
                dg_agent_contrast = weather_det.new_zeros(())
            if snow_denoiser_enabled:
                snow_voxel_weather, snow_voxel_clean, \
                    snow_voxel_novelty_ratio, snow_voxel_mean_keep, \
                    snow_voxel_suppressed_ratio, \
                    snow_voxel_effective_positive_weight, \
                    snow_voxel_positive_probability, \
                    snow_voxel_negative_probability = \
                    snow_voxel_denoising_losses(
                        clean_encoded, weather_encoded, opt)
            else:
                snow_voxel_weather = weather_det.new_zeros(())
                snow_voxel_clean = weather_det.new_zeros(())
                snow_voxel_novelty_ratio = 0.0
                snow_voxel_mean_keep = 1.0
                snow_voxel_suppressed_ratio = 0.0
                snow_voxel_effective_positive_weight = 1.0
                snow_voxel_positive_probability = 0.0
                snow_voxel_negative_probability = 0.0
            if cure_enabled:
                with torch.no_grad():
                    cure_targets = counterfactual_utility_targets(
                        teacher, teacher_encoded,
                        teacher_weather_encoded, teacher_out,
                        teacher_weather_out, labels,
                        weather_ego['record_len'], opt)
                if opt.cure_location == 'post_fusion':
                    cure_targets = aggregate_cure_targets(
                        cure_targets, weather_ego['record_len'])
                    cure_prediction_source = weather_out
                else:
                    cure_prediction_source = weather_encoded
                cure_recovery, cure_harm, cure_utility = \
                    cure_supervision_losses(
                        cure_prediction_source, cure_targets, opt)
                if opt.lambda_cure_feature > 0:
                    if opt.cure_location == 'post_fusion':
                        cure_feature = \
                            cure_fused_feature_restoration_loss(
                                weather_out, teacher_out, cure_targets,
                                opt.feature_background_floor)
                    else:
                        cure_feature = cure_feature_restoration_loss(
                            weather_encoded, teacher_encoded, cure_targets,
                            opt.feature_background_floor)
                else:
                    cure_feature = weather_det.new_zeros(())
                if opt.lambda_cure_identity > 0:
                    if opt.cure_location == 'post_fusion':
                        cure_identity = cure_fused_identity_loss(
                            clean_out, teacher_out,
                            opt.feature_background_floor)
                    else:
                        cure_identity = feature_restoration_loss(
                            clean_encoded, teacher_encoded, opt)
                else:
                    cure_identity = weather_det.new_zeros(())
            else:
                cure_targets = {
                    'neighbor_count': 0,
                    'valid_count': 0,
                    'recovery': weather_det.new_zeros((1,)),
                    'harm': weather_det.new_zeros((1,)),
                }
                cure_recovery = weather_det.new_zeros(())
                cure_harm = weather_det.new_zeros(())
                cure_utility = weather_det.new_zeros(())
                cure_feature = weather_det.new_zeros(())
                cure_identity = weather_det.new_zeros(())
            if cure_enabled and cure_targets['valid_count'] > 0:
                cure_valid = cure_targets['valid']
                cure_recovery_target_mean = float(
                    cure_targets['recovery'][cure_valid].mean().item())
                cure_harm_target_ratio = float(
                    cure_targets['harm'][cure_valid].mean().item())
                if index == 0:
                    clean_values = cure_targets[
                        'clean_utility'][cure_valid]
                    weather_values = cure_targets[
                        'weather_utility'][cure_valid]
                    recovery_values = cure_targets[
                        'recovery'][cure_valid]
                    print(
                        '\nCURE target audit: neighbors=%d valid_cells=%d '
                        'clean_u=%.6f[%.6f,%.6f] '
                        'weather_u=%.6f[%.6f,%.6f] '
                        'recovery_mean=%.6f recovery_pos=%.4f '
                        'harm_ratio=%.4f' %
                        (cure_targets['neighbor_count'],
                         cure_targets['valid_count'],
                         clean_values.mean().item(),
                         clean_values.min().item(),
                         clean_values.max().item(),
                         weather_values.mean().item(),
                         weather_values.min().item(),
                         weather_values.max().item(),
                         recovery_values.mean().item(),
                         (recovery_values > 0.01).float().mean().item(),
                         cure_harm_target_ratio))
            else:
                cure_recovery_target_mean = 0.0
                cure_harm_target_ratio = 0.0
            total = (
                opt.lambda_weather_det * weather_det +
                opt.lambda_clean_det * clean_det +
                opt.lambda_pred_cls * pred_cls +
                opt.lambda_pred_reg * pred_reg +
                opt.lambda_clean_comm * clean_comm +
                opt.lambda_weather_comm * weather_comm +
                opt.lambda_counterfactual_gain * counterfactual_gain +
                opt.lambda_weather_feature * weather_feature +
                opt.lambda_clean_feature * clean_feature +
                opt.lambda_dg_agent_align * dg_agent_align +
                opt.lambda_dg_fusion_align * dg_fusion_align +
                opt.lambda_dg_agent_contrast * dg_agent_contrast +
                opt.lambda_cure_recovery * cure_recovery +
                opt.lambda_cure_harm * cure_harm +
                opt.lambda_cure_utility * cure_utility +
                opt.lambda_cure_feature * cure_feature +
                opt.lambda_cure_identity * cure_identity +
                opt.lambda_snow_voxel_denoise * snow_voxel_weather +
                opt.lambda_snow_voxel_clean * snow_voxel_clean)
            total.backward()
            optimizer.step()
            completed_step = epoch * len(train_loader) + index + 1
            if opt.save_every_steps > 0 and \
                    completed_step % opt.save_every_steps == 0:
                step_path = os.path.join(
                    saved_path, 'net_step%d.pth' % completed_step)
                torch.save(student.state_dict(), step_path)
                print('\nSaved intermediate checkpoint to %s' % step_path)
            if hypes['lr_scheduler']['core_method'] == 'cosineannealwarm':
                scheduler.step_update(epoch * len(train_loader) + index)

            step = epoch * len(train_loader) + index
            if index % 10 == 0:
                writer.add_scalar(
                    'Train/weather_det', weather_det.item(), step)
                writer.add_scalar(
                    'Train/clean_det', clean_det.item(), step)
                writer.add_scalar(
                    'Train/pred_cls_kd', pred_cls.item(), step)
                writer.add_scalar(
                    'Train/pred_reg_kd', pred_reg.item(), step)
                writer.add_scalar(
                    'Train/clean_comm_preserve', clean_comm.item(), step)
                writer.add_scalar(
                    'Train/weather_comm_preserve', weather_comm.item(), step)
                writer.add_scalar('Train/total', total.item(), step)
                writer.add_scalar(
                    'Train/positive_anchors', positive_count, step)
                writer.add_scalar(
                    'Train/recoverability_mean', recoverability_mean, step)
                writer.add_scalar(
                    'Train/recoverability_anchors',
                    recoverability_count, step)
                writer.add_scalar(
                    'Train/counterfactual_mean',
                    counterfactual_mean, step)
                writer.add_scalar(
                    'Train/counterfactual_anchors',
                    counterfactual_count, step)
                writer.add_scalar(
                    'Train/counterfactual_gain_loss',
                    counterfactual_gain.item(), step)
                writer.add_scalar(
                    'Train/counterfactual_gain_anchors',
                    counterfactual_gain_count, step)
                writer.add_scalar(
                    'Train/weather_feature_restore',
                    weather_feature.item(), step)
                writer.add_scalar(
                    'Train/clean_feature_identity',
                    clean_feature.item(), step)
                writer.add_scalar(
                    'Train/dg_agent_align',
                    dg_agent_align.item(), step)
                writer.add_scalar(
                    'Train/dg_fusion_align',
                    dg_fusion_align.item(), step)
                writer.add_scalar(
                    'Train/dg_agent_contrast',
                    dg_agent_contrast.item(), step)
                writer.add_scalar(
                    'Train/cure_recovery_loss',
                    cure_recovery.item(), step)
                writer.add_scalar(
                    'Train/cure_harm_loss',
                    cure_harm.item(), step)
                writer.add_scalar(
                    'Train/cure_utility_loss',
                    cure_utility.item(), step)
                writer.add_scalar(
                    'Train/cure_feature_loss',
                    cure_feature.item(), step)
                writer.add_scalar(
                    'Train/cure_identity_loss',
                    cure_identity.item(), step)
                writer.add_scalar(
                    'Train/cure_recovery_target_mean',
                    cure_recovery_target_mean, step)
                writer.add_scalar(
                    'Train/cure_harm_target_ratio',
                    cure_harm_target_ratio, step)
                writer.add_scalar(
                    'Train/snow_voxel_weather_loss',
                    snow_voxel_weather.item(), step)
                writer.add_scalar(
                    'Train/snow_voxel_clean_loss',
                    snow_voxel_clean.item(), step)
                writer.add_scalar(
                    'Train/snow_voxel_novelty_ratio',
                    snow_voxel_novelty_ratio, step)
                writer.add_scalar(
                    'Train/snow_voxel_mean_keep',
                    snow_voxel_mean_keep, step)
                writer.add_scalar(
                    'Train/snow_voxel_suppressed_ratio',
                    snow_voxel_suppressed_ratio, step)
                writer.add_scalar(
                    'Train/snow_voxel_effective_positive_weight',
                    snow_voxel_effective_positive_weight, step)
                writer.add_scalar(
                    'Train/snow_voxel_positive_probability',
                    snow_voxel_positive_probability, step)
                writer.add_scalar(
                    'Train/snow_voxel_negative_probability',
                    snow_voxel_negative_probability, step)
            progress.set_description(
                'weather %.4f clean %.4f kd_cls %.4f kd_reg %.4f '
                'comm_c %.6f comm_w %.6f pos %d rec %.3f/%d '
                'cf %.3f/%d cfg %.4f/%d feat %.4f/%.4f '
                'dg %.4f/%.4f/%.4f '
                'cure %.3f/%.3f/%.3f ft %.4f id %.4f '
                'target %.3f/%.3f n %d snow %.3f/%.3f '
                'novel %.3f pw %.1f p+ %.3f p- %.3f '
                'keep %.3f sup %.3f total %.4f' %
                (weather_det.item(), clean_det.item(), pred_cls.item(),
                 pred_reg.item(), clean_comm.item(), weather_comm.item(),
                 positive_count, recoverability_mean,
                 recoverability_count, counterfactual_mean,
                 counterfactual_count, counterfactual_gain.item(),
                 counterfactual_gain_count, weather_feature.item(),
                 clean_feature.item(), dg_agent_align.item(),
                 dg_fusion_align.item(), dg_agent_contrast.item(),
                 cure_recovery.item(),
                 cure_harm.item(), cure_utility.item(),
                 cure_feature.item(), cure_identity.item(),
                 cure_recovery_target_mean, cure_harm_target_ratio,
                 cure_targets['neighbor_count'],
                 snow_voxel_weather.item(), snow_voxel_clean.item(),
                 snow_voxel_novelty_ratio,
                 snow_voxel_effective_positive_weight,
                 snow_voxel_positive_probability,
                 snow_voxel_negative_probability, snow_voxel_mean_keep,
                 snow_voxel_suppressed_ratio, total.item()))
            progress.update(1)
        progress.close()
        if augmentation_mode == 'mixed_weather':
            total_mixed = sum(mixed_weather_counts.values())
            rain_fraction = (
                mixed_weather_counts['physics_rain'] / total_mixed
                if total_mixed else 0.0)
            print(
                'Mixed weather scenes: physics_rain=%d v2x_dgw_awa=%d '
                'rain_fraction=%.4f' %
                (mixed_weather_counts['physics_rain'],
                 mixed_weather_counts['v2x_dgw_awa'],
                 rain_fraction))
        elif augmentation_mode == 'physics_fog':
            formatted_counts = ' '.join(
                '%.3f=%d' % (alpha, count)
                for alpha, count in sorted(fog_alpha_counts.items()))
            print('Physics-Fog alpha scenes: %s' % formatted_counts)
        elif augmentation_mode == 'physics_snow':
            if snowfall_rates:
                print(
                    'Physics-Snow scenes: count=%d rate_min=%.3f '
                    'rate_mean=%.3f rate_max=%.3f' %
                    (len(snowfall_rates), min(snowfall_rates),
                     float(np.mean(snowfall_rates)),
                     max(snowfall_rates)))
            else:
                print('Physics-Snow scenes: count=0')

        torch.save(
            student.state_dict(),
            os.path.join(saved_path, 'net_epoch%d.pth' % (epoch + 1)))
        validation = evaluate_student(
            student, val_loader, criterion, hypes, opt, device)
        print('At epoch %d, validation clean_det %.6f weather_det %.6f '
              'clean_com %.6f weather_com %.6f' %
              (epoch, validation['clean_det'], validation['weather_det'],
               validation['clean_com'], validation['weather_com']))
        for name, value in validation.items():
            writer.add_scalar('Validate/%s' % name, value, epoch)

    writer.close()
    print('Training Finished, checkpoints saved to %s' % saved_path)


if __name__ == '__main__':
    main()
