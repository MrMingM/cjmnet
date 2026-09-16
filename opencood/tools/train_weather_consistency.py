# -*- coding: utf-8 -*-
"""
Reliability-guided clean/weather consistency training for Where2comm.

This script keeps the original detector loss and adds a frozen clean teacher.
The student sees an online weather-degraded version of the same clean sample.
"""

import argparse
import copy
import os
import statistics

import torch
import torch.nn.functional as F
import tqdm
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils


def train_parser():
    parser = argparse.ArgumentParser(
        description='Reliability-guided weather consistency training.')
    parser.add_argument('--model_dir', required=True,
                        help='Original Where2comm checkpoint directory.')
    parser.add_argument('--hypes_yaml', default='',
                        help='Optional yaml. Defaults to model_dir/config.yaml.')
    parser.add_argument('--teacher_model_dir', default='',
                        help='Teacher checkpoint directory. Defaults to model_dir.')
    parser.add_argument('--epochs', default=5, type=int,
                        help='Fine-tuning epochs.')
    parser.add_argument('--batch_size', default=0, type=int,
                        help='Override train_params.batch_size if > 0.')
    parser.add_argument('--lr', default=0.0, type=float,
                        help='Override optimizer lr if > 0.')
    parser.add_argument('--warmup_epochs', default=0, type=int,
                        help='Fine-tuning warm-up epochs. Defaults to 0.')
    parser.add_argument('--train_batchnorm', action='store_true',
                        help='Update BatchNorm statistics and affine parameters.')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--lambda_fuse', default=0.05, type=float)
    parser.add_argument('--lambda_conf', default=0.1, type=float)
    parser.add_argument('--lambda_clean_conf', default=1.0, type=float,
                        help='Full-map clean communication-logit preservation weight.')
    parser.add_argument('--lambda_clean_det', default=1.0, type=float,
                        help='Detection loss weight for clean student branch.')
    parser.add_argument('--lambda_weather_det', default=0.5, type=float,
                        help='Detection loss weight for corrupted/weather student branch.')
    parser.add_argument('--lambda_reliability', default=1.0, type=float)
    parser.add_argument('--fg_thresh', default=0.2, type=float,
                        help='Teacher confidence threshold for foreground mask.')
    parser.add_argument('--dropout_base', default=0.05, type=float,
                        help='Base voxel dropout probability for student.')
    parser.add_argument('--dropout_range', default=0.35, type=float,
                        help='Extra dropout probability at far range.')
    parser.add_argument('--dropout_max', default=0.65, type=float,
                        help='Maximum voxel dropout probability.')
    parser.add_argument('--weather_stat_path', default='',
                        help='Clean distance statistics npz for reliability loss weight.')
    parser.add_argument('--save_name', default='weather_consistency',
                        help='Suffix appended to the saved run name.')
    return parser.parse_args()


def clone_nested(data):
    if torch.is_tensor(data):
        return data.clone()
    if isinstance(data, dict):
        return {k: clone_nested(v) for k, v in data.items()}
    if isinstance(data, list):
        return [clone_nested(v) for v in data]
    if isinstance(data, tuple):
        return tuple(clone_nested(v) for v in data)
    return copy.deepcopy(data)


def configure_weather_loss(hypes, opt):
    fusion_cfg = hypes['model']['args']['where2comm_fusion']
    weather_cfg = fusion_cfg.setdefault('weather_reliability', {})
    weather_cfg.update({
        'enable': True,
        'apply_to': 'loss',
        'density_enable': True,
        'isolation_enable': False,
        'alpha_density': 0.5,
        'beta_isolation': 0.0,
        'reliability_floor': 0.9,
        'empty_cell_reliability': 0.55,
        'use_clean_stats': bool(opt.weather_stat_path)
    })
    if opt.weather_stat_path:
        weather_cfg['stat_path'] = opt.weather_stat_path


def configure_weather_augmentation(hypes, opt):
    """Normalize augmentation configuration while preserving old commands."""
    augmentation_cfg = hypes.setdefault('weather_augmentation', {})
    augmentation_cfg.setdefault('mode', 'current_voxel_dropout')
    augmentation_cfg.setdefault(
        'seed', hypes.get('wild_setting', {}).get('seed', 20))
    current_cfg = augmentation_cfg.setdefault(
        'current_voxel_dropout', {})
    current_cfg.setdefault('dropout_base', opt.dropout_base)
    current_cfg.setdefault('dropout_range', opt.dropout_range)
    current_cfg.setdefault('dropout_max', opt.dropout_max)

    valid_modes = [
        'none', 'current_voxel_dropout', 'v2x_dgw_awa',
        'pre_voxel_matched_dropout', 'physics_rain', 'physics_fog',
        'physics_snow', 'mixed_weather']
    if augmentation_cfg['mode'] not in valid_modes:
        raise ValueError('Unsupported weather augmentation mode: %s' %
                         augmentation_cfg['mode'])
    return augmentation_cfg


def setup_hypes(opt):
    config_path = opt.hypes_yaml or os.path.join(opt.model_dir, 'config.yaml')
    hypes = yaml_utils.load_yaml(config_path)
    hypes['name'] = '%s_%s' % (hypes.get('name', 'where2comm'),
                               opt.save_name)
    hypes['train_params']['epoches'] = opt.epochs
    if 'lr_scheduler' in hypes and 'epoches' in hypes['lr_scheduler']:
        hypes['lr_scheduler']['epoches'] = opt.epochs
    if opt.batch_size > 0:
        hypes['train_params']['batch_size'] = opt.batch_size
    if opt.lr > 0:
        hypes['optimizer']['lr'] = opt.lr
    if hypes.get('lr_scheduler', {}).get('core_method') == 'cosineannealwarm':
        # The original Where2comm recipe uses a 10-epoch warm-up at 2e-5.
        # Reusing it for a short fine-tuning run silently overrides --lr.
        hypes['lr_scheduler']['warmup_epoches'] = max(
            0, min(opt.warmup_epochs, opt.epochs))
        if opt.lr > 0:
            hypes['lr_scheduler']['warmup_lr'] = opt.lr
            hypes['lr_scheduler']['lr_min'] = min(
                hypes['lr_scheduler'].get('lr_min', opt.lr), opt.lr)
    configure_weather_loss(hypes, opt)
    augmentation_cfg = configure_weather_augmentation(hypes, opt)
    hypes['weather_consistency'] = {
        'teacher_model_dir': opt.teacher_model_dir or opt.model_dir,
        'student_init_model_dir': opt.model_dir,
        'lambda_fuse': opt.lambda_fuse,
        'lambda_conf': opt.lambda_conf,
        'lambda_clean_conf': opt.lambda_clean_conf,
        'lambda_clean_det': opt.lambda_clean_det,
        'lambda_weather_det': opt.lambda_weather_det,
        'lambda_reliability': opt.lambda_reliability,
        'fg_thresh': opt.fg_thresh,
        'augmentation_mode': augmentation_cfg['mode'],
        'weather_augmentation': copy.deepcopy(augmentation_cfg),
        'warmup_epochs': opt.warmup_epochs,
        'train_batchnorm': opt.train_batchnorm,
        'weather_stat_path': opt.weather_stat_path
    }
    return hypes


def corrupt_student_voxels(ego_dict, hypes, opt):
    processed = ego_dict['processed_lidar']
    voxel_coords = processed['voxel_coords']
    if voxel_coords.numel() == 0:
        return

    lidar_range = hypes['preprocess']['cav_lidar_range']
    voxel_size = hypes['preprocess']['args']['voxel_size']
    x_min, y_min = float(lidar_range[0]), float(lidar_range[1])
    x_max, y_max = float(lidar_range[3]), float(lidar_range[4])
    vx, vy = float(voxel_size[0]), float(voxel_size[1])

    coords = voxel_coords.to(dtype=torch.float32)
    x_center = x_min + (coords[:, 3] + 0.5) * vx
    y_center = y_min + (coords[:, 2] + 0.5) * vy
    dist = torch.sqrt(x_center * x_center + y_center * y_center)
    max_dist = max((x_max * x_max + y_max * y_max) ** 0.5, 1.0)
    range_ratio = torch.clamp(dist / max_dist, min=0.0, max=1.0)
    dropout_cfg = hypes['weather_augmentation'][
        'current_voxel_dropout']
    dropout_base = float(dropout_cfg['dropout_base'])
    dropout_range = float(dropout_cfg['dropout_range'])
    dropout_max = float(dropout_cfg['dropout_max'])
    drop_prob = dropout_base + dropout_range * range_ratio
    drop_prob = torch.clamp(drop_prob, min=0.0, max=dropout_max)
    keep_mask = torch.rand_like(drop_prob) > drop_prob

    if keep_mask.sum() == 0:
        keep_mask[torch.randint(0, keep_mask.numel(), (1,), device=keep_mask.device)] = True

    processed['voxel_features'] = processed['voxel_features'][keep_mask]
    processed['voxel_coords'] = processed['voxel_coords'][keep_mask]
    processed['voxel_num_points'] = processed['voxel_num_points'][keep_mask]


def build_teacher_student_inputs(batch_ego, hypes, opt):
    """Build clean teacher, clean student, and weather student inputs.

    All returned dictionaries keep identical labels, poses, pairwise
    transformations, agent order, and ``record_len``. Only
    ``processed_lidar`` differs for the weather student.
    """
    mode = hypes['weather_augmentation']['mode']
    weather_processed = batch_ego.pop(
        'processed_lidar_weather', None)
    teacher_ego = batch_ego
    student_clean_ego = clone_nested(batch_ego)
    student_weather_ego = clone_nested(batch_ego)

    if mode == 'current_voxel_dropout':
        if weather_processed is not None:
            raise AssertionError(
                'current_voxel_dropout unexpectedly received a point-level branch')
        corrupt_student_voxels(student_weather_ego, hypes, opt)
    elif mode in [
            'v2x_dgw_awa', 'pre_voxel_matched_dropout', 'physics_rain',
            'physics_fog', 'physics_snow', 'mixed_weather']:
        if weather_processed is None:
            raise KeyError(
                '%s requires processed_lidar_weather from the dataset' % mode)
        student_weather_ego['processed_lidar'] = weather_processed
    elif mode == 'none':
        if weather_processed is not None:
            raise AssertionError(
                'augmentation=none unexpectedly received a weather branch')
    else:
        raise ValueError('Unsupported augmentation mode: %s' % mode)

    clean_record_len = student_clean_ego['record_len']
    weather_record_len = student_weather_ego['record_len']
    if not torch.equal(clean_record_len, weather_record_len):
        raise AssertionError('clean/weather record_len changed')
    return teacher_ego, student_clean_ego, student_weather_ego


def print_weather_augmentation_stats(batch_stats):
    """Print first-batch point/pillar statistics for pre-voxel weather."""
    if not batch_stats:
        return
    for batch_index, scene_stats in enumerate(batch_stats):
        for agent_index, stats in enumerate(scene_stats):
            print(
                'Weather batch=%d agent=%d cav=%s raw=%d clean_points=%d '
                'range_points=%d final_points=%d clean_voxels=%d '
                'weather_voxels=%d clean_pts/pillar=%.2f '
                'weather_pts/pillar=%.2f mean_dist=%.2f max_dist=%.2f '
                'fallback=%s clean_empty_align=%s drop_p_mean=%.4f '
                'frame=%s local_aug=%d' %
                (batch_index, agent_index, stats.get('cav_id', '?'),
                 stats['raw_point_count'], stats['clean_point_count'],
                 stats['after_range_point_count'],
                 stats['final_point_count'], stats['clean_voxel_count'],
                 stats['weather_voxel_count'],
                 stats['clean_mean_points_per_pillar'],
                 stats['weather_mean_points_per_pillar'],
                 stats['mean_distance'], stats['max_distance'],
                 stats['fallback_used'],
                 stats.get('clean_empty_alignment_used', False),
                 stats.get('drop_probability_mean', -1.0),
                  stats.get('perturbation_frame', '?'),
                  stats.get('local_augmented_point_count', -1)))
            if 'rain_rate' in stats:
                print(
                    'PhysicsRain rate=%.2fmm/h retained=%d false=%d lost=%d '
                    'input_I=%.4f output_I=%.4f foreground=%d/%d '
                    'alpha=%.8f input_hist=%s output_hist=%s' %
                    (stats['rain_rate'],
                     stats['retained_original_point_count'],
                     stats['false_return_count'],
                     stats['lost_point_count'],
                     stats['input_intensity_mean'],
                     stats['output_intensity_mean'],
                     stats.get('clean_foreground_point_count', -1),
                     stats.get('weather_foreground_point_count', -1),
                     stats['extinction_coefficient'],
                     stats['input_distance_histogram'],
                     stats['output_distance_histogram']))
            if 'fog_alpha' in stats:
                print(
                    'PhysicsFog alpha=%.4f lookup=%.4f MOR=%.1fm '
                    'retained=%d fog=%d lost=%d input_I=%.4f '
                    'output_I=%.4f' %
                    (stats['fog_alpha'], stats['fog_lookup_alpha'],
                     stats['meteorological_optical_range'],
                     stats['retained_original_point_count'],
                     stats['fog_return_count'],
                     stats['lost_point_count'],
                     stats['input_intensity_mean'],
                     stats['output_intensity_mean']))
            if 'snowfall_rate' in stats:
                print(
                    'PhysicsSnow rate=%.3fmm/h occupancy=%.3e '
                    'hazard=%.6f/m trans=%.4f intercepted=%d attenuated=%d '
                    'snow=%d retained=%d lost=%d input_I=%.4f '
                    'output_I=%.4f' %
                    (stats['snowfall_rate'], stats['occupancy_ratio'],
                     stats['intercept_hazard'],
                     stats['mean_two_way_transmittance'],
                     stats['intercepted_point_count'],
                     stats['attenuated_point_count'],
                     stats['snow_return_count'],
                     stats['retained_original_point_count'],
                     stats['lost_point_count'],
                     stats['input_intensity_mean'],
                     stats['output_intensity_mean']))


def scene_reliability_from_agent_map(agent_r_map, record_len, target_size):
    if agent_r_map is None:
        return None
    split_maps = torch.tensor_split(agent_r_map, torch.cumsum(record_len, dim=0)[:-1].cpu())
    scene_maps = []
    for maps in split_maps:
        scene_maps.append(maps.mean(dim=0, keepdim=True))
    r_scene = torch.cat(scene_maps, dim=0)
    if r_scene.shape[-2:] != target_size:
        r_scene = F.interpolate(r_scene, size=target_size,
                                mode='bilinear', align_corners=False)
    return torch.clamp(r_scene, min=0.0, max=1.0)


def weighted_mean(value_map, weight_map):
    denom = weight_map.sum().clamp(min=1.0)
    return (value_map * weight_map).sum() / denom


def consistency_losses(student_out, teacher_out, student_ego, opt):
    teacher_psm = teacher_out['psm'].detach()
    student_psm = student_out['psm']
    fg_mask = (teacher_psm.sigmoid().max(dim=1, keepdim=True)[0] >
               opt.fg_thresh).to(student_psm.dtype)
    if fg_mask.sum() < 1:
        fg_mask = torch.ones_like(fg_mask)

    r_map = scene_reliability_from_agent_map(
        student_out.get('weather_R_map', None),
        student_ego['record_len'],
        fg_mask.shape[-2:])
    if r_map is None:
        r_map = torch.ones_like(fg_mask)
    reliability_weight = 1.0 + opt.lambda_reliability * (1.0 - r_map)
    feat_weight = fg_mask * reliability_weight

    # Where2comm builds its mask from psm_single at a low threshold (often
    # 0.01), so communication distillation must cover the whole map.
    student_comm = student_out['single_confidence']
    teacher_comm = teacher_out['single_confidence'].detach()
    comm_weight = reliability_weight
    if comm_weight.shape[-2:] != student_comm.shape[-2:]:
        comm_weight = F.interpolate(
            comm_weight, size=student_comm.shape[-2:],
            mode='bilinear', align_corners=False)
    conf_diff = (student_comm - teacher_comm).pow(2).mean(
        dim=1, keepdim=True)
    conf_loss = weighted_mean(conf_diff, comm_weight)

    student_feat = student_out['fused_feature']
    teacher_feat = teacher_out['fused_feature'].detach()
    if feat_weight.shape[-2:] != student_feat.shape[-2:]:
        feat_weight = F.interpolate(feat_weight, size=student_feat.shape[-2:],
                                    mode='bilinear', align_corners=False)
    student_feat = F.normalize(student_feat, dim=1)
    teacher_feat = F.normalize(teacher_feat, dim=1)
    fuse_diff = (student_feat - teacher_feat).abs().mean(dim=1, keepdim=True)
    fuse_loss = weighted_mean(fuse_diff, feat_weight)
    return fuse_loss, conf_loss


def clean_communication_loss(student_out, teacher_out):
    student_comm = student_out['single_confidence']
    teacher_comm = teacher_out['single_confidence'].detach()
    return (student_comm - teacher_comm).pow(2).mean()


def load_model_from_dir(hypes, model_dir, device, trainable):
    model = train_utils.create_model(hypes)
    _, model = train_utils.load_saved_model(model_dir, model)
    model.to(device)
    if not trainable:
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
    return model


def freeze_batchnorm(model):
    count = 0
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
            for param in module.parameters():
                param.requires_grad = False
            count += 1
    return count


def keep_batchnorm_eval(model):
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()


def main():
    opt = train_parser()
    hypes = setup_hypes(opt)

    print('-----------------Dataset Building------------------')
    train_dataset = build_dataset(hypes, visualize=False, train=True)
    val_dataset = build_dataset(hypes, visualize=False, train=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=hypes['train_params']['batch_size'],
        num_workers=opt.num_workers,
        collate_fn=train_dataset.collate_batch_train,
        shuffle=True,
        pin_memory=False,
        drop_last=True)
    val_loader = DataLoader(
        val_dataset,
        batch_size=hypes['train_params']['batch_size'],
        num_workers=opt.num_workers,
        collate_fn=train_dataset.collate_batch_train,
        shuffle=False,
        pin_memory=False,
        drop_last=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    teacher_dir = opt.teacher_model_dir or opt.model_dir

    print('---------------Creating Teacher/Student------------------')
    teacher = load_model_from_dir(hypes, teacher_dir, device, trainable=False)
    student = load_model_from_dir(hypes, opt.model_dir, device, trainable=True)
    if not opt.train_batchnorm:
        frozen_bn_count = freeze_batchnorm(student)
        print('Frozen BatchNorm layers: %d' % frozen_bn_count)

    criterion = train_utils.create_loss(hypes)
    optimizer = train_utils.setup_optimizer(hypes, student)
    scheduler = train_utils.setup_lr_schedular(hypes, optimizer, len(train_loader))
    if hypes['lr_scheduler']['core_method'] == 'cosineannealwarm':
        print('Fine-tuning LR config: lr=%.7f, warmup_lr=%.7f, '
              'warmup_epochs=%d, lr_min=%.7f' %
              (hypes['optimizer']['lr'],
               hypes['lr_scheduler']['warmup_lr'],
               hypes['lr_scheduler']['warmup_epoches'],
               hypes['lr_scheduler']['lr_min']))
    saved_path = train_utils.setup_train(hypes)
    writer = SummaryWriter(saved_path)

    print('Weather consistency training start')
    augmentation_mode = hypes['weather_augmentation']['mode']
    print('Weather augmentation mode: %s' % augmentation_mode)
    for epoch in range(opt.epochs):
        if hasattr(train_dataset, 'set_weather_augmentation_epoch'):
            train_dataset.set_weather_augmentation_epoch(epoch)
        if hypes['lr_scheduler']['core_method'] != 'cosineannealwarm':
            scheduler.step(epoch)
        if hypes['lr_scheduler']['core_method'] == 'cosineannealwarm':
            scheduler.step_update(epoch * len(train_loader))
        for param_group in optimizer.param_groups:
            print('learning rate %.7f' % param_group['lr'])

        pbar = tqdm.tqdm(total=len(train_loader), leave=True)
        for i, batch_data in enumerate(train_loader):
            student.train()
            if not opt.train_batchnorm:
                keep_batchnorm_eval(student)
            optimizer.zero_grad()

            augmentation_stats = batch_data['ego'].pop(
                'weather_augmentation_stats', None)
            if i == 0 and augmentation_mode in [
                    'v2x_dgw_awa', 'pre_voxel_matched_dropout',
                    'physics_rain', 'physics_fog', 'physics_snow',
                    'mixed_weather']:
                print_weather_augmentation_stats(augmentation_stats)
            batch_data = train_utils.to_device(batch_data, device)
            teacher_ego, student_clean_ego, student_weather_ego = \
                build_teacher_student_inputs(
                    batch_data['ego'], hypes, opt)
            teacher_ego['return_aux'] = True
            student_clean_ego['return_aux'] = True
            student_weather_ego['return_aux'] = True

            with torch.no_grad():
                teacher_out = teacher(teacher_ego)
            student_clean_out = student(student_clean_ego)
            student_weather_out = student(student_weather_ego)

            clean_det_loss = criterion(student_clean_out,
                                       batch_data['ego']['label_dict'])
            weather_det_loss = criterion(student_weather_out,
                                         batch_data['ego']['label_dict'])
            fuse_loss, conf_loss = consistency_losses(
                student_weather_out, teacher_out, student_weather_ego, opt)
            clean_conf_loss = clean_communication_loss(
                student_clean_out, teacher_out)
            final_loss = opt.lambda_clean_det * clean_det_loss + \
                opt.lambda_weather_det * weather_det_loss + \
                opt.lambda_fuse * fuse_loss + \
                opt.lambda_conf * conf_loss + \
                opt.lambda_clean_conf * clean_conf_loss

            final_loss.backward()
            optimizer.step()
            if hypes['lr_scheduler']['core_method'] == 'cosineannealwarm':
                scheduler.step_update(epoch * len(train_loader) + i)

            if i % 10 == 0:
                global_step = epoch * len(train_loader) + i
                writer.add_scalar('Train/clean_det_loss',
                                  clean_det_loss.item(), global_step)
                writer.add_scalar('Train/weather_det_loss',
                                  weather_det_loss.item(), global_step)
                writer.add_scalar('Train/fuse_loss', fuse_loss.item(), global_step)
                writer.add_scalar('Train/conf_loss', conf_loss.item(), global_step)
                writer.add_scalar('Train/clean_conf_loss',
                                  clean_conf_loss.item(), global_step)
                writer.add_scalar('Train/total_loss', final_loss.item(), global_step)
            pbar.set_description(
                'clean %.4f weather %.4f fuse %.4f conf %.4f '
                'clean_conf %.4f total %.4f' %
                (clean_det_loss.item(), weather_det_loss.item(),
                 fuse_loss.item(), conf_loss.item(), clean_conf_loss.item(),
                 final_loss.item()))
            pbar.update(1)
        pbar.close()

        if epoch % hypes['train_params']['save_freq'] == 0:
            torch.save(student.state_dict(),
                       os.path.join(saved_path, 'net_epoch%d.pth' % (epoch + 1)))

        if epoch % hypes['train_params']['eval_freq'] == 0:
            valid_ave_loss = []
            with torch.no_grad():
                student.eval()
                for batch_data in val_loader:
                    batch_data = train_utils.to_device(batch_data, device)
                    out = student(batch_data['ego'])
                    val_loss = criterion(out, batch_data['ego']['label_dict'])
                    valid_ave_loss.append(val_loss.item())
            valid_ave_loss = statistics.mean(valid_ave_loss)
            print('At epoch %d, validation det loss is %f' %
                  (epoch, valid_ave_loss))
            writer.add_scalar('Validate_Loss', valid_ave_loss, epoch)

    print('Training Finished, checkpoints saved to %s' % saved_path)


if __name__ == '__main__':
    main()
