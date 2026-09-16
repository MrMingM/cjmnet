# -*- coding: utf-8 -*-
"""
Batch inference for Where2comm on OPV2V-w weather splits.

Example:
python opencood/tools/test_weather_where2comm.py ^
  --model_dir opencood/logs/point_pillar_where2comm_xxx ^
  --weather clean=D:/data/OPV2V/test ^
  --weather fog=D:/data/OPV2V-w/fog/test ^
  --weather rain=D:/data/OPV2V-w/rain/test ^
  --weather snow=D:/data/OPV2V-w/snow/test
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
from torch.utils.data import DataLoader
from tqdm import tqdm

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils
from opencood.utils import eval_utils


def parse_args():
    parser = argparse.ArgumentParser(
        description='Batch test Where2comm on multiple weather validation sets.')
    parser.add_argument('--model_dir', required=True, type=str,
                        help='Directory containing config.yaml and checkpoint.')
    parser.add_argument('--weather', action='append', required=True,
                        help='Weather split in the form name=/path/to/validate_dir. '
                             'Use this argument multiple times.')
    parser.add_argument('--eval_epoch', default=0, type=int,
                        help='Checkpoint epoch to evaluate, e.g. 50 loads net_epoch50.pth. '
                             'Use 0 to load latest/max epoch.')
    parser.add_argument('--output_csv', default='', type=str,
                        help='Where to save summary CSV. Defaults to model_dir/weather_eval.csv.')
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--max_cav', default=0, type=int,
                        help='Override train_params.max_cav and model.args.max_cav. '
                             'Use 0 to keep config.yaml.')
    parser.add_argument('--device', default='auto', type=str,
                        help='Device for inference: auto, cpu, cuda, cuda:0, cuda:1, etc.')
    parser.add_argument('--global_sort_detections', action='store_true')
    parser.add_argument('--use_uncertainty', choices=['keep', 'true', 'false'],
                        default='keep',
                        help='Override where2comm communication.use_uncertainty at test time.')
    parser.add_argument('--uncertainty_alpha', default=None, type=float,
                        help='Soft uncertainty penalty alpha. If set, enables uncertainty and uses '
                             'communication = confidence * (1 - alpha * entropy).')
    return parser.parse_args()


def parse_weather_items(items):
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

def set_uncertainty_override(hypes, override, uncertainty_alpha):
    communication = hypes['model']['args']['where2comm_fusion']['communication']
    if override != 'keep':
        communication['use_uncertainty'] = override == 'true'
    if uncertainty_alpha is not None:
        communication['use_uncertainty'] = True
        communication['uncertainty_alpha'] = uncertainty_alpha
        print('Override uncertainty_alpha to %.4f' % uncertainty_alpha)


def set_max_cav_override(hypes, max_cav):
    if max_cav <= 0:
        return
    hypes.setdefault('train_params', {})['max_cav'] = max_cav
    if 'model' in hypes and 'args' in hypes['model']:
        hypes['model']['args']['max_cav'] = max_cav
    print('Override max_cav to %d' % max_cav)



def resolve_device(device_arg):
    if device_arg == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_arg)


def load_model_for_eval(model_dir, model, eval_epoch):
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
                         num_workers, global_sort_detections):
    hypes['validate_dir'] = validate_dir

    print('\n' + '=' * 80)
    print('Weather: %s' % weather_name)
    print('validate_dir: %s' % validate_dir)
    print('=' * 80)

    dataset = build_dataset(hypes, visualize=True, train=False)
    data_loader = DataLoader(dataset,
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
    cav_counts = []

    model.eval()
    for _, batch_data in tqdm(enumerate(data_loader), total=len(data_loader)):
        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)
            record_len = batch_data['ego'].get('record_len', None)
            if record_len is not None:
                cav_counts.extend([int(x) for x in record_len.detach().cpu().view(-1).tolist()])
            output_dict = OrderedDict()
            output_dict['ego'] = model(batch_data['ego'])

            com = output_dict['ego'].get('com', None)
            if com is not None:
                communication_rates.append(float(com.detach().cpu()))

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
    avg_cav = sum(cav_counts) / len(cav_counts) if cav_counts else -1.0
    min_cav = min(cav_counts) if cav_counts else -1
    max_cav = max(cav_counts) if cav_counts else -1

    print('Weather %s: AP@0.3 %.4f, AP@0.5 %.4f, AP@0.7 %.4f, Com %.4f, CAV %.2f [%d, %d]' %
          (weather_name, ap30, ap50, ap70, avg_com, avg_cav, min_cav, max_cav))

    return {
        'weather': weather_name,
        'validate_dir': validate_dir,
        'samples': len(dataset),
        'ap30': ap30,
        'ap50': ap50,
        'ap70': ap70,
        'communication_rate': avg_com,
        'avg_cav': avg_cav,
        'min_cav': min_cav,
        'max_cav': max_cav,
    }


def main():
    opt = parse_args()
    weather_dirs = parse_weather_items(opt.weather)
    output_csv = opt.output_csv or os.path.join(opt.model_dir, 'weather_eval.csv')

    config_path = os.path.join(opt.model_dir, 'config.yaml')
    hypes = yaml_utils.load_yaml(config_path)
    set_max_cav_override(hypes, opt.max_cav)
    set_uncertainty_override(hypes, opt.use_uncertainty, opt.uncertainty_alpha)

    print('Creating Model')
    model = train_utils.create_model(hypes)
    device = resolve_device(opt.device)
    if device.type == 'cuda':
        if device.index is not None:
            torch.cuda.set_device(device)
        torch.cuda.empty_cache()

    print('Loading Model from checkpoint')
    model = load_model_for_eval(opt.model_dir, model, opt.eval_epoch)
    model.to(device)

    rows = []
    for weather_name, validate_dir in weather_dirs:
        rows.append(evaluate_one_weather(model, copy.deepcopy(hypes), weather_name,
                                         validate_dir, device,
                                         opt.num_workers,
                                         opt.global_sort_detections))

    os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=['weather', 'validate_dir', 'samples',
                        'ap30', 'ap50', 'ap70', 'communication_rate',
                        'avg_cav', 'min_cav', 'max_cav'])
        writer.writeheader()
        writer.writerows(rows)

    print('\nSaved summary to %s' % output_csv)


if __name__ == '__main__':
    main()