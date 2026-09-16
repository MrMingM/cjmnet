import argparse
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils, inference_utils
from opencood.utils import eval_utils

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', required=True, type=str)
    parser.add_argument('--fusion_method', default='intermediate', type=str)
    opt = parser.parse_args()

    hypes = yaml_utils.load_yaml(None, opt)
    
    print('Creating Model...')
    model = train_utils.create_model(hypes)
    if torch.cuda.is_available():
        model.cuda()

    _, model = train_utils.load_saved_model(opt.model_dir, model)
    model.eval()

    weathers = [
        ('clean', '/data/scd/datasets/opv2v_official_data_dumping/test'),
        ('fog', '/data/cjm/datasets/opv2v-w/fog/test'),
        ('rain', '/data/cjm/datasets/opv2v-w/rain/test'),
        ('snow', '/data/cjm/datasets/opv2v-w/snow/test')
    ]

    for weather_name, val_dir in weathers:
        print(f"\n================ Evaluating {weather_name} ================")
        hypes['validate_dir'] = val_dir
        dataset = build_dataset(hypes, visualize=True, train=False)
        data_loader = DataLoader(dataset, batch_size=1, num_workers=4, collate_fn=dataset.collate_batch_test, shuffle=False)

        result_stat = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},
                       0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},
                       0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}

        for i, batch_data in enumerate(tqdm(data_loader)):
            with torch.no_grad():
                batch_data = train_utils.to_device(batch_data, 'cuda' if torch.cuda.is_available() else 'cpu')
                pred_box_tensor, pred_score, gt_box_tensor = inference_utils.inference_intermediate_fusion(batch_data, model, dataset)
                eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score, gt_box_tensor, result_stat, 0.3)
                eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score, gt_box_tensor, result_stat, 0.5)
                eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score, gt_box_tensor, result_stat, 0.7)
        
        print(f"\nResults for {weather_name}:")
        eval_utils.eval_final_results(result_stat, opt.model_dir, False)

if __name__ == '__main__':
    main()
