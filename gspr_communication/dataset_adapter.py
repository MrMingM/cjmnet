"""Local dataset subclass: preserve true per-CAV transforms without patching OpenCOOD."""
import numpy as np
import torch
from opencood.data_utils.datasets.intermediate_fusion_dataset import IntermediateFusionDataset


class CommunicationDataset(IntermediateFusionDataset):
    def __init__(self, params, train):
        args = params['fusion'].get('args', {})
        if isinstance(args, dict) and not args.get('proj_first', True):
            raise ValueError('Communication v1 requires proj_first=True')
        wild = params.get('wild_setting', {})
        if wild.get('async', False) or wild.get('loc_err', False):
            raise ValueError('v1 does not support delay / localization perturbations')
        super().__init__(params, visualize=False, train=train)

    def __getitem__(self, index):
        self._communication_transforms = []
        sample = super().__getitem__(index)
        matrices = np.asarray(self._communication_transforms, dtype=np.float32)
        if len(matrices) != sample['ego']['cav_num']:
            raise ValueError('CAV pose/feature order mismatch')
        sample['ego']['communication_transforms'] = matrices
        sample['ego']['communication_sample_index'] = int(index)
        return sample

    def get_item_single_car(self, selected_cav_base, ego_pose, **kwargs):
        result = super().get_item_single_car(selected_cav_base, ego_pose, **kwargs)
        self._communication_transforms.append(np.array(
            selected_cav_base['params']['transformation_matrix'], copy=True))
        return result

    def collate_batch_train(self, batch):
        result = super().collate_batch_train(batch)
        result['ego']['communication_transforms'] = torch.from_numpy(np.concatenate([
            x['ego']['communication_transforms'] for x in batch], axis=0))
        result['ego']['communication_sample_index'] = torch.tensor([
            x['ego']['communication_sample_index'] for x in batch])
        return result
