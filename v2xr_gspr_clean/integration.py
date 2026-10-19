"""Server-only GSPR adapter; original V2X-R model/VFE/backbone code stays intact."""
import importlib.util
from pathlib import Path
import sys

import torch
from opencood.models.point_pillar_intermediate import PointPillarIntermediate
from opencood.loss.point_pillar_loss_opencood import PointPillarLossOpenCOOD
from auditlib import ROOT, SETTINGS, read_json, sha256

OFFICIAL_LOSS_METHOD = 'point_pillar_loss_opencood'


def reliability_class():
    path = ROOT / 'attfuse_gspr/reliability.py'
    expected = read_json(Path(__file__).parent / 'PINS.json')['files']['attfuse_gspr/reliability.py']
    if sha256(path) != expected:
        raise ValueError('Frozen reliability network changed')
    name = '_v2xr_gspr_original_reliability'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name].GeometryFirstPillarReliability


class PointPillarGsprOfficial(PointPillarIntermediate):
    def __init__(self, args):
        # Initialize exactly the official base first. Extra GSPR initialization does not
        # advance the global CPU RNG used by the subsequent official training loaders.
        super().__init__(args)
        if self.modality != 'processed_lidar':
            raise ValueError('This experiment is LiDAR only')
        rng = torch.get_rng_state()
        try:
            self.gspr = reliability_class()(SETTINGS['model'], args['voxel_size'], args['lidar_range'])
        finally:
            torch.set_rng_state(rng)
        self.force_unit_weights = False  # Synthetic equivalence check only; never training/evaluation.
        self._point_outputs = None
        self.pillar_vfe.register_forward_pre_hook(self._before_vfe)
        self.pillar_vfe.pfn_layers[0].register_forward_pre_hook(self._before_pfn)

    def _before_vfe(self, module, inputs):
        batch = inputs[0]
        features, counts = batch['voxel_features'], batch['voxel_num_points']
        if features.ndim != 3 or features.shape[-1] != 4 or features.shape[0] == 0 or (counts <= 0).any():
            raise ValueError('Expected nonempty valid official XYZI voxels')
        self._point_outputs = self.gspr(features, counts, batch['voxel_coords'])
        r = self._point_outputs['point_reliability']
        if not torch.isfinite(r).all() or not torch.isfinite(self._point_outputs['point_evidence']).all():
            raise ValueError('Nonfinite GSPR output')

    def _before_pfn(self, module, inputs):
        features = inputs[0]
        if self._point_outputs is None:
            raise ValueError('PFN called without point reliability')
        r = self._point_outputs['point_reliability']
        if tuple(r.shape) != tuple(features.shape[:2]):
            raise ValueError('GSPR weights do not align with official decorated point slots')
        if self.force_unit_weights:
            return inputs  # Server synthetic identity check: leave original tensor exactly intact.
        return (features * r.unsqueeze(-1),)

    def forward(self, data_dict):
        self._point_outputs = None
        result = super().forward(data_dict)  # Unchanged official VFE/scatter/attention/head forward.
        if self._point_outputs is None:
            raise ValueError('GSPR insertion was not executed')
        result.update(self._point_outputs)
        return result


class GsprOfficialLoss(PointPillarLossOpenCOOD):
    def __init__(self, args):
        super().__init__(args)
        # The official loss logs running means and resets its counter at 500.
        # A buffer includes this additional logging accumulator in resume state_dict.
        self.register_buffer('_gspr_regularizer_sum', torch.zeros((), dtype=torch.float64))

    def forward(self, output_dict, target_dict, epoch=1, train=True):
        if 'point_reliability_target' in output_dict:
            raise ValueError('This registered experiment has no point/noise supervision')
        previous_count = self.loss_cnt
        detection = super().forward(output_dict, target_dict, epoch=epoch, train=train)
        valid = output_dict['point_valid_mask']
        if not valid.any():
            raise ValueError('No valid points for GSPR regularization')
        settings = SETTINGS['loss']
        keep = (output_dict['point_reliability'][valid].mean() - settings['target_keep_rate']).square()
        evidence = output_dict['point_evidence'][valid].square().mean()
        regularizer = settings['keep_weight'] * keep + settings['evidence_weight'] * evidence
        total = detection + regularizer
        if self.loss_cnt <= previous_count:
            self._gspr_regularizer_sum.zero_()
        self._gspr_regularizer_sum.add_(regularizer.detach().item())
        self.loss_dict['total_loss'] += self._gspr_regularizer_sum.item() / self.loss_cnt
        self.loss_dict['gspr_keep'] = keep.detach().item()
        self.loss_dict['gspr_evidence'] = evidence.detach().item()
        return total

    def logging(self, epoch, batch_id, batch_len, writer, pbar=None):
        super().logging(epoch, batch_id, batch_len, writer, pbar=pbar)
        step = epoch * batch_len + batch_id
        writer.add_scalar('GSPR_keep', self.loss_dict['gspr_keep'], step)
        writer.add_scalar('GSPR_evidence', self.loss_dict['gspr_evidence'], step)


def install():
    from opencood.tools import train_utils
    if getattr(train_utils.create_model, '_v2xr_gspr_adapter', False):
        return

    def model_factory(hypes):
        if hypes['model']['core_method'] != 'point_pillar_intermediate':
            raise ValueError('Unexpected base model')
        metadata = hypes.get('gspr_experiment')
        if metadata is not None and metadata != SETTINGS:
            raise ValueError('GSPR metadata differs from registered settings')
        print('Explicit GSPR model adapter: integration.PointPillarGsprOfficial', flush=True)
        return PointPillarGsprOfficial(hypes['model']['args'])

    def loss_factory(hypes):
        if hypes['loss']['core_method'] != OFFICIAL_LOSS_METHOD:
            raise ValueError('Expected official loss ' + OFFICIAL_LOSS_METHOD + ', got ' + str(hypes['loss']['core_method']))
        return GsprOfficialLoss(hypes['loss']['args'])

    model_factory._v2xr_gspr_adapter = True
    train_utils.create_model = model_factory
    train_utils.create_loss = loss_factory
