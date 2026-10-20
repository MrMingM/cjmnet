"""Explicit factories for C/D and Scatter -> FiLM -> original AttBEVBackbone."""
import torch
from torch import nn
from auditlib import CONFIG, ROOT, load_frozen

OBSERVED_FILM = []


class SpatialFiLM(nn.Module):
    def __init__(self):
        super().__init__()
        cfg = CONFIG['film']
        self.net = nn.Sequential(nn.Linear(2 * cfg['channels'], cfg['hidden']), nn.SiLU(),
                                 nn.Linear(cfg['hidden'], 2 * cfg['channels']))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.last = None
        self.observations = []

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != 64:
            raise ValueError('Scatter insertion must be [N,64,H,W], got ' + str(tuple(x.shape)))
        stats = torch.cat((x.mean((2, 3)), x.var((2, 3), unbiased=False).add(1e-6).sqrt()), 1)
        dg, beta = self.net(stats).chunk(2, 1)
        gamma = 1 + dg
        if not torch.isfinite(gamma).all() or not torch.isfinite(beta).all():
            raise ValueError('Nonfinite FiLM coefficients')
        self.last = dict(shape=list(x.shape), gamma=describe(gamma), beta=describe(beta),
            delta_rms=float(dg.detach().square().mean().sqrt()),
            beta_rms=float(beta.detach().square().mean().sqrt()),
            between_cav_gamma_rms=float(gamma.detach().var(dim=0, unbiased=False).mean().sqrt()),
            between_cav_beta_rms=float(beta.detach().var(dim=0, unbiased=False).mean().sqrt()))
        self.observations.append(self.last)
        OBSERVED_FILM.append(self.last)
        return gamma[:, :, None, None] * x + beta[:, :, None, None]


def describe(x):
    flat = x.detach().float().flatten()
    return dict(mean=float(flat.mean()), std=float(flat.std(unbiased=False)),
        min=float(flat.min()), max=float(flat.max()),
        quantiles=[float(v) for v in torch.quantile(flat, flat.new_tensor([.05, .5, .95]))])


def add_film(model):
    rng = torch.get_rng_state()
    try:
        model.film = SpatialFiLM()
    finally:
        torch.set_rng_state(rng)
    def after_scatter(module, inputs, result):
        result['spatial_features'] = model.film(result['spatial_features'])
        return result
    model.scatter.register_forward_hook(after_scatter)
    return model


def create(hypes, variant):
    from opencood.models.point_pillar_intermediate import PointPillarIntermediate
    if hypes['model']['core_method'] != 'point_pillar_intermediate':
        raise ValueError('Unexpected official base model')
    if variant.startswith('D'):
        integration = load_frozen('gspr_integration', 'integration.py', ROOT / 'v2xr_gspr_clean')
        model = integration.PointPillarGsprOfficial(hypes['model']['args'])
    elif variant.startswith('C'):
        model = PointPillarIntermediate(hypes['model']['args'])
    else:
        raise ValueError('Unknown variant: ' + variant)
    if variant.endswith('FiLM'):
        add_film(model)
    def finite_detection(module, inputs, result):
        if any(not torch.isfinite(result[key]).all() for key in ('psm', 'rm')):
            raise ValueError('Nonfinite detector logits/regression before official postprocessing')
    model.register_forward_hook(finite_detection)
    return model


def criterion(hypes, variant, film=False):
    from opencood.loss.point_pillar_loss_opencood import PointPillarLossOpenCOOD
    if hypes['loss']['core_method'] != 'point_pillar_loss_opencood':
        raise ValueError('Unexpected official detection loss')
    if variant.startswith('D') and not film:
        integration = load_frozen('gspr_integration', 'integration.py', ROOT / 'v2xr_gspr_clean')
        return integration.GsprOfficialLoss(hypes['loss']['args'])
    return PointPillarLossOpenCOOD(hypes['loss']['args'])


def install(variant):
    from opencood.tools import train_utils
    train_utils.create_model = lambda h: create(h, variant)
    train_utils.create_loss = lambda h: criterion(h, variant, variant.endswith('FiLM'))


def frozen_backbone(model):
    model.eval()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith('film.'))
    model.film.train()


def strict_backbone(model, base):
    actual = model.state_dict()
    keys = {k for k in actual if not k.startswith('film.')}
    if keys != set(base) or any(not torch.equal(actual[k].detach().cpu(), base[k].cpu()) for k in keys):
        raise ValueError('Frozen backbone parameter/buffer changed')


def bn_layers(model):
    layers = {n: m for n, m in model.named_modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)}
    if not layers or any(not m.track_running_stats for m in layers.values()):
        raise ValueError('No applicable tracked BN layers')
    return layers


def start_bn(model):
    model.eval()
    model.requires_grad_(False)
    layers = bn_layers(model)
    for layer in layers.values():
        layer.reset_running_stats()
        layer.momentum = None
        layer.train()
    if any(module.training for module in model.modules() if not isinstance(module, nn.modules.batchnorm._BatchNorm)):
        raise ValueError('Non-BN module entered training mode during calibration')
    return layers


def bn_difference(before, after, layers):
    allowed = {n + '.' + suffix for n in layers for suffix in
               ('running_mean', 'running_var', 'num_batches_tracked')}
    if before.keys() != after.keys():
        raise ValueError('BN calibration changed model keys')
    changed = {k for k in before if not torch.equal(before[k].cpu(), after[k].detach().cpu())}
    if not changed <= allowed:
        raise ValueError('Non-BN state changed: ' + repr(changed - allowed))
    summaries = {}
    for name, layer in layers.items():
        if int(after[name + '.num_batches_tracked']) <= 0:
            raise ValueError('BN layer did not execute: ' + name)
        summaries[name] = dict(type=type(layer).__name__,
            num_batches_tracked=int(after[name + '.num_batches_tracked']),
            mean_delta=describe(after[name + '.running_mean'].cpu() - before[name + '.running_mean'].cpu()),
            variance_delta=describe(after[name + '.running_var'].cpu() - before[name + '.running_var'].cpu()))
    if not any(k.endswith(('running_mean', 'running_var')) for k in changed):
        raise ValueError('BN recalibration had no statistical effect')
    return dict(only_allowed_buffers_changed=True, changed=sorted(changed), layers=summaries)
