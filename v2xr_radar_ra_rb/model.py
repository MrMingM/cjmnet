"""Keep the official LR1 forward and attach two sparse feature hooks."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from empty_input import preserve_empty_vehicle_slots
from v2xr_lr1.lr1_model import LR1Model, LR1Loss
from enhancement import RadarSupport, RadarCompensation, keys, lookup, point_statistics


class Model(LR1Model):
    def __init__(self, args, settings):
        super().__init__(args, 'LR1')
        preserve_empty_vehicle_slots(self)
        self.ra_on, self.rb_on = settings['ra'], settings['rb']
        self.mode = 'normal'
        self.observe = False
        self.aux = None
        self.observation = {}
        self.extra_used = set()
        with torch.random.fork_rng(devices=[]):
            self.ra = (RadarSupport(args['voxel_size'], (self.scatter.nx, self.scatter.ny),
                                   radius=settings['radius'], neighbors=settings['neighbors'])
                       if settings['ra'] else None)
            self.rb = RadarCompensation(settings['ordinary']) if settings['rb'] else None
        self.radar_pillar_vfe.register_forward_hook(self.radar_features)
        self.backbone.register_forward_pre_hook(self.compensate)
        # Identical computation, with frozen GSPR activations recomputed on backward.
        native_gspr = self.gspr.forward

        def gspr(*args):
            if self.training and torch.is_grad_enabled():
                return checkpoint(native_gspr, *args, use_reentrant=False)
            return native_gspr(*args)

        self.gspr.forward = gspr

    def forward(self, data):
        if 'paired_weather' in data:
            raise ValueError('Training-only labels reached the detector')
        self.aux, self.observation, self.extra_used = None, {}, set()
        self.radar_stats = None
        if self.mode != 'normal':
            if self.training:
                raise ValueError('Radar corruption is evaluation-only')
            data = dict(data)
            radar = data['processed_radar']
            coords = radar['voxel_coords']
            if self.mode == 'radar_off':
                data['processed_radar'] = {k: v[:0] for k, v in radar.items()}
            elif self.mode == 'radar_shuffle':
                # Deterministically permute whole pillars within each vehicle, retaining the occupancy set.
                features = radar['voxel_features'].clone()
                counts = radar['voxel_num_points'].clone()
                for car in coords[:, 0].unique():
                    idx = torch.where(coords[:, 0] == car)[0]
                    perm = idx.roll(max(1, len(idx)//2))
                    delta = (coords[idx, 2:] - coords[perm, 2:]).flip(-1).to(features.dtype) * .4
                    slots = torch.arange(features.shape[1], device=features.device)[None] < radar['voxel_num_points'][perm, None]
                    features[idx] = radar['voxel_features'][perm]
                    features[idx, :, :2] += delta[:, None] * slots[..., None]
                    counts[idx] = radar['voxel_num_points'][perm]
                data['processed_radar'] = dict(radar, voxel_features=features, voxel_num_points=counts)
            else:
                raise ValueError(f'Unknown diagnostic {self.mode}')
        result = super().forward(data)
        self.radar_stats = None
        return result

    def radar_features(self, _, __, radar):
        coords = radar['voxel_coords']
        if self.ra is None or not (self.ra_on or self.rb_on):
            return radar
        features, stats = self.ra(radar['pillar_features'], radar, self.ra_on)
        radar['pillar_features'] = features
        self.radar_stats = coords, stats
        if len(coords) and self.ra_on:
            self.extra_used.add('ra')
        if self.observe and len(coords):
            self.observation.update(ra_neighbors=float(stats['neighbors'].float().mean()),
                                    ra_distance=float(stats['distance'].mean()),
                                    ra_residual=float(stats['residual'].mean()))
        return radar

    def compensate(self, _, inputs):
        if not self.rb_on or self.radar_stats is None:
            return
        coords, stats = self.radar_stats
        if not len(coords):
            return
        self.extra_used.add('rb')
        c, _, y, x = coords.long().unbind(1)
        raw = self._raw
        nx, ny = self.scatter.nx, self.scatter.ny
        rid = keys(coords, ny, nx)
        idx, observed = lookup(keys(raw['voxel_coords'], ny, nx), rid)
        count, _, coverage = point_statistics(raw)
        fields = []
        for values, unknown in ((count/32, 0.), (coverage, 0.),
                                (None if self._point_outputs is None else self._point_outputs['pillar_reliability'].reshape(-1), -1.),
                                (None if self._point_outputs is None else self._point_outputs['pillar_uncertainty'].reshape(-1), -1.)):
            if len(count):
                fields.append(torch.where(observed, values[idx], count.new_full(idx.shape, unknown)))
            else:
                fields.append(stats['count'].new_full(idx.shape, unknown))
        points, span, reliability, uncertainty = fields
        state = torch.stack((observed.to(points.dtype), points, span, reliability, uncertainty,
            stats['count']/32, stats['spread'], stats['support'], stats['neighbors']/8,
            stats['distance']/self.ra.radius), -1)
        spatial = inputs[0]['spatial_features']
        delta, opportunity = self.rb(spatial[c, :64, y, x], spatial[c, 64:, y, x], state, stats['support'])
        amended = spatial.clone()
        amended[c, :64, y, x] += delta
        inputs[0]['spatial_features'] = amended
        if self.training:
            self.aux = dict(coords=coords, gate=opportunity, delta=delta, reliability=reliability,
                            support=stats['support'], observed=observed)
        if self.observe:
            amplitude = delta.detach().square().mean(-1).sqrt()
            self.observation.update(radar_cells=len(coords), gate_sum=float(opportunity.sum()),
                gate_sq_sum=float(opportunity.square().sum()), residual_sum=float(amplitude.sum()),
                missing_lidar=int((~observed).sum()),
                corrected_missing=int(((~observed) & (amplitude > 1e-6)).sum()),
                corrected_low_reliability=int((observed & (reliability < .5) & (amplitude > 1e-6)).sum()),
                low_reliability=int((observed & (reliability < .5)).sum()),
                corrected_observed=int((observed & (amplitude > 1e-6)).sum()),
                observed=int(observed.sum()))


class Loss(LR1Loss):
    def __init__(self, native, model, settings):
        super().__init__(native, model)
        self.settings = settings
        self.pair = None

    def forward(self, prediction, target):
        model = self._model()
        # Legitimate empty LiDAR has no point-level regularizer.
        loss = self.native(prediction, target) if model._point_outputs is None else super().forward(prediction, target)
        if not model.training or not self.settings['paired']:
            if self.pair is not None:
                raise ValueError('Pair data in evaluation or unpaired training')
            return loss
        if self.pair is None:
            raise ValueError('F3 training requires paired supervision')
        if model.aux is None:
            return loss
        a = model.aux
        coords, targets = self.pair
        nx, ny = model.scatter.nx, model.scatter.ny
        idx, found = lookup(keys(coords, ny, nx), keys(a['coords'], ny, nx))
        if len(targets) != len(a['coords']) or not found.all():
            raise ValueError('Paired supervision and real radar coordinates disagree')
        damage, known_fog, healthy = targets[idx].unbind(-1)
        weight = known_fog * a['support'].detach()
        gate_loss = F.binary_cross_entropy(a['gate'], damage, reduction='none')
        weather = (weight * gate_loss).sum() / weight.sum().clamp_min(1e-6)
        protect = healthy * a['observed'] * (a['reliability'].detach() >= .7)
        preserve = (protect * a['delta'].square().mean(-1)).sum() / protect.sum().clamp_min(1)
        self.last_terms.update(weather=float(weather.detach()), preserve=float(preserve.detach()),
                               supervised=int(known_fog.sum()), protected=int(protect.sum()))
        return loss + self.settings['weather_weight']*weather + self.settings['preserve_weight']*preserve

