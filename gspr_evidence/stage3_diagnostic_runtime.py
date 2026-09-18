"""Isolated frozen-output interventions; no monkey patches to the trained model."""
import copy
import math
import time
import numpy as np
from .stage3_analysis import greedy_assignment
from .stage3_trace import trace_branch, polygon_ious, save_proposals
from .stage3_diagnostic_analysis import watched_anchors, compact_targets, compare_targets, subset_edges


def weight_prediction(model, encoded, mode, peer=0, scale=None, probes=None, head_shape=None):
    """Keys and values stay fixed for query/uniform interventions.

    zero_ego_value fixes the ORIGINAL weights and removes only ego's value term.
    This is an artificial intervention, not a deployable method.
    """
    import torch
    fused, stats = [], []
    for level_index, x in enumerate(encoded['levels']):
        logits = (x[:1]*x).sum(1, keepdim=True)/math.sqrt(x.shape[1])
        original = logits.softmax(0)
        weights, values, used_logits = original, x, logits
        if scale is None or scale == level_index:
            if mode == 'peer_query':
                used_logits = (x[peer:peer+1]*x).sum(1, keepdim=True)/math.sqrt(x.shape[1])
                weights = used_logits.softmax(0)
            elif mode == 'uniform':
                weights = torch.ones_like(original)/len(x)
            elif mode == 'zero_ego_value':
                values = x.clone(); values[0] = 0
            elif mode != 'identity':
                raise ValueError(mode)
        fused.append((weights*values).sum(0, keepdim=True))
        sites = {}
        for cid, (row, col) in (probes or {}).items():
            h = min(x.shape[2]-1, int((row+.5)*x.shape[2]/head_shape[0]))
            w = min(x.shape[3]-1, int((col+.5)*x.shape[3]/head_shape[1]))
            sites[str(cid)] = dict(feature_cell=[h, w], weights=weights[:, 0, h, w].tolist(),
                original_weights=original[:, 0, h, w].tolist(),
                dot_product_logits=used_logits[:, 0, h, w].tolist(),
                source_feature_norms=x[:, :, h, w].norm(dim=1).tolist())
        stats.append(dict(scale=level_index, anchor_site_probes=sites,
            probe_note='Normalized anchor-center to scale-grid cell mapping, not the entire detector receptive field. Uniform mode ignores recorded dot-product logits.',
            mean_source_weights=weights.mean((1, 2, 3)).tolist(),
            mean_abs_weight_change=float((weights-original).abs().mean())))
    base = model.engine.base
    joined = torch.cat([d(x) for d, x in zip(base.backbone.deblocks, fused)], 1)
    return dict(psm=base.cls_head(joined), rm=base.reg_head(joined)), stats


class FrameAudit:
    def __init__(self, ds, batch, predictions, a_rows, out, index, save_dense=False):
        self.ds, self.batch, self.predictions = ds, batch, predictions
        self.a_rows, self.out, self.index = a_rows, out, index
        self.targets = sorted(a_rows)
        self.watched = watched_anchors(a_rows)
        self.threshold = ds.post_processor.params['target_args']['score_threshold']
        self.save_dense = save_dense
        self.branches = []
        self.full_trace = trace_branch(ds, batch, predictions['full'])
        self.gt = self.full_trace['gt']
        self.baseline_ids = set(int(j) for j in self.full_trace['assignment'] if j >= 0)
        ids = self.full_trace['ids']['range'][self.full_trace['assignment'] < 0]
        self.baseline_fp_boxes = self.full_trace['corners'][ids]
        self.baseline_fp = len(ids)
        self.full_targets = compact_targets(self.full_trace, self.targets, self.watched, self.threshold)
        for j, a in a_rows.items():
            if self.full_targets[str(j)]['failure_stage'] != a['failure_stage']:
                raise AssertionError('A/B disappearance stage mismatch')

    def add(self, name, family, prediction, subset=None, dataset=None, trace=None, **metadata):
        started = time.monotonic()
        ds = dataset or self.ds
        tr = trace if trace is not None else trace_branch(ds, self.batch, prediction)
        np.testing.assert_allclose(tr['gt'], self.gt, rtol=0, atol=1e-6)
        matched = set(int(j) for j in tr['assignment'] if j >= 0)
        fids = tr['ids']['range'][tr['assignment'] < 0]
        correspondence = greedy_assignment(tr['scores'][fids],
            polygon_ious(tr['corners'][fids], self.baseline_fp_boxes))
        threshold = ds.post_processor.params['target_args']['score_threshold']
        details = compact_targets(tr, self.targets, self.watched, threshold)
        row = dict(name=name, family=family, subset=subset, targets=details,
            recovered_candidates=sorted(set(self.targets) & matched),
            lost_baseline_gt=sorted(self.baseline_ids-matched), matched_gt=sorted(matched),
            total_matched_gt=len(matched), frame_fp=len(fids), fp_count_delta=len(fids)-self.baseline_fp,
            new_fp_count=int((correspondence < 0).sum()),
            final_candidate_ids=tr['ids']['range'].tolist(),
            final_boxes=tr['corners'][tr['ids']['range']].tolist(),
            final_scores=tr['scores'][tr['ids']['range']].tolist(),
            final_assigned_gt=tr['assignment'].tolist(),
            versus_full=compare_targets(self.full_targets, details), **metadata)
        if self.save_dense:
            path = self.out/'proposals'/f'{self.index}_{name}.npz'
            path.parent.mkdir(exist_ok=True)
            save_proposals(path, tr, self.targets)
            row['proposals'] = str(path.relative_to(self.out))
        self.branches.append(row)
        row['trace_seconds'] = time.monotonic()-started
        print(f'frame={self.index} branch={name} recovered={len(row["recovered_candidates"])} '
              f'lost={len(row["lost_baseline_gt"])} new_fp={row["new_fp_count"]} '
              f'trace_seconds={row["trace_seconds"]:.2f}', flush=True)
        return row

    def interventions(self, model, encoded):
        import torch
        full = self.predictions['full']
        anchor_shape = self.full_trace['anchor_shape'][:-1]
        if len(anchor_shape) != 3 or list(anchor_shape[:2]) != list(full['psm'].shape[-2:]):
            raise ValueError('Expected H/W/anchor layout for attention probes')
        probes = {cid: np.unravel_index(cid, anchor_shape)[:2]
                  for ids in self.watched.values() for cid in ids}
        def weighted(mode, peer=0, scale=None):
            return weight_prediction(model, encoded, mode, peer, scale, probes, anchor_shape[:2])
        identity, self.baseline_weights = weighted('identity')
        for key in ('psm', 'rm'):
            torch.testing.assert_close(identity[key], full[key], atol=2e-4, rtol=2e-4)
        # An identity intervention must also preserve the actual final decisions.
        identity_trace = trace_branch(self.ds, self.batch, identity)
        for key in ('assignment',):
            np.testing.assert_array_equal(identity_trace[key], self.full_trace[key])
        np.testing.assert_array_equal(identity_trace['ids']['range'], self.full_trace['ids']['range'])
        del identity_trace, identity
        peers = sorted({p['peer_index'] for a in self.a_rows.values() for p in a['peers']})
        for peer in peers:
            single = self.predictions['peer_'+str(peer)]
            self.add(f'peer_{peer}', 'peer_alone', single, peer=peer)
            self.add(f'peer_score_{peer}', 'peer_score_full_geometry', dict(full, psm=single['psm']), peer=peer)
            self.add(f'peer_geometry_{peer}', 'full_score_peer_geometry', dict(full, rm=single['rm']), peer=peer)
            # All scales + one scale at a time: localize which level responds.
            for scale in [None] + list(range(len(encoded['levels']))):
                pred, stats = weighted('peer_query', peer, scale)
                label = 'all' if scale is None else str(scale)
                self.add(f'query_{peer}_scale_{label}', 'peer_query_scale_'+label, pred,
                         peer=peer, weight_statistics=stats,
                         interpretation='Only query changes; keys/values fixed. Rescue is not proof the original query is uniquely causal.')
        for mode in ('uniform', 'zero_ego_value'):
            pred, stats = weighted(mode)
            self.add(mode, mode, pred, weight_statistics=stats)
        # Predeclared full-output factorial sweep. No model forward, no test tuning.
        base_nms = float(self.ds.post_processor.params['nms_thresh'])
        for score in sorted({float(self.threshold), .05, .1}):
            for nms in sorted({base_nms, .3, .5}):
                if score == self.threshold and nms == base_nms:
                    continue
                local = copy.copy(self.ds)
                local.post_processor = copy.copy(self.ds.post_processor)
                local.post_processor.params = copy.deepcopy(self.ds.post_processor.params)
                local.post_processor.params['target_args']['score_threshold'] = score
                local.post_processor.params['nms_thresh'] = nms
                self.add(f'score_{score:g}_nms_{nms:g}', 'postprocess', full, dataset=local,
                         score_threshold=score, nms_threshold=nms)

    def result(self, weather, input_hash):
        peers = {b['peer']: b for b in self.branches if b['family'] == 'peer_alone'}
        pairs = []
        for b in self.branches:
            if b['family'] == 'subset' and len(b['subset']) == 2 and b['subset'][1] in peers:
                peer = b['subset'][1]
                pairs.append(dict(peer=peer, targets=compare_targets(peers[peer]['targets'], b['targets'])))
        return dict(sample_index=self.index, weather=weather, input_sha256=input_hash,
            candidate_targets=self.targets, target_metadata={str(j): {k: a[k] for k in
                ('scene', 'distance', 'failure_stage')} for j, a in self.a_rows.items()},
            baseline_targets=self.full_targets, baseline_weight_statistics=self.baseline_weights, branches=self.branches,
            peer_to_ego_peer=pairs, source_addition_edges=subset_edges(self.branches),
            warning='Artificial frozen-feature interventions, selected failure frames only. Not AP; source indices are frame-local.')
