"""Layer observations for native and explicitly labeled BN counterfactuals; no training/scoring."""
import numpy as np


def batchnorm_observation(mean, variance, running_mean, running_var, gamma, beta,
                          input_samples, output_samples, eps, use_batch_statistics=False):
    """Describe distribution drift and check sampled eval arithmetic using CPU float64."""
    arrays = [np.asarray(x, dtype=np.float64) for x in
              (mean, variance, running_mean, running_var, gamma, beta)]
    mean, variance, running_mean, running_var, gamma, beta = arrays
    if any(x.ndim != 1 or x.shape != mean.shape for x in arrays):
        raise ValueError('BatchNorm channel statistics do not align')
    input_samples, output_samples = (np.asarray(x, dtype=np.float64) for x in
                                     (input_samples, output_samples))
    if input_samples.shape != output_samples.shape or input_samples.ndim != 2 or input_samples.shape[0] != len(mean):
        raise ValueError('BatchNorm sampled input/output do not align')
    if not np.isfinite(eps) or eps <= 0 or np.any(running_var < 0) or np.any(variance < 0):
        raise ValueError('Invalid BatchNorm variance/epsilon')
    denominator = np.sqrt(running_var + eps)
    used_mean, used_var = (mean, variance) if use_batch_statistics else (running_mean, running_var)
    expected = ((input_samples - used_mean[:, None]) / np.sqrt(used_var[:, None] + eps)
                * gamma[:, None] + beta[:, None])
    error = np.abs(output_samples - expected)
    finite = all(np.isfinite(x).all() for x in (*arrays, input_samples, output_samples, expected))
    formula = dict(finite=bool(finite), samples=int(input_samples.size),
        basis='current_batch_population' if use_batch_statistics else 'stored_running',
        max_absolute_error=float(error.max()),
        max_relative_error_with_floor_1=float((error / np.maximum(np.abs(expected), 1.)).max()),
        elementwise_close=bool(finite and np.all(error <= 1e-5 + 1e-4 * np.abs(expected))),
        scope='batch index 0; up to 8 evenly spaced spatial/point positions per channel; CPU float64 formula; diagnostic only')
    return dict(batch_mean=mean.tolist(), batch_population_variance=variance.tolist(),
        running_mean=running_mean.tolist(), running_variance=running_var.tolist(),
        batch_to_running_std_ratio_max=float(np.sqrt((variance + eps) / (running_var + eps)).max()),
        mean_offset_in_running_std_max=float((np.abs(mean - running_mean) / denominator).max()),
        gamma_abs_max=float(np.abs(gamma).max()), beta_abs_max=float(np.abs(beta).max()),
        sampled_normalization_formula=formula,
        sampled_eval_formula=None if use_batch_statistics else formula)


def trace_summary(rows):
    """Threshold crossings describe magnitude only; they never accept/reject an experiment."""
    observations = []
    for row in rows:
        for slot, value in row.get('outputs', {}).items():
            maximum = value.get('finite_abs_max')
            if maximum is not None and np.isfinite(maximum):
                observations.append(dict(index=row['index'], module=row['module'],
                    operator=row['operator'], slot=slot, absolute_maximum=maximum))
    first = {}
    for threshold in (100, 1000, 1000000):
        first[str(threshold)] = next((dict(value) for value in observations
                                     if value['absolute_maximum'] >= threshold), None)
    bn = [dict(index=row['index'], module=row['module'],
        std_ratio=row['BN']['batch_to_running_std_ratio_max'],
        mean_offset=row['BN']['mean_offset_in_running_std_max'],
        formula=row['BN']['sampled_normalization_formula']) for row in rows if 'BN' in row]
    return dict(observed_calls=len(rows), first_observed_output_at_or_above=first,
        largest_outputs=sorted(observations, key=lambda v: v['absolute_maximum'], reverse=True)[:8],
        BN_observed_calls=len(bn),
        BN_largest_std_mismatch=sorted(bn, key=lambda value: value['std_ratio'], reverse=True)[:5],
        BN_sample_formula_failures=[value for value in bn if not value['formula']['elementwise_close']],
        nonfinite_outputs=[dict(module=row['module'], slot=slot) for row in rows
                           for slot, value in row.get('outputs', {}).items() if value.get('all_finite') is False],
        observation_errors=[dict(module=row['module'], error=row['observation_error'])
                            for row in rows if 'observation_error' in row],
        evaluation_or_selection_gate=False,
        limitation='Hooks and sampled formulas are observations, not a causal proof. Any current-batch counterfactual is explicitly labeled in the enclosing report. No training, checkpoint updates or AP acceptance.')


class LayerTrace:
    def __init__(self, batch_statistics=False):
        self.rows = []
        self.bn_before = None
        self.bn_after = None
        self.state_observation_errors = []
        self.batch_statistics = batch_statistics
        self.model_before = None
        self.model_after = None

    def install(self, model, stack):
        import torch
        from torch import nn
        from nms_numeric import array_hash

        from nms_common import identity

        def fingerprints():
            all_state = {name: array_hash(value.detach().cpu().numpy()) for name, value in model.state_dict().items()}
            bn = {name: value for name, value in all_state.items()
                  if name.endswith(('running_mean', 'running_var', 'num_batches_tracked'))}
            return identity(all_state), bn

        def before(module, inputs):
            try:
                self.model_before, self.bn_before = fingerprints()
            except Exception as error:
                self.state_observation_errors.append(repr(error))
            return None

        def after(module, inputs, output):
            try:
                self.model_after, self.bn_after = fingerprints()
            except Exception as error:
                self.state_observation_errors.append(repr(error))
            return None

        stack.callback(model.register_forward_pre_hook(before).remove)
        stack.callback(model.register_forward_hook(after).remove)

        def describe(value):
            if not torch.is_tensor(value):
                return {}
            x = value.detach()
            if not x.numel():
                return dict(shape=list(x.shape), dtype=str(x.dtype), elements=0,
                            finite_abs_max=None)
            low, high = torch.aminmax(x)
            low, high = float(low), float(high)
            return dict(shape=list(x.shape), dtype=str(x.dtype), elements=x.numel(),
                minimum=low, maximum=high, finite_abs_max=max(abs(low), abs(high)),
                all_finite=bool(torch.isfinite(x).all()))

        def slots(value, label):
            if torch.is_tensor(value):
                return {label: describe(value)}
            if isinstance(value, dict):
                selected = ('pillar_features', 'spatial_features', 'spatial_features_2d', 'psm', 'rm')
                return {label + '/' + key: describe(value[key]) for key in selected
                        if key in value and torch.is_tensor(value[key])}
            if isinstance(value, (tuple, list)):
                return {label + '/' + str(i): describe(x) for i, x in enumerate(value) if torch.is_tensor(x)}
            return {}

        def observe(name):
            def hook(module, inputs, output):
                row = dict(index=len(self.rows), module=name,
                           operator=type(module).__module__ + '.' + type(module).__name__,
                           training=bool(module.training))
                self.rows.append(row)
                try:
                    # Some official feature modules mutate their dictionary argument in place.
                    row['input_observed_after_forward'] = slots(inputs, 'input')
                    row['outputs'] = slots(output, 'output')
                    if isinstance(module, nn.modules.batchnorm._BatchNorm):
                        if module.training or not module.track_running_stats:
                            raise ValueError('Observed BatchNorm is not the expected tracked eval operation')
                        x = inputs[0].detach()
                        dimensions = tuple(i for i in range(x.ndim) if i != 1)
                        variance, mean = torch.var_mean(x.float(), dim=dimensions, unbiased=False)
                        channels = x.shape[1]
                        first_x = x[0].reshape(channels, -1)
                        first_y = output.detach()[0].reshape(channels, -1)
                        positions = np.unique(np.linspace(0, first_x.shape[1] - 1,
                                                         min(8, first_x.shape[1]), dtype=np.int64)).tolist()
                        cpu = lambda value: value.detach().cpu().numpy()
                        gamma = np.ones(channels) if module.weight is None else cpu(module.weight)
                        beta = np.zeros(channels) if module.bias is None else cpu(module.bias)
                        row['BN'] = batchnorm_observation(cpu(mean), cpu(variance),
                            cpu(module.running_mean), cpu(module.running_var), gamma, beta,
                            cpu(first_x[:, positions]), cpu(first_y[:, positions]), module.eps,
                            use_batch_statistics=self.batch_statistics)
                        row['BN'].update(momentum=module.momentum, eps=module.eps,
                            num_batches_tracked=int(module.num_batches_tracked))
                except Exception as error:
                    # An observer failure must not replace, repair or suppress native model output.
                    row['observation_error'] = repr(error)
                return None
            return hook

        types = (nn.Conv2d, nn.ConvTranspose2d, nn.Linear, nn.modules.batchnorm._BatchNorm)
        for name, module in model.named_modules():
            group = name in ('pillar_vfe', 'scatter', 'backbone', 'cls_head', 'reg_head')
            fusion = name.startswith('backbone.fuse_modules.') and name.count('.') == 2
            if isinstance(module, types) or group or fusion:
                handle = module.register_forward_hook(observe(name))
                stack.callback(handle.remove)

    def report(self):
        result = trace_summary(self.rows)
        result.update(BN_buffers_exact_after_forward=(self.bn_before == self.bn_after
            if self.bn_before is not None and self.bn_after is not None else None),
            all_model_state_exact_after_forward=(self.model_before == self.model_after
                if self.model_before is not None and self.model_after is not None else None),
            model_state_before_sha256=self.model_before, model_state_after_sha256=self.model_after,
            state_observation_errors=self.state_observation_errors)
        return dict(summary=result, rows=self.rows, no_training=True,
                    eligible_for_AP_or_selection=False, numerical_values_modified=self.batch_statistics,
                    BN_current_batch_counterfactual=self.batch_statistics)
