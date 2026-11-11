"""Substitute exact single-vehicle or paired-context fusion outputs."""
from diag_model import error
import torch


def run(model, original, alternatives, known):
    changed, single = alternatives['both'], alternatives['single']
    solo = model.ego_fusion(original)
    modes = {'base':original.prediction,'true_single':single.prediction,
             'single_all':model.predict(solo),'context_all':model.predict(changed.post)}
    measurement,raw={},{}
    for i,(pre,multi,one,context) in enumerate(zip(original.pre,original.post,solo,changed.post)):
        for name,value in (('single',one),('context',context)):
            per_scale=list(original.post)
            per_scale[i]=value
            modes[f'{name}_s{i}']=model.predict(per_scale)
        measurement[i]=dict(
            collaboration=(multi-one,one),
            context_fused=(context-multi,multi),
            context_ego_pre=(changed.pre[i][:1]-pre[:1],pre[:1]),
            context_neighbor_pre=(changed.pre[i][1:]-pre[1:],pre[1:]),
            single_batch_pre=(pre[:1]-single.pre[i],single.pre[i]),
            single_batch_post=(one-single.post[i],single.post[i]))
        raw[i]=dict(pre_weather=pre,pre_context=changed.pre[i],post_multi=multi,
                    post_solo=one,post_true_single=single.post[i],post_context=context)
    same=error(modes['single_all'],modes['true_single'])
    return modes,measurement,raw,dict(layer_single_vs_true_single_max=same,
        layer_single_close=all(torch.allclose(modes['single_all'][k],modes['true_single'][k],
                                             atol=2e-5,rtol=1e-5) for k in ('psm','rm')))
