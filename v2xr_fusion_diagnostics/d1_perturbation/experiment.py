"""Factor the original AttFusion into weights and feature values."""
import torch
from diag_model import attention, fuse


def run(model, original, alternatives, known):
    changed = alternatives['neighbor']
    modes = {'base': original.prediction, 'neighbor_clean': changed.prediction}
    replacements = {name:[] for name in ('content','weights','combined')}
    measurements, raw = {}, {}
    errors = []
    for i,(before,after,base,native) in enumerate(zip(original.pre,changed.pre,original.post,model.backbone.fuse_modules)):
        torch.testing.assert_close(before[:1],after[:1],atol=2e-5,rtol=1e-5)
        hybrid = torch.cat((before[:1],after[1:]))
        a0,a1 = attention(before,native),attention(hybrid,native)
        torch.testing.assert_close(fuse(a0,before),base,atol=1e-6,rtol=1e-6)
        content,weights,combined = fuse(a0,hybrid),fuse(a1,before),fuse(a1,hybrid)
        torch.testing.assert_close(combined,changed.post[i],atol=2e-5,rtol=1e-5)
        errors.append(float((combined-changed.post[i]).abs().max()))
        for key,value in zip(replacements,(content,weights,combined)):
            replacements[key].append(value)
        shape=before.shape[-2:]
        row0=a0[:,0,:].T.reshape(1,-1,*shape)
        row1=a1[:,0,:].T.reshape(1,-1,*shape)
        neighbor_before=before[1:] if len(before)>1 else before[:1]*0
        neighbor_after=hybrid[1:] if len(before)>1 else neighbor_before
        measurements[i] = {
            'neighbor_input':(neighbor_after-neighbor_before,neighbor_before),
            'attention_ego_row':(row1-row0,row0),
            'content':(content-base,base),'weights':(weights-base,base),
            'combined':(combined-base,base),
            'interaction':(combined-content-weights+base,base),
        }
        raw[i] = dict(pre_weather=before,pre_clean_neighbor=hybrid,attention_weather=a0,
                      attention_clean_neighbor=a1,post_weather=base,post_content=content,
                      post_weights=weights,post_combined=combined)
    for name,values in replacements.items():
        modes[name] = model.predict(values)
        if known:
            for i,value in enumerate(values):
                per_scale=list(original.post)
                per_scale[i]=value
                modes[f'{name}_s{i}']=model.predict(per_scale)
    return modes,measurements,raw,dict(combined_vs_native_max=max(errors))
