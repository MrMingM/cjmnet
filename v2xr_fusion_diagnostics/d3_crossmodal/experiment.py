"""Measure LiDAR context by Radar presence interactions before and after fusion."""
import torch


def run(model, original, alternatives, known):
    w0=alternatives['weather_r0']
    if torch.count_nonzero(w0.spatial[:,64:]):
        raise ValueError('Empty radar produced a nonzero scatter feature')
    modes={'base':original.prediction,'weather_r0':w0.prediction}
    measurement={i:{} for i in (-1,0,1,2)}
    raw={i:{} for i in (-1,0,1,2)}
    for scope in ('ego','neighbor','both'):
        c1,c0=(alternatives[f'{scope}_r{r}'] for r in (1,0))
        modes[f'{scope}_r1'],modes[f'{scope}_r0']=c1.prediction,c0.prediction
        if not torch.equal(c1.spatial[:,64:],original.spatial[:,64:]):
            raise ValueError('Radar encoder output changed with LiDAR-only intervention')
        if torch.count_nonzero(c0.spatial[:,64:]):
            raise ValueError('Empty radar branch is nonzero')
        if not torch.equal(c1.spatial[:,:64],c0.spatial[:,:64]):
            raise ValueError('Radar switch changed LiDAR encoder output')
        if not torch.equal(w0.spatial[:,:64],original.spatial[:,:64]):
            raise ValueError('Radar switch changed weather LiDAR encoder output')
        for i in (-1,0,1,2):
            stages = {'input':(original.spatial,w0.spatial,c1.spatial,c0.spatial)} if i==-1 else {
                'pre':tuple(s.pre[i][:1] for s in (original,w0,c1,c0)),
                'neighbor_pre':tuple(s.pre[i][1:] for s in (original,w0,c1,c0)),
                'post':tuple(s.post[i] for s in (original,w0,c1,c0)),
            }
            for stage,(a,b,c,d) in stages.items():
                prefix=f'{scope}_{stage}'
                measurement[i].update({
                    prefix+'_lidar_r1':(c-a,a),prefix+'_lidar_r0':(d-b,b),
                    prefix+'_radar_weather':(a-b,a),prefix+'_radar_clean':(c-d,c),
                    prefix+'_interaction':((c-a)-(d-b),a),
                })
                if known:
                    raw[i].update({prefix+'_'+key:value for key,value in zip(('w1','w0','c1','c0'),(a,b,c,d))})
    return modes,measurement,raw,dict(radar_features_equal=True,empty_radar_zero=True)
