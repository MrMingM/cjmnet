"""Scenario-block uncertainty and out-of-scene statistical controls."""
import numpy as np


def ratio_interval(blocks):
    values = np.asarray(list(blocks.values()), dtype=float)
    if not len(values) or not values[:,1].sum():
        return dict(value=None, interval95=None, scenes=len(values), n=0)
    draws = np.random.default_rng(20261112).integers(len(values), size=(2000,len(values)))
    samples = values[draws].sum(axis=1)
    valid = samples[:,1]>0
    return dict(value=float(values[:,0].sum()/values[:,1].sum()),
                interval95=np.quantile(samples[valid,0]/samples[valid,1],[.025,.975]).tolist()
                if len(values)>1 and valid.any() else None, scenes=len(values), n=int(values[:,1].sum()))


def correlation(a,b):
    a,b=np.asarray(a),np.asarray(b)
    if len(a)<3:
        return None
    ac,bc=a-a.mean(0),b-b.mean(0)
    scale=(np.linalg.norm(ac,axis=0)*np.linalg.norm(bc,axis=0))
    return [float(x/y) if y>0 else None for x,y in zip((ac*bc).sum(0),scale)]


def correlation_blocks(a,b,scene):
    a,b,scene=np.asarray(a),np.asarray(b),np.asarray(scene)
    if len(a)<3:
        return dict(n=len(a),scenes=len(set(scene)),r=None,interval95=None)
    cells=[]
    for s in sorted(set(scene)):
        x,y=a[scene==s],b[scene==s]
        cells.append(np.stack((np.full(x.shape[1],len(x)),x.sum(0),y.sum(0),
                               (x*x).sum(0),(y*y).sum(0),(x*y).sum(0))))
    cells=np.asarray(cells)
    n,x,y,xx,yy,xy=cells[np.random.default_rng(20261112).integers(len(cells),size=(2000,len(cells)))].sum(1).transpose(1,0,2)
    den=np.sqrt(np.maximum((xx-x*x/n)*(yy-y*y/n),0))
    draws=np.divide(xy-x*y/n,den,out=np.full_like(den,np.nan),where=den>0)
    ci=[np.quantile(col[np.isfinite(col)],[.025,.975]).tolist() if len(cells)>1 and np.isfinite(col).any()
        else None for col in draws.T]
    return dict(n=len(a),scenes=len(cells),r=correlation(a,b),interval95=ci)
