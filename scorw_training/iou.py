"""Exact disjoint-envelope shortcut; positive candidates use frozen IoU arithmetic."""
from contextlib import contextmanager
import numpy as np


@contextmanager
def exact_iou(common,audit):
    original=common.compute_iou;cache={}
    def bounds(p):
        k=id(p)
        if k not in cache:
            if not p.is_valid or p.is_empty or not np.isfinite(p.area) or p.area<=0:
                raise ValueError('Invalid BEV polygon; no geometry repair')
            cache[k]=(p,p.bounds)
        return cache[k][1]
    def call(box,boxes):
        a=bounds(box);out=np.zeros(len(boxes),dtype=np.float32)
        if not len(boxes):return out
        b=np.asarray([bounds(p) for p in boxes])
        keep=np.flatnonzero((b[:,0]<a[2])&(b[:,2]>a[0])&(b[:,1]<a[3])&(b[:,3]>a[1]))
        if len(keep):out[keep]=audit.call(original,box,[boxes[i] for i in keep])
        return out
    common.compute_iou=call
    try:yield
    finally:common.compute_iou=original
