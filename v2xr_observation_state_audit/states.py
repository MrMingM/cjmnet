"""Measurement-derived labels and present-frame geometry."""
import numpy as np
from scipy.spatial import cKDTree

STEP = .8


def cell_sums(points, origin, shape, weights=None):
    h,w=shape
    ij=np.floor((points[:,:2]-origin)/STEP).astype(int)
    valid=((ij>=0)&(ij<[w,h])).all(1)
    ij=ij[valid]
    return np.bincount(ij[:,1]*w+ij[:,0],weights=None if weights is None else weights[valid],
                       minlength=w*h).reshape(h,w)


def rays(points, centers):
    """Evidence near the horizontal plane of the ego sensor, not whole-volume occupancy."""
    xyz=points[:,:3].astype(float)
    length=np.linalg.norm(xyz,axis=1)
    use=length>.5
    unit=xyz[use]/length[use,None]
    distance=length[use]
    q=np.column_stack((centers,np.zeros(len(centers))))
    r=np.linalg.norm(q,axis=1)
    direction=q/np.maximum(r[:,None],.01)
    result=np.zeros((len(q),2),dtype=np.int32)
    if not len(unit):
        return result
    tree=cKDTree(unit)
    # Bound memory and exclude the near sensor region, where angular cones are very broad.
    for start in range(0,len(q),512):
        sl=slice(start,start+512)
        neighbors=tree.query_ball_point(direction[sl],np.minimum(.22,.42/np.maximum(r[sl],1.)))
        for index,ids in enumerate(neighbors,start):
            if r[index]<2 or not ids:
                continue
            ids=np.asarray(ids)
            along=unit[ids]@direction[index]
            end=distance[ids]*along
            # Count separate angular directions rather than repeated collinear returns.
            angles=np.rint(unit[ids]/.002).astype(int)
            result[index,0]=len(np.unique(angles[end>r[index]+.6],axis=0))
            result[index,1]=len(np.unique(angles[(end<r[index]-.6)&(end>r[index]-12)],axis=0))
    return result


def labels(lidar, radar, clean, ray):
    free=(lidar==0)&(ray[:,0]>=3)
    blocked=(lidar==0)&(ray[:,0]==0)&(ray[:,1]>=3)
    # Unknown outside returned directions is explicit. For A/C, negatives are occupied cells.
    a=np.where(free,1,np.where((lidar>0)|blocked,0,-1))
    c=np.where(blocked,1,np.where((lidar>0)|free,0,-1))
    b=np.where(clean>=3,(lidar==0).astype(int),-1)
    d=np.where(lidar==0,(radar>0).astype(int),-1)
    return np.column_stack((a,b,c,d))
