"""Fixed grid masks, raw/voxel counts and paired frame bootstrap."""
import numpy as np
import torch
from opencood.pcdet_utils.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_cpu


class Grid:
    def __init__(self, model, cfg):
        self.nx,self.ny = int(model.scatter.nx),int(model.scatter.ny)
        self.size=self.nx*self.ny
        self.dx,self.dy,_=cfg['model']['args']['voxel_size']
        self.x0,self.y0=cfg['model']['args']['lidar_range'][:2]
        x,y=self.x0+(np.arange(self.nx)+.5)*self.dx,self.y0+(np.arange(self.ny)+.5)*self.dy
        xx,yy=np.meshgrid(x,y)
        self.centers=np.column_stack((xx.ravel(),yy.ravel(),np.zeros(self.size))).astype(np.float32)

    def cell(self, xy):
        x=np.floor((xy[:,0]-self.x0)/self.dx).astype(int)
        y=np.floor((xy[:,1]-self.y0)/self.dy).astype(int)
        return y*self.nx+x, (x>=0)&(x<self.nx)&(y>=0)&(y<self.ny)

    def raw(self, ego, sensor):
        counts=np.zeros(self.size,dtype=np.int64)
        for points in ego['projected_'+sensor][0]:
            points=np.asarray(points)
            idx,valid=self.cell(points[:,:2])
            counts+=np.bincount(idx[valid],minlength=self.size)
        return counts

    def gt(self, targets):
        mask=np.zeros(self.size,dtype=bool)
        for target in targets:
            # Fixed official box convention h,w,l. Test horizontal cell centers.
            x,y,z,h,w,l,yaw=target['box']
            radius=float(np.hypot(w,l))/2
            pts=self.centers
            roi=(abs(pts[:,0]-x)<=radius)&(abs(pts[:,1]-y)<=radius)
            box=np.array([[x,y,0,l,w,2,yaw]],dtype=np.float32)
            mask[roi] |= np.asarray(points_in_boxes_cpu(pts[roi],box))[0].astype(bool)
        return mask

    def voxel(self, ego, sensor, weights=None):
        d=ego['processed_'+sensor]
        coords=d['voxel_coords'].detach().cpu().numpy()
        idx=coords[:,2]*self.nx+coords[:,3]
        if weights is None:
            weights=d['voxel_num_points'].detach().cpu().numpy()
        return np.bincount(idx,weights=weights,minlength=self.size)


def table(lidar,radar,gt):
    mask=(lidar>0).astype(int)*2+(radar<=0)
    return np.stack((np.bincount(mask,minlength=4),
                     np.bincount(mask[gt],minlength=4)),axis=-1)


def summarize(rows, groups):
    a=np.asarray(rows,dtype=np.int64)
    totals=a.sum(0)
    rates=totals[:,1]/totals[:,0]
    # Resample whole scenes to account for temporal dependence within each scene.
    scenes=np.stack([a[np.asarray(groups)==g].sum(0) for g in sorted(set(groups))])
    rng=np.random.default_rng(20261113)
    sampled=scenes[rng.integers(0,len(scenes),(4000,len(scenes)))].sum(1)
    risk=sampled[:,:,1]/sampled[:,:,0]
    lo,hi=np.quantile(risk[:,0]-risk[:,1],[.025,.975])
    return dict(rows=['L0_R1','L0_R0','L1_R1','L1_R0'],count=totals.tolist(),
        probability=rates.tolist(),scene_count=len(scenes),lift=float(rates[0]-rates[1]),
        scene_bootstrap_95ci=[float(lo),float(hi)])

