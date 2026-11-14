"""Six predicted-box cells, using ego-frame observed points only."""
import numpy as np
from v2xr_griq_experiments.features import Cloud

L = ('count','effective','mean_r','mean_u','occupancy','lidar_missing','gspr_missing')
R = ('count','density','spread','offset','background_density','radar_missing')
DL, DR = len(L),len(R)
CELLS = 6


def coordinates(points, box):
    a = points[:,:3]-box[:3]
    c,s = np.cos(box[6]),np.sin(box[6])
    return np.column_stack((c*a[:,0]+s*a[:,1],-s*a[:,0]+c*a[:,1],a[:,2]))


def grid(local, box, halo=False):
    # Length is divided into three bins, width into two. Boundary goes right/up.
    uv = local[:,:2]/box[[5,4]]
    if halo:
        uv = uv/1.5
    scaled = (uv+.5)*[3,2]
    rounded = np.rint(scaled)
    scaled = np.where(np.abs(scaled-rounded)<1e-7,rounded,scaled)
    x = np.clip(np.floor(scaled[:,0]),0,2).astype(int)
    y = np.clip(np.floor(scaled[:,1]),0,1).astype(int)
    return 3*y+x


def extract_local(boxes, context, inside):
    (ls,rs),xyz,r,u,_ = context
    lidar,radar,gspr = Cloud(ls),Cloud(rs),Cloud([xyz])
    arr = np.zeros((len(boxes),CELLS,DL+DR),dtype=np.float32)
    for i,box in enumerate(boxes):
        halo = box.copy()
        halo[4:6] *= 1.5
        radius = np.linalg.norm(halo[3:6])/2+1e-4
        lid,lp = lidar.near(box,radius)
        rid,rp = radar.near(box,radius)
        gid,gp = gspr.near(box,radius)
        gl = inside(gp,box)
        rv = inside(rp,box)
        local_r = coordinates(rp[rv],box)
        lc = grid(coordinates(lp[inside(lp,box)],box),box)
        gc = grid(coordinates(gp[gl],box),box)
        rc = grid(local_r,box)
        bg = grid(coordinates(rp[inside(rp,halo)&~rv],box),box,halo=True)
        area = float(box[4]*box[5])/CELLS
        for k in range(CELLS):
            ln = int(np.count_nonzero(lc==k))
            sub = gid[gl][gc==k]
            rp_k = local_r[rc==k]
            rn = len(rp_k)
            center = box[[5,4]] * np.array([(k%3+.5)/3-.5,(k//3+.5)/2-.5])
            offset = np.linalg.norm(rp_k[:,:2].mean(0)-center) if rn else 0.
            spread = np.linalg.norm(rp_k.std(0)) if rn else 0.
            arr[i,k] = [ln,np.sum(r[sub]*(1-u[sub])),r[sub].mean() if len(sub) else 0.,
                        u[sub].mean() if len(sub) else 0.,ln>0,ln==0,len(sub)==0,
                        rn,rn/area,spread,offset,np.count_nonzero(bg==k)/(area*1.25),rn==0]
    if not np.isfinite(arr).all():
        raise ValueError('Invalid local evidence')
    return arr


def intervene(x, kind, rng=None):
    x = x.copy()
    if kind=='permute':
        for i in range(len(x)):
            x[i,:,DL:] = x[i,rng.permutation(CELLS),DL:]
    elif kind=='flatten_radar':
        # Equalize evidence within the observed cells while retaining the occupancy and totals.
        radar = x[:,:,DL:]
        observed = radar[:,:,0]>0
        n = observed.sum(1,keepdims=True).clip(1)
        for j in (0,1,2,3):
            radar[:,:,j] = observed*((radar[:,:,j]*observed).sum(1,keepdims=True)/n)
        radar[:,:,4] = radar[:,:,4].mean(1,keepdims=True)
        radar[:,:,-1] = ~observed
    elif kind=='one_cell':
        l,r = x[:,:,:DL],x[:,:,DL:]
        g = ~l[:,:,6].astype(bool)
        observed = r[:,:,0]>0
        for j in (2,3):
            l[:,:,j] = ((l[:,:,j]*g).sum(1,keepdims=True)/g.sum(1,keepdims=True).clip(1))
            r[:,:,j] = ((r[:,:,j]*observed).sum(1,keepdims=True)/observed.sum(1,keepdims=True).clip(1))
        for j in (0,1):
            l[:,:,j] = l[:,:,j].mean(1,keepdims=True)
        for j in (0,1,4):
            r[:,:,j] = r[:,:,j].mean(1,keepdims=True)
        l[:,:,4] = l[:,:,0]>0
        l[:,:,5] = l[:,:,0]==0
        l[:,:,6] = ~g.any(1,keepdims=True)
        r[:,:,5] = r[:,:,0]==0
    return x
