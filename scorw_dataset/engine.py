"""Amortize world-field queries; every return still runs the unchanged R3 receiver."""
import math
import numpy as np
from scorw_repair.response import single_response,validate
from shared_weather_benchmark.geometry import pose_matrix
from scorw_dataset.core import identity

ALGORITHM="r3-receiver-prefetched-identical-field-queries-v1"


def queries(origin,end,c,max_query_points):
    distance=float(np.linalg.norm(end-origin))
    if not math.isfinite(distance) or distance<=1e-5:raise ValueError("Invalid original ray; no silent deletion")
    n=max(1,math.ceil(distance/c["response_step_m"]))
    endpoints=end[None,:]
    length=np.linalg.norm(endpoints-origin,axis=1)
    k=int(np.maximum(1,np.ceil(length/c["attenuation_step_m"]).astype(np.int64))[0])
    if n+1+k>max_query_points:raise MemoryError("Single ray exceeds frozen query memory budget")
    s=(np.arange(n)+.5)*(distance/n)
    prefix=origin+(end-origin)[None,:]*(s/distance)[:,None]
    u=(np.arange(k,dtype=np.float64)+.5)/k
    hard=origin+(endpoints-origin)[:,None,:]*u[None,:,None]
    return [prefix,endpoints,hard.reshape(-1,3)]


class Prefetched:
    def __init__(self,origin,origin_alpha,positions,values):
        self.origin=origin;self.origin_alpha=origin_alpha;self.positions=positions;self.values=values;self.slot=0
    def alpha_numpy(self,points):
        k=self.slot;self.slot+=1
        if k>=len(self.positions) or not np.array_equal(points,self.positions[k]):
            raise ValueError("Receiver query changed; prefetch optimization is no longer equivalent")
        return self.values[k]
    def alpha(self,point,t=0):
        if t!=0 or not np.array_equal(point,self.origin):raise ValueError("Unexpected scalar receiver query")
        return self.origin_alpha


def simulate(raw,pose,field,receiver,key,batch_size=16,max_query_points=200000):
    raw=np.asarray(raw)
    if raw.ndim!=2 or raw.shape[1]!=4 or not np.isfinite(raw).all() or (raw[:,3]<0).any() or (raw[:,3]>1).any():
        raise ValueError("Invalid Clean XYZI")
    m=np.asarray(pose_matrix(pose));origin=m[:3,3]
    world=raw[:,:3].astype(float)@m[:3,:3].T+origin
    n=len(raw);weather=raw.astype(float).copy()
    kinds=np.zeros(n,dtype=np.uint8);emitted=np.zeros(n,dtype=bool);tau=np.zeros(n);hard=np.zeros(n);soft=np.zeros(n)
    overlap=np.zeros(n,dtype=bool);origin_alpha=field.alpha(origin.tolist())
    offset=0
    while offset<n:
        prepared=[];count=0
        for i in range(offset,min(n,offset+batch_size)):
            q=queries(origin,world[i],receiver,max_query_points);size=sum(len(p) for p in q)
            if size>max_query_points:raise MemoryError("Single ray exceeds frozen query memory budget")
            if prepared and count+size>max_query_points:break
            prepared.append(q);count+=size
        positions=[p for q in prepared for p in q]
        flat=field.alpha_numpy(np.concatenate(positions,axis=0))
        cursor=0
        for k,q in enumerate(prepared):
            values=[]
            for p in q:values.append(flat[cursor:cursor+len(p)]);cursor+=len(p)
            cached=Prefetched(origin,origin_alpha,q,values);i=offset+k
            row=single_response(cached,origin,world[i],float(raw[i,3]),receiver)
            if cached.slot!=3:raise ValueError("Incomplete prefetched receiver query sequence")
            kinds[i]=row["origin"];emitted[i]=row.get("emitted",True);tau[i]=row["tau"]
            hard[i]=row["hard_signal"];soft[i]=row["soft_signal"];overlap[i]=row["unidentifiable_overlap"]
            if kinds[i]!=0:weather[i,:3]=raw[i,:3]*(row["range_m"]/float(np.linalg.norm(raw[i,:3])))
            weather[i,3]=row["intensity"]
        offset+=len(prepared)
    out=dict(original_xyzi=raw.copy(),weather_xyzi=weather.astype(np.float32),emitted=emitted,origin=kinds,
        retained=emitted&(kinds==0),integral=tau,hard_signal=hard,soft_signal=soft,overlap_unknown=overlap,
        original_index=np.arange(n,dtype=np.int64),point_identity=np.asarray([identity([*key,i]) for i in range(n)],dtype="U64"),
        frame_identity=np.asarray(key))
    validate(out)
    return out
