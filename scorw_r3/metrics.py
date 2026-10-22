"""Exact greedy score-order BEV matching and VOC2010 global AP on fixed GT."""
import numpy as np
from scorw_r3.core import Blocked


def match_matrix(ious,scores,threshold):
    a=np.asarray(ious);scores=np.asarray(scores)
    if a.ndim!=2 or len(a)!=len(scores) or not np.isfinite(a).all() or not np.isfinite(scores).all():
        raise Blocked("R3_INVALID_MATCH_MATRIX")
    available=list(range(a.shape[1]));tp=[];fp=[];matched=[];assigned={};false=[]
    order=np.argsort(-scores)
    for p in order:
        values=a[p,available]
        if len(values) and max(values)>=threshold:
            g=available.pop(int(np.argmax(values)));tp.append(1);fp.append(0);matched.append(g);assigned[str(g)]=int(p)
        else:tp.append(0);fp.append(1);false.append(int(p))
    return dict(tp=tp,fp=fp,score=scores[order].tolist(),gt=a.shape[1],matched_gt=matched,
        assignments=assigned,unmatched_prediction=false,TP=sum(tp),FP=sum(fp),FN=a.shape[1]-sum(tp))


def ap(records):
    rows=list(records);gt=sum(r["gt"] for r in rows)
    if not gt:return None
    score=np.asarray([v for r in rows for v in r["score"]]);order=np.argsort(-score)
    tp=np.asarray([v for r in rows for v in r["tp"]])[order];fp=np.asarray([v for r in rows for v in r["fp"]])[order]
    rec=(np.cumsum(tp)/gt).tolist();prec=(np.cumsum(tp)/np.maximum(1,np.cumsum(tp+fp))).tolist()
    mrec=[0.,*rec,1.];mpre=[0.,*prec,0.]
    for k in range(len(mpre)-2,-1,-1):mpre[k]=max(mpre[k],mpre[k+1])
    return float(sum((mrec[k]-mrec[k-1])*mpre[k] for k in range(1,len(mrec)) if mrec[k]!=mrec[k-1]))


def iou_matrix(boxes,gt,audit):
    from opencood.utils import common_utils
    p=common_utils.convert_format(np.asarray(boxes)) if len(boxes) else []
    g=common_utils.convert_format(np.asarray(gt)) if len(gt) else []
    return np.asarray([audit.call(common_utils.compute_iou,b,g) for b in p],dtype=float).reshape(len(p),len(g))


def target_category(ego,neighbors,fusion):
    neighbor=any(neighbors)
    if not ego and neighbor:return "A_neighbor_detects_fusion_recovers" if fusion else "B_neighbor_detects_fusion_misses"
    if not ego and not neighbor and not fusion:return "C_all_miss"
    if ego and not fusion:return "D_ego_detects_fusion_loses"
    if not ego and fusion:return "fusion_only_recovery"
    return "ego_and_fusion_detect"
