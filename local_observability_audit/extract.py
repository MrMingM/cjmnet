"""GT-assisted mechanism features and independent prediction matching."""
import itertools
import numpy as np
from . import coordinates as co
from .observability import information

def source_rows(batch, metadata, reliability, hypes, record, weather, cfg):
    processed = {k:v.detach().cpu().numpy() for k,v in batch['ego']['processed_lidar'].items()}
    sources = co.retained_points(processed, reliability, len(metadata),
        hypes['preprocess']['args']['voxel_size'],hypes['preprocess']['cav_lidar_range'])
    ego = batch['ego']
    mask = ego['object_bbx_mask'].detach().cpu().numpy()[0].astype(bool)
    boxes = co.canonical(ego['object_bbx_center'].detach().cpu().numpy()[0][mask],hypes['postprocess']['order'])
    rows = []
    for object_id, box in zip(ego['object_ids'], boxes):
        for i, (points, r, u) in enumerate(sources):
            select = co.crop(points,box,cfg['roi_expansion'])
            p, r, u = points[select], r[select], u[select]
            source = metadata[i]
            origin = source['transform'][:3,3]
            delta = origin[:2]-box[:2]
            angle = (np.arctan2(delta[1],delta[0])-box[6]+np.pi)%(2*np.pi)-np.pi
            obs = information(p[:,:2],co.bev_state(box),r,cfg)
            rows.append({'frame_id': record['frame_id'], 'scene_id': record['scene_id'],
                'weather':weather, 'gt_object_id':str(object_id), 'gt_box':box,
                'source_id':source['source_id'], 'source_index':i, 'is_ego':i==0,
                'source_pose':source['pose'], 'source_to_ego':source['transform'],
                'source_distance':float(np.linalg.norm(delta)), 'viewing_angle':float(angle),
                'point_count':int(co.crop(source['points'],box,cfg['roi_expansion']).sum()),
                'valid_point_count':len(p), 'measurement_points':'retained_voxel_slots',
                'mean_gspr_reliability':float(r.mean()) if len(r) else None,
                'mean_gspr_uncertainty':float(u.mean()) if len(u) else None,
                'gspr_valid':bool(len(p)), 'gspr_context':'full_selected_frame_forward',
                'observability':obs})
    return rows, boxes, [str(x) for x in ego['object_ids']]

def match_predictions(prediction, boxes, cfg):
    from shapely.geometry import Polygon
    corners = np.asarray(prediction['corners']).reshape(-1,8,3)
    scores = np.asarray(prediction['scores'])
    if len(scores)!=len(corners) or not np.isfinite(corners).all() or not np.isfinite(scores).all():
        raise ValueError('Invalid predictions')
    pboxes = [co.corners_to_box(c) for c in corners]
    pp = [Polygon(c[:4,:2]) for c in corners]
    gp = [Polygon(co.box_corners(b)[:4,:2]) for b in boxes]
    matches, taken = {}, set()
    # Standard score-ordered, one-to-one BEV IoU matching; same threshold in every subset.
    for i in sorted(range(len(scores)),key=lambda k:(-scores[k],k)):
        candidates=[]
        for j, g in enumerate(gp):
            if j in taken:
                continue
            inter=pp[i].intersection(g).area
            iou=inter/max(pp[i].area+g.area-inter,1e-12)
            candidates.append((iou,-j,j,inter))
        if not candidates:
            continue
        iou, _, j, area = max(candidates)
        if iou < cfg['match_bev_iou']:
            continue
        taken.add(j)
        pred, gt = pboxes[i], boxes[j]
        err=np.abs(co.bev_state(pred)-co.bev_state(gt))
        err[4]=co.angle_error(pred[6],gt[6],cfg['angle_period'])
        overlap=max(0.,min(pred[2]+pred[5]/2,gt[2]+gt[5]/2)-max(pred[2]-pred[5]/2,gt[2]-gt[5]/2))
        inter3=area*overlap
        iou3=inter3/max(np.prod(pred[3:6])+np.prod(gt[3:6])-inter3,1e-12)
        matches[j]={'prediction_index':i,'score':float(scores[i]),'errors':err,
            'box':pred,'bev_iou':float(iou),'iou3d':float(iou3),
            'mean_scaled_error':float(np.mean(err/np.asarray(cfg['parameter_scales'])))}
    return matches

def contributions(predictions, boxes, ids, n, cfg):
    matches={k:match_predictions(p,boxes,cfg) for k,p in predictions.items()}
    singles,pairs=[],[]
    for o, object_id in enumerate(ids):
        base=matches['0'].get(o)
        for i in range(1,n):
            added=matches[f'0,{i}'].get(o)
            common=base is not None and added is not None
            singles.append({'gt_object_id':object_id,'source_index':i,'regression_valid':common,
                'ego_detected':base is not None,'added_detected':added is not None,
                'recovery':base is None and added is not None,'lost':base is not None and added is None,
                'utility':base['errors']-added['errors'] if common else None,
                'ego_match':base,'added_match':added})
        for i,j in itertools.combinations(range(1,n),2):
            a,b,c=(matches[k].get(o) for k in (f'0,{i}',f'0,{j}',f'0,{i},{j}'))
            common=all(x is not None for x in (a,b,c))
            pairs.append({'gt_object_id':object_id,'i':i,'j':j,'regression_valid':common,
                'utility':np.minimum(a['errors'],b['errors'])-c['errors'] if common else None,
                'gain':min(a['mean_scaled_error'],b['mean_scaled_error'])-c['mean_scaled_error'] if common else None,
                'a_match':a,'b_match':b,'pair_match':c})
    return {'singles':singles,'pairs':pairs,'match_policy':'score-ordered one-to-one BEV IoU',
            'object_count':len(ids),'subset_count':len(predictions)}
