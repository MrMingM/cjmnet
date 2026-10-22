"""Small fixed-weight adapter: same R2 project-first preprocessing and postprocessor."""
import hashlib
import numpy as np

from scorw_r3.core import Blocked,identity,array_hash
from scorw_r3.metrics import iou_matrix,match_matrix
from scorw_repair.core import install_dataset_guard
from scorw_repair.range_adapter import literal_range_filter
from scorw_repair.iou_audit import IoUAudit
from shared_weather_benchmark.geometry import pose_matrix,inverse


def validate_project_first(h):
    """OpenCOOD uses args: [] for defaults; preserve its project-first meaning."""
    fusion=h.get("fusion")
    if not isinstance(fusion,dict):raise Blocked("R3_INVALID_FUSION_CONFIG")
    args=fusion.get("args",{})
    if isinstance(args,dict):
        if args.get("proj_first",True) is not True:raise Blocked("R3_PROJECT_FIRST_REQUIRED")
    elif not (isinstance(args,list) and not args):
        raise Blocked("R3_UNSUPPORTED_FUSION_ARGS: expected a mapping or the legacy empty list")
    return dict(proj_first=True,args_type=type(args).__name__,
        meaning="legacy empty list or absent option uses the frozen default proj_first=True")


class Runtime:
    def __init__(self,c,old,output):
        install_dataset_guard()
        import torch
        from attfuse_code.point_pillar_intermediate import PointPillarIntermediate
        from opencood.hypes_yaml.yaml_utils import load_yaml
        from opencood.data_utils.pre_processor import build_preprocessor
        from opencood.data_utils.post_processor import build_postprocessor
        from opencood.utils import pcd_utils,box_utils,common_utils
        if not torch.cuda.is_available():raise Blocked("R3_GPU_UNAVAILABLE")
        self.torch=torch;self.pcd=pcd_utils;self.box=box_utils;self.common=common_utils
        self.h=load_yaml(c["model"]["config"])
        if self.h["model"]["core_method"]!="point_pillar_intermediate":raise Blocked("R3_PURE_ATTFUSE_REQUIRED")
        fusion_config=validate_project_first(self.h)
        torch.manual_seed(old.config["seed"]);torch.cuda.manual_seed_all(old.config["seed"])
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        self.model=PointPillarIntermediate(self.h["model"]["args"]).eval()
        state=torch.load(c["model"]["checkpoint"],map_location="cpu",weights_only=False)
        self.model.load_state_dict(state,strict=True);del state
        self.initial_state=self.state_hash();original=self.model.scatter

        class EmptyAwareScatter(torch.nn.Module):
            def forward(self,batch):
                features=batch["pillar_features"].reshape(-1,original.num_bev_features);coords=batch["voxel_coords"]
                if len(features)!=len(coords):raise Blocked("R3_EMPTY_SCATTER_SHAPE_ERROR")
                count=int(batch["record_len"].sum().item())
                a=torch.zeros((count,original.num_bev_features,original.nz*original.nx*original.ny),dtype=features.dtype,device=features.device)
                for k in range(count):
                    mask=coords[:,0]==k;cc=coords[mask]
                    index=(cc[:,1]+cc[:,2]*original.nx+cc[:,3]).long();a[k,:,index]=features[mask].t()
                batch["spatial_features"]=a.view(count,original.num_bev_features*original.nz,original.ny,original.nx)
                return batch
        adapted=EmptyAwareScatter()
        test=dict(pillar_features=torch.arange(128,dtype=torch.float32).reshape(2,64),
            voxel_coords=torch.tensor([[0,0,0,0],[1,0,0,1]]),record_len=torch.tensor([2]))
        if not torch.equal(original(dict(test))["spatial_features"],adapted(dict(test))["spatial_features"]):
            raise Blocked("R3_SCATTER_NONEMPTY_EQUIVALENCE_FAILED")
        for feature,coords in ((torch.ones(64),torch.tensor([[0,0,0,0]])),(torch.empty((0,64)),torch.empty((0,4),dtype=torch.int64))):
            a=adapted(dict(pillar_features=feature,voxel_coords=coords,record_len=torch.tensor([2])))["spatial_features"]
            if a.shape[0]!=2 or torch.count_nonzero(a[1]).item():raise Blocked("R3_EMPTY_CAV_SLOT_ERROR")
        self.model.scatter=adapted
        self.pre=build_preprocessor(self.h["preprocess"],False);self.post=build_postprocessor(self.h["postprocess"],False)
        self.device=torch.device("cuda:0");self.model.to(self.device)
        self.anchors=torch.from_numpy(self.post.generate_anchor_box()).to(self.device)
        self.audit=IoUAudit(output)
        self.model_audit=dict(status="passed",weights=old.model,training=False,
            model_state_sha256=self.initial_state,seed=old.config["seed"],torch=torch.__version__,hip=torch.version.hip,
            gpu=torch.cuda.get_device_name(0),preprocess=self.h["preprocess"],postprocess=self.h["postprocess"],
            fusion_config=fusion_config,
            neighbor_frame="neighbor local cloud -> frozen ego coordinates -> same one-CAV detector and fixed ego GT",
            scatter="same R2 empty-slot adaptation; nonempty exact equivalence checked",REAL_WORLD_VALIDATION_MISSING=True)

    def state_hash(self):
        return identity({k:hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest() for k,v in sorted(self.model.state_dict().items())})

    def project(self,cloud,pose,ego_pose,original_indices=None):
        cloud=np.asarray(cloud,dtype=np.float32)
        indices=np.arange(len(cloud),dtype=np.int64) if original_indices is None else np.asarray(original_indices)
        if len(indices)!=len(cloud) or (len(indices) and max(indices)>=2**24):raise Blocked("R3_SOURCE_SLOT_ENCODING_ERROR")
        pc=np.column_stack((cloud,indices.astype(np.float32)))
        pc=self.pcd.mask_ego_points(pc.copy())
        m=np.asarray(inverse(pose_matrix(ego_pose)))@np.asarray(pose_matrix(pose))
        pc[:,:3]=pc[:,:3]@m[:3,:3].T+m[:3,3]
        pc=self.pcd.mask_points_by_range(pc,self.h["preprocess"]["cav_lidar_range"])
        return np.ascontiguousarray(pc[:,:4]),pc[:,4].astype(np.int64)

    def predict(self,clouds,gt,diagnose=True):
        from opencood.tools.train_utils import to_device
        torch=self.torch
        features=[self.pre.preprocess(np.asarray(p,dtype=np.float32)) for p in clouds]
        data=dict(processed_lidar=to_device(self.pre.collate_batch(features),self.device),
            record_len=torch.tensor([len(clouds)],dtype=torch.int64,device=self.device),anchor_box=self.anchors,
            transformation_matrix=torch.eye(4,device=self.device))
        with torch.inference_mode():
            pred=self.model(data)
            if not all(torch.isfinite(pred[k]).all() for k in ("psm","rm")):raise Blocked("R3_NONFINITE_MODEL_HEAD")
            with literal_range_filter(self.box),self.audit.guard(self.common):
                boxes,scores=self.post.post_process({"ego":data},{"ego":pred})
            if boxes is None:boxes=np.empty((0,8,3),dtype=np.float32);scores=np.empty(0,dtype=np.float32)
            else:boxes=boxes.cpu().numpy();scores=scores.cpu().numpy()
            if not np.isfinite(boxes).all() or not np.isfinite(scores).all():raise Blocked("R3_NONFINITE_DETECTOR_OUTPUT")
            matrix=iou_matrix(boxes,gt,self.audit)
            matches={str(t):match_matrix(matrix,scores,t) for t in (.3,.5,.7)}
            diagnostics=self.raw_diagnostics(pred,gt,matrix,matches["0.7"]) if diagnose else None
        return dict(boxes=boxes.tolist(),scores=scores.tolist(),matches=matches,diagnostics=diagnostics,
            input_hash=identity([array_hash(p) for p in clouds]),output_hash=identity(dict(boxes=array_hash(boxes),scores=array_hash(scores))))

    def raw_diagnostics(self,pred,gt,final_iou,matched):
        torch=self.torch
        scores=torch.sigmoid(pred["psm"].permute(0,2,3,1)).reshape(-1).cpu().numpy()
        raw=self.post.delta_to_boxes3d(pred["rm"],self.anchors)[0]
        corners=self.box.boxes_to_corners_3d(raw,order=self.post.params["order"]).cpu().numpy()
        finite=np.isfinite(corners).all(axis=(1,2));threshold=self.post.params["target_args"]["score_threshold"]
        if not finite.all():raise Blocked("R3_NONFINITE_RAW_ANCHOR_DECODE")
        safe=corners[finite];sc=scores[finite]
        lo=safe[:,:4,:2].min(axis=1);hi=safe[:,:4,:2].max(axis=1);out=[]
        for k,g in enumerate(gt):
            if k in matched["matched_gt"]:
                out.append(dict(reason="detected",raw_scope="not_needed"));continue
            low=g[:4,:2].min(axis=0);high=g[:4,:2].max(axis=0)
            slots=np.flatnonzero(((hi>=low)&(lo<=high)).all(axis=1))
            polygons=self.common.convert_format(safe[slots]) if len(slots) else []
            valid=np.asarray([p.is_valid and not p.is_empty and np.isfinite(p.area) and p.area>0 for p in polygons],dtype=bool)
            invalid=int((~valid).sum())+int((~finite).sum())
            candidate=safe[slots[valid]];candidate_score=sc[slots[valid]]
            values=iou_matrix(candidate,g[None,:,:],self.audit).reshape(-1)
            best=float(max(values)) if len(values) else 0.
            good=candidate_score[values>=.7]
            post_best=float(max(final_iou[:,k])) if len(final_iou) else 0.
            if invalid:reason="unknown_invalid_raw_candidates"
            elif best>=.7 and len(good) and max(good)<=threshold:reason="score_below_frozen_threshold"
            elif post_best>=.7:reason="GT_assignment_competition"
            elif best>=.7:reason="postprocess_filter_or_NMS_not_uniquely_identified"
            elif best>0:reason="localization_below_AP70_IoU"
            else:reason="no_overlapping_raw_candidate"
            out.append(dict(reason=reason,raw_best_iou=best,final_best_iou=post_best,
                best_score_at_iou70=float(max(good)) if len(good) else None,
                frozen_score_threshold=threshold,raw_overlap_candidates=len(slots),invalid_raw_candidates=invalid,
                causal_claim=False,scope="all finite raw-anchor boxes with BEV bounding overlap; invalid raw candidates explicitly unresolved"))
        return out

    def finish(self):
        self.audit.save()
        if self.model.training or self.state_hash()!=self.initial_state:raise Blocked("R3_FROZEN_MODEL_STATE_CHANGED")
