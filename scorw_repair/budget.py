"""Count the complete frozen workload before waveform generation or GPU inference."""
from scorw_repair.core import Blocked, identity, atomic_json


def plan(config, candidates, source_rows, output=None):
    index={}
    for r in source_rows:
        key=(r["scene"],r["timestamp"],r["cav"])
        if key in index:raise Blocked("R2_DUPLICATE_INPUT_IDENTITY")
        index[key]=r
    selected=[]
    for scene in sorted({f["scene"] for f in candidates}):
        selected.extend([f for f in candidates if f["scene"]==scene][:config["r2"]["full_frames_per_scene"]])
    rows=[]; problems=[]
    pool=config["pool_size"]
    if pool!=20:raise Blocked("R2_REQUIRES_COMPLETE_TWENTY_FIELD_POOL")
    for frame in selected:
        ego=frame["ego"]
        cavs=[ego]+sorted(c for c in frame["active_cavs"] if c!=ego)[:config["r2"]["cavs"]-1]
        if len(cavs)!=config["r2"]["cavs"] or len(set(cavs))!=len(cavs):
            raise Blocked("R2_FROZEN_CAV_COUNT_UNAVAILABLE")
        clouds=[]
        for cav in cavs:
            r=index[(frame["scene"],frame["timestamp"],cav)]
            n=r["points"]
            if isinstance(n,bool) or not isinstance(n,int) or n<1:raise Blocked("R2_INVALID_CLEAR_POINT_COUNT")
            clouds.append(dict(cav=cav,points=n,pcd_sha256=r["pcd_sha256"],yaml_sha256=r["yaml_sha256"]))
        cost=(pool+1)*sum(r["points"] for r in clouds)
        row=dict(scene=frame["scene"],timestamp=frame["timestamp"],ego=ego,cavs=cavs,clouds=clouds,
            ray_field_pairs=cost,field_responses_per_ray=pool+1,includes="20 pooled fields plus B2 constant; B3/B4 reuse; B0 zero-fog and B1 separate")
        rows.append(row)
        if cost>config["r2"]["maximum_ray_field_pairs_per_frame"]:
            problems.append(dict(scope="frame",scene=frame["scene"],timestamp=frame["timestamp"],required=cost,
                limit=config["r2"]["maximum_ray_field_pairs_per_frame"]))
    total=sum(r["ray_field_pairs"] for r in rows)
    if total>config["r2"]["maximum_ray_field_pairs"]:
        problems.append(dict(scope="run",required=total,limit=config["r2"]["maximum_ray_field_pairs"]))
    result=dict(status="passed" if rows and not problems else "blocked",frames=rows,required_ray_field_pairs=total,
        per_frame_limit=config["r2"]["maximum_ray_field_pairs_per_frame"],run_limit=config["r2"]["maximum_ray_field_pairs"],
        problems=problems,selection_uses_weather_or_AP=False,all_fields_preserved=True,
        selection_identity=identity(rows),original_points=sum(c["points"] for r in rows for c in r["clouds"]))
    if output is not None:atomic_json(output/"R2_COMPUTE_PLAN.json",result)
    if result["status"]!="passed":
        raise Blocked("R2_COMPUTE_BUDGET_EXCEEDED_BEFORE_ANY_RESPONSE: required="+str(total)+
            "; run_limit="+str(result["run_limit"])+"; details in R2_COMPUTE_PLAN.json")
    return selected,result
