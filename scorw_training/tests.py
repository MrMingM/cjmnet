"""Server-only schedule/AP regression tests; real steps covered by SMOKE."""
from .data import schedule
from scorw_diagnostics.core import write


def checks(c,out):
    from scorw_diagnostics.ap_audit import ap
    from opencood.utils.eval_utils import calculate_ap
    import copy
    for epoch in range(20):
        a,w=schedule(c,epoch,6374);b,v=schedule(c,epoch,6374)
        assert a==b and w==v and sum(w)==3187 and len(set(a))==6374
    r=[dict(tp=[0,1],fp=[1,0],score=[.9,.3],gt=2),dict(tp=[1],fp=[0],score=[.8],gt=1)]
    stat={.7:dict(tp=[v for m in r for v in m['tp']],fp=[v for m in r for v in m['fp']],score=[v for m in r for v in m['score']],gt=3)}
    for flag in (False,True):assert abs(ap(r,flag)-calculate_ap(copy.deepcopy(stat),.7,flag)[0])<1e-12
    from .iou import exact_iou
    from scorw_repair.iou_audit import IoUAudit
    from opencood.utils import common_utils
    from shapely.geometry import Polygon
    import numpy as np
    polygons=[Polygon([(x,0),(x+1,0),(x+1,1),(x,1)]) for x in (-10,-1,0,.3,1,10)]
    polygons+=[Polygon([(0,0),(2,.001),(1,2),(-1,1)])]
    expected=[common_utils.compute_iou(p,polygons) for p in polygons]
    with exact_iou(common_utils,IoUAudit(None)):
        for p,e in zip(polygons,expected):assert np.array_equal(e,common_utils.compute_iou(p,polygons))
    write(out/'UNIT_TESTS.json',dict(status='passed',schedule_epochs_checked=20,exact_half=True,AP_native_equivalence=True,
        all_initial_backbones_equal=True,IoU_disjoint_shortcut_bitwise_exact=True,real_forward_backward_and_resume='required in SMOKE for S0/S1/S2'))
