"""Preserve frozen IoU values; record warnings and stop on invalid geometry/results."""
from contextlib import contextmanager
import warnings
import numpy as np

from scorw_repair.core import Blocked,atomic_json


class IoUAudit:
    def __init__(self,output=None):
        self.output=output
        self.report=dict(status="passed",calls=0,intersection_warnings_with_finite_result=0,
            invalid_geometry=0,invalid_results=0,values_changed=False,geometry_repairs=False,
            policy="same frozen IoU computation; warnings recorded, finite valid results preserved; NaN/invalid geometry stops evaluation")

    def save(self):
        if self.output is not None:atomic_json(self.output/"R2_IOU_AUDIT.json",self.report)

    def call(self,original,box,boxes):
        self.report["calls"]+=1
        for p in [box,*boxes]:
            if not p.is_valid or p.is_empty or not np.isfinite(p.area) or p.area<=0:
                self.report.update(status="failed",invalid_geometry=self.report["invalid_geometry"]+1)
                self.save();raise Blocked("R2_INVALID_BEV_POLYGON")
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always",RuntimeWarning)
            result=original(box,boxes)
        values=np.asarray(result)
        if values.shape!=(len(boxes),) or not np.isfinite(values).all() or (values<0).any() or (values>1).any():
            self.report.update(status="failed",invalid_results=self.report["invalid_results"]+1)
            self.save();raise Blocked("R2_NONFINITE_OR_INVALID_IOU")
        known=0
        for w in captured:
            if issubclass(w.category,RuntimeWarning) and str(w.message)=="invalid value encountered in intersection":
                known+=1
            else:warnings.warn_explicit(str(w.message),w.category,w.filename,w.lineno)
        if known:
            self.report["intersection_warnings_with_finite_result"]+=known
            self.save()
        return result

    @contextmanager
    def guard(self,module):
        original=module.compute_iou
        module.compute_iou=lambda box,boxes:self.call(original,box,boxes)
        try:yield
        finally:module.compute_iou=original
