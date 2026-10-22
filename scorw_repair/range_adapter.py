"""Use the literal frozen GT range without executing historical dataset loaders."""
import ast
from contextlib import contextmanager

from scorw_repair.core import REPO, Blocked


def gt_range():
    path=REPO/"opencood/data_utils/datasets/__init__.py"
    tree=ast.parse(path.read_text("utf-8"))
    values=[ast.literal_eval(node.value) for node in tree.body if isinstance(node,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=="GT_RANGE" for t in node.targets)]
    if values!=[[-140,-40,-3,140,40,1]]:
        raise Blocked("R2_FROZEN_GT_RANGE_CHANGED")
    return values[0]


def mask(boxes):
    import torch
    bounds=gt_range()
    lo=torch.Tensor(bounds[:2]).reshape(1,1,-1).to(boxes.device)
    hi=torch.Tensor(bounds[3:5]).reshape(1,1,-1).to(boxes.device)
    return torch.all(torch.all(boxes[:,:,:2]>=lo,dim=-1)&torch.all(boxes[:,:,:2]<=hi,dim=-1),dim=-1)


@contextmanager
def literal_range_filter(box_utils):
    # Same tensor operations/range as the pinned original function; avoid its lazy dataset-package import.
    original=box_utils.get_mask_for_boxes_within_range_torch
    box_utils.get_mask_for_boxes_within_range_torch=mask
    try:
        yield
    finally:
        box_utils.get_mask_for_boxes_within_range_torch=original
