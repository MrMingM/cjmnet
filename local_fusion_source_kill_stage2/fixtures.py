"""Synthetic fixture ONLY: no data files, Torch, checkpoints or real AP."""
import ast
from types import SimpleNamespace
import numpy as np
from .common import KEEP, ROOT


class Tensor:
    def __init__(self, value):
        self.value = float(value)
        self.shape = (1, 1, 1, 8)
        self.device = 'fixture'
    def clone(self):
        return Tensor(self.value)


class Mask:
    shape = (1, 1, 1, 8)
    def __getitem__(self, key):
        return self


class GT:
    def __len__(self):
        return 2
    def detach(self):
        return self
    def cpu(self):
        return self
    def numpy(self):
        return np.ones((2, 8, 3), dtype=np.float32)


class Dataset:
    calls = 0
    def post_process(self, batch, prediction):
        self.calls += 1
        value = prediction['ego']
        return value['psm'].value, value['rm'].value, GT()


def oracle_fixture():
    scope = dict(np=np, KEEP=KEEP)
    def state(post, target_index, threshold):
        score, geometry, _ = post
        detected = score >= .5 and geometry >= .5
        return dict(matched={0,1} if detected else set(), fp=[],
                    focal=dict(detected=detected, score=score, iou=geometry))
    def proxy(dataset, batch, prediction):
        return dict(scores=np.full(8, prediction['psm'].value, dtype=np.float32),
                    corners=np.full((8,8,3), prediction['rm'].value, dtype=np.float32), anchor_num=1, hw=(1,8))
    scope.update(_state=state, _proxy_cache=proxy,
        count_new_false_positives=lambda a,b,t:0,
        detection_outcome=lambda a,b,c,t: (state((a,b,c),0,t)['matched'], []),
        masks_for_frame=lambda *args: (np.ones((1,8), dtype=bool), [], {'fixture':True}),
        cu=SimpleNamespace(convert_format=lambda a:list(np.asarray(a)),
                           compute_iou=lambda x,y:np.array([float(np.asarray(p).mean()) for p in y])),
        torch=SimpleNamespace(bool=bool, as_tensor=lambda *a,**k:Mask(),
                              where=lambda m,x,y:Tensor(x.value)))
    tree = ast.parse((ROOT/'local_fusion_task_source_oracle/oracle.py').read_text(encoding='utf-8'))
    names = {'run_oracle','_actions','_proxy_tuple','_shortlist_actions','_roi_anchor_ids',
             '_action_key','_target_order','compose_prediction'}
    module = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[])
    exec(compile(module, '<original Oracle AST / synthetic fixture>', 'exec'), scope)
    return SimpleNamespace(**scope)


def ap_fixture(stats, threshold, global_sort):
    """VOC-style fixture calculation, deliberately marked non-scientific."""
    if global_sort:
        raise ValueError('Fixture forbids global sort')
    row = stats[threshold]
    tp, fp = np.cumsum(row['tp']), np.cumsum(row['fp'])
    recall = tp/row['gt']
    precision = tp/np.maximum(tp+fp,1)
    r = np.r_[0,recall,1]
    p = np.r_[0,precision,0]
    p = np.maximum.accumulate(p[::-1])[::-1]
    ids = np.flatnonzero(r[1:] != r[:-1])+1
    return float(np.sum((r[ids]-r[ids-1])*p[ids])),r,p
