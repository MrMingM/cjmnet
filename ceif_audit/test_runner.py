"""CPU orchestration smoke. Synthetic geometry/postprocessing, real AP accumulation.

This checks files/modes/contracts, not the server detector or real-data AP.
"""
import importlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from .test_audit import box


class RunnerTests(unittest.TestCase):
    def test_end_to_end_artifacts_and_metric_protocol(self):
        from gspr_evidence.test_evidence import toy_model
        from gspr_evidence import runtime as rt
        from . import run
        model, encoded = toy_model()
        model.lidar_range = [0,-5,-3,20,5,3]
        cloud = torch.tensor([[[5.,0,0],[5,.05,0],[5,-.05,0]],
                              [[10.,0,0],[10,.1,0],[10,-.1,0]]])
        processed = dict(voxel_features=cloud, voxel_num_points=torch.tensor([3,3]),
                         voxel_coords=torch.tensor([[0,0,1,0],[1,0,1,1]]))
        rel = dict(point_evidence=torch.tensor([[[3.,6.]]*3,[[18.,1.]]*3]),
                   point_valid_mask=torch.ones(2,3,dtype=torch.bool))
        class GSPR(torch.nn.Module):
            def forward(self, *args):
                return rel
        model.engine.base.gspr = GSPR()
        model.engine.base.forward = lambda inp: model.run(encoded, 'full')[0]
        model.encode = lambda inp: encoded
        inp = dict(processed_lidar=processed, clouds=[cloud[0],cloud[1]],
                   transforms=torch.eye(4)[None].repeat(2,1,1))
        target = torch.from_numpy(box()[None])
        class Dataset:
            len_record = [2]
            def post_process(self, batch, pred):
                score = torch.sigmoid(pred['ego']['psm'].mean()).reshape(1)
                return target.clone(), score, target.clone()
        loader = [{'ego': {'communication_sample_index': [i]}} for i in range(2)]
        # Replace only geometry backend and train utility dependencies, leaving the
        # actual checkout eval_utils/AP implementation active.
        common = types.ModuleType('opencood.utils.common_utils')
        common.torch_tensor_to_numpy = lambda x: x.detach().cpu().numpy() if torch.is_tensor(x) else x
        common.convert_format = lambda x: list(x)
        def ious(a, others):
            low, high = a[:4,:2].min(0), a[:4,:2].max(0)
            values = []
            for b in others:
                bl, bh = b[:4,:2].min(0), b[:4,:2].max(0)
                inter = np.maximum(np.minimum(high,bh)-np.maximum(low,bl),0).prod()
                values.append(inter/(np.prod(high-low)+np.prod(bh-bl)-inter))
            return np.asarray(values)
        common.compute_iou = ious
        train = types.ModuleType('opencood.tools.train_utils')
        train.to_device = lambda batch, device: batch
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); checkpoint = root/'weights.pth'; checkpoint.write_bytes(b'unchanged')
            frontend = root/'config.yaml'; frontend.write_text('test: true')
            output = root/'clean'
            def new_output(path):
                p = Path(path); p.mkdir(); return p
            with patch.dict(sys.modules, {'opencood.utils.common_utils': common,
                                          'opencood.tools.train_utils': train}):
                evaluation = importlib.import_module('opencood.utils.eval_utils')
                with patch.object(evaluation, 'common_utils', common), \
                     patch.multiple(rt, load_config=lambda *a: ({'seed':1}, {'validate_dir':'/data/scd/datasets/opv2v_official_data_dumping/validate'}),
                         device=lambda: torch.device('cpu'), load_model=lambda *a: (model,rt.sha256(checkpoint)),
                         make_loader=lambda *a,**k: (Dataset(),loader,[0,1]), new_output=new_output,
                         input_branch=lambda *a: inp, contract=lambda *a: {'synthetic':True}), \
                     patch.object(run, 'assignments', lambda b,s,g: __import__('ceif_audit.scoring',fromlist=['greedy']).greedy(
                         np.asarray([ious(a,g) for a in b]).reshape(len(b),len(g)),s)), \
                     patch.object(sys, 'argv', ['audit','--weather','clean','--smoke','2',
                         '--frontend-config',str(frontend),'--frontend-checkpoint',str(checkpoint),
                         '--output-dir',str(output)]):
                    run.main()
            summary = json.loads((output/'summary.json').read_text())
            self.assertEqual(summary['totals']['frames'],2)
            self.assertEqual(set(summary['results']),set(run.MODES))
            self.assertFalse(summary['protocol']['global_sort'])
            self.assertTrue(summary['protocol']['smoke'])
            self.assertEqual(summary['results']['baseline']['ap70'],1)
            self.assertGreater(summary['totals']['evaluated_actions'],0)
            self.assertTrue((output/'actions.jsonl').read_text())
            self.assertTrue((output/'audit.md').is_file())
            self.assertTrue((output/'baseline'/'eval.yaml').is_file())


if __name__ == '__main__':
    unittest.main()
