"""CPU integration: real optimizer/loss/checkpoints, synthetic features and validation."""
import copy
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from .test_min import packet,EXTENT


class TrainingTests(unittest.TestCase):
    def test_training_checkpoint_resume_and_frozen_frontend(self):
        from . import train
        model=torch.nn.Module(); model.engine=torch.nn.Module(); model.engine.base=torch.nn.Module()
        base=model.engine.base
        base.cls_head=torch.nn.Conv2d(2,1,1); base.reg_head=torch.nn.Conv2d(2,7,1)
        model.lidar_range=EXTENT
        before=copy.deepcopy(model.state_dict())
        feature=torch.randn(1,2,4,4)
        observations=packet([[1.5,1.5,0],[2.5,2.5,0]],[.8,0],[1,.2])
        observations.update(labels=torch.tensor([1.,0.]),label_weights=torch.tensor([.9,.9]))
        labels=dict(pos_equal_one=torch.ones(1,4,4,1),targets=torch.zeros(1,4,4,7))
        batch={'ego':dict(communication_sample_index=[0],label_dict=labels)}
        loader_calls=[]
        def loader(hypes,options,training,weather,epoch=0,smoke=0):
            loader_calls.append((training,weather))
            return None,[batch],[0]
        def validation(model,decoder,variants,hypes,options,settings,device,smoke):
            return dict(results={name:dict(ap70=.5) for name in variants})
        stub=types.ModuleType('opencood.tools.train_utils'); stub.to_device=lambda b,d:b
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); frontend=root/'frontend.yaml'; frontend.write_text('test: true')
            weights=root/'frontend.pth'; weights.write_bytes(b'fixed')
            output=root/'run'
            argv=['train','--frontend-config',str(frontend),'--frontend-checkpoint',str(weights),
                  '--output-dir',str(output),'--epochs','1','--decoder-epochs','1','--queries','16','--hidden','4','--smoke','1']
            def new_output(path):
                value=Path(path); value.mkdir(); return value
            with patch.dict(sys.modules,{'opencood.tools.train_utils':stub}), \
                 patch.multiple(train.rt,load_config=lambda *a: ({'seed':1},
                     {'root_dir':'/synthetic/train','validate_dir':'/synthetic/validate','loss':{'args':{'cls_weight':1.,'reg':2.}}}),
                     device=lambda:torch.device('cpu'),load_model=lambda *a:(model,train.rt.sha256(weights)),new_output=new_output), \
                 patch.object(train,'contract',lambda options,frontend,digest,settings:dict(prototype=settings,sha=digest)), \
                 patch.object(train,'loader',loader),patch.object(train,'prepare',lambda *a,**k:(feature,observations)), \
                 patch.object(train,'validate',validation),patch.object(sys,'argv',argv):
                train.main()
                state=torch.load(output/'ceif_best.pth',map_location='cpu',weights_only=True)
                self.assertEqual(state['variant'],'ceif')
                self.assertTrue((output/'aux_only_best.pth').exists())
                self.assertGreater(float(state['model']['residual.4.weight'].abs().sum()),0)
                count=len(loader_calls)
                with patch.object(sys,'argv',argv+['--resume']): train.main()
                self.assertEqual(len(loader_calls),count)  # finished epochs are not repeated
            for key,value in model.state_dict().items(): torch.testing.assert_close(value,before[key],atol=0,rtol=0)
            self.assertIn((True,'mixed'),loader_calls)
            self.assertIn((False,'clean'),loader_calls)
            self.assertFalse(json.loads((output/'protocol.json').read_text())['test_used_for_selection'])
            self.assertTrue(json.loads((output/'training_complete.json').read_text())['smoke'])


if __name__=='__main__': unittest.main()
