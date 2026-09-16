import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from .network import QueryDecoder,MiniCEIF,project_observations,interval_loss,stencil
from .observations import decoder_loss,make_packet

EXTENT=[0,0,-2,4,4,2]


def packet(xyz,lo,hi):
    return dict(xyz=torch.tensor(xyz,dtype=torch.float32).reshape(-1,3),
                lower=torch.tensor(lo,dtype=torch.float32),upper=torch.tensor(hi,dtype=torch.float32))


class MiniTests(unittest.TestCase):
    def decoder(self):
        decoder=QueryDecoder(2)
        with torch.no_grad():
            decoder.weight.weight.zero_(); decoder.weight.weight[0,0]=2**.5
            decoder.bias.weight.zero_()
        return decoder.requires_grad_(False)

    def test_unknown_identity_even_at_extreme_logits(self):
        decoder=self.decoder(); feature=torch.full((1,2,4,4),100.)
        p=packet([[1.5,1.5,0]],[0],[1])
        torch.testing.assert_close(project_observations(feature,decoder,p,EXTENT,.5),feature,atol=0,rtol=0)

    def test_hit_and_free_have_opposite_local_effects(self):
        decoder=self.decoder(); feature=torch.zeros(1,2,4,4)
        for lo,hi,sign in ((.9,1,1),(0,.1,-1)):
            p=packet([[1.5,1.5,0]],[lo],[hi])
            result=project_observations(feature,decoder,p,EXTENT,.5)
            self.assertGreater(sign*result[0,0,1,1],0)
            self.assertEqual(int((result!=0).sum()),1)
            self.assertLess(interval_loss(decoder(result,p['xyz'],EXTENT),p['lower'],p['upper']),
                            interval_loss(decoder(feature,p['xyz'],EXTENT),p['lower'],p['upper']))

    def test_conflict_is_not_unknown(self):
        decoder=self.decoder(); feature=torch.zeros(1,2,4,4)
        p=packet([[1.5,1.5,0]],[.9],[.1])
        self.assertGreater(float(interval_loss(decoder(feature,p['xyz'],EXTENT),p['lower'],p['upper'])),.3)
        self.assertTrue(torch.isfinite(project_observations(feature,decoder,p,EXTENT,.5)).all())

    def test_control_initial_identity_and_empty_packet(self):
        model=MiniCEIF(2,4); decoder=self.decoder(); feature=torch.randn(1,2,4,4)
        p=packet([],[],[])
        torch.testing.assert_close(model(feature,decoder,p,EXTENT,False),feature,atol=0,rtol=0)
        torch.testing.assert_close(model(feature,decoder,p,EXTENT,True),feature,atol=0,rtol=0)

    def test_gradient_through_projection_and_frozen_head(self):
        model=MiniCEIF(2,4); decoder=self.decoder(); feature=torch.zeros(1,2,4,4)
        head=torch.nn.Conv2d(2,1,1).requires_grad_(False)
        with torch.no_grad(): head.weight.fill_(1); head.bias.zero_()
        p=packet([[1.5,1.5,0]],[.9],[1])
        result=model(feature,decoder,p,EXTENT,True)
        loss=(head(result)-1).square().mean(); loss.backward()
        self.assertGreater(float(model.residual[-1].bias.grad.abs().sum()),0)
        self.assertGreater(float(model.step_logit.grad.abs()),0)
        self.assertTrue(all(v.grad is None for v in decoder.parameters()))
        self.assertIsNone(head.weight.grad)

    def test_decoder_training_then_freeze_and_roundtrip(self):
        torch.manual_seed(7); decoder=QueryDecoder(2)
        feature=torch.zeros(1,2,4,4); feature[0,0,1,1]=1; feature[0,0,2,2]=-1
        p=packet([[1.5,1.5,0],[2.5,2.5,0]],[.9,0],[1,.1])
        p.update(labels=torch.tensor([1.,0.]),label_weights=torch.tensor([.9,.9]))
        optimizer=torch.optim.Adam(decoder.parameters(),lr=.05)
        before=float(decoder_loss(decoder(feature,p['xyz'],EXTENT),p))
        for _ in range(20):
            optimizer.zero_grad(); loss=decoder_loss(decoder(feature,p['xyz'],EXTENT),p)
            loss.backward(); optimizer.step()
        self.assertLess(float(loss),before)
        replica=QueryDecoder(2); replica.load_state_dict(copy.deepcopy(decoder.state_dict()))
        torch.testing.assert_close(replica(feature,p['xyz'],EXTENT),decoder(feature,p['xyz'],EXTENT))

    def test_stencil_weights_and_query_order(self):
        p=packet([[.1,.1,0],[2.25,3.5,0]],[.8,0],[1,.2])
        ids,weights,_=stencil(p['xyz'],(4,4),EXTENT)
        torch.testing.assert_close(weights.sum(1),torch.ones(2))
        decoder=self.decoder(); feature=torch.zeros(1,2,4,4)
        shuffled={k:v.flip(0) for k,v in p.items()}
        torch.testing.assert_close(project_observations(feature,decoder,p,EXTENT,.5),
                                   project_observations(feature,decoder,shuffled,EXTENT,.5))

    def test_observation_packet_uses_actual_hits_and_rays(self):
        from ceif_audit.test_audit import source
        sources=[source([[5,0,0],[5,1,0],[5,2,0]]),source([[10,2,0],[10,3,0],[10,4,0]])]
        model=SimpleNamespace(engine=SimpleNamespace(base=None),grid=(2,4),lidar_range=[0,-5,-3,20,5,3])
        encoded={'obs':torch.zeros(2,13,2,4)}
        masks=torch.ones(2,1,2,4)
        with patch('ceif_min.observations.build_sources',return_value=sources):
            a=make_packet(model,{},encoded,masks,41,32)
            b=make_packet(model,{},encoded,masks,41,32)
        for key in ('xyz','lower','upper','labels','label_weights'):
            torch.testing.assert_close(a[key],b[key])
        self.assertTrue((a['labels']==0).any() and (a['labels']==1).any())
        self.assertTrue((a['lower'][a['labels']==1]>=.89).all())
        self.assertTrue((a['upper'][a['labels']==0]<=.11).all())
        self.assertGreater(a['extra_bytes'],0)


if __name__=='__main__': unittest.main()
