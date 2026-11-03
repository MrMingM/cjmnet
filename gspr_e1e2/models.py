"""A1's GSPR/PFN/BN intact; E1 changes messages, E2 adds original B2 C after scatter."""
import torch
from bridge import load
from modules import Enhancement

base=load('original_a1_models','gspr_ggca/models.py')
OBSERVATIONS=base.OBSERVATIONS
OBSERVED_FILM=base.OBSERVED_FILM

class Model(base.Model):
    def __init__(self,args,variant):
        if variant not in ('E0','E1','E2'):raise ValueError('Unknown extension variant')
        super().__init__(args,'A1')
        self.variant=variant
        self.c_enabled=variant=='E2'
        self.c_force_zero=False
        self.c_last={}
        self._e_before={}
        if variant=='E1':self.ggca.enable_s()
        if variant=='E2':
            rng=torch.get_rng_state()
            try:self.repe_c=Enhancement()
            finally:torch.set_rng_state(rng)
            self.scatter.register_forward_hook(self._enhance)

    def _enhance(self,module,inputs,result):
        if not self.c_enabled:return result
        x=result['spatial_features']
        occupied=torch.zeros_like(x[:,:1],dtype=torch.bool)
        coords=self._raw['voxel_coords'].long()
        occupied[coords[:,0],0,coords[:,2],coords[:,3]]=self._raw['voxel_num_points']>0
        changed=self.repe_c(x,{},occupied,constrained=False,state_gate=False)
        strength=self.repe_c.last['strength'];mass=self.repe_c.last['source_mass']
        delta=(changed-x).detach().square().mean(1,keepdim=True).sqrt()
        def group(mask):
            return dict(cells=int(mask.sum()),receiving=int((mask&(mass>1e-6)).sum()),
                mean_strength=float(strength[mask].mean()) if mask.any() else 0.,
                delta_rms_mean=float(delta[mask].mean()) if mask.any() else 0.,
                nonzero=int((mask&(delta>0)).sum()))
        self.c_last=dict(observed=group(occupied),empty=group(~occupied),
            source_observed_cells=int(occupied.sum()),source_mass_mean=float(mass.mean()),
            source_rule='original occupied grid only; target occupancy unrestricted',
            gate_mean=float(strength.mean()),force_zero=self.c_force_zero)
        result['spatial_features']=x if self.c_force_zero else changed
        return result

    def repe_check_gradients(self):
        super().repe_check_gradients()
        if hasattr(self,'repe_c'):
            grads=[p.grad for p in self.repe_c.parameters() if p.grad is not None]
            if not grads or any(not torch.isfinite(g).all() for g in grads) or not any(g.abs().sum()>0 for g in grads):
                raise ValueError('Missing/nonfinite/zero C gradient')
            self._e_before={k:p.detach().clone() for k,p in self.repe_c.named_parameters()}

    def repe_statistics(self,loss):
        row=super().repe_statistics(loss)
        if self.variant=='E2':
            row['propagation']=self.c_last
            row['C_gradient_l2']={k:float(p.grad.detach().norm()) for k,p in self.repe_c.named_parameters() if p.grad is not None}
            row['C_update_l2']={k:float((p.detach()-self._e_before[k]).norm()) for k,p in self.repe_c.named_parameters()}
        return row

    def forward(self,data_dict):
        if self.variant=='E2':self.c_last={}
        result=super().forward(data_dict)
        if not self.training and OBSERVATIONS:
            OBSERVATIONS[-1]['propagation']=self.c_last if self.variant=='E2' else {}
        return result

def create(hypes,variant):return Model(hypes['model']['args'],variant)
def criterion(hypes,variant,film=False):return base.criterion(hypes,'A1',film)
def install(variant):
    from opencood.tools import train_utils
    train_utils.create_model=lambda h:create(h,variant)
    train_utils.create_loss=lambda h:criterion(h,variant)
def frozen_backbone(*args):raise ValueError('No frozen-backbone fine-tuning in E12')
strict_backbone=frozen_backbone
