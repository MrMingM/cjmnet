"""Add point self-retention at exactly one A1 message insertion; reuse B2 C verbatim."""
import ast
import math
import torch
from torch import nn
from torch.nn import functional as F
from bridge import ROOT, load

old=load('a1_modules','gspr_ggca/modules.py')
masked_softmax=old.masked_softmax
Enhancement=load('b2_modules','gspr_repe/modules.py').Enhancement

def a1_message_extension():
    tree=ast.parse((ROOT/'gspr_ggca/modules.py').read_text(encoding='utf-8'))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='GGCA')
    fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='chunk')
    fn.name='_s_chunk'
    inserted=0
    for i,n in enumerate(fn.body):
        if (isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)
                and n.targets[0].id=='messages' and isinstance(n.value,ast.Call)
                and ast.unparse(n.value.func)=='torch.where'):
            fn.body.insert(i+1,ast.parse('messages = self.mix_self(messages, f, attention, neighbor_prior, mask, prior, u, valid)').body[0])
            inserted+=1
            break
    if inserted!=1:
        raise ValueError('A1 message insertion point changed')
    # The original 12 diagnostic columns stay unchanged; add separate S columns.
    ret=fn.body[-1]
    if not isinstance(ret,ast.Return) or ast.unparse(ret.value)!='(result, diag)':
        raise ValueError('A1 return contract changed')
    fn.body.insert(-1,ast.parse('diag = torch.cat((diag, self._s_diag), -1)').body[0])
    namespace=dict(old.__dict__)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),
        str(ROOT/'gspr_ggca/modules.py')+':S_MESSAGE_ONLY','exec'),namespace)
    return namespace['_s_chunk']

class SelfRetention(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.cap=cfg['alpha_cap']
        self.logits=nn.Parameter(torch.tensor([cfg['initial_bias']]+[
            math.log(math.expm1(x)) for x in cfg['initial_gains']],dtype=torch.float32))

    def forward(self,prior,u,attention,neighbor_prior,mask):
        trust=(attention*neighbor_prior).sum(-1)
        evidence=(neighbor_prior*mask).sum(-1)/mask.shape[-1]
        gap=prior-trust
        gain=F.softplus(self.logits[1:])
        alpha=self.cap*prior*torch.sigmoid(self.logits[0]+gain[0]*gap+
            gain[1]*(1-evidence)-gain[2]*u)
        alpha=torch.where(mask.any(-1),alpha,torch.zeros_like(alpha))
        return alpha,trust,evidence

class GGCA(old.GGCA):
    _s_calculate=a1_message_extension()
    def __init__(self,settings,channels=64):
        super().__init__(settings,channels)
        self.s_enabled=False
        self.control='adaptive'
        self.control_constant=None
        self.control_alpha=None
        self.capture=False
        self.captures=[]
        self._control_cursor=0

    def enable_s(self):
        self.s=SelfRetention(self.cfg['S'])
        self.s_enabled=True

    def mix_self(self,messages,f,attention,neighbor_prior,mask,prior,u,valid):
        alpha,trust,evidence=self.s(prior,u,attention,neighbor_prior,mask)
        if self.control=='off':alpha=torch.zeros_like(alpha)
        elif self.control=='uniform':
            alpha=torch.full_like(alpha,float(self.control_constant))*valid*mask.any(-1)
        elif self.control=='shuffled':
            alpha=self.control_alpha[self._control_cursor:self._control_cursor+len(f)]
            self._control_cursor+=len(f)
        mixed=alpha[...,None]*f+(1-alpha[...,None])*messages
        if self.capture:
            self.captures.append(dict(alpha=alpha.detach().clone(),valid=valid.detach().clone(),
                has_neighbor=mask.any(-1).detach().clone()))
        n=valid.sum(1).clamp_min(1)
        mean=lambda x:(x*valid).sum(1)/n
        self._s_diag=torch.stack((mean(alpha),alpha.masked_fill(~valid,0).amax(1),
            mean((messages-f).square().mean(-1).sqrt()),
            mean((mixed-f).square().mean(-1).sqrt()),mean(evidence),mean(trust),mean(prior),
            ((alpha>0)&valid).sum(1).to(f.dtype)), -1).detach()
        return mixed

    def chunk(self,*args):
        if self.s_enabled:
            if args[-1]!=1:raise ValueError('E1 must remain A1, without A2 geometry')
            return self._s_calculate(*args)
        return super().chunk(*args)

    def forward(self,*args):
        self.captures=[]
        self._control_cursor=0
        result=super().forward(*args)
        if self.s_enabled and self.last['pillars']:
            d=self.pillar_rows
            self.last['self_retention']=dict(alpha_mean_pillar=float(d[:,12].mean()),
                alpha_max=float(d[:,13].max()),point_self_difference_rms=float(d[:,14].mean()),
                mixed_self_difference_rms=float(d[:,15].mean()),neighbor_evidence=float(d[:,16].mean()),
                neighbor_trust=float(d[:,17].mean()),point_prior=float(d[:,18].mean()),
                active_points=int(d[:,19].sum()),control=self.control,
                alpha_histogram_by_pillar=torch.histc(d[:,12],bins=10,min=0,max=.5).tolist())
        return result
