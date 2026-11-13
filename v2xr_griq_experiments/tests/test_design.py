"""Small CPU invariants. Real native smoke is run by the entrypoint."""
import numpy as np
import torch

from v2xr_griq_experiments.dataset import partitions, pool_mask
from v2xr_griq_experiments.training import Quality, pair_indices


def main():
    assert pool_mask(np.array([.9,.21,.05,.01]),np.arange(4),.2,.02,1).tolist()==[True,True,False,False]
    split=partitions([(str(i),i) for i in range(32)])
    assert sorted(map(len,split.values()))==[5,5,22] and len(set(sum(split.values(),[])))==32
    dual=Quality(10,'dual')
    x=torch.randn(8,10)
    assert torch.allclose(dual.logits(x).sigmoid(),dual(x).sigmoid().prod(1),atol=1e-6)
    assert torch.nn.functional.softplus(torch.tensor(-2.)) < torch.nn.functional.softplus(torch.tensor(2.))
    print('DESIGN TESTS PASS',flush=True)


if __name__ == '__main__':
    main()
