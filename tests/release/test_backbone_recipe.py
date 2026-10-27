import copy

import pytest
import torch

from spatial_cbm.training.backbone_recipe import mix_batch, update_ema
from spatial_cbm.training.config import Training


@pytest.mark.parametrize('cutmix', [False, True])
def test_mixing_preserves_label_and_pixel_mass(cutmix):
    torch.manual_seed(12)
    targets=torch.arange(8)
    images=targets.float()[:,None,None,None].expand(-1,3,16,16).clone()
    cfg=Training(mixup_alpha=0. if cutmix else .2,cutmix_alpha=1. if cutmix else 0.)
    a,b,lam=mix_batch(images,targets,cfg)
    torch.testing.assert_close(images.mean((1,2,3)),lam*a+(1-lam)*b)
    assert 0 <= lam <= 1


def test_ema_averages_float_state_and_copies_counters():
    model=torch.nn.BatchNorm2d(2)
    ema=copy.deepcopy(model)
    with torch.no_grad():
        model.weight.fill_(3)
        model.running_mean.fill_(2)
        model.num_batches_tracked.fill_(7)
    update_ema(ema,model,.9999,1)
    decay=2/11
    torch.testing.assert_close(ema.weight,torch.full((2,),decay+3*(1-decay)))
    torch.testing.assert_close(ema.running_mean,torch.full((2,),2*(1-decay)))
    assert ema.num_batches_tracked.item()==7
