import math
from pathlib import Path

import pytest
import torch

from spatial_cbm.layers import pool_residual_spatial_logits
from spatial_cbm.training.config import Config, Model, load
from spatial_cbm.training.engine import fit
from spatial_cbm.training.export import export
from spatial_cbm.training.metrics import assess
from spatial_cbm.training.model import ConceptHead, load_selected
from test_training import config, inputs, threads


@pytest.mark.parametrize('fraction,k',[(.05,40),(.1,79),(.15,118),(.2,157),(.25,196)])
def test_topk_values_and_gradients(fraction,k):
    maps = torch.arange(784,dtype=torch.float32).reshape(1,1,28,28).requires_grad_()
    pooled = pool_residual_spatial_logits(maps,pooling='topk',topk_fraction=fraction)
    torch.testing.assert_close(pooled,torch.tensor([[783-(k-1)/2]]))
    pooled.sum().backward()
    expected = torch.zeros_like(maps).flatten()
    expected[-k:] = 1/k
    torch.testing.assert_close(maps.grad.flatten(),expected)


@pytest.mark.parametrize('mode',['lse','topk'])
def test_pooling_preserves_negative_constant_maps(mode):
    maps = torch.full((2,3,28,28),-4.)
    torch.testing.assert_close(pool_residual_spatial_logits(maps,pooling=mode),torch.full((2,3),-4.))


def test_lse_default_unchanged_and_topk_full_mean():
    maps = torch.randn(2,3,7,7)
    expected = torch.logsumexp(maps.flatten(2),dim=-1)-math.log(49)
    torch.testing.assert_close(pool_residual_spatial_logits(maps),expected,rtol=0,atol=0)
    torch.testing.assert_close(pool_residual_spatial_logits(maps,pooling='topk',topk_fraction=1),maps.mean((2,3)))


def test_head_pooling_uses_same_maps_and_initial_weights():
    torch.manual_seed(7)
    baseline = ConceptHead(Model(width=8),(2,4,8),3)
    torch.manual_seed(7)
    topk = ConceptHead(Model(width=8,spatial_pooling='topk',topk_fraction=.15),(2,4,8),3)
    for key,value in baseline.state_dict().items():
        torch.testing.assert_close(value,topk.state_dict()[key],rtol=0,atol=0)
    features = dict(conv3=torch.randn(2,2,8,8),conv4=torch.randn(2,4,4,4),conv5=torch.randn(2,8,2,2))
    a,b = baseline(features),topk(features)
    torch.testing.assert_close(a['spatial_maps'],b['spatial_maps'],rtol=0,atol=0)
    expected = b['spatial_maps'].flatten(2).topk(math.ceil(64*.15),dim=-1).values.mean(-1)
    torch.testing.assert_close(b['spatial_logits'],expected,rtol=0,atol=0)
    torch.testing.assert_close(b['final_logits'],b['global_logits']+.1*expected,rtol=0,atol=0)


@pytest.mark.parametrize('options',[dict(spatial_pooling='unknown'),dict(topk_fraction=0),
    dict(topk_fraction=-.1),dict(topk_fraction=1.1),dict(topk_fraction=float('nan'))])
def test_pooling_config_rejects_invalid_values(options):
    with pytest.raises(ValueError):
        Config(model=Model(weights='IMAGENET1K_V1',**options)).validate()


def test_pooling_survives_yaml_checkpoint_calibration_and_export(tmp_path):
    import yaml
    prepared,_ = inputs(tmp_path)
    cfg = config(tmp_path,prepared)
    cfg.model.spatial_pooling = 'topk'
    cfg.model.topk_fraction = .15
    path = tmp_path/'config.yaml'
    path.write_text(yaml.safe_dump(cfg.asdict()))
    cfg = load(path)
    result = fit(cfg)
    assert fit(cfg,resume=True)==result
    assert assess(cfg)['checkpoint_epoch']==result['best_epoch']
    assert assess(cfg,test=True)['checkpoint_epoch']==result['best_epoch']
    export(cfg)
    _,head,_,_,_ = load_selected(cfg)
    assert head.residual_spatial_pooling=='topk' and head.residual_topk_fraction==.15
    protocol = torch.load(Path(cfg.training.output)/'best.pt',weights_only=False)['protocol']
    assert protocol['config']['model']['spatial_pooling']=='topk'
    assert protocol['config']['model']['topk_fraction']==.15
