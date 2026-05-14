import torch
import torch.nn as nn

from gcbm.imagenet_models import DualBranchConceptHead, MultiScaleDualBranchConceptHead
from gcbm.sg_model import DualBranchConceptLayer, MultiScaleConceptLayer, pool_concept_maps, pool_residual_spatial_logits


def test_pool_concept_maps_avg_and_topk():
    maps = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    assert torch.allclose(pool_concept_maps(maps, pooling="avg"), torch.tensor([[2.5]]))
    assert torch.allclose(pool_concept_maps(maps, pooling="topk", topk_fraction=0.5), torch.tensor([[3.5]]))


def test_lse_residual_pooling_preserves_constant_maps():
    maps = torch.full((2, 3, 4, 4), 7.0)
    pooled = pool_residual_spatial_logits(maps, pooling="lse")
    assert torch.allclose(pooled, torch.full((2, 3), 7.0), atol=1e-6)
    assert torch.allclose(pool_residual_spatial_logits(maps, pooling="avg"), torch.full((2, 3), 7.0))


def test_dual_branch_concept_layer_shapes_and_residual():
    global_layer = nn.Linear(4, 2, bias=False)
    spatial_layer = nn.Conv2d(3, 2, kernel_size=1, bias=False)
    with torch.no_grad():
        global_layer.weight.fill_(1.0)
        spatial_layer.weight.fill_(0.5)
    layer = DualBranchConceptLayer(global_layer, spatial_layer, spatial_stage="conv4", residual_alpha=0.25, residual_spatial_pooling="avg")
    feats = {"conv5": torch.ones(5, 4, 2, 2), "conv4": torch.ones(5, 3, 2, 2)}
    out = layer(feats)
    assert out["global_logits"].shape == (5, 2)
    assert out["spatial_maps"].shape == (5, 2, 2, 2)
    assert torch.allclose(out["final_logits"], out["global_logits"] + 0.25 * out["spatial_logits"])


def test_multiscale_concept_layer_shapes():
    layer = MultiScaleConceptLayer(nn.Linear(8, 3), nn.Conv2d(5, 3, kernel_size=1), conv4_dim=4, conv5_dim=8, fusion_dim=5)
    feats = {"conv4": torch.randn(2, 4, 7, 7), "conv5": torch.randn(2, 8, 4, 4)}
    out = layer(feats)
    assert out["global_logits"].shape == (2, 3)
    assert out["spatial_maps"].shape == (2, 3, 7, 7)


def test_imagenet_head_state_dict_keys_remain_checkpoint_compatible():
    dual_keys = set(DualBranchConceptHead(2, "conv5", 0.1, "avg").state_dict())
    assert "global_head.weight" in dual_keys
    assert "spatial.weight" in dual_keys
    assert not any(key.startswith("global_layer.") for key in dual_keys)
    multi_keys = set(MultiScaleDualBranchConceptHead(2, 0.1, "avg").state_dict())
    assert {"global_head.weight", "spatial.weight", "conv4_proj.weight", "conv5_proj.weight"} <= multi_keys
    assert not any(key.startswith("spatial_layer.") for key in multi_keys)
