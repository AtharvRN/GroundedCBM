import torch

from gcbm.losses import sgcbm_concept_losses, soft_align_kl_loss, weighted_concept_bce


def test_weighted_concept_bce_increases_positive_weighted_loss():
    logits = torch.tensor([[-2.0, 0.0]])
    targets = torch.tensor([[1.0, 0.0]])
    unweighted = weighted_concept_bce(logits, targets, pos_weight=1.0)
    weighted = weighted_concept_bce(logits, targets, pos_weight=3.0)
    assert weighted > unweighted


def test_soft_align_kl_loss_returns_zero_for_no_valid_masks():
    maps = torch.randn(2, 3, 4, 4)
    indices = torch.full((2, 1), -1, dtype=torch.long)
    targets = torch.zeros(2, 1, 4, 4)
    valid = torch.zeros(2, 1, dtype=torch.bool)
    loss = soft_align_kl_loss(maps, indices, targets, valid)
    assert loss.item() == 0.0


def test_soft_align_kl_loss_is_finite_for_valid_mask():
    maps = torch.zeros(1, 2, 2, 2)
    indices = torch.tensor([[1]])
    targets = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])
    valid = torch.tensor([[True]])
    loss = soft_align_kl_loss(maps, indices, targets, valid)
    assert torch.isfinite(loss)
    assert loss.item() > 0.0


def test_sgcbm_concept_losses_returns_global_and_mask_terms():
    logits = torch.zeros(1, 2)
    spatial = torch.zeros(1, 2, 2, 2)
    global_targets = torch.tensor([[1.0, 0.0]])
    indices = torch.tensor([[0]])
    targets = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])
    valid = torch.tensor([[True]])
    loss_global, loss_mask = sgcbm_concept_losses(logits, spatial, global_targets, indices, targets, valid)
    assert torch.isfinite(loss_global)
    assert torch.isfinite(loss_mask)
