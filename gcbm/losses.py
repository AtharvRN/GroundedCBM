from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import torch.nn as nn
import torch
import torch.nn.functional as F


NINF = -100
LOGGER = logging.getLogger(__name__)


class TwoWayLoss(nn.Module):
    def __init__(self, tp: float = 4.0, tn: float = 1.0) -> None:
        super().__init__()
        self.tp = float(tp)
        self.tn = float(tn)
        LOGGER.info("Initializing TwoWayLoss with Tp=%s and Tn=%s", self.tp, self.tn)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        class_mask = (y > 0).any(dim=0)
        sample_mask = (y > 0).any(dim=1)
        positive_mask = y.masked_fill(y <= 0, NINF).masked_fill(y > 0, 0.0)
        positive_class = torch.logsumexp(-x / self.tp + positive_mask, dim=0).mul(self.tp)[class_mask]
        positive_sample = torch.logsumexp(-x / self.tp + positive_mask, dim=1).mul(self.tp)[sample_mask]
        negative_mask = y.masked_fill(y != 0, NINF).masked_fill(y == 0, 0.0)
        negative_class = torch.logsumexp(x / self.tn + negative_mask, dim=0).mul(self.tn)[class_mask]
        negative_sample = torch.logsumexp(x / self.tn + negative_mask, dim=1).mul(self.tn)[sample_mask]
        return F.softplus(negative_class + positive_class).mean() + F.softplus(negative_sample + positive_sample).mean()


def get_loss(
    loss_type: str,
    num_concepts: int,
    num_samples: int,
    concept_counts: List[float],
    cbl_pos_weight: float,
    cbl_auto_weight: bool = False,
    tp: float = 4.0,
    device: str = "cuda",
) -> nn.Module:
    if loss_type == "bce":
        LOGGER.info("Using BCE loss for training CBL")
        if cbl_auto_weight:
            LOGGER.info("Using automatic positive weighting with scale %s", cbl_pos_weight)
            pos_count = torch.tensor(concept_counts, device=device)
            neg_count = num_samples - pos_count
            scale = (neg_count / pos_count) * cbl_pos_weight
            return nn.BCEWithLogitsLoss(pos_weight=scale).to(device)
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([cbl_pos_weight] * num_concepts, device=device)).to(device)
    if loss_type == "twoway":
        LOGGER.info("Using TwoWay loss for training CBL")
        return TwoWayLoss(tp=tp).to(device)
    raise NotImplementedError(f"Loss {loss_type} is not implemented")


def weighted_concept_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: float = 1.0,
) -> torch.Tensor:
    """Binary concept loss with a scalar positive-class weight."""
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    weights = torch.where(
        targets > 0.5,
        torch.full_like(targets, float(pos_weight)),
        torch.ones_like(targets),
    )
    return (bce * weights).sum() / torch.clamp(weights.sum(), min=1.0)


def soft_align_kl_loss(
    map_logits: torch.Tensor,
    mask_indices: torch.Tensor,
    mask_targets: torch.Tensor,
    mask_valid: torch.Tensor,
    concept_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """KL-align selected concept maps to target masks.

    `mask_indices`, `mask_targets`, and `mask_valid` are padded per-image
    tensors. Only valid concept/mask pairs contribute to the loss.
    """
    map_logits = F.interpolate(
        map_logits,
        size=mask_targets.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    batch_losses = []
    for batch_index in range(map_logits.shape[0]):
        valid = mask_valid[batch_index]
        if not bool(valid.any()):
            continue
        concept_ids = mask_indices[batch_index][valid]
        pred = map_logits[batch_index].index_select(0, concept_ids)
        pred_flat = pred.flatten(1).float()
        target_mass = mask_targets[batch_index][valid].flatten(1).float().clamp(min=0.0)
        target_mass_sum = target_mass.sum(dim=1, keepdim=True)
        valid_targets = target_mass_sum.squeeze(1) > 0.0

        per_concept_loss = torch.zeros(
            (pred_flat.shape[0],),
            device=pred.device,
            dtype=pred_flat.dtype,
        )
        if bool(valid_targets.any()):
            target_dist = torch.zeros_like(target_mass)
            target_dist[valid_targets] = target_mass[valid_targets] / torch.clamp(
                target_mass_sum[valid_targets],
                min=1e-6,
            )
            pred_log_dist = F.log_softmax(pred_flat[valid_targets], dim=1)
            per_concept_loss[valid_targets] = F.kl_div(
                pred_log_dist,
                target_dist[valid_targets],
                reduction="none",
            ).sum(dim=1)

        if concept_weights is None:
            batch_losses.append(per_concept_loss.mean())
            continue

        weights = concept_weights[batch_index].index_select(0, concept_ids).to(per_concept_loss.dtype)
        batch_losses.append((per_concept_loss * weights).sum() / torch.clamp(weights.sum(), min=1.0))

    return torch.stack(batch_losses).mean() if batch_losses else map_logits.sum() * 0.0


def sgcbm_concept_losses(
    concept_logits: torch.Tensor,
    spatial_logits: torch.Tensor,
    global_targets: torch.Tensor,
    mask_indices: torch.Tensor,
    mask_targets: torch.Tensor,
    mask_valid: torch.Tensor,
    global_pos_weight: float = 1.0,
    local_trust_weights: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """SG-CBM concept-layer losses: global BCE and spatial soft-align KL."""
    loss_global = weighted_concept_bce(
        concept_logits,
        global_targets,
        pos_weight=global_pos_weight,
    )
    loss_mask = soft_align_kl_loss(
        spatial_logits,
        mask_indices,
        mask_targets,
        mask_valid,
        concept_weights=local_trust_weights,
    )
    return loss_global, loss_mask
