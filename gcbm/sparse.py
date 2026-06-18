from __future__ import annotations

import torch


def threshold_weight_truncation(weight: torch.Tensor, sparsity: float) -> torch.Tensor:
    """Globally keep the largest-magnitude final-layer weights.

    This matches the NEC post-processing used by VLG-CBM: for a classifier
    weight matrix of shape [classes, concepts], sparsity=NEC/num_concepts keeps
    approximately NEC * num_classes nonzero entries.
    """
    numel = int(weight.numel())
    keep = int(round(float(sparsity) * numel))
    if keep <= 0:
        return torch.zeros_like(weight)
    if keep >= numel:
        return weight.clone().detach()
    threshold = weight.abs().flatten().topk(keep).values[-1]
    sparse_weight = weight.clone().detach()
    sparse_weight[weight.abs() < threshold] = 0
    return sparse_weight
