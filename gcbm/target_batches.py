from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import torch


def pad_sparse_targets(
    mask_indices: Sequence[torch.Tensor],
    mask_targets: Sequence[torch.Tensor],
    *,
    mask_h: int,
    mask_w: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad sparse per-image concept masks into batched tensors."""
    batch_size = len(mask_indices)
    max_k = max((int(indices.numel()) for indices in mask_indices), default=0)
    if max_k == 0:
        return (
            torch.full((batch_size, 1), -1, dtype=torch.long),
            torch.zeros((batch_size, 1, int(mask_h), int(mask_w)), dtype=torch.float32),
            torch.zeros((batch_size, 1), dtype=torch.bool),
        )

    idx_pad = torch.full((batch_size, max_k), -1, dtype=torch.long)
    mask_pad = torch.zeros((batch_size, max_k, int(mask_h), int(mask_w)), dtype=torch.float32)
    valid = torch.zeros((batch_size, max_k), dtype=torch.bool)
    for batch_index, (indices, masks) in enumerate(zip(mask_indices, mask_targets)):
        count = int(indices.numel())
        if count == 0:
            continue
        idx_pad[batch_index, :count] = indices
        mask_pad[batch_index, :count] = masks
        valid[batch_index, :count] = True
    return idx_pad, mask_pad, valid


def batch_targets_to_device(
    batch: Dict[str, Any],
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        batch["global_targets"].to(device, non_blocking=True),
        batch["mask_indices"].to(device, non_blocking=True),
        batch["mask_targets"].to(device, non_blocking=True),
        batch["mask_valid"].to(device, non_blocking=True),
    )
