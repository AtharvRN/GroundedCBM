from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

NCCMode = Literal["all_classes", "predicted_class", "target_class"]


@dataclass
class NCCAccumulator:
    tau: float
    mode: NCCMode
    total_count: int = 0
    total_ncc: float = 0.0

    def update(self, counts: torch.Tensor) -> None:
        counts = counts.detach().float().cpu()
        self.total_count += int(counts.numel())
        self.total_ncc += float(counts.sum().item())

    @property
    def mean(self) -> float:
        if self.total_count == 0:
            return float("nan")
        return self.total_ncc / float(self.total_count)

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "ncc_tau": float(self.tau),
            "ncc_mode": self.mode,
            "ncc": float(self.mean),
            "ncc_count": int(self.total_count),
        }


def coverage_counts(abs_contrib: torch.Tensor, tau: float) -> torch.Tensor:
    """Return the fewest top concepts needed to cover tau of each row's mass."""
    if abs_contrib.ndim < 2:
        raise ValueError("abs_contrib must have shape [..., concepts]")
    if not 0.0 <= float(tau) <= 1.0:
        raise ValueError("tau must be in [0, 1]")

    flat = abs_contrib.reshape(-1, abs_contrib.shape[-1]).float()
    if float(tau) <= 0.0:
        return torch.zeros(flat.shape[0], dtype=torch.long, device=flat.device).reshape(abs_contrib.shape[:-1])

    total = flat.sum(dim=-1)
    sorted_contrib = torch.sort(flat, dim=-1, descending=True).values
    cumsum = sorted_contrib.cumsum(dim=-1)
    threshold = total * float(tau)
    counts = (cumsum >= threshold.unsqueeze(-1)).int().argmax(dim=-1).long() + 1
    counts = torch.where(total > 0, counts, torch.zeros_like(counts))
    return counts.reshape(abs_contrib.shape[:-1])


def ncc_counts_for_batch(
    concept_logits: torch.Tensor,
    weight: torch.Tensor,
    *,
    tau: float = 0.95,
    mode: NCCMode = "all_classes",
    bias: torch.Tensor | None = None,
    targets: torch.Tensor | None = None,
    class_chunk_size: int = 64,
) -> torch.Tensor:
    """Compute NCC counts for one batch and one linear classifier.

    concept_logits has shape [batch, concepts] and weight has shape
    [classes, concepts]. In all_classes mode, the returned tensor has shape
    [batch, classes]; otherwise it has shape [batch].
    """
    if concept_logits.ndim != 2:
        raise ValueError("concept_logits must have shape [batch, concepts]")
    if weight.ndim != 2:
        raise ValueError("weight must have shape [classes, concepts]")
    if concept_logits.shape[1] != weight.shape[1]:
        raise ValueError(
            f"concept dim mismatch: logits={tuple(concept_logits.shape)} weight={tuple(weight.shape)}"
        )

    concept_logits = concept_logits.float()
    weight = weight.float()
    mode = str(mode)

    if mode == "predicted_class":
        if bias is None:
            bias = torch.zeros(weight.shape[0], dtype=weight.dtype, device=weight.device)
        logits = concept_logits @ weight.t() + bias.to(concept_logits.device).float()
        classes = logits.argmax(dim=-1)
        selected_weight = weight.to(concept_logits.device)[classes]
        return coverage_counts((concept_logits * selected_weight).abs(), tau)

    if mode == "target_class":
        if targets is None:
            raise ValueError("targets are required for target_class NCC")
        classes = targets.to(concept_logits.device).long()
        selected_weight = weight.to(concept_logits.device)[classes]
        return coverage_counts((concept_logits * selected_weight).abs(), tau)

    if mode != "all_classes":
        raise ValueError(f"unsupported NCC mode: {mode}")

    counts = []
    weight_device = weight.to(concept_logits.device)
    chunk = max(1, int(class_chunk_size))
    for start in range(0, weight_device.shape[0], chunk):
        class_weights = weight_device[start : start + chunk]
        contrib = (concept_logits[:, None, :] * class_weights[None, :, :]).abs()
        counts.append(coverage_counts(contrib, tau))
    return torch.cat(counts, dim=1)
