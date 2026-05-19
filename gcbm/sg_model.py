from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def pool_concept_maps(
    maps: torch.Tensor,
    *,
    pooling: str = "avg",
    topk_fraction: float = 0.2,
) -> torch.Tensor:
    """Pool concept maps into one logit per concept."""
    mode = str(pooling).lower()
    if maps.ndim == 2:
        return maps
    if mode == "avg":
        return F.adaptive_avg_pool2d(maps, 1).flatten(1)
    if mode != "topk":
        raise ValueError(f"Unsupported concept-map pooling mode: {pooling}")
    flat = maps.flatten(2)
    fraction = min(max(float(topk_fraction), 0.0), 1.0)
    k = max(1, int(math.ceil(flat.shape[-1] * fraction)))
    values, _ = flat.topk(k=k, dim=-1)
    return values.mean(dim=-1)


def pool_residual_spatial_logits(
    maps: torch.Tensor,
    *,
    pooling: str = "lse",
    temperature: float = 1.0,
) -> torch.Tensor:
    """Pool spatial residual maps while preserving constant-map scale."""
    flat = maps.flatten(2)
    mode = str(pooling).lower()
    if mode == "avg":
        return flat.mean(dim=-1)
    if mode == "lse":
        temp = max(float(temperature), 1e-6)
        num_patches = flat.shape[-1]
        return temp * torch.logsumexp(flat / temp, dim=-1) - temp * math.log(max(num_patches, 1))
    raise ValueError(f"Unsupported residual spatial pooling mode: {pooling}")


def _select_stage(feats, stage: str):
    if isinstance(feats, dict):
        return feats[str(stage).lower()]
    return feats


class DualBranchConceptLayer(nn.Module):
    """Shared SG-CBM dual-branch concept layer.

    The global branch can be a linear head over pooled conv5 features or a
    spatial head that returns concept maps. The spatial branch always returns
    concept maps for localization and soft-align supervision.
    """

    def __init__(
        self,
        global_layer: nn.Module,
        spatial_layer: nn.Module,
        *,
        spatial_stage: str = "conv5",
        global_stage: str = "conv5",
        global_pool: bool = True,
        residual_alpha: float = 0.0,
        residual_spatial_pooling: str = "lse",
        learn_spatial_residual_scale: bool = False,
    ) -> None:
        super().__init__()
        self.global_layer = global_layer
        self.spatial_layer = spatial_layer
        self.spatial_stage = str(spatial_stage).lower()
        self.global_stage = str(global_stage).lower()
        self.global_pool = bool(global_pool)
        self.residual_alpha = float(residual_alpha)
        self.residual_spatial_pooling = str(residual_spatial_pooling).lower()
        self.log_spatial_scale = nn.Parameter(torch.zeros(())) if learn_spatial_residual_scale else None

    def forward_global(self, feats) -> torch.Tensor:
        global_feats = _select_stage(feats, self.global_stage)
        if self.global_pool and global_feats.ndim == 4:
            global_feats = global_feats.mean(dim=(2, 3))
        return self.global_layer(global_feats)

    def forward_spatial(self, feats) -> torch.Tensor:
        return self.spatial_layer(_select_stage(feats, self.spatial_stage))

    def forward_both(self, feats) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward_global(feats), self.forward_spatial(feats)

    def forward(self, feats) -> Dict[str, torch.Tensor]:
        global_outputs, spatial_maps = self.forward_both(feats)
        global_logits = pool_concept_maps(global_outputs, pooling="avg")
        spatial_logits = pool_residual_spatial_logits(
            spatial_maps,
            pooling=self.residual_spatial_pooling,
        )
        spatial_scale = 1.0 if self.log_spatial_scale is None else torch.exp(self.log_spatial_scale)
        final_logits = global_logits + self.residual_alpha * spatial_scale * spatial_logits
        return {
            "global_logits": global_logits,
            "spatial_logits": spatial_logits,
            "spatial_maps": spatial_maps,
            "final_logits": final_logits,
        }


class MultiScaleConceptLayer(nn.Module):
    """Shared SG-CBM multiscale conv4/conv5 spatial fusion layer."""

    def __init__(
        self,
        global_layer: nn.Module,
        spatial_layer: nn.Module,
        *,
        conv4_dim: int,
        conv5_dim: int,
        fusion_dim: Optional[int] = None,
        global_pool: bool = True,
        residual_alpha: float = 0.0,
        residual_spatial_pooling: str = "lse",
        learn_spatial_residual_scale: bool = False,
    ) -> None:
        super().__init__()
        fusion_dim = int(fusion_dim or conv5_dim)
        self.global_layer = global_layer
        self.spatial_layer = spatial_layer
        self.conv4_proj = nn.Conv2d(int(conv4_dim), fusion_dim, kernel_size=1, bias=False)
        self.conv5_proj = nn.Conv2d(int(conv5_dim), fusion_dim, kernel_size=1, bias=False)
        self.global_pool = bool(global_pool)
        self.residual_alpha = float(residual_alpha)
        self.residual_spatial_pooling = str(residual_spatial_pooling).lower()
        self.log_spatial_scale = nn.Parameter(torch.zeros(())) if learn_spatial_residual_scale else None

    def fuse_spatial_features(self, feats) -> torch.Tensor:
        if not isinstance(feats, dict):
            raise TypeError("MultiScaleConceptLayer expects a dict with conv4 and conv5 features")
        conv4 = feats["conv4"]
        conv5 = feats["conv5"]
        conv5_up = F.interpolate(
            self.conv5_proj(conv5),
            size=conv4.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return F.relu(self.conv4_proj(conv4) + conv5_up, inplace=False)

    def forward_global(self, feats) -> torch.Tensor:
        global_feats = _select_stage(feats, "conv5")
        if self.global_pool and global_feats.ndim == 4:
            global_feats = global_feats.mean(dim=(2, 3))
        return self.global_layer(global_feats)

    def forward_spatial(self, feats) -> torch.Tensor:
        return self.spatial_layer(self.fuse_spatial_features(feats))

    def forward_both(self, feats) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward_global(feats), self.forward_spatial(feats)

    def forward(self, feats) -> Dict[str, torch.Tensor]:
        global_outputs, spatial_maps = self.forward_both(feats)
        global_logits = pool_concept_maps(global_outputs, pooling="avg")
        spatial_logits = pool_residual_spatial_logits(
            spatial_maps,
            pooling=self.residual_spatial_pooling,
        )
        spatial_scale = 1.0 if self.log_spatial_scale is None else torch.exp(self.log_spatial_scale)
        final_logits = global_logits + self.residual_alpha * spatial_scale * spatial_logits
        return {
            "global_logits": global_logits,
            "spatial_logits": spatial_logits,
            "spatial_maps": spatial_maps,
            "final_logits": final_logits,
        }
