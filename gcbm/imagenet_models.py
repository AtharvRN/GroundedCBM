from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50

from gcbm.imagenet_targets import load_concepts
from gcbm.sg_model import pool_residual_spatial_logits as shared_pool_residual_spatial_logits


def get_resnet50_weights(version: str) -> ResNet50_Weights:
    normalized = str(version or "v2").lower()
    if normalized in {"v1", "imagenet1k_v1", "imagenet1k-v1"}:
        return ResNet50_Weights.IMAGENET1K_V1
    if normalized in {"v2", "imagenet1k_v2", "imagenet1k-v2"}:
        return ResNet50_Weights.IMAGENET1K_V2
    raise ValueError(f"Unsupported ResNet-50 weights version: {version!r}")


class ResNet50Conv45(nn.Module):
    def __init__(self, weights_version: str = "v2") -> None:
        super().__init__()
        self.weights_version = str(weights_version or "v2").lower()
        model = resnet50(weights=get_resnet50_weights(self.weights_version))
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        conv4 = self.layer3(x)
        conv5 = self.layer4(conv4)
        return {"conv4": conv4, "conv5": conv5}


class SharedConceptHead(nn.Module):
    def __init__(self, n_concepts: int, spatial_stage: str) -> None:
        super().__init__()
        in_channels = 1024 if spatial_stage == "conv4" else 2048
        self.spatial_stage = spatial_stage
        self.spatial = nn.Conv2d(in_channels, n_concepts, kernel_size=1, bias=True)

    def forward(self, feats: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        spatial_maps = self.spatial(feats[self.spatial_stage])
        pooled = F.adaptive_avg_pool2d(spatial_maps, 1).flatten(1)
        return {
            "global_logits": pooled,
            "spatial_logits": torch.zeros_like(pooled),
            "spatial_maps": spatial_maps,
            "final_logits": pooled,
        }


class GlobalOnlyConceptHead(nn.Module):
    def __init__(self, n_concepts: int) -> None:
        super().__init__()
        self.global_head = nn.Linear(2048, n_concepts, bias=True)

    def forward(self, feats: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        global_feats = F.adaptive_avg_pool2d(feats["conv5"], 1).flatten(1)
        global_logits = self.global_head(global_feats)
        return {
            "global_logits": global_logits,
            "final_logits": global_logits,
        }


def pool_residual_spatial_logits(spatial_maps: torch.Tensor, pooling: str) -> torch.Tensor:
    return shared_pool_residual_spatial_logits(spatial_maps, pooling=pooling)


class DualBranchConceptHead(nn.Module):
    def __init__(
        self,
        n_concepts: int,
        spatial_stage: str,
        residual_alpha: float,
        residual_spatial_pooling: str,
        learn_spatial_residual_scale: bool = False,
    ) -> None:
        super().__init__()
        in_channels = 1024 if spatial_stage == "conv4" else 2048
        self.spatial_stage = spatial_stage
        self.global_head = nn.Linear(2048, n_concepts, bias=True)
        self.spatial = nn.Conv2d(in_channels, n_concepts, kernel_size=1, bias=True)
        self.residual_alpha = float(residual_alpha)
        self.residual_spatial_pooling = residual_spatial_pooling
        self.log_spatial_scale = nn.Parameter(torch.zeros(())) if learn_spatial_residual_scale else None

    def forward(self, feats: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        global_feats = F.adaptive_avg_pool2d(feats["conv5"], 1).flatten(1)
        global_logits = self.global_head(global_feats)
        spatial_maps = self.spatial(feats[self.spatial_stage])
        spatial_logits = pool_residual_spatial_logits(spatial_maps, self.residual_spatial_pooling)
        spatial_scale = 1.0 if self.log_spatial_scale is None else torch.exp(self.log_spatial_scale)
        final_logits = global_logits + self.residual_alpha * spatial_scale * spatial_logits
        return {
            "global_logits": global_logits,
            "spatial_logits": spatial_logits,
            "spatial_maps": spatial_maps,
            "final_logits": final_logits,
        }


class MultiScaleDualBranchConceptHead(nn.Module):
    def __init__(
        self,
        n_concepts: int,
        residual_alpha: float,
        residual_spatial_pooling: str,
        learn_spatial_residual_scale: bool = False,
        fusion_dim: int = 2048,
    ) -> None:
        super().__init__()
        self.global_head = nn.Linear(2048, n_concepts, bias=True)
        self.conv4_proj = nn.Conv2d(1024, fusion_dim, kernel_size=1, bias=False)
        self.conv5_proj = nn.Conv2d(2048, fusion_dim, kernel_size=1, bias=False)
        self.spatial = nn.Conv2d(fusion_dim, n_concepts, kernel_size=1, bias=True)
        self.residual_alpha = float(residual_alpha)
        self.residual_spatial_pooling = residual_spatial_pooling
        self.log_spatial_scale = nn.Parameter(torch.zeros(())) if learn_spatial_residual_scale else None

    def _fuse(self, feats: Dict[str, torch.Tensor]) -> torch.Tensor:
        conv4 = feats["conv4"]
        conv5_up = F.interpolate(
            self.conv5_proj(feats["conv5"]),
            size=conv4.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return F.relu(self.conv4_proj(conv4) + conv5_up, inplace=False)

    def forward(self, feats: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        global_feats = F.adaptive_avg_pool2d(feats["conv5"], 1).flatten(1)
        global_logits = self.global_head(global_feats)
        spatial_maps = self.spatial(self._fuse(feats))
        spatial_logits = pool_residual_spatial_logits(spatial_maps, self.residual_spatial_pooling)
        spatial_scale = 1.0 if self.log_spatial_scale is None else torch.exp(self.log_spatial_scale)
        final_logits = global_logits + self.residual_alpha * spatial_scale * spatial_logits
        return {
            "global_logits": global_logits,
            "spatial_logits": spatial_logits,
            "spatial_maps": spatial_maps,
            "final_logits": final_logits,
        }


def build_model(cfg: Any, n_concepts: int) -> Tuple[nn.Module, nn.Module]:
    backbone = ResNet50Conv45(weights_version=getattr(cfg, "resnet50_weights", "v2")).to(cfg.device)
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    backbone.eval()
    if cfg.channels_last:
        backbone.to(memory_format=torch.channels_last)
    if cfg.branch_arch == "global_only":
        head = GlobalOnlyConceptHead(n_concepts=n_concepts)
    elif cfg.spatial_branch_mode == "multiscale_conv45":
        if cfg.branch_arch != "dual":
            raise ValueError("multiscale_conv45 requires branch_arch=dual")
        head = MultiScaleDualBranchConceptHead(
            n_concepts=n_concepts,
            residual_alpha=cfg.residual_alpha,
            residual_spatial_pooling=getattr(cfg, "residual_spatial_pooling", "avg"),
            learn_spatial_residual_scale=bool(getattr(cfg, "learn_spatial_residual_scale", False)),
        )
    elif cfg.branch_arch == "dual":
        head = DualBranchConceptHead(
            n_concepts=n_concepts,
            spatial_stage=cfg.spatial_stage,
            residual_alpha=cfg.residual_alpha,
            residual_spatial_pooling=getattr(cfg, "residual_spatial_pooling", "avg"),
            learn_spatial_residual_scale=bool(getattr(cfg, "learn_spatial_residual_scale", False)),
        )
    else:
        head = SharedConceptHead(n_concepts=n_concepts, spatial_stage=cfg.spatial_stage)
    head = head.to(cfg.device)
    if cfg.channels_last:
        head.to(memory_format=torch.channels_last)
    return backbone, head


def init_global_head_from_vlg(head: nn.Module, cfg: Any, concepts: Sequence[str]) -> None:
    if not cfg.vlg_init_path:
        return
    if not hasattr(head, "global_head"):
        print(f"[vlg_init] skipping: head type {type(head).__name__} has no global_head", flush=True)
        return

    vlg_state = torch.load(cfg.vlg_init_path, map_location="cpu")
    if isinstance(vlg_state, dict) and "state_dict" in vlg_state and isinstance(vlg_state["state_dict"], dict):
        vlg_state = vlg_state["state_dict"]
    weight = vlg_state.get("model.0.weight")
    bias = vlg_state.get("model.0.bias")
    if weight is None or bias is None:
        raise KeyError(f"Could not find VLG weights in {cfg.vlg_init_path}")

    vlg_concepts = load_concepts(cfg.vlg_concepts_path)
    if len(vlg_concepts) != int(weight.shape[0]):
        raise ValueError(
            f"VLG concept count mismatch: {len(vlg_concepts)} concepts for weight rows {int(weight.shape[0])}"
        )
    vlg_concept_to_idx = {concept: idx for idx, concept in enumerate(vlg_concepts)}
    target_head = head.global_head
    if tuple(weight.shape) != tuple(target_head.weight.shape):
        if int(weight.shape[1]) != int(target_head.weight.shape[1]):
            raise ValueError(f"VLG init feature dim mismatch: {tuple(weight.shape)} vs {tuple(target_head.weight.shape)}")
    matched = 0
    with torch.no_grad():
        for our_idx, concept in enumerate(concepts):
            vlg_idx = vlg_concept_to_idx.get(concept)
            if vlg_idx is None:
                continue
            target_head.weight[our_idx].copy_(weight[vlg_idx])
            target_head.bias[our_idx].copy_(bias[vlg_idx])
            matched += 1
    print(f"[vlg_init] matched {matched}/{len(concepts)} concepts from {cfg.vlg_init_path}", flush=True)

    if cfg.freeze_global_head:
        for parameter in target_head.parameters():
            parameter.requires_grad = False
        print("[vlg_init] global head frozen", flush=True)
