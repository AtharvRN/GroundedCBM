import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

import clip
from data import utils as data_utils
from glm_saga.elasticnet import IndexedTensorDataset
from methods.common import build_run_dir, save_args, write_artifacts

# Try to import timm for ViT support
try:
    import timm
    from einops import rearrange
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    timm = None
    rearrange = None
from methods.lf import (
    TransformedSubset,
    cos_similarity_cubed,
    get_lf_concepts,
    subset_targets,
    use_original_label_free_protocol,
)
from model.cbm import train_dense_final, train_sparse_final


RESNET_SPATIAL_BACKBONES = frozenset(
    {
        "resnet18_cub",
        "resnet50_cub",
        "resnet50_cub_mm",
        "resnet50",
        "resnet50_partimagenetpp_scratch",
    }
)


class RawSubset(Dataset):
    def __init__(self, base_dataset: Dataset, indices: Iterable[int]):
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.targets = subset_targets(base_dataset, self.indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        image, target = self.base_dataset[self.indices[idx]]
        return image, target


def use_partimagenetpp_train_val_split(args) -> bool:
    return (
        getattr(args, "dataset", None) == "partimagenetpp"
        and bool(os.environ.get("PARTIMAGENETPP_TRAIN_MANIFEST"))
        and bool(os.environ.get("PARTIMAGENETPP_VAL_MANIFEST"))
        and float(getattr(args, "partimagenetpp_train_val_split", 0.0)) > 0.0
    )


def pil_collate(batch):
    images = [sample[0] for sample in batch]
    labels = torch.tensor([sample[1] for sample in batch], dtype=torch.long)
    return images, labels


class CLIPSpatialRN50Backbone(nn.Module):
    def __init__(self, backbone_name: str, device: str = "cuda"):
        super().__init__()
        if backbone_name != "clip_RN50":
            raise NotImplementedError(
                "SALF first pass currently supports only clip_RN50 as the spatial backbone."
            )
        model, preprocess = clip.load(backbone_name[5:], device=device)
        visual = model.visual.float()
        self.visual = visual
        self.preprocess = preprocess
        self.output_dim = 2048
        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        x = x.type(self.visual.conv1.weight.dtype)
        x = self.visual.relu1(self.visual.bn1(self.visual.conv1(x)))
        x = self.visual.relu2(self.visual.bn2(self.visual.conv2(x)))
        x = self.visual.relu3(self.visual.bn3(self.visual.conv3(x)))
        x = self.visual.avgpool(x)
        x = self.visual.layer1(x)
        x = self.visual.layer2(x)
        x = self.visual.layer3(x)
        x = self.visual.layer4(x)
        return x.float()


class CLIPSpatialViTBackbone(nn.Module):
    def __init__(self, backbone_name: str, device: str = "cuda"):
        super().__init__()
        if not backbone_name.startswith("clip_"):
            raise ValueError(
                f"CLIPSpatialViTBackbone expects a clip_* backbone name, got {backbone_name}."
            )
        model, preprocess = clip.load(backbone_name[5:], device=device)
        visual = model.visual.float()
        required = ("conv1", "class_embedding", "positional_embedding", "ln_pre", "transformer", "ln_post")
        missing = [attr for attr in required if not hasattr(visual, attr)]
        if missing:
            raise NotImplementedError(
                f"CLIPSpatialViTBackbone requires a CLIP ViT visual backbone. Missing={missing} for {backbone_name}."
            )
        self.visual = visual
        self.preprocess = preprocess
        self.device = device

        pos_len = int(self.visual.positional_embedding.shape[0])
        grid_tokens = pos_len - 1
        grid = int(round(grid_tokens**0.5))
        if grid * grid != grid_tokens:
            raise RuntimeError(
                f"Unexpected CLIP ViT positional embedding length={pos_len} for {backbone_name} (not square grid)."
            )
        self.grid = grid
        self.output_dim = int(self.visual.conv1.out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        x = x.type(self.visual.conv1.weight.dtype)

        x = self.visual.conv1(x)  # [B, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # [B, width, grid^2]
        x = x.permute(0, 2, 1)  # [B, grid^2, width]
        x = torch.cat(
            [
                self.visual.class_embedding.to(x.dtype)
                + torch.zeros(
                    x.shape[0],
                    1,
                    x.shape[-1],
                    dtype=x.dtype,
                    device=x.device,
                ),
                x,
            ],
            dim=1,
        )  # [B, grid^2 + 1, width]
        x = x + self.visual.positional_embedding.to(x.dtype)
        x = self.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        # Apply the post-transformer layer norm to patch tokens (CLIP applies it
        # to the class token only in encode_image).
        patch_tokens = self.visual.ln_post(x[:, 1:, :])  # [B, grid^2, width]
        patch_tokens = patch_tokens.permute(0, 2, 1).reshape(
            patch_tokens.shape[0],
            patch_tokens.shape[-1],
            self.grid,
            self.grid,
        )
        return patch_tokens.float()


class TIMMViTSpatialBackbone(nn.Module):
    """
    ViT backbone using timm library for SAVLG dual-branch architecture.
    Returns separate global (CLS token) and spatial (patch tokens) features.
    """

    def __init__(self, model_name: str, pretrained: bool = True, device: str = "cuda"):
        super().__init__()
        if not TIMM_AVAILABLE:
            raise ImportError(
                "timm and einops not installed. Run: pip install timm einops"
            )

        print(f"[TIMMViTSpatialBackbone] Loading {model_name}...", flush=True)

        # Load ViT model
        self.vit = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
        ).to(device)

        # Get model parameters
        self.embed_dim = self.vit.embed_dim
        self.num_patches = self.vit.patch_embed.num_patches
        self.grid = int(self.num_patches ** 0.5)
        self.output_dim = self.embed_dim
        self.device = device
        self.model_name = model_name

        # Build timm's standard preprocessing transform for this model
        data_config = timm.data.resolve_model_data_config(self.vit)
        self.preprocess = timm.data.create_transform(**data_config, is_training=False)

        print(f"[TIMMViTSpatialBackbone] Loaded {model_name}", flush=True)
        print(f"  embed_dim: {self.embed_dim}", flush=True)
        print(f"  grid_size: {self.grid}×{self.grid}", flush=True)
        print(f"  num_patches: {self.num_patches}", flush=True)

    def forward(self, x: torch.Tensor) -> dict:
        """
        Returns dict with 'global' and 'spatial' keys for SAVLG dual-branch.

        Args:
            x: B × 3 × H × W

        Returns:
            dict with:
                'global': B × embed_dim (CLS token features)
                'spatial': B × embed_dim × grid × grid (patch token features)
        """
        x = x.to(self.device)

        # Extract all tokens (CLS + patches)
        tokens = self.vit.forward_features(x)  # B × (1+num_patches) × embed_dim

        # Split CLS and patch tokens
        cls_token = tokens[:, 0]  # B × embed_dim
        patch_tokens = tokens[:, 1:]  # B × num_patches × embed_dim

        # Reshape patch tokens to spatial grid
        spatial_feats = rearrange(
            patch_tokens,
            'b (h w) c -> b c h w',
            h=self.grid,
            w=self.grid
        )  # B × embed_dim × grid × grid

        return {
            'global': cls_token.float(),
            'spatial': spatial_feats.float(),
        }


class DualBranchViTConceptLayer(nn.Module):
    """
    Concept layer for ViT that handles separate global and spatial inputs.
    Unlike ResNet-based SAVLG, global and spatial are not derived from same maps.
    """

    def __init__(self, embed_dim: int, n_concepts: int, device: str = "cuda"):
        super().__init__()

        # Global branch: Linear projection from CLS token
        self.global_projection = nn.Linear(embed_dim, n_concepts, bias=True).to(device)

        # Spatial branch: 1×1 conv from patch tokens
        self.spatial_projection = nn.Conv2d(embed_dim, n_concepts, kernel_size=1, bias=True).to(device)

        print(f"[DualBranchViTConceptLayer] Created", flush=True)
        print(f"  embed_dim: {embed_dim}", flush=True)
        print(f"  n_concepts: {n_concepts}", flush=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """For compatibility, forward on spatial features only."""
        if isinstance(x, dict):
            return self.spatial_projection(x['spatial'])
        return self.spatial_projection(x)

    def forward_global(self, x: torch.Tensor) -> torch.Tensor:
        """Forward global branch only."""
        if isinstance(x, dict):
            return self.global_projection(x['global'])
        raise ValueError("Expected dict with 'global' key for ViT")

    def forward_spatial(self, x: torch.Tensor) -> torch.Tensor:
        """Forward spatial branch only."""
        if isinstance(x, dict):
            return self.spatial_projection(x['spatial'])
        return self.spatial_projection(x)

    def forward_both(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward both branches from ViT features dict.

        Args:
            x: dict with 'global' (B × embed_dim) and 'spatial' (B × embed_dim × H × W)

        Returns:
            global_outputs: B × n_concepts (logits)
            spatial_maps: B × n_concepts × H × W (logits)
        """
        if not isinstance(x, dict):
            raise ValueError("ViT concept layer expects dict with 'global' and 'spatial' keys")

        global_outputs = self.global_projection(x['global'])  # B × n_concepts
        spatial_maps = self.spatial_projection(x['spatial'])  # B × n_concepts × H × W

        return global_outputs, spatial_maps


class SpatialBackbone(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        device: str = "cuda",
        spatial_stage: str = "conv5",
        checkpoint_path: str = "",
    ):
        super().__init__()
        self.device = device
        self.backbone_name = backbone_name
        self.spatial_stage = spatial_stage
        self.checkpoint_path = str(checkpoint_path or "")
        if backbone_name.startswith("clip_"):
            if backbone_name == "clip_RN50":
                if spatial_stage != "conv5":
                    raise ValueError(
                        f"clip_RN50 spatial backbone only supports spatial_stage='conv5', got {spatial_stage}."
                    )
                clip_backbone = CLIPSpatialRN50Backbone(backbone_name, device=device)
                self.backbone = clip_backbone
                self.preprocess = clip_backbone.preprocess
                self.output_dim = clip_backbone.output_dim
                self.stage_dims = {"conv5": clip_backbone.output_dim}
                return

            if spatial_stage != "conv5":
                raise ValueError(
                    f"{backbone_name} spatial backbone currently supports only spatial_stage='conv5', got {spatial_stage}."
                )
            clip_backbone = CLIPSpatialViTBackbone(backbone_name, device=device)
            self.backbone = clip_backbone
            self.preprocess = clip_backbone.preprocess
            self.output_dim = clip_backbone.output_dim
            # Keep the conv-stage naming to avoid requiring SAVLG/SALF config changes.
            self.stage_dims = {"conv5": clip_backbone.output_dim}
            return

        # Handle timm ViT models
        if backbone_name.startswith("vit_") or backbone_name.startswith("swin_"):
            if not TIMM_AVAILABLE:
                raise ImportError(
                    f"Requested ViT backbone {backbone_name} but timm is not installed. "
                    "Run: pip install timm einops"
                )
            if spatial_stage != "conv5":
                raise ValueError(
                    f"ViT backbones currently support only spatial_stage='conv5', got {spatial_stage}."
                )
            vit_backbone = TIMMViTSpatialBackbone(backbone_name, pretrained=True, device=device)
            self.backbone = vit_backbone
            self.preprocess = vit_backbone.preprocess
            self.output_dim = vit_backbone.output_dim
            self.stage_dims = {"conv5": vit_backbone.output_dim}
            self.is_vit = True  # Flag to indicate ViT backbone
            return

        self.is_vit = False  # ResNet-style backbone
        if backbone_name in RESNET_SPATIAL_BACKBONES:
            print(
                f"[SpatialBackbone] loading backbone={backbone_name} stage={spatial_stage}",
                flush=True,
            )
            if backbone_name == "resnet50_partimagenetpp_scratch":
                if not self.checkpoint_path:
                    raise ValueError(
                        "backbone=resnet50_partimagenetpp_scratch requires --backbone_checkpoint."
                    )
                target_model, preprocess = data_utils.load_resnet50_checkpoint(
                    self.checkpoint_path, device
                )
            else:
                target_model, preprocess = data_utils.get_target_model(backbone_name, device)
            print(
                f"[SpatialBackbone] target model ready for backbone={backbone_name}",
                flush=True,
            )
            if hasattr(target_model, "features"):
                feature_children = list(target_model.features.children())
                self.res_init = feature_children[0].to(device).float().eval()
                self.res_layer1 = feature_children[1].to(device).float().eval()
                self.res_layer2 = feature_children[2].to(device).float().eval()
                self.res_layer3 = feature_children[3].to(device).float().eval()
                self.res_layer4 = feature_children[4].to(device).float().eval()
            else:
                self.res_conv1 = target_model.conv1.to(device).float().eval()
                self.res_bn1 = target_model.bn1.to(device).float().eval()
                self.res_relu = target_model.relu.to(device).float().eval()
                self.res_maxpool = target_model.maxpool.to(device).float().eval()
                self.res_layer1 = target_model.layer1.to(device).float().eval()
                self.res_layer2 = target_model.layer2.to(device).float().eval()
                self.res_layer3 = target_model.layer3.to(device).float().eval()
                self.res_layer4 = target_model.layer4.to(device).float().eval()
            if backbone_name == "resnet18_cub":
                self.stage_dims = {
                    "conv3": 128,
                    "conv4": 256,
                    "conv5": 512,
                }
            else:
                self.stage_dims = {
                    "conv3": 512,
                    "conv4": 1024,
                    "conv5": 2048,
                }
            if spatial_stage not in self.stage_dims:
                raise ValueError(
                    f"Unsupported spatial_stage={spatial_stage} for {backbone_name}. Expected one of {sorted(self.stage_dims)}."
                )
            self.preprocess = preprocess
            self.output_dim = self.stage_dims[spatial_stage]
            return
        raise NotImplementedError(
            f"SALF first pass currently supports only clip_RN50, {', '.join(sorted(RESNET_SPATIAL_BACKBONES))}, and supported ViTs as spatial backbones, got {backbone_name}."
        )

    def get_stage_dim(self, stage_name: str) -> int:
        if stage_name not in self.stage_dims:
            raise ValueError(
                f"Unsupported stage_name={stage_name} for backbone={self.backbone_name}. Expected one of {sorted(self.stage_dims)}."
            )
        return int(self.stage_dims[stage_name])

    def _forward_resnet_cub_stages(
        self,
        x: torch.Tensor,
        stage_names: Iterable[str],
    ) -> dict[str, torch.Tensor]:
        requested = set(stage_names)
        missing = requested.difference(self.stage_dims)
        if missing:
            raise ValueError(
                f"Unsupported {self.backbone_name} stage request(s): {sorted(missing)}. Expected subset of {sorted(self.stage_dims)}."
            )

        x = x.to(self.device)
        if hasattr(self, "res_conv1"):
            x = self.res_conv1(x)
            x = self.res_bn1(x)
            x = self.res_relu(x)
            x = self.res_maxpool(x)
            x = self.res_layer1(x)
        else:
            x = self.res_init(x)
            x = self.res_layer1(x)
        conv3 = self.res_layer2(x).float()
        conv4 = self.res_layer3(conv3).float()
        conv5 = self.res_layer4(conv4).float()

        outputs: dict[str, torch.Tensor] = {}
        if "conv3" in requested:
            outputs["conv3"] = conv3
        if "conv4" in requested:
            outputs["conv4"] = conv4
        if "conv5" in requested:
            outputs["conv5"] = conv5
        return outputs

    def forward_multistage(
        self,
        x: torch.Tensor,
        stage_names: Sequence[str],
    ) -> dict[str, torch.Tensor]:
        if self.backbone_name.startswith("clip_"):
            requested = set(stage_names)
            if requested != {"conv5"}:
                raise ValueError(
                    f"{self.backbone_name} only supports conv5 feature extraction, got {sorted(requested)}."
                )
            conv5 = self.backbone(x.to(self.device)).float()
            return {"conv5": conv5}
        if self.backbone_name in RESNET_SPATIAL_BACKBONES:
            return self._forward_resnet_cub_stages(x, stage_names)
        raise NotImplementedError(
            f"Unsupported multistage request for backbone={self.backbone_name}."
        )

    def forward(self, x: torch.Tensor):
        if self.backbone_name in RESNET_SPATIAL_BACKBONES:
            return self._forward_resnet_cub_stages(x, [self.spatial_stage])[self.spatial_stage]

        # ViT backbones return dict, others return tensor
        output = self.backbone(x.to(self.device))
        if isinstance(output, dict):
            return output  # ViT: return dict as-is
        return output.float()  # ResNet/CLIP: return tensor


class ConceptSpatialMLP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_concepts: int,
        num_hidden: int = 1,
        hidden_dim: int | None = None,
        use_batchnorm: bool = False,
    ):
        super().__init__()
        hdim = (
            int(hidden_dim)
            if hidden_dim is not None and int(hidden_dim) > 0
            else int(in_channels)
        )
        layers = []
        dim = int(in_channels)
        for _ in range(max(1, int(num_hidden))):
            layers.append(nn.Conv2d(dim, hdim, kernel_size=1, bias=not use_batchnorm))
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(hdim))
            layers.append(nn.ReLU(inplace=True))
            dim = hdim
        layers.append(nn.Conv2d(dim, n_concepts, kernel_size=1, bias=False))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class DualBranchSpatialConceptLayer(nn.Module):
    def __init__(self, global_layer: nn.Module, spatial_layer: nn.Module):
        super().__init__()
        self.global_layer = global_layer
        self.spatial_layer = spatial_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial_layer(x)

    def forward_global(self, x: torch.Tensor) -> torch.Tensor:
        return self.global_layer(x)

    def forward_spatial(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial_layer(x)

    def forward_both(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.global_layer(x), self.spatial_layer(x)


def build_single_spatial_concept_layer(args, in_channels: int, n_concepts: int) -> nn.Module:
    if args.cbl_type == "linear":
        return nn.Conv2d(
            in_channels,
            n_concepts,
            kernel_size=1,
            bias=bool(getattr(args, "salf_cbl_bias", False)),
        ).to(args.device)
    if args.cbl_type == "mlp":
        return ConceptSpatialMLP(
            in_channels=in_channels,
            n_concepts=n_concepts,
            num_hidden=max(1, int(args.cbl_hidden_layers)),
            hidden_dim=(
                int(args.cbl_hidden_dim)
                if getattr(args, "cbl_hidden_dim", 0) > 0
                else None
            ),
            use_batchnorm=bool(args.cbl_use_batchnorm),
        ).to(args.device)
    raise ValueError(f"Unsupported SALF cbl_type={args.cbl_type}")


def build_spatial_concept_layer(args, in_channels: int, n_concepts: int, is_vit: bool = False) -> nn.Module:
    """
    Build concept projection layer.

    Args:
        is_vit: If True, builds ViT-specific dual-branch layer that handles separate global/spatial inputs
    """
    if is_vit:
        # ViT requires dual-branch architecture (global and spatial are separate)
        print(f"[build_spatial_concept_layer] Building ViT dual-branch concept layer", flush=True)
        return DualBranchViTConceptLayer(
            embed_dim=in_channels,
            n_concepts=n_concepts,
            device=args.device,
        )

    # ResNet-style: single spatial map, global derived by pooling
    branch_arch = str(getattr(args, "savlg_branch_arch", "shared")).lower()
    if branch_arch == "shared":
        return build_single_spatial_concept_layer(args, in_channels, n_concepts)
    if branch_arch == "dual":
        return DualBranchSpatialConceptLayer(
            global_layer=build_single_spatial_concept_layer(args, in_channels, n_concepts),
            spatial_layer=build_single_spatial_concept_layer(args, in_channels, n_concepts),
        ).to(args.device)
    raise ValueError(f"Unsupported SAVLG branch architecture: {branch_arch}")


def draw_prompt(pil_img: Image.Image, center: Tuple[int, int], radius: int) -> Image.Image:
    img = pil_img.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    x, y = center
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline="red", width=3)
    return img


def infer_clip_input_size(clip_preprocess) -> int:
    for transform in getattr(clip_preprocess, "transforms", []):
        size = getattr(transform, "size", None)
        if isinstance(size, int):
            return int(size)
        if isinstance(size, (tuple, list)) and len(size) >= 1:
            return int(size[0])
    return 224


def resize_and_center_crop_for_prompt(
    pil_img: Image.Image,
    clip_preprocess,
) -> Image.Image:
    input_size = infer_clip_input_size(clip_preprocess)
    img = pil_img.convert("RGB")
    img = TF.resize(img, input_size, interpolation=InterpolationMode.BICUBIC)
    img = TF.center_crop(img, [input_size, input_size])
    return img


def _get_clip_normalize_stats(clip_preprocess) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    for transform in getattr(clip_preprocess, "transforms", []):
        mean = getattr(transform, "mean", None)
        std = getattr(transform, "std", None)
        if mean is not None and std is not None:
            return tuple(float(x) for x in mean), tuple(float(x) for x in std)
    # OpenAI CLIP defaults
    return (
        (0.48145466, 0.4578275, 0.40821073),
        (0.26862954, 0.26130258, 0.27577711),
    )


def _build_prompt_grid_metadata(
    image_size: int,
    grid_h: int,
    grid_w: int,
    radius: int,
    ring_half_width: float = 1.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xs = np.linspace(
        int(radius),
        max(int(radius), image_size - int(radius) - 1),
        int(grid_w),
    ).astype(int)
    ys = np.linspace(
        int(radius),
        max(int(radius), image_size - int(radius) - 1),
        int(grid_h),
    ).astype(int)
    centers = torch.tensor([(x, y) for y in ys for x in xs], dtype=torch.float32)

    yy, xx = torch.meshgrid(
        torch.arange(image_size, dtype=torch.float32),
        torch.arange(image_size, dtype=torch.float32),
        indexing="ij",
    )
    yy = yy.unsqueeze(0)
    xx = xx.unsqueeze(0)
    cx = centers[:, 0].view(-1, 1, 1)
    cy = centers[:, 1].view(-1, 1, 1)
    dist = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    masks = ((dist >= float(radius) - ring_half_width) & (dist <= float(radius) + ring_half_width)).float()
    return centers, masks


def _prepare_base_clip_tensors(
    pil_images: Sequence[Image.Image],
    clip_preprocess,
) -> torch.Tensor:
    processed = []
    for pil in pil_images:
        pil = resize_and_center_crop_for_prompt(pil, clip_preprocess)
        processed.append(TF.to_tensor(pil))
    return torch.stack(processed, dim=0)


def _apply_prompt_masks_to_batch(
    base_images: torch.Tensor,
    prompt_masks: torch.Tensor,
    clip_preprocess,
) -> torch.Tensor:
    mean, std = _get_clip_normalize_stats(clip_preprocess)
    mean_t = torch.tensor(mean, dtype=base_images.dtype, device=base_images.device).view(1, 1, 3, 1, 1)
    std_t = torch.tensor(std, dtype=base_images.dtype, device=base_images.device).view(1, 1, 3, 1, 1)
    red = torch.tensor([1.0, 0.0, 0.0], dtype=base_images.dtype, device=base_images.device).view(1, 1, 3, 1, 1)

    # base_images: [B, 3, H, W], prompt_masks: [G, H, W]
    base = base_images.unsqueeze(1)
    mask = prompt_masks.to(device=base_images.device, dtype=base_images.dtype).unsqueeze(0).unsqueeze(2)
    prompted = base * (1.0 - mask) + red * mask
    prompted = (prompted - mean_t) / std_t
    return prompted.reshape(-1, prompted.shape[2], prompted.shape[3], prompted.shape[4])


def create_salf_splits(args):
    backbone = SpatialBackbone(
        args.backbone,
        device=args.device,
        spatial_stage=getattr(args, "savlg_spatial_stage", "conv5"),
        checkpoint_path=getattr(args, "backbone_checkpoint", ""),
    )

    clip_name = getattr(args, "lf_clip_name", None) or args.backbone
    clip_model, clip_preprocess = clip.load(clip_name.replace("clip_", ""), device=args.device)
    clip_model = clip_model.float().eval()

    has_explicit_partimagenetpp_splits = (
        getattr(args, "dataset", None) == "partimagenetpp"
        and bool(os.environ.get("PARTIMAGENETPP_TRAIN_MANIFEST"))
        and bool(os.environ.get("PARTIMAGENETPP_VAL_MANIFEST"))
    )
    if use_original_label_free_protocol(args) or has_explicit_partimagenetpp_splits:
        base_train_raw = data_utils.get_data(f"{args.dataset}_train", None)
        base_val_raw = data_utils.get_data(f"{args.dataset}_val", None)
        max_train = int(getattr(args, "max_train_images", 0) or 0)
        max_test = int(getattr(args, "max_test_images", 0) or 0)
        train_total = min(len(base_train_raw), max_train) if max_train > 0 else len(base_train_raw)
        val_total = min(len(base_val_raw), max_test) if max_test > 0 else len(base_val_raw)
        train_indices = list(range(train_total))
        inner_val_indices = []
        if use_partimagenetpp_train_val_split(args):
            val_fraction = float(getattr(args, "partimagenetpp_train_val_split", 0.0))
            if not 0.0 < val_fraction < 1.0:
                raise ValueError(
                    "partimagenetpp_train_val_split must be in (0, 1) when enabled."
                )
            n_val = max(1, int(round(val_fraction * train_total)))
            if n_val >= train_total:
                raise ValueError(
                    "partimagenetpp_train_val_split leaves no training examples."
                )
            generator = torch.Generator().manual_seed(args.seed)
            shuffled = torch.randperm(train_total, generator=generator).tolist()
            inner_val_indices = shuffled[:n_val]
            train_indices = shuffled[n_val:]
        val_indices = list(range(val_total))
        train_raw = RawSubset(base_train_raw, train_indices)
        val_raw = RawSubset(
            base_train_raw if inner_val_indices else base_val_raw,
            inner_val_indices if inner_val_indices else val_indices,
        )
        train_dataset = TransformedSubset(base_train_raw, train_indices, backbone.preprocess)
        val_dataset = TransformedSubset(
            base_train_raw if inner_val_indices else base_val_raw,
            inner_val_indices if inner_val_indices else val_indices,
            backbone.preprocess,
        )
        test_dataset = TransformedSubset(base_val_raw, val_indices, backbone.preprocess)
        return train_raw, val_raw, train_dataset, val_dataset, test_dataset, backbone, clip_model, clip_preprocess

    base_train_raw = data_utils.get_data(f"{args.dataset}_train", None)
    max_train = int(getattr(args, "max_train_images", 0) or 0)
    total = min(len(base_train_raw), max_train) if max_train > 0 else len(base_train_raw)
    n_val = int(args.val_split * total)
    if args.val_split > 0 and n_val == 0 and total > 1:
        n_val = 1
    n_train = total - n_val
    generator = torch.Generator().manual_seed(args.seed)
    train_subset, val_subset = torch.utils.data.random_split(
        list(range(total)),
        [n_train, n_val],
        generator=generator,
    )
    train_raw = RawSubset(base_train_raw, train_subset.indices)
    val_raw = RawSubset(base_train_raw, val_subset.indices)
    train_dataset = TransformedSubset(base_train_raw, train_subset.indices, backbone.preprocess)
    val_dataset = TransformedSubset(base_train_raw, val_subset.indices, backbone.preprocess)
    base_test = data_utils.get_data(f"{args.dataset}_val", None)
    max_test = int(getattr(args, "max_test_images", 0) or 0)
    test_total = min(len(base_test), max_test) if max_test > 0 else len(base_test)
    test_dataset = TransformedSubset(base_test, list(range(test_total)), backbone.preprocess)
    return train_raw, val_raw, train_dataset, val_dataset, test_dataset, backbone, clip_model, clip_preprocess


def _concept_cache_base(args, split_name: str, concepts: Sequence[str]) -> str:
    concept_hash = hashlib.sha1("\n".join(concepts).encode("utf-8")).hexdigest()[:16]
    clip_name = (getattr(args, "lf_clip_name", None) or args.backbone).replace("/", "_").replace("-", "_")
    source = getattr(args, "spatial_source", "prompt_grid")
    radius_tag = f"_r{int(getattr(args, 'prompt_radius', 3))}" if source == "prompt_grid" else ""
    stem = (
        f"{args.dataset}_{split_name}_salf_{args.backbone}_clip_{clip_name}_"
        f"{source}_gh{int(args.grid_h)}_gw{int(args.grid_w)}{radius_tag}_{concept_hash}"
    )
    return os.path.join(args.activation_dir, stem)


def _save_cache_order(cache_base: str, dataset_len: int) -> None:
    with open(cache_base + "_meta.json", "w") as f:
        json.dump({"dataset_len": dataset_len}, f, indent=2)


def _load_cached_spatial_sims(cache_base: str, dataset_len: int, force_recompute: bool):
    cache_path = cache_base + "_P.pt"
    meta_path = cache_base + "_meta.json"
    if not os.path.exists(cache_path) or force_recompute:
        return None
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
        if int(meta.get("dataset_len", -1)) != int(dataset_len):
            raise RuntimeError(
                f"Cached SALF spatial sims length mismatch at {cache_path}: "
                f"{meta.get('dataset_len')} cached vs {dataset_len} current."
            )
    logger.info("Loading cached SALF spatial sims from {}", cache_path)
    return torch.load(cache_path, weights_only=False)


def compute_spatial_sims_prompt_grid(
    args,
    raw_dataset: Dataset,
    clip_model,
    clip_preprocess,
    concepts: Sequence[str],
    split_name: str,
) -> torch.Tensor:
    cache_base = _concept_cache_base(args, split_name, concepts)
    os.makedirs(os.path.dirname(cache_base), exist_ok=True)
    cached = _load_cached_spatial_sims(
        cache_base, len(raw_dataset), getattr(args, "recompute_spatial_sims", False)
    )
    if cached is not None:
        return cached

    spatial_batch_size = int(getattr(args, "spatial_batch_size", 128) or 128)
    prompt_batch_size = int(getattr(args, "prompt_batch_size", 1024) or 1024)
    spatial_num_workers = int(
        getattr(args, "spatial_num_workers", args.num_workers) or args.num_workers
    )

    loader = DataLoader(
        raw_dataset,
        batch_size=spatial_batch_size,
        shuffle=False,
        num_workers=spatial_num_workers,
        collate_fn=pil_collate,
        persistent_workers=spatial_num_workers > 0,
    )
    tokens = clip.tokenize([str(concept) for concept in concepts]).to(args.device)
    with torch.no_grad():
        text_emb = clip_model.encode_text(tokens).float()
        text_emb = F.normalize(text_emb, dim=1)

    input_size = infer_clip_input_size(clip_preprocess)
    _, prompt_masks = _build_prompt_grid_metadata(
        image_size=input_size,
        grid_h=int(args.grid_h),
        grid_w=int(args.grid_w),
        radius=int(args.prompt_radius),
    )

    all_P: List[torch.Tensor] = []
    logger.info(
        "Computing SALF prompt-grid similarities for {} with spatial_batch_size={} prompt_batch_size={}",
        split_name,
        spatial_batch_size,
        prompt_batch_size,
    )
    for pil_images, _ in tqdm(loader, desc=f"SALF P {split_name}"):
        base_images = _prepare_base_clip_tensors(pil_images, clip_preprocess)
        prompted_images = _apply_prompt_masks_to_batch(
            base_images,
            prompt_masks,
            clip_preprocess,
        )

        sims_chunks = []
        for start in range(0, prompted_images.shape[0], prompt_batch_size):
            image_tensor = prompted_images[start : start + prompt_batch_size].to(args.device)
            with torch.no_grad():
                img_emb = clip_model.encode_image(image_tensor).float()
                img_emb = F.normalize(img_emb, dim=1)
                sim = img_emb @ text_emb.T
            sims_chunks.append(sim.cpu())

        sims = torch.cat(sims_chunks, dim=0)
        batch_size = len(pil_images)
        sim_maps = sims.view(batch_size, int(args.grid_h), int(args.grid_w), -1)
        all_P.append(sim_maps)

    P = torch.cat(all_P, dim=0)
    torch.save(P, cache_base + "_P.pt")
    _save_cache_order(cache_base, len(raw_dataset))
    logger.info("Saved SALF spatial similarities to {}", cache_base + "_P.pt")
    return P


def compute_clip_scores_from_P(P_train: torch.Tensor, score_mode: str, topk: int, quantile: float) -> torch.Tensor:
    flat = P_train.reshape(-1, P_train.shape[-1])
    if score_mode == "mean":
        return flat.mean(dim=0)
    if score_mode == "topk":
        k = min(max(1, int(topk)), int(flat.shape[0]))
        return torch.topk(flat, k=k, dim=0).values.mean(dim=0)
    if score_mode == "quantile":
        q = float(quantile)
        return torch.quantile(flat, q=q, dim=0)
    raise ValueError(f"Unsupported clip_score_mode={score_mode}")


def cbl_loss(pred_maps: torch.Tensor, target_maps: torch.Tensor) -> torch.Tensor:
    bsz, n_concepts, _, _ = pred_maps.shape
    pred = pred_maps.permute(0, 2, 3, 1).reshape(-1, n_concepts)
    tgt = target_maps.reshape(-1, n_concepts)
    return -cos_similarity_cubed(tgt, pred).mean()


def train_spatial_cbl(
    args,
    backbone: SpatialBackbone,
    concept_layer: nn.Module,
    train_loader: DataLoader,
    train_P: torch.Tensor,
    val_loader: DataLoader,
    val_P: torch.Tensor,
) -> nn.Module:
    backbone.eval()
    for parameter in backbone.parameters():
        parameter.requires_grad = False

    concept_layer.train()
    optimizer_name = str(getattr(args, "cbl_optimizer", "adam")).lower()
    weight_decay = float(getattr(args, "cbl_weight_decay", 0.0) or 0.0)
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            concept_layer.parameters(),
            lr=args.cbl_lr,
            weight_decay=weight_decay,
        )
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            concept_layer.parameters(),
            lr=args.cbl_lr,
            weight_decay=weight_decay,
        )
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            concept_layer.parameters(),
            lr=args.cbl_lr,
            weight_decay=weight_decay,
            momentum=float(getattr(args, "cbl_momentum", 0.9)),
        )
    else:
        raise ValueError(f"Unsupported SALF optimizer: {optimizer_name}")
    best_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    min_delta = float(getattr(args, "cbl_min_delta", 0.0) or 0.0)
    min_epochs = int(getattr(args, "cbl_min_epochs", 0) or 0)
    patience = int(getattr(args, "cbl_early_stop_patience", 0) or 0)

    for epoch in range(int(args.cbl_epochs)):
        running = 0.0
        for batch_idx, (images, _) in enumerate(
            tqdm(train_loader, desc=f"SALF CBL epoch {epoch + 1}")
        ):
            images = images.to(args.device)
            target = train_P[
                batch_idx * train_loader.batch_size : batch_idx * train_loader.batch_size
                + images.size(0)
            ].to(args.device)

            def compute_loss():
                with torch.no_grad():
                    feats = backbone(images)
                spatial_feats = feats["spatial"] if isinstance(feats, dict) else feats
                concept_maps = concept_layer(spatial_feats)
                concept_maps = F.interpolate(
                    concept_maps,
                    size=(int(args.grid_h), int(args.grid_w)),
                    mode="bilinear",
                    align_corners=False,
                )
                return cbl_loss(concept_maps, target)

            loss = compute_loss()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * images.size(0)

        train_loss = running / max(len(train_loader.dataset), 1)
        concept_layer.eval()
        with torch.no_grad():
            val_running = 0.0
            for batch_idx, (images, _) in enumerate(val_loader):
                images = images.to(args.device)
                feats = backbone(images)
                spatial_feats = feats["spatial"] if isinstance(feats, dict) else feats
                concept_maps = concept_layer(spatial_feats)
                concept_maps = F.interpolate(
                    concept_maps,
                    size=(int(args.grid_h), int(args.grid_w)),
                    mode="bilinear",
                    align_corners=False,
                )
                target = val_P[
                    batch_idx * val_loader.batch_size : batch_idx * val_loader.batch_size
                    + images.size(0)
                ].to(args.device)
                val_running += float(cbl_loss(concept_maps, target).item()) * images.size(0)
            val_loss = val_running / max(len(val_loader.dataset), 1)
        concept_layer.train()
        if val_loss < best_loss - min_delta:
            best_loss = val_loss
            epochs_without_improvement = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in concept_layer.state_dict().items()
            }
        else:
            epochs_without_improvement += 1
        logger.info(
            "[SALF CBL] epoch={} train_loss={:.6f} val_loss={:.6f} best_val={:.6f} stale_epochs={}",
            epoch,
            train_loss,
            val_loss,
            best_loss,
            epochs_without_improvement,
        )
        if patience > 0 and epoch + 1 >= min_epochs and epochs_without_improvement >= patience:
            logger.info(
                "[SALF CBL] early stopping at epoch={} after {} stale epochs",
                epoch,
                epochs_without_improvement,
            )
            break

    if best_state is not None:
        concept_layer.load_state_dict(best_state, strict=True)
    return concept_layer


def extract_global_concepts(
    args,
    backbone: SpatialBackbone,
    concept_layer: nn.Module,
    loader: DataLoader,
) -> Tuple[torch.Tensor, torch.Tensor]:
    backbone.eval()
    concept_layer.eval()
    concept_features = []
    labels = []
    with torch.no_grad():
        for images, target in tqdm(loader, desc="SALF concept extraction"):
            images = images.to(args.device)
            feats = backbone(images)
            spatial_feats = feats["spatial"] if isinstance(feats, dict) else feats
            maps = concept_layer(spatial_feats)
            pooled = pool_salf_maps(args, maps)
            concept_features.append(pooled.cpu())
            labels.append(target)
    return torch.cat(concept_features, dim=0), torch.cat(labels, dim=0)


def pool_salf_maps(args, maps: torch.Tensor) -> torch.Tensor:
    """Pool SALF spatial maps into one concept score per concept."""
    if maps.ndim <= 2:
        return maps
    mode = str(getattr(args, "salf_pool_mode", "avg") or "avg").lower()
    if mode in {"avg", "average", "mean"}:
        return F.adaptive_avg_pool2d(maps, 1).flatten(1)
    if mode == "max":
        return maps.flatten(2).max(dim=2).values
    if mode == "softmax":
        flat = maps.flatten(2)
        weights = F.softmax(flat, dim=2)
        return (flat * weights).sum(dim=2)
    raise ValueError(f"Unsupported salf_pool_mode={mode!r}; expected avg, max, or softmax.")


def evaluate_salf_accuracy(
    args,
    backbone: SpatialBackbone,
    concept_layer: nn.Module,
    mean: torch.Tensor,
    std: torch.Tensor,
    final_layer: nn.Module,
    dataset: Dataset,
) -> float:
    loader = DataLoader(
        dataset,
        batch_size=args.cbl_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    correct = 0
    total = 0
    with torch.no_grad():
        for images, target in tqdm(loader, desc="SALF eval", leave=False):
            images = images.to(args.device)
            feats = backbone(images)
            spatial_feats = feats["spatial"] if isinstance(feats, dict) else feats
            maps = concept_layer(spatial_feats)
            pooled = pool_salf_maps(args, maps)
            pooled = (pooled - mean.to(args.device)) / std.to(args.device)
            pred = final_layer(pooled).argmax(dim=-1).cpu()
            correct += int((pred == target).sum().item())
            total += int(target.numel())
    return correct / max(total, 1)


@dataclass
class SALFArtifacts:
    run_dir: str
    cache_train_path: str
    cache_val_path: str


def train_salf_cbm(args):
    if getattr(args, "spatial_source", "prompt_grid") != "prompt_grid":
        raise NotImplementedError(
            "First-pass SALF port currently supports only spatial_source=prompt_grid."
        )

    save_dir = build_run_dir(args.save_dir, args.dataset, args.model_name)
    logger.add(
        os.path.join(save_dir, "train.log"),
        format="{time} {level} {message}",
        level="DEBUG",
    )
    logger.info("Saving SALF-CBM model to {}", save_dir)
    save_args(args, save_dir)

    classes = data_utils.get_classes(args.dataset)
    raw_concepts = get_lf_concepts(args)
    (
        train_raw,
        val_raw,
        train_dataset,
        val_dataset,
        test_dataset,
        backbone,
        clip_model,
        clip_preprocess,
    ) = create_salf_splits(args)

    if use_partimagenetpp_train_val_split(args):
        full_train_raw = RawSubset(
            train_raw.base_dataset,
            list(range(len(train_raw.indices) + len(val_raw.indices))),
        )
        P_full_train = compute_spatial_sims_prompt_grid(
            args, full_train_raw, clip_model, clip_preprocess, raw_concepts, "train"
        )
        P_train = P_full_train.index_select(
            0, torch.tensor(train_raw.indices, dtype=torch.long)
        )
        P_val = P_full_train.index_select(
            0, torch.tensor(val_raw.indices, dtype=torch.long)
        )
    else:
        P_train = compute_spatial_sims_prompt_grid(
            args, train_raw, clip_model, clip_preprocess, raw_concepts, "train"
        )
        P_val = compute_spatial_sims_prompt_grid(
            args, val_raw, clip_model, clip_preprocess, raw_concepts, "val"
        )

    if getattr(args, "clip_cutoff", None) is not None:
        score_mode = getattr(args, "clip_score_mode", "topk")
        topk = int(getattr(args, "clip_topk", 500))
        quantile = float(getattr(args, "clip_quantile", 0.995))
        train_scores = compute_clip_scores_from_P(P_train, score_mode, topk, quantile)
        keep_mask = train_scores >= float(args.clip_cutoff)
        if int(keep_mask.sum().item()) == 0:
            raise RuntimeError("All SALF concepts were removed by clip_cutoff.")
        P_train = P_train[..., keep_mask]
        P_val = P_val[..., keep_mask]
        concepts = [raw_concepts[i] for i in range(len(raw_concepts)) if keep_mask[i]]
        logger.info(
            "SALF concept filtering kept {}/{} concepts with clip_cutoff={}",
            len(concepts),
            len(raw_concepts),
            args.clip_cutoff,
        )
    else:
        concepts = list(raw_concepts)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.cbl_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.cbl_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    is_vit = getattr(backbone, "is_vit", False)
    concept_layer = build_spatial_concept_layer(args, backbone.output_dim, len(concepts), is_vit=is_vit)
    concept_layer = train_spatial_cbl(
        args, backbone, concept_layer, train_loader, P_train, val_loader, P_val
    )
    torch.save(
        concept_layer.state_dict(), os.path.join(save_dir, "concept_layer_cbl.pt")
    )
    with open(os.path.join(save_dir, "concepts_cbl.txt"), "w") as f:
        f.write("\n".join(concepts))
    logger.info("Saved SALF concept-head checkpoint before sparse-final training")

    train_concepts, train_labels = extract_global_concepts(args, backbone, concept_layer, train_loader)
    val_concepts, val_labels = extract_global_concepts(args, backbone, concept_layer, val_loader)

    train_mean = train_concepts.mean(dim=0, keepdim=True)
    train_std = torch.clamp(train_concepts.std(dim=0, keepdim=True), min=1e-6)
    train_concepts = (train_concepts - train_mean) / train_std
    val_concepts = (val_concepts - train_mean) / train_std

    train_final_loader = DataLoader(
        IndexedTensorDataset(train_concepts, train_labels),
        batch_size=args.saga_batch_size,
        shuffle=True,
    )
    val_final_loader = DataLoader(
        TensorDataset(val_concepts, val_labels),
        batch_size=args.saga_batch_size,
        shuffle=False,
    )
    final_layer = nn.Linear(len(concepts), len(classes)).to(args.device)
    final_layer.weight.data.zero_()
    final_layer.bias.data.zero_()
    if args.dense:
        output_proj = train_dense_final(
            final_layer,
            train_final_loader,
            val_final_loader,
            args.saga_n_iters,
            args.dense_lr,
            device=args.device,
        )
    else:
        output_proj = train_sparse_final(
            final_layer,
            train_final_loader,
            val_final_loader,
            args.saga_n_iters,
            args.saga_lam,
            step_size=args.saga_step_size,
            device=args.device,
            path_steps=getattr(args, "saga_path_steps", 1),
            min_lam_ratio=getattr(args, "saga_min_lam_ratio", 1.0),
        )

    selected_final = output_proj["best"]
    W_g = selected_final["weight"]
    b_g = selected_final["bias"]
    final_layer.load_state_dict({"weight": W_g, "bias": b_g})
    logger.info(
        "Selected SALF sparse final head: lambda={} val_loss={} val_acc={}",
        float(selected_final["lam"]),
        selected_final["metrics"].get("loss_val"),
        selected_final["metrics"].get("acc_val"),
    )

    if getattr(args, "skip_train_val_eval", False):
        train_accuracy = None
        val_accuracy = None
    else:
        train_accuracy = evaluate_salf_accuracy(
            args, backbone, concept_layer, train_mean, train_std, final_layer, train_dataset
        )
        val_accuracy = evaluate_salf_accuracy(
            args, backbone, concept_layer, train_mean, train_std, final_layer, val_dataset
        )
    if getattr(args, "skip_test_eval", False):
        test_accuracy = None
    else:
        test_accuracy = evaluate_salf_accuracy(
            args, backbone, concept_layer, train_mean, train_std, final_layer, test_dataset
        )

    with open(os.path.join(save_dir, "concepts.txt"), "w") as f:
        f.write("\n".join(concepts))
    torch.save(concept_layer.state_dict(), os.path.join(save_dir, "concept_layer.pt"))
    torch.save(W_g, os.path.join(save_dir, "W_g.pt"))
    torch.save(b_g, os.path.join(save_dir, "b_g.pt"))
    torch.save(train_mean, os.path.join(save_dir, "proj_mean.pt"))
    torch.save(train_std, os.path.join(save_dir, "proj_std.pt"))

    train_metrics = {"accuracy": train_accuracy}
    val_metrics = {"accuracy": val_accuracy}
    test_metrics = {"accuracy": test_accuracy}
    for filename, payload in (
        ("train_metrics.json", train_metrics),
        ("val_metrics.json", val_metrics),
        ("test_metrics.json", test_metrics),
    ):
        with open(os.path.join(save_dir, filename), "w") as f:
            json.dump(payload, f, indent=2)

    path0 = selected_final
    metrics_payload = {
        key: float(path0[key]) for key in ("lam", "lr", "alpha", "time")
    }
    metrics_payload["metrics"] = path0["metrics"]
    nnz = int((W_g.abs() > 1e-5).sum().item())
    total = int(W_g.numel())
    metrics_payload["sparsity"] = {
        "Non-zero weights": nnz,
        "Total weights": total,
        "Percentage non-zero": nnz / max(total, 1),
    }
    with open(os.path.join(save_dir, "metrics.txt"), "w") as f:
        json.dump(metrics_payload, f, indent=2)

    method_log = {
        "cbm_variant": "salf_cbm",
        "protocol": (
            "lf_matched_general_domain" if use_original_label_free_protocol(args) else "unified_adaptation"
        ),
        "spatial_source": getattr(args, "spatial_source", "prompt_grid"),
        "clip_name": (getattr(args, "lf_clip_name", None) or args.backbone),
        "grid_h": int(args.grid_h),
        "grid_w": int(args.grid_w),
        "prompt_radius": int(getattr(args, "prompt_radius", 3)),
        "concept_filters": {
            "clip_cutoff": getattr(args, "clip_cutoff", None),
            "clip_score_mode": getattr(args, "clip_score_mode", "topk"),
        },
        "concept_bottleneck_layer": {
            "type": args.cbl_type,
            "hidden_layers": args.cbl_hidden_layers if args.cbl_type == "mlp" else 0,
            "use_batchnorm": bool(args.cbl_use_batchnorm) if args.cbl_type == "mlp" else False,
            "bias": bool(getattr(args, "salf_cbl_bias", False)) if args.cbl_type == "linear" else None,
            "optimizer": str(getattr(args, "cbl_optimizer", "adam")),
        },
        "sparse_final_layer": {
            "solver": "glm_saga",
            "lam": args.saga_lam,
            "path_steps": int(getattr(args, "saga_path_steps", 1)),
            "min_lam_ratio": float(getattr(args, "saga_min_lam_ratio", 1.0)),
            "selected_lam": float(selected_final["lam"]),
            "saga_iters": args.saga_n_iters,
            "saga_batch_size": args.saga_batch_size,
        },
        "salf_pool_mode": getattr(args, "salf_pool_mode", "avg"),
        "spatial_cache_paths": {
            "train": _concept_cache_base(args, "train", concepts) + "_P.pt",
            "val": _concept_cache_base(args, "val", concepts) + "_P.pt",
        },
    }
    with open(os.path.join(save_dir, "method_log.json"), "w") as f:
        json.dump(method_log, f, indent=2)

    write_artifacts(
        save_dir,
        {
            "model_name": args.model_name,
            "dataset": args.dataset,
            "backbone": args.backbone,
            "concept_layer_format": "concept_layer.pt",
            "normalization_format": ["proj_mean.pt", "proj_std.pt"],
            "final_layer_format": ["W_g.pt", "b_g.pt"],
            "spatial_cache_format": ["*_P.pt", "*_meta.json"],
            "sparse_eval_style": "not_yet_supported",
        },
    )
    if test_accuracy is None:
        logger.info("SALF-CBM test accuracy skipped")
    elif train_accuracy is None or val_accuracy is None:
        logger.info("SALF-CBM test accuracy={:.4f} (train/val eval skipped)", test_accuracy)
    else:
        logger.info(
            "SALF-CBM train accuracy={:.4f} val accuracy={:.4f} test accuracy={:.4f}",
            train_accuracy,
            val_accuracy,
            test_accuracy,
        )
    return save_dir
