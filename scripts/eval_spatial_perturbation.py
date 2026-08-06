#!/usr/bin/env python3
"""Evaluate whether SG-CBM spatial maps are faithful under region perturbations."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data import utils as data_utils
from methods.salf import SpatialBackbone, build_spatial_concept_layer, pool_salf_maps
from methods.savlg import (
    build_savlg_concept_layer,
    compute_savlg_concept_logits,
    forward_savlg_backbone,
    forward_savlg_concept_layer,
)
from model.cbm import Backbone, BackboneCLIP, ConceptLayer

try:
    from gcbm.imagenet_eval import load_run_config
    from gcbm.imagenet_models import build_model as build_imagenet_model
    from gcbm.imagenet_targets import load_concepts as load_imagenet_concepts
    from gcbm.runtime import configure_runtime
    from gcbm.training_utils import prepare_images as prepare_imagenet_images
except Exception:  # pragma: no cover - optional outside ImageNet SG-CBM runs.
    load_run_config = None
    build_imagenet_model = None
    load_imagenet_concepts = None
    configure_runtime = None
    prepare_imagenet_images = None


SAVLG_DEFAULTS = {
    "backbone": "resnet18_cub",
    "backbone_checkpoint": "",
    "feature_layer": "layer4",
    "savlg_branch_arch": "dual",
    "savlg_global_head_mode": "vlg_linear",
    "savlg_global_hidden_dim": 0,
    "savlg_global_hidden_layers": 0,
    "savlg_global_use_batchnorm": False,
    "savlg_pooling": "avg",
    "savlg_residual_spatial_alpha": 0.1,
    "savlg_residual_spatial_pooling": "lse",
    "savlg_spatial_branch_mode": "multiscale_conv45",
    "savlg_spatial_stage": "conv5",
    "savlg_topk_fraction": 0.2,
    "savlg_mil_temperature": 1.0,
}


def parse_fractions(raw: str) -> List[float]:
    values = [float(item) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("At least one deletion fraction is required.")
    for value in values:
        if not 0.0 < value < 1.0:
            raise ValueError(f"Invalid fraction {value}; expected value in (0, 1).")
    return values


def load_run_args(artifact_dir: Path, device: str) -> SimpleNamespace:
    args_path = artifact_dir / "args.txt"
    if not args_path.is_file():
        raise FileNotFoundError(f"Missing saved args file: {args_path}")
    payload = json.loads(args_path.read_text())
    for key, value in SAVLG_DEFAULTS.items():
        payload.setdefault(key, value)
    payload["device"] = device
    return SimpleNamespace(**payload)


def load_lines(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def find_final_layer_normalization(artifact_dir: Path) -> Path:
    norm_path = artifact_dir / "final_layer_normalization.pt"
    if norm_path.is_file():
        return norm_path
    candidates = sorted(
        artifact_dir.glob("*/final_layer_normalization.pt"),
        key=lambda path: str(path),
    )
    if candidates:
        return candidates[0]
    candidates = sorted(
        artifact_dir.glob("**/final_layer_normalization.pt"),
        key=lambda path: (len(path.relative_to(artifact_dir).parts), str(path)),
    )
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"Missing final_layer_normalization.pt under {artifact_dir}")


def infer_unified_dataset_name(cfg, artifact_dir: Path) -> str:
    explicit = getattr(cfg, "dataset", "")
    if explicit:
        return explicit
    haystack = " ".join(
        str(getattr(cfg, key, ""))
        for key in ("train_root", "train_manifest", "val_manifest", "annotation_dir", "concept_file", "val_root", "save_dir", "run_name", "precomputed_target_dir")
    )
    haystack = f"{artifact_dir} {haystack}".lower()
    for name in ("places365", "partimagenetpp", "cub", "imagenet"):
        if name in haystack:
            return name
    return "imagenet"


def load_imagenet_val_targets(devkit_dir: Path, label_order: str) -> List[int]:
    label_path = devkit_dir / "data" / "ILSVRC2012_validation_ground_truth.txt"
    if label_order == "ilsvrc_id":
        labels: List[int] = []
        with label_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    labels.append(int(line) - 1)
        if len(labels) != 50000:
            raise RuntimeError(f"expected 50000 ImageNet validation labels, got {len(labels)}")
        return labels

    meta_path = devkit_dir / "data" / "meta.mat"
    if meta_path.is_file():
        from scipy.io import loadmat

        payload = loadmat(meta_path, squeeze_me=True, struct_as_record=False)
        synsets = payload["synsets"]
        id_to_wnid: Dict[int, str] = {}
        for syn in synsets:
            ilsvrc_id = int(syn.ILSVRC2012_ID)
            if 1 <= ilsvrc_id <= 1000 and int(syn.num_children) == 0:
                id_to_wnid[ilsvrc_id] = str(syn.WNID)
        if len(id_to_wnid) != 1000:
            raise RuntimeError(f"expected 1000 ImageNet leaf synsets, got {len(id_to_wnid)}")
        wnids = sorted(id_to_wnid.values())
        class_to_idx = {wnid: idx for idx, wnid in enumerate(wnids)}
        labels: List[int] = []
        with label_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    raw_id = int(line)
                    labels.append(class_to_idx[id_to_wnid[raw_id]])
        if len(labels) != 50000:
            raise RuntimeError(f"expected 50000 ImageNet validation labels, got {len(labels)}")
        return labels

    valprep_path = devkit_dir / "data" / "valprep.sh"
    if valprep_path.is_file():
        import re

        val_to_wnid: Dict[int, str] = {}
        pattern = re.compile(r"ILSVRC2012_val_(\d{8})\.JPEG\s+([a-z]\d{8})")
        for line in valprep_path.read_text().splitlines():
            match = pattern.search(line)
            if match is None:
                continue
            val_to_wnid[int(match.group(1)) - 1] = match.group(2)
        if len(val_to_wnid) != 50000:
            raise RuntimeError(f"expected 50000 valprep entries, got {len(val_to_wnid)}")
        wnids = sorted(set(val_to_wnid.values()))
        if len(wnids) != 1000:
            raise RuntimeError(f"expected 1000 valprep WNIDs, got {len(wnids)}")
        class_to_idx = {wnid: idx for idx, wnid in enumerate(wnids)}
        return [class_to_idx[val_to_wnid[idx]] for idx in range(50000)]

    raise FileNotFoundError(
        f"Need {meta_path} or {valprep_path} to build sorted-WNID ImageNet labels."
    )


def resolve_imagenet_label_order(model_name: str, label_order: str) -> str:
    if label_order != "auto":
        return label_order
    return "sorted_wnid"


def imagenet_val_index(path: Path) -> int | None:
    stem = path.stem
    prefix = "ILSVRC2012_val_"
    if not stem.startswith(prefix):
        return None
    try:
        return int(stem[len(prefix) :]) - 1
    except ValueError:
        return None


class FlatImageNetValDataset(Dataset):
    def __init__(self, root: Path, transform, labels: List[int] | None = None) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"Missing ImageNet val root: {self.root}")
        self.paths = sorted(self.root.glob("*.JPEG"))
        if not self.paths:
            self.paths = sorted(self.root.rglob("*.JPEG"))
        if not self.paths:
            raise FileNotFoundError(f"No JPEG files found under ImageNet val root: {self.root}")
        self.transform = transform
        self.labels = labels

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        with Image.open(path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        label = -1
        if self.labels is not None:
            val_idx = imagenet_val_index(path)
            if val_idx is not None and 0 <= val_idx < len(self.labels):
                label = self.labels[val_idx]
        return image, torch.tensor(label, dtype=torch.long)


def load_sgcbm(artifact_dir: Path, device: str):
    run_args = load_run_args(artifact_dir, device)
    concepts = load_lines(artifact_dir / "concepts.txt")
    classes = data_utils.get_classes(run_args.dataset)

    backbone = SpatialBackbone(
        run_args.backbone,
        device=device,
        spatial_stage=getattr(run_args, "savlg_spatial_stage", "conv5"),
        checkpoint_path=getattr(run_args, "backbone_checkpoint", ""),
    )
    concept_layer = build_savlg_concept_layer(run_args, backbone, len(concepts))
    state = torch.load(artifact_dir / "concept_layer.pt", map_location=device)
    missing, unexpected = concept_layer.load_state_dict(state, strict=False)
    unexpected = [key for key in unexpected if key != "spatial_layer.bias"]
    if missing or unexpected:
        raise RuntimeError(f"Incompatible SG-CBM concept layer: missing={missing}, unexpected={unexpected}")
    concept_layer.eval()
    backbone.eval()

    final_layer = nn.Linear(len(concepts), len(classes)).to(device)
    final_layer.weight.data.copy_(torch.load(artifact_dir / "W_g.pt", map_location=device))
    final_layer.bias.data.copy_(torch.load(artifact_dir / "b_g.pt", map_location=device))
    final_layer.eval()

    mean = torch.load(artifact_dir / "proj_mean.pt", map_location=device).flatten().to(device)
    std = torch.load(artifact_dir / "proj_std.pt", map_location=device).flatten().to(device)
    std = torch.where(std.abs() < 1e-6, torch.ones_like(std), std)
    return run_args, concepts, backbone, concept_layer, final_layer, mean, std


def load_imagenet_sgcbm(
    artifact_dir: Path,
    device: str,
    batch_size: int,
    workers: int,
    prefetch_factor: int,
    persistent_workers: bool,
    pin_memory: bool,
):
    if load_run_config is None or build_imagenet_model is None:
        raise ImportError("ImageNet SG-CBM dependencies are unavailable.")
    cfg_args = argparse.Namespace(
        device=device,
        batch_size=batch_size,
        workers=workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
    )
    cfg = load_run_config(artifact_dir, cfg_args)
    configure_runtime(cfg)
    dataset_name = infer_unified_dataset_name(cfg, artifact_dir)
    cfg.dataset = dataset_name
    concepts_path = artifact_dir / "concepts.txt"
    if dataset_name == "imagenet" and load_imagenet_concepts is not None:
        concepts = load_imagenet_concepts(str(concepts_path))
    else:
        concepts = load_lines(concepts_path)
    classes = data_utils.get_classes(dataset_name)
    backbone, concept_layer = build_imagenet_model(cfg, n_concepts=len(concepts))
    state_path = artifact_dir / "concept_head_best.pt"
    if not state_path.is_file():
        state_path = artifact_dir / "concept_head_latest.pt"
    concept_layer.load_state_dict(torch.load(state_path, map_location=device))
    concept_layer.eval()
    backbone.eval()

    final_layer = nn.Linear(len(concepts), len(classes)).to(device)
    final_layer.weight.data.zero_()
    final_layer.bias.data.zero_()
    final_layer.eval()

    norm_path = find_final_layer_normalization(artifact_dir)
    payload = torch.load(norm_path, map_location=device)
    mean = payload["mean"].flatten().to(device)
    std = payload["std"].flatten().to(device)
    std = torch.where(std.abs() < 1e-6, torch.ones_like(std), std)
    return cfg, concepts, backbone, concept_layer, final_layer, mean, std


@torch.no_grad()
def forward_sgcbm(images, run_args, backbone, concept_layer, final_layer, mean, std):
    feats = forward_savlg_backbone(backbone, images, run_args)
    global_outputs, spatial_maps = forward_savlg_concept_layer(concept_layer, feats)
    _, _, concept_logits = compute_savlg_concept_logits(
        global_outputs,
        spatial_maps,
        run_args,
        concept_layer=concept_layer,
    )
    class_logits = final_layer((concept_logits - mean) / std)
    return concept_logits, spatial_maps, class_logits


@torch.no_grad()
def forward_imagenet_sgcbm(images, run_args, backbone, concept_layer, final_layer, mean, std):
    images = prepare_imagenet_images(images, run_args)
    outputs = concept_layer(backbone(images))
    concept_logits = outputs["final_logits"]
    spatial_maps = outputs["spatial_maps"]
    class_logits = final_layer((concept_logits - mean) / std)
    return concept_logits, spatial_maps, class_logits


def _load_stat(artifact_dir: Path, names: Tuple[str, ...], device: str) -> torch.Tensor:
    for name in names:
        path = artifact_dir / name
        if path.is_file():
            return torch.load(path, map_location=device).flatten().to(device)
    joined = ", ".join(names)
    raise FileNotFoundError(f"Missing projection statistic in {artifact_dir}; tried {joined}")


def load_vlg(artifact_dir: Path, device: str):
    run_args = load_run_args(artifact_dir, device)
    concepts = load_lines(artifact_dir / "concepts.txt")
    classes = data_utils.get_classes(run_args.dataset)

    if str(run_args.backbone).startswith("clip_"):
        backbone = BackboneCLIP(
            run_args.backbone,
            use_penultimate=bool(getattr(run_args, "use_clip_penultimate", False)),
            device=device,
        )
    else:
        feature_layer = str(getattr(run_args, "feature_layer", ""))
        backbone = Backbone(run_args.backbone, feature_layer, device)
        cam_layer = data_utils.BACKBONE_VISUALIZATION_TARGET_LAYER.get(str(run_args.backbone))
        if cam_layer and cam_layer != feature_layer:
            backbone.vlg_cam_feature_vals = {}

            def cam_hook(module, input, output):
                backbone.vlg_cam_feature_vals[output.device] = output

            backbone.backbone.get_submodule(cam_layer).register_forward_hook(cam_hook)

    if (artifact_dir / "cbl.pt").is_file():
        concept_layer = ConceptLayer.from_pretrained(str(artifact_dir), device)
    elif (artifact_dir / "W_c.pt").is_file():
        weight = torch.load(artifact_dir / "W_c.pt", map_location=device).float()
        concept_layer = nn.Linear(int(weight.shape[1]), int(weight.shape[0]), bias=False).to(device)
        concept_layer.load_state_dict({"weight": weight})
    else:
        raise FileNotFoundError(f"Missing VLG concept layer in {artifact_dir}; expected cbl.pt or W_c.pt")

    final_layer = nn.Linear(len(concepts), len(classes)).to(device)
    if (artifact_dir / "final.pt").is_file():
        final_state = torch.load(artifact_dir / "final.pt", map_location=device)
        final_layer.load_state_dict(final_state)
    elif (artifact_dir / "W_g.pt").is_file() and (artifact_dir / "b_g.pt").is_file():
        final_layer.weight.data.copy_(torch.load(artifact_dir / "W_g.pt", map_location=device))
        final_layer.bias.data.copy_(torch.load(artifact_dir / "b_g.pt", map_location=device))
    else:
        final_layer.weight.data.zero_()
        final_layer.bias.data.zero_()
    final_layer.eval()
    concept_layer.eval()
    backbone.eval()

    mean = _load_stat(artifact_dir, ("train_concept_features_mean.pt", "proj_mean.pt"), device)
    std = _load_stat(artifact_dir, ("train_concept_features_std.pt", "proj_std.pt"), device)
    std = torch.where(std.abs() < 1e-6, torch.ones_like(std), std)
    return run_args, concepts, backbone, concept_layer, final_layer, mean, std


class OfficialSalfLayer(nn.Module):
    def __init__(self, weight: torch.Tensor, map_size: Tuple[int, int], pooling: str) -> None:
        super().__init__()
        if weight.ndim == 2:
            weight = weight[:, :, None, None]
        if weight.ndim != 4:
            raise ValueError(f"expected SALF W_c to have 2 or 4 dims, got {tuple(weight.shape)}")
        self.register_buffer("weight", weight.float())
        self.map_size = tuple(map_size)
        self.pooling = pooling

    def forward(self, feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if feats.ndim == 4:
            if tuple(feats.shape[-2:]) != self.map_size:
                feats = F.interpolate(feats, size=self.map_size, mode="bilinear", align_corners=False)
            maps = F.conv2d(feats, self.weight)
            if self.pooling == "softmax":
                n, c, h, w = maps.shape
                patches = maps.view(n, c, h * w)
                weights = F.softmax(patches, dim=2)
                concepts = (patches * weights).sum(dim=2)
            else:
                concepts = F.adaptive_avg_pool2d(maps, 1).flatten(1)
            return concepts, maps
        concepts = F.linear(torch.flatten(feats, 1), self.weight.squeeze(-1).squeeze(-1))
        maps = concepts[:, :, None, None]
        return concepts, maps


class CallableBackbone(nn.Module):
    def __init__(self, forward_fn, preprocess) -> None:
        super().__init__()
        self.forward_fn = forward_fn
        self.preprocess = preprocess

    def forward(self, images: torch.Tensor):
        return self.forward_fn(images)


def parse_hw(raw: str) -> Tuple[int, int]:
    parts = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if len(parts) != 2:
        raise ValueError(f"Expected H,W map size, got {raw!r}")
    return parts[0], parts[1]


def load_salf_imagenet_official(
    artifact_dir: Path,
    device: str,
    backbone_name: str,
    feature_layer: str,
    map_size: str,
    pooling: str,
):
    from scripts.eval_salf_imagenet_nec_tar import build_backbone

    run_args = SimpleNamespace(dataset="imagenet")
    concepts = load_lines(artifact_dir / "concepts.txt")
    classes = data_utils.get_classes("imagenet")
    backbone_fn, preprocess = build_backbone(backbone_name, feature_layer, device)
    backbone = CallableBackbone(backbone_fn, preprocess).to(device).eval()
    concept_layer = OfficialSalfLayer(
        torch.load(artifact_dir / "W_c.pt", map_location=device),
        map_size=parse_hw(map_size),
        pooling=pooling,
    ).to(device).eval()

    final_layer = nn.Linear(len(concepts), len(classes)).to(device)
    final_layer.weight.data.copy_(torch.load(artifact_dir / "W_g.pt", map_location=device))
    final_layer.bias.data.copy_(torch.load(artifact_dir / "b_g.pt", map_location=device))
    final_layer.eval()

    mean = torch.load(artifact_dir / "proj_mean.pt", map_location=device).flatten().to(device)
    std = torch.load(artifact_dir / "proj_std.pt", map_location=device).flatten().to(device)
    std = torch.where(std.abs() < 1e-6, torch.ones_like(std), std)
    return run_args, concepts, backbone, concept_layer, final_layer, mean, std


@torch.no_grad()
def forward_salf_imagenet_official(images, run_args, backbone, concept_layer, final_layer, mean, std):
    concept_logits, spatial_maps = concept_layer(backbone(images))
    class_logits = final_layer((concept_logits - mean) / std)
    return concept_logits, spatial_maps, class_logits


@torch.no_grad()
def forward_vlg(images, run_args, backbone, concept_layer, final_layer, mean, std):
    concept_logits = concept_layer(backbone(images))
    feature_map = getattr(backbone, "vlg_cam_feature_vals", backbone.feature_vals)[
        concept_logits.device
    ]
    linear = concept_layer.model[0] if hasattr(concept_layer, "model") else concept_layer
    spatial_maps = F.conv2d(feature_map, linear.weight[:, :, None, None], bias=None)
    spatial_maps = F.relu(spatial_maps)
    class_logits = final_layer((concept_logits - mean) / std)
    return concept_logits, spatial_maps, class_logits


def compute_vlg_gradcam_maps(images, backbone, concept_layer, final_layer, mean, std):
    backbone.zero_grad(set_to_none=True)
    concept_layer.zero_grad(set_to_none=True)
    final_layer.zero_grad(set_to_none=True)
    with torch.enable_grad():
        concept_logits = concept_layer(backbone(images))
        feature_map = getattr(backbone, "vlg_cam_feature_vals", backbone.feature_vals)[
            concept_logits.device
        ]
        feature_map.retain_grad()
        class_logits = final_layer((concept_logits - mean) / std)
        pred_class = class_logits.argmax(dim=1)
        score = class_logits[torch.arange(images.shape[0], device=images.device), pred_class].sum()
        score.backward()
        grads = feature_map.grad
        if grads is None:
            raise RuntimeError("Failed to compute Grad-CAM gradients for VLG-CBM feature map.")
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cams = F.relu((weights * feature_map).sum(dim=1)).detach()
    backbone.zero_grad(set_to_none=True)
    concept_layer.zero_grad(set_to_none=True)
    final_layer.zero_grad(set_to_none=True)
    return cams


def load_salf(artifact_dir: Path, device: str):
    run_args = load_run_args(artifact_dir, device)
    for key, value in {
        "grid_h": 7,
        "grid_w": 7,
        "salf_pool_mode": "avg",
        "cbl_type": "linear",
        "cbl_hidden_dim": 0,
        "cbl_hidden_layers": 0,
        "cbl_use_batchnorm": False,
    }.items():
        if not hasattr(run_args, key):
            setattr(run_args, key, value)
    concepts = load_lines(artifact_dir / "concepts.txt")
    classes = data_utils.get_classes(run_args.dataset)

    backbone = SpatialBackbone(
        run_args.backbone,
        device=device,
        spatial_stage=getattr(run_args, "savlg_spatial_stage", "conv5"),
        checkpoint_path=getattr(run_args, "backbone_checkpoint", ""),
    )
    state = torch.load(artifact_dir / "concept_layer.pt", map_location=device)
    if (
        not hasattr(run_args, "salf_cbl_bias")
        and any(key.endswith("bias") for key in state)
    ):
        run_args.salf_cbl_bias = True
    concept_layer = build_spatial_concept_layer(run_args, backbone.output_dim, len(concepts))
    missing, unexpected = concept_layer.load_state_dict(state, strict=False)
    unexpected = [
        key for key in unexpected if key not in {"global_layer.bias", "spatial_layer.bias"}
    ]
    if missing or unexpected:
        raise RuntimeError(f"Incompatible SALF-CBM concept layer: missing={missing}, unexpected={unexpected}")
    concept_layer.eval()
    backbone.eval()

    final_layer = nn.Linear(len(concepts), len(classes)).to(device)
    final_layer.weight.data.copy_(torch.load(artifact_dir / "W_g.pt", map_location=device))
    final_layer.bias.data.copy_(torch.load(artifact_dir / "b_g.pt", map_location=device))
    final_layer.eval()

    mean = torch.load(artifact_dir / "proj_mean.pt", map_location=device).flatten().to(device)
    std = torch.load(artifact_dir / "proj_std.pt", map_location=device).flatten().to(device)
    std = torch.where(std.abs() < 1e-6, torch.ones_like(std), std)
    return run_args, concepts, backbone, concept_layer, final_layer, mean, std


@torch.no_grad()
def forward_salf(images, run_args, backbone, concept_layer, final_layer, mean, std):
    feats = backbone(images)
    spatial_feats = feats["spatial"] if isinstance(feats, dict) else feats
    spatial_maps = concept_layer(spatial_feats)
    concept_logits = pool_salf_maps(run_args, spatial_maps)
    class_logits = final_layer((concept_logits - mean) / std)
    return concept_logits, spatial_maps, class_logits


def minmax_maps(maps: torch.Tensor) -> torch.Tensor:
    flat = maps.flatten(1)
    mins = flat.min(dim=1).values[:, None, None]
    maxs = flat.max(dim=1).values[:, None, None]
    return (maps - mins) / (maxs - mins + 1e-6)


def top_fraction_mask(maps: torch.Tensor, out_hw: Tuple[int, int], fraction: float) -> torch.Tensor:
    maps = F.interpolate(maps[:, None], size=out_hw, mode="bilinear", align_corners=False)[:, 0]
    maps = minmax_maps(maps)
    batch, height, width = maps.shape
    k = max(1, min(height * width, int(round(fraction * height * width))))
    flat = maps.flatten(1)
    threshold = flat.topk(k, dim=1).values[:, -1].view(batch, 1, 1)
    return (maps >= threshold).to(maps.dtype)


def top_patch_mask(
    maps: torch.Tensor,
    out_hw: Tuple[int, int],
    patch_size: int,
    top_blocks: int,
) -> torch.Tensor:
    maps = F.interpolate(maps[:, None], size=out_hw, mode="bilinear", align_corners=False)
    maps = minmax_maps(maps[:, 0])[:, None]
    batch, _, height, width = maps.shape
    patch_size = max(1, min(patch_size, height, width))
    grid_h = max(1, height // patch_size)
    grid_w = max(1, width // patch_size)
    pooled = F.avg_pool2d(maps, kernel_size=patch_size, stride=patch_size)
    pooled = pooled[:, :, :grid_h, :grid_w].flatten(1)
    top_blocks = max(1, min(top_blocks, pooled.shape[1]))
    indices = pooled.topk(top_blocks, dim=1).indices
    mask = torch.zeros((batch, height, width), device=maps.device, dtype=maps.dtype)
    for batch_idx in range(batch):
        for idx in indices[batch_idx].tolist():
            y = (idx // grid_w) * patch_size
            x = (idx % grid_w) * patch_size
            mask[batch_idx, y : y + patch_size, x : x + patch_size] = 1.0
    return mask


def random_patch_mask(
    batch: int,
    height: int,
    width: int,
    patch_size: int,
    top_blocks: int,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    patch_size = max(1, min(patch_size, height, width))
    grid_h = max(1, height // patch_size)
    grid_w = max(1, width // patch_size)
    total_blocks = grid_h * grid_w
    top_blocks = max(1, min(top_blocks, total_blocks))
    scores = torch.rand((batch, total_blocks), device=device, generator=generator)
    indices = scores.topk(top_blocks, dim=1).indices
    mask = torch.zeros((batch, height, width), device=device)
    for batch_idx in range(batch):
        for idx in indices[batch_idx].tolist():
            y = (idx // grid_w) * patch_size
            x = (idx % grid_w) * patch_size
            mask[batch_idx, y : y + patch_size, x : x + patch_size] = 1.0
    return mask


def random_fraction_mask(
    batch: int,
    height: int,
    width: int,
    fraction: float,
    device: torch.device,
    generator: torch.Generator,
    mode: str = "box",
) -> torch.Tensor:
    k = max(1, min(height * width, int(round(fraction * height * width))))
    mask = torch.zeros((batch, height * width), device=device)
    if mode == "pixel":
        scores = torch.rand((batch, height * width), device=device, generator=generator)
        idx = scores.topk(k, dim=1).indices
        mask.scatter_(1, idx, 1.0)
        return mask.view(batch, height, width)
    if mode != "box":
        raise ValueError(f"Unknown random mask mode: {mode}")

    box_h = max(1, min(height, int(round(math.sqrt(k)))))
    box_w = max(1, min(width, int(math.ceil(k / box_h))))
    max_y = max(0, height - box_h)
    max_x = max(0, width - box_w)
    ys = torch.randint(0, max_y + 1, (batch,), device=device, generator=generator)
    xs = torch.randint(0, max_x + 1, (batch,), device=device, generator=generator)
    mask_2d = mask.view(batch, height, width)
    for row, y, x in zip(mask_2d, ys.tolist(), xs.tolist()):
        row[y : y + box_h, x : x + box_w] = 1.0
    if box_h * box_w > k:
        flat = mask_2d.flatten(1)
        active = flat.nonzero(as_tuple=False)
        for batch_idx in range(batch):
            idx = active[active[:, 0] == batch_idx, 1]
            drop_n = idx.numel() - k
            if drop_n > 0:
                perm = torch.randperm(idx.numel(), device=device, generator=generator)[:drop_n]
                flat[batch_idx, idx[perm]] = 0.0
    return mask_2d


def apply_mask(images: torch.Tensor, mask: torch.Tensor, fill_value: float, insertion: bool) -> torch.Tensor:
    mask = mask[:, None]
    fill = torch.full_like(images, fill_value)
    if insertion:
        return images * mask + fill * (1.0 - mask)
    return images * (1.0 - mask) + fill * mask


def init_metrics(fractions: Iterable[float]) -> Dict[str, float]:
    out: Dict[str, float] = {
        "n": 0.0,
        "accuracy_before_sum": 0.0,
        "top_concept_logit_sum": 0.0,
    }
    for frac in fractions:
        tag = f"f{frac:g}"
        for name in (
            "class_logit_drop_selected",
            "class_prob_drop_selected",
            "concept_logit_drop_selected",
            "normalized_concept_logit_drop_selected",
            "accuracy_after_selected_sum",
            "class_logit_drop_random",
            "class_prob_drop_random",
            "concept_logit_drop_random",
            "normalized_concept_logit_drop_random",
            "accuracy_after_random_sum",
            "class_prob_insertion_selected",
            "class_prob_insertion_random",
        ):
            out[f"{tag}_{name}"] = 0.0
    return out


def choose_concepts(concept_logits, norm_concepts, class_logits, final_layer, mode: str):
    pred_class = class_logits.argmax(dim=1)
    if mode == "top_activation":
        concept_idx = norm_concepts.argmax(dim=1)
    elif mode == "top_class_contribution":
        weights = final_layer.weight[pred_class]
        contribution = weights * norm_concepts
        concept_idx = contribution.argmax(dim=1)
    elif mode == "top_active_class_contribution":
        weights = final_layer.weight[pred_class]
        contribution = weights * norm_concepts
        masked = contribution.masked_fill(norm_concepts <= 0, -torch.inf)
        concept_idx = masked.argmax(dim=1)
    else:
        raise ValueError(f"Unknown concept selection mode: {mode}")
    return pred_class, concept_idx


def choose_topk_concepts(concept_logits, norm_concepts, class_logits, final_layer, mode: str, k: int):
    pred_class = class_logits.argmax(dim=1)
    k = max(1, min(k, norm_concepts.shape[1]))
    if mode == "top_activation":
        concept_idx = norm_concepts.topk(k, dim=1).indices
        selected_mask = torch.ones_like(concept_idx, dtype=torch.bool)
    elif mode == "top_class_contribution":
        weights = final_layer.weight[pred_class]
        contribution = weights * norm_concepts
        concept_idx = contribution.topk(k, dim=1).indices
        selected_mask = torch.ones_like(concept_idx, dtype=torch.bool)
    elif mode == "top_active_class_contribution":
        weights = final_layer.weight[pred_class]
        contribution = weights * norm_concepts
        eligible = norm_concepts > 0
        masked = contribution.masked_fill(~eligible, -torch.inf)
        concept_idx = masked.topk(k, dim=1).indices
        selected_mask = eligible.gather(1, concept_idx)
    else:
        raise ValueError(f"Unknown concept selection mode: {mode}")
    return pred_class, concept_idx, selected_mask


def finalize(
    metrics: Dict[str, float],
    fractions: List[float],
    mode: str,
    random_trials: int,
    top_concepts_per_image: int,
) -> Dict:
    n = max(1.0, metrics["n"])
    result = {
        "n": int(metrics["n"]),
        "concept_selection": mode,
        "top_concepts_per_image": top_concepts_per_image,
        "random_trials": random_trials,
        "accuracy_before": metrics["accuracy_before_sum"] / n,
        "mean_selected_concept_logit": metrics["top_concept_logit_sum"] / n,
        "fractions": {},
    }
    for frac in fractions:
        tag = f"f{frac:g}"
        entry = {}
        for key, value in metrics.items():
            prefix = f"{tag}_"
            if key.startswith(prefix):
                entry[key[len(prefix):]] = value / n
        entry["selected_minus_random_class_logit_drop"] = (
            entry["class_logit_drop_selected"] - entry["class_logit_drop_random"]
        )
        entry["selected_minus_random_concept_logit_drop"] = (
            entry["concept_logit_drop_selected"] - entry["concept_logit_drop_random"]
        )
        entry["selected_minus_random_normalized_concept_logit_drop"] = (
            entry["normalized_concept_logit_drop_selected"]
            - entry["normalized_concept_logit_drop_random"]
        )
        result["fractions"][str(frac)] = entry
    return result


def accumulate_selected_concepts(
    images,
    labels,
    concept_logits,
    spatial_maps,
    class_logits,
    forward_fn,
    run_args,
    backbone,
    concept_layer,
    final_layer,
    mean,
    std,
    fractions,
    random_trials,
    random_mask_mode,
    deletion_region,
    mask_maps,
    patch_size,
    top_blocks,
    fill_value,
    generator,
    metrics,
    skip_insertion,
    concept_selection,
    top_concepts_per_image,
):
    batch = images.shape[0]
    device = images.device
    probs = class_logits.softmax(dim=1)
    norm_concepts = (concept_logits - mean) / std
    pred_class, concept_idx_2d, selected_mask_2d = choose_topk_concepts(
        concept_logits,
        norm_concepts,
        class_logits,
        final_layer,
        concept_selection,
        top_concepts_per_image,
    )
    k = concept_idx_2d.shape[1]
    pair_count = int(selected_mask_2d.sum().item())
    if pair_count == 0:
        return
    row_2d = torch.arange(batch, device=device)[:, None]
    selected_rows = row_2d.expand(batch, k)[selected_mask_2d]
    concept_idx = concept_idx_2d[selected_mask_2d]
    repeated_images = images[selected_rows]
    repeated_labels = labels[selected_rows]
    repeated_pred_class = pred_class[selected_rows]
    repeated_base_class_logit = class_logits[selected_rows, repeated_pred_class]
    repeated_base_class_prob = probs[selected_rows, repeated_pred_class]
    repeated_base_concept = concept_logits[selected_rows, concept_idx]
    repeated_base_norm_concept = norm_concepts[selected_rows, concept_idx]
    repeated_mean = mean[concept_idx]
    repeated_std = std[concept_idx]
    if mask_maps is None:
        selected_maps = spatial_maps[selected_rows, concept_idx]
    else:
        selected_maps = mask_maps[selected_rows]

    metrics["n"] += float(pair_count)
    metrics["accuracy_before_sum"] += (
        class_logits.argmax(dim=1)[selected_rows] == repeated_labels
    ).float().sum().item()
    metrics["top_concept_logit_sum"] += repeated_base_concept.sum().item()

    row = torch.arange(pair_count, device=device)
    for frac in fractions:
        tag = f"f{frac:g}"
        if deletion_region == "top_patch":
            mask = top_patch_mask(
                selected_maps,
                images.shape[-2:],
                patch_size=patch_size,
                top_blocks=top_blocks,
            )
        else:
            mask = top_fraction_mask(selected_maps, images.shape[-2:], frac)
        deleted = apply_mask(repeated_images, mask, fill_value, insertion=False)
        del_concepts, _, del_logits = forward_fn(
            deleted, run_args, backbone, concept_layer, final_layer, mean, std
        )
        del_probs = del_logits.softmax(dim=1)
        metrics[f"{tag}_class_logit_drop_selected"] += (
            repeated_base_class_logit - del_logits[row, repeated_pred_class]
        ).sum().item()
        metrics[f"{tag}_class_prob_drop_selected"] += (
            repeated_base_class_prob - del_probs[row, repeated_pred_class]
        ).sum().item()
        metrics[f"{tag}_concept_logit_drop_selected"] += (
            repeated_base_concept - del_concepts[row, concept_idx]
        ).sum().item()
        del_norm_concept = (del_concepts[row, concept_idx] - repeated_mean) / repeated_std
        metrics[f"{tag}_normalized_concept_logit_drop_selected"] += (
            repeated_base_norm_concept - del_norm_concept
        ).sum().item()
        metrics[f"{tag}_accuracy_after_selected_sum"] += (
            del_logits.argmax(dim=1) == repeated_labels
        ).float().sum().item()
        if not skip_insertion:
            inserted = apply_mask(repeated_images, mask, fill_value, insertion=True)
            _, _, ins_logits = forward_fn(
                inserted, run_args, backbone, concept_layer, final_layer, mean, std
            )
            ins_probs = ins_logits.softmax(dim=1)
            metrics[f"{tag}_class_prob_insertion_selected"] += (
                ins_probs[row, repeated_pred_class]
            ).sum().item()

        random_class_drop = random_prob_drop = random_concept_drop = 0.0
        random_norm_concept_drop = 0.0
        random_acc = random_insert_prob = 0.0
        for _ in range(random_trials):
            if deletion_region == "top_patch":
                random_mask = random_patch_mask(
                    pair_count,
                    images.shape[-2],
                    images.shape[-1],
                    patch_size=patch_size,
                    top_blocks=top_blocks,
                    device=device,
                    generator=generator,
                )
            else:
                random_mask = random_fraction_mask(
                    pair_count,
                    images.shape[-2],
                    images.shape[-1],
                    frac,
                    device,
                    generator,
                    mode=random_mask_mode,
                )
            random_deleted = apply_mask(repeated_images, random_mask, fill_value, insertion=False)
            rand_concepts, _, rand_logits = forward_fn(
                random_deleted, run_args, backbone, concept_layer, final_layer, mean, std
            )
            rand_probs = rand_logits.softmax(dim=1)
            random_class_drop += (
                repeated_base_class_logit - rand_logits[row, repeated_pred_class]
            ).sum().item()
            random_prob_drop += (
                repeated_base_class_prob - rand_probs[row, repeated_pred_class]
            ).sum().item()
            random_concept_drop += (
                repeated_base_concept - rand_concepts[row, concept_idx]
            ).sum().item()
            rand_norm_concept = (rand_concepts[row, concept_idx] - repeated_mean) / repeated_std
            random_norm_concept_drop += (
                repeated_base_norm_concept - rand_norm_concept
            ).sum().item()
            random_acc += (rand_logits.argmax(dim=1) == repeated_labels).float().sum().item()
            if not skip_insertion:
                random_inserted = apply_mask(repeated_images, random_mask, fill_value, insertion=True)
                _, _, rand_ins_logits = forward_fn(
                    random_inserted, run_args, backbone, concept_layer, final_layer, mean, std
                )
                rand_ins_probs = rand_ins_logits.softmax(dim=1)
                random_insert_prob += rand_ins_probs[row, repeated_pred_class].sum().item()
        inv_trials = 1.0 / max(1, random_trials)
        metrics[f"{tag}_class_logit_drop_random"] += random_class_drop * inv_trials
        metrics[f"{tag}_class_prob_drop_random"] += random_prob_drop * inv_trials
        metrics[f"{tag}_concept_logit_drop_random"] += random_concept_drop * inv_trials
        metrics[f"{tag}_normalized_concept_logit_drop_random"] += (
            random_norm_concept_drop * inv_trials
        )
        metrics[f"{tag}_accuracy_after_random_sum"] += random_acc * inv_trials
        if not skip_insertion:
            metrics[f"{tag}_class_prob_insertion_random"] += random_insert_prob * inv_trials


def repeated_by_concept(tensor: torch.Tensor, n_concepts: int) -> torch.Tensor:
    return tensor[:, None].expand(*tensor.shape[:1], n_concepts, *tensor.shape[1:]).reshape(
        tensor.shape[0] * n_concepts, *tensor.shape[1:]
    )


def accumulate_all_concepts(
    images,
    labels,
    concept_logits,
    spatial_maps,
    class_logits,
    forward_fn,
    run_args,
    backbone,
    concept_layer,
    final_layer,
    mean,
    std,
    fractions,
    random_trials,
    random_mask_mode,
    fill_value,
    generator,
    concept_chunk_size,
    metrics,
    skip_insertion,
):
    batch, num_concepts = concept_logits.shape
    device = images.device
    probs = class_logits.softmax(dim=1)
    pred_class = class_logits.argmax(dim=1)
    base_class_logit = class_logits[torch.arange(batch, device=device), pred_class]
    base_class_prob = probs[torch.arange(batch, device=device), pred_class]

    for start in range(0, num_concepts, concept_chunk_size):
        end = min(num_concepts, start + concept_chunk_size)
        chunk = end - start
        pair_count = batch * chunk
        concept_idx = torch.arange(start, end, device=device).repeat(batch)
        repeated_images = images[:, None].expand(batch, chunk, *images.shape[1:]).reshape(
            pair_count, *images.shape[1:]
        )
        repeated_labels = labels[:, None].expand(batch, chunk).reshape(pair_count)
        repeated_pred_class = pred_class[:, None].expand(batch, chunk).reshape(pair_count)
        repeated_base_class_logit = base_class_logit[:, None].expand(batch, chunk).reshape(pair_count)
        repeated_base_class_prob = base_class_prob[:, None].expand(batch, chunk).reshape(pair_count)
        repeated_base_concept = concept_logits[:, start:end].reshape(pair_count)
        selected_maps = spatial_maps[:, start:end].reshape(pair_count, *spatial_maps.shape[-2:])

        metrics["n"] += float(pair_count)
        metrics["accuracy_before_sum"] += (
            class_logits.argmax(dim=1)[:, None].expand(batch, chunk).reshape(pair_count)
            == repeated_labels
        ).float().sum().item()
        metrics["top_concept_logit_sum"] += repeated_base_concept.sum().item()

        for frac in fractions:
            tag = f"f{frac:g}"
            mask = top_fraction_mask(selected_maps, images.shape[-2:], frac)
            deleted = apply_mask(repeated_images, mask, fill_value, insertion=False)
            del_concepts, _, del_logits = forward_fn(
                deleted, run_args, backbone, concept_layer, final_layer, mean, std
            )
            del_probs = del_logits.softmax(dim=1)
            row = torch.arange(pair_count, device=device)
            metrics[f"{tag}_class_logit_drop_selected"] += (
                repeated_base_class_logit - del_logits[row, repeated_pred_class]
            ).sum().item()
            metrics[f"{tag}_class_prob_drop_selected"] += (
                repeated_base_class_prob - del_probs[row, repeated_pred_class]
            ).sum().item()
            metrics[f"{tag}_concept_logit_drop_selected"] += (
                repeated_base_concept - del_concepts[row, concept_idx]
            ).sum().item()
            metrics[f"{tag}_accuracy_after_selected_sum"] += (
                del_logits.argmax(dim=1) == repeated_labels
            ).float().sum().item()
            if not skip_insertion:
                inserted = apply_mask(repeated_images, mask, fill_value, insertion=True)
                _, _, ins_logits = forward_fn(
                    inserted, run_args, backbone, concept_layer, final_layer, mean, std
                )
                ins_probs = ins_logits.softmax(dim=1)
                metrics[f"{tag}_class_prob_insertion_selected"] += (
                    ins_probs[row, repeated_pred_class]
                ).sum().item()

            random_class_drop = random_prob_drop = random_concept_drop = 0.0
            random_acc = random_insert_prob = 0.0
            for _ in range(random_trials):
                random_mask = random_fraction_mask(
                    pair_count,
                    images.shape[-2],
                    images.shape[-1],
                    frac,
                    device,
                    generator,
                    mode=random_mask_mode,
                )
                random_deleted = apply_mask(repeated_images, random_mask, fill_value, insertion=False)
                rand_concepts, _, rand_logits = forward_fn(
                    random_deleted, run_args, backbone, concept_layer, final_layer, mean, std
                )
                rand_probs = rand_logits.softmax(dim=1)
                random_class_drop += (
                    repeated_base_class_logit - rand_logits[row, repeated_pred_class]
                ).sum().item()
                random_prob_drop += (
                    repeated_base_class_prob - rand_probs[row, repeated_pred_class]
                ).sum().item()
                random_concept_drop += (
                    repeated_base_concept - rand_concepts[row, concept_idx]
                ).sum().item()
                random_acc += (rand_logits.argmax(dim=1) == repeated_labels).float().sum().item()
                if not skip_insertion:
                    random_inserted = apply_mask(
                        repeated_images, random_mask, fill_value, insertion=True
                    )
                    _, _, rand_ins_logits = forward_fn(
                        random_inserted, run_args, backbone, concept_layer, final_layer, mean, std
                    )
                    rand_ins_probs = rand_ins_logits.softmax(dim=1)
                    random_insert_prob += rand_ins_probs[row, repeated_pred_class].sum().item()
            inv_trials = 1.0 / max(1, random_trials)
            metrics[f"{tag}_class_logit_drop_random"] += random_class_drop * inv_trials
            metrics[f"{tag}_class_prob_drop_random"] += random_prob_drop * inv_trials
            metrics[f"{tag}_concept_logit_drop_random"] += random_concept_drop * inv_trials
            metrics[f"{tag}_accuracy_after_random_sum"] += random_acc * inv_trials
            if not skip_insertion:
                metrics[f"{tag}_class_prob_insertion_random"] += random_insert_prob * inv_trials


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact_dir", required=True)
    parser.add_argument(
        "--model_name",
        default="sgcbm",
        choices=["sgcbm", "imagenet_sgcbm", "savlg_cbm", "salf_cbm", "salf_imagenet_official", "vlg_cbm", "cub_cbm"],
    )
    parser.add_argument("--nec", type=int, default=0, help="Load W_g@NEC=<nec>.pt / b_g@NEC=<nec>.pt from artifact_dir or --nec_dir.")
    parser.add_argument("--nec_dir", default="", help="Optional directory containing W_g@NEC=*.pt / b_g@NEC=*.pt.")
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--imagenet_val_root", default="", help="Flat or ImageFolder ImageNet validation root for ImageNet perturbation.")
    parser.add_argument("--imagenet_devkit_dir", default="", help="Optional ILSVRC2012 devkit root with data/meta.mat and validation labels.")
    parser.add_argument(
        "--imagenet_label_order",
        default="auto",
        choices=["auto", "sorted_wnid", "ilsvrc_id"],
        help="ImageNet label order for flat val labels. auto uses sorted WNID for SG/VLG and ILSVRC id for official SALF.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--no_pin_memory", action="store_true")
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--fractions", default="0.05,0.1,0.2")
    parser.add_argument("--random_trials", type=int, default=3)
    parser.add_argument("--random_mask_mode", default="box", choices=["box", "pixel"])
    parser.add_argument("--deletion_region", default="top_fraction", choices=["top_fraction", "top_patch"])
    parser.add_argument("--mask_source", default="concept_map", choices=["concept_map", "gradcam"])
    parser.add_argument("--patch_size", type=int, default=32)
    parser.add_argument("--top_blocks", type=int, default=1)
    parser.add_argument("--top_concepts_per_image", type=int, default=1)
    parser.add_argument(
        "--concept_selection",
        default="top_class_contribution",
        choices=[
            "top_class_contribution",
            "top_active_class_contribution",
            "top_activation",
            "all_concepts",
        ],
    )
    parser.add_argument("--concept_chunk_size", type=int, default=32)
    parser.add_argument("--skip_insertion", action="store_true")
    parser.add_argument(
        "--relu_concepts_for_eval",
        action="store_true",
        help="Clamp concept logits to be non-negative before normalization/classification during eval only.",
    )
    parser.add_argument("--fill_value", type=float, default=0.0)
    parser.add_argument("--salf_imagenet_backbone", default="resnet50_imagenet")
    parser.add_argument("--salf_imagenet_feature_layer", default="layer4")
    parser.add_argument("--salf_imagenet_map_size", default="12,12")
    parser.add_argument("--salf_imagenet_pooling", default="softmax", choices=["avg", "softmax"])
    parser.add_argument("--output", default="")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    artifact_dir = Path(args.artifact_dir)
    fractions = parse_fractions(args.fractions)
    pin_memory = not args.no_pin_memory
    if args.model_name == "imagenet_sgcbm":
        run_args, concepts, backbone, concept_layer, final_layer, mean, std = load_imagenet_sgcbm(
            artifact_dir,
            args.device,
            batch_size=args.batch_size,
            workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            persistent_workers=args.persistent_workers,
            pin_memory=pin_memory,
        )
        forward_fn = forward_imagenet_sgcbm
    elif args.model_name == "salf_imagenet_official":
        run_args, concepts, backbone, concept_layer, final_layer, mean, std = load_salf_imagenet_official(
            artifact_dir,
            args.device,
            backbone_name=args.salf_imagenet_backbone,
            feature_layer=args.salf_imagenet_feature_layer,
            map_size=args.salf_imagenet_map_size,
            pooling=args.salf_imagenet_pooling,
        )
        forward_fn = forward_salf_imagenet_official
    elif args.model_name == "salf_cbm":
        run_args, concepts, backbone, concept_layer, final_layer, mean, std = load_salf(
            artifact_dir, args.device
        )
        forward_fn = forward_salf
    elif args.model_name in {"vlg_cbm", "cub_cbm"}:
        run_args, concepts, backbone, concept_layer, final_layer, mean, std = load_vlg(
            artifact_dir, args.device
        )
        forward_fn = forward_vlg
    else:
        run_args, concepts, backbone, concept_layer, final_layer, mean, std = load_sgcbm(
            artifact_dir, args.device
        )
        forward_fn = forward_sgcbm
    if args.nec > 0:
        nec_dir = Path(args.nec_dir) if args.nec_dir else artifact_dir
        weight_path = nec_dir / f"W_g@NEC={args.nec}.pt"
        bias_path = nec_dir / f"b_g@NEC={args.nec}.pt"
        if not weight_path.is_file() or not bias_path.is_file():
            raise FileNotFoundError(f"Missing NEC={args.nec} final layer: {weight_path} / {bias_path}")
        final_layer.weight.data.copy_(torch.load(weight_path, map_location=args.device))
        final_layer.bias.data.copy_(torch.load(bias_path, map_location=args.device))
        final_layer.eval()
    if args.relu_concepts_for_eval:
        base_forward_fn = forward_fn

        def forward_relu_concepts(images, run_args, backbone, concept_layer, final_layer, mean, std):
            concept_logits, spatial_maps, _ = base_forward_fn(
                images, run_args, backbone, concept_layer, final_layer, mean, std
            )
            concept_logits = concept_logits.clamp_min(0.0)
            class_logits = final_layer((concept_logits - mean) / std)
            return concept_logits, spatial_maps, class_logits

        forward_fn = forward_relu_concepts
    if args.mask_source == "gradcam" and args.model_name not in {"vlg_cbm", "cub_cbm"}:
        raise ValueError("--mask_source gradcam is currently implemented for VLG-CBM/CUB-CBM checkpoints.")

    dataset_name = f"{run_args.dataset}_{args.split}"
    imagenet_label_order = ""
    if run_args.dataset == "imagenet" and args.split == "val" and args.imagenet_val_root:
        preprocess = getattr(backbone, "preprocess", None)
        if preprocess is None:
            preprocess = data_utils.get_resnet_imagenet_preprocess()
        labels = None
        if args.imagenet_devkit_dir:
            imagenet_label_order = resolve_imagenet_label_order(args.model_name, args.imagenet_label_order)
            labels = load_imagenet_val_targets(Path(args.imagenet_devkit_dir), imagenet_label_order)
            label_mode = f"imagenet_devkit_{imagenet_label_order}"
        else:
            label_mode = "baseline_prediction"
        dataset = FlatImageNetValDataset(Path(args.imagenet_val_root), preprocess, labels=labels)
    else:
        preprocess = getattr(backbone, "preprocess", None)
        if preprocess is None:
            preprocess = data_utils.get_resnet_imagenet_preprocess()
        dataset = data_utils.get_data(dataset_name, preprocess)
        label_mode = "dataset_label"
    if args.max_images > 0:
        dataset = Subset(dataset, list(range(min(args.max_images, len(dataset)))))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.persistent_workers and args.num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )

    generator = torch.Generator(device=args.device).manual_seed(args.seed + 17)
    metrics = init_metrics(fractions)

    for images, labels in tqdm(loader, desc="spatial perturbation"):
        images = images.to(args.device, non_blocking=True)
        labels = labels.to(args.device, non_blocking=True)
        concept_logits, spatial_maps, class_logits = forward_fn(
            images, run_args, backbone, concept_layer, final_layer, mean, std
        )
        if (labels < 0).all():
            labels = class_logits.argmax(dim=1).detach()
        elif (labels < 0).any():
            raise RuntimeError(
                "Only some ImageNet labels are missing. Check --imagenet_val_root filenames and --imagenet_devkit_dir."
            )
        mask_maps = None
        if args.mask_source == "gradcam":
            mask_maps = compute_vlg_gradcam_maps(
                images, backbone, concept_layer, final_layer, mean, std
            )
        if args.concept_selection == "all_concepts":
            if args.mask_source != "concept_map":
                raise ValueError("--concept_selection all_concepts requires --mask_source concept_map.")
            accumulate_all_concepts(
                images=images,
                labels=labels,
                concept_logits=concept_logits,
                spatial_maps=spatial_maps,
                class_logits=class_logits,
                forward_fn=forward_fn,
                run_args=run_args,
                backbone=backbone,
                concept_layer=concept_layer,
                final_layer=final_layer,
                mean=mean,
                std=std,
                fractions=fractions,
                random_trials=args.random_trials,
                random_mask_mode=args.random_mask_mode,
                fill_value=args.fill_value,
                generator=generator,
                concept_chunk_size=args.concept_chunk_size,
                metrics=metrics,
                skip_insertion=args.skip_insertion,
            )
            continue
        if args.top_concepts_per_image > 1 or args.concept_selection == "top_active_class_contribution":
            accumulate_selected_concepts(
                images=images,
                labels=labels,
                concept_logits=concept_logits,
                spatial_maps=spatial_maps,
                class_logits=class_logits,
                forward_fn=forward_fn,
                run_args=run_args,
                backbone=backbone,
                concept_layer=concept_layer,
                final_layer=final_layer,
                mean=mean,
                std=std,
                fractions=fractions,
                random_trials=args.random_trials,
                random_mask_mode=args.random_mask_mode,
                deletion_region=args.deletion_region,
                mask_maps=mask_maps,
                patch_size=args.patch_size,
                top_blocks=args.top_blocks,
                fill_value=args.fill_value,
                generator=generator,
                metrics=metrics,
                skip_insertion=args.skip_insertion,
                concept_selection=args.concept_selection,
                top_concepts_per_image=args.top_concepts_per_image,
            )
            continue
        norm_concepts = (concept_logits - mean) / std
        probs = class_logits.softmax(dim=1)
        pred_class, concept_idx = choose_concepts(
            concept_logits, norm_concepts, class_logits, final_layer, args.concept_selection
        )
        batch = images.shape[0]
        if mask_maps is None:
            selected_maps = spatial_maps[torch.arange(batch, device=args.device), concept_idx]
        else:
            selected_maps = mask_maps
        base_class_logit = class_logits[torch.arange(batch, device=args.device), pred_class]
        base_class_prob = probs[torch.arange(batch, device=args.device), pred_class]
        base_concept_logit = concept_logits[torch.arange(batch, device=args.device), concept_idx]

        metrics["n"] += float(batch)
        metrics["accuracy_before_sum"] += (class_logits.argmax(dim=1) == labels).float().sum().item()
        metrics["top_concept_logit_sum"] += base_concept_logit.sum().item()

        for frac in fractions:
            tag = f"f{frac:g}"
            if args.deletion_region == "top_patch":
                mask = top_patch_mask(
                    selected_maps,
                    images.shape[-2:],
                    patch_size=args.patch_size,
                    top_blocks=args.top_blocks,
                )
            else:
                mask = top_fraction_mask(selected_maps, images.shape[-2:], frac)
            deleted = apply_mask(images, mask, args.fill_value, insertion=False)
            inserted = apply_mask(images, mask, args.fill_value, insertion=True)
            del_concepts, _, del_logits = forward_fn(
                deleted, run_args, backbone, concept_layer, final_layer, mean, std
            )
            ins_concepts, _, ins_logits = forward_fn(
                inserted, run_args, backbone, concept_layer, final_layer, mean, std
            )
            del_probs = del_logits.softmax(dim=1)
            ins_probs = ins_logits.softmax(dim=1)
            metrics[f"{tag}_class_logit_drop_selected"] += (
                base_class_logit - del_logits[torch.arange(batch, device=args.device), pred_class]
            ).sum().item()
            metrics[f"{tag}_class_prob_drop_selected"] += (
                base_class_prob - del_probs[torch.arange(batch, device=args.device), pred_class]
            ).sum().item()
            metrics[f"{tag}_concept_logit_drop_selected"] += (
                base_concept_logit - del_concepts[torch.arange(batch, device=args.device), concept_idx]
            ).sum().item()
            metrics[f"{tag}_accuracy_after_selected_sum"] += (
                del_logits.argmax(dim=1) == labels
            ).float().sum().item()
            metrics[f"{tag}_class_prob_insertion_selected"] += (
                ins_probs[torch.arange(batch, device=args.device), pred_class]
            ).sum().item()

            random_class_drop = random_prob_drop = random_concept_drop = 0.0
            random_acc = random_insert_prob = 0.0
            for _ in range(args.random_trials):
                if args.deletion_region == "top_patch":
                    random_mask = random_patch_mask(
                        batch,
                        images.shape[-2],
                        images.shape[-1],
                        patch_size=args.patch_size,
                        top_blocks=args.top_blocks,
                        device=images.device,
                        generator=generator,
                    )
                else:
                    random_mask = random_fraction_mask(
                        batch,
                        images.shape[-2],
                        images.shape[-1],
                        frac,
                        images.device,
                        generator,
                        mode=args.random_mask_mode,
                    )
                random_deleted = apply_mask(images, random_mask, args.fill_value, insertion=False)
                random_inserted = apply_mask(images, random_mask, args.fill_value, insertion=True)
                rand_concepts, _, rand_logits = forward_fn(
                    random_deleted, run_args, backbone, concept_layer, final_layer, mean, std
                )
                _, _, rand_ins_logits = forward_fn(
                    random_inserted, run_args, backbone, concept_layer, final_layer, mean, std
                )
                rand_probs = rand_logits.softmax(dim=1)
                rand_ins_probs = rand_ins_logits.softmax(dim=1)
                random_class_drop += (
                    base_class_logit - rand_logits[torch.arange(batch, device=args.device), pred_class]
                ).sum().item()
                random_prob_drop += (
                    base_class_prob - rand_probs[torch.arange(batch, device=args.device), pred_class]
                ).sum().item()
                random_concept_drop += (
                    base_concept_logit - rand_concepts[torch.arange(batch, device=args.device), concept_idx]
                ).sum().item()
                random_acc += (rand_logits.argmax(dim=1) == labels).float().sum().item()
                random_insert_prob += (
                    rand_ins_probs[torch.arange(batch, device=args.device), pred_class]
                ).sum().item()
            inv_trials = 1.0 / max(1, args.random_trials)
            metrics[f"{tag}_class_logit_drop_random"] += random_class_drop * inv_trials
            metrics[f"{tag}_class_prob_drop_random"] += random_prob_drop * inv_trials
            metrics[f"{tag}_concept_logit_drop_random"] += random_concept_drop * inv_trials
            metrics[f"{tag}_accuracy_after_random_sum"] += random_acc * inv_trials
            metrics[f"{tag}_class_prob_insertion_random"] += random_insert_prob * inv_trials

    result = finalize(
        metrics,
        fractions,
        args.concept_selection,
        args.random_trials,
        args.top_concepts_per_image,
    )
    result.update(
        {
            "artifact_dir": str(artifact_dir),
            "model_name": args.model_name,
            "nec": args.nec,
            "dataset": run_args.dataset,
            "split": args.split,
            "label_mode": label_mode,
            "imagenet_label_order": imagenet_label_order,
            "fill_value_normalized_space": args.fill_value,
            "random_mask_mode": args.random_mask_mode,
            "deletion_region": args.deletion_region,
            "mask_source": args.mask_source,
            "patch_size": args.patch_size,
            "top_blocks": args.top_blocks,
            "skip_insertion": args.skip_insertion,
            "concept_chunk_size": args.concept_chunk_size,
            "num_concepts": len(concepts),
        }
    )
    output = Path(args.output) if args.output else artifact_dir / "spatial_perturbation_metrics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"saved to {output}")


if __name__ == "__main__":
    main()
