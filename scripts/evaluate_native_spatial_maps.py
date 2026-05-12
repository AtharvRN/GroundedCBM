import argparse
import json
import os
import re
import sys
from argparse import Namespace
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data import utils as data_utils
from methods.common import load_run_info
from methods.lf import make_projection_layer
from methods.salf import SpatialBackbone, build_spatial_concept_layer
from methods.savlg import (
    build_savlg_concept_layer,
    compute_savlg_concept_logits,
    forward_savlg_backbone,
    forward_savlg_concept_layer,
)
from model.cbm import Backbone, ConceptLayer


def parse_thresholds(raw: str) -> List[float]:
    values = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))
    if not values:
        raise ValueError("Expected at least one threshold value.")
    return values


def format_threshold_key(value: float, mode: str) -> str:
    if mode == "mean":
        return "mean"
    if mode == "percentile":
        if float(value).is_integer():
            return f"p{int(value)}"
        return f"p{value}"
    return str(value)


def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def load_concepts(path: str) -> List[str]:
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def resolve_cub_metadata_root(path: Optional[str]) -> Optional[str]:
    candidates = []
    if path:
        candidates.append(path)
    candidates.extend(
        [
            os.path.join(REPO_ROOT, "datasets", "CUB_200_2011"),
            os.path.expanduser("~/Downloads/CUB_200_2011"),
        ]
    )
    for candidate in candidates:
        if candidate and os.path.exists(os.path.join(candidate, "images.txt")):
            return candidate
    return None


def resolve_annotation_split_dir(annotation_root: str, dataset: str, split_name: str) -> str:
    candidates = [
        os.path.join(annotation_root, f"{dataset}_{split_name}"),
    ]
    if split_name in {"test", "val"}:
        candidates.append(os.path.join(annotation_root, f"{dataset}_val"))
        candidates.append(os.path.join(annotation_root, f"{dataset}_test"))
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError(
        f"Could not find annotation directory for dataset={dataset} split={split_name} under {annotation_root}"
    )


def add_default_args(args: Namespace) -> Namespace:
    defaults = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "cbl_type": "linear",
        "cbl_hidden_layers": 1,
        "cbl_use_batchnorm": False,
        "cbl_hidden_dim": 0,
        "num_workers": 0,
        "use_clip_penultimate": False,
        "savlg_spatial_stage": "conv5",
        "savlg_spatial_branch_mode": "shared_stage",
        "savlg_residual_spatial_alpha": 0.0,
        "savlg_residual_spatial_pooling": "lse",
    }
    for key, value in defaults.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    return args


def normalize_box(
    box: Sequence[float],
    image_size: Tuple[int, int],
) -> Optional[Tuple[float, float, float, float]]:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    x1, y1, x2, y2 = [float(v) for v in box]
    width, height = int(image_size[0]), int(image_size[1])
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) > 1.5:
        if width <= 0 or height <= 0:
            return None
        x1, x2 = x1 / width, x2 / width
        y1, y2 = y1 / height, y2 / height
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1 = float(np.clip(x1, 0.0, 1.0))
    x2 = float(np.clip(x2, 0.0, 1.0))
    y1 = float(np.clip(y1, 0.0, 1.0))
    y2 = float(np.clip(y2, 0.0, 1.0))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def rasterize_box_union(
    boxes: Sequence[Sequence[float]],
    image_size: Tuple[int, int],
    map_h: int,
    map_w: int,
) -> np.ndarray:
    norm_boxes = []
    for box in boxes:
        norm = normalize_box(box, image_size=image_size)
        if norm is not None:
            norm_boxes.append(norm)
    if not norm_boxes:
        return np.zeros((map_h, map_w), dtype=np.bool_)

    boxes_arr = np.asarray(norm_boxes, dtype=np.float32)
    x1 = boxes_arr[:, 0][:, None, None]
    y1 = boxes_arr[:, 1][:, None, None]
    x2 = boxes_arr[:, 2][:, None, None]
    y2 = boxes_arr[:, 3][:, None, None]

    px1 = (np.arange(map_w, dtype=np.float32) / float(map_w))[None, None, :]
    px2 = ((np.arange(map_w, dtype=np.float32) + 1.0) / float(map_w))[None, None, :]
    py1 = (np.arange(map_h, dtype=np.float32) / float(map_h))[None, :, None]
    py2 = ((np.arange(map_h, dtype=np.float32) + 1.0) / float(map_h))[None, :, None]

    overlap_w = np.minimum(px2, x2) - np.maximum(px1, x1)
    overlap_h = np.minimum(py2, y2) - np.maximum(py1, y1)
    return ((overlap_w > 0.0) & (overlap_h > 0.0)).any(axis=0)


def tight_box_from_mask(mask: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
    ys, xs = np.nonzero(mask)
    if ys.size == 0 or xs.size == 0:
        return None
    x1 = float(xs.min())
    y1 = float(ys.min())
    x2 = float(xs.max() + 1)
    y2 = float(ys.max() + 1)
    return x1, y1, x2, y2


def box_iou(box_a: Optional[Tuple[float, float, float, float]], box_b: Optional[Tuple[float, float, float, float]]) -> float:
    if box_a is None or box_b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def tight_boxes_from_masks(masks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if masks.ndim != 3:
        raise ValueError(f"Expected masks shape [C,H,W], got {masks.shape}")
    num_concepts, map_h, map_w = masks.shape
    if num_concepts == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.bool_)
    valid = masks.reshape(num_concepts, -1).any(axis=1)
    boxes = np.zeros((num_concepts, 4), dtype=np.float32)
    if not valid.any():
        return boxes, valid

    ys_any = masks.any(axis=2)
    xs_any = masks.any(axis=1)
    y_min = ys_any.argmax(axis=1).astype(np.float32)
    y_max = (map_h - ys_any[:, ::-1].argmax(axis=1)).astype(np.float32)
    x_min = xs_any.argmax(axis=1).astype(np.float32)
    x_max = (map_w - xs_any[:, ::-1].argmax(axis=1)).astype(np.float32)
    boxes[:, 0] = x_min
    boxes[:, 1] = y_min
    boxes[:, 2] = x_max
    boxes[:, 3] = y_max
    return boxes, valid


def box_iou_vectorized(
    pred_boxes: np.ndarray,
    pred_valid: np.ndarray,
    gt_boxes: np.ndarray,
    gt_valid: np.ndarray,
) -> np.ndarray:
    ix1 = np.maximum(pred_boxes[:, 0], gt_boxes[:, 0])
    iy1 = np.maximum(pred_boxes[:, 1], gt_boxes[:, 1])
    ix2 = np.minimum(pred_boxes[:, 2], gt_boxes[:, 2])
    iy2 = np.minimum(pred_boxes[:, 3], gt_boxes[:, 3])

    inter_w = np.maximum(0.0, ix2 - ix1)
    inter_h = np.maximum(0.0, iy2 - iy1)
    inter = inter_w * inter_h

    area_pred = np.maximum(0.0, pred_boxes[:, 2] - pred_boxes[:, 0]) * np.maximum(0.0, pred_boxes[:, 3] - pred_boxes[:, 1])
    area_gt = np.maximum(0.0, gt_boxes[:, 2] - gt_boxes[:, 0]) * np.maximum(0.0, gt_boxes[:, 3] - gt_boxes[:, 1])
    union = area_pred + area_gt - inter

    valid = pred_valid & gt_valid & (union > 0.0)
    out = np.zeros_like(inter, dtype=np.float32)
    out[valid] = inter[valid] / union[valid]
    return out


def average_precision_binary(labels: Sequence[int], scores: Sequence[float]) -> Optional[float]:
    if not labels:
        return None
    labels_np = np.asarray(labels, dtype=np.int32)
    scores_np = np.asarray(scores, dtype=np.float32)
    num_pos = int(labels_np.sum())
    if num_pos == 0:
        return None
    order = np.argsort(-scores_np, kind="mergesort")
    sorted_labels = labels_np[order]
    tp = 0.0
    precision_sum = 0.0
    for rank, label in enumerate(sorted_labels, start=1):
        if int(label) != 1:
            continue
        tp += 1.0
        precision_sum += tp / float(rank)
    return precision_sum / float(num_pos)


def segmentation_average_precision(gt_mask: np.ndarray, score_map: np.ndarray) -> Optional[float]:
    """Chefer/SALF-style pixel AP over foreground and background channels."""
    gt_flat = gt_mask.astype(np.int32, copy=False).reshape(-1)
    score_flat = score_map.astype(np.float32, copy=False).reshape(-1)
    labels = np.concatenate([1 - gt_flat, gt_flat], axis=0)
    scores = np.concatenate([1.0 - score_flat, score_flat], axis=0)
    return average_precision_binary(labels.tolist(), scores.tolist())


def spatial_distribution_from_map(score_map: np.ndarray) -> np.ndarray:
    """Softmax-normalize one spatial map into a probability distribution."""
    flat = score_map.astype(np.float64, copy=False).reshape(-1)
    if flat.size == 0:
        return flat.reshape(score_map.shape)
    flat = flat - np.max(flat)
    exp = np.exp(flat)
    denom = float(exp.sum())
    if denom <= 0.0 or not np.isfinite(denom):
        return np.full(score_map.shape, 1.0 / float(max(flat.size, 1)), dtype=np.float64)
    return (exp / denom).reshape(score_map.shape)


def mask_iou_from_score_map(score_map: np.ndarray, gt_mask: np.ndarray, threshold: float) -> float:
    pred_mask = score_map >= float(threshold)
    intersection = np.logical_and(pred_mask, gt_mask).sum(dtype=np.float64)
    union = np.logical_or(pred_mask, gt_mask).sum(dtype=np.float64)
    if union <= 0.0:
        return 0.0
    return float(intersection / union)


class IndexedPreprocessDataset(Dataset):
    def __init__(self, base_dataset, preprocess):
        self.base_dataset = base_dataset
        self.preprocess = preprocess

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, _ = self.base_dataset[idx]
        width, height = image.size
        tensor = self.preprocess(image)
        return tensor, idx, width, height


def collate_indexed(batch):
    images = torch.stack([row[0] for row in batch], dim=0)
    indices = torch.tensor([row[1] for row in batch], dtype=torch.long)
    widths = torch.tensor([row[2] for row in batch], dtype=torch.long)
    heights = torch.tensor([row[3] for row in batch], dtype=torch.long)
    return images, indices, widths, heights


@dataclass
class SpatialMapModel:
    name: str
    load_path: str
    args: Namespace
    concepts: List[str]
    concept_to_idx: Dict[str, int]
    backbone: SpatialBackbone
    concept_layer: torch.nn.Module
    preprocess: object
    model_name: str
    proj_mean: Optional[torch.Tensor]
    proj_std: Optional[torch.Tensor]
    eval_kind: str
    supports_native_maps: bool
    gradcam_backbone: Optional[Backbone]

    def predict_maps(self, images: torch.Tensor) -> torch.Tensor:
        if not self.supports_native_maps:
            raise RuntimeError(
                f"Model {self.name} ({self.model_name}) does not expose native spatial maps. "
                "Use --map_source gradcam instead."
            )
        with torch.no_grad():
            if self.eval_kind == "savlg":
                feats = forward_savlg_backbone(self.backbone, images, self.args)
                _, spatial_maps = forward_savlg_concept_layer(self.concept_layer, feats)
            else:
                feats = self.backbone(images.to(self.args.device))
                spatial_maps = self.concept_layer(feats)
        return spatial_maps

    def predict_maps_with_global(self, images: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Predict spatial maps and global logits (for gating).
        Returns: (spatial_maps, global_logits) where global_logits is None for non-SAVLG models.
        """
        if not self.supports_native_maps:
            raise RuntimeError(
                f"Model {self.name} ({self.model_name}) does not expose native spatial maps. "
                "Use --map_source gradcam instead."
            )
        with torch.no_grad():
            if self.eval_kind == "savlg":
                feats = forward_savlg_backbone(self.backbone, images, self.args)
                global_outputs, spatial_maps = forward_savlg_concept_layer(self.concept_layer, feats)
                # Compute global logits from global_outputs
                global_logits = compute_savlg_concept_logits(
                    global_outputs,
                    spatial_maps,
                    self.args,
                )[2]  # Returns (pooled, residual, final_logits)
                return spatial_maps, global_logits
            else:
                feats = self.backbone(images.to(self.args.device))
                spatial_maps = self.concept_layer(feats)
                return spatial_maps, None

    def forward_for_gradcam(
        self,
        images: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        images = images.to(self.args.device)
        if self.eval_kind == "savlg":
            feats = forward_savlg_backbone(self.backbone, images, self.args)
            global_outputs, spatial_maps = forward_savlg_concept_layer(self.concept_layer, feats)
            grad_source = feats["conv5"] if isinstance(feats, dict) else feats
            logits = compute_savlg_concept_logits(
                global_outputs,
                spatial_maps,
                self.args,
            )[2]
            return grad_source, spatial_maps, logits
        if self.eval_kind == "salf":
            feats = self.backbone(images)
            global_outputs = self.concept_layer(feats)
            spatial_maps = global_outputs
            logits = F.adaptive_avg_pool2d(global_outputs, 1).flatten(1)
            return feats, spatial_maps, logits
        if self.eval_kind in {"vlg", "lf"}:
            if self.gradcam_backbone is None:
                raise RuntimeError(f"Model {self.name} is missing gradcam_backbone.")
            _ = self.gradcam_backbone.backbone(images)
            feats = self.gradcam_backbone.feature_vals[images.device].float()
            pooled = feats.mean(dim=[2, 3])
            logits = self.concept_layer(pooled)
            return feats, None, logits
        raise ValueError(f"Unsupported eval_kind={self.eval_kind}")


def apply_global_spatial_gating(
    spatial_maps: torch.Tensor,
    global_logits: torch.Tensor,
    gating_mode: str = "hard",
) -> torch.Tensor:
    """
    Apply global-to-spatial gating to prevent absent concept hallucination.

    Args:
        spatial_maps: Raw spatial logits [B x C x H x W]
        global_logits: Global concept logits [B x C]
        gating_mode: 'hard' for multiplicative, 'soft' for floor-based

    Returns:
        Gated spatial maps [B x C x H x W]
    """
    global_probs = torch.sigmoid(global_logits)  # B x C

    if gating_mode == "hard":
        # Pure multiplicative gating: gate = global_prob
        gate = global_probs.unsqueeze(-1).unsqueeze(-1)  # B x C x 1 x 1
    elif gating_mode == "soft":
        # Soft gating with floor: gate = 0.2 + 0.8 * global_prob (range [0.2, 1.0])
        gate = (0.2 + 0.8 * global_probs).unsqueeze(-1).unsqueeze(-1)
    else:
        raise ValueError(f"Unknown gating_mode: {gating_mode}")

    # Apply gating in log-space for numerical stability
    # log(spatial_gated) = log(spatial) + log(gate)
    spatial_logits_gated = spatial_maps + torch.log(gate + 1e-8)

    return spatial_logits_gated


def normalize_eval_maps(
    maps: torch.Tensor,
    map_normalization: str,
    proj_mean: Optional[torch.Tensor],
    proj_std: Optional[torch.Tensor],
) -> torch.Tensor:
    if map_normalization == "concept_zscore_minmax":
        map_normalization = "proj_zscore_minmax"
    if map_normalization == "sigmoid":
        return torch.sigmoid(maps)
    if map_normalization == "minmax":
        min_vals = maps.amin(dim=(2, 3), keepdim=True)
        max_vals = maps.amax(dim=(2, 3), keepdim=True)
        return (maps - min_vals) / torch.clamp(max_vals - min_vals, min=1e-6)
    if map_normalization == "proj_zscore_minmax":
        if proj_mean is None or proj_std is None:
            raise RuntimeError("proj_zscore_minmax requires proj_mean/proj_std.")
        maps = (maps - proj_mean.to(maps.device)) / proj_std.to(maps.device)
        min_vals = maps.amin(dim=(2, 3), keepdim=True)
        max_vals = maps.amax(dim=(2, 3), keepdim=True)
        return (maps - min_vals) / torch.clamp(max_vals - min_vals, min=1e-6)
    raise ValueError(f"Unsupported map_normalization={map_normalization}")


def predict_gradcam_maps(
    model: SpatialMapModel,
    images: torch.Tensor,
    concept_indices: torch.Tensor,
    gradcam_chunk_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    grad_source, _, logits = model.forward_for_gradcam(images)
    grad_source = grad_source.requires_grad_(True)
    selected_logits = logits.index_select(1, concept_indices.to(logits.device))
    cams = []
    num_concepts = selected_logits.shape[1]
    for start in range(0, num_concepts, gradcam_chunk_size):
        stop = min(start + gradcam_chunk_size, num_concepts)
        chunk_logits = selected_logits[:, start:stop]
        chunk_size = chunk_logits.shape[1]
        grad_outputs = torch.zeros(
            (chunk_size, chunk_logits.shape[0], chunk_size),
            device=chunk_logits.device,
            dtype=chunk_logits.dtype,
        )
        diag = torch.arange(chunk_size, device=chunk_logits.device)
        grad_outputs[diag, :, diag] = 1.0
        grads = torch.autograd.grad(
            outputs=chunk_logits,
            inputs=grad_source,
            grad_outputs=grad_outputs,
            is_grads_batched=True,
            retain_graph=stop < num_concepts,
            create_graph=False,
        )[0]
        alpha = grads.mean(dim=(-1, -2))
        chunk_cams = torch.einsum("kbc,bchw->kbhw", alpha, grad_source)
        chunk_cams = F.relu(chunk_cams).permute(1, 0, 2, 3).contiguous()
        cams.append(chunk_cams)
    return torch.cat(cams, dim=1), selected_logits.detach()


def infer_eval_kind(model_name: str) -> str:
    lowered = str(model_name).lower()
    if lowered.startswith("savlg"):
        return "savlg"
    if lowered.startswith("salf"):
        return "salf"
    if lowered.startswith("vlg"):
        return "vlg"
    if lowered.startswith("lf"):
        return "lf"
    raise ValueError(f"Unsupported model_name for evaluator: {model_name}")


def load_proj_stats(load_path: str) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    proj_mean_path = os.path.join(load_path, "proj_mean.pt")
    proj_std_path = os.path.join(load_path, "proj_std.pt")
    if not (os.path.exists(proj_mean_path) and os.path.exists(proj_std_path)):
        return None, None
    proj_mean = torch.load(proj_mean_path, map_location="cpu").float().flatten()
    proj_std = torch.load(proj_std_path, map_location="cpu").float().flatten()
    proj_std = torch.clamp(proj_std, min=1e-6)
    return proj_mean, proj_std


def resolve_gradcam_feature_layer(args: Namespace) -> str:
    feature_layer = data_utils.BACKBONE_VISUALIZATION_TARGET_LAYER.get(str(args.backbone))
    if feature_layer is None:
        raise ValueError(
            f"No spatial Grad-CAM feature layer registered for backbone {args.backbone}. "
            "Add it to data.utils.BACKBONE_VISUALIZATION_TARGET_LAYER first."
        )
    return feature_layer


def load_spatial_model(load_path: str, device: Optional[str], name: Optional[str] = None) -> SpatialMapModel:
    run_info = load_run_info(load_path)
    raw_args = load_json(os.path.join(load_path, "args.txt"))
    args = add_default_args(Namespace(**raw_args))
    if device is not None:
        args.device = device
    concepts = load_concepts(os.path.join(load_path, "concepts.txt"))
    model_name = str(run_info.get("model_name", raw_args.get("model_name", "unknown")))
    eval_kind = infer_eval_kind(model_name)
    proj_mean, proj_std = load_proj_stats(load_path)
    gradcam_backbone = None

    if eval_kind in {"savlg", "salf"}:
        backbone = SpatialBackbone(
            args.backbone,
            device=args.device,
            spatial_stage=getattr(args, "savlg_spatial_stage", "conv5"),
        )
        if eval_kind == "savlg":
            concept_layer = build_savlg_concept_layer(args, backbone, len(concepts))
        else:
            concept_layer = build_spatial_concept_layer(args, backbone.output_dim, len(concepts))
        state_dict = torch.load(os.path.join(load_path, "concept_layer.pt"), map_location=args.device)
        if isinstance(concept_layer, torch.nn.Conv2d) and "bias" not in state_dict and concept_layer.bias is not None:
            logger.info("Checkpoint {} has no concept-layer bias; loading with zero bias for backward compatibility.", load_path)
            concept_layer.load_state_dict(state_dict, strict=False)
            with torch.no_grad():
                concept_layer.bias.zero_()
        else:
            concept_layer.load_state_dict(state_dict)
        backbone.eval()
        concept_layer.eval()
        preprocess = backbone.preprocess
        supports_native_maps = True
    elif eval_kind == "vlg":
        feature_layer = resolve_gradcam_feature_layer(args)
        backbone = None
        gradcam_backbone = Backbone(args.backbone, feature_layer, args.device)
        concept_layer = ConceptLayer.from_pretrained(load_path, device=args.device)
        gradcam_backbone.eval()
        concept_layer.eval()
        preprocess = gradcam_backbone.preprocess
        supports_native_maps = False
    elif eval_kind == "lf":
        if str(args.backbone).startswith("clip"):
            raise ValueError("Grad-CAM evaluation for LF-CBM is currently only supported for spatial CNN backbones.")
        feature_layer = resolve_gradcam_feature_layer(args)
        backbone = None
        gradcam_backbone = Backbone(args.backbone, feature_layer, args.device)
        concept_format = str(run_info.get("concept_layer_format", ""))
        if concept_format == "W_c.pt":
            W_c = torch.load(os.path.join(load_path, "W_c.pt"), map_location=args.device)
            concept_layer = torch.nn.Linear(W_c.shape[1], W_c.shape[0], bias=False).to(args.device)
            concept_layer.load_state_dict({"weight": W_c})
        else:
            concept_layer = make_projection_layer(
                args,
                input_dim=data_utils.BACKBONE_ENCODING_DIMENSION[args.backbone],
                n_concepts=len(concepts),
            )
            concept_layer.load_state_dict(
                torch.load(os.path.join(load_path, "concept_layer.pt"), map_location=args.device)
            )
        gradcam_backbone.eval()
        concept_layer.eval()
        preprocess = gradcam_backbone.preprocess
        supports_native_maps = False
    else:
        raise ValueError(f"Unsupported eval_kind={eval_kind}")
    label = name or os.path.basename(load_path.rstrip("/"))
    return SpatialMapModel(
        name=label,
        load_path=load_path,
        args=args,
        concepts=concepts,
        concept_to_idx={concept: idx for idx, concept in enumerate(concepts)},
        backbone=backbone,
        concept_layer=concept_layer,
        preprocess=preprocess,
        model_name=model_name,
        proj_mean=proj_mean,
        proj_std=proj_std,
        eval_kind=eval_kind,
        supports_native_maps=supports_native_maps,
        gradcam_backbone=gradcam_backbone,
    )


class AnnotationStore:
    def __init__(
        self,
        annotation_dir: str,
        dataset: str,
        split_name: str,
        valid_concepts: Iterable[str],
        annotation_threshold: float,
    ):
        self.split_dir = resolve_annotation_split_dir(annotation_dir, dataset, split_name)
        self.valid_concepts = set(valid_concepts)
        self.annotation_threshold = float(annotation_threshold)
        self._cache: Dict[int, Dict[str, List[List[float]]]] = {}

    def get(self, idx: int) -> Dict[str, List[List[float]]]:
        cached = self._cache.get(int(idx))
        if cached is not None:
            return cached

        ann_path = os.path.join(self.split_dir, f"{int(idx)}.json")
        per_image: Dict[str, List[List[float]]] = {}
        if os.path.exists(ann_path):
            payload = load_json(ann_path)
            for row in payload[1:]:
                score = float(row.get("logit", row.get("score", 0.0)))
                if score < self.annotation_threshold:
                    continue
                label = data_utils.format_concept(str(row.get("label", "")))
                if label not in self.valid_concepts:
                    continue
                box = row.get("box")
                if not isinstance(box, list) or len(box) != 4:
                    continue
                per_image.setdefault(label, []).append([float(v) for v in box])
        self._cache[int(idx)] = per_image
        return per_image


def _parse_manifest_image_index(image_id) -> Optional[int]:
    if image_id is None:
        return None
    if isinstance(image_id, int):
        return int(image_id)
    if isinstance(image_id, str):
        # Accept formats like "image_000123" or "000123" or "123".
        m = re.search(r"(\\d+)", image_id)
        if m:
            return int(m.group(1))
    return None


def _downsample_mask_to_grid(mask: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Downsample a binary mask to a coarse grid using adaptive max pooling."""
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape={mask.shape}")
    if int(out_h) <= 0 or int(out_w) <= 0:
        raise ValueError(f"Invalid target size: {(out_h, out_w)}")
    mask_t = torch.from_numpy(mask.astype(np.float32, copy=False)).unsqueeze(0).unsqueeze(0)
    pooled = F.adaptive_max_pool2d(mask_t, output_size=(int(out_h), int(out_w)))[0, 0]
    return (pooled > 0.0).cpu().numpy().astype(np.bool_, copy=False)


def _resize_mask_nearest(mask: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape={mask.shape}")
    mask_t = torch.from_numpy(mask.astype(np.float32, copy=False)).unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(mask_t, size=(int(out_h), int(out_w)), mode="nearest")[0, 0]
    return (resized > 0.0).cpu().numpy().astype(np.bool_, copy=False)


class GroundedSAM2MaskStore:
    """Loads per-concept binary masks produced by GroundedSAM2 from a manifest.json."""

    def __init__(
        self,
        manifest_path: str,
        valid_concepts: Iterable[str],
        min_score: Optional[float] = None,
        cache_max_entries: int = 16,
    ):
        self.manifest_path = os.path.abspath(str(manifest_path))
        self.manifest_root = os.path.dirname(self.manifest_path)
        self.valid_concepts = set(valid_concepts)
        self.min_score = None if min_score is None else float(min_score)
        self.cache_max_entries = max(int(cache_max_entries), 0)

        payload = load_json(self.manifest_path)
        records = payload.get("records", [])
        if not isinstance(records, list):
            raise ValueError(f"Expected manifest records list in {self.manifest_path}")

        by_image: Dict[int, List[dict]] = {}
        for rec in records:
            if not isinstance(rec, dict):
                continue
            image_idx = _parse_manifest_image_index(rec.get("image_id"))
            if image_idx is None:
                continue
            label = data_utils.format_concept(str(rec.get("label", "")))
            if label not in self.valid_concepts:
                continue
            rec = dict(rec)
            rec["label"] = label
            by_image.setdefault(int(image_idx), []).append(rec)
        self._by_image = by_image

        self._bundle_cache: "OrderedDict[str, np.lib.npyio.NpzFile]" = OrderedDict()

    def _resolve_npz_path(self, bundle_npz: str) -> str:
        bundle_npz = str(bundle_npz)
        if os.path.isabs(bundle_npz):
            return bundle_npz
        return os.path.join(self.manifest_root, bundle_npz)

    def _load_bundle(self, npz_path: str):
        npz_path = self._resolve_npz_path(npz_path)
        cached = self._bundle_cache.get(npz_path)
        if cached is not None:
            self._bundle_cache.move_to_end(npz_path)
            return cached
        bundle = np.load(npz_path, allow_pickle=False)
        if self.cache_max_entries > 0:
            self._bundle_cache[npz_path] = bundle
            self._bundle_cache.move_to_end(npz_path)
            while len(self._bundle_cache) > self.cache_max_entries:
                _old_path, old_bundle = self._bundle_cache.popitem(last=False)
                try:
                    old_bundle.close()
                except Exception:
                    pass
        return bundle

    def get_masks(
        self,
        idx: int,
        image_size: Tuple[int, int],
        target_h: int,
        target_w: int,
        concept_pos: Dict[str, int],
        interpolate_to_full_image: bool,
    ) -> np.ndarray:
        records = self._by_image.get(int(idx), [])
        if not records:
            return np.zeros((len(concept_pos), int(target_h), int(target_w)), dtype=np.bool_)
        num_concepts = len(concept_pos)
        out = np.zeros((num_concepts, int(target_h), int(target_w)), dtype=np.bool_)
        width, height = int(image_size[0]), int(image_size[1])
        for rec in records:
            score = rec.get("sam2_score", rec.get("score", None))
            if score is not None and self.min_score is not None and float(score) < self.min_score:
                continue
            concept = rec.get("label")
            if not isinstance(concept, str):
                continue
            concept_idx = concept_pos.get(concept)
            if concept_idx is None:
                continue
            bundle_npz = rec.get("bundle_npz")
            mask_index = rec.get("mask_index")
            if bundle_npz is None or mask_index is None:
                continue
            try:
                mask_index = int(mask_index)
            except Exception:
                continue
            bundle = self._load_bundle(str(bundle_npz))
            masks = bundle.get("masks")
            if masks is None or mask_index < 0 or mask_index >= int(getattr(masks, "shape", [0])[0]):
                continue
            mask = masks[mask_index]
            if mask.ndim != 2:
                continue
            mask = mask.astype(np.bool_, copy=False)

            # Align mask to expected reference space.
            if interpolate_to_full_image:
                if mask.shape != (height, width):
                    mask = _resize_mask_nearest(mask, out_h=height, out_w=width)
                if (target_h, target_w) != (height, width):
                    mask = _resize_mask_nearest(mask, out_h=target_h, out_w=target_w)
            else:
                # Downsample to concept-map grid resolution.
                mask = _downsample_mask_to_grid(mask, out_h=target_h, out_w=target_w)

            out[concept_idx] |= mask
        return out

def _normalize_concept_text(text: str) -> str:
    return data_utils.format_concept(str(text)).lower().replace("-", " ")


def infer_cub_parts_for_concept(concept: str) -> Set[int]:
    text = _normalize_concept_text(concept)
    matched: Set[int] = set()
    keyword_map = {
        1: ["back", "upperparts", "upper parts", "mantle"],
        2: ["beak", "bill", "hooked bill", "hooked beak"],
        3: ["belly", "underparts", "under parts", "abdomen", "underside", "venter"],
        4: ["breast", "chest"],
        5: ["crown", "cap", "head"],
        6: ["forehead", "brow"],
        7: ["left eye", "eye"],
        8: ["left leg", "leg", "foot", "feet", "claw", "talon"],
        9: ["left wing", "wing", "wings", "wingbar", "wing bar"],
        10: ["nape", "hindneck", "back of neck", "neck"],
        11: ["right eye", "eye"],
        12: ["right leg", "leg", "foot", "feet", "claw", "talon"],
        13: ["right wing", "wing", "wings", "wingbar", "wing bar"],
        14: ["tail"],
        15: ["throat", "chin", "gular"],
    }
    for part_id, keywords in keyword_map.items():
        if any(keyword in text for keyword in keywords):
            matched.add(part_id)
    return matched


class CUBPartStore:
    def __init__(self, metadata_root: str, raw_dataset):
        self.metadata_root = metadata_root
        self.raw_dataset = raw_dataset
        self._idx_to_points: Dict[int, Dict[int, Tuple[float, float]]] = {}
        self._idx_to_visible_count: Dict[int, int] = {}

        images_txt = os.path.join(metadata_root, "images.txt")
        part_locs_txt = os.path.join(metadata_root, "parts", "part_locs.txt")
        if not os.path.exists(images_txt) or not os.path.exists(part_locs_txt):
            raise FileNotFoundError(
                f"CUB metadata root {metadata_root} is missing images.txt or parts/part_locs.txt."
            )

        relpath_to_image_id: Dict[str, int] = {}
        with open(images_txt, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                image_id_str, relpath = line.split(" ", 1)
                relpath_to_image_id[relpath] = int(image_id_str)

        image_id_to_points: Dict[int, Dict[int, Tuple[float, float]]] = {}
        image_id_to_visible_count: Dict[int, int] = {}
        with open(part_locs_txt, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                image_id_str, part_id_str, x_str, y_str, visible_str = line.split()
                image_id = int(image_id_str)
                part_id = int(part_id_str)
                visible = int(visible_str)
                if visible != 1:
                    continue
                image_id_to_points.setdefault(image_id, {})[part_id] = (float(x_str), float(y_str))
                image_id_to_visible_count[image_id] = image_id_to_visible_count.get(image_id, 0) + 1

        root = getattr(raw_dataset, "root", None)
        if root is None:
            raise ValueError("Expected raw_dataset to expose .root for CUB part evaluation.")

        for sample_idx, sample in enumerate(getattr(raw_dataset, "samples", [])):
            path = sample[0]
            relpath = os.path.relpath(path, root)
            image_id = relpath_to_image_id.get(relpath)
            if image_id is None:
                continue
            self._idx_to_points[sample_idx] = image_id_to_points.get(image_id, {})
            self._idx_to_visible_count[sample_idx] = image_id_to_visible_count.get(image_id, 0)

    def get(self, idx: int) -> Dict[int, Tuple[float, float]]:
        return self._idx_to_points.get(int(idx), {})

    def visible_count(self, idx: int) -> int:
        return self._idx_to_visible_count.get(int(idx), 0)


def evaluate_model(
    model: SpatialMapModel,
    raw_dataset,
    eval_concepts: Sequence[str],
    annotation_store: AnnotationStore,
    mask_store: Optional[GroundedSAM2MaskStore],
    activation_thresholds: Sequence[float],
    threshold_mode: str,
    map_normalization: str,
    box_iou_thresholds: Sequence[float],
    batch_size: int,
    num_workers: int,
    max_images: Optional[int],
    interpolate_to_full_image: bool,
    gt_cache_max_entries: int,
    map_source: str,
    gradcam_chunk_size: int,
    point_store: Optional[CUBPartStore],
    concept_score_threshold: float,
    topk_concepts_per_image: Optional[int],
    eval_subset_mode: str,
    gt_cache: Optional[Dict[Tuple[int, int, int], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]] = None,
    apply_gating: bool = False,
    gating_mode: str = "hard",
) -> dict:
    logger.info("Starting model eval for {} ({})", model.name, model.model_name)
    if apply_gating:
        if model.eval_kind == "savlg":
            logger.info("Gating enabled: mode={}, applying global-to-spatial gating", gating_mode)
        else:
            logger.warning("Gating requested but model is not SAVLG ({}), gating will be skipped", model.eval_kind)
    concept_indices = torch.tensor([model.concept_to_idx[c] for c in eval_concepts], dtype=torch.long)
    concept_pos = {concept: idx for idx, concept in enumerate(eval_concepts)}
    concept_part_ids = [infer_cub_parts_for_concept(concept) for concept in eval_concepts]
    proj_mean = None
    proj_std = None
    if map_normalization in {"proj_zscore_minmax", "concept_zscore_minmax"}:
        if model.proj_mean is None or model.proj_std is None:
            raise RuntimeError(
                f"Model {model.name} is missing proj_mean.pt/proj_std.pt required for map_normalization=proj_zscore_minmax."
            )
        proj_mean = model.proj_mean.index_select(0, concept_indices).view(1, -1, 1, 1)
        proj_std = model.proj_std.index_select(0, concept_indices).view(1, -1, 1, 1)
    if gt_cache is None:
        gt_cache = OrderedDict()

    labels_rows: List[np.ndarray] = []
    scores_rows = [
        [[] for _ in box_iou_thresholds]
        for _ in activation_thresholds
    ]
    localized_positive_counts = [
        [0 for _ in box_iou_thresholds]
        for _ in activation_thresholds
    ]
    positive_gt_counts = [
        [0 for _ in box_iou_thresholds]
        for _ in activation_thresholds
    ]
    iou_sums = [0.0 for _ in activation_thresholds]
    iou_counts = [0 for _ in activation_thresholds]
    seg_correct = [0 for _ in activation_thresholds]
    seg_total = [0 for _ in activation_thresholds]
    seg_intersections = [np.zeros((2,), dtype=np.float64) for _ in activation_thresholds]
    seg_unions = [np.zeros((2,), dtype=np.float64) for _ in activation_thresholds]
    seg_ap_scores: List[float] = []
    seg_foreground_ap_scores: List[float] = []
    concept_region_count = 0
    mass_in_gt_sum = 0.0
    point_hit_sum = 0.0
    mask_iou_at_0p5_sum = 0.0
    point_hits = [0 for _ in activation_thresholds]
    point_total = [0 for _ in activation_thresholds]
    point_visible_total = 0

    dataset = IndexedPreprocessDataset(raw_dataset, model.preprocess)
    if max_images is not None:
        indices = list(range(min(max_images, len(dataset))))
        dataset = torch.utils.data.Subset(dataset, indices)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_indexed,
    )
    logger.info(
        "{} dataloader ready: batches={} batch_size={} num_workers={}",
        model.name,
        len(loader),
        batch_size,
        num_workers,
    )

    for images, sample_indices, widths, heights in tqdm(loader, desc=f"{model.name} map eval"):
        if map_source == "native":
            if apply_gating and model.eval_kind == "savlg":
                # Use predict_maps_with_global to get both spatial and global logits
                raw_maps_full, global_logits_full = model.predict_maps_with_global(images)
                # Select concepts
                raw_maps = raw_maps_full.index_select(1, concept_indices.to(model.args.device))
                global_logits = global_logits_full.index_select(1, concept_indices.to(model.args.device))
                # Apply gating
                raw_maps = apply_global_spatial_gating(raw_maps, global_logits, gating_mode=gating_mode)
                score_batch = torch.sigmoid(raw_maps).amax(dim=(2, 3)).cpu().numpy()
            else:
                raw_maps = model.predict_maps(images).index_select(1, concept_indices.to(model.args.device))
                score_batch = raw_maps.amax(dim=(2, 3)).cpu().numpy()
        elif map_source == "gradcam":
            raw_maps, logits = predict_gradcam_maps(
                model=model,
                images=images,
                concept_indices=concept_indices,
                gradcam_chunk_size=gradcam_chunk_size,
            )
            score_batch = torch.sigmoid(logits).cpu().numpy()
        else:
            raise ValueError(f"Unsupported map_source={map_source}")
        maps = normalize_eval_maps(
            raw_maps,
            map_normalization=map_normalization,
            proj_mean=proj_mean,
            proj_std=proj_std,
        ).cpu()
        batch_size_now, num_concepts, map_h, map_w = maps.shape

        for batch_idx in range(batch_size_now):
            image_idx = int(sample_indices[batch_idx].item())
            image_size = (int(widths[batch_idx].item()), int(heights[batch_idx].item()))
            point_visible_total += int(point_store.visible_count(image_idx)) if point_store is not None else 0
            target_h = image_size[1] if interpolate_to_full_image else map_h
            target_w = image_size[0] if interpolate_to_full_image else map_w
            cache_key = (image_idx, target_h, target_w)
            use_cache = gt_cache_max_entries > 0
            cached = gt_cache.get(cache_key) if use_cache else None
            if cached is None:
                if mask_store is not None:
                    gt_masks = mask_store.get_masks(
                        image_idx,
                        image_size=image_size,
                        target_h=target_h,
                        target_w=target_w,
                        concept_pos=concept_pos,
                        interpolate_to_full_image=interpolate_to_full_image,
                    )
                else:
                    gt_boxes = annotation_store.get(image_idx)
                    gt_masks = np.zeros((num_concepts, target_h, target_w), dtype=np.bool_)
                    for concept, boxes in gt_boxes.items():
                        concept_idx = concept_pos.get(concept)
                        if concept_idx is None:
                            continue
                        mask = rasterize_box_union(boxes, image_size=image_size, map_h=target_h, map_w=target_w)
                        if mask.any():
                            gt_masks[concept_idx] = mask
                gt_present = gt_masks.reshape(num_concepts, -1).any(axis=1)
                gt_tight_boxes, gt_box_valid = tight_boxes_from_masks(gt_masks)
                cached = (gt_masks, gt_present, gt_tight_boxes, gt_box_valid)
                if use_cache:
                    gt_cache[cache_key] = cached
                    if isinstance(gt_cache, OrderedDict):
                        gt_cache.move_to_end(cache_key)
                        while len(gt_cache) > int(gt_cache_max_entries):
                            gt_cache.popitem(last=False)
            elif use_cache and isinstance(gt_cache, OrderedDict):
                gt_cache.move_to_end(cache_key)
            gt_masks, gt_present, gt_tight_boxes, gt_box_valid = cached

            sample_scores = score_batch[batch_idx]
            labels_rows.append(gt_present.astype(np.int8, copy=True))
            if eval_subset_mode == "gt_present":
                eval_indices = np.flatnonzero(gt_present).astype(np.int64)
            elif (
                eval_subset_mode == "topk"
                and topk_concepts_per_image is not None
                and 0 < int(topk_concepts_per_image) < num_concepts
            ):
                topk = int(topk_concepts_per_image)
                eval_indices = np.argpartition(sample_scores, -topk)[-topk:]
                eval_indices = eval_indices[np.argsort(sample_scores[eval_indices])[::-1]]
            else:
                eval_indices = np.arange(num_concepts, dtype=np.int64)
            eval_indices_t = torch.as_tensor(eval_indices, dtype=torch.long)

            if eval_indices.size > 0:
                sample_maps_t = maps[batch_idx : batch_idx + 1].index_select(1, eval_indices_t)
                sample_raw_maps_t = raw_maps[batch_idx : batch_idx + 1].index_select(
                    1, eval_indices_t.to(raw_maps.device)
                )
                if interpolate_to_full_image and (target_h != map_h or target_w != map_w):
                    sample_maps_t = F.interpolate(
                        sample_maps_t,
                        size=(target_h, target_w),
                        mode="bilinear",
                        align_corners=False,
                    )
                    sample_raw_maps_t = F.interpolate(
                        sample_raw_maps_t,
                        size=(target_h, target_w),
                        mode="bilinear",
                        align_corners=False,
                    )
                sample_maps = sample_maps_t[0].detach().numpy()
                sample_raw_maps = sample_raw_maps_t[0].detach().cpu().numpy()
                gt_masks_eval = gt_masks[eval_indices]
                gt_present_eval = gt_present[eval_indices]
                gt_tight_boxes_eval = gt_tight_boxes[eval_indices]
                gt_box_valid_eval = gt_box_valid[eval_indices]
            else:
                sample_maps = np.zeros((0, target_h, target_w), dtype=np.float32)
                sample_raw_maps = np.zeros((0, target_h, target_w), dtype=np.float32)
                gt_masks_eval = np.zeros((0, target_h, target_w), dtype=np.bool_)
                gt_present_eval = np.zeros((0,), dtype=np.bool_)
                gt_tight_boxes_eval = np.zeros((0, 4), dtype=np.int32)
                gt_box_valid_eval = np.zeros((0,), dtype=np.bool_)
            if gt_present_eval.any():
                positive_score_maps = sample_maps[gt_present_eval]
                positive_raw_maps = sample_raw_maps[gt_present_eval]
                positive_gt_masks = gt_masks_eval[gt_present_eval]
                for score_map, raw_map, gt_mask in zip(
                    positive_score_maps, positive_raw_maps, positive_gt_masks
                ):
                    seg_ap = segmentation_average_precision(gt_mask, score_map)
                    if seg_ap is not None:
                        seg_ap_scores.append(float(seg_ap))
                    fg_ap = average_precision_binary(
                        gt_mask.astype(np.int32, copy=False).reshape(-1).tolist(),
                        score_map.astype(np.float32, copy=False).reshape(-1).tolist(),
                    )
                    if fg_ap is not None:
                        seg_foreground_ap_scores.append(float(fg_ap))
                    pred_dist = spatial_distribution_from_map(raw_map)
                    mass_in_gt_sum += float((pred_dist * gt_mask.astype(np.float64, copy=False)).sum())
                    peak_idx = int(pred_dist.reshape(-1).argmax())
                    point_hit_sum += float(gt_mask.reshape(-1)[peak_idx])
                    mask_iou_at_0p5_sum += mask_iou_from_score_map(score_map, gt_mask, 0.5)
                    concept_region_count += 1
            for thr_idx, activation_threshold in enumerate(activation_thresholds):
                if threshold_mode == "fixed":
                    pred_masks = sample_maps >= float(activation_threshold)
                elif threshold_mode == "percentile":
                    q = float(activation_threshold) / 100.0
                    q = min(max(q, 0.0), 1.0)
                    cutoff = np.quantile(sample_maps, q=q, axis=(1, 2), keepdims=True)
                    pred_masks = sample_maps >= cutoff
                elif threshold_mode == "mean":
                    cutoff = sample_maps.mean(axis=(1, 2), keepdims=True)
                    pred_masks = sample_maps >= cutoff
                else:
                    raise ValueError(f"Unsupported threshold_mode={threshold_mode}")

                if pred_masks.shape[0] == 0:
                    continue

                if point_store is not None:
                    points = point_store.get(image_idx)
                    if points:
                        concept_present = sample_scores >= float(concept_score_threshold)
                        for part_id, (x_px, y_px) in points.items():
                            matched_concepts = [
                                local_idx
                                for local_idx, concept_idx in enumerate(eval_indices.tolist())
                                if concept_present[concept_idx] and part_id in concept_part_ids[concept_idx]
                            ]
                            if not matched_concepts:
                                continue
                            point_total[thr_idx] += 1
                            x_idx = int(np.clip(np.floor(float(x_px) / max(image_size[0], 1) * target_w), 0, target_w - 1))
                            y_idx = int(np.clip(np.floor(float(y_px) / max(image_size[1], 1) * target_h), 0, target_h - 1))
                            hit = any(bool(pred_masks[cidx, y_idx, x_idx]) for cidx in matched_concepts)
                            if hit:
                                point_hits[thr_idx] += 1

                if gt_present.any():
                    positive_masks = pred_masks[gt_present_eval]
                    positive_gt = gt_masks_eval[gt_present_eval]
                    intersection = np.logical_and(positive_masks, positive_gt).sum(axis=(1, 2)).astype(np.float32)
                    union = np.logical_or(positive_masks, positive_gt).sum(axis=(1, 2)).astype(np.float32)
                    ious = np.where(union > 0.0, intersection / np.maximum(union, 1.0), 0.0)
                    iou_sums[thr_idx] += float(ious.sum())
                    iou_counts[thr_idx] += int(gt_present.sum())

                    seg_correct[thr_idx] += int((positive_masks == positive_gt).sum())
                    seg_total[thr_idx] += int(positive_gt.size)
                    fg_inter = np.logical_and(positive_masks, positive_gt).sum(dtype=np.float64)
                    fg_union = np.logical_or(positive_masks, positive_gt).sum(dtype=np.float64)
                    bg_pred = ~positive_masks
                    bg_gt = ~positive_gt
                    bg_inter = np.logical_and(bg_pred, bg_gt).sum(dtype=np.float64)
                    bg_union = np.logical_or(bg_pred, bg_gt).sum(dtype=np.float64)
                    seg_intersections[thr_idx] += np.asarray([bg_inter, fg_inter], dtype=np.float64)
                    seg_unions[thr_idx] += np.asarray([bg_union, fg_union], dtype=np.float64)

                pred_boxes, pred_box_valid = tight_boxes_from_masks(pred_masks)
                iou_to_gt = box_iou_vectorized(
                    pred_boxes,
                    pred_box_valid,
                    gt_tight_boxes_eval,
                    gt_box_valid_eval,
                )
                for tau_idx, box_iou_threshold in enumerate(box_iou_thresholds):
                    threshold = float(box_iou_threshold)
                    localized = np.zeros((num_concepts,), dtype=np.float32)
                    negative_with_pred = pred_box_valid & (~gt_present_eval)
                    localized[eval_indices[negative_with_pred]] = sample_scores[eval_indices[negative_with_pred]].astype(np.float32)
                    positive_localized = pred_box_valid & gt_present_eval & (iou_to_gt >= threshold)
                    localized_positive_counts[thr_idx][tau_idx] += int(positive_localized.sum())
                    positive_gt_counts[thr_idx][tau_idx] += int(gt_present_eval.sum())
                    localized[eval_indices[positive_localized]] = sample_scores[eval_indices[positive_localized]].astype(np.float32)
                    scores_rows[thr_idx][tau_idx].append(localized)

    labels_matrix = np.stack(labels_rows, axis=0) if labels_rows else np.zeros((0, len(eval_concepts)), dtype=np.int8)

    mean_iou = {}
    for thr_idx, threshold in enumerate(activation_thresholds):
        key = format_threshold_key(threshold, threshold_mode)
        mean_iou[key] = (
            float(iou_sums[thr_idx]) / float(max(iou_counts[thr_idx], 1))
        )

    segmentation_pixel_accuracy = {}
    segmentation_mean_iou = {}
    segmentation_foreground_iou = {}
    for thr_idx, threshold in enumerate(activation_thresholds):
        key = format_threshold_key(threshold, threshold_mode)
        segmentation_pixel_accuracy[key] = (
            float(seg_correct[thr_idx]) / float(max(seg_total[thr_idx], 1))
        )
        class_ious = seg_intersections[thr_idx] / np.maximum(seg_unions[thr_idx], 1.0)
        segmentation_mean_iou[key] = float(class_ious.mean())
        segmentation_foreground_iou[key] = float(class_ious[1])

    map_summary = {}
    box_acc_summary = {}
    per_concept_ap = {}
    for thr_idx, activation_threshold in enumerate(activation_thresholds):
        key = format_threshold_key(activation_threshold, threshold_mode)
        map_summary[key] = {}
        box_acc_summary[key] = {}
        per_concept_ap[key] = {}
        for tau_idx, box_iou_threshold in enumerate(box_iou_thresholds):
            if scores_rows[thr_idx][tau_idx]:
                score_matrix = np.stack(scores_rows[thr_idx][tau_idx], axis=0)
            else:
                score_matrix = np.zeros((0, len(eval_concepts)), dtype=np.float32)
            concept_aps = {}
            for concept_idx, concept in enumerate(eval_concepts):
                ap = average_precision_binary(
                    labels_matrix[:, concept_idx].tolist(),
                    score_matrix[:, concept_idx].tolist(),
                )
                if ap is not None:
                    concept_aps[concept] = float(ap)
            per_concept_ap[key][str(box_iou_threshold)] = concept_aps
            map_summary[key][str(box_iou_threshold)] = (
                float(np.mean(list(concept_aps.values()))) if concept_aps else 0.0
            )
            box_acc_summary[key][str(box_iou_threshold)] = (
                float(localized_positive_counts[thr_idx][tau_idx]) / float(max(positive_gt_counts[thr_idx][tau_idx], 1))
            )

    max_box_acc_summary = {}
    if activation_thresholds:
        for tau in box_iou_thresholds:
            tau_key = str(tau)
            best_threshold_key = None
            best_box_acc = -1.0
            for threshold in activation_thresholds:
                threshold_key = format_threshold_key(threshold, threshold_mode)
                box_acc = float(box_acc_summary[threshold_key][tau_key])
                if box_acc > best_box_acc:
                    best_box_acc = box_acc
                    best_threshold_key = threshold_key
            max_box_acc_summary[tau_key] = {
                "box_acc": float(max(best_box_acc, 0.0)),
                "best_activation_threshold": best_threshold_key,
            }

    gt_frequency = {
        concept: int(labels_matrix[:, idx].sum())
        for idx, concept in enumerate(eval_concepts)
    }
    point_localization = {}
    for thr_idx, threshold in enumerate(activation_thresholds):
        key = format_threshold_key(threshold, threshold_mode)
        point_localization[key] = {
            "hit_rate": float(point_hits[thr_idx]) / float(max(point_total[thr_idx], 1)),
            "matched_part_hits": int(point_hits[thr_idx]),
            "matched_part_total": int(point_total[thr_idx]),
            "visible_part_total": int(point_visible_total),
            "coverage": float(point_total[thr_idx]) / float(max(point_visible_total, 1)),
            "concept_score_threshold": float(concept_score_threshold),
            "topk_concepts_per_image": (
                int(topk_concepts_per_image) if topk_concepts_per_image is not None else None
            ),
            "eval_subset_mode": str(eval_subset_mode),
        }
    logger.info("Finished model eval for {}", model.name)
    return {
        "model_name": model.model_name,
        "load_path": model.load_path,
        "num_eval_concepts": len(eval_concepts),
        "mean_iou_by_activation_threshold": mean_iou,
        "segmentation_pixel_accuracy_by_activation_threshold": segmentation_pixel_accuracy,
        "segmentation_mean_iou_by_activation_threshold": segmentation_mean_iou,
        "segmentation_foreground_iou_by_activation_threshold": segmentation_foreground_iou,
        "segmentation_map": float(np.mean(seg_ap_scores)) if seg_ap_scores else 0.0,
        "segmentation_foreground_map": (
            float(np.mean(seg_foreground_ap_scores)) if seg_foreground_ap_scores else 0.0
        ),
        "segmentation_ap_count": int(len(seg_ap_scores)),
        "concept_region_metrics": {
            "mass_in_gt": float(mass_in_gt_sum / float(max(concept_region_count, 1))),
            "point_hit": float(point_hit_sum / float(max(concept_region_count, 1))),
            "mask_iou_at_0p5": float(mask_iou_at_0p5_sum / float(max(concept_region_count, 1))),
            "count": int(concept_region_count),
            "mass_source": "softmax(raw_map)",
            "mask_iou_threshold": 0.5,
        },
        "map_by_activation_and_box_iou_threshold": map_summary,
        "box_acc_by_activation_and_box_iou_threshold": box_acc_summary,
        "max_box_acc_by_box_iou_threshold": max_box_acc_summary,
        "per_concept_ap": per_concept_ap,
        "gt_frequency": gt_frequency,
        "point_localization_by_activation_threshold": point_localization,
        "topk_concepts_per_image": (
            int(topk_concepts_per_image) if topk_concepts_per_image is not None else None
        ),
        "eval_subset_mode": str(eval_subset_mode),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate native or Grad-CAM concept maps against GroundingDINO annotations.")
    parser.add_argument("--load_paths", nargs="+", required=True, help="Checkpoint directories to compare.")
    parser.add_argument("--names", nargs="*", default=None, help="Optional display names matching --load_paths order.")
    parser.add_argument("--annotation_dir", type=str, required=True, help="Root annotation directory, e.g. outputs.")
    parser.add_argument("--dataset", type=str, default="cub")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Alias for --max_images. If both are set, the smaller value is used.",
    )
    parser.add_argument("--annotation_threshold", type=float, default=0.15)
    parser.add_argument("--activation_thresholds", type=str, default="0.3,0.5,0.7")
    parser.add_argument(
        "--threshold_mode",
        type=str,
        default="fixed",
        choices=["fixed", "percentile", "mean"],
        help="How activation thresholds are interpreted: fixed values, per-map percentiles, or per-map mean-value thresholding.",
    )
    parser.add_argument(
        "--map_source",
        type=str,
        default="native",
        choices=["native", "gradcam"],
        help="Use native spatial concept maps or concept-logit Grad-CAM maps.",
    )
    parser.add_argument(
        "--map_normalization",
        type=str,
        default="sigmoid",
        choices=["sigmoid", "minmax", "proj_zscore_minmax", "concept_zscore_minmax"],
        help="Map normalization before thresholding. concept_zscore_minmax uses per-concept proj_mean/proj_std, then per-map min-max.",
    )
    parser.add_argument(
        "--gradcam_chunk_size",
        type=int,
        default=32,
        help="Number of concepts per autograd chunk when map_source=gradcam.",
    )
    parser.add_argument(
        "--interpolate_to_full_image",
        action="store_true",
        help="If set, upsample each concept map to the original image size before IoU/AP computation.",
    )
    parser.add_argument(
        "--gt_cache_max_entries",
        type=int,
        default=0,
        help="Max number of GT mask entries to cache in-memory. 0 disables GT mask caching (recommended for full-image eval).",
    )
    parser.add_argument(
        "--gt_source",
        type=str,
        default="gdino_boxes",
        choices=["gdino_boxes", "groundedsam2_masks"],
        help="Ground-truth localization geometry source. gdino_boxes rasterizes annotation boxes; groundedsam2_masks loads per-concept masks from a GroundedSAM2 manifest.json.",
    )
    parser.add_argument(
        "--groundedsam2_manifest",
        type=str,
        default=None,
        help="Path to GroundedSAM2 manifest.json (required when --gt_source=groundedsam2_masks).",
    )
    parser.add_argument(
        "--groundedsam2_min_score",
        type=float,
        default=None,
        help="If set, ignore GroundedSAM2 records with sam2_score below this threshold.",
    )
    parser.add_argument(
        "--groundedsam2_bundle_cache_max_entries",
        type=int,
        default=16,
        help="Max number of per-image NPZ bundles to cache in-memory while reading GroundedSAM2 masks.",
    )
    parser.add_argument(
        "--cub_metadata_root",
        type=str,
        default=None,
        help="Path to CUB_200_2011 metadata root for paper-style part-point localization evaluation.",
    )
    parser.add_argument(
        "--concept_score_threshold",
        type=float,
        default=0.5,
        help="Minimum per-concept score for a concept to be considered predicted in paper-style point localization eval.",
    )
    parser.add_argument(
        "--topk_concepts_per_image",
        type=int,
        default=None,
        help="If set, only materialize maps for the top-k concepts per image by concept score; non-top-k concepts receive zero localization score.",
    )
    parser.add_argument(
        "--eval_subset_mode",
        type=str,
        default="all",
        choices=["all", "topk", "gt_present"],
        help="Per-image concept subset to materialize during eval.",
    )
    parser.add_argument("--box_iou_thresholds", type=str, default="0.3,0.5,0.7")
    parser.add_argument(
        "--concept_mode",
        type=str,
        default="intersection",
        choices=["intersection", "union"],
        help="How to align concepts across checkpoints before evaluation.",
    )
    parser.add_argument("--output", type=str, default=None, help="Optional JSON output path.")
    parser.add_argument(
        "--apply_gating",
        action="store_true",
        help="Apply global-to-spatial gating to prevent absent concept hallucination (SAVLG only).",
    )
    parser.add_argument(
        "--gating_mode",
        type=str,
        default="hard",
        choices=["hard", "soft"],
        help="Gating mode: 'hard' for multiplicative gating, 'soft' for gating with floor=0.2.",
    )
    args = parser.parse_args()

    if args.names is not None and len(args.names) not in (0, len(args.load_paths)):
        raise ValueError("--names must be omitted or match the number of --load_paths.")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive.")
    if args.max_images is not None and args.max_images <= 0:
        raise ValueError("--max_images must be positive.")
    if args.gt_cache_max_entries < 0:
        raise ValueError("--gt_cache_max_entries must be >= 0.")
    if args.gradcam_chunk_size <= 0:
        raise ValueError("--gradcam_chunk_size must be positive.")
    if args.topk_concepts_per_image is not None and args.topk_concepts_per_image <= 0:
        raise ValueError("--topk_concepts_per_image must be positive when set.")
    if args.eval_subset_mode == "topk" and args.topk_concepts_per_image is None:
        raise ValueError("--topk_concepts_per_image is required when --eval_subset_mode=topk.")
    if args.gt_source == "groundedsam2_masks" and not args.groundedsam2_manifest:
        raise ValueError("--groundedsam2_manifest is required when --gt_source=groundedsam2_masks.")
    if args.limit is not None:
        args.max_images = args.limit if args.max_images is None else min(args.max_images, args.limit)
    if args.map_source == "gradcam" and args.map_normalization == "sigmoid":
        logger.warning("map_source=gradcam does not work well with sigmoid normalization; switching to minmax.")
        args.map_normalization = "minmax"

    names = args.names or [None] * len(args.load_paths)
    logger.info("Loading {} checkpoints", len(args.load_paths))
    models = [
        load_spatial_model(load_path=load_path, device=args.device, name=name)
        for load_path, name in zip(args.load_paths, names)
    ]
    if len({model.args.dataset for model in models}) != 1:
        raise ValueError("All checkpoints must come from the same dataset.")
    dataset_name = args.dataset or models[0].args.dataset
    split_dataset_name = f"{dataset_name}_val" if args.split in {"val", "test"} else f"{dataset_name}_train"
    raw_dataset = data_utils.get_data(split_dataset_name, None)

    concept_sets = [set(model.concepts) for model in models]
    if args.concept_mode == "union" and len(models) > 1:
        raise ValueError("concept_mode=union is only supported for a single checkpoint. Use intersection for comparisons.")

    if args.concept_mode == "intersection":
        eval_concepts = sorted(set.intersection(*concept_sets))
    else:
        eval_concepts = sorted(set.union(*concept_sets))
    if not eval_concepts:
        raise RuntimeError("No common concepts found across checkpoints.")

    annotation_store = AnnotationStore(
        annotation_dir=args.annotation_dir,
        dataset=dataset_name,
        split_name=args.split,
        valid_concepts=eval_concepts,
        annotation_threshold=args.annotation_threshold,
    )
    mask_store = None
    if args.gt_source == "groundedsam2_masks":
        mask_store = GroundedSAM2MaskStore(
            manifest_path=args.groundedsam2_manifest,
            valid_concepts=eval_concepts,
            min_score=args.groundedsam2_min_score,
            cache_max_entries=args.groundedsam2_bundle_cache_max_entries,
        )
        logger.info("GroundedSAM2 mask store ready from {}", os.path.abspath(args.groundedsam2_manifest))
    point_store = None
    if dataset_name == "cub":
        cub_metadata_root = resolve_cub_metadata_root(args.cub_metadata_root)
        if cub_metadata_root is not None:
            point_store = CUBPartStore(metadata_root=cub_metadata_root, raw_dataset=raw_dataset)
            logger.info("CUB point-localization metadata ready from {}", cub_metadata_root)
        else:
            logger.warning("CUB metadata root not found; paper-style point localization eval will be skipped.")
    logger.info(
        "Annotation store ready (lazy loading) from {}",
        annotation_store.split_dir,
    )

    activation_thresholds = parse_thresholds(args.activation_thresholds)
    box_iou_thresholds = parse_thresholds(args.box_iou_thresholds)
    if args.threshold_mode == "percentile":
        for t in activation_thresholds:
            if not (0.0 <= float(t) <= 100.0):
                raise ValueError("For threshold_mode=percentile, activation_thresholds must be in [0, 100].")
    elif args.threshold_mode == "mean":
        # Mean-threshold mode binarizes each concept map using its own mean value,
        # matching the segmentation-thresholding protocol used in Chefer et al. (CVPR 2021).
        activation_thresholds = [0.0]

    results = {
        "dataset": dataset_name,
        "split": args.split,
        "annotation_dir": args.annotation_dir,
        "annotation_threshold": float(args.annotation_threshold),
        "activation_thresholds": activation_thresholds,
        "threshold_mode": args.threshold_mode,
        "map_source": args.map_source,
        "map_normalization": args.map_normalization,
        "gradcam_chunk_size": int(args.gradcam_chunk_size),
        "interpolate_to_full_image": bool(args.interpolate_to_full_image),
        "gt_cache_max_entries": int(args.gt_cache_max_entries),
        "gt_source": str(args.gt_source),
        "groundedsam2_manifest": os.path.abspath(args.groundedsam2_manifest) if args.groundedsam2_manifest else None,
        "groundedsam2_min_score": float(args.groundedsam2_min_score) if args.groundedsam2_min_score is not None else None,
        "box_iou_thresholds": box_iou_thresholds,
        "concept_mode": args.concept_mode,
        "num_images": min(len(raw_dataset), args.max_images) if args.max_images is not None else len(raw_dataset),
        "num_eval_concepts": len(eval_concepts),
        "eval_concepts": eval_concepts,
        "models": {},
    }

    logger.info(
        "Evaluating {} checkpoints on {} images with {} shared annotated concepts",
        len(models),
        results["num_images"],
        len(eval_concepts),
    )

    gt_cache: Dict[Tuple[int, int, int], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for model in models:
        missing = [concept for concept in eval_concepts if concept not in model.concept_to_idx]
        if missing:
            raise RuntimeError(f"Model {model.name} is missing {len(missing)} eval concepts.")
        results["models"][model.name] = evaluate_model(
            model=model,
            raw_dataset=raw_dataset,
            eval_concepts=eval_concepts,
            annotation_store=annotation_store,
            mask_store=mask_store,
            activation_thresholds=activation_thresholds,
            threshold_mode=args.threshold_mode,
            map_normalization=args.map_normalization,
            box_iou_thresholds=box_iou_thresholds,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_images=args.max_images,
            interpolate_to_full_image=bool(args.interpolate_to_full_image),
            gt_cache_max_entries=int(args.gt_cache_max_entries),
            map_source=args.map_source,
            gradcam_chunk_size=int(args.gradcam_chunk_size),
            point_store=point_store,
            concept_score_threshold=float(args.concept_score_threshold),
            topk_concepts_per_image=args.topk_concepts_per_image,
            eval_subset_mode=args.eval_subset_mode,
            gt_cache=gt_cache,
            apply_gating=bool(args.apply_gating),
            gating_mode=args.gating_mode,
        )

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Wrote evaluation summary to {}", args.output)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
