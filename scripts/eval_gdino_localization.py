#!/usr/bin/env python3
"""Unified GDINO-box localization evaluation for CUB and ImageNet SG-CBM runs."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import tarfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gcbm.imagenet_eval import (  # noqa: E402
    VAL_RE,
    load_run_config,
    resolve_source_run_dir,
)
from gcbm.imagenet_annotation_index import (  # noqa: E402
    build_filename_to_annotation_path,
    load_annotation_payload,
    resolve_val_annotation_dir,
)
from gcbm.imagenet_config import Config  # noqa: E402
from gcbm.imagenet_targets import build_gdino_targets, load_concepts  # noqa: E402
from gcbm.runtime import configure_runtime  # noqa: E402
from gcbm.training_utils import prepare_images  # noqa: E402
from gcbm.imagenet_models import build_model  # noqa: E402

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate SG-CBM native spatial maps against GDINO pseudo boxes on CUB or ImageNet."
    )
    parser.add_argument("--dataset", required=True, choices=["cub", "imagenet"])
    parser.add_argument("--gcbm_path", required=True, help="Path to a trained SG-CBM run directory.")
    parser.add_argument("--annotation_dir", required=True, help="Directory containing GDINO annotation JSONs.")
    parser.add_argument("--output", required=True, help="Output JSON path.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", "--workers", dest="num_workers", type=int, default=0)
    parser.add_argument("--max_images", type=int, default=0, help="Optional cap on evaluated images. 0 means full split.")
    parser.add_argument("--activation_thresholds", default="0.3,0.5,0.7,0.9")
    parser.add_argument("--box_iou_thresholds", default="0.1,0.3,0.5")
    parser.add_argument(
        "--threshold_mode",
        default="fixed",
        choices=["fixed", "percentile", "mean"],
        help="fixed thresholds, per-map percentiles, or per-map mean thresholding.",
    )
    parser.add_argument(
        "--map_normalization",
        default="concept_zscore_minmax",
        choices=["minmax", "sigmoid", "proj_zscore_minmax", "concept_zscore_minmax"],
        help="CUB concept_zscore_minmax uses saved proj_mean/proj_std; ImageNet uses per-map zscore then min-max.",
    )
    parser.add_argument("--annotation_threshold", type=float, default=0.15)
    parser.add_argument("--log_every", type=int, default=1000)

    imagenet = parser.add_argument_group("ImageNet inputs")
    imagenet.add_argument("--val_root", default="", help="Extracted ImageNet val root, flat or ImageFolder-style.")
    imagenet.add_argument("--val_tar", default="", help="Official ImageNet val tar. Used when --val_root is not set.")
    imagenet.add_argument(
        "--annotation_val_root",
        default="",
        help="Optional ImageNet val root used to map annotation JSONs by filename.",
    )
    imagenet.add_argument("--prefetch_factor", type=int, default=2)
    imagenet.add_argument("--persistent_workers", action="store_true")
    imagenet.add_argument("--pin_memory", action="store_true")
    return parser.parse_args()


def parse_float_list(raw: str) -> List[float]:
    return [float(x.strip()) for x in str(raw).split(",") if x.strip()]


def threshold_keys(args: argparse.Namespace, thresholds: Sequence[float]) -> List[str]:
    if args.threshold_mode == "mean" or str(args.activation_thresholds).strip().lower() in {"mean", "meanthr"}:
        return ["mean"]
    if args.threshold_mode == "percentile":
        return [f"p{int(t)}" if float(t).is_integer() else f"p{t}" for t in thresholds]
    return [str(t) for t in thresholds]


def init_state(keys: Sequence[str], box_iou_thresholds: Sequence[float]) -> Dict[str, Any]:
    return {
        "images_seen": 0,
        "images_with_targets": 0,
        "instances": 0,
        "distribution": {"soft_iou": 0.0, "mass_in_gt": 0.0, "point_hit": 0.0, "mask_iou_at_0p5": 0.0},
        "thresholds": {
            key: {
                "iou_sum": 0.0,
                "mask_iou_sum": 0.0,
                "dice_sum": 0.0,
                "point_hit_sum": 0.0,
                "coverage_sum": 0.0,
                "count": 0,
                "box_hits": {str(t): 0 for t in box_iou_thresholds},
            }
            for key in keys
        },
    }


def normalize_box(box: Sequence[float], image_size: Tuple[int, int]) -> Optional[Tuple[float, float, float, float]]:
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


def rasterize_box_union(boxes: Sequence[Sequence[float]], image_size: Tuple[int, int], map_h: int, map_w: int) -> np.ndarray:
    norm_boxes = [box for box in (normalize_box(box, image_size) for box in boxes) if box is not None]
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


def tight_boxes_from_masks(masks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if masks.ndim != 3:
        raise ValueError(f"Expected [K,H,W] masks, got {masks.shape}")
    count, map_h, map_w = masks.shape
    valid = masks.reshape(count, -1).any(axis=1)
    boxes = np.zeros((count, 4), dtype=np.float32)
    if count == 0 or not valid.any():
        return boxes, valid
    ys_any = masks.any(axis=2)
    xs_any = masks.any(axis=1)
    boxes[:, 0] = xs_any.argmax(axis=1).astype(np.float32)
    boxes[:, 1] = ys_any.argmax(axis=1).astype(np.float32)
    boxes[:, 2] = (map_w - xs_any[:, ::-1].argmax(axis=1)).astype(np.float32)
    boxes[:, 3] = (map_h - ys_any[:, ::-1].argmax(axis=1)).astype(np.float32)
    return boxes, valid


def box_iou_vectorized(pred_boxes: np.ndarray, pred_valid: np.ndarray, gt_boxes: np.ndarray, gt_valid: np.ndarray) -> np.ndarray:
    ix1 = np.maximum(pred_boxes[:, 0], gt_boxes[:, 0])
    iy1 = np.maximum(pred_boxes[:, 1], gt_boxes[:, 1])
    ix2 = np.minimum(pred_boxes[:, 2], gt_boxes[:, 2])
    iy2 = np.minimum(pred_boxes[:, 3], gt_boxes[:, 3])
    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
    area_pred = np.maximum(0.0, pred_boxes[:, 2] - pred_boxes[:, 0]) * np.maximum(0.0, pred_boxes[:, 3] - pred_boxes[:, 1])
    area_gt = np.maximum(0.0, gt_boxes[:, 2] - gt_boxes[:, 0]) * np.maximum(0.0, gt_boxes[:, 3] - gt_boxes[:, 1])
    union = area_pred + area_gt - inter
    valid = pred_valid & gt_valid & (union > 0.0)
    out = np.zeros_like(inter, dtype=np.float32)
    out[valid] = inter[valid] / union[valid]
    return out


def spatial_distribution_from_map(score_map: np.ndarray) -> np.ndarray:
    flat = score_map.astype(np.float64, copy=False).reshape(-1)
    flat = flat - np.max(flat)
    exp = np.exp(flat)
    denom = float(exp.sum())
    if denom <= 0.0 or not np.isfinite(denom):
        return np.full(score_map.shape, 1.0 / float(max(flat.size, 1)), dtype=np.float64)
    return (exp / denom).reshape(score_map.shape)


def pred_masks_for_threshold(score_maps: np.ndarray, threshold: float, mode: str) -> np.ndarray:
    if mode == "mean":
        cutoff = score_maps.mean(axis=(1, 2), keepdims=True)
        return score_maps >= cutoff
    if mode == "percentile":
        q = min(max(float(threshold) / 100.0, 0.0), 1.0)
        cutoff = np.quantile(score_maps, q=q, axis=(1, 2), keepdims=True)
        return score_maps >= cutoff
    return score_maps >= float(threshold)


def update_metrics(
    state: Dict[str, Any],
    score_maps: np.ndarray,
    raw_maps: np.ndarray,
    gt_masks: np.ndarray,
    thresholds: Sequence[float],
    keys: Sequence[str],
    threshold_mode: str,
    box_iou_thresholds: Sequence[float],
) -> None:
    if score_maps.shape[0] == 0:
        return
    gt_masks = gt_masks.astype(np.bool_, copy=False)
    state["instances"] += int(score_maps.shape[0])

    for score_map, raw_map, gt_mask in zip(score_maps, raw_maps, gt_masks):
        pred_dist = spatial_distribution_from_map(raw_map)
        gt_dist = gt_mask.astype(np.float64, copy=False)
        gt_sum = float(gt_dist.sum())
        if gt_sum > 0:
            gt_dist = gt_dist / gt_sum
        inter = np.minimum(pred_dist, gt_dist).sum()
        union = np.maximum(pred_dist, gt_dist).sum()
        state["distribution"]["soft_iou"] += float(inter / max(union, 1e-12))
        state["distribution"]["mass_in_gt"] += float((pred_dist * gt_mask).sum())
        state["distribution"]["point_hit"] += float(gt_mask.reshape(-1)[int(pred_dist.reshape(-1).argmax())])
        mask_05 = score_map >= 0.5
        mask_union = np.logical_or(mask_05, gt_mask).sum()
        state["distribution"]["mask_iou_at_0p5"] += float(
            np.logical_and(mask_05, gt_mask).sum() / max(mask_union, 1)
        )

    gt_boxes, gt_box_valid = tight_boxes_from_masks(gt_masks)
    for threshold, key in zip(thresholds, keys):
        pred_masks = pred_masks_for_threshold(score_maps, threshold, threshold_mode)
        pred_boxes, pred_box_valid = tight_boxes_from_masks(pred_masks)
        box_ious = box_iou_vectorized(pred_boxes, pred_box_valid, gt_boxes, gt_box_valid)
        inter = np.logical_and(pred_masks, gt_masks).sum(axis=(1, 2)).astype(np.float64)
        union = np.logical_or(pred_masks, gt_masks).sum(axis=(1, 2)).astype(np.float64)
        pred_sum = pred_masks.sum(axis=(1, 2)).astype(np.float64)
        gt_sum = gt_masks.sum(axis=(1, 2)).astype(np.float64)
        mask_iou = np.where(union > 0, inter / np.maximum(union, 1.0), 0.0)
        dice = np.where((pred_sum + gt_sum) > 0, 2.0 * inter / np.maximum(pred_sum + gt_sum, 1.0), 0.0)
        point_hits = []
        coverages = []
        for pred_mask, score_map, gt_mask in zip(pred_masks, score_maps, gt_masks):
            point_hits.append(float(gt_mask.reshape(-1)[int(score_map.reshape(-1).argmax())]))
            coverages.append(float(pred_mask[gt_mask].mean()) if bool(gt_mask.any()) else 0.0)
        bucket = state["thresholds"][key]
        bucket["count"] += int(score_maps.shape[0])
        bucket["iou_sum"] += float(box_ious.sum())
        bucket["mask_iou_sum"] += float(mask_iou.sum())
        bucket["dice_sum"] += float(dice.sum())
        bucket["point_hit_sum"] += float(np.sum(point_hits))
        bucket["coverage_sum"] += float(np.sum(coverages))
        for tau in box_iou_thresholds:
            bucket["box_hits"][str(tau)] += int((box_ious >= float(tau)).sum())


def finalize(state: Dict[str, Any], keys: Sequence[str], box_iou_thresholds: Sequence[float]) -> Dict[str, Any]:
    instances = max(int(state["instances"]), 1)
    threshold_metrics: Dict[str, Any] = {}
    best_mean_iou = {"threshold": None, "value": None}
    best_box_acc = {str(t): {"threshold": None, "value": None} for t in box_iou_thresholds}
    for key in keys:
        bucket = state["thresholds"][key]
        count = max(int(bucket["count"]), 1)
        box_acc = {str(t): float(bucket["box_hits"][str(t)] / count) for t in box_iou_thresholds}
        metric = {
            "mean_iou": float(bucket["iou_sum"] / count),
            "mask_iou": float(bucket["mask_iou_sum"] / count),
            "dice": float(bucket["dice_sum"] / count),
            "point_hit": float(bucket["point_hit_sum"] / count),
            "coverage": float(bucket["coverage_sum"] / count),
            "box_acc": box_acc,
        }
        threshold_metrics[key] = metric
        if best_mean_iou["value"] is None or metric["mask_iou"] > float(best_mean_iou["value"]):
            best_mean_iou = {"threshold": key, "value": metric["mask_iou"]}
        for tau in box_iou_thresholds:
            tau_key = str(tau)
            value = box_acc[tau_key]
            if best_box_acc[tau_key]["value"] is None or value > float(best_box_acc[tau_key]["value"]):
                best_box_acc[tau_key] = {"threshold": key, "value": value}
    return {
        "images_seen": int(state["images_seen"]),
        "images_with_targets": int(state["images_with_targets"]),
        "instances": int(state["instances"]),
        "distribution_metrics": {k: float(v / instances) for k, v in state["distribution"].items()},
        "threshold_metrics": threshold_metrics,
        "best_mean_iou": best_mean_iou,
        "best_box_acc": best_box_acc,
    }


def normalize_imagenet_maps(maps: torch.Tensor, mode: str) -> torch.Tensor:
    maps = maps.float()
    if mode == "sigmoid":
        return torch.sigmoid(maps)
    if mode in {"concept_zscore_minmax", "proj_zscore_minmax"}:
        mean = maps.mean(dim=(1, 2), keepdim=True)
        std = maps.std(dim=(1, 2), keepdim=True, unbiased=False).clamp_min(1e-6)
        maps = (maps - mean) / std
    min_v = maps.amin(dim=(1, 2), keepdim=True)
    max_v = maps.amax(dim=(1, 2), keepdim=True)
    return (maps - min_v) / (max_v - min_v).clamp_min(1e-6)


def normalize_cub_maps(
    maps: torch.Tensor,
    mode: str,
    proj_mean: Optional[torch.Tensor],
    proj_std: Optional[torch.Tensor],
) -> torch.Tensor:
    if mode == "concept_zscore_minmax":
        mode = "proj_zscore_minmax"
    if mode == "sigmoid":
        return torch.sigmoid(maps)
    if mode == "proj_zscore_minmax":
        if proj_mean is None or proj_std is None:
            raise RuntimeError("proj_zscore_minmax requires proj_mean/proj_std.")
        maps = (maps - proj_mean.to(maps.device)) / proj_std.to(maps.device)
    min_v = maps.amin(dim=(2, 3), keepdim=True)
    max_v = maps.amax(dim=(2, 3), keepdim=True)
    return (maps - min_v) / (max_v - min_v).clamp_min(1e-6)


class IndexedPreprocessDataset(Dataset):
    def __init__(self, base_dataset, preprocess) -> None:
        self.base_dataset = base_dataset
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        image, _target = self.base_dataset[idx]
        width, height = image.size
        return self.preprocess(image), idx, width, height


def collate_indexed(batch):
    images = torch.stack([row[0] for row in batch], dim=0)
    indices = torch.tensor([row[1] for row in batch], dtype=torch.long)
    widths = torch.tensor([row[2] for row in batch], dtype=torch.long)
    heights = torch.tensor([row[3] for row in batch], dtype=torch.long)
    return images, indices, widths, heights


def resolve_annotation_split_dir(annotation_root: str, dataset: str, split_name: str) -> Path:
    root = Path(annotation_root)
    candidates = [root / f"{dataset}_{split_name}"]
    if split_name in {"test", "val"}:
        candidates.extend([root / f"{dataset}_val", root / f"{dataset}_test"])
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"Could not find {dataset}_{split_name} annotations under {annotation_root}")


class CubAnnotationStore:
    def __init__(self, annotation_dir: str, valid_concepts: Sequence[str], threshold: float) -> None:
        self.split_dir = resolve_annotation_split_dir(annotation_dir, "cub", "val")
        self.valid_concepts = set(valid_concepts)
        self.threshold = float(threshold)
        self.cache: Dict[int, Dict[str, List[List[float]]]] = {}

    def get(self, idx: int) -> Dict[str, List[List[float]]]:
        cached = self.cache.get(int(idx))
        if cached is not None:
            return cached
        path = self.split_dir / f"{int(idx)}.json"
        out: Dict[str, List[List[float]]] = {}
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            for row in payload[1:] if isinstance(payload, list) else payload.get("concepts", []):
                score = float(row.get("logit", row.get("score", 0.0)))
                if score < self.threshold:
                    continue
                label = str(row.get("label", row.get("name", "")))
                label = " ".join(label.lower().replace("_", " ").split())
                if label not in self.valid_concepts:
                    continue
                box = row.get("box")
                if isinstance(box, list) and len(box) == 4:
                    out.setdefault(label, []).append([float(v) for v in box])
        self.cache[int(idx)] = out
        return out


class ImageNetValDataset(Dataset):
    def __init__(
        self,
        val_root: Path,
        annotation_val_dir: Path,
        annotation_val_root: Optional[Path],
        input_size: int,
    ) -> None:
        try:
            dataset = ImageFolder(str(val_root))
            self.samples = [Path(path) for path, _target in dataset.samples]
        except FileNotFoundError:
            self.samples = sorted(val_root.glob("*.JPEG")) or sorted(val_root.rglob("*.JPEG"))
            if not self.samples:
                raise
        self.annotation_val_dir = annotation_val_dir
        self.filename_to_annotation_path = build_filename_to_annotation_path(annotation_val_dir, annotation_val_root)
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path = self.samples[index]
        image_name = image_path.name
        match = VAL_RE.search(image_name)
        if match is None:
            raise ValueError(f"ImageNet val filename does not match expected pattern: {image_path}")
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image_size = (int(image.size[0]), int(image.size[1]))
            tensor = self.transform(image)
        annotation = load_annotation_payload(
            self.annotation_val_dir,
            int(match.group(1)),
            image_name,
            filename_to_annotation_path=self.filename_to_annotation_path,
        )
        return tensor, annotation, image_size, image_name


def imagenet_collate(batch):
    images, annotations, image_sizes, names = zip(*batch)
    return list(images), list(annotations), list(image_sizes), list(names)


def eval_cub(args: argparse.Namespace, thresholds: Sequence[float], keys: Sequence[str], box_iou_thresholds: Sequence[float]) -> Dict[str, Any]:
    from data import utils as data_utils
    from gcbm.savlg_eval_common import _load_args, _load_concepts
    from methods.salf import SpatialBackbone
    from methods.savlg import build_savlg_concept_layer, forward_savlg_backbone, forward_savlg_concept_layer

    run_args = _load_args(args.gcbm_path, args.device, args.annotation_dir)
    if getattr(run_args, "skip_test_eval", False):
        run_args.skip_test_eval = False
    backbone = SpatialBackbone(
        run_args.backbone,
        device=run_args.device,
        spatial_stage=getattr(run_args, "savlg_spatial_stage", "conv5"),
    )
    concepts = _load_concepts(args.gcbm_path, run_args)
    concept_layer = build_savlg_concept_layer(run_args, backbone, len(concepts)).to(run_args.device)
    concept_layer.load_state_dict(torch.load(Path(args.gcbm_path) / "concept_layer.pt", map_location=run_args.device))
    backbone.eval()
    concept_layer.eval()
    raw_dataset = data_utils.get_data("cub_val", None)
    dataset = IndexedPreprocessDataset(raw_dataset, backbone.preprocess)
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=False,
        collate_fn=collate_indexed,
    )
    annotation_store = CubAnnotationStore(args.annotation_dir, concepts, float(args.annotation_threshold))
    concept_to_idx = {concept: idx for idx, concept in enumerate(concepts)}
    state = init_state(keys, box_iou_thresholds)
    concept_indices_all = torch.arange(len(concepts), dtype=torch.long)
    if args.map_normalization in {"proj_zscore_minmax", "concept_zscore_minmax"}:
        mean_path = Path(args.gcbm_path) / "proj_mean.pt"
        std_path = Path(args.gcbm_path) / "proj_std.pt"
        if not mean_path.exists() or not std_path.exists():
            raise RuntimeError("CUB concept_zscore_minmax requires proj_mean.pt/proj_std.pt in the run directory.")
        proj_mean = torch.load(mean_path, map_location="cpu").float().flatten().clamp(min=-1e12)
        proj_std = torch.load(std_path, map_location="cpu").float().flatten().clamp_min(1e-6)
        proj_mean_all = proj_mean.index_select(0, concept_indices_all).view(1, -1, 1, 1)
        proj_std_all = proj_std.index_select(0, concept_indices_all).view(1, -1, 1, 1)
    else:
        proj_mean_all = None
        proj_std_all = None
    start = time.perf_counter()
    next_log = max(int(args.log_every), 1)
    with torch.no_grad():
        for images, indices, widths, heights in loader:
            if int(args.max_images) > 0 and state["images_seen"] >= int(args.max_images):
                break
            if int(args.max_images) > 0 and state["images_seen"] + images.shape[0] > int(args.max_images):
                keep = int(args.max_images) - state["images_seen"]
                images, indices, widths, heights = images[:keep], indices[:keep], widths[:keep], heights[:keep]
            images = images.to(run_args.device, non_blocking=True)
            feats = forward_savlg_backbone(backbone, images, run_args)
            _global_outputs, raw_maps_full = forward_savlg_concept_layer(concept_layer, feats)
            score_maps_full = normalize_cub_maps(
                raw_maps_full,
                mode=args.map_normalization,
                proj_mean=proj_mean_all,
                proj_std=proj_std_all,
            ).cpu()
            raw_maps_full_cpu = raw_maps_full.detach().cpu()
            for batch_idx, image_idx in enumerate(indices.tolist()):
                gt_boxes = annotation_store.get(int(image_idx))
                concept_indices: List[int] = []
                gt_masks: List[np.ndarray] = []
                map_h, map_w = int(score_maps_full.shape[-2]), int(score_maps_full.shape[-1])
                image_size = (int(widths[batch_idx].item()), int(heights[batch_idx].item()))
                for concept, boxes in gt_boxes.items():
                    concept_idx = concept_to_idx.get(concept)
                    if concept_idx is None:
                        continue
                    mask = rasterize_box_union(boxes, image_size=image_size, map_h=map_h, map_w=map_w)
                    if mask.any():
                        concept_indices.append(int(concept_idx))
                        gt_masks.append(mask)
                if concept_indices:
                    state["images_with_targets"] += 1
                    idx_t = torch.as_tensor(concept_indices, dtype=torch.long)
                    update_metrics(
                        state,
                        score_maps_full[batch_idx].index_select(0, idx_t).numpy(),
                        raw_maps_full_cpu[batch_idx].index_select(0, idx_t).numpy(),
                        np.stack(gt_masks, axis=0),
                        thresholds,
                        keys,
                        args.threshold_mode if args.threshold_mode != "mean" else "mean",
                        box_iou_thresholds,
                    )
            state["images_seen"] += int(images.shape[0])
            if args.log_every > 0 and state["images_seen"] >= next_log:
                elapsed = time.perf_counter() - start
                print(f"[gdino-loc:cub] n={state['images_seen']} ips={state['images_seen']/max(elapsed,1e-6):.2f}", flush=True)
                while next_log <= state["images_seen"]:
                    next_log += max(int(args.log_every), 1)
    return finalize(state, keys, box_iou_thresholds)


def eval_imagenet(args: argparse.Namespace, thresholds: Sequence[float], keys: Sequence[str], box_iou_thresholds: Sequence[float]) -> Dict[str, Any]:
    if args.threshold_mode == "percentile":
        raise SystemExit("ImageNet GDINO localization does not support --threshold_mode percentile.")
    if not args.val_root and not args.val_tar:
        raise SystemExit("ImageNet GDINO localization requires one of --val_root or --val_tar.")
    artifact_dir = Path(args.gcbm_path).resolve()
    source_run_dir = resolve_source_run_dir(artifact_dir)
    try:
        cfg = load_run_config(source_run_dir, argparse.Namespace(**vars(args), workers=int(args.num_workers)))
    except TypeError as exc:
        if "unexpected keyword argument" not in str(exc):
            raise
        payload = json.loads((source_run_dir / "config.json").read_text(encoding="utf-8"))
        valid_fields = {field.name for field in dataclasses.fields(Config)}
        payload = {key: value for key, value in payload.items() if key in valid_fields}
        payload.setdefault("feature_storage_dtype", "fp16")
        payload.setdefault("saga_table_device", "cpu")
        payload.setdefault("dense_lr", 1e-3)
        payload.setdefault("dense_n_iters", 20)
        payload.setdefault("train_random_transforms", True)
        payload.setdefault("learn_spatial_residual_scale", False)
        payload["device"] = args.device
        payload["batch_size"] = int(args.batch_size)
        payload["workers"] = int(args.num_workers)
        payload["prefetch_factor"] = int(args.prefetch_factor)
        payload["persistent_workers"] = bool(args.persistent_workers)
        payload["pin_memory"] = bool(args.pin_memory)
        payload["skip_final_layer"] = True
        payload["print_config"] = False
        valid_fields = {field.name for field in dataclasses.fields(Config)}
        cfg = Config(**{key: value for key, value in payload.items() if key in valid_fields})
    configure_runtime(cfg)
    concepts = load_concepts(str(source_run_dir / "concepts.txt"))
    concept_to_idx = {name: idx for idx, name in enumerate(concepts)}
    backbone, head = build_model(cfg, n_concepts=len(concepts))
    head.load_state_dict(torch.load(source_run_dir / "concept_head_best.pt", map_location=cfg.device))
    backbone.eval()
    head.eval()
    annotation_val_dir = resolve_val_annotation_dir(Path(args.annotation_dir).resolve())
    annotation_val_root = Path(args.annotation_val_root).resolve() if args.annotation_val_root else None
    state = init_state(keys, box_iou_thresholds)
    start = time.perf_counter()
    next_log = max(int(args.log_every), 1)

    def process_batch(images: List[torch.Tensor], annotations: List[List[Dict[str, Any]]], image_sizes: List[Tuple[int, int]]) -> None:
        batch = prepare_images(torch.stack(images, dim=0), cfg)
        with torch.no_grad():
            feats = backbone(batch)
            outputs = head(feats)
            _global_targets, mask_indices, mask_targets, mask_valid = build_gdino_targets(
                annotations, image_sizes, concept_to_idx, len(concepts), cfg, cfg.device
            )
            raw_maps = F.interpolate(outputs["spatial_maps"], size=mask_targets.shape[-2:], mode="bilinear", align_corners=False).float()
        for batch_idx in range(raw_maps.shape[0]):
            valid = mask_valid[batch_idx]
            if not bool(valid.any()):
                continue
            concept_ids = mask_indices[batch_idx][valid]
            gt = mask_targets[batch_idx][valid]
            target_valid = gt.flatten(1).sum(dim=1) > 0
            if not bool(target_valid.any()):
                continue
            state["images_with_targets"] += 1
            concept_ids = concept_ids[target_valid]
            gt = gt[target_valid]
            pred = raw_maps[batch_idx].index_select(0, concept_ids)
            score_maps = normalize_imagenet_maps(pred, args.map_normalization)
            update_metrics(
                state,
                score_maps.detach().cpu().numpy(),
                pred.detach().cpu().numpy(),
                (gt > 0.0).detach().cpu().numpy(),
                thresholds,
                keys,
                "mean" if args.threshold_mode == "mean" else "fixed",
                box_iou_thresholds,
            )

    if args.val_root:
        dataset = ImageNetValDataset(
            Path(args.val_root).resolve(),
            annotation_val_dir,
            annotation_val_root,
            cfg.input_size,
        )
        loader = DataLoader(
            dataset,
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=int(args.num_workers),
            pin_memory=bool(args.pin_memory),
            collate_fn=imagenet_collate,
            **(
                {"prefetch_factor": int(args.prefetch_factor), "persistent_workers": bool(args.persistent_workers)}
                if int(args.num_workers) > 0
                else {}
            ),
        )
        for images, annotations, image_sizes, _names in loader:
            if int(args.max_images) > 0 and state["images_seen"] >= int(args.max_images):
                break
            if int(args.max_images) > 0 and state["images_seen"] + len(images) > int(args.max_images):
                keep = int(args.max_images) - state["images_seen"]
                images, annotations, image_sizes = images[:keep], annotations[:keep], image_sizes[:keep]
            process_batch(images, annotations, image_sizes)
            state["images_seen"] += len(images)
            if args.log_every > 0 and state["images_seen"] >= next_log:
                elapsed = time.perf_counter() - start
                print(f"[gdino-loc:imagenet] n={state['images_seen']} ips={state['images_seen']/max(elapsed,1e-6):.2f}", flush=True)
                while next_log <= state["images_seen"]:
                    next_log += max(int(args.log_every), 1)
    else:
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(cfg.input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        filename_map = build_filename_to_annotation_path(annotation_val_dir, annotation_val_root)
        images: List[torch.Tensor] = []
        annotations: List[List[Dict[str, Any]]] = []
        image_sizes: List[Tuple[int, int]] = []
        with tarfile.open(Path(args.val_tar).resolve(), "r|*") as tf:
            for member in tf:
                if not member.isfile():
                    continue
                match = VAL_RE.search(Path(member.name).name)
                if match is None:
                    continue
                handle = tf.extractfile(member)
                if handle is None:
                    continue
                with Image.open(handle) as image:
                    image = image.convert("RGB")
                    image_sizes.append((int(image.size[0]), int(image.size[1])))
                    images.append(transform(image))
                annotations.append(
                    load_annotation_payload(annotation_val_dir, int(match.group(1)), Path(member.name).name, filename_map)
                )
                if len(images) >= int(args.batch_size):
                    process_batch(images, annotations, image_sizes)
                    state["images_seen"] += len(images)
                    images.clear()
                    annotations.clear()
                    image_sizes.clear()
                if int(args.max_images) > 0 and state["images_seen"] >= int(args.max_images):
                    break
        if images and (int(args.max_images) <= 0 or state["images_seen"] < int(args.max_images)):
            process_batch(images, annotations, image_sizes)
            state["images_seen"] += len(images)
    return finalize(state, keys, box_iou_thresholds)


def main() -> None:
    args = parse_args()
    thresholds = [0.0] if args.threshold_mode == "mean" or str(args.activation_thresholds).strip().lower() in {"mean", "meanthr"} else parse_float_list(args.activation_thresholds)
    box_iou_thresholds = parse_float_list(args.box_iou_thresholds)
    keys = threshold_keys(args, thresholds)
    metrics = (
        eval_cub(args, thresholds, keys, box_iou_thresholds)
        if args.dataset == "cub"
        else eval_imagenet(args, thresholds, keys, box_iou_thresholds)
    )
    payload = {
        "dataset": args.dataset,
        "gcbm_path": str(Path(args.gcbm_path).resolve()),
        "annotation_dir": str(Path(args.annotation_dir).resolve()),
        "map_normalization": args.map_normalization,
        "threshold_mode": "mean" if keys == ["mean"] else args.threshold_mode,
        "activation_thresholds": keys,
        "box_iou_thresholds": [str(t) for t in box_iou_thresholds],
        "metrics": metrics,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
