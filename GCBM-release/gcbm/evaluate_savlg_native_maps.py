import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Allow direct script execution without requiring an external PYTHONPATH.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from methods.savlg import (
    _annotation_split_dir,
    build_savlg_concept_layer,
    compute_savlg_concept_logits,
    create_savlg_splits,
    forward_savlg_backbone,
    forward_savlg_concept_layer,
)
from model.utils import get_bbox_iou
from gcbm.visualize_savlg_examples import (
    _load_args,
    _load_concepts,
    _normalize_map,
    _union_boxes,
)
from data import utils as data_utils


class IndexedDataset(Dataset):
    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        image, target = self.base[idx]
        return idx, image, target


def _parse_csv_floats(x: str):
    return [float(v.strip()) for v in x.split(",") if v.strip()]


def _point_in_box(x: int, y: int, box: List[float]) -> bool:
    x1, y1, x2, y2 = box
    return float(x1) <= float(x) <= float(x2) and float(y1) <= float(y) <= float(y2)


def _mask_box_coverage(mask: torch.Tensor, box: List[float]) -> float:
    if mask.ndim != 2:
        raise ValueError(f"Expected [H,W] mask, got shape {tuple(mask.shape)}")
    h, w = int(mask.shape[0]), int(mask.shape[1])
    x1, y1, x2, y2 = box
    x1_i = max(0, min(w - 1, int(x1)))
    y1_i = max(0, min(h - 1, int(y1)))
    x2_i = max(0, min(w, int(x2)))
    y2_i = max(0, min(h, int(y2)))
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    box_mask = mask[y1_i:y2_i, x1_i:x2_i]
    return float(box_mask.float().mean().item())


def _rasterize_box_mask(h: int, w: int, box: List[float]) -> torch.Tensor:
    x1, y1, x2, y2 = box
    x1_i = max(0, min(w - 1, int(x1)))
    y1_i = max(0, min(h - 1, int(y1)))
    x2_i = max(0, min(w, int(x2)))
    y2_i = max(0, min(h, int(y2)))
    mask = torch.zeros((h, w), dtype=torch.bool)
    if x2_i > x1_i and y2_i > y1_i:
        mask[y1_i:y2_i, x1_i:x2_i] = True
    return mask


def _mask_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> float:
    if pred_mask.shape != gt_mask.shape:
        raise ValueError(f"Mask shape mismatch: {tuple(pred_mask.shape)} vs {tuple(gt_mask.shape)}")
    pred_mask = pred_mask.bool()
    gt_mask = gt_mask.bool()
    inter = (pred_mask & gt_mask).sum().item()
    union = (pred_mask | gt_mask).sum().item()
    if union <= 0:
        return 0.0
    return float(inter / union)


def _mask_dice(pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> float:
    if pred_mask.shape != gt_mask.shape:
        raise ValueError(f"Mask shape mismatch: {tuple(pred_mask.shape)} vs {tuple(gt_mask.shape)}")
    pred_mask = pred_mask.bool()
    gt_mask = gt_mask.bool()
    inter = (pred_mask & gt_mask).sum().item()
    pred_sum = pred_mask.sum().item()
    gt_sum = gt_mask.sum().item()
    denom = pred_sum + gt_sum
    if denom <= 0:
        return 0.0
    return float((2.0 * inter) / denom)


def _mask_confusion_counts(pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> Tuple[int, int, int]:
    if pred_mask.shape != gt_mask.shape:
        raise ValueError(f"Mask shape mismatch: {tuple(pred_mask.shape)} vs {tuple(gt_mask.shape)}")
    pred_mask = pred_mask.bool()
    gt_mask = gt_mask.bool()
    tp = int((pred_mask & gt_mask).sum().item())
    fp = int((pred_mask & ~gt_mask).sum().item())
    fn = int((~pred_mask & gt_mask).sum().item())
    return tp, fp, fn


def _box_mask_distribution(h: int, w: int, box: List[float]) -> torch.Tensor:
    mask = _rasterize_box_mask(h, w, box).to(torch.float32)
    total = mask.sum()
    if total <= 0:
        return mask
    return mask / total


def _distribution_soft_iou(pred_dist: torch.Tensor, gt_dist: torch.Tensor) -> float:
    if pred_dist.shape != gt_dist.shape:
        raise ValueError(
            f"Distribution shape mismatch: {tuple(pred_dist.shape)} vs {tuple(gt_dist.shape)}"
        )
    pred_dist = pred_dist.detach().float()
    gt_dist = gt_dist.detach().float()
    inter = torch.minimum(pred_dist, gt_dist).sum().item()
    union = torch.maximum(pred_dist, gt_dist).sum().item()
    if union <= 0:
        return 0.0
    return float(inter / union)


def _mass_in_box(pred_dist: torch.Tensor, gt_mask: torch.Tensor) -> float:
    if pred_dist.shape != gt_mask.shape:
        raise ValueError(
            f"Distribution/mask shape mismatch: {tuple(pred_dist.shape)} vs {tuple(gt_mask.shape)}"
        )
    pred_dist = pred_dist.detach().float()
    gt_mask = gt_mask.detach().to(torch.float32)
    return float((pred_dist * gt_mask).sum().item())


def _normalize_map_with_mode(x: torch.Tensor, mode: str) -> torch.Tensor:
    x = x.detach().float()
    mode = str(mode).lower()
    if x.ndim == 2:
        reduce_dims = None
    elif x.ndim == 3:
        reduce_dims = (1, 2)
    else:
        raise ValueError(f"Expected 2D or 3D maps, got shape {tuple(x.shape)}")
    if mode == "minmax":
        if reduce_dims is None:
            return _normalize_map(x)
        x = x - x.amin(dim=reduce_dims, keepdim=True)
        denom = x.amax(dim=reduce_dims, keepdim=True).clamp_min(1e-6)
        return x / denom
    if mode == "sigmoid":
        return torch.sigmoid(x)
    if mode == "concept_zscore_minmax":
        if reduce_dims is None:
            x = (x - x.mean()) / x.std(unbiased=False).clamp_min(1e-6)
            return _normalize_map(x)
        mean = x.mean(dim=reduce_dims, keepdim=True)
        std = x.std(dim=reduce_dims, keepdim=True, unbiased=False).clamp_min(1e-6)
        x = (x - mean) / std
        x = x - x.amin(dim=reduce_dims, keepdim=True)
        denom = x.amax(dim=reduce_dims, keepdim=True).clamp_min(1e-6)
        return x / denom
    raise ValueError(f"Unsupported map_normalization={mode}")


def _boxes_from_masks(masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # masks: [K, H, W] bool on any device
    if masks.ndim != 3:
        raise ValueError(f"Expected [K,H,W] masks, got shape {tuple(masks.shape)}")
    valid = masks.flatten(1).any(dim=1)
    rows_any = masks.any(dim=2)
    cols_any = masks.any(dim=1)

    y1 = rows_any.float().argmax(dim=1)
    x1 = cols_any.float().argmax(dim=1)
    y2 = masks.shape[1] - 1 - rows_any.flip(1).float().argmax(dim=1)
    x2 = masks.shape[2] - 1 - cols_any.flip(1).float().argmax(dim=1)

    boxes = torch.stack([x1, y1, x2, y2], dim=1).to(torch.float32)
    return boxes, valid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_path", type=str, required=True)
    parser.add_argument("--annotation_dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--activation_thresholds", type=str, default="0.9")
    parser.add_argument("--box_iou_thresholds", type=str, default="0.1,0.3,0.5,0.7")
    parser.add_argument("--map_normalization", type=str, default="concept_zscore_minmax")
    parser.add_argument("--eval_subset_mode", type=str, default="gt_present", choices=["gt_present"])
    parser.add_argument("--threshold_protocol", type=str, default="explicit", choices=["explicit", "meanthr"])
    parser.add_argument(
        "--threshold_source",
        type=str,
        default="normalized_map",
        choices=["normalized_map", "pred_dist"],
    )
    parser.add_argument(
        "--compute_distribution_metrics",
        action="store_true",
        help="Also compute threshold-free soft metrics from pred_dist. By default only the selected threshold source is materialized.",
    )
    args_ns = parser.parse_args()

    args = _load_args(args_ns.load_path, args_ns.device, args_ns.annotation_dir)
    _, _, _, _, test_dataset, backbone = create_savlg_splits(args)
    if args_ns.max_images is not None:
        keep = min(args_ns.max_images, len(test_dataset))
        test_dataset = torch.utils.data.Subset(test_dataset, list(range(keep)))
    dataset = IndexedDataset(test_dataset)

    concepts = _load_concepts(args_ns.load_path, args)
    concept_to_idx = {name: idx for idx, name in enumerate(concepts)}
    ann_dir = _annotation_split_dir(args.annotation_dir, args.dataset, "val")

    concept_layer = build_savlg_concept_layer(args, backbone, len(concepts)).to(args.device)
    concept_layer.load_state_dict(torch.load(os.path.join(args_ns.load_path, "concept_layer.pt"), map_location=args.device))
    concept_layer.eval()
    backbone.eval()

    loader = DataLoader(
        dataset,
        batch_size=args_ns.batch_size,
        shuffle=False,
        num_workers=args_ns.num_workers,
        pin_memory=False,
    )

    use_meanthr = args_ns.threshold_protocol == "meanthr" or str(args_ns.activation_thresholds).strip().lower() == "meanthr"
    activation_thresholds = [] if use_meanthr else _parse_csv_floats(args_ns.activation_thresholds)
    iou_thresholds = _parse_csv_floats(args_ns.box_iou_thresholds)
    need_pred_dist = args_ns.threshold_source == "pred_dist" or args_ns.compute_distribution_metrics
    per_threshold_ious = {"meanthr": []} if use_meanthr else {thr: [] for thr in activation_thresholds}
    per_threshold_point_hits = {"meanthr": []} if use_meanthr else {thr: [] for thr in activation_thresholds}
    per_threshold_coverages = {"meanthr": []} if use_meanthr else {thr: [] for thr in activation_thresholds}
    per_threshold_mask_ious = {"meanthr": []} if use_meanthr else {thr: [] for thr in activation_thresholds}
    per_threshold_mask_dices = {"meanthr": []} if use_meanthr else {thr: [] for thr in activation_thresholds}
    per_threshold_tp = {"meanthr": 0} if use_meanthr else {thr: 0 for thr in activation_thresholds}
    per_threshold_fp = {"meanthr": 0} if use_meanthr else {thr: 0 for thr in activation_thresholds}
    per_threshold_fn = {"meanthr": 0} if use_meanthr else {thr: 0 for thr in activation_thresholds}
    distribution_soft_ious = []
    distribution_mass_in_box = []
    distribution_point_hits = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="native map eval"):
            indices, images, _targets = batch
            images = images.to(args.device, non_blocking=True)
            feats = forward_savlg_backbone(backbone, images, args)
            _global_outputs, spatial_maps = forward_savlg_concept_layer(concept_layer, feats)
            img_h, img_w = int(images.shape[-2]), int(images.shape[-1])

            for b, ds_idx in enumerate(indices.tolist()):
                ann_path = os.path.join(ann_dir, f"{int(ds_idx)}.json")
                if not os.path.exists(ann_path):
                    continue
                with open(ann_path, "r") as f:
                    payload = json.load(f)
                gt_boxes_by_concept = {}
                for ann in payload[1:]:
                    if not isinstance(ann, dict):
                        continue
                    label = ann.get("label")
                    if isinstance(label, str):
                        label = data_utils.canonicalize_concept_label(label)
                    box = ann.get("box")
                    if not isinstance(label, str) or label not in concept_to_idx:
                        continue
                    if not isinstance(box, list) or len(box) != 4:
                        continue
                    gt_boxes_by_concept.setdefault(label, []).append([float(v) for v in box])
                if not gt_boxes_by_concept:
                    continue
                concept_indices: List[int] = []
                gt_boxes: List[List[float]] = []
                for concept_name, concept_boxes in gt_boxes_by_concept.items():
                    gt_box = _union_boxes(concept_boxes)
                    if gt_box is None:
                        continue
                    concept_indices.append(concept_to_idx[concept_name])
                    gt_boxes.append(gt_box)
                if not concept_indices:
                    continue
                concept_idx_tensor = torch.as_tensor(concept_indices, device=spatial_maps.device, dtype=torch.long)
                maps_k_native = spatial_maps[b].index_select(0, concept_idx_tensor)
                maps_k = F.interpolate(
                    maps_k_native.unsqueeze(1),
                    size=(img_h, img_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
                pred_dist_k = F.softmax(maps_k.flatten(1), dim=1).view_as(maps_k) if need_pred_dist else None
                norm_maps_k = (
                    _normalize_map_with_mode(maps_k, args_ns.map_normalization)
                    if args_ns.threshold_source == "normalized_map"
                    else None
                )
                threshold_maps_k = pred_dist_k if args_ns.threshold_source == "pred_dist" else norm_maps_k
                gt_boxes_cpu = gt_boxes
                if args_ns.compute_distribution_metrics:
                    assert pred_dist_k is not None
                    argmax_flat_dist = pred_dist_k.flatten(1).argmax(dim=1)
                    argmax_y_dist = (argmax_flat_dist // pred_dist_k.shape[-1]).cpu().tolist()
                    argmax_x_dist = (argmax_flat_dist % pred_dist_k.shape[-1]).cpu().tolist()
                    pred_dist_cpu = pred_dist_k.cpu()
                    for pred_dist, gt_box, px, py in zip(
                        pred_dist_cpu, gt_boxes_cpu, argmax_x_dist, argmax_y_dist
                    ):
                        gt_mask = _rasterize_box_mask(int(pred_dist.shape[0]), int(pred_dist.shape[1]), gt_box)
                        gt_dist = _box_mask_distribution(int(pred_dist.shape[0]), int(pred_dist.shape[1]), gt_box)
                        distribution_soft_ious.append(_distribution_soft_iou(pred_dist, gt_dist))
                        distribution_mass_in_box.append(_mass_in_box(pred_dist, gt_mask))
                        distribution_point_hits.append(
                            1.0 if _point_in_box(int(px), int(py), gt_box) else 0.0
                        )
                if use_meanthr:
                    thr_key = "meanthr"
                    thr_tensor = threshold_maps_k.mean(dim=(1, 2), keepdim=True)
                    pred_masks = threshold_maps_k >= thr_tensor
                    pred_boxes_t, valid_t = _boxes_from_masks(pred_masks)
                    pred_boxes = pred_boxes_t.cpu().tolist()
                    valid = valid_t.cpu().tolist()
                    argmax_flat = threshold_maps_k.flatten(1).argmax(dim=1)
                    argmax_y = (argmax_flat // threshold_maps_k.shape[-1]).cpu().tolist()
                    argmax_x = (argmax_flat % threshold_maps_k.shape[-1]).cpu().tolist()
                    pred_masks_cpu = pred_masks.cpu()
                    for pred_box, is_valid, gt_box, px, py, pred_mask in zip(
                        pred_boxes, valid, gt_boxes_cpu, argmax_x, argmax_y, pred_masks_cpu
                    ):
                        iou = 0.0 if not is_valid else float(get_bbox_iou(pred_box, gt_box))
                        per_threshold_ious[thr_key].append(iou)
                        per_threshold_point_hits[thr_key].append(1.0 if _point_in_box(int(px), int(py), gt_box) else 0.0)
                        per_threshold_coverages[thr_key].append(_mask_box_coverage(pred_mask, gt_box))
                        gt_mask = _rasterize_box_mask(int(pred_mask.shape[0]), int(pred_mask.shape[1]), gt_box)
                        per_threshold_mask_ious[thr_key].append(_mask_iou(pred_mask, gt_mask))
                        per_threshold_mask_dices[thr_key].append(_mask_dice(pred_mask, gt_mask))
                        tp, fp, fn = _mask_confusion_counts(pred_mask, gt_mask)
                        per_threshold_tp[thr_key] += tp
                        per_threshold_fp[thr_key] += fp
                        per_threshold_fn[thr_key] += fn
                else:
                    for thr in activation_thresholds:
                        pred_masks = threshold_maps_k >= float(thr)
                        pred_boxes_t, valid_t = _boxes_from_masks(pred_masks)
                        pred_boxes = pred_boxes_t.cpu().tolist()
                        valid = valid_t.cpu().tolist()
                        argmax_flat = threshold_maps_k.flatten(1).argmax(dim=1)
                        argmax_y = (argmax_flat // threshold_maps_k.shape[-1]).cpu().tolist()
                        argmax_x = (argmax_flat % threshold_maps_k.shape[-1]).cpu().tolist()
                        pred_masks_cpu = pred_masks.cpu()
                        for pred_box, is_valid, gt_box, px, py, pred_mask in zip(
                            pred_boxes, valid, gt_boxes_cpu, argmax_x, argmax_y, pred_masks_cpu
                        ):
                            iou = 0.0 if not is_valid else float(get_bbox_iou(pred_box, gt_box))
                            per_threshold_ious[thr].append(iou)
                            per_threshold_point_hits[thr].append(1.0 if _point_in_box(int(px), int(py), gt_box) else 0.0)
                            per_threshold_coverages[thr].append(_mask_box_coverage(pred_mask, gt_box))
                            gt_mask = _rasterize_box_mask(int(pred_mask.shape[0]), int(pred_mask.shape[1]), gt_box)
                            per_threshold_mask_ious[thr].append(_mask_iou(pred_mask, gt_mask))
                            per_threshold_mask_dices[thr].append(_mask_dice(pred_mask, gt_mask))
                            tp, fp, fn = _mask_confusion_counts(pred_mask, gt_mask)
                            per_threshold_tp[thr] += tp
                            per_threshold_fp[thr] += fp
                            per_threshold_fn[thr] += fn

    results = {
        "load_path": args_ns.load_path,
        "map_normalization": args_ns.map_normalization,
        "threshold_source": args_ns.threshold_source,
        "compute_distribution_metrics": bool(args_ns.compute_distribution_metrics),
        "eval_subset_mode": args_ns.eval_subset_mode,
        "threshold_protocol": "meanthr" if use_meanthr else "explicit",
        "num_images": len(dataset),
        "num_gt_instances": int(len(next(iter(per_threshold_ious.values()))) if per_threshold_ious else 0),
        "threshold_metrics": {},
    }
    if args_ns.compute_distribution_metrics:
        results["distribution_metrics"] = {
            "soft_iou": float(sum(distribution_soft_ious) / max(len(distribution_soft_ious), 1)),
            "mass_in_box": float(
                sum(distribution_mass_in_box) / max(len(distribution_mass_in_box), 1)
            ),
            "point_hit": float(
                sum(distribution_point_hits) / max(len(distribution_point_hits), 1)
            ),
        }

    best_mean_iou_thr = None
    best_mean_iou = None
    best_box_acc = {iou_thr: {"threshold": None, "value": None} for iou_thr in iou_thresholds}

    metric_keys = ["meanthr"] if use_meanthr else activation_thresholds
    for thr in metric_keys:
        ious = per_threshold_ious[thr]
        if not ious:
            mean_iou = 0.0
        else:
            mean_iou = float(sum(ious) / len(ious))
        box_acc = {}
        for iou_thr in iou_thresholds:
            value = 0.0 if not ious else float(sum(i >= iou_thr for i in ious) / len(ious))
            box_acc[str(iou_thr)] = value
            cur = best_box_acc[iou_thr]
            if cur["value"] is None or value > cur["value"]:
                cur["value"] = value
                cur["threshold"] = thr
        point_hit = 0.0 if not per_threshold_point_hits[thr] else float(sum(per_threshold_point_hits[thr]) / len(per_threshold_point_hits[thr]))
        coverage = 0.0 if not per_threshold_coverages[thr] else float(sum(per_threshold_coverages[thr]) / len(per_threshold_coverages[thr]))
        mask_iou = 0.0 if not per_threshold_mask_ious[thr] else float(sum(per_threshold_mask_ious[thr]) / len(per_threshold_mask_ious[thr]))
        dice = 0.0 if not per_threshold_mask_dices[thr] else float(sum(per_threshold_mask_dices[thr]) / len(per_threshold_mask_dices[thr]))
        tp = int(per_threshold_tp[thr])
        fp = int(per_threshold_fp[thr])
        fn = int(per_threshold_fn[thr])
        precision = float(tp / max(tp + fp, 1))
        recall = float(tp / max(tp + fn, 1))
        f1 = float((2.0 * precision * recall) / max(precision + recall, 1e-12))
        results["threshold_metrics"][str(thr)] = {
            "mean_iou": mean_iou,
            "mask_iou": mask_iou,
            "dice": dice,
            "pixel_counts": {
                "tp": tp,
                "fp": fp,
                "fn": fn,
            },
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "box_acc": box_acc,
            "mAP": box_acc,
            "point_hit": point_hit,
            "coverage": coverage,
        }
        if best_mean_iou is None or mean_iou > best_mean_iou:
            best_mean_iou = mean_iou
            best_mean_iou_thr = thr

    results["best_mean_iou"] = {"threshold": best_mean_iou_thr, "value": best_mean_iou}
    results["best_box_acc"] = {str(k): v for k, v in best_box_acc.items()}
    if use_meanthr and args_ns.compute_distribution_metrics:
        results["meanthr_metrics"] = {
            **results["threshold_metrics"]["meanthr"],
            **results["distribution_metrics"],
        }

    output_path = Path(args_ns.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
