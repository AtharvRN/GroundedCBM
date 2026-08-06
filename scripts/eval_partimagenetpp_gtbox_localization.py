#!/usr/bin/env python3
"""Evaluate PartImageNet++ spatial concept maps against human GT part annotations."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval_gdino_localization import (  # noqa: E402
    finalize,
    init_state,
    normalize_imagenet_maps,
    parse_float_list,
    threshold_keys,
    update_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate concept localization on PartImageNet++ human GT boxes or segmentation polygons."
    )
    parser.add_argument("--gcbm_path", required=True, help="Trained CBM run directory.")
    parser.add_argument(
        "--model_name",
        default="savlg_cbm",
        choices=["savlg_cbm", "sgcbm", "sg_cbm", "salf_cbm", "vlg_cbm"],
        help="Model family to load.",
    )
    parser.add_argument("--val_manifest", required=True, help="PartImageNet++ val JSONL manifest.")
    parser.add_argument("--train_manifest", default="", help="PartImageNet++ train JSONL manifest.")
    parser.add_argument("--gt_boxes_jsonl", default="", help="Val GT boxes JSONL in manifest order.")
    parser.add_argument(
        "--gt_segments_jsonl",
        default="",
        help="Val human GT segmentation-polygons JSONL in manifest order.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", "--workers", dest="num_workers", type=int, default=4)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument(
        "--evaluation_map_size",
        type=int,
        default=0,
        help=(
            "Optional square grid for localization scoring. Maps are bilinearly resized "
            "before normalization and GT is rasterized at this same resolution. "
            "Use this to compare models with different native map sizes."
        ),
    )
    parser.add_argument("--activation_thresholds", default="0.3,0.5,0.7,0.9")
    parser.add_argument("--box_iou_thresholds", default="0.1,0.3,0.5")
    parser.add_argument(
        "--threshold_mode",
        choices=["fixed", "percentile", "mean"],
        default="fixed",
    )
    parser.add_argument(
        "--map_normalization",
        choices=["minmax", "sigmoid", "proj_zscore_minmax", "concept_zscore_minmax"],
        default="concept_zscore_minmax",
        help="For PartImageNet++ this uses per-map zscore followed by min-max, matching ImageNet behavior.",
    )
    parser.add_argument(
        "--distribution_map_normalization",
        choices=["raw", "zscore"],
        default="raw",
        help=(
            "Map values used only by RMA/soft-IoU spatial distributions. "
            "zscore removes arbitrary per-map logit temperature before softmax."
        ),
    )
    parser.add_argument(
        "--distribution_only",
        action="store_true",
        help="Compute only threshold-free distribution metrics (RMA, pointing, soft IoU).",
    )
    parser.add_argument(
        "--vlg_map_method",
        choices=["gradcam", "cam"],
        default="gradcam",
        help=(
            "VLG-CBM localization source. Grad-CAM is the established protocol for "
            "global-only concept heads; cam is retained only for reproducing older runs."
        ),
    )
    parser.add_argument("--log_every", type=int, default=500)
    args = parser.parse_args()
    if bool(args.gt_boxes_jsonl) == bool(args.gt_segments_jsonl):
        parser.error("Pass exactly one of --gt_boxes_jsonl or --gt_segments_jsonl.")
    return args


def _canonicalize(text: str) -> str:
    from data import utils as data_utils

    return data_utils.canonicalize_concept_label(str(text))


def load_concepts(path: Path) -> List[str]:
    return [_canonicalize(line.strip()) for line in path.read_text().splitlines() if line.strip()]


def load_gt_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"GT boxes file is empty: {path}")
    for expected, row in enumerate(rows):
        if int(row.get("row_index", expected)) != expected:
            raise RuntimeError(f"GT row_index mismatch at row {expected}: {row.get('row_index')}")
    return rows


def _state_dict_num_outputs(state_dict: Dict[str, torch.Tensor], concepts: Sequence[str]) -> int:
    for key in ("spatial_layer.weight", "global_layer.weight", "weight"):
        value = state_dict.get(key)
        if isinstance(value, torch.Tensor):
            return int(value.shape[0])
    return len(concepts)


def _load_savlg(args: argparse.Namespace, run_args: argparse.Namespace, concepts: Sequence[str]):
    from methods.savlg import build_savlg_concept_layer, create_savlg_splits

    _train_cbl, _val_cbl, _train_dataset, _val_dataset, test_dataset, backbone = create_savlg_splits(run_args)
    state_dict = torch.load(Path(args.gcbm_path) / "concept_layer.pt", map_location=run_args.device)
    concept_layer = build_savlg_concept_layer(run_args, backbone, _state_dict_num_outputs(state_dict, concepts))
    concept_layer.load_state_dict(state_dict)
    return backbone, concept_layer, test_dataset


def _load_salf(args: argparse.Namespace, run_args: argparse.Namespace, concepts: Sequence[str]):
    from data import utils as data_utils
    from methods.salf import SpatialBackbone, build_spatial_concept_layer

    backbone = SpatialBackbone(
        run_args.backbone,
        device=run_args.device,
        spatial_stage=getattr(run_args, "savlg_spatial_stage", "conv5"),
        checkpoint_path=getattr(run_args, "backbone_checkpoint", ""),
    )
    dataset = data_utils.get_data(f"{run_args.dataset}_val", preprocess=backbone.preprocess)
    if int(args.max_images) > 0:
        dataset = Subset(dataset, list(range(min(int(args.max_images), len(dataset)))))
    state_dict = torch.load(Path(args.gcbm_path) / "concept_layer.pt", map_location=run_args.device)
    if "weight" in state_dict and state_dict["weight"].ndim == 4:
        concept_layer = torch.nn.Conv2d(
            int(state_dict["weight"].shape[1]),
            int(state_dict["weight"].shape[0]),
            kernel_size=1,
            bias=("bias" in state_dict),
        ).to(run_args.device)
    elif "spatial_layer.weight" in state_dict:
        concept_layer = build_spatial_concept_layer(
            run_args,
            backbone.output_dim,
            int(state_dict["spatial_layer.weight"].shape[0]),
            is_vit=getattr(backbone, "is_vit", False),
        )
    else:
        raise ValueError(f"Unsupported SALF concept layer format at {Path(args.gcbm_path) / 'concept_layer.pt'}")
    concept_layer.load_state_dict(state_dict)
    return backbone, concept_layer, dataset


def _load_vlg(args: argparse.Namespace, concepts: Sequence[str]):
    from data import utils as data_utils
    from gcbm.imagenet_config import Config
    from gcbm.imagenet_models import build_model
    from gcbm.runtime import configure_runtime

    gcbm_path = Path(args.gcbm_path)
    payload = json.loads((gcbm_path / "config.json").read_text())
    payload.setdefault("feature_storage_dtype", "fp16")
    payload.setdefault("saga_table_device", "cpu")
    payload.setdefault("dense_lr", 1e-3)
    payload.setdefault("dense_n_iters", 20)
    payload.setdefault("train_random_transforms", False)
    payload.setdefault("learn_spatial_residual_scale", False)
    payload["device"] = args.device
    payload["batch_size"] = int(args.batch_size)
    payload["workers"] = int(args.num_workers)
    payload["prefetch_factor"] = 2
    payload["persistent_workers"] = False
    payload["pin_memory"] = str(args.device).startswith("cuda")
    payload["skip_final_layer"] = True
    payload["print_config"] = False
    valid_fields = {field.name for field in dataclasses.fields(Config)}
    cfg = Config(**{key: value for key, value in payload.items() if key in valid_fields})
    configure_runtime(cfg)

    backbone, head = build_model(cfg, n_concepts=len(concepts))
    head.load_state_dict(torch.load(gcbm_path / "concept_head_best.pt", map_location=cfg.device))
    backbone.eval()
    head.eval()

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(int(cfg.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    dataset = data_utils.get_data("partimagenetpp_val", preprocess=transform)
    return cfg, backbone, head, dataset


def load_model(args: argparse.Namespace):
    gcbm_path = Path(args.gcbm_path)
    concepts = load_concepts(gcbm_path / "concepts.txt")
    model_name = str(args.model_name).lower().replace("-", "_")

    train_manifest = str(args.train_manifest or os.environ.get("PARTIMAGENETPP_TRAIN_MANIFEST", ""))
    if train_manifest:
        os.environ["PARTIMAGENETPP_TRAIN_MANIFEST"] = train_manifest
    os.environ["PARTIMAGENETPP_VAL_MANIFEST"] = str(args.val_manifest)

    if model_name == "vlg_cbm":
        run_args, backbone, concept_layer, test_dataset = _load_vlg(args, concepts)
        backbone.eval()
        concept_layer.eval()
        return run_args, backbone, concept_layer, concepts, test_dataset

    run_args = argparse.Namespace(**json.loads((gcbm_path / "args.txt").read_text()))
    run_args.device = args.device
    run_args.cbl_batch_size = int(args.batch_size)
    run_args.saga_batch_size = int(args.batch_size)
    run_args.num_workers = int(args.num_workers)
    run_args.skip_test_eval = False
    if int(args.max_images) > 0:
        run_args.max_test_images = int(args.max_images)

    if model_name in {"savlg_cbm", "sgcbm", "sg_cbm"}:
        backbone, concept_layer, test_dataset = _load_savlg(args, run_args, concepts)
    elif model_name == "salf_cbm":
        backbone, concept_layer, test_dataset = _load_salf(args, run_args, concepts)
    else:
        raise ValueError(f"Unsupported model_name={args.model_name}")
    backbone.eval()
    concept_layer.eval()
    return run_args, backbone, concept_layer, concepts, test_dataset


def forward_concept_maps(
    model_name: str,
    backbone,
    concept_layer,
    images: torch.Tensor,
    run_args,
    *,
    vlg_map_method: str = "gradcam",
) -> torch.Tensor:
    normalized = str(model_name).lower().replace("-", "_")
    if normalized in {"savlg_cbm", "sgcbm", "sg_cbm"}:
        from methods.savlg import forward_savlg_backbone, forward_savlg_concept_layer

        feats = forward_savlg_backbone(backbone, images, run_args)
        _global_outputs, spatial_maps = forward_savlg_concept_layer(concept_layer, feats)
        return spatial_maps
    if normalized == "vlg_cbm":
        from gcbm.training_utils import prepare_images

        feats = backbone(prepare_images(images, run_args))
        outputs = concept_layer(feats)
        if "spatial_maps" in outputs:
            return outputs["spatial_maps"]
        if hasattr(concept_layer, "global_head"):
            weight = concept_layer.global_head.weight[:, :, None, None]
            if vlg_map_method == "cam":
                return F.conv2d(feats["conv5"], weight, concept_layer.global_head.bias)

            # For z_c = GAP(A) @ w_c + b_c, Grad-CAM's channel weights are
            # alpha_c,k = mean_ij(d z_c / d A_kij) = w_c,k / (H * W).
            # The positive factor 1/(H*W) vanishes during per-map normalization,
            # so this is exactly Grad-CAM without materializing a backward pass
            # for every image/concept pair.
            return F.relu(F.conv2d(feats["conv5"], weight, bias=None))
        raise RuntimeError("VLG head does not expose spatial maps or global_head weights")
    feats = backbone(images)
    maps = concept_layer(feats)
    if isinstance(maps, tuple):
        maps = maps[0]
    return maps


def rasterize_polygon_union(
    instances: Sequence[Sequence[Sequence[float]]],
    image_size: tuple[int, int],
    map_h: int,
    map_w: int,
) -> np.ndarray:
    """Rasterize human polygons in the ResNet ImageNet eval frame to map cells."""
    width, height = image_size
    canvas = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    for instance in instances:
        for polygon in instance:
            if len(polygon) < 6 or len(polygon) % 2:
                continue
            points = [(float(polygon[i]), float(polygon[i + 1])) for i in range(0, len(polygon), 2)]
            draw.polygon(points, fill=255)

    # The scratch ResNet-50 evaluation path uses Resize(256), CenterCrop(224).
    # Apply the same spatial operations before reducing the GT to the model map.
    canvas = TF.resize(canvas, 256, interpolation=InterpolationMode.NEAREST)
    canvas = TF.center_crop(canvas, [224, 224])
    tensor = torch.from_numpy(np.asarray(canvas, dtype=np.uint8) > 0).float()[None, None]
    return F.adaptive_max_pool2d(tensor, output_size=(map_h, map_w))[0, 0].bool().numpy()


def rasterize_box_union_in_eval_frame(
    boxes: Sequence[Sequence[float]],
    image_size: tuple[int, int],
    map_h: int,
    map_w: int,
) -> np.ndarray:
    """Rasterize source-coordinate boxes in the model's resize/crop frame.

    The generic ImageNet evaluator uses normalized full-image boxes.  That is
    incorrect here because both CBM variants consume a Resize(256), CenterCrop(224)
    image.  Keep box and polygon GT in exactly the same evaluation frame.
    """
    width, height = image_size
    canvas = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    for box in boxes:
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        x1, y1, x2, y2 = (float(value) for value in box)
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        draw.rectangle((x1, y1, x2, y2), fill=255)

    canvas = TF.resize(canvas, 256, interpolation=InterpolationMode.NEAREST)
    canvas = TF.center_crop(canvas, [224, 224])
    tensor = torch.from_numpy(np.asarray(canvas, dtype=np.uint8) > 0).float()[None, None]
    return F.adaptive_max_pool2d(tensor, output_size=(map_h, map_w))[0, 0].bool().numpy()


def init_segmentation_state(keys: Sequence[str], n_concepts: int, ap_bins: int = 512) -> Dict[str, Any]:
    return {
        "n_concepts": int(n_concepts),
        "ap_bins": int(ap_bins),
        "positive_hist": np.zeros((n_concepts, ap_bins), dtype=np.int64),
        "negative_hist": np.zeros((n_concepts, ap_bins), dtype=np.int64),
        "thresholds": {
            key: {
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "tn": 0,
                "by_concept": {},
            }
            for key in keys
        },
    }


def normalize_distribution_maps(maps: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "raw":
        return maps
    mean = maps.mean(dim=(1, 2), keepdim=True)
    std = maps.std(dim=(1, 2), keepdim=True, unbiased=False).clamp_min(1e-6)
    return (maps - mean) / std


def update_segmentation_metrics(
    state: Dict[str, Any],
    score_maps: np.ndarray,
    gt_masks: np.ndarray,
    concept_indices: Sequence[int],
    thresholds: Sequence[float],
    keys: Sequence[str],
    threshold_mode: str,
) -> None:
    if score_maps.shape[0] == 0:
        return
    gt_masks = gt_masks.astype(np.bool_, copy=False)
    bins = int(state["ap_bins"])
    for score_map, gt_mask, concept_idx in zip(score_maps, gt_masks, concept_indices):
        bin_ids = np.minimum((np.clip(score_map, 0.0, 1.0) * (bins - 1)).astype(np.int32), bins - 1)
        state["positive_hist"][concept_idx] += np.bincount(bin_ids[gt_mask], minlength=bins)
        state["negative_hist"][concept_idx] += np.bincount(bin_ids[~gt_mask], minlength=bins)

    for threshold, key in zip(thresholds, keys):
        if threshold_mode == "mean":
            pred_masks = score_maps >= score_maps.mean(axis=(1, 2), keepdims=True)
        elif threshold_mode == "percentile":
            cutoff = np.quantile(
                score_maps,
                q=min(max(float(threshold) / 100.0, 0.0), 1.0),
                axis=(1, 2),
                keepdims=True,
            )
            pred_masks = score_maps >= cutoff
        else:
            pred_masks = score_maps >= float(threshold)
        bucket = state["thresholds"][key]
        for pred_mask, gt_mask, concept_idx in zip(pred_masks, gt_masks, concept_indices):
            tp = int(np.logical_and(pred_mask, gt_mask).sum())
            fp = int(np.logical_and(pred_mask, ~gt_mask).sum())
            fn = int(np.logical_and(~pred_mask, gt_mask).sum())
            tn = int(np.logical_and(~pred_mask, ~gt_mask).sum())
            bucket["tp"] += tp
            bucket["fp"] += fp
            bucket["fn"] += fn
            bucket["tn"] += tn
            per_concept = bucket["by_concept"].setdefault(int(concept_idx), [0, 0, 0])
            per_concept[0] += tp
            per_concept[1] += fp
            per_concept[2] += fn


def _ap_from_hist(positive_hist: np.ndarray, negative_hist: np.ndarray) -> float | None:
    positives = int(positive_hist.sum())
    if positives == 0:
        return None
    tp = np.cumsum(positive_hist[::-1], dtype=np.float64)
    fp = np.cumsum(negative_hist[::-1], dtype=np.float64)
    precision = tp / np.maximum(tp + fp, 1.0)
    recall = tp / float(positives)
    return float(np.sum((recall - np.concatenate(([0.0], recall[:-1]))) * precision))


def finalize_segmentation_metrics(state: Dict[str, Any], keys: Sequence[str]) -> Dict[str, Any]:
    positive_hist = state["positive_hist"]
    negative_hist = state["negative_hist"]
    per_concept_ap = [
        _ap_from_hist(positive_hist[index], negative_hist[index])
        for index in range(int(state["n_concepts"]))
    ]
    valid_ap = [value for value in per_concept_ap if value is not None]
    micro_ap = _ap_from_hist(positive_hist.sum(axis=0), negative_hist.sum(axis=0))
    thresholds: Dict[str, Any] = {}
    best_miou = {"threshold": None, "value": None}
    for key in keys:
        bucket = state["thresholds"][key]
        tp, fp, fn, tn = (int(bucket[name]) for name in ("tp", "fp", "fn", "tn"))
        micro_iou = float(tp / max(tp + fp + fn, 1))
        micro_dice = float(2 * tp / max(2 * tp + fp + fn, 1))
        concept_ious = [
            counts[0] / max(sum(counts), 1)
            for counts in bucket["by_concept"].values()
            if counts[0] + counts[2] > 0
        ]
        concept_dice = [
            2 * counts[0] / max(2 * counts[0] + counts[1] + counts[2], 1)
            for counts in bucket["by_concept"].values()
            if counts[0] + counts[2] > 0
        ]
        metric = {
            "pixel_accuracy": float((tp + tn) / max(tp + fp + fn + tn, 1)),
            "foreground_precision": float(tp / max(tp + fp, 1)),
            "foreground_recall": float(tp / max(tp + fn, 1)),
            "micro_iou": micro_iou,
            "micro_dice": micro_dice,
            "mIoU": float(np.mean(concept_ious)) if concept_ious else 0.0,
            "macro_dice": float(np.mean(concept_dice)) if concept_dice else 0.0,
            "concepts_with_gt": len(concept_ious),
        }
        thresholds[key] = metric
        if best_miou["value"] is None or metric["mIoU"] > float(best_miou["value"]):
            best_miou = {"threshold": key, "value": metric["mIoU"]}
    return {
        "pixel_mAP": float(np.mean(valid_ap)) if valid_ap else 0.0,
        "pixel_micro_ap": float(micro_ap) if micro_ap is not None else 0.0,
        "ap_concepts_with_gt": len(valid_ap),
        "threshold_metrics": thresholds,
        "best_mIoU": best_miou,
        "definitions": {
            "mIoU": "Macro mean of generic-part foreground IoU, aggregating TP/FP/FN per concept across the 10k set.",
            "pixel_mAP": "Macro average precision over pixel scores within GT-present image-part pairs; this is semantic pixel AP, not COCO instance-mask AP.",
            "pixel_accuracy": "Pairwise binary pixel accuracy over GT-present image-part masks; background dominance makes it supporting-only.",
        },
    }


def select_targets_for_row(
    row: Dict[str, Any],
    concept_to_idx: Dict[str, int],
    map_h: int,
    map_w: int,
    source: str,
) -> tuple[List[int], np.ndarray]:
    targets_by_concept = row.get("boxes" if source == "boxes" else "segmentations", {})
    if not isinstance(targets_by_concept, dict):
        return [], np.zeros((0, map_h, map_w), dtype=np.bool_)
    concept_indices: List[int] = []
    masks: List[np.ndarray] = []
    width = int(row.get("width") or 0)
    height = int(row.get("height") or 0)
    if width <= 0 or height <= 0:
        return [], np.zeros((0, map_h, map_w), dtype=np.bool_)
    for raw_concept, target in sorted(targets_by_concept.items()):
        concept = _canonicalize(raw_concept)
        concept_idx = concept_to_idx.get(concept)
        if concept_idx is None:
            continue
        mask = (
            rasterize_box_union_in_eval_frame(
                target, image_size=(width, height), map_h=map_h, map_w=map_w
            )
            if source == "boxes"
            else rasterize_polygon_union(target, image_size=(width, height), map_h=map_h, map_w=map_w)
        )
        if mask.any():
            concept_indices.append(concept_idx)
            masks.append(mask)
    if not masks:
        return [], np.zeros((0, map_h, map_w), dtype=np.bool_)
    return concept_indices, np.stack(masks, axis=0)


def main() -> None:
    args = parse_args()
    start = time.time()
    thresholds = parse_float_list(args.activation_thresholds)
    box_iou_thresholds = parse_float_list(args.box_iou_thresholds)
    keys = threshold_keys(args, thresholds)
    if args.distribution_only:
        thresholds = []
        keys = []
    gt_source = "boxes" if args.gt_boxes_jsonl else "segments"
    gt_path = Path(args.gt_boxes_jsonl or args.gt_segments_jsonl)
    gt_rows = load_gt_rows(gt_path)

    run_args, backbone, concept_layer, concepts, test_dataset = load_model(args)
    n_eval = min(len(test_dataset), len(gt_rows))
    if int(args.max_images) > 0:
        n_eval = min(n_eval, int(args.max_images))
    if n_eval < len(test_dataset):
        test_dataset = Subset(test_dataset, list(range(n_eval)))

    concept_to_idx = {concept: idx for idx, concept in enumerate(concepts)}
    state = init_state(keys, box_iou_thresholds)
    segmentation_state = (
        init_segmentation_state(keys, len(concepts))
        if gt_source == "segments" and not args.distribution_only
        else None
    )
    skipped_no_box = 0
    skipped_unmatched = 0
    loader = DataLoader(
        test_dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=str(args.device).startswith("cuda"),
    )

    cursor = 0
    with torch.no_grad():
        for batch_idx, (images, _labels) in enumerate(loader):
            images = images.to(run_args.device, non_blocking=True)
            spatial_maps = forward_concept_maps(
                args.model_name,
                backbone,
                concept_layer,
                images,
                run_args,
                vlg_map_method=args.vlg_map_method,
            )
            spatial_maps = spatial_maps.detach().float()
            if int(spatial_maps.shape[1]) > len(concepts):
                spatial_maps = spatial_maps[:, : len(concepts)]
            elif int(spatial_maps.shape[1]) < len(concepts):
                raise RuntimeError(
                    f"concept maps have fewer channels ({spatial_maps.shape[1]}) than concepts ({len(concepts)})"
                )
            batch_size, _num_concepts, native_map_h, native_map_w = spatial_maps.shape
            map_h = int(args.evaluation_map_size) if int(args.evaluation_map_size) > 0 else native_map_h
            map_w = int(args.evaluation_map_size) if int(args.evaluation_map_size) > 0 else native_map_w
            for local_idx in range(batch_size):
                row_idx = cursor + local_idx
                if row_idx >= n_eval:
                    break
                state["images_seen"] += 1
                row = gt_rows[row_idx]
                if not row.get("boxes" if gt_source == "boxes" else "segmentations"):
                    skipped_no_box += 1
                    continue
                concept_indices, gt_masks = select_targets_for_row(
                    row, concept_to_idx, map_h, map_w, gt_source
                )
                if not concept_indices:
                    skipped_unmatched += 1
                    continue
                state["images_with_targets"] += 1
                raw_maps_t = spatial_maps[local_idx, concept_indices]
                if tuple(raw_maps_t.shape[-2:]) != (map_h, map_w):
                    # Resize only the active GT concepts. Resizing all ~783 maps
                    # to 224x224 would require tens of GB per batch.
                    raw_maps_t = F.interpolate(
                        raw_maps_t[:, None],
                        size=(map_h, map_w),
                        mode="bilinear",
                        align_corners=False,
                    )[:, 0]
                raw_maps_t = raw_maps_t.cpu()
                score_maps_t = normalize_imagenet_maps(raw_maps_t, args.map_normalization)
                distribution_maps_t = normalize_distribution_maps(
                    raw_maps_t, args.distribution_map_normalization
                )
                update_metrics(
                    state,
                    score_maps=score_maps_t.numpy(),
                    raw_maps=distribution_maps_t.numpy(),
                    gt_masks=gt_masks,
                    thresholds=thresholds,
                    keys=keys,
                    threshold_mode=args.threshold_mode,
                    box_iou_thresholds=box_iou_thresholds,
                )
                if segmentation_state is not None:
                    update_segmentation_metrics(
                        segmentation_state,
                        score_maps_t.numpy(),
                        gt_masks,
                        concept_indices,
                        thresholds,
                        keys,
                        args.threshold_mode,
                    )
            cursor += batch_size
            if args.log_every > 0 and (cursor >= n_eval or cursor % int(args.log_every) < batch_size):
                print(
                    f"[pinpp gtbox loc] images={min(cursor, n_eval)}/{n_eval} "
                    f"with_targets={state['images_with_targets']} instances={state['instances']}",
                    flush=True,
                )

    metrics = finalize(state, keys, box_iou_thresholds)
    payload = {
        "dataset": "partimagenetpp",
        "gcbm_path": str(Path(args.gcbm_path)),
        "model_name": args.model_name,
        "localization_source": (
            f"global_head_conv5_{args.vlg_map_method}"
            if str(args.model_name).lower().replace("-", "_") == "vlg_cbm"
            else "native_spatial_maps"
        ),
        "val_manifest": str(Path(args.val_manifest)),
        "gt_source": "human_coco_boxes" if gt_source == "boxes" else "human_coco_segmentation_polygons",
        "gt_annotation_jsonl": str(gt_path),
        "concept_count": len(concepts),
        "map_normalization": args.map_normalization,
        "distribution_map_normalization": args.distribution_map_normalization,
        "distribution_only": bool(args.distribution_only),
        "evaluation_map_size": int(args.evaluation_map_size),
        "threshold_mode": args.threshold_mode,
        "activation_thresholds": thresholds,
        "box_iou_thresholds": box_iou_thresholds,
        "skipped_no_box": skipped_no_box,
        "skipped_unmatched": skipped_unmatched,
        "elapsed_sec": time.time() - start,
        "metrics": metrics,
    }
    if segmentation_state is not None:
        payload["segmentation_metrics"] = finalize_segmentation_metrics(segmentation_state, keys)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload["metrics"], indent=2, sort_keys=True), flush=True)
    print(f"[pinpp gtbox loc] wrote {output}", flush=True)


if __name__ == "__main__":
    main()
