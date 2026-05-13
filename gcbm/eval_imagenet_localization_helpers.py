import argparse
import json
import sys
import time
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gcbm.imagenet_core import (
    Config,
    SafeImageFolderWithAnnotations,
    amp_dtype,
    build_loader,
    build_model,
    configure_runtime,
    load_concepts,
    prepare_images,
    split_train_val,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate standalone ImageNet SAVLG spatial maps against precomputed concept masks."
    )
    parser.add_argument("--artifact_dir", required=True, help="Run directory with config.json and concept_head_best.pt.")
    parser.add_argument("--train_root", default="", help="Override ImageNet train root. Required if config path is absent.")
    parser.add_argument(
        "--precomputed_target_dir",
        default="",
        help="Override precomputed target root. Falls back to artifact_dir/precomputed_targets.",
    )
    parser.add_argument("--output_json", default="")
    parser.add_argument("--split", default="val", choices=["train", "val"], help="Evaluate the train split or held-out val split.")
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--activation_thresholds", default="0.3,0.5,0.7,0.9,mean")
    parser.add_argument("--box_iou_thresholds", default="0.1,0.3,0.5")
    parser.add_argument(
        "--map_normalization",
        default="concept_zscore_minmax",
        choices=["minmax", "sigmoid", "concept_zscore_minmax"],
    )
    parser.add_argument("--gt_threshold", type=float, default=0.0, help="GT soft-box cells > threshold are treated as positive.")
    parser.add_argument("--log_every", type=int, default=25)
    return parser.parse_args()


def load_config(artifact_dir: Path, args: argparse.Namespace) -> Config:
    payload = json.loads((artifact_dir / "config.json").read_text())
    payload.setdefault("feature_storage_dtype", "fp16")
    payload.setdefault("saga_table_device", "cpu")
    payload.setdefault("dense_lr", 1e-3)
    payload.setdefault("dense_n_iters", 20)
    payload.setdefault("train_random_transforms", True)
    payload.setdefault("learn_spatial_residual_scale", False)
    payload["device"] = args.device
    payload["batch_size"] = int(args.batch_size)
    payload["workers"] = int(args.workers)
    payload["prefetch_factor"] = int(args.prefetch_factor)
    payload["skip_final_layer"] = True
    payload["print_config"] = False
    if args.train_root:
        payload["train_root"] = args.train_root
    if args.precomputed_target_dir:
        payload["precomputed_target_dir"] = args.precomputed_target_dir
    valid_fields = {field.name for field in fields(Config)}
    return Config(**{key: value for key, value in payload.items() if key in valid_fields})


def resolve_precomputed_target_dir(artifact_dir: Path, cfg: Config, override: str) -> Path:
    candidates: List[Path] = []
    if override:
        candidates.append(Path(override))
    candidates.extend(
        [
            artifact_dir / "precomputed_targets",
            Path(cfg.precomputed_target_dir) if cfg.precomputed_target_dir else Path(),
        ]
    )
    for candidate in candidates:
        if str(candidate) and (candidate / "train" / "metadata.json").is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not find precomputed targets. Pass --precomputed_target_dir or copy "
        "precomputed_targets into the artifact directory."
    )


def resolve_train_root(cfg: Config) -> Path:
    train_root = Path(cfg.train_root)
    if train_root.is_dir():
        return train_root.resolve()
    raise FileNotFoundError(
        f"ImageNet train root is not available: {train_root}. "
        "Pass --train_root with an extracted ImageNet train directory."
    )


def parse_thresholds(raw: str) -> Tuple[List[float], bool]:
    values: List[float] = []
    include_mean = False
    for token in raw.split(","):
        token = token.strip().lower()
        if not token:
            continue
        if token == "mean":
            include_mean = True
        else:
            values.append(float(token))
    return values, include_mean


def normalize_maps(maps: torch.Tensor, mode: str) -> torch.Tensor:
    maps = maps.float()
    if mode == "sigmoid":
        return torch.sigmoid(maps)
    if mode == "concept_zscore_minmax":
        mean = maps.mean(dim=(1, 2), keepdim=True)
        std = maps.std(dim=(1, 2), keepdim=True, unbiased=False).clamp_min(1e-6)
        maps = (maps - mean) / std
    min_v = maps.amin(dim=(1, 2), keepdim=True)
    max_v = maps.amax(dim=(1, 2), keepdim=True)
    return (maps - min_v) / (max_v - min_v).clamp_min(1e-6)


def boxes_from_masks(masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    valid = masks.flatten(1).any(dim=1)
    rows_any = masks.any(dim=2)
    cols_any = masks.any(dim=1)
    y1 = rows_any.float().argmax(dim=1)
    x1 = cols_any.float().argmax(dim=1)
    y2 = masks.shape[1] - rows_any.flip(1).float().argmax(dim=1)
    x2 = masks.shape[2] - cols_any.flip(1).float().argmax(dim=1)
    boxes = torch.stack([x1, y1, x2, y2], dim=1).float()
    return boxes, valid


def box_iou(pred_boxes: torch.Tensor, pred_valid: torch.Tensor, gt_boxes: torch.Tensor, gt_valid: torch.Tensor) -> torch.Tensor:
    ix1 = torch.maximum(pred_boxes[:, 0], gt_boxes[:, 0])
    iy1 = torch.maximum(pred_boxes[:, 1], gt_boxes[:, 1])
    ix2 = torch.minimum(pred_boxes[:, 2], gt_boxes[:, 2])
    iy2 = torch.minimum(pred_boxes[:, 3], gt_boxes[:, 3])
    inter = (ix2 - ix1).clamp_min(0) * (iy2 - iy1).clamp_min(0)
    area_pred = (pred_boxes[:, 2] - pred_boxes[:, 0]).clamp_min(0) * (pred_boxes[:, 3] - pred_boxes[:, 1]).clamp_min(0)
    area_gt = (gt_boxes[:, 2] - gt_boxes[:, 0]).clamp_min(0) * (gt_boxes[:, 3] - gt_boxes[:, 1]).clamp_min(0)
    union = area_pred + area_gt - inter
    valid = pred_valid & gt_valid & (union > 0)
    out = torch.zeros_like(inter)
    out[valid] = inter[valid] / union[valid]
    return out


def update_threshold_metrics(
    metrics: Dict[str, Any],
    key: str,
    score_maps: torch.Tensor,
    pred_masks: torch.Tensor,
    gt_masks: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_box_valid: torch.Tensor,
    box_iou_thresholds: Sequence[float],
) -> torch.Tensor:
    pred_masks = pred_masks.bool()
    gt_masks = gt_masks.bool()
    inter = (pred_masks & gt_masks).flatten(1).sum(dim=1).float()
    pred_sum = pred_masks.flatten(1).sum(dim=1).float()
    gt_sum = gt_masks.flatten(1).sum(dim=1).float()
    union = pred_sum + gt_sum - inter
    iou = torch.where(union > 0, inter / union, torch.zeros_like(union))
    dice = torch.where(pred_sum + gt_sum > 0, 2.0 * inter / (pred_sum + gt_sum).clamp_min(1), torch.zeros_like(inter))
    argmax = score_maps.flatten(1).argmax(dim=1)
    point_hit = gt_masks.flatten(1).gather(1, argmax[:, None]).squeeze(1).float()
    pred_boxes, pred_box_valid = boxes_from_masks(pred_masks)
    box_ious = box_iou(pred_boxes, pred_box_valid, gt_boxes, gt_box_valid)

    state = metrics.setdefault(
        key,
        {
            "instances": 0,
            "mask_iou_sum": 0.0,
            "dice_sum": 0.0,
            "point_hit_sum": 0.0,
            "box_iou_sum": 0.0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "box_acc_counts": {str(t): 0 for t in box_iou_thresholds},
        },
    )
    count = int(gt_masks.shape[0])
    state["instances"] += count
    state["mask_iou_sum"] += float(iou.sum().item())
    state["dice_sum"] += float(dice.sum().item())
    state["point_hit_sum"] += float(point_hit.sum().item())
    state["box_iou_sum"] += float(box_ious.sum().item())
    state["tp"] += int(inter.sum().item())
    state["fp"] += int((pred_sum - inter).sum().item())
    state["fn"] += int((gt_sum - inter).sum().item())
    for threshold in box_iou_thresholds:
        state["box_acc_counts"][str(threshold)] += int((box_ious >= float(threshold)).sum().item())
    return box_ious


def finalize_threshold_metrics(raw: Dict[str, Any], box_iou_thresholds: Sequence[float]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, state in raw.items():
        instances = max(int(state["instances"]), 1)
        tp = int(state["tp"])
        fp = int(state["fp"])
        fn = int(state["fn"])
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        out[key] = {
            "instances": int(state["instances"]),
            "mask_iou": float(state["mask_iou_sum"] / instances),
            "dice": float(state["dice_sum"] / instances),
            "point_hit": float(state["point_hit_sum"] / instances),
            "box_iou": float(state["box_iou_sum"] / instances),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float((2.0 * precision * recall) / max(precision + recall, 1e-12)),
            "box_acc": {
                str(threshold): float(state["box_acc_counts"][str(threshold)] / instances)
                for threshold in box_iou_thresholds
            },
            "pixel_counts": {"tp": tp, "fp": fp, "fn": fn},
        }
    return out


@torch.no_grad()
def evaluate(backbone: torch.nn.Module, head: torch.nn.Module, loader: torch.utils.data.DataLoader, cfg: Config, args: argparse.Namespace) -> Dict[str, Any]:
    thresholds, include_mean_threshold = parse_thresholds(args.activation_thresholds)
    box_iou_thresholds = [float(x.strip()) for x in args.box_iou_thresholds.split(",") if x.strip()]
    threshold_raw: Dict[str, Any] = {}
    distribution = {
        "instances": 0,
        "soft_iou_sum": 0.0,
        "mass_in_gt_sum": 0.0,
        "point_hit_sum": 0.0,
    }
    images_seen = 0
    images_with_targets = 0
    start = time.perf_counter()
    backbone.eval()
    head.eval()
    for step, batch in enumerate(loader, start=1):
        images = prepare_images(batch["images"], cfg)
        with torch.autocast(
            device_type="cuda",
            dtype=amp_dtype(cfg.amp),
            enabled=(str(cfg.device).startswith("cuda") and amp_dtype(cfg.amp) is not None),
        ):
            feats = backbone(images)
            outputs = head(feats)
            spatial_maps = F.interpolate(
                outputs["spatial_maps"],
                size=batch["mask_targets"].shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).float()
        images_seen += int(images.shape[0])
        mask_indices = batch["mask_indices"].to(cfg.device, non_blocking=cfg.pin_memory)
        mask_targets = batch["mask_targets"].to(cfg.device, non_blocking=cfg.pin_memory).float()
        mask_valid = batch["mask_valid"].to(cfg.device, non_blocking=cfg.pin_memory)

        for batch_index in range(spatial_maps.shape[0]):
            valid = mask_valid[batch_index]
            if not bool(valid.any()):
                continue
            concept_ids = mask_indices[batch_index][valid]
            gt = mask_targets[batch_index][valid]
            target_mass = gt.flatten(1).sum(dim=1)
            target_valid = target_mass > 0
            if not bool(target_valid.any()):
                continue
            images_with_targets += 1
            concept_ids = concept_ids[target_valid]
            gt = gt[target_valid]
            pred = spatial_maps[batch_index].index_select(0, concept_ids)
            gt_masks = gt > float(args.gt_threshold)
            gt_boxes, gt_box_valid = boxes_from_masks(gt_masks)
            score_maps = normalize_maps(pred, args.map_normalization)

            pred_dist = F.softmax(pred.flatten(1), dim=1).view_as(pred)
            gt_dist = gt.flatten(1) / gt.flatten(1).sum(dim=1, keepdim=True).clamp_min(1e-6)
            pred_dist_flat = pred_dist.flatten(1)
            soft_inter = torch.minimum(pred_dist_flat, gt_dist).sum(dim=1)
            soft_union = torch.maximum(pred_dist_flat, gt_dist).sum(dim=1).clamp_min(1e-6)
            argmax = pred_dist_flat.argmax(dim=1)
            dist_point_hit = gt_masks.flatten(1).gather(1, argmax[:, None]).squeeze(1).float()
            distribution["instances"] += int(gt.shape[0])
            distribution["soft_iou_sum"] += float((soft_inter / soft_union).sum().item())
            distribution["mass_in_gt_sum"] += float((pred_dist * gt_masks.float()).flatten(1).sum(dim=1).sum().item())
            distribution["point_hit_sum"] += float(dist_point_hit.sum().item())

            for threshold in thresholds:
                key = str(threshold)
                pred_masks = score_maps >= float(threshold)
                update_threshold_metrics(
                    threshold_raw,
                    key,
                    score_maps,
                    pred_masks,
                    gt_masks,
                    gt_boxes,
                    gt_box_valid,
                    box_iou_thresholds,
                )
            if include_mean_threshold:
                pred_masks = score_maps >= score_maps.mean(dim=(1, 2), keepdim=True)
                update_threshold_metrics(
                    threshold_raw,
                    "mean",
                    score_maps,
                    pred_masks,
                    gt_masks,
                    gt_boxes,
                    gt_box_valid,
                    box_iou_thresholds,
                )

        if args.log_every > 0 and step % int(args.log_every) == 0:
            elapsed = time.perf_counter() - start
            instances = int(distribution["instances"])
            print(
                f"[loc-eval] step={step}/{len(loader)} images={images_seen} "
                f"instances={instances} ips={images_seen / max(elapsed, 1e-6):.2f}",
                flush=True,
            )

    elapsed = time.perf_counter() - start
    instances = max(int(distribution["instances"]), 1)
    return {
        "images_seen": images_seen,
        "images_with_targets": images_with_targets,
        "instances": int(distribution["instances"]),
        "elapsed_sec": elapsed,
        "images_per_second": images_seen / max(elapsed, 1e-6),
        "distribution_metrics": {
            "soft_iou": float(distribution["soft_iou_sum"] / instances),
            "mass_in_gt": float(distribution["mass_in_gt_sum"] / instances),
            "point_hit": float(distribution["point_hit_sum"] / instances),
        },
        "threshold_metrics": finalize_threshold_metrics(threshold_raw, box_iou_thresholds),
    }


def main() -> None:
    args = parse_args()
    artifact_dir = Path(args.artifact_dir).resolve()
    output_json = Path(args.output_json).resolve() if args.output_json else artifact_dir / f"localization_{args.split}.json"
    cfg = load_config(artifact_dir, args)
    precomputed_target_dir = resolve_precomputed_target_dir(artifact_dir, cfg, args.precomputed_target_dir)
    cfg.precomputed_target_dir = str(precomputed_target_dir)
    cfg.train_root = str(resolve_train_root(cfg))
    configure_runtime(cfg)

    precomputed_concepts = precomputed_target_dir / "concepts.txt"
    base_concepts = load_concepts(str(precomputed_concepts if precomputed_concepts.exists() else Path(cfg.concept_file)))
    checkpoint_concepts = load_concepts(str(artifact_dir / "concepts.txt"))

    dataset_full = SafeImageFolderWithAnnotations(
        root=cfg.train_root,
        annotation_dir=cfg.annotation_dir,
        concepts=base_concepts,
        input_size=cfg.input_size,
        min_image_bytes=cfg.min_image_bytes,
        split="train",
        manifest=cfg.train_manifest,
    )
    dataset_full.attach_precomputed_targets(cfg.precomputed_target_dir, cfg)
    filter_summary_path = artifact_dir / "concept_filter_summary.json"
    concept_filter_summary: Optional[Dict[str, Any]] = None
    if filter_summary_path.exists():
        concept_filter_summary = json.loads(filter_summary_path.read_text())
        keep_indices = concept_filter_summary.get("keep_indices", [])
        if keep_indices:
            dataset_full.apply_concept_filter(keep_indices)
    if list(dataset_full.concepts) != checkpoint_concepts:
        raise ValueError(
            f"Concept mismatch after filtering: dataset has {len(dataset_full.concepts)} concepts, "
            f"checkpoint has {len(checkpoint_concepts)} concepts."
        )

    train_dataset, val_dataset = split_train_val(
        dataset_full,
        val_split=cfg.val_split,
        max_train_images=cfg.max_train_images,
        max_val_images=cfg.max_val_images,
        seed=cfg.seed,
    )
    eval_dataset = train_dataset if args.split == "train" else val_dataset
    if args.max_images > 0:
        eval_dataset = torch.utils.data.Subset(eval_dataset, list(range(min(int(args.max_images), len(eval_dataset)))))

    loader = build_loader(eval_dataset, cfg, shuffle=False, drop_last=False)
    backbone, head = build_model(cfg, n_concepts=len(checkpoint_concepts))
    head.load_state_dict(torch.load(artifact_dir / "concept_head_best.pt", map_location=cfg.device))

    payload = {
        "artifact_dir": str(artifact_dir),
        "train_root": cfg.train_root,
        "precomputed_target_dir": str(precomputed_target_dir),
        "split": args.split,
        "max_images": int(args.max_images),
        "n_concepts": len(checkpoint_concepts),
        "concept_filter": concept_filter_summary,
        "config": {
            "batch_size": int(cfg.batch_size),
            "workers": int(cfg.workers),
            "device": cfg.device,
            "map_normalization": args.map_normalization,
            "activation_thresholds": args.activation_thresholds,
            "box_iou_thresholds": args.box_iou_thresholds,
            "gt_threshold": float(args.gt_threshold),
        },
        "metrics": evaluate(backbone, head, loader, cfg, args),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
