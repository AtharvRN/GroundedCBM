#!/usr/bin/env python3
"""Evaluate SAVLG localization on CUB part annotations.

This script uses a concept->part mapping artifact plus the official CUB part
point annotations. It filters to concepts that were mapped to supported parts,
then evaluates whether each concept's spatial map localizes the corresponding
annotated part(s) for each image.

Primary metrics:
- point_hit: whether the map peak falls within a radius around any mapped part
- mean_normalized_distance: min peak-to-part distance normalized by image diag

Optional thresholded metrics:
- point_in_mask@thr: whether any mapped part point falls inside the thresholded mask
- mask_iou@thr / dice@thr: overlap with a small disk rasterized around the part
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class IndexedDataset(Dataset):
    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        image, target = self.base[idx]
        return idx, image, target


def format_concept(s: str) -> str:
    s = s.lower()
    for token in ["-", ",", ".", "(", ")"]:
        s = s.replace(token, " ")
    if s.startswith("a "):
        s = s[2:]
    elif s.startswith("an "):
        s = s[3:]
    return " ".join(s.split())


def canonicalize_concept_label(s: str) -> str:
    return format_concept(s)


def _parse_csv_floats(x: str):
    return [float(v.strip()) for v in x.split(",") if v.strip()]


def _normalize_map_with_mode(maps: torch.Tensor, mode: str) -> torch.Tensor:
    maps = maps.detach().float()
    if mode == "sigmoid":
        return torch.sigmoid(maps)
    if mode == "concept_zscore_minmax":
        mean = maps.flatten(1).mean(dim=1).view(-1, 1, 1)
        std = maps.flatten(1).std(dim=1, unbiased=False).view(-1, 1, 1).clamp_min(1e-6)
        maps = (maps - mean) / std
    elif mode != "minmax":
        raise ValueError(f"Unsupported map normalization mode: {mode}")
    mins = maps.flatten(1).min(dim=1).values.view(-1, 1, 1)
    maxs = maps.flatten(1).max(dim=1).values.view(-1, 1, 1)
    return (maps - mins) / (maxs - mins).clamp_min(1e-6)


def resolve_base_index(ds: Dataset, idx: int) -> int:
    if isinstance(ds, torch.utils.data.Subset):
        return resolve_base_index(ds.dataset, int(ds.indices[idx]))
    indices = getattr(ds, "indices", None)
    base_dataset = getattr(ds, "base_dataset", None)
    if indices is not None and base_dataset is not None:
        return resolve_base_index(base_dataset, int(indices[idx]))
    return int(idx)


def sample_path_from_dataset(ds: Dataset, idx: int) -> str:
    if isinstance(ds, torch.utils.data.Subset):
        return sample_path_from_dataset(ds.dataset, int(ds.indices[idx]))
    base_dataset = getattr(ds, "base_dataset", None)
    indices = getattr(ds, "indices", None)
    if base_dataset is not None and indices is not None:
        return sample_path_from_dataset(base_dataset, int(indices[idx]))
    samples = getattr(ds, "samples", None) or getattr(ds, "imgs", None)
    if samples is None:
        raise RuntimeError("Dataset does not expose base_dataset/indices or samples/imgs")
    return str(samples[int(idx)][0])


def load_images_index(images_txt: Path) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    for line in images_txt.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        image_id_str, relpath = line.split(" ", 1)
        tail = "/".join(Path(relpath).parts[-2:])
        mapping[tail] = int(image_id_str)
    return mapping


def load_parts(parts_txt: Path) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for line in parts_txt.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        part_id_str, name = line.split(" ", 1)
        out[int(part_id_str)] = name.strip()
    return out


def load_part_locs(part_locs_txt: Path, part_names: Dict[int, str]) -> Dict[int, Dict[str, Tuple[float, float]]]:
    out: Dict[int, Dict[str, Tuple[float, float]]] = {}
    for line in part_locs_txt.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        image_id_str, part_id_str, x_str, y_str, visible_str = line.split()
        if int(visible_str) != 1:
            continue
        image_id = int(image_id_str)
        part_id = int(part_id_str)
        out.setdefault(image_id, {})[part_names[part_id]] = (float(x_str), float(y_str))
    return out


def load_mapping(mapping_json: Path) -> Dict[str, List[str]]:
    payload = json.loads(mapping_json.read_text())
    out: Dict[str, List[str]] = {}
    items = payload.get("mappings") or payload.get("concepts") or []
    for item in items:
        if item.get("keep"):
            out[canonicalize_concept_label(item["concept"])] = list(item.get("exact_parts", []))
    return out


def disk_mask(h: int, w: int, center_x: float, center_y: float, radius_px: float) -> torch.Tensor:
    ys = torch.arange(h, dtype=torch.float32).unsqueeze(1)
    xs = torch.arange(w, dtype=torch.float32).unsqueeze(0)
    dist2 = (xs - float(center_x)) ** 2 + (ys - float(center_y)) ** 2
    return dist2 <= float(radius_px) ** 2


def merged_disk_mask(h: int, w: int, points: Sequence[Tuple[float, float]], radius_px: float) -> torch.Tensor:
    mask = torch.zeros((h, w), dtype=torch.bool)
    for x, y in points:
        mask |= disk_mask(h, w, x, y, radius_px)
    return mask


def point_in_any_disk(px: int, py: int, points: Sequence[Tuple[float, float]], radius_px: float) -> bool:
    for x, y in points:
        if (float(px) - float(x)) ** 2 + (float(py) - float(y)) ** 2 <= float(radius_px) ** 2:
            return True
    return False


def min_normalized_distance(px: int, py: int, points: Sequence[Tuple[float, float]], diag: float) -> float:
    best = None
    for x, y in points:
        d = math.sqrt((float(px) - float(x)) ** 2 + (float(py) - float(y)) ** 2) / max(diag, 1e-8)
        best = d if best is None else min(best, d)
    return 1.0 if best is None else float(best)


def build_dataset_image_ids(ds: Dataset, images_index: Dict[str, int]) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for ds_idx in range(len(ds)):
        sample_path = sample_path_from_dataset(ds, ds_idx)
        tail = "/".join(Path(sample_path).parts[-2:])
        image_id = images_index.get(tail)
        if image_id is not None:
            out[int(ds_idx)] = int(image_id)
    return out


def preload_mapped_gt_concepts(
    ann_split_dir: Path,
    dataset_base_indices: Sequence[int],
    image_ids_by_ds_idx: Dict[int, int],
    image_part_names_by_id: Dict[int, set[str]],
    concept_to_parts: Dict[str, List[str]],
    concept_to_idx: Dict[str, int],
) -> Dict[int, List[Tuple[str, List[str]]]]:
    preloaded: Dict[int, List[Tuple[str, List[str]]]] = {}
    needed_base = set(int(x) for x in dataset_base_indices)
    for ann_path in ann_split_dir.glob("*.json"):
        try:
            base_idx = int(ann_path.stem)
        except ValueError:
            continue
        if base_idx not in needed_base:
            continue
        ds_idx = base_idx
        image_id = image_ids_by_ds_idx.get(ds_idx)
        if image_id is None:
            continue
        visible_parts = image_part_names_by_id.get(image_id, set())
        if not visible_parts:
            continue
        payload = json.loads(ann_path.read_text())
        gt_concepts: List[Tuple[str, List[str]]] = []
        for ann in payload[1:]:
            label = ann.get("label")
            if not isinstance(label, str):
                continue
            label = canonicalize_concept_label(label)
            if label in concept_to_parts and label in concept_to_idx:
                exact_parts = [p for p in concept_to_parts[label] if p in visible_parts]
                if exact_parts:
                    gt_concepts.append((label, exact_parts))
        if gt_concepts:
            preloaded[base_idx] = gt_concepts
    return preloaded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_path", type=str, required=True)
    parser.add_argument("--annotation_dir", type=str, required=True)
    parser.add_argument(
        "--annotation_cache_json",
        type=str,
        default=None,
        help="Optional precomputed part-aligned annotation cache JSON created by precompute_cub_part_annotation_cache.py",
    )
    parser.add_argument("--cub_root", type=str, required=True, help="Path to local CUB_200_2011 root")
    parser.add_argument("--mapping_json", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--map_normalization", type=str, default="concept_zscore_minmax")
    parser.add_argument("--point_source", type=str, default="normalized_map", choices=["normalized_map", "pred_dist"])
    parser.add_argument("--activation_thresholds", type=str, default="0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--radius_fracs", type=str, default="0.01,0.02,0.05,0.1")
    parser.add_argument("--disk_radius_frac", type=float, default=0.03)
    return parser.parse_args()


def main() -> None:
    args_ns = parse_args()
    from gcbm.savlg_eval_common import _load_args, _load_concepts
    from methods.savlg import (
        build_savlg_concept_layer,
        create_savlg_splits,
        forward_savlg_backbone,
        forward_savlg_concept_layer,
    )

    args = _load_args(args_ns.load_path, args_ns.device, args_ns.annotation_dir)
    if getattr(args, "skip_test_eval", False):
        print(
            "[cub-parts] overriding saved skip_test_eval=True to force evaluation on dataset_val",
            flush=True,
        )
        args.skip_test_eval = False
    _, _, _, _, test_dataset, backbone = create_savlg_splits(args)
    if args_ns.max_images is not None:
        keep = min(args_ns.max_images, len(test_dataset))
        test_dataset = torch.utils.data.Subset(test_dataset, list(range(keep)))
    dataset = IndexedDataset(test_dataset)

    cub_root = Path(args_ns.cub_root)
    images_index = load_images_index(cub_root / "images.txt")
    part_names = load_parts(cub_root / "parts" / "parts.txt")
    part_locs = load_part_locs(cub_root / "parts" / "part_locs.txt", part_names)
    concept_to_parts = load_mapping(Path(args_ns.mapping_json))

    concepts = _load_concepts(args_ns.load_path, args)
    concept_to_idx = {canonicalize_concept_label(name): idx for idx, name in enumerate(concepts)}

    concept_layer = build_savlg_concept_layer(args, backbone, len(concepts)).to(args.device)
    concept_layer.load_state_dict(torch.load(os.path.join(args_ns.load_path, "concept_layer.pt"), map_location=args.device))
    concept_layer.eval()
    backbone.eval()

    loader = DataLoader(dataset, batch_size=args_ns.batch_size, shuffle=False, num_workers=args_ns.num_workers, pin_memory=False)

    thresholds = _parse_csv_floats(args_ns.activation_thresholds)
    radii = _parse_csv_floats(args_ns.radius_fracs)
    threshold_tensor = torch.tensor(thresholds, dtype=torch.float32)

    point_hits_sum = {r: 0.0 for r in radii}
    point_hits_count = 0
    mean_dist_sum = 0.0
    mean_dist_count = 0
    threshold_point_hits_sum = {thr: 0.0 for thr in thresholds}
    threshold_mask_iou_sum = {thr: 0.0 for thr in thresholds}
    threshold_dice_sum = {thr: 0.0 for thr in thresholds}
    threshold_count = {thr: 0 for thr in thresholds}
    threshold_tp = {thr: 0 for thr in thresholds}
    threshold_fp = {thr: 0 for thr in thresholds}
    threshold_fn = {thr: 0 for thr in thresholds}

    ann_split_dir = Path(args.annotation_dir) / f"{args.dataset}_test"
    if not ann_split_dir.is_dir():
        ann_split_dir = Path(args.annotation_dir) / f"{args.dataset}_val"

    dataset_base_indices = [resolve_base_index(dataset.base, i) for i in range(len(dataset))]
    image_ids_by_ds_idx = build_dataset_image_ids(dataset.base, images_index)
    image_part_names_by_id = {img_id: set(parts.keys()) for img_id, parts in part_locs.items()}
    if args_ns.annotation_cache_json:
        cache_payload = json.loads(Path(args_ns.annotation_cache_json).read_text())
        mapped_gt_concepts_by_base_idx = {
            int(base_idx): [(str(item["label"]), list(item["exact_parts"])) for item in items]
            for base_idx, items in (cache_payload.get("items_by_base_idx") or {}).items()
        }
    else:
        mapped_gt_concepts_by_base_idx = preload_mapped_gt_concepts(
            ann_split_dir=ann_split_dir,
            dataset_base_indices=dataset_base_indices,
            image_ids_by_ds_idx=image_ids_by_ds_idx,
            image_part_names_by_id=image_part_names_by_id,
            concept_to_parts=concept_to_parts,
            concept_to_idx=concept_to_idx,
        )

    with torch.no_grad():
        for batch in tqdm(loader, desc="cub part eval"):
            indices, images, _targets = batch
            images = images.to(args.device, non_blocking=True)
            feats = forward_savlg_backbone(backbone, images, args)
            _global_outputs, spatial_maps = forward_savlg_concept_layer(concept_layer, feats)
            img_h, img_w = int(images.shape[-2]), int(images.shape[-1])

            for b, ds_idx in enumerate(indices.tolist()):
                base_idx = resolve_base_index(dataset.base, ds_idx)
                image_id = image_ids_by_ds_idx.get(int(ds_idx))
                if image_id is None:
                    continue
                image_parts = part_locs.get(image_id, {})
                if not image_parts:
                    continue

                gt_concepts = mapped_gt_concepts_by_base_idx.get(int(base_idx), [])
                if not gt_concepts:
                    continue

                concept_idx_tensor = torch.as_tensor([concept_to_idx[label] for label, _ in gt_concepts], device=spatial_maps.device, dtype=torch.long)
                maps_k_native = spatial_maps[b].index_select(0, concept_idx_tensor)
                maps_k = F.interpolate(
                    maps_k_native.unsqueeze(1),
                    size=(img_h, img_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
                if args_ns.point_source == "pred_dist":
                    score_maps = F.softmax(maps_k.flatten(1), dim=1).view_as(maps_k)
                else:
                    score_maps = _normalize_map_with_mode(maps_k, args_ns.map_normalization)

                argmax_flat = score_maps.flatten(1).argmax(dim=1)
                argmax_y = (argmax_flat // score_maps.shape[-1]).cpu().tolist()
                argmax_x = (argmax_flat % score_maps.shape[-1]).cpu().tolist()
                score_maps_cpu = score_maps.cpu()
                diag = math.sqrt(float(img_h * img_h + img_w * img_w))
                disk_radius_px = float(args_ns.disk_radius_frac) * diag

                points_per_concept: List[List[Tuple[float, float]]] = []
                gt_masks: List[torch.Tensor] = []
                point_indicator_masks: List[torch.Tensor] = []

                for (_label, exact_parts), px, py in zip(gt_concepts, argmax_x, argmax_y):
                    points = [image_parts[p] for p in exact_parts]
                    points_per_concept.append(points)
                    mean_dist_sum += min_normalized_distance(int(px), int(py), points, diag)
                    mean_dist_count += 1
                    point_hits_count += 1
                    for r in radii:
                        point_hits_sum[r] += 1.0 if point_in_any_disk(int(px), int(py), points, float(r) * diag) else 0.0

                    gt_masks.append(merged_disk_mask(img_h, img_w, points, disk_radius_px))
                    point_mask = torch.zeros((img_h, img_w), dtype=torch.bool)
                    for x, y in points:
                        xi = int(round(x))
                        yi = int(round(y))
                        if 0 <= yi < img_h and 0 <= xi < img_w:
                            point_mask[yi, xi] = True
                    point_indicator_masks.append(point_mask)

                if not gt_masks:
                    continue

                gt_masks_tensor = torch.stack(gt_masks, dim=0)
                point_indicator_tensor = torch.stack(point_indicator_masks, dim=0)
                pred_masks = score_maps_cpu.unsqueeze(0) >= threshold_tensor[:, None, None, None]
                gt_masks_exp = gt_masks_tensor.unsqueeze(0)
                point_indicator_exp = point_indicator_tensor.unsqueeze(0)

                inter = (pred_masks & gt_masks_exp).flatten(2).sum(dim=2)
                pred_sum = pred_masks.flatten(2).sum(dim=2)
                gt_sum = gt_masks_exp.flatten(2).sum(dim=2)
                union = (pred_masks | gt_masks_exp).flatten(2).sum(dim=2)
                mask_iou_vals = torch.where(union > 0, inter.float() / union.float(), torch.zeros_like(union, dtype=torch.float32))
                dice_vals = torch.where(
                    (pred_sum + gt_sum) > 0,
                    (2.0 * inter.float()) / (pred_sum + gt_sum).float(),
                    torch.zeros_like(pred_sum, dtype=torch.float32),
                )
                fp_vals = (pred_masks & ~gt_masks_exp).flatten(2).sum(dim=2)
                fn_vals = (~pred_masks & gt_masks_exp).flatten(2).sum(dim=2)
                point_in_mask_vals = (pred_masks & point_indicator_exp).flatten(2).any(dim=2).float()

                num_instances = gt_masks_tensor.shape[0]
                for i, thr in enumerate(thresholds):
                    threshold_count[thr] += int(num_instances)
                    threshold_point_hits_sum[thr] += float(point_in_mask_vals[i].sum().item())
                    threshold_mask_iou_sum[thr] += float(mask_iou_vals[i].sum().item())
                    threshold_dice_sum[thr] += float(dice_vals[i].sum().item())
                    threshold_tp[thr] += int(inter[i].sum().item())
                    threshold_fp[thr] += int(fp_vals[i].sum().item())
                    threshold_fn[thr] += int(fn_vals[i].sum().item())

    results = {
        "load_path": args_ns.load_path,
        "mapping_json": args_ns.mapping_json,
        "cub_root": args_ns.cub_root,
        "point_source": args_ns.point_source,
        "map_normalization": args_ns.map_normalization,
        "disk_radius_frac": args_ns.disk_radius_frac,
        "num_images": len(dataset),
        "num_gt_instances": mean_dist_count,
        "point_metrics": {
            "mean_normalized_distance": float(mean_dist_sum / max(mean_dist_count, 1)),
            "point_hit": {str(r): float(point_hits_sum[r] / max(point_hits_count, 1)) for r in radii},
        },
        "threshold_metrics": {},
    }

    best_mask_iou = {"threshold": None, "value": None}
    best_dice = {"threshold": None, "value": None}
    best_point_in_mask = {"threshold": None, "value": None}
    for thr in thresholds:
        point_in_mask = float(threshold_point_hits_sum[thr] / max(threshold_count[thr], 1))
        mask_iou = float(threshold_mask_iou_sum[thr] / max(threshold_count[thr], 1))
        dice = float(threshold_dice_sum[thr] / max(threshold_count[thr], 1))
        tp = int(threshold_tp[thr])
        fp = int(threshold_fp[thr])
        fn = int(threshold_fn[thr])
        precision = float(tp / max(tp + fp, 1))
        recall = float(tp / max(tp + fn, 1))
        f1 = float((2.0 * precision * recall) / max(precision + recall, 1e-12))
        results["threshold_metrics"][str(thr)] = {
            "point_in_mask": point_in_mask,
            "mask_iou": mask_iou,
            "dice": dice,
            "pixel_counts": {"tp": tp, "fp": fp, "fn": fn},
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        if best_mask_iou["value"] is None or mask_iou > best_mask_iou["value"]:
            best_mask_iou = {"threshold": thr, "value": mask_iou}
        if best_dice["value"] is None or dice > best_dice["value"]:
            best_dice = {"threshold": thr, "value": dice}
        if best_point_in_mask["value"] is None or point_in_mask > best_point_in_mask["value"]:
            best_point_in_mask = {"threshold": thr, "value": point_in_mask}

    results["best_mask_iou"] = best_mask_iou
    results["best_dice"] = best_dice
    results["best_point_in_mask"] = best_point_in_mask

    out = Path(args_ns.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
