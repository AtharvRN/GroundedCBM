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
from PIL import Image

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


def _normalize_map_with_mode(
    maps: torch.Tensor,
    mode: str,
    proj_mean: Optional[torch.Tensor] = None,
    proj_std: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    maps = maps.detach().float()
    if mode == "sigmoid":
        return torch.sigmoid(maps)
    if mode == "concept_zscore_minmax":
        mode = "proj_zscore_minmax"
    if mode == "proj_zscore_minmax":
        if proj_mean is None or proj_std is None:
            raise RuntimeError("proj_zscore_minmax requires proj_mean.pt/proj_std.pt in the checkpoint directory.")
        maps = (maps - proj_mean.to(maps.device)) / proj_std.to(maps.device)
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


def resize_short_edge_size(image_size: Tuple[int, int], resize_size: int) -> Tuple[int, int]:
    width, height = image_size
    if width <= 0 or height <= 0:
        return int(resize_size), int(resize_size)
    if width == height:
        return int(resize_size), int(resize_size)
    if width < height:
        return int(resize_size), int(resize_size * height / width)
    return int(resize_size * width / height), int(resize_size)


def infer_resize_size(backbone_name: str, crop_size: int) -> int:
    if backbone_name == "resnet50_cub_mm" or crop_size == 448:
        return 600
    return 256


def transform_point_for_model_input(
    point: Tuple[float, float],
    image_size: Tuple[int, int],
    crop_size: int,
    resize_size: int,
) -> Optional[Tuple[float, float]]:
    # CUB part annotations are in original image pixels. The model sees the
    # deterministic evaluation transform, so apply Resize(short edge) +
    # CenterCrop before comparing points to upsampled concept maps.
    width, height = image_size
    if width <= 0 or height <= 0:
        return None
    resized_width, resized_height = resize_short_edge_size(image_size, resize_size)
    scale_x = resized_width / float(width)
    scale_y = resized_height / float(height)
    x = float(point[0]) * scale_x
    y = float(point[1]) * scale_y
    crop_left = max(int(round((resized_width - crop_size) / 2.0)), 0)
    crop_top = max(int(round((resized_height - crop_size) / 2.0)), 0)
    x -= crop_left
    y -= crop_top
    if x < 0.0 or y < 0.0 or x >= crop_size or y >= crop_size:
        return None
    return x, y


def sample_image_size(ds: Dataset, idx: int, cache: Dict[int, Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    if idx in cache:
        return cache[idx]
    sample_path = sample_path_from_dataset(ds, idx)
    try:
        with Image.open(sample_path) as image:
            size = image.size
    except OSError:
        return None
    cache[idx] = size
    return size


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


def batched_oracle_metrics(
    score_maps: torch.Tensor,
    points_per_target: Sequence[Sequence[Tuple[float, float]]],
    gt_masks: torch.Tensor,
    point_masks: torch.Tensor,
    thresholds: Sequence[float],
    radii: Sequence[float],
    diag: float,
    chunk_size: int,
) -> Dict[str, object]:
    device = score_maps.device
    score_flat = score_maps.flatten(1)
    gt_flat = gt_masks.to(device=device, dtype=torch.float32).flatten(1)
    point_flat = point_masks.to(device=device, dtype=torch.float32).flatten(1)
    gt_sum = gt_flat.sum(dim=1).view(1, -1)
    flat_argmax = score_flat.argmax(dim=1)
    argmax_y = (flat_argmax // score_maps.shape[-1]).float()
    argmax_x = (flat_argmax % score_maps.shape[-1]).float()

    mean_dist_sum = 0.0
    point_hits_sum = {r: 0.0 for r in radii}
    point_count = 0
    for points in points_per_target:
        point_tensor = torch.tensor(points, dtype=torch.float32, device=device)
        dx = argmax_x[:, None] - point_tensor[None, :, 0]
        dy = argmax_y[:, None] - point_tensor[None, :, 1]
        per_concept_dist = torch.sqrt(dx.square() + dy.square()).amin(dim=1) / max(float(diag), 1e-8)
        best_idx = int(per_concept_dist.argmin().item())
        best_dist = float(per_concept_dist[best_idx].item())
        best_x = int(argmax_x[best_idx].item())
        best_y = int(argmax_y[best_idx].item())
        mean_dist_sum += best_dist
        point_count += 1
        for r in radii:
            point_hits_sum[r] += 1.0 if point_in_any_disk(best_x, best_y, points, float(r) * diag) else 0.0

    threshold_out: Dict[float, Dict[str, float]] = {}
    chunk_size = max(int(chunk_size), 1)
    for thr in thresholds:
        best_point = torch.zeros((gt_flat.shape[0],), dtype=torch.float32, device=device)
        best_iou = torch.zeros((gt_flat.shape[0],), dtype=torch.float32, device=device)
        best_dice = torch.zeros((gt_flat.shape[0],), dtype=torch.float32, device=device)
        for start in range(0, score_flat.shape[0], chunk_size):
            pred = (score_flat[start : start + chunk_size] >= thr).float()
            pred_sum = pred.sum(dim=1).view(-1, 1)
            inter = pred @ gt_flat.T
            union = pred_sum + gt_sum - inter
            iou = torch.where(union > 0, inter / union.clamp_min(1.0), torch.zeros_like(union))
            dice = torch.where(
                (pred_sum + gt_sum) > 0,
                (2.0 * inter) / (pred_sum + gt_sum).clamp_min(1.0),
                torch.zeros_like(inter),
            )
            point_hit = ((pred @ point_flat.T) > 0).float()
            best_point = torch.maximum(best_point, point_hit.max(dim=0).values)
            best_iou = torch.maximum(best_iou, iou.max(dim=0).values)
            best_dice = torch.maximum(best_dice, dice.max(dim=0).values)
        threshold_out[thr] = {
            "point_in_mask_sum": float(best_point.sum().item()),
            "mask_iou_sum": float(best_iou.sum().item()),
            "dice_sum": float(best_dice.sum().item()),
            "count": int(gt_flat.shape[0]),
        }
    return {
        "mean_dist_sum": mean_dist_sum,
        "point_hits_sum": point_hits_sum,
        "point_count": point_count,
        "thresholds": threshold_out,
    }


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
    parser.add_argument(
        "--map_normalization",
        type=str,
        default="concept_zscore_minmax",
        choices=["minmax", "sigmoid", "proj_zscore_minmax", "concept_zscore_minmax"],
        help="Map normalization. For CUB, concept_zscore_minmax uses saved proj_mean.pt/proj_std.pt, then min-max scales each concept map.",
    )
    parser.add_argument("--point_source", type=str, default="normalized_map", choices=["normalized_map", "pred_dist"])
    parser.add_argument(
        "--compute_concept_oracle",
        action="store_true",
        help="Also evaluate every concept map for each part target and report the best concept score per metric.",
    )
    parser.add_argument(
        "--oracle_chunk_size",
        type=int,
        default=128,
        help="Number of concepts per chunk for all-concept oracle mask metrics.",
    )
    parser.add_argument("--activation_thresholds", type=str, default="0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--radius_fracs", type=str, default="0.01,0.02,0.05,0.1")
    parser.add_argument("--disk_radius_frac", type=float, default=0.03)
    parser.add_argument(
        "--resize_size",
        type=int,
        default=None,
        help="Short-edge resize used before center crop. Defaults to 256 for 224px ResNet CUB and 600 for 448px MMPretrain CUB.",
    )
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
    if args_ns.map_normalization in {"concept_zscore_minmax", "proj_zscore_minmax"}:
        mean_path = Path(args_ns.load_path) / "proj_mean.pt"
        std_path = Path(args_ns.load_path) / "proj_std.pt"
        if not mean_path.exists() or not std_path.exists():
            raise RuntimeError("CUB concept_zscore_minmax requires proj_mean.pt/proj_std.pt in the checkpoint directory.")
        proj_mean_all = torch.load(mean_path, map_location="cpu").float().flatten()
        proj_std_all = torch.load(std_path, map_location="cpu").float().flatten().clamp_min(1e-6)
        if proj_mean_all.numel() < len(concepts) or proj_std_all.numel() < len(concepts):
            raise RuntimeError(
                f"proj_mean/proj_std size mismatch: got {proj_mean_all.numel()}/{proj_std_all.numel()} "
                f"for {len(concepts)} concepts."
            )
    else:
        proj_mean_all = None
        proj_std_all = None

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
    oracle_point_hits_sum = {r: 0.0 for r in radii}
    oracle_point_hits_count = 0
    oracle_mean_dist_sum = 0.0
    oracle_mean_dist_count = 0
    oracle_threshold_point_hits_sum = {thr: 0.0 for thr in thresholds}
    oracle_threshold_mask_iou_sum = {thr: 0.0 for thr in thresholds}
    oracle_threshold_dice_sum = {thr: 0.0 for thr in thresholds}
    oracle_threshold_count = {thr: 0 for thr in thresholds}

    ann_split_dir = Path(args.annotation_dir) / f"{args.dataset}_test"
    if not ann_split_dir.is_dir():
        ann_split_dir = Path(args.annotation_dir) / f"{args.dataset}_val"

    dataset_base_indices = [resolve_base_index(dataset.base, i) for i in range(len(dataset))]
    image_ids_by_ds_idx = build_dataset_image_ids(dataset.base, images_index)
    image_size_by_ds_idx: Dict[int, Tuple[int, int]] = {}
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
                image_size = sample_image_size(dataset.base, int(ds_idx), image_size_by_ds_idx)
                if image_size is None:
                    continue
                resize_size = int(args_ns.resize_size or infer_resize_size(getattr(args, "backbone", ""), img_h))
                valid_gt_concepts: List[Tuple[str, List[Tuple[float, float]]]] = []
                for label, exact_parts in gt_concepts:
                    points = [
                        transformed
                        for p in exact_parts
                        if p in image_parts
                        for transformed in [
                            transform_point_for_model_input(
                                image_parts[p],
                                image_size=image_size,
                                crop_size=img_h,
                                resize_size=resize_size,
                            )
                        ]
                        if transformed is not None
                    ]
                    if points:
                        valid_gt_concepts.append((label, points))
                if not valid_gt_concepts:
                    continue

                concept_idx_tensor = torch.as_tensor([concept_to_idx[label] for label, _ in valid_gt_concepts], device=spatial_maps.device, dtype=torch.long)
                if proj_mean_all is not None and proj_std_all is not None:
                    concept_idx_cpu = concept_idx_tensor.cpu()
                    proj_mean = proj_mean_all.index_select(0, concept_idx_cpu).view(-1, 1, 1)
                    proj_std = proj_std_all.index_select(0, concept_idx_cpu).view(-1, 1, 1)
                else:
                    proj_mean = None
                    proj_std = None
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
                    score_maps = _normalize_map_with_mode(
                        maps_k,
                        args_ns.map_normalization,
                        proj_mean=proj_mean,
                        proj_std=proj_std,
                    )
                if args_ns.compute_concept_oracle:
                    all_maps = F.interpolate(
                        spatial_maps[b].unsqueeze(1),
                        size=(img_h, img_w),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(1)
                    if args_ns.point_source == "pred_dist":
                        oracle_score_maps = F.softmax(all_maps.flatten(1), dim=1).view_as(all_maps)
                    else:
                        oracle_mean = proj_mean_all.view(-1, 1, 1) if proj_mean_all is not None else None
                        oracle_std = proj_std_all.view(-1, 1, 1) if proj_std_all is not None else None
                        oracle_score_maps = _normalize_map_with_mode(
                            all_maps,
                            args_ns.map_normalization,
                            proj_mean=oracle_mean,
                            proj_std=oracle_std,
                        )
                    oracle_score_maps_for_metrics = oracle_score_maps
                else:
                    oracle_score_maps_for_metrics = None

                argmax_flat = score_maps.flatten(1).argmax(dim=1)
                argmax_y = (argmax_flat // score_maps.shape[-1]).cpu().tolist()
                argmax_x = (argmax_flat % score_maps.shape[-1]).cpu().tolist()
                score_maps_cpu = score_maps.cpu()
                diag = math.sqrt(float(img_h * img_h + img_w * img_w))
                disk_radius_px = float(args_ns.disk_radius_frac) * diag

                points_per_concept: List[List[Tuple[float, float]]] = []
                gt_masks: List[torch.Tensor] = []
                point_indicator_masks: List[torch.Tensor] = []

                for (_label, points), px, py in zip(valid_gt_concepts, argmax_x, argmax_y):
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

                if oracle_score_maps_for_metrics is not None:
                    oracle_batch = batched_oracle_metrics(
                        oracle_score_maps_for_metrics,
                        points_per_concept,
                        gt_masks_tensor,
                        point_indicator_tensor,
                        thresholds,
                        radii,
                        diag,
                        max(int(args_ns.oracle_chunk_size), 1),
                    )
                    oracle_mean_dist_sum += float(oracle_batch["mean_dist_sum"])
                    oracle_mean_dist_count += int(oracle_batch["point_count"])
                    oracle_point_hits_count += int(oracle_batch["point_count"])
                    for r, value in oracle_batch["point_hits_sum"].items():
                        oracle_point_hits_sum[r] += float(value)
                    for thr, values in oracle_batch["thresholds"].items():
                        oracle_threshold_count[thr] += int(values["count"])
                        oracle_threshold_point_hits_sum[thr] += float(values["point_in_mask_sum"])
                        oracle_threshold_mask_iou_sum[thr] += float(values["mask_iou_sum"])
                        oracle_threshold_dice_sum[thr] += float(values["dice_sum"])

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
    if args_ns.compute_concept_oracle:
        oracle_threshold_metrics_out: Dict[str, Dict[str, float]] = {}
        oracle_best_mask_iou = {"threshold": None, "value": None}
        oracle_best_dice = {"threshold": None, "value": None}
        oracle_best_point_in_mask = {"threshold": None, "value": None}
        for thr in thresholds:
            point_in_mask = float(oracle_threshold_point_hits_sum[thr] / max(oracle_threshold_count[thr], 1))
            mask_iou = float(oracle_threshold_mask_iou_sum[thr] / max(oracle_threshold_count[thr], 1))
            dice = float(oracle_threshold_dice_sum[thr] / max(oracle_threshold_count[thr], 1))
            oracle_threshold_metrics_out[str(thr)] = {
                "point_in_mask": point_in_mask,
                "mask_iou": mask_iou,
                "dice": dice,
            }
            if oracle_best_mask_iou["value"] is None or mask_iou > oracle_best_mask_iou["value"]:
                oracle_best_mask_iou = {"threshold": thr, "value": mask_iou}
            if oracle_best_dice["value"] is None or dice > oracle_best_dice["value"]:
                oracle_best_dice = {"threshold": thr, "value": dice}
            if oracle_best_point_in_mask["value"] is None or point_in_mask > oracle_best_point_in_mask["value"]:
                oracle_best_point_in_mask = {"threshold": thr, "value": point_in_mask}
        results["concept_oracle"] = {
            "definition": "For each part target, all concept maps are evaluated and the best concept score is selected independently for each metric.",
            "num_gt_instances": oracle_mean_dist_count,
            "point_metrics": {
                "mean_normalized_distance": float(oracle_mean_dist_sum / max(oracle_mean_dist_count, 1)),
                "point_hit": {
                    str(r): float(oracle_point_hits_sum[r] / max(oracle_point_hits_count, 1))
                    for r in radii
                },
            },
            "threshold_metrics": oracle_threshold_metrics_out,
            "best_mask_iou": oracle_best_mask_iou,
            "best_dice": oracle_best_dice,
            "best_point_in_mask": oracle_best_point_in_mask,
        }

    out = Path(args_ns.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
