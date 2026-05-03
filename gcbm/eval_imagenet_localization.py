import argparse
import json
import re
import sys
import tarfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gcbm.eval_imagenet_localization_helpers import (
    boxes_from_masks,
    box_iou,
    finalize_threshold_metrics,
    normalize_maps,
    parse_thresholds,
    update_threshold_metrics,
)
from gcbm.imagenet_annotation_index import (
    build_filename_to_annotation_path,
    load_annotation_payload,
    resolve_val_annotation_dir,
)
from gcbm.imagenet_core import (
    Config,
    annotation_entries,
    amp_dtype,
    build_gdino_targets,
    build_model,
    canonicalize_concept_label,
    configure_runtime,
    load_concepts,
    prepare_images,
)


VAL_RE = re.compile(r"ILSVRC2012_val_(\d{8})\.JPEG$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate G-CBM localization on ImageNet validation data."
    )
    parser.add_argument("--artifact_dir", required=True)
    parser.add_argument("--val_tar", default="", help="Official ImageNet val tar. Used when --val_root is not set.")
    parser.add_argument(
        "--val_root",
        default="",
        help="Optional extracted ImageNet val ImageFolder root. Enables parallel image decode with DataLoader workers.",
    )
    parser.add_argument(
        "--annotation_dir",
        required=True,
        help="Directory containing imagenet_val/*.json, or the imagenet_val directory itself.",
    )
    parser.add_argument(
        "--annotation_val_root",
        default="",
        help="Optional reorganized ImageNet val ImageFolder root used when annotations are keyed by ImageFolder dataset index.",
    )
    parser.add_argument("--devkit_dir", default="", help="Optional ImageNet devkit, only used to sanity-check val label count.")
    parser.add_argument("--output_json", default="")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--activation_thresholds", default="0.3,0.5,0.7,0.9,mean")
    parser.add_argument("--box_iou_thresholds", default="0.1,0.3,0.5")
    parser.add_argument(
        "--map_normalization",
        default="concept_zscore_minmax",
        choices=["minmax", "sigmoid", "concept_zscore_minmax"],
    )
    parser.add_argument("--gt_threshold", type=float, default=0.0)
    parser.add_argument(
        "--compute_map",
        action="store_true",
        help=(
            "Compute CUB-style concept mAP. Scores are stored only for concepts "
            "observed in GDINO val annotations; TP boxes are sparse."
        ),
    )
    parser.add_argument(
        "--map_score_key",
        default="final_logits",
        choices=["final_logits", "spatial_logits", "global_logits"],
        help="Concept score used for AP ranking.",
    )
    parser.add_argument(
        "--map_score_dtype",
        default="float16",
        choices=["float16", "float32"],
        help="CPU dtype for accumulated AP score chunks.",
    )
    parser.add_argument(
        "--map_max_concepts",
        type=int,
        default=0,
        help="Optional debug cap on AP concepts after annotation-frequency sorting. 0 means all observed concepts.",
    )
    parser.add_argument("--log_every", type=int, default=1000)
    return parser.parse_args()


def resolve_source_run_dir(artifact_dir: Path) -> Path:
    source_run_file = artifact_dir / "source_run_dir.txt"
    if source_run_file.exists():
        source_run_dir = Path(source_run_file.read_text().strip()).resolve()
        if source_run_dir.is_dir():
            return source_run_dir
    return artifact_dir


def load_run_config(config_dir: Path, args: argparse.Namespace) -> Config:
    payload = json.loads((config_dir / "config.json").read_text())
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
    payload["persistent_workers"] = bool(args.persistent_workers)
    payload["pin_memory"] = bool(args.pin_memory)
    payload["skip_final_layer"] = True
    payload["print_config"] = False
    return Config(**payload)


def load_val_label_count(devkit_dir: Path) -> Optional[int]:
    if not devkit_dir:
        return None
    labels_path = devkit_dir / "data" / "ILSVRC2012_validation_ground_truth.txt"
    meta_path = devkit_dir / "data" / "meta.mat"
    if not labels_path.is_file() or not meta_path.is_file():
        return None
    # Load meta too, matching the classification eval's validation that this is a real devkit.
    loadmat(meta_path, squeeze_me=True, struct_as_record=False)
    with labels_path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def build_transform(input_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def load_annotation(
    annotation_val_dir: Path,
    image_index_1based: int,
    image_name: str,
    filename_to_annotation_path: Optional[Dict[str, Path]] = None,
) -> List[Dict[str, Any]]:
    return load_annotation_payload(
        annotation_val_dir=annotation_val_dir,
        image_index_1based=image_index_1based,
        image_name=image_name,
        filename_to_annotation_path=filename_to_annotation_path,
    )


def collect_observed_ap_concepts(
    annotation_val_dir: Path,
    concept_to_idx: Dict[str, int],
    concept_threshold: float,
    max_concepts: int,
) -> Tuple[List[int], Dict[int, int]]:
    """Choose the localization concepts included in CUB-style mAP.

    AP treats images without GT for a concept as negatives, so evaluating
    concepts that never appear in GDINO val annotations only creates millions
    of uninformative negatives. This one-time scan scopes mAP to observed
    benchmark concepts while keeping the later per-batch computation dense and
    vectorized over that observed subset.
    """
    counts: Dict[int, int] = {}
    for path in sorted(annotation_val_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        entries = annotation_entries(payload if isinstance(payload, list) else payload.get("concepts", []))
        present_in_image: set[int] = set()
        for ann in entries:
            if not isinstance(ann, dict):
                continue
            label = ann.get("label", ann.get("name"))
            if not isinstance(label, str):
                continue
            if float(ann.get("logit", 0.0)) < float(concept_threshold):
                continue
            concept_idx = concept_to_idx.get(canonicalize_concept_label(label))
            if concept_idx is not None:
                present_in_image.add(int(concept_idx))
        for concept_idx in present_in_image:
            counts[concept_idx] = counts.get(concept_idx, 0) + 1
    ordered = sorted(counts, key=lambda idx: (-counts[idx], idx))
    if max_concepts > 0:
        ordered = ordered[: int(max_concepts)]
        counts = {idx: counts[idx] for idx in ordered}
    return ordered, counts


def init_ap_state(
    ap_concept_indices: Sequence[int],
    n_concepts: int,
    device: str,
    score_dtype: str,
) -> Dict[str, Any]:
    concept_to_ap_index = torch.full((n_concepts,), -1, dtype=torch.long, device=device)
    if ap_concept_indices:
        concept_to_ap_index[torch.tensor(ap_concept_indices, dtype=torch.long, device=device)] = torch.arange(
            len(ap_concept_indices),
            dtype=torch.long,
            device=device,
        )
    return {
        "concept_indices": list(ap_concept_indices),
        "concept_indices_device": torch.tensor(ap_concept_indices, dtype=torch.long, device=device),
        "concept_to_ap_index": concept_to_ap_index,
        "score_dtype": np.float16 if score_dtype == "float16" else np.float32,
        "score_chunks": [],
        "gt_counts": np.zeros((len(ap_concept_indices),), dtype=np.int64),
        "tp_rows": defaultdict(list),
        "tp_cols": defaultdict(list),
    }


def append_ap_scores(ap_state: Optional[Dict[str, Any]], outputs: Dict[str, torch.Tensor], score_key: str) -> None:
    if ap_state is None or not ap_state["concept_indices"]:
        return
    if score_key not in outputs:
        raise KeyError(f"{score_key} not found in model outputs; available keys={sorted(outputs)}")
    concept_indices = ap_state["concept_indices_device"].to(outputs[score_key].device, non_blocking=True)
    scores = outputs[score_key].detach().float().index_select(1, concept_indices).cpu().numpy()
    ap_state["score_chunks"].append(scores.astype(ap_state["score_dtype"], copy=False))


def update_ap_ground_truth_counts(ap_state: Optional[Dict[str, Any]], concept_ids: torch.Tensor) -> Optional[torch.Tensor]:
    if ap_state is None:
        return None
    ap_cols = ap_state["concept_to_ap_index"].index_select(0, concept_ids)
    valid = ap_cols >= 0
    if bool(valid.any()):
        cols = ap_cols[valid].detach().cpu().numpy().astype(np.int64, copy=False)
        np.add.at(ap_state["gt_counts"], cols, 1)
    return ap_cols


def update_ap_true_positives(
    ap_state: Optional[Dict[str, Any]],
    metric_key: str,
    image_row: int,
    ap_cols: Optional[torch.Tensor],
    box_ious: torch.Tensor,
    box_iou_thresholds: Sequence[float],
) -> None:
    if ap_state is None or ap_cols is None:
        return
    valid = ap_cols >= 0
    if not bool(valid.any()):
        return
    cols = ap_cols[valid].detach().cpu().numpy().astype(np.int64, copy=False)
    ious = box_ious[valid].detach().cpu().numpy()
    for threshold in box_iou_thresholds:
        hit_cols = cols[ious >= float(threshold)]
        if hit_cols.size == 0:
            continue
        key = f"{metric_key}@{threshold:g}"
        ap_state["tp_rows"][key].extend([int(image_row)] * int(hit_cols.size))
        ap_state["tp_cols"][key].extend(hit_cols.astype(int).tolist())


def average_precision(sorted_tp: np.ndarray, total_gt: int) -> float:
    if total_gt <= 0:
        return float("nan")
    if not bool(sorted_tp.any()):
        return 0.0
    tp_cum = np.cumsum(sorted_tp, dtype=np.float64)
    ranks = np.arange(1, sorted_tp.size + 1, dtype=np.float64)
    return float((tp_cum[sorted_tp] / ranks[sorted_tp]).sum() / float(total_gt))


def finalize_ap_metrics(
    ap_state: Optional[Dict[str, Any]],
    threshold_keys: Sequence[str],
    box_iou_thresholds: Sequence[float],
) -> Dict[str, Any]:
    if ap_state is None:
        return {}
    if not ap_state["score_chunks"]:
        return {"enabled": True, "concept_count": len(ap_state["concept_indices"]), "error": "no scores accumulated"}

    scores = np.concatenate(ap_state["score_chunks"], axis=0)
    n_images, n_ap_concepts = scores.shape
    gt_counts = ap_state["gt_counts"]
    valid_concepts = gt_counts > 0
    total_gt_pairs = int(gt_counts[valid_concepts].sum())

    tp_by_metric: Dict[str, Dict[int, np.ndarray]] = {}
    for key, rows in ap_state["tp_rows"].items():
        grouped: DefaultDict[int, List[int]] = defaultdict(list)
        for row, col in zip(rows, ap_state["tp_cols"][key]):
            grouped[int(col)].append(int(row))
        tp_by_metric[key] = {
            col: np.unique(np.asarray(row_values, dtype=np.int64))
            for col, row_values in grouped.items()
        }

    metric_keys = [f"{threshold_key}@{iou:g}" for threshold_key in threshold_keys for iou in box_iou_thresholds]
    ap_values: Dict[str, List[float]] = {key: [] for key in metric_keys}
    weighted_ap_sums: Dict[str, float] = {key: 0.0 for key in metric_keys}

    for concept_col in np.where(valid_concepts)[0]:
        order = np.argsort(-scores[:, concept_col], kind="mergesort")
        gt_count = int(gt_counts[concept_col])
        for metric_key in metric_keys:
            tp = np.zeros((n_images,), dtype=bool)
            tp_rows = tp_by_metric.get(metric_key, {}).get(int(concept_col))
            if tp_rows is not None and tp_rows.size:
                tp[tp_rows] = True
            ap = average_precision(tp[order], gt_count)
            ap_values[metric_key].append(ap)
            weighted_ap_sums[metric_key] += ap * gt_count

    per_threshold: Dict[str, Any] = {}
    for threshold_key in threshold_keys:
        per_threshold[threshold_key] = {}
        for iou in box_iou_thresholds:
            metric_key = f"{threshold_key}@{iou:g}"
            values = np.asarray(ap_values[metric_key], dtype=np.float64)
            per_threshold[threshold_key][str(iou)] = {
                "mAP": float(values.mean()) if values.size else float("nan"),
                "weighted_mAP": float(weighted_ap_sums[metric_key] / max(total_gt_pairs, 1)),
                "concepts_with_gt": int(values.size),
            }

    best_by_iou: Dict[str, Any] = {}
    for iou in box_iou_thresholds:
        best_threshold = ""
        best_value = float("-inf")
        for threshold_key in threshold_keys:
            value = float(per_threshold[threshold_key][str(iou)]["mAP"])
            if value > best_value:
                best_value = value
                best_threshold = threshold_key
        best_by_iou[str(iou)] = {"threshold": best_threshold, "mAP": best_value}

    return {
        "enabled": True,
        "score_dtype": "float16" if ap_state["score_dtype"] == np.float16 else "float32",
        "score_shape": [int(n_images), int(n_ap_concepts)],
        "concept_count": int(n_ap_concepts),
        "concepts_with_gt": int(valid_concepts.sum()),
        "total_gt_pairs": total_gt_pairs,
        "per_threshold": per_threshold,
        "best_by_iou": best_by_iou,
    }


def iter_tar_samples(
    val_tar: Path,
    annotation_val_dir: Path,
    transform: transforms.Compose,
    max_images: int,
    filename_to_annotation_path: Optional[Dict[str, Path]] = None,
):
    seen = 0
    with tarfile.open(val_tar, "r|*") as tf:
        print(f"[loc-val] streaming tar members from {val_tar}", flush=True)
        for member in tf:
            if not member.isfile():
                continue
            match = VAL_RE.search(Path(member.name).name)
            if match is None:
                continue
            image_index = int(match.group(1))
            handle = tf.extractfile(member)
            if handle is None:
                raise FileNotFoundError(member.name)
            with Image.open(handle) as image:
                image = image.convert("RGB")
                image_size = (int(image.size[0]), int(image.size[1]))
                tensor = transform(image)
            annotation = load_annotation(
                annotation_val_dir,
                image_index,
                Path(member.name).name,
                filename_to_annotation_path=filename_to_annotation_path,
            )
            seen += 1
            yield tensor, annotation, image_size, member.name
            if max_images > 0 and seen >= max_images:
                break


class ImageFolderLocalizationDataset(Dataset):
    def __init__(
        self,
        val_root: Path,
        annotation_val_dir: Path,
        transform: transforms.Compose,
        filename_to_annotation_path: Optional[Dict[str, Path]] = None,
    ) -> None:
        try:
            dataset = ImageFolder(str(val_root))
            self.samples = [Path(path) for path, _target in dataset.samples]
        except FileNotFoundError:
            # ImageNet val is often extracted as a flat directory of
            # ILSVRC2012_val_*.JPEG files rather than an ImageFolder tree.
            self.samples = sorted(val_root.glob("*.JPEG"))
            if not self.samples:
                self.samples = sorted(val_root.rglob("*.JPEG"))
            if not self.samples:
                raise
        self.annotation_val_dir = annotation_val_dir
        self.transform = transform
        self.filename_to_annotation_path = filename_to_annotation_path

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path = self.samples[index]
        image_name = Path(image_path).name
        match = VAL_RE.search(image_name)
        if match is None:
            raise ValueError(f"ImageNet val filename does not match expected pattern: {image_path}")
        image_index = int(match.group(1))
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image_size = (int(image.size[0]), int(image.size[1]))
            tensor = self.transform(image)
        annotation = load_annotation(
            self.annotation_val_dir,
            image_index,
            image_name,
            filename_to_annotation_path=self.filename_to_annotation_path,
        )
        return tensor, annotation, image_size, image_name


def localization_collate(batch):
    images, annotations, image_sizes, names = zip(*batch)
    return list(images), list(annotations), list(image_sizes), list(names)


def build_extracted_val_loader(
    val_root: Path,
    annotation_val_dir: Path,
    transform: transforms.Compose,
    cfg: Config,
    args: argparse.Namespace,
    filename_to_annotation_path: Optional[Dict[str, Path]] = None,
) -> DataLoader:
    dataset = ImageFolderLocalizationDataset(
        val_root=val_root,
        annotation_val_dir=annotation_val_dir,
        transform=transform,
        filename_to_annotation_path=filename_to_annotation_path,
    )
    kwargs: Dict[str, Any] = {
        "dataset": dataset,
        "batch_size": int(cfg.batch_size),
        "shuffle": False,
        "num_workers": int(args.workers),
        "pin_memory": bool(args.pin_memory),
        "collate_fn": localization_collate,
    }
    if int(args.workers) > 0:
        kwargs["prefetch_factor"] = int(args.prefetch_factor)
        kwargs["persistent_workers"] = bool(args.persistent_workers)
    return DataLoader(**kwargs)


def evaluate_batch(
    images: List[torch.Tensor],
    annotations: List[List[Dict[str, Any]]],
    image_sizes: List[Tuple[int, int]],
    backbone: torch.nn.Module,
    head: torch.nn.Module,
    cfg: Config,
    concept_to_idx: Dict[str, int],
    n_concepts: int,
    args: argparse.Namespace,
    threshold_raw: Dict[str, Any],
    distribution: Dict[str, float],
    thresholds: Sequence[float],
    include_mean_threshold: bool,
    box_iou_thresholds: Sequence[float],
    ap_state: Optional[Dict[str, Any]] = None,
    image_offset: int = 0,
) -> Tuple[int, int]:
    batch = prepare_images(torch.stack(images, dim=0), cfg)
    with torch.no_grad():
        with torch.autocast(
            device_type="cuda",
            dtype=amp_dtype(cfg.amp),
            enabled=(str(cfg.device).startswith("cuda") and amp_dtype(cfg.amp) is not None),
        ):
            feats = backbone(batch)
            outputs = head(feats)
            global_targets, mask_indices, mask_targets, mask_valid = build_gdino_targets(
                annotations,
                image_sizes,
                concept_to_idx,
                n_concepts,
                cfg,
                cfg.device,
            )
            del global_targets
            append_ap_scores(ap_state, outputs, args.map_score_key)
            spatial_maps = F.interpolate(
                outputs["spatial_maps"],
                size=mask_targets.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).float()

    images_with_targets = 0
    instance_count = 0
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
        ap_cols = update_ap_ground_truth_counts(ap_state, concept_ids)
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
        instance_count += int(gt.shape[0])
        distribution["instances"] += int(gt.shape[0])
        distribution["soft_iou_sum"] += float((soft_inter / soft_union).sum().item())
        distribution["mass_in_gt_sum"] += float((pred_dist * gt_masks.float()).flatten(1).sum(dim=1).sum().item())
        distribution["point_hit_sum"] += float(dist_point_hit.sum().item())

        for threshold in thresholds:
            key = str(threshold)
            box_ious = update_threshold_metrics(
                threshold_raw,
                key,
                score_maps,
                score_maps >= float(threshold),
                gt_masks,
                gt_boxes,
                gt_box_valid,
                box_iou_thresholds,
            )
            update_ap_true_positives(
                ap_state,
                key,
                image_offset + batch_index,
                ap_cols,
                box_ious,
                box_iou_thresholds,
            )
        if include_mean_threshold:
            box_ious = update_threshold_metrics(
                threshold_raw,
                "mean",
                score_maps,
                score_maps >= score_maps.mean(dim=(1, 2), keepdim=True),
                gt_masks,
                gt_boxes,
                gt_box_valid,
                box_iou_thresholds,
            )
            update_ap_true_positives(
                ap_state,
                "mean",
                image_offset + batch_index,
                ap_cols,
                box_ious,
                box_iou_thresholds,
            )
    return images_with_targets, instance_count


def main() -> None:
    args = parse_args()
    artifact_dir = Path(args.artifact_dir).resolve()
    source_run_dir = resolve_source_run_dir(artifact_dir)
    val_tar = Path(args.val_tar).resolve() if args.val_tar else None
    val_root = Path(args.val_root).resolve() if args.val_root else None
    if val_tar is None and val_root is None:
        raise ValueError("one of --val_tar or --val_root is required")
    annotation_val_dir = resolve_val_annotation_dir(Path(args.annotation_dir).resolve())
    annotation_val_root = Path(args.annotation_val_root).resolve() if args.annotation_val_root else None
    filename_to_annotation_path = build_filename_to_annotation_path(annotation_val_dir, annotation_val_root)
    output_json = (
        Path(args.output_json).resolve()
        if args.output_json
        else artifact_dir / "localization_imagenet_val_tar.json"
    )
    cfg = load_run_config(source_run_dir, args)
    configure_runtime(cfg)

    concepts = load_concepts(str(source_run_dir / "concepts.txt"))
    concept_to_idx = {name: idx for idx, name in enumerate(concepts)}
    backbone, head = build_model(cfg, n_concepts=len(concepts))
    head.load_state_dict(torch.load(source_run_dir / "concept_head_best.pt", map_location=cfg.device))
    backbone.eval()
    head.eval()

    thresholds, include_mean_threshold = parse_thresholds(args.activation_thresholds)
    box_iou_thresholds = [float(x.strip()) for x in args.box_iou_thresholds.split(",") if x.strip()]
    threshold_keys = [str(threshold) for threshold in thresholds]
    if include_mean_threshold:
        threshold_keys.append("mean")
    threshold_raw: Dict[str, Any] = {}
    distribution: Dict[str, float] = {
        "instances": 0,
        "soft_iou_sum": 0.0,
        "mass_in_gt_sum": 0.0,
        "point_hit_sum": 0.0,
    }
    transform = build_transform(cfg.input_size)
    label_count = load_val_label_count(Path(args.devkit_dir).resolve()) if args.devkit_dir else None
    ap_state: Optional[Dict[str, Any]] = None
    ap_concept_counts: Dict[int, int] = {}
    if bool(args.compute_map):
        if int(args.map_max_concepts) > 0:
            ap_concepts, ap_concept_counts = collect_observed_ap_concepts(
                annotation_val_dir,
                concept_to_idx,
                cfg.concept_threshold,
                int(args.map_max_concepts),
            )
        else:
            # Avoid a slow pre-scan over tens of thousands of annotation JSONs.
            # Full ImageNet val scores are only ~430 MB as fp16
            # (50k images x 4309 concepts), and finalization automatically
            # drops concepts with zero GT count before averaging mAP.
            ap_concepts = list(range(len(concepts)))
        ap_state = init_ap_state(
            ap_concepts,
            len(concepts),
            cfg.device,
            args.map_score_dtype,
        )
        print(
            f"[loc-val] CUB-style AP enabled: concepts={len(ap_concepts)} "
            f"score_key={args.map_score_key} score_dtype={args.map_score_dtype}",
            flush=True,
        )

    start = time.perf_counter()
    total_images = 0
    images_with_targets = 0
    last_name = ""
    if val_root is not None:
        loader = build_extracted_val_loader(
            val_root,
            annotation_val_dir,
            transform,
            cfg,
            args,
            filename_to_annotation_path=filename_to_annotation_path,
        )
        print(
            f"[loc-val] loading extracted val root {val_root} with workers={int(args.workers)} "
            f"prefetch_factor={int(args.prefetch_factor)}",
            flush=True,
        )
        for images, annotations, image_sizes, names in loader:
            if int(args.max_images) > 0 and total_images + len(images) > int(args.max_images):
                keep = int(args.max_images) - total_images
                if keep <= 0:
                    break
                images = images[:keep]
                annotations = annotations[:keep]
                image_sizes = image_sizes[:keep]
                names = names[:keep]
            batch_with_targets, _ = evaluate_batch(
                images,
                annotations,
                image_sizes,
                backbone,
                head,
                cfg,
                concept_to_idx,
                len(concepts),
                args,
                threshold_raw,
                distribution,
                thresholds,
                include_mean_threshold,
                box_iou_thresholds,
                ap_state=ap_state,
                image_offset=total_images,
            )
            total_images += len(images)
            images_with_targets += batch_with_targets
            last_name = names[-1] if names else last_name
            if args.log_every > 0 and total_images % int(args.log_every) == 0:
                elapsed = time.perf_counter() - start
                print(
                    f"[loc-val] n={total_images} images_with_targets={images_with_targets} "
                    f"instances={int(distribution['instances'])} ips={total_images / max(elapsed, 1e-6):.2f} last={last_name}",
                    flush=True,
                )
            if int(args.max_images) > 0 and total_images >= int(args.max_images):
                break
    else:
        images: List[torch.Tensor] = []
        annotations: List[List[Dict[str, Any]]] = []
        image_sizes: List[Tuple[int, int]] = []
        assert val_tar is not None
        for image_tensor, annotation, image_size, name in iter_tar_samples(
            val_tar,
            annotation_val_dir,
            transform,
            int(args.max_images),
            filename_to_annotation_path=filename_to_annotation_path,
        ):
            images.append(image_tensor)
            annotations.append(annotation)
            image_sizes.append(image_size)
            last_name = name
            if len(images) >= int(cfg.batch_size):
                batch_with_targets, _ = evaluate_batch(
                    images,
                    annotations,
                    image_sizes,
                    backbone,
                    head,
                    cfg,
                    concept_to_idx,
                    len(concepts),
                    args,
                    threshold_raw,
                    distribution,
                    thresholds,
                    include_mean_threshold,
                    box_iou_thresholds,
                    ap_state=ap_state,
                    image_offset=total_images,
                )
                total_images += len(images)
                images_with_targets += batch_with_targets
                if args.log_every > 0 and total_images % int(args.log_every) == 0:
                    elapsed = time.perf_counter() - start
                    print(
                        f"[loc-val] n={total_images} images_with_targets={images_with_targets} "
                        f"instances={int(distribution['instances'])} ips={total_images / max(elapsed, 1e-6):.2f} last={last_name}",
                        flush=True,
                    )
                images.clear()
                annotations.clear()
                image_sizes.clear()
        if images:
            batch_with_targets, _ = evaluate_batch(
                images,
                annotations,
                image_sizes,
                backbone,
                head,
                cfg,
                concept_to_idx,
                len(concepts),
                args,
                threshold_raw,
                distribution,
                thresholds,
                include_mean_threshold,
                box_iou_thresholds,
                ap_state=ap_state,
                image_offset=total_images,
            )
            total_images += len(images)
            images_with_targets += batch_with_targets

    elapsed = time.perf_counter() - start
    instances = max(int(distribution["instances"]), 1)
    map_metrics = finalize_ap_metrics(ap_state, threshold_keys, box_iou_thresholds)
    payload = {
        "artifact_dir": str(artifact_dir),
        "source_run_dir": str(source_run_dir),
        "val_tar": str(val_tar) if val_tar is not None else "",
        "val_root": str(val_root) if val_root is not None else "",
        "annotation_val_dir": str(annotation_val_dir),
        "annotation_val_root": str(annotation_val_root) if annotation_val_root is not None else "",
        "devkit_label_count": label_count,
        "n_concepts": len(concepts),
        "config": {
            "batch_size": int(cfg.batch_size),
            "device": cfg.device,
            "map_normalization": args.map_normalization,
            "activation_thresholds": args.activation_thresholds,
            "box_iou_thresholds": args.box_iou_thresholds,
            "gt_threshold": float(args.gt_threshold),
            "max_images": int(args.max_images),
            "workers": int(args.workers),
            "prefetch_factor": int(args.prefetch_factor),
            "persistent_workers": bool(args.persistent_workers),
            "pin_memory": bool(args.pin_memory),
            "compute_map": bool(args.compute_map),
            "map_score_key": args.map_score_key,
            "map_score_dtype": args.map_score_dtype,
            "map_max_concepts": int(args.map_max_concepts),
        },
        "metrics": {
            "images_seen": total_images,
            "images_with_targets": images_with_targets,
            "instances": int(distribution["instances"]),
            "elapsed_sec": elapsed,
            "images_per_second": total_images / max(elapsed, 1e-6),
            "distribution_metrics": {
                "soft_iou": float(distribution["soft_iou_sum"] / instances),
                "mass_in_gt": float(distribution["mass_in_gt_sum"] / instances),
                "point_hit": float(distribution["point_hit_sum"] / instances),
            },
            "threshold_metrics": finalize_threshold_metrics(threshold_raw, box_iou_thresholds),
            "map_metrics": map_metrics,
        },
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
