#!/usr/bin/env python3
"""Evaluate concept presence scores against GDINO or CUB-part ground truth."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gcbm.imagenet_annotation_index import build_filename_to_annotation_path, load_annotation_payload, resolve_val_annotation_dir
from gcbm.imagenet_config import Config
from gcbm.imagenet_eval import VAL_RE, load_run_config, resolve_source_run_dir
from gcbm.imagenet_models import build_model
from gcbm.imagenet_targets import canonicalize_concept_label as canonicalize_imagenet_concept
from gcbm.runtime import configure_runtime
from gcbm.training_utils import prepare_images

ImageFile.LOAD_TRUNCATED_IMAGES = True


PART_GROUP_TO_IDS = {
    "back": [1],
    "beak": [2],
    "belly": [3],
    "breast": [4],
    "crown": [5],
    "eye": [7, 11],
    "leg": [8, 12],
    "tail": [14],
    "throat": [15],
    "wing": [9, 13],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CBM concept scores against concept-presence ground truth.")
    parser.add_argument("--dataset", required=True, choices=["cub", "imagenet", "partimagenetpp", "places365"])
    parser.add_argument(
        "--gt_source",
        default="gdino",
        choices=["gdino", "cub_parts", "partimagenetpp_manifest", "partimagenetpp_boxes"],
    )
    parser.add_argument("--load_paths", nargs="+", required=True, help="One or more CBM run directories.")
    parser.add_argument("--model_names", nargs="+", default=None, help="Model types matching --load_paths: sgcbm/vlg_cbm/savlg_cbm/salf_cbm/lf_cbm.")
    parser.add_argument("--names", nargs="+", default=None, help="Display names matching --load_paths.")
    parser.add_argument("--annotation_dir", default="", help="GDINO annotation root.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", "--workers", dest="num_workers", type=int, default=4)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--concept_scope", choices=["own", "common"], default="own", help="Evaluate each model's concepts or only concepts common to all models.")
    parser.add_argument("--normalization", choices=["model_default", "none", "split_zscore", "train_zscore", "saved_zscore", "saved_zscore_minmax", "concept_zscore_minmax", "sigmoid", "minmax", "split_minmax", "train_minmax"], default="model_default")
    parser.add_argument("--threshold", type=float, default=0.0, help="Threshold applied after normalization for precision/recall/F1.")
    parser.add_argument("--gdino_threshold", type=float, default=0.15)
    parser.add_argument("--cub_score_source", choices=["concept_prediction", "nec_features"], default="concept_prediction", help="For CUB SG-CBM, use raw concept-prediction logits or NEC-normalized features.")
    parser.add_argument("--save_per_concept", action="store_true")
    parser.add_argument("--log_every", type=int, default=500)
    parser.add_argument("--savlg_alpha_override", type=float, default=None)
    parser.add_argument("--savlg_branch_norm_mode", default="none")
    parser.add_argument("--disable_activation_cache", action="store_true")

    imagenet = parser.add_argument_group("ImageNet inputs")
    imagenet.add_argument("--val_root", default="")
    imagenet.add_argument("--annotation_val_root", default="", help="ImageFolder root used when ImageNet val annotations were generated.")
    imagenet.add_argument("--annotation_mapping_json", default="", help="Optional ImageNet filename-to-annotation mapping JSON.")
    imagenet.add_argument("--prefetch_factor", type=int, default=2)
    imagenet.add_argument("--persistent_workers", action="store_true")
    imagenet.add_argument("--pin_memory", action="store_true")

    places = parser.add_argument_group("Places365 inputs")
    places.add_argument("--places365_val_manifest", default="")

    cub_parts = parser.add_argument_group("CUB part GT inputs")
    cub_parts.add_argument("--concept_part_mapping", default="data/cub_concept_part_mapping_gpt54.json")
    cub_parts.add_argument("--part_locs", default="datasets/CUB/part_locs.txt")
    cub_parts.add_argument("--images_txt", default="datasets/CUB/images.txt")
    cub_parts.add_argument("--split_txt", default="datasets/CUB/train_test_split.txt")

    pinpp = parser.add_argument_group("PartImageNet++ inputs")
    pinpp.add_argument("--partimagenetpp_train_manifest", default="")
    pinpp.add_argument("--partimagenetpp_val_manifest", default="")
    pinpp.add_argument("--partimagenetpp_gt_boxes_jsonl", default="")
    return parser.parse_args()


def canonicalize_concept(text: str, dataset: str) -> str:
    if dataset == "imagenet":
        return canonicalize_imagenet_concept(text)
    try:
        from data import utils as data_utils

        return data_utils.canonicalize_concept_label(text)
    except Exception:
        return " ".join(str(text).lower().replace("-", " ").replace("_", " ").split())


def load_concepts(path: Path, dataset: str) -> List[str]:
    return [canonicalize_concept(line.strip(), dataset) for line in path.read_text().splitlines() if line.strip()]


def parse_annotation_items(payload: Any) -> Iterable[dict]:
    if isinstance(payload, dict):
        items = payload.get("concepts", [])
    else:
        items = payload
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, dict)]


def annotation_label_and_score(item: dict) -> Tuple[str, float]:
    label = str(item.get("label", item.get("concept", item.get("name", ""))))
    score = item.get("logit", item.get("score", item.get("confidence", 0.0)))
    try:
        return label, float(score)
    except Exception:
        return label, 0.0


def load_gdino_gt_indexed(annotation_dir: Path, indices: Sequence[int], concepts: Sequence[str], dataset: str, threshold: float) -> np.ndarray:
    concept_to_col = {concept: idx for idx, concept in enumerate(concepts)}
    gt = np.zeros((len(indices), len(concepts)), dtype=np.float32)
    for row, image_idx in enumerate(indices):
        path = annotation_dir / f"{int(image_idx)}.json"
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        for item in parse_annotation_items(payload):
            label, score = annotation_label_and_score(item)
            col = concept_to_col.get(canonicalize_concept(label, dataset))
            if col is not None and score > threshold:
                gt[row, col] = 1.0
    return gt


def load_cub_part_visibility(part_locs_path: Path, images_txt_path: Path, split_path: Path) -> Tuple[dict[int, dict[str, bool]], dict[int, str], set[int]]:
    test_ids: set[int] = set()
    for line in split_path.read_text().splitlines():
        if not line.strip():
            continue
        image_id, split = line.strip().split()
        if split == "0":
            test_ids.add(int(image_id))
    image_id_to_path: dict[int, str] = {}
    for line in images_txt_path.read_text().splitlines():
        if not line.strip():
            continue
        image_id, rel_path = line.strip().split(" ", 1)
        image_id_to_path[int(image_id)] = rel_path
    part_vis_by_image: dict[int, dict[int, bool]] = {}
    for line in part_locs_path.read_text().splitlines():
        if not line.strip():
            continue
        image_id_raw, part_id_raw, _x, _y, visible_raw = line.strip().split()
        image_id, part_id, visible = int(image_id_raw), int(part_id_raw), int(visible_raw)
        if image_id in test_ids:
            part_vis_by_image.setdefault(image_id, {})[part_id] = bool(visible)
    grouped: dict[int, dict[str, bool]] = {}
    for image_id, part_vis in part_vis_by_image.items():
        grouped[image_id] = {group: any(part_vis.get(part_id, False) for part_id in part_ids) for group, part_ids in PART_GROUP_TO_IDS.items()}
    return grouped, image_id_to_path, test_ids


def load_concept_part_mapping(path: Path, dataset: str) -> Dict[str, str]:
    payload = json.loads(path.read_text())
    entries = payload.get("mappings", payload if isinstance(payload, list) else [])
    mapping: Dict[str, str] = {}
    for item in entries:
        if not isinstance(item, dict):
            continue
        keep = bool(item.get("keep", True))
        part_group = item.get("part_group", item.get("part", ""))
        concept = item.get("concept", "")
        if keep and part_group:
            mapping[canonicalize_concept(str(concept), dataset)] = str(part_group)
    return mapping


def load_cub_parts_gt(indices: Sequence[int], concepts: Sequence[str], args: argparse.Namespace) -> np.ndarray:
    part_vis, image_id_to_path, test_ids = load_cub_part_visibility(Path(args.part_locs), Path(args.images_txt), Path(args.split_txt))
    concept_to_part = load_concept_part_mapping(Path(args.concept_part_mapping), args.dataset)
    test_image_ids = [image_id for image_id in sorted(test_ids) if image_id in part_vis and image_id in image_id_to_path]
    gt = np.zeros((len(indices), len(concepts)), dtype=np.float32)
    for row, dataset_idx in enumerate(indices):
        if int(dataset_idx) >= len(test_image_ids):
            continue
        image_id = test_image_ids[int(dataset_idx)]
        group_vis = part_vis.get(image_id, {})
        for col, concept in enumerate(concepts):
            part_group = concept_to_part.get(concept)
            if part_group and group_vis.get(part_group, False):
                gt[row, col] = 1.0
    return gt


def partimagenetpp_manifest_path(args: argparse.Namespace, split: str) -> Path:
    attr = f"partimagenetpp_{split}_manifest"
    explicit = str(getattr(args, attr, "") or "")
    env_name = f"PARTIMAGENETPP_{split.upper()}_MANIFEST"
    fallback = os.environ.get(env_name) or os.environ.get("PARTIMAGENETPP_MANIFEST", "")
    path = explicit or fallback
    if not path:
        raise ValueError(f"--{attr} or ${env_name} is required for PartImageNet++ concept accuracy")
    return Path(path).resolve()


def configure_partimagenetpp_env(args: argparse.Namespace) -> None:
    if getattr(args, "partimagenetpp_train_manifest", ""):
        os.environ["PARTIMAGENETPP_TRAIN_MANIFEST"] = str(Path(args.partimagenetpp_train_manifest).resolve())
    if getattr(args, "partimagenetpp_val_manifest", ""):
        os.environ["PARTIMAGENETPP_VAL_MANIFEST"] = str(Path(args.partimagenetpp_val_manifest).resolve())


def load_partimagenetpp_presence_gt(indices: Sequence[int], concepts: Sequence[str], args: argparse.Namespace) -> np.ndarray:
    manifest_path = partimagenetpp_manifest_path(args, "val")
    concept_to_col = {concept: idx for idx, concept in enumerate(concepts)}
    rows: List[dict] = []
    with manifest_path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    gt = np.zeros((len(indices), len(concepts)), dtype=np.float32)
    for row, dataset_idx in enumerate(indices):
        if int(dataset_idx) >= len(rows):
            continue
        payload = rows[int(dataset_idx)]
        labels = payload.get("labels", payload.get("queried_parts", []))
        for label in labels if isinstance(labels, list) else []:
            col = concept_to_col.get(canonicalize_concept(str(label), "partimagenetpp"))
            if col is not None:
                gt[row, col] = 1.0
    return gt


def partimagenetpp_gt_boxes_path(args: argparse.Namespace) -> Path:
    explicit = str(getattr(args, "partimagenetpp_gt_boxes_jsonl", "") or "")
    fallback = os.environ.get("PARTIMAGENETPP_GT_BOXES_JSONL", "")
    path = explicit or fallback
    if not path:
        raise ValueError("--partimagenetpp_gt_boxes_jsonl or $PARTIMAGENETPP_GT_BOXES_JSONL is required")
    return Path(path).resolve()


def load_partimagenetpp_box_presence_gt(indices: Sequence[int], concepts: Sequence[str], args: argparse.Namespace) -> np.ndarray:
    gt_path = partimagenetpp_gt_boxes_path(args)
    concept_to_col = {concept: idx for idx, concept in enumerate(concepts)}
    rows: List[dict] = []
    with gt_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    gt = np.zeros((len(indices), len(concepts)), dtype=np.float32)
    for row, dataset_idx in enumerate(indices):
        dataset_idx = int(dataset_idx)
        if dataset_idx >= len(rows):
            continue
        payload = rows[dataset_idx]
        boxes_by_concept = payload.get("boxes", {})
        if not isinstance(boxes_by_concept, dict):
            continue
        for label, boxes in boxes_by_concept.items():
            if not boxes:
                continue
            col = concept_to_col.get(canonicalize_concept(str(label), "partimagenetpp"))
            if col is not None:
                gt[row, col] = 1.0
    return gt


def safe_roc_auc(gt: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(gt)) < 2:
        return float("nan")
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(gt, scores))


def safe_average_precision(gt: np.ndarray, scores: np.ndarray) -> float:
    if gt.sum() <= 0:
        return float("nan")
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(gt, scores))


def macro_average_precision(gt: np.ndarray, scores: np.ndarray) -> float:
    per_concept = [safe_average_precision(gt[:, idx], scores[:, idx]) for idx in range(gt.shape[1])]
    valid = [value for value in per_concept if not math.isnan(value)]
    return float(np.mean(valid)) if valid else float("nan")


def precision_at_k(gt: np.ndarray, scores: np.ndarray, k: int = 5) -> float:
    if gt.size == 0 or scores.shape[1] == 0:
        return float("nan")
    k = min(int(k), int(scores.shape[1]))
    top_idx = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    hits = np.take_along_axis(gt, top_idx, axis=1)
    return float(hits.mean())


def confusion_metrics(gt: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    gt_bool = gt.astype(bool)
    pred_bool = pred.astype(bool)
    tp = int((gt_bool & pred_bool).sum())
    fp = int((~gt_bool & pred_bool).sum())
    fn = int((gt_bool & ~pred_bool).sum())
    tn = int((~gt_bool & ~pred_bool).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    accuracy = (tp + tn) / max(tp + fp + fn + tn, 1)
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def best_f1_metrics(gt: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    if gt.sum() <= 0:
        payload = confusion_metrics(gt, scores > 0)
        payload["threshold"] = 0.0
        return payload
    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(gt, scores)
    f1 = 2.0 * precision * recall / np.maximum(precision + recall, 1e-12)
    best_idx = int(np.nanargmax(f1))
    threshold = float(thresholds[min(best_idx, len(thresholds) - 1)]) if len(thresholds) else 0.0
    payload = confusion_metrics(gt, scores >= threshold)
    payload["threshold"] = threshold
    payload["f1"] = float(f1[best_idx])
    payload["precision"] = float(precision[best_idx])
    payload["recall"] = float(recall[best_idx])
    return payload


def per_concept_metrics(gt: np.ndarray, scores: np.ndarray, concepts: Sequence[str], threshold: float) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for idx, concept in enumerate(concepts):
        gt_col = gt[:, idx].reshape(-1)
        score_col = scores[:, idx].reshape(-1)
        row: Dict[str, Any] = {
            "concept": concept,
            "gt_positive_rate": float(gt_col.mean()) if gt_col.size else 0.0,
            "auroc": safe_roc_auc(gt_col, score_col),
            "ap": safe_average_precision(gt_col, score_col),
        }
        row.update({f"threshold_{key}": value for key, value in confusion_metrics(gt_col, score_col >= threshold).items()})
        rows.append(row)
    return rows


def apply_normalization(
    scores: torch.Tensor,
    mode: str,
    *,
    train_scores: Optional[torch.Tensor] = None,
    saved_mean: Optional[torch.Tensor] = None,
    saved_std: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    mode = str(mode).lower()
    x = scores.float()
    if mode in {"model_default", "none"}:
        return x
    if mode == "sigmoid":
        return torch.sigmoid(x)
    if mode in {"minmax", "split_minmax"}:
        lo = x.amin(dim=0, keepdim=True)
        hi = x.amax(dim=0, keepdim=True)
        return (x - lo) / (hi - lo).clamp_min(1e-6)
    if mode == "train_minmax":
        if train_scores is None:
            raise ValueError("train_minmax normalization requires train concept scores")
        train = train_scores.float()
        lo = train.amin(dim=0, keepdim=True)
        hi = train.amax(dim=0, keepdim=True)
        return (x - lo) / (hi - lo).clamp_min(1e-6)
    if mode == "split_zscore":
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
        return (x - mean) / std
    if mode == "train_zscore":
        if train_scores is None:
            raise ValueError("train_zscore normalization requires train concept scores")
        mean = train_scores.float().mean(dim=0, keepdim=True)
        std = train_scores.float().std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
        return (x - mean) / std
    if mode == "saved_zscore":
        if saved_mean is None or saved_std is None:
            raise ValueError("saved_zscore normalization requires saved projection stats")
        return (x - saved_mean.reshape(1, -1).float()) / saved_std.reshape(1, -1).float().clamp_min(1e-6)
    if mode in {"saved_zscore_minmax", "concept_zscore_minmax"}:
        if saved_mean is None or saved_std is None:
            raise ValueError(f"{mode} normalization requires saved projection stats")
        z = (x - saved_mean.reshape(1, -1).float()) / saved_std.reshape(1, -1).float().clamp_min(1e-6)
        lo = z.amin(dim=0, keepdim=True)
        hi = z.amax(dim=0, keepdim=True)
        return (z - lo) / (hi - lo).clamp_min(1e-6)
    raise ValueError(f"Unsupported normalization: {mode}")


def subset_by_concepts(
    scores: torch.Tensor,
    concepts: Sequence[str],
    wanted: Sequence[str],
    *,
    train_scores: Optional[torch.Tensor] = None,
    saved_mean: Optional[torch.Tensor] = None,
    saved_std: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    concept_to_idx = {concept: idx for idx, concept in enumerate(concepts)}
    indices = torch.tensor([concept_to_idx[concept] for concept in wanted], dtype=torch.long)
    out_scores = scores.index_select(1, indices)
    out_train = train_scores.index_select(1, indices) if train_scores is not None else None
    out_mean = saved_mean.index_select(0, indices) if saved_mean is not None else None
    out_std = saved_std.index_select(0, indices) if saved_std is not None else None
    return out_scores, out_train, out_mean, out_std


def align_score_columns_to_concepts(scores: torch.Tensor, concepts: Sequence[str], context: str) -> torch.Tensor:
    if int(scores.shape[1]) == len(concepts):
        return scores
    if int(scores.shape[1]) > len(concepts):
        cache = getattr(align_score_columns_to_concepts, "_warned", set())
        key = (context, int(scores.shape[1]), len(concepts))
        if key not in cache:
            print(
                f"[concept-acc] {context}: score columns {scores.shape[1]} > named concepts {len(concepts)}; "
                "using the leading named columns and dropping unnamed extras",
                flush=True,
            )
            cache.add(key)
            setattr(align_score_columns_to_concepts, "_warned", cache)
        return scores[:, : len(concepts)]
    raise RuntimeError(f"{context}: score columns {scores.shape[1]} < named concepts {len(concepts)}")


def cub_annotation_dir(args: argparse.Namespace) -> Path:
    if not args.annotation_dir:
        raise ValueError("--annotation_dir is required for CUB GDINO concept accuracy")
    root = Path(args.annotation_dir)
    split_dir = root / "cub_val"
    return split_dir if split_dir.is_dir() else root


def infer_model_name(path: Path, explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    args_path = path / "args.txt"
    if args_path.exists():
        payload = json.loads(args_path.read_text())
        return str(payload.get("model_name", "vlg_cbm"))
    config_path = path / "config.json"
    if config_path.exists():
        payload = json.loads(config_path.read_text())
        return str(payload.get("model", payload.get("branch_arch", "sgcbm")))
    return "sgcbm"


def extract_cub_scores(path: Path, model_name: str, args: argparse.Namespace) -> Dict[str, Any]:
    normalized_model = str(model_name).lower().replace("-", "_")
    if args.cub_score_source == "concept_prediction" and normalized_model in {"savlg_cbm", "sg_cbm", "sgcbm"}:
        return extract_cub_savlg_raw_scores(path, args)
    if args.cub_score_source == "concept_prediction" and normalized_model in {"vlg_cbm", "cub_cbm"}:
        return extract_cub_vlg_raw_scores(path, args)
    if args.cub_score_source == "concept_prediction" and normalized_model == "salf_cbm":
        return extract_cub_salf_raw_scores(path, args)

    from evaluations.sparse_utils import build_nec_feature_set

    feature_set, run_args = build_nec_feature_set(
        str(path),
        model_name,
        annotation_dir=args.annotation_dir or None,
        cbl_batch_size=args.batch_size,
        saga_batch_size=args.batch_size,
        num_workers=args.num_workers,
        savlg_alpha_override=args.savlg_alpha_override,
        disable_activation_cache=args.disable_activation_cache,
        max_images=args.max_images or None,
        savlg_branch_norm_mode=args.savlg_branch_norm_mode,
    )
    scores = feature_set.test_features.float()
    train_scores = feature_set.train_features.float() if feature_set.train_features is not None else None
    concepts = [canonicalize_concept(c, "cub") for c in feature_set.concepts if c]
    n = int(scores.shape[0])
    indices = list(range(n))
    # The CUB extractors return each model's default concept-score space. For
    # VLG/SALF/SG-CBM this already includes the saved projection normalization,
    # so do not expose proj_mean/proj_std here and risk double-normalizing.
    saved_mean = saved_std = None
    return {"scores": scores, "train_scores": train_scores, "concepts": concepts, "indices": indices, "run_args": vars(run_args), "saved_mean": saved_mean, "saved_std": saved_std}


def extract_cub_vlg_raw_scores(path: Path, args: argparse.Namespace) -> Dict[str, Any]:
    from model.cbm import Backbone, BackboneCLIP, ConceptLayer
    from data import utils as data_utils

    run_args = argparse.Namespace(**json.loads((path / "args.txt").read_text()))
    run_args.device = args.device
    run_args.cbl_batch_size = int(args.batch_size)
    run_args.num_workers = int(args.num_workers)
    concepts = load_concepts(path / "concepts.txt", "cub")
    if str(run_args.backbone).startswith("clip_"):
        backbone = BackboneCLIP(run_args.backbone, device=run_args.device, use_penultimate=getattr(run_args, "use_clip_penultimate", False))
    else:
        backbone = Backbone(run_args.backbone, run_args.feature_layer, run_args.device)
    backbone_path = path / "backbone.pt"
    if backbone_path.exists():
        backbone.backbone.load_state_dict(torch.load(backbone_path, map_location=run_args.device))
    concept_layer = ConceptLayer.from_pretrained(str(path), run_args.device)
    backbone.eval()
    concept_layer.eval()

    def extract(split: str) -> Tuple[torch.Tensor, torch.Tensor]:
        dataset = data_utils.get_data(f"{run_args.dataset}_{split}", preprocess=backbone.preprocess)
        if int(args.max_images) > 0:
            dataset = Subset(dataset, list(range(min(int(args.max_images), len(dataset)))))
        loader = DataLoader(dataset, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers))
        score_chunks: List[torch.Tensor] = []
        label_chunks: List[torch.Tensor] = []
        with torch.no_grad():
            for images, labels in loader:
                scores = concept_layer(backbone(images.to(run_args.device)))
                score_chunks.append(scores.detach().cpu().float())
                label_chunks.append(labels.detach().cpu())
        return torch.cat(score_chunks, dim=0), torch.cat(label_chunks, dim=0)

    scores, test_labels = extract("val")
    train_scores = None
    if str(args.normalization).lower() in {"train_zscore", "train_minmax"}:
        train_scores, _train_labels = extract("train")
    saved_mean = torch.load(path / "proj_mean.pt", map_location="cpu") if (path / "proj_mean.pt").exists() else None
    saved_std = torch.load(path / "proj_std.pt", map_location="cpu") if (path / "proj_std.pt").exists() else None
    if saved_mean is not None:
        saved_mean = saved_mean.reshape(-1)[: len(concepts)]
    if saved_std is not None:
        saved_std = saved_std.reshape(-1)[: len(concepts)]
    return {
        "scores": scores,
        "train_scores": train_scores,
        "concepts": concepts,
        "indices": list(range(int(scores.shape[0]))),
        "run_args": vars(run_args),
        "saved_mean": saved_mean,
        "saved_std": saved_std,
        "test_labels": test_labels,
    }


def extract_cub_salf_raw_scores(path: Path, args: argparse.Namespace) -> Dict[str, Any]:
    from data import utils as data_utils
    from methods.salf import SpatialBackbone, build_spatial_concept_layer, pool_salf_maps

    run_args = argparse.Namespace(**json.loads((path / "args.txt").read_text()))
    run_args.device = args.device
    run_args.cbl_batch_size = int(args.batch_size)
    run_args.num_workers = int(args.num_workers)
    concepts = load_concepts(path / "concepts.txt", getattr(run_args, "dataset", args.dataset))
    backbone = SpatialBackbone(
        run_args.backbone,
        device=run_args.device,
        checkpoint_path=getattr(run_args, "backbone_checkpoint", ""),
    )
    state_dict = torch.load(path / "concept_layer.pt", map_location=run_args.device)
    if "weight" in state_dict and state_dict["weight"].ndim == 4:
        concept_layer = torch.nn.Conv2d(int(state_dict["weight"].shape[1]), int(state_dict["weight"].shape[0]), kernel_size=1, bias=("bias" in state_dict)).to(run_args.device)
    elif "spatial_layer.weight" in state_dict:
        n_outputs = int(state_dict["spatial_layer.weight"].shape[0])
        concept_layer = build_spatial_concept_layer(
            run_args,
            backbone.output_dim,
            n_outputs,
            is_vit=getattr(backbone, "is_vit", False),
        )
    else:
        raise ValueError(f"Unsupported SALF concept layer format at {path / 'concept_layer.pt'}")
    concept_layer.load_state_dict(state_dict)
    backbone.eval()
    concept_layer.eval()

    def extract(split: str) -> Tuple[torch.Tensor, torch.Tensor]:
        dataset = data_utils.get_data(f"{run_args.dataset}_{split}", preprocess=backbone.preprocess)
        if int(args.max_images) > 0:
            dataset = Subset(dataset, list(range(min(int(args.max_images), len(dataset)))))
        loader = DataLoader(dataset, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers))
        score_chunks: List[torch.Tensor] = []
        label_chunks: List[torch.Tensor] = []
        with torch.no_grad():
            for images, labels in loader:
                maps = concept_layer(backbone(images.to(run_args.device)))
                if isinstance(maps, tuple):
                    maps = maps[0]
                scores = pool_salf_maps(run_args, maps) if maps.ndim > 2 else maps
                scores = align_score_columns_to_concepts(scores.detach().cpu().float(), concepts, f"SALF {split}")
                score_chunks.append(scores)
                label_chunks.append(labels.detach().cpu())
        return torch.cat(score_chunks, dim=0), torch.cat(label_chunks, dim=0)

    scores, test_labels = extract("val")
    train_scores = None
    if str(args.normalization).lower() in {"train_zscore", "train_minmax"}:
        train_scores, _train_labels = extract("train")
    return {"scores": scores, "train_scores": train_scores, "concepts": concepts, "indices": list(range(int(scores.shape[0]))), "run_args": vars(run_args), "saved_mean": None, "saved_std": None, "test_labels": test_labels}


def extract_cub_savlg_raw_scores(path: Path, args: argparse.Namespace) -> Dict[str, Any]:
    from methods.savlg import build_savlg_concept_layer, compute_savlg_concept_logits, create_savlg_splits, forward_savlg_backbone, forward_savlg_concept_layer

    run_args = argparse.Namespace(**json.loads((path / "args.txt").read_text()))
    run_args.device = args.device
    run_args.cbl_batch_size = int(args.batch_size)
    run_args.saga_batch_size = int(args.batch_size)
    run_args.num_workers = int(args.num_workers)
    if args.annotation_dir:
        run_args.annotation_dir = args.annotation_dir
    if args.disable_activation_cache:
        run_args.use_activation_cache = False
        run_args.disable_activation_cache = True
    if getattr(run_args, "skip_test_eval", False):
        run_args.skip_test_eval = False

    concepts = load_concepts(path / "concepts.txt", getattr(run_args, "dataset", args.dataset))
    _train_cbl, _val_cbl, train_dataset, _val_dataset, test_dataset, backbone = create_savlg_splits(run_args)
    if int(args.max_images) > 0:
        train_dataset = Subset(train_dataset, list(range(min(int(args.max_images), len(train_dataset)))))
        test_dataset = Subset(test_dataset, list(range(min(int(args.max_images), len(test_dataset)))))

    state_dict = torch.load(path / "concept_layer.pt", map_location=run_args.device)
    if "spatial_layer.weight" in state_dict:
        n_outputs = int(state_dict["spatial_layer.weight"].shape[0])
    elif "global_layer.weight" in state_dict:
        n_outputs = int(state_dict["global_layer.weight"].shape[0])
    elif "weight" in state_dict:
        n_outputs = int(state_dict["weight"].shape[0])
    else:
        n_outputs = len(concepts)
    concept_layer = build_savlg_concept_layer(run_args, backbone, n_outputs)
    concept_layer.load_state_dict(state_dict)
    concept_layer.eval()
    backbone.eval()

    def extract(dataset: Dataset) -> Tuple[torch.Tensor, torch.Tensor]:
        loader = DataLoader(dataset, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers))
        score_chunks: List[torch.Tensor] = []
        label_chunks: List[torch.Tensor] = []
        with torch.no_grad():
            for images, labels in loader:
                features = forward_savlg_backbone(backbone, images.to(run_args.device), run_args)
                global_logits, spatial_maps = forward_savlg_concept_layer(concept_layer, features)
                _global, _spatial, final_logits = compute_savlg_concept_logits(global_logits, spatial_maps, run_args)
                final_logits = align_score_columns_to_concepts(
                    final_logits.detach().cpu().float(),
                    concepts,
                    "SAVLG",
                )
                score_chunks.append(final_logits)
                label_chunks.append(labels.detach().cpu())
        return torch.cat(score_chunks, dim=0), torch.cat(label_chunks, dim=0)

    scores, test_labels = extract(test_dataset)
    train_scores = None
    if str(args.normalization).lower() in {"train_zscore", "train_minmax"}:
        train_scores, _train_labels = extract(train_dataset)
    return {
        "scores": scores,
        "train_scores": train_scores,
        "concepts": concepts,
        "indices": list(range(int(scores.shape[0]))),
        "run_args": vars(run_args),
        "saved_mean": None,
        "saved_std": None,
        "test_labels": test_labels,
    }


def load_filename_to_annotation_mapping(mapping_path: Path, annotation_val_dir: Path) -> Dict[str, Path]:
    payload = json.loads(mapping_path.read_text())
    entries = payload.get("items", payload if isinstance(payload, list) else [])
    mapping: Dict[str, Path] = {}
    for item in entries:
        if not isinstance(item, dict):
            continue
        name = item.get("image_name", item.get("filename", ""))
        ann_file = item.get("annotation_file", "")
        if not name:
            continue
        if not ann_file and "annotation_index" in item:
            ann_file = f"{int(item['annotation_index'])}.json"
        if ann_file:
            mapping[Path(str(name)).name] = annotation_val_dir / str(ann_file)
    if not mapping:
        raise RuntimeError(f"no filename mapping entries found in {mapping_path}")
    return mapping


class ImageNetConceptDataset(Dataset):
    def __init__(
        self,
        val_root: Path,
        annotation_dir: Path,
        annotation_val_root: Optional[Path],
        annotation_mapping_json: Optional[Path],
        input_size: int = 224,
        transform: Optional[Any] = None,
    ) -> None:
        self.val_root = val_root
        self.annotation_dir = annotation_dir
        self.annotation_val_root = annotation_val_root
        if annotation_mapping_json is not None:
            self.filename_map = load_filename_to_annotation_mapping(annotation_mapping_json, annotation_dir)
        elif annotation_val_root is not None:
            self.filename_map = build_filename_to_annotation_path(annotation_dir, annotation_val_root)
        else:
            default_mapping = annotation_dir.parent / "imagenet_val_filename_to_annotation.json"
            self.filename_map = load_filename_to_annotation_mapping(default_mapping, annotation_dir) if default_mapping.exists() else None
        self.transform = transform or transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        if any(p.is_dir() for p in val_root.iterdir()):
            self.dataset = ImageFolder(str(val_root))
            self.samples = [(Path(path), label) for path, label in self.dataset.samples]
        else:
            self.dataset = None
            self.samples = [(path, -1) for path in sorted(val_root.iterdir()) if VAL_RE.search(path.name)]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        path, _label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            image_size = (int(image.size[0]), int(image.size[1]))
            tensor = self.transform(image)
        match = VAL_RE.search(path.name)
        image_index_1based = int(match.group(1)) if match else index + 1
        annotation = load_annotation_payload(self.annotation_dir, image_index_1based, path.name, self.filename_map)
        return {"image": tensor, "annotation": annotation, "image_size": image_size, "name": path.name}


def imagenet_collate(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "images": torch.stack([item["image"] for item in batch], dim=0),
        "annotations": [item["annotation"] for item in batch],
        "image_sizes": [item["image_size"] for item in batch],
        "names": [item["name"] for item in batch],
    }


def load_imagenet_config(source_run_dir: Path, args: argparse.Namespace) -> Config:
    cfg_args = argparse.Namespace(**vars(args), workers=int(args.num_workers))
    try:
        cfg = load_run_config(source_run_dir, cfg_args)
    except TypeError:
        payload = json.loads((source_run_dir / "config.json").read_text())
        payload.setdefault("feature_storage_dtype", "fp16")
        payload.setdefault("saga_table_device", "cpu")
        payload.setdefault("dense_lr", 1e-3)
        payload.setdefault("dense_n_iters", 20)
        payload.setdefault("train_random_transforms", False)
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
    return cfg


def imagenet_annotation_inputs(args: argparse.Namespace) -> Tuple[Path, Optional[Path], Optional[Path]]:
    if not args.val_root:
        raise ValueError("--val_root is required for ImageNet concept accuracy")
    if not args.annotation_dir:
        raise ValueError("--annotation_dir is required for ImageNet concept accuracy")
    annotation_val_dir = resolve_val_annotation_dir(Path(args.annotation_dir).resolve())
    annotation_val_root = Path(args.annotation_val_root).resolve() if args.annotation_val_root else None
    annotation_mapping_json = Path(args.annotation_mapping_json).resolve() if args.annotation_mapping_json else None
    return annotation_val_dir, annotation_val_root, annotation_mapping_json


def resolve_split_annotation_dir(annotation_dir: Path, split_name: str) -> Path:
    if (annotation_dir / "0.json").is_file():
        return annotation_dir.resolve()
    candidate = annotation_dir / split_name
    if (candidate / "0.json").is_file():
        return candidate.resolve()
    raise FileNotFoundError(f"Could not find {split_name}/0.json under {annotation_dir}")


def make_imagenet_dataset(args: argparse.Namespace, transform: Optional[Any] = None, input_size: int = 224) -> Dataset:
    annotation_val_dir, annotation_val_root, annotation_mapping_json = imagenet_annotation_inputs(args)
    dataset: Dataset = ImageNetConceptDataset(
        Path(args.val_root).resolve(),
        annotation_val_dir,
        annotation_val_root,
        annotation_mapping_json,
        input_size=input_size,
        transform=transform,
    )
    if args.max_images > 0:
        dataset = Subset(dataset, list(range(min(args.max_images, len(dataset)))))
    return dataset


def make_imagenet_loader(dataset: Dataset, args: argparse.Namespace) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=bool(args.pin_memory),
        collate_fn=imagenet_collate,
        **({"prefetch_factor": int(args.prefetch_factor), "persistent_workers": bool(args.persistent_workers)} if int(args.num_workers) > 0 else {}),
    )


class Places365ConceptDataset(Dataset):
    def __init__(self, manifest: Path, annotation_dir: Path, input_size: int, transform: Optional[Any] = None) -> None:
        self.rows: List[Dict[str, Any]] = []
        with manifest.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))
        if not self.rows:
            raise ValueError(f"Places365 manifest has no rows: {manifest}")
        self.annotation_dir = annotation_dir
        self.transform = transform or transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def __len__(self) -> int:
        return len(self.rows)

    @staticmethod
    def _resolve_image_path(path: Path) -> Path:
        if path.exists():
            return path
        candidates: List[Path] = []
        places_root = os.environ.get("PLACES365_ROOT", "")
        if places_root:
            root = Path(places_root)
            if not path.is_absolute():
                candidates.append(root / path)
            candidates.extend((root / path.name, root / "val_256" / path.name))
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return path

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.rows[index]
        image_path = self._resolve_image_path(Path(row["path"]))
        annotation_index = int(row.get("annotation_index", row.get("sample_index", index)))
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        annotation_path = self.annotation_dir / f"{annotation_index}.json"
        if annotation_path.is_file():
            payload = json.loads(annotation_path.read_text(encoding="utf-8"))
            annotations = payload if isinstance(payload, list) else payload.get("concepts", [])
        else:
            annotations = []
        return {"image": tensor, "annotation": annotations}


def places365_collate(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "images": torch.stack([item["image"] for item in batch], dim=0),
        "annotations": [item["annotation"] for item in batch],
    }


def places365_manifest_path(args: argparse.Namespace) -> Path:
    path = str(getattr(args, "places365_val_manifest", "") or "")
    if not path:
        raise ValueError("--places365_val_manifest is required for Places365 concept accuracy")
    return Path(path).resolve()


def make_places365_dataset(args: argparse.Namespace, transform: Optional[Any] = None, input_size: int = 224) -> Dataset:
    if not args.annotation_dir:
        raise ValueError("--annotation_dir is required for Places365 concept accuracy")
    annotation_val_dir = resolve_split_annotation_dir(Path(args.annotation_dir).resolve(), "places365_val")
    dataset: Dataset = Places365ConceptDataset(
        places365_manifest_path(args),
        annotation_val_dir,
        input_size=input_size,
        transform=transform,
    )
    if args.max_images > 0:
        dataset = Subset(dataset, list(range(min(args.max_images, len(dataset)))))
    return dataset


def make_places365_loader(dataset: Dataset, args: argparse.Namespace) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=bool(args.pin_memory),
        collate_fn=places365_collate,
        **({"prefetch_factor": int(args.prefetch_factor), "persistent_workers": bool(args.persistent_workers)} if int(args.num_workers) > 0 else {}),
    )


def extract_imagenet_from_forward(
    *,
    args: argparse.Namespace,
    concepts: Sequence[str],
    forward_scores,
    transform: Optional[Any],
    input_size: int,
    device: str,
    log_tag: str,
    saved_mean: Optional[torch.Tensor] = None,
    saved_std: Optional[torch.Tensor] = None,
    run_args: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    dataset = make_imagenet_dataset(args, transform=transform, input_size=input_size)
    loader = make_imagenet_loader(dataset, args)
    scores: List[torch.Tensor] = []
    gt_rows: List[np.ndarray] = []
    start = time.perf_counter()
    seen = 0
    concept_to_idx = {concept: idx for idx, concept in enumerate(concepts)}
    with torch.no_grad():
        for batch in loader:
            batch_scores = forward_scores(batch["images"].to(device))
            scores.append(batch_scores.detach().cpu().float())
            gt_rows.append(gdino_gt_from_annotations(batch["annotations"], concepts, concept_to_idx, "imagenet", args.gdino_threshold))
            seen += int(batch["images"].shape[0])
            if args.log_every > 0 and seen % int(args.log_every) == 0:
                elapsed = time.perf_counter() - start
                print(f"[concept-acc:{log_tag}] n={seen} ips={seen / max(elapsed, 1e-6):.2f}", flush=True)
    return {
        "scores": torch.cat(scores, dim=0),
        "train_scores": None,
        "concepts": list(concepts),
        "gt": np.concatenate(gt_rows, axis=0),
        "indices": list(range(sum(row.shape[0] for row in gt_rows))),
        "run_args": run_args or {},
        "saved_mean": saved_mean,
        "saved_std": saved_std,
    }


def extract_places365_from_forward(
    *,
    args: argparse.Namespace,
    concepts: Sequence[str],
    forward_scores,
    transform: Optional[Any],
    input_size: int,
    device: str,
    log_tag: str,
    saved_mean: Optional[torch.Tensor] = None,
    saved_std: Optional[torch.Tensor] = None,
    run_args: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    dataset = make_places365_dataset(args, transform=transform, input_size=input_size)
    loader = make_places365_loader(dataset, args)
    scores: List[torch.Tensor] = []
    gt_rows: List[np.ndarray] = []
    start = time.perf_counter()
    seen = 0
    concept_to_idx = {concept: idx for idx, concept in enumerate(concepts)}
    with torch.no_grad():
        for batch in loader:
            batch_scores = forward_scores(batch["images"].to(device))
            scores.append(batch_scores.detach().cpu().float())
            gt_rows.append(gdino_gt_from_annotations(batch["annotations"], concepts, concept_to_idx, "places365", args.gdino_threshold))
            seen += int(batch["images"].shape[0])
            if args.log_every > 0 and seen % int(args.log_every) == 0:
                elapsed = time.perf_counter() - start
                print(f"[concept-acc:{log_tag}] n={seen} ips={seen / max(elapsed, 1e-6):.2f}", flush=True)
    return {
        "scores": torch.cat(scores, dim=0),
        "train_scores": None,
        "concepts": list(concepts),
        "gt": np.concatenate(gt_rows, axis=0),
        "indices": list(range(sum(row.shape[0] for row in gt_rows))),
        "run_args": run_args or {},
        "saved_mean": saved_mean,
        "saved_std": saved_std,
    }


def load_legacy_imagenet_args(path: Path, args: argparse.Namespace) -> argparse.Namespace:
    if (path / "args.txt").exists():
        run_args = argparse.Namespace(**json.loads((path / "args.txt").read_text()))
    else:
        run_args = argparse.Namespace(dataset="imagenet", backbone="resnet50", feature_layer="layer4", device=args.device)
    run_args.device = args.device
    run_args.cbl_batch_size = int(args.batch_size)
    run_args.num_workers = int(args.num_workers)
    return run_args


def load_legacy_places365_args(path: Path, args: argparse.Namespace) -> argparse.Namespace:
    if (path / "args.txt").exists():
        run_args = argparse.Namespace(**json.loads((path / "args.txt").read_text()))
    else:
        run_args = argparse.Namespace(dataset="places365", backbone="resnet50", feature_layer="layer4", device=args.device)
    run_args.device = args.device
    run_args.backbone = getattr(run_args, "backbone", "resnet50")
    run_args.feature_layer = getattr(run_args, "feature_layer", "layer4")
    run_args.use_clip_penultimate = getattr(run_args, "use_clip_penultimate", False)
    run_args.cbl_batch_size = int(args.batch_size)
    run_args.num_workers = int(args.num_workers)
    return run_args


def load_projection_stats(path: Path, n_concepts: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    mean_path = path / "proj_mean.pt"
    std_path = path / "proj_std.pt"
    if not mean_path.exists() or not std_path.exists():
        return None, None
    mean = torch.load(mean_path, map_location="cpu").float().flatten()
    std = torch.load(std_path, map_location="cpu").float().flatten()
    if int(mean.numel()) != int(n_concepts) or int(std.numel()) != int(n_concepts):
        print(
            f"[concept-acc] ignoring projection stats in {path}: "
            f"mean/std size {mean.numel()}/{std.numel()} != n_concepts {n_concepts}",
            flush=True,
        )
        return None, None
    return mean, std


def extract_imagenet_legacy_linear_scores(path: Path, model_name: str, args: argparse.Namespace) -> Dict[str, Any]:
    from model.cbm import Backbone, BackboneCLIP, ConceptLayer

    run_args = load_legacy_imagenet_args(path, args)
    concepts = load_concepts(path / "concepts.txt", "imagenet")
    if str(run_args.backbone).startswith("clip_"):
        backbone = BackboneCLIP(run_args.backbone, device=run_args.device, use_penultimate=getattr(run_args, "use_clip_penultimate", False))
    else:
        backbone = Backbone(run_args.backbone, run_args.feature_layer, run_args.device)
    cbl_path = path / "cbl.pt"
    wc_path = path / "W_c.pt"
    if model_name == "lf_cbm":
        if not wc_path.exists():
            raise RuntimeError(f"LF-CBM checkpoint {path} is missing W_c.pt")
        W_c = torch.load(wc_path, map_location=run_args.device).float()
        if int(W_c.shape[0]) != len(concepts):
            raise RuntimeError(
                f"LF-CBM W_c rows ({W_c.shape[0]}) do not match concepts.txt ({len(concepts)}) in {path}. "
                "This checkpoint cannot be evaluated faithfully without the matching LF concept list."
            )
        concept_layer = torch.nn.Linear(int(W_c.shape[1]), int(W_c.shape[0]), bias=False).to(run_args.device)
        concept_layer.load_state_dict({"weight": W_c})
    elif cbl_path.exists():
        state = torch.load(cbl_path, map_location="cpu")
        out_features = int(state["model.0.weight"].shape[0]) if isinstance(state, dict) and "model.0.weight" in state else len(concepts)
        if out_features == len(concepts):
            concept_layer = ConceptLayer.from_pretrained(str(path), run_args.device)
        elif wc_path.exists():
            W_c = torch.load(wc_path, map_location=run_args.device).float()
            if int(W_c.shape[0]) != len(concepts):
                raise RuntimeError(f"{model_name} W_c rows ({W_c.shape[0]}) do not match concepts.txt ({len(concepts)}) in {path}")
            concept_layer = torch.nn.Linear(int(W_c.shape[1]), int(W_c.shape[0]), bias=False).to(run_args.device)
            concept_layer.load_state_dict({"weight": W_c})
        else:
            raise RuntimeError(f"{model_name} cbl.pt output count ({out_features}) does not match concepts.txt ({len(concepts)}) in {path}")
    elif wc_path.exists():
        W_c = torch.load(wc_path, map_location=run_args.device).float()
        if int(W_c.shape[0]) != len(concepts):
            raise RuntimeError(f"{model_name} W_c rows ({W_c.shape[0]}) do not match concepts.txt ({len(concepts)}) in {path}")
        concept_layer = torch.nn.Linear(int(W_c.shape[1]), int(W_c.shape[0]), bias=False).to(run_args.device)
        concept_layer.load_state_dict({"weight": W_c})
    else:
        raise RuntimeError(f"No cbl.pt or W_c.pt found in legacy ImageNet checkpoint {path}")
    backbone.eval()
    concept_layer.eval()
    saved_mean, saved_std = load_projection_stats(path, len(concepts))

    def forward_scores(images: torch.Tensor) -> torch.Tensor:
        return concept_layer(backbone(images))

    return extract_imagenet_from_forward(
        args=args,
        concepts=concepts,
        forward_scores=forward_scores,
        transform=backbone.preprocess,
        input_size=224,
        device=run_args.device,
        log_tag=f"imagenet:{model_name}",
        saved_mean=saved_mean,
        saved_std=saved_std,
        run_args=vars(run_args),
    )


def extract_places365_legacy_linear_scores(path: Path, model_name: str, args: argparse.Namespace) -> Dict[str, Any]:
    from model.cbm import Backbone, BackboneCLIP, ConceptLayer

    run_args = load_legacy_places365_args(path, args)
    concepts = load_concepts(path / "concepts.txt", "places365")
    if str(run_args.backbone).startswith("clip_"):
        backbone = BackboneCLIP(run_args.backbone, device=run_args.device, use_penultimate=run_args.use_clip_penultimate)
    else:
        backbone = Backbone(run_args.backbone, run_args.feature_layer, run_args.device)
    cbl_path = path / "cbl.pt"
    wc_path = path / "W_c.pt"
    if model_name == "lf_cbm":
        if not wc_path.exists():
            raise RuntimeError(f"LF-CBM checkpoint {path} is missing W_c.pt")
        W_c = torch.load(wc_path, map_location=run_args.device).float()
        concept_layer = torch.nn.Linear(int(W_c.shape[1]), int(W_c.shape[0]), bias=False).to(run_args.device)
        concept_layer.load_state_dict({"weight": W_c})
    elif cbl_path.exists():
        state = torch.load(cbl_path, map_location="cpu")
        if not isinstance(state, dict) or "model.0.weight" not in state:
            raise RuntimeError(f"Unsupported VLG-CBM cbl.pt format in {path}")
        linear_weight_keys = [key for key in state if key.startswith("model.") and key.endswith(".weight")]
        first_weight = state["model.0.weight"]
        concept_layer = ConceptLayer(
            in_features=int(first_weight.shape[1]),
            out_features=int(first_weight.shape[0]),
            num_hidden=max(0, len(linear_weight_keys) - 1),
            bias="model.0.bias" in state,
            device=run_args.device,
        )
        concept_layer.load_state_dict(state)
    elif wc_path.exists():
        W_c = torch.load(wc_path, map_location=run_args.device).float()
        concept_layer = torch.nn.Linear(int(W_c.shape[1]), int(W_c.shape[0]), bias=False).to(run_args.device)
        concept_layer.load_state_dict({"weight": W_c})
    else:
        raise RuntimeError(f"No cbl.pt or W_c.pt found in legacy Places365 checkpoint {path}")
    backbone.eval()
    concept_layer.eval()
    saved_mean, saved_std = load_projection_stats(path, len(concepts))

    def forward_scores(images: torch.Tensor) -> torch.Tensor:
        raw_scores = concept_layer(backbone(images))
        return align_score_columns_to_concepts(raw_scores, concepts, f"places365:{model_name}")

    return extract_places365_from_forward(
        args=args,
        concepts=concepts,
        forward_scores=forward_scores,
        transform=backbone.preprocess,
        input_size=224,
        device=run_args.device,
        log_tag=f"places365:{model_name}",
        saved_mean=saved_mean,
        saved_std=saved_std,
        run_args=vars(run_args),
    )


def extract_imagenet_salf_scores(path: Path, args: argparse.Namespace) -> Dict[str, Any]:
    from methods.salf import SpatialBackbone

    run_args = load_legacy_imagenet_args(path, args)
    concepts = load_concepts(path / "concepts.txt", "imagenet")
    backbone = SpatialBackbone(run_args.backbone, device=run_args.device, spatial_stage=getattr(run_args, "savlg_spatial_stage", "conv5"))
    if (path / "concept_layer.pt").exists():
        state_dict = torch.load(path / "concept_layer.pt", map_location=run_args.device)
        if "weight" not in state_dict or state_dict["weight"].ndim != 4:
            raise ValueError(f"Unsupported SALF concept layer format at {path / 'concept_layer.pt'}")
        weight = state_dict["weight"]
        has_bias = "bias" in state_dict
    elif (path / "W_c.pt").exists():
        weight = torch.load(path / "W_c.pt", map_location=run_args.device).float()
        if weight.ndim == 2:
            weight = weight[:, :, None, None]
        state_dict = {"weight": weight}
        has_bias = False
    else:
        raise RuntimeError(f"No concept_layer.pt or W_c.pt found in SALF checkpoint {path}")
    if int(weight.shape[0]) != len(concepts):
        raise RuntimeError(f"SALF concept layer outputs ({weight.shape[0]}) do not match concepts.txt ({len(concepts)}) in {path}")
    concept_layer = torch.nn.Conv2d(int(weight.shape[1]), int(weight.shape[0]), kernel_size=1, bias=has_bias).to(run_args.device)
    concept_layer.load_state_dict(state_dict)
    backbone.eval()
    concept_layer.eval()
    saved_mean, saved_std = load_projection_stats(path, len(concepts))

    def forward_scores(images: torch.Tensor) -> torch.Tensor:
        maps = concept_layer(backbone(images))
        if isinstance(maps, tuple):
            maps = maps[0]
        return maps.flatten(2).max(dim=2).values if maps.ndim > 2 else maps

    return extract_imagenet_from_forward(
        args=args,
        concepts=concepts,
        forward_scores=forward_scores,
        transform=backbone.preprocess,
        input_size=224,
        device=run_args.device,
        log_tag="imagenet:salf_cbm",
        saved_mean=saved_mean,
        saved_std=saved_std,
        run_args=vars(run_args),
    )


def extract_imagenet_unified_scores(path: Path, args: argparse.Namespace) -> Dict[str, Any]:
    if not args.val_root:
        raise ValueError("--val_root is required for ImageNet concept accuracy")
    if not args.annotation_dir:
        raise ValueError("--annotation_dir is required for ImageNet concept accuracy")
    source_run_dir = resolve_source_run_dir(path)
    cfg = load_imagenet_config(source_run_dir, args)
    configure_runtime(cfg)
    concepts = load_concepts(source_run_dir / "concepts.txt", "imagenet")
    backbone, head = build_model(cfg, n_concepts=len(concepts))
    head.load_state_dict(torch.load(source_run_dir / "concept_head_best.pt", map_location=cfg.device))
    backbone.eval()
    head.eval()
    def forward_scores(images: torch.Tensor) -> torch.Tensor:
        outputs = head(backbone(prepare_images(images, cfg)))
        return outputs["final_logits"]

    saved_mean = saved_std = None
    norm_path = path / "final_layer_normalization.pt"
    if not norm_path.exists():
        norm_path = source_run_dir / "final_layer_normalization.pt"
    if norm_path.exists():
        payload = torch.load(norm_path, map_location="cpu")
        saved_mean = payload.get("mean")
        saved_std = payload.get("std")
    return extract_imagenet_from_forward(
        args=args,
        concepts=concepts,
        forward_scores=forward_scores,
        transform=None,
        input_size=cfg.input_size,
        device=cfg.device,
        log_tag="imagenet",
        saved_mean=saved_mean,
        saved_std=saved_std,
        run_args=dataclasses.asdict(cfg),
    )


def extract_places365_unified_scores(path: Path, args: argparse.Namespace) -> Dict[str, Any]:
    source_run_dir = resolve_source_run_dir(path)
    cfg = load_imagenet_config(source_run_dir, args)
    configure_runtime(cfg)
    concepts = load_concepts(source_run_dir / "concepts.txt", "places365")
    backbone, head = build_model(cfg, n_concepts=len(concepts))
    head.load_state_dict(torch.load(source_run_dir / "concept_head_best.pt", map_location=cfg.device))
    backbone.eval()
    head.eval()

    def forward_scores(images: torch.Tensor) -> torch.Tensor:
        outputs = head(backbone(prepare_images(images, cfg)))
        return outputs["final_logits"]

    saved_mean = saved_std = None
    norm_path = path / "final_layer_normalization.pt"
    if not norm_path.exists():
        norm_path = source_run_dir / "final_layer_normalization.pt"
    if norm_path.exists():
        payload = torch.load(norm_path, map_location="cpu")
        saved_mean = payload.get("mean")
        saved_std = payload.get("std")
    return extract_places365_from_forward(
        args=args,
        concepts=concepts,
        forward_scores=forward_scores,
        transform=None,
        input_size=cfg.input_size,
        device=cfg.device,
        log_tag="places365",
        saved_mean=saved_mean,
        saved_std=saved_std,
        run_args=dataclasses.asdict(cfg),
    )


def extract_partimagenetpp_unified_scores(path: Path, args: argparse.Namespace) -> Dict[str, Any]:
    from data import utils as data_utils

    configure_partimagenetpp_env(args)
    source_run_dir = resolve_source_run_dir(path)
    cfg = load_imagenet_config(source_run_dir, args)
    configure_runtime(cfg)
    concepts = load_concepts(source_run_dir / "concepts.txt", "partimagenetpp")
    backbone, head = build_model(cfg, n_concepts=len(concepts))
    head.load_state_dict(torch.load(source_run_dir / "concept_head_best.pt", map_location=cfg.device))
    backbone.eval()
    head.eval()

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(cfg.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    dataset: Dataset = data_utils.get_data("partimagenetpp_val", preprocess=transform)
    if int(args.max_images) > 0:
        dataset = Subset(dataset, list(range(min(int(args.max_images), len(dataset)))))
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=bool(args.pin_memory),
        **(
            {"prefetch_factor": int(args.prefetch_factor), "persistent_workers": bool(args.persistent_workers)}
            if int(args.num_workers) > 0
            else {}
        ),
    )

    scores: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    start = time.perf_counter()
    seen = 0
    with torch.no_grad():
        for images, targets in loader:
            outputs = head(backbone(prepare_images(images, cfg)))
            scores.append(outputs["final_logits"].detach().cpu().float())
            labels.append(targets.detach().cpu())
            seen += int(images.shape[0])
            if args.log_every > 0 and seen % int(args.log_every) == 0:
                elapsed = time.perf_counter() - start
                print(f"[concept-acc:partimagenetpp:unified] n={seen} ips={seen / max(elapsed, 1e-6):.2f}", flush=True)

    saved_mean = saved_std = None
    norm_path = path / "final_layer_normalization.pt"
    if not norm_path.exists():
        norm_path = source_run_dir / "final_layer_normalization.pt"
    if norm_path.exists():
        payload = torch.load(norm_path, map_location="cpu")
        saved_mean = payload.get("mean")
        saved_std = payload.get("std")
    return {
        "scores": torch.cat(scores, dim=0),
        "train_scores": None,
        "concepts": concepts,
        "indices": list(range(sum(int(chunk.shape[0]) for chunk in scores))),
        "run_args": dataclasses.asdict(cfg),
        "saved_mean": saved_mean,
        "saved_std": saved_std,
        "test_labels": torch.cat(labels, dim=0) if labels else torch.empty(0, dtype=torch.long),
    }


def extract_imagenet_scores(path: Path, model_name: str, args: argparse.Namespace) -> Dict[str, Any]:
    normalized = str(model_name).lower().replace("-", "_")
    if normalized == "salf_cbm":
        return extract_imagenet_salf_scores(path, args)
    if normalized in {"lf_cbm", "vlg_cbm", "cub_cbm"} and (path / "args.txt").exists() and not (path / "config.json").exists():
        return extract_imagenet_legacy_linear_scores(path, normalized, args)
    return extract_imagenet_unified_scores(path, args)


def extract_partimagenetpp_scores(path: Path, model_name: str, args: argparse.Namespace) -> Dict[str, Any]:
    configure_partimagenetpp_env(args)
    normalized = str(model_name).lower().replace("-", "_")
    if (path / "config.json").exists() or (path / "source_run_dir.txt").exists():
        return extract_partimagenetpp_unified_scores(path, args)
    if normalized == "salf_cbm":
        return extract_cub_salf_raw_scores(path, args)
    if normalized in {"savlg_cbm", "sg_cbm", "sgcbm"}:
        return extract_cub_savlg_raw_scores(path, args)
    if normalized in {"vlg_cbm", "cub_cbm"}:
        return extract_cub_vlg_raw_scores(path, args)
    return extract_cub_scores(path, model_name, args)


def extract_places365_scores(path: Path, model_name: str, args: argparse.Namespace) -> Dict[str, Any]:
    normalized = str(model_name).lower().replace("-", "_")
    if normalized in {"lf_cbm", "vlg_cbm", "cub_cbm"} and (path / "args.txt").exists() and not (path / "config.json").exists():
        return extract_places365_legacy_linear_scores(path, normalized, args)
    if normalized not in {"sgcbm", "sg_cbm", "savlg_cbm"}:
        raise RuntimeError(f"Unsupported Places365 concept accuracy checkpoint type: {model_name}")
    return extract_places365_unified_scores(path, args)


def gdino_gt_from_annotations(
    annotations: Sequence[Sequence[dict]],
    concepts: Sequence[str],
    concept_to_idx: Dict[str, int],
    dataset: str,
    threshold: float,
) -> np.ndarray:
    gt = np.zeros((len(annotations), len(concepts)), dtype=np.float32)
    for row, items in enumerate(annotations):
        for item in parse_annotation_items(items):
            label, score = annotation_label_and_score(item)
            idx = concept_to_idx.get(canonicalize_concept(label, dataset))
            if idx is not None and score > threshold:
                gt[row, idx] = 1.0
    return gt


def evaluate_one(
    display_name: str,
    path: Path,
    model_name: str,
    args: argparse.Namespace,
    common_concepts: Optional[Sequence[str]],
) -> Dict[str, Any]:
    if args.dataset == "imagenet":
        extracted = extract_imagenet_scores(path, model_name, args)
    elif args.dataset == "partimagenetpp":
        extracted = extract_partimagenetpp_scores(path, model_name, args)
    elif args.dataset == "places365":
        extracted = extract_places365_scores(path, model_name, args)
    else:
        extracted = extract_cub_scores(path, model_name, args)
    concepts = list(extracted["concepts"])
    wanted = list(common_concepts) if common_concepts is not None else concepts
    scores, train_scores, saved_mean, saved_std = subset_by_concepts(
        extracted["scores"],
        concepts,
        wanted,
        train_scores=extracted.get("train_scores"),
        saved_mean=extracted.get("saved_mean"),
        saved_std=extracted.get("saved_std"),
    )
    scores = apply_normalization(scores, args.normalization, train_scores=train_scores, saved_mean=saved_mean, saved_std=saved_std)
    if "gt" in extracted:
        gt_full = extracted["gt"]
        concept_to_idx = {concept: idx for idx, concept in enumerate(concepts)}
        gt = gt_full[:, [concept_to_idx[concept] for concept in wanted]]
    elif args.gt_source == "cub_parts":
        gt = load_cub_parts_gt(extracted["indices"], wanted, args)
    elif args.dataset == "partimagenetpp":
        if args.gt_source == "partimagenetpp_boxes" or getattr(args, "partimagenetpp_gt_boxes_jsonl", ""):
            gt = load_partimagenetpp_box_presence_gt(extracted["indices"], wanted, args)
        else:
            gt = load_partimagenetpp_presence_gt(extracted["indices"], wanted, args)
    else:
        gt = load_gdino_gt_indexed(cub_annotation_dir(args), extracted["indices"], wanted, args.dataset, args.gdino_threshold)
    if int(args.max_images) > 0:
        keep = min(int(args.max_images), int(scores.shape[0]), int(gt.shape[0]))
        scores = scores[:keep]
        gt = gt[:keep]
    score_np = scores.detach().cpu().numpy().astype(np.float64, copy=False)
    gt_np = gt.astype(np.float64, copy=False)
    flat_gt = gt_np.reshape(-1)
    flat_scores = score_np.reshape(-1)
    threshold_metrics = confusion_metrics(flat_gt, flat_scores >= float(args.threshold))
    output: Dict[str, Any] = {
        "name": display_name,
        "model_name": model_name,
        "load_path": str(path),
        "n_images": int(gt_np.shape[0]),
        "n_concepts": int(gt_np.shape[1]),
        "n_pairs": int(flat_gt.size),
        "gt_positive_rate": float(flat_gt.mean()) if flat_gt.size else 0.0,
        "score_mean": float(np.mean(flat_scores)) if flat_scores.size else float("nan"),
        "score_std": float(np.std(flat_scores)) if flat_scores.size else float("nan"),
        "auroc": safe_roc_auc(flat_gt, flat_scores),
        "ap": safe_average_precision(flat_gt, flat_scores),
        "macro_ap": macro_average_precision(gt_np, score_np),
        "p_at_5": precision_at_k(gt_np, score_np, k=5),
        "threshold": float(args.threshold),
        "threshold_metrics": threshold_metrics,
        "best_f1": best_f1_metrics(flat_gt, flat_scores),
    }
    if args.save_per_concept:
        output["per_concept"] = per_concept_metrics(gt_np, score_np, wanted, float(args.threshold))
    return output


def resolve_common_concepts(paths: Sequence[Path], model_names: Sequence[str], dataset: str) -> List[str]:
    common: Optional[set[str]] = None
    for path, model_name in zip(paths, model_names):
        concept_path = path / "concepts.txt"
        if not concept_path.exists() and (path / "source_run_dir.txt").exists():
            source = Path((path / "source_run_dir.txt").read_text().strip())
            concept_path = source / "concepts.txt"
        concepts = set(load_concepts(concept_path, dataset))
        common = concepts if common is None else common & concepts
    return sorted(common or [])


def main() -> None:
    args = parse_args()
    paths = [Path(path).resolve() for path in args.load_paths]
    explicit_models = args.model_names or [None] * len(paths)
    if len(explicit_models) != len(paths):
        raise ValueError("--model_names must match --load_paths")
    names = args.names or [path.name for path in paths]
    if len(names) != len(paths):
        raise ValueError("--names must match --load_paths")
    model_names = [infer_model_name(path, explicit) for path, explicit in zip(paths, explicit_models)]
    common = resolve_common_concepts(paths, model_names, args.dataset) if args.concept_scope == "common" else None
    if common is not None and not common:
        raise RuntimeError("No common concepts across requested models")

    results: Dict[str, Any] = {}
    for display_name, path, model_name in zip(names, paths, model_names):
        print(f"[concept-acc] evaluating {display_name} model={model_name} path={path}", flush=True)
        results[display_name] = evaluate_one(display_name, path, model_name, args, common)
        brief = results[display_name]
        print(
            f"[concept-acc] {display_name}: auroc={brief['auroc']:.4f} ap={brief['ap']:.4f} "
            f"macro_ap={brief['macro_ap']:.4f} p@5={brief['p_at_5']:.4f} "
            f"best_f1={brief['best_f1']['f1']:.4f} pos_rate={brief['gt_positive_rate']:.4f}",
            flush=True,
        )
    payload = {
        "dataset": args.dataset,
        "gt_source": (
            "partimagenetpp_boxes"
            if args.dataset == "partimagenetpp" and (args.gt_source == "partimagenetpp_boxes" or getattr(args, "partimagenetpp_gt_boxes_jsonl", ""))
            else ("partimagenetpp_manifest" if args.dataset == "partimagenetpp" else args.gt_source)
        ),
        "normalization": args.normalization,
        "cub_score_source": args.cub_score_source,
        "concept_scope": args.concept_scope,
        "common_concepts": common,
        "gdino_threshold": float(args.gdino_threshold),
        "threshold": float(args.threshold),
        "results": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
