from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from gcbm.spatial_targets import rasterize_box_target
from gcbm.target_batches import pad_sparse_targets


def load_concepts(path: str | Path) -> List[str]:
    """Load a medical concept list from txt or JSON concept-bank files."""
    path = Path(path)
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "concepts" in payload:
            payload = payload["concepts"]
        if isinstance(payload, dict):
            concepts: List[str] = []
            for values in payload.values():
                if isinstance(values, list):
                    concepts.extend(str(item).strip() for item in values if str(item).strip())
            return list(dict.fromkeys(concepts))
        if isinstance(payload, list):
            return [str(item).strip() for item in payload if str(item).strip()]
        raise ValueError(f"Unsupported concept JSON schema: {path}")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def canonical_path(path: str | Path) -> str:
    return os.path.normpath(str(path)).replace("\\", "/")


def path_match_keys(path: str | Path) -> List[str]:
    """Build robust keys for matching annotation img_path to dataset image_path."""
    p = canonical_path(path)
    keys = {p}
    parts = [part for part in p.split("/") if part]
    for width in (3, 4, 5, 6):
        if len(parts) >= width:
            keys.add("/".join(parts[-width:]))
    for marker in ("CheXpert-v1.0-small/", "files/"):
        if marker in p:
            keys.add(marker + p.split(marker, 1)[1])
    return list(keys)


def confidence_to_unit(value: float, mode: str = "auto") -> float:
    x = float(value)
    if mode == "clip":
        return float(np.clip(x, 0.0, 1.0))
    if mode == "sigmoid":
        return float(1.0 / (1.0 + np.exp(-x)))
    if 0.0 <= x <= 1.0:
        return x
    return float(1.0 / (1.0 + np.exp(-x)))


def calibrate_presence(scores: np.ndarray, *, pos_thresh: float, neg_thresh: float, mode: str = "linear") -> np.ndarray:
    if mode == "none":
        return np.clip(scores, 0.0, 1.0).astype(np.float32)
    denom = max(float(pos_thresh) - float(neg_thresh), 1e-6)
    return np.clip((scores - float(neg_thresh)) / denom, 0.0, 1.0).astype(np.float32)


def build_dataset_path_index(dataset: Dataset) -> Dict[str, int]:
    if not hasattr(dataset, "get_image_path"):
        raise TypeError("Medical annotation alignment requires dataset.get_image_path(index)")
    buckets: Dict[str, List[int]] = {}
    for index in range(len(dataset)):
        image_path = getattr(dataset, "get_image_path")(index)
        for key in path_match_keys(image_path):
            buckets.setdefault(key, []).append(index)
    return {key: rows[0] for key, rows in buckets.items() if len(rows) == 1}


def _annotation_files(annotation_dir: str | Path) -> List[Tuple[int, Path]]:
    files: List[Tuple[int, Path]] = []
    for path in Path(annotation_dir).glob("*.json"):
        if path.name == "image_annotation_index.json":
            continue
        if path.stem.isdigit():
            files.append((int(path.stem), path))
    return sorted(files, key=lambda item: item[0])


def _read_annotation(path: Path) -> Tuple[str, List[Dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload or not isinstance(payload[0], dict):
        raise ValueError(f"Unsupported annotation schema in {path}")
    return canonical_path(payload[0].get("img_path", "")), [entry for entry in payload[1:] if isinstance(entry, dict)]


def _dataset_image_size(dataset: Dataset, row: int, fallback: Tuple[int, int]) -> Tuple[int, int]:
    if hasattr(dataset, "get_image_size"):
        try:
            return tuple(int(v) for v in getattr(dataset, "get_image_size")(row))  # type: ignore[return-value]
        except Exception:
            pass
    return fallback


def build_medical_targets(
    dataset: Dataset,
    *,
    annotation_dir: str | Path,
    concepts: Sequence[str],
    mask_h: int,
    mask_w: int,
    concept_threshold: float = 0.15,
    neg_threshold: float = 0.02,
    confidence_mode: str = "auto",
    presence_mode: str = "soft",
    target_mode: str = "soft_box",
    box_transform: str = "resize_center_crop",
    input_size: int = 224,
    resize_size: int = 256,
    annotation_ref_size: Tuple[int, int] = (1024, 1024),
    patch_iou_thresh: float = 0.5,
    min_box_confidence: Optional[float] = None,
    allow_index_fallback: bool = False,
) -> Dict[str, Any]:
    """Convert medical grounding JSONs into SG-CBM concept and mask targets."""
    concept_to_idx = {str(concept): idx for idx, concept in enumerate(concepts)}
    n_rows = len(dataset)
    n_concepts = len(concepts)
    presence_scores = np.zeros((n_rows, n_concepts), dtype=np.float32)
    sparse_masks: List[Dict[int, np.ndarray]] = [dict() for _ in range(n_rows)]
    path_index = build_dataset_path_index(dataset)
    min_box_conf = float(concept_threshold if min_box_confidence is None else min_box_confidence)

    matched = 0
    unmatched = 0
    for file_index, path in _annotation_files(annotation_dir):
        try:
            image_path, entries = _read_annotation(path)
        except Exception:
            continue
        row = None
        for key in path_match_keys(image_path):
            if key in path_index:
                row = path_index[key]
                break
        if row is None and allow_index_fallback and 0 <= file_index < n_rows:
            row = file_index
        if row is None:
            unmatched += 1
            continue
        matched += 1
        image_size = _dataset_image_size(dataset, row, annotation_ref_size)
        for entry in entries:
            label = str(entry.get("label", "")).strip()
            concept_idx = concept_to_idx.get(label)
            if concept_idx is None:
                continue
            score = confidence_to_unit(float(entry.get("logit", 0.0)), confidence_mode)
            presence_scores[row, concept_idx] = max(presence_scores[row, concept_idx], score)
            box = entry.get("box")
            if not isinstance(box, (list, tuple)) or len(box) != 4 or score < min_box_conf:
                continue
            mask = rasterize_box_target(
                box,
                image_size=image_size,
                target_mode=target_mode,
                mask_h=int(mask_h),
                mask_w=int(mask_w),
                iou_thresh=float(patch_iou_thresh),
                transform=box_transform,
                input_size=int(input_size),
                resize_size=int(resize_size),
            )
            if mask is None:
                continue
            existing = sparse_masks[row].get(concept_idx)
            if existing is None:
                sparse_masks[row][concept_idx] = mask
            else:
                np.maximum(existing, mask, out=existing)

    if presence_mode == "binary":
        global_targets = (presence_scores >= float(concept_threshold)).astype(np.float32)
    elif presence_mode == "soft":
        global_targets = calibrate_presence(
            presence_scores,
            pos_thresh=float(concept_threshold),
            neg_thresh=float(neg_threshold),
            mode="linear",
        )
    else:
        raise ValueError(f"Unsupported presence_mode: {presence_mode}")

    mask_indices: List[torch.Tensor] = []
    mask_targets: List[torch.Tensor] = []
    for entry in sparse_masks:
        if not entry:
            mask_indices.append(torch.zeros((0,), dtype=torch.long))
            mask_targets.append(torch.zeros((0, int(mask_h), int(mask_w)), dtype=torch.float32))
            continue
        keys = sorted(entry)
        mask_indices.append(torch.tensor(keys, dtype=torch.long))
        mask_targets.append(torch.from_numpy(np.stack([entry[key] for key in keys]).astype(np.float32)))

    idx_pad, mask_pad, valid_pad = pad_sparse_targets(mask_indices, mask_targets, mask_h=int(mask_h), mask_w=int(mask_w))
    return {
        "global_targets": torch.from_numpy(global_targets),
        "presence_scores": torch.from_numpy(presence_scores),
        "mask_indices": idx_pad,
        "mask_targets": mask_pad,
        "mask_valid": valid_pad,
        "matched_annotations": matched,
        "unmatched_annotations": unmatched,
        "num_concepts": n_concepts,
        "num_images": n_rows,
    }


class MedicalTargetDataset(Dataset):
    """Attach precomputed SG-CBM concept targets to a medical image dataset."""

    def __init__(self, base_dataset: Dataset, target_payload: Dict[str, Any]) -> None:
        self.base_dataset = base_dataset
        self.global_targets = target_payload["global_targets"].float()
        self.mask_indices = target_payload["mask_indices"].long()
        self.mask_targets = target_payload["mask_targets"].float()
        self.mask_valid = target_payload["mask_valid"].bool()
        if len(base_dataset) != int(self.global_targets.shape[0]):
            raise ValueError("base_dataset and target payload length mismatch")

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = self.base_dataset[index]
        return {
            **item,
            "global_targets": self.global_targets[index],
            "mask_indices": self.mask_indices[index],
            "mask_targets": self.mask_targets[index],
            "mask_valid": self.mask_valid[index],
        }


def medical_target_collate(batch: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    items = list(batch)
    return {
        "image": torch.stack([item["image"] for item in items]),
        "label": torch.stack([item["target"] for item in items]),
        "global_targets": torch.stack([item["global_targets"] for item in items]),
        "mask_indices": torch.stack([item["mask_indices"] for item in items]),
        "mask_targets": torch.stack([item["mask_targets"] for item in items]),
        "mask_valid": torch.stack([item["mask_valid"] for item in items]),
        "sample_id": [str(item["sample_id"]) for item in items],
        "image_path": [str(item["image_path"]) for item in items],
        "index": torch.tensor([int(item["index"]) for item in items], dtype=torch.long),
    }
