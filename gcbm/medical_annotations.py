from __future__ import annotations

import json
import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
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


def _image_size_from_path(path: str, fallback: Tuple[int, int]) -> Tuple[int, int]:
    try:
        from PIL import Image

        with Image.open(path) as image:
            return image.size
    except Exception:
        return fallback


def _can_use_index_alignment(dataset: Dataset, files: Sequence[Tuple[int, Path]]) -> bool:
    if not files:
        return False
    candidates = [0, 1, 2, len(files) // 2, len(files) - 1]
    seen = set()
    for pos in candidates:
        if pos < 0 or pos >= len(files) or pos in seen:
            continue
        seen.add(pos)
        file_index, path = files[pos]
        if file_index < 0 or file_index >= len(dataset):
            return False
        try:
            image_path, _entries = _read_annotation(path)
        except Exception:
            return False
        dataset_path = getattr(dataset, "get_image_path")(file_index)
        if not (set(path_match_keys(image_path)) & set(path_match_keys(dataset_path))):
            return False
    return True


def _process_indexed_annotation_file_worker(args: Tuple[Any, ...]) -> Tuple[Optional[int], Dict[int, float], Dict[int, np.ndarray], bool]:
    (
        file_item,
        concept_to_idx,
        n_rows,
        confidence_mode,
        min_box_conf,
        target_mode,
        mask_h,
        mask_w,
        patch_iou_thresh,
        box_transform,
        input_size,
        resize_size,
        annotation_ref_size,
    ) = args
    file_index, path = file_item
    if file_index < 0 or file_index >= n_rows:
        return None, {}, {}, False
    try:
        image_path, entries = _read_annotation(Path(path))
    except Exception:
        return None, {}, {}, False
    scores: Dict[int, float] = {}
    masks: Dict[int, np.ndarray] = {}
    image_size = _image_size_from_path(image_path, annotation_ref_size)
    for entry in entries:
        label = str(entry.get("label", "")).strip()
        concept_idx = concept_to_idx.get(label)
        if concept_idx is None:
            continue
        score = confidence_to_unit(float(entry.get("logit", 0.0)), confidence_mode)
        scores[concept_idx] = max(scores.get(concept_idx, 0.0), score)
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
        existing = masks.get(concept_idx)
        if existing is None:
            masks[concept_idx] = mask
        else:
            np.maximum(existing, mask, out=existing)
    return file_index, scores, masks, True


def _process_indexed_annotation_chunk_worker(args: Tuple[Any, ...]) -> List[Tuple[Optional[int], Dict[int, float], Dict[int, np.ndarray], bool]]:
    file_items, *common_args = args
    return [
        _process_indexed_annotation_file_worker((file_item, *common_args))
        for file_item in file_items
    ]


def _process_indexed_presence_chunk_worker(args: Tuple[Any, ...]) -> List[Tuple[Optional[int], Dict[int, float], bool]]:
    file_items, concept_to_idx, n_rows, confidence_mode = args
    results: List[Tuple[Optional[int], Dict[int, float], bool]] = []
    for file_index, path in file_items:
        if file_index < 0 or file_index >= n_rows:
            results.append((None, {}, False))
            continue
        try:
            _image_path, entries = _read_annotation(Path(path))
        except Exception:
            results.append((None, {}, False))
            continue
        scores: Dict[int, float] = {}
        for entry in entries:
            label = str(entry.get("label", "")).strip()
            concept_idx = concept_to_idx.get(label)
            if concept_idx is None:
                continue
            score = confidence_to_unit(float(entry.get("logit", 0.0)), confidence_mode)
            scores[concept_idx] = max(scores.get(concept_idx, 0.0), score)
        results.append((file_index, scores, True))
    return results


def _chunks(items: Sequence[Any], size: int) -> List[List[Any]]:
    return [list(items[start : start + size]) for start in range(0, len(items), size)]


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
    num_workers: int = 0,
    progress_every: int = 10000,
) -> Dict[str, Any]:
    """Convert medical grounding JSONs into SG-CBM concept and mask targets."""
    concept_to_idx = {str(concept): idx for idx, concept in enumerate(concepts)}
    n_rows = len(dataset)
    n_concepts = len(concepts)
    presence_scores = np.zeros((n_rows, n_concepts), dtype=np.float32)
    sparse_masks: List[Dict[int, np.ndarray]] = [dict() for _ in range(n_rows)]
    path_index: Optional[Dict[str, int]] = None
    min_box_conf = float(concept_threshold if min_box_confidence is None else min_box_confidence)
    files = _annotation_files(annotation_dir)
    use_index_alignment = _can_use_index_alignment(dataset, files)
    if not use_index_alignment:
        path_index = build_dataset_path_index(dataset)

    def process_file(file_item: Tuple[int, Path]) -> Tuple[Optional[int], Dict[int, float], Dict[int, np.ndarray], bool]:
        file_index, path = file_item
        try:
            image_path, entries = _read_annotation(path)
        except Exception:
            return None, {}, {}, False
        row = None
        if use_index_alignment and 0 <= file_index < n_rows:
            row = file_index
        else:
            assert path_index is not None
            for key in path_match_keys(image_path):
                if key in path_index:
                    row = path_index[key]
                    break
        if row is None and allow_index_fallback and 0 <= file_index < n_rows:
            row = file_index
        if row is None:
            return None, {}, {}, False

        scores: Dict[int, float] = {}
        masks: Dict[int, np.ndarray] = {}
        image_size = _dataset_image_size(dataset, row, annotation_ref_size)
        for entry in entries:
            label = str(entry.get("label", "")).strip()
            concept_idx = concept_to_idx.get(label)
            if concept_idx is None:
                continue
            score = confidence_to_unit(float(entry.get("logit", 0.0)), confidence_mode)
            scores[concept_idx] = max(scores.get(concept_idx, 0.0), score)
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
            existing = masks.get(concept_idx)
            if existing is None:
                masks[concept_idx] = mask
            else:
                np.maximum(existing, mask, out=existing)
        return row, scores, masks, True

    def merge_result(row: Optional[int], scores: Dict[int, float], masks: Dict[int, np.ndarray], ok: bool) -> None:
        nonlocal matched, unmatched
        if not ok:
            unmatched += 1
            return
        if row is None:
            unmatched += 1
            return
        matched += 1
        for concept_idx, score in scores.items():
            presence_scores[row, concept_idx] = max(presence_scores[row, concept_idx], score)
        row_masks = sparse_masks[row]
        for concept_idx, mask in masks.items():
            existing = row_masks.get(concept_idx)
            if existing is None:
                row_masks[concept_idx] = mask
            else:
                np.maximum(existing, mask, out=existing)

    matched = 0
    unmatched = 0
    started = time.perf_counter()
    workers = max(int(num_workers), 0)
    print(
        f"[medical targets] building {len(files)} annotation targets with workers={workers} index_alignment={use_index_alignment}",
        flush=True,
    )
    if workers <= 1:
        next_report = int(progress_every)
        for processed, file_item in enumerate(files, start=1):
            merge_result(*process_file(file_item))
            if progress_every > 0 and processed >= next_report:
                elapsed = time.perf_counter() - started
                print(f"[medical targets] processed={processed}/{len(files)} matched={matched} unmatched={unmatched} elapsed={elapsed:.1f}s", flush=True)
                next_report += int(progress_every)
    elif use_index_alignment:
        common_args = (
            concept_to_idx,
            n_rows,
            confidence_mode,
            min_box_conf,
            target_mode,
            int(mask_h),
            int(mask_w),
            float(patch_iou_thresh),
            box_transform,
            int(input_size),
            int(resize_size),
            annotation_ref_size,
        )
        chunk_size = 256
        worker_args = [
            (
                [(file_index, str(path)) for file_index, path in chunk],
                *common_args,
            )
            for chunk in _chunks(files, chunk_size)
        ]
        with ProcessPoolExecutor(max_workers=workers) as executor:
            arg_iter = iter(worker_args)
            pending = set()
            for _ in range(min(workers * 4, len(worker_args))):
                pending.add(executor.submit(_process_indexed_annotation_chunk_worker, next(arg_iter)))
            processed = 0
            next_report = int(progress_every)
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    for result in future.result():
                        processed += 1
                        merge_result(*result)
                    try:
                        pending.add(executor.submit(_process_indexed_annotation_chunk_worker, next(arg_iter)))
                    except StopIteration:
                        pass
                    if progress_every > 0 and processed >= next_report:
                        elapsed = time.perf_counter() - started
                        print(f"[medical targets] processed={processed}/{len(files)} matched={matched} unmatched={unmatched} elapsed={elapsed:.1f}s", flush=True)
                        next_report += int(progress_every)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            file_iter = iter(files)
            pending = set()
            for _ in range(min(workers * 4, len(files))):
                pending.add(executor.submit(process_file, next(file_iter)))
            processed = 0
            next_report = int(progress_every)
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    processed += 1
                    merge_result(*future.result())
                    try:
                        pending.add(executor.submit(process_file, next(file_iter)))
                    except StopIteration:
                        pass
                    if progress_every > 0 and processed >= next_report:
                        elapsed = time.perf_counter() - started
                        print(f"[medical targets] processed={processed}/{len(files)} matched={matched} unmatched={unmatched} elapsed={elapsed:.1f}s", flush=True)
                        next_report += int(progress_every)
    elapsed = time.perf_counter() - started
    print(f"[medical targets] done files={len(files)} matched={matched} unmatched={unmatched} elapsed={elapsed:.1f}s", flush=True)

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

    return {
        "global_targets": torch.from_numpy(global_targets),
        "presence_scores": torch.from_numpy(presence_scores),
        "mask_indices": mask_indices,
        "mask_targets": mask_targets,
        "matched_annotations": matched,
        "unmatched_annotations": unmatched,
        "num_concepts": n_concepts,
        "num_images": n_rows,
    }


def build_medical_presence_targets(
    dataset: Dataset,
    *,
    annotation_dir: str | Path,
    concepts: Sequence[str],
    concept_threshold: float = 0.15,
    confidence_mode: str = "auto",
    allow_index_fallback: bool = False,
    num_workers: int = 0,
    progress_every: int = 50000,
) -> Dict[str, Any]:
    """Build binary concept-presence targets without rasterizing spatial masks.

    This mirrors the VLG-CBM concept-cache path and is used to select the same
    frequency-filtered concept bank before SG-CBM lazy spatial training.
    """
    concept_to_idx = {str(concept): idx for idx, concept in enumerate(concepts)}
    n_rows = len(dataset)
    n_concepts = len(concepts)
    presence_scores = np.zeros((n_rows, n_concepts), dtype=np.float32)
    files = _annotation_files(annotation_dir)
    use_index_alignment = _can_use_index_alignment(dataset, files)
    path_index: Optional[Dict[str, int]] = None if use_index_alignment else build_dataset_path_index(dataset)

    matched = 0
    unmatched = 0
    started = time.perf_counter()
    print(
        f"[medical targets] scanning concept presence files={len(files)} workers={max(int(num_workers), 0)} index_alignment={use_index_alignment}",
        flush=True,
    )

    def merge_presence(row: Optional[int], scores: Dict[int, float], ok: bool) -> None:
        nonlocal matched, unmatched
        if not ok or row is None:
            unmatched += 1
            return
        matched += 1
        for concept_idx, score in scores.items():
            presence_scores[row, concept_idx] = max(presence_scores[row, concept_idx], score)

    workers = max(int(num_workers), 0)
    processed = 0
    next_report = int(progress_every)
    if workers > 1 and use_index_alignment:
        chunk_size = 256
        worker_args = [
            (
                [(file_index, str(path)) for file_index, path in chunk],
                concept_to_idx,
                n_rows,
                confidence_mode,
            )
            for chunk in _chunks(files, chunk_size)
        ]
        with ProcessPoolExecutor(max_workers=workers) as executor:
            arg_iter = iter(worker_args)
            pending = set()
            for _ in range(min(workers * 4, len(worker_args))):
                pending.add(executor.submit(_process_indexed_presence_chunk_worker, next(arg_iter)))
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    for row, scores, ok in future.result():
                        processed += 1
                        merge_presence(row, scores, ok)
                    try:
                        pending.add(executor.submit(_process_indexed_presence_chunk_worker, next(arg_iter)))
                    except StopIteration:
                        pass
                    if progress_every > 0 and processed >= next_report:
                        elapsed = time.perf_counter() - started
                        print(
                            f"[medical targets] presence processed={processed}/{len(files)} matched={matched} unmatched={unmatched} elapsed={elapsed:.1f}s",
                            flush=True,
                        )
                        next_report += int(progress_every)
    else:
        for processed, (file_index, path) in enumerate(files, start=1):
            try:
                image_path, entries = _read_annotation(path)
            except Exception:
                unmatched += 1
                continue
            row = None
            if use_index_alignment and 0 <= file_index < n_rows:
                row = file_index
            elif path_index is not None:
                for key in path_match_keys(image_path):
                    if key in path_index:
                        row = path_index[key]
                        break
            if row is None and allow_index_fallback and 0 <= file_index < n_rows:
                row = file_index
            if row is None:
                unmatched += 1
                continue
            scores: Dict[int, float] = {}
            for entry in entries:
                label = str(entry.get("label", "")).strip()
                concept_idx = concept_to_idx.get(label)
                if concept_idx is None:
                    continue
                score = confidence_to_unit(float(entry.get("logit", 0.0)), confidence_mode)
                scores[concept_idx] = max(scores.get(concept_idx, 0.0), score)
            merge_presence(row, scores, True)
            if progress_every > 0 and processed >= next_report:
                elapsed = time.perf_counter() - started
                print(
                    f"[medical targets] presence processed={processed}/{len(files)} matched={matched} unmatched={unmatched} elapsed={elapsed:.1f}s",
                    flush=True,
                )
                next_report += int(progress_every)

    global_targets = (presence_scores >= float(concept_threshold)).astype(np.float32)
    elapsed = time.perf_counter() - started
    print(f"[medical targets] presence done matched={matched} unmatched={unmatched} elapsed={elapsed:.1f}s", flush=True)
    return {
        "global_targets": torch.from_numpy(global_targets),
        "presence_scores": torch.from_numpy(presence_scores),
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
        self.mask_indices, self.mask_targets = self._compact_masks(target_payload)
        if len(base_dataset) != int(self.global_targets.shape[0]):
            raise ValueError("base_dataset and target payload length mismatch")

    @staticmethod
    def _compact_masks(target_payload: Dict[str, Any]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        indices = target_payload["mask_indices"]
        masks = target_payload["mask_targets"]
        valid = target_payload.get("mask_valid")
        if isinstance(indices, torch.Tensor):
            if indices.ndim != 2 or not isinstance(masks, torch.Tensor) or valid is None:
                raise ValueError("Padded medical target payload must include 2D mask_indices, mask_targets, and mask_valid")
            valid = valid.bool()
            compact_indices: List[torch.Tensor] = []
            compact_masks: List[torch.Tensor] = []
            for row in range(indices.shape[0]):
                row_valid = valid[row]
                compact_indices.append(indices[row][row_valid].long().cpu())
                compact_masks.append(masks[row][row_valid].float().cpu())
            return compact_indices, compact_masks
        return [item.long().cpu() for item in indices], [item.float().cpu() for item in masks]

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = self.base_dataset[index]
        return {
            **item,
            "global_targets": self.global_targets[index],
            "mask_indices": self.mask_indices[index],
            "mask_targets": self.mask_targets[index],
        }


def medical_target_collate(batch: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    items = list(batch)
    mask_h = int(items[0]["mask_targets"].shape[-2])
    mask_w = int(items[0]["mask_targets"].shape[-1])
    mask_indices, mask_targets, mask_valid = pad_sparse_targets(
        [item["mask_indices"] for item in items],
        [item["mask_targets"] for item in items],
        mask_h=mask_h,
        mask_w=mask_w,
    )
    return {
        "image": torch.stack([item["image"] for item in items]),
        "label": torch.stack([item["target"] for item in items]),
        "global_targets": torch.stack([item["global_targets"] for item in items]),
        "mask_indices": mask_indices,
        "mask_targets": mask_targets,
        "mask_valid": mask_valid,
        "sample_id": [str(item["sample_id"]) for item in items],
        "image_path": [str(item["image_path"]) for item in items],
        "index": torch.tensor([int(item["index"]) for item in items], dtype=torch.long),
    }


class LazyMedicalTargetDataset(Dataset):
    """Attach SG-CBM targets by reading one medical annotation JSON per sample.

    This is the full-scale path for large CheXpert/MIMIC runs. It keeps only
    dataset metadata in memory and rasterizes masks inside DataLoader workers,
    so full spatial targets are never resident for the whole training split.
    """

    def __init__(
        self,
        base_dataset: Dataset,
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
    ) -> None:
        self.base_dataset = base_dataset
        self.annotation_dir = Path(annotation_dir)
        self.concepts = list(concepts)
        self.concept_to_idx = {str(concept): idx for idx, concept in enumerate(concepts)}
        self.mask_h = int(mask_h)
        self.mask_w = int(mask_w)
        self.concept_threshold = float(concept_threshold)
        self.neg_threshold = float(neg_threshold)
        self.confidence_mode = str(confidence_mode)
        self.presence_mode = str(presence_mode)
        self.target_mode = str(target_mode)
        self.box_transform = str(box_transform)
        self.input_size = int(input_size)
        self.resize_size = int(resize_size)
        self.annotation_ref_size = annotation_ref_size
        self.patch_iou_thresh = float(patch_iou_thresh)
        self.min_box_conf = float(concept_threshold if min_box_confidence is None else min_box_confidence)
        self.allow_index_fallback = bool(allow_index_fallback)
        self.files = _annotation_files(self.annotation_dir)
        self.index_aligned = _can_use_index_alignment(base_dataset, self.files)
        self.file_by_index = {file_index: path for file_index, path in self.files}
        self.file_by_path_key: Optional[Dict[str, Path]] = None
        if not self.index_aligned:
            self.file_by_path_key = self._build_file_by_path_key()
        print(
            f"[medical targets] lazy annotation dataset files={len(self.files)} index_alignment={self.index_aligned}",
            flush=True,
        )

    def _build_file_by_path_key(self) -> Dict[str, Path]:
        mapping: Dict[str, Path] = {}
        for _file_index, path in self.files:
            try:
                image_path, _entries = _read_annotation(path)
            except Exception:
                continue
            for key in path_match_keys(image_path):
                mapping.setdefault(key, path)
        return mapping

    def __len__(self) -> int:
        return len(self.base_dataset)

    def _annotation_path(self, index: int, image_path: str) -> Optional[Path]:
        if self.index_aligned:
            path = self.file_by_index.get(int(index))
            if path is not None:
                return path
        if self.file_by_path_key is None:
            return None
        for key in path_match_keys(image_path):
            path = self.file_by_path_key.get(key)
            if path is not None:
                return path
        if self.allow_index_fallback:
            return self.file_by_index.get(int(index))
        return None

    def _empty_targets(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        global_targets = torch.zeros((len(self.concepts),), dtype=torch.float32)
        mask_indices = torch.zeros((0,), dtype=torch.long)
        mask_targets = torch.zeros((0, self.mask_h, self.mask_w), dtype=torch.float32)
        return global_targets, mask_indices, mask_targets

    def _build_targets_for_item(self, index: int, image_path: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        path = self._annotation_path(index, image_path)
        if path is None:
            return self._empty_targets()
        try:
            _annotation_image_path, entries = _read_annotation(path)
        except Exception:
            return self._empty_targets()

        presence_scores = np.zeros((len(self.concepts),), dtype=np.float32)
        sparse_masks: Dict[int, np.ndarray] = {}
        image_size = _dataset_image_size(self.base_dataset, index, self.annotation_ref_size)
        for entry in entries:
            label = str(entry.get("label", "")).strip()
            concept_idx = self.concept_to_idx.get(label)
            if concept_idx is None:
                continue
            score = confidence_to_unit(float(entry.get("logit", 0.0)), self.confidence_mode)
            presence_scores[concept_idx] = max(presence_scores[concept_idx], score)
            box = entry.get("box")
            if not isinstance(box, (list, tuple)) or len(box) != 4 or score < self.min_box_conf:
                continue
            mask = rasterize_box_target(
                box,
                image_size=image_size,
                target_mode=self.target_mode,
                mask_h=self.mask_h,
                mask_w=self.mask_w,
                iou_thresh=self.patch_iou_thresh,
                transform=self.box_transform,
                input_size=self.input_size,
                resize_size=self.resize_size,
            )
            if mask is None:
                continue
            existing = sparse_masks.get(concept_idx)
            if existing is None:
                sparse_masks[concept_idx] = mask
            else:
                np.maximum(existing, mask, out=existing)

        if self.presence_mode == "binary":
            global_targets_np = (presence_scores >= self.concept_threshold).astype(np.float32)
        elif self.presence_mode == "soft":
            global_targets_np = calibrate_presence(
                presence_scores,
                pos_thresh=self.concept_threshold,
                neg_thresh=self.neg_threshold,
                mode="linear",
            )
        else:
            raise ValueError(f"Unsupported presence_mode: {self.presence_mode}")

        if not sparse_masks:
            mask_indices = torch.zeros((0,), dtype=torch.long)
            mask_targets = torch.zeros((0, self.mask_h, self.mask_w), dtype=torch.float32)
        else:
            keys = sorted(sparse_masks)
            mask_indices = torch.tensor(keys, dtype=torch.long)
            mask_targets = torch.from_numpy(np.stack([sparse_masks[key] for key in keys]).astype(np.float32))
        return torch.from_numpy(global_targets_np), mask_indices, mask_targets

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = self.base_dataset[index]
        global_targets, mask_indices, mask_targets = self._build_targets_for_item(index, str(item["image_path"]))
        return {
            **item,
            "global_targets": global_targets,
            "mask_indices": mask_indices,
            "mask_targets": mask_targets,
        }
