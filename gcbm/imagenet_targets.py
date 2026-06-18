from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

from gcbm.imagenet_config import Config
from gcbm.spatial_targets import (
    PREPROCESS_RESIZE_SIZE,
    normalize_box,
    rasterize_box_iou as _shared_rasterize_box_iou,
    rasterize_box_soft_occupancy as _shared_rasterize_box_soft_occupancy,
    rasterize_box_target as _shared_rasterize_box_target,
    resize_short_edge_size,
    transform_box_for_resize_center_crop,
)
from gcbm.target_batches import batch_targets_to_device as _batch_targets_to_device
from gcbm.target_batches import pad_sparse_targets


IMAGENET_LABEL_ALIASES = {
    "website": "a web page",
    "beer bottle": "a bottle with a long neck",
    "wine bottle": "a bottle with a long neck",
    "soda bottle": "a glass or plastic bottle",
    "ski": "a pair of skis",
    "metal nail": "nails",
}


def format_concept(text: str) -> str:
    text = text.lower()
    for ch in "-,.()":
        text = text.replace(ch, " ")
    if text.startswith("a "):
        text = text[2:]
    elif text.startswith("an "):
        text = text[3:]
    return " ".join(text.split())


def canonicalize_concept_label(text: str) -> str:
    normalized = format_concept(text)
    return format_concept(IMAGENET_LABEL_ALIASES.get(normalized, normalized))


def load_concepts(path: str) -> List[str]:
    with open(path, "r") as handle:
        concepts = [canonicalize_concept_label(line.strip()) for line in handle if line.strip()]
    return list(dict.fromkeys(concepts))


def load_run_concepts(cfg: Config) -> List[str]:
    concepts = load_concepts(cfg.concept_file)
    if cfg.mode == "precompute_targets" or not cfg.precomputed_target_dir:
        return concepts
    precomputed_concepts = Path(cfg.precomputed_target_dir) / "concepts.txt"
    if not precomputed_concepts.exists():
        return concepts
    target_concepts = load_concepts(str(precomputed_concepts))
    if target_concepts != concepts:
        print(
            f"[concept_filter] using {len(target_concepts)} concepts from {precomputed_concepts} "
            f"instead of {len(concepts)} concepts from {cfg.concept_file}",
            flush=True,
        )
        return target_concepts
    return concepts


class PrecomputedTargetStore:
    """Memory-mapped target cache for CBL training.

    This contains supervision only: dense global concept labels plus sparse
    per-concept spatial masks. It does not contain ResNet activations or SAVLG
    concept features.
    """

    def __init__(self, root: Path) -> None:
        metadata = json.loads((root / "metadata.json").read_text())
        self.root = root
        self.metadata = metadata
        self.n_examples = int(metadata["n_examples"])
        self.n_concepts = int(metadata["n_concepts"])
        self.mask_h = int(metadata["mask_h"])
        self.mask_w = int(metadata["mask_w"])
        self.global_targets = np.load(root / "global_targets.npy", mmap_mode="r")
        self.offsets = np.load(root / "offsets.npy", mmap_mode="r")
        self.concept_ids = np.load(root / "concept_ids.npy", mmap_mode="r")
        self.mask_targets = np.load(root / "mask_targets.npy", mmap_mode="r")
        self.keep_indices: Optional[np.ndarray] = None
        self.concept_remap: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return self.n_examples

    def validate_target_geometry(self, cfg: Config) -> None:
        # Old precompute caches were rasterized in original-image coordinates.
        # Refuse them here so spatial training cannot silently mix frames.
        expected_frame = "resize_short_edge_then_center_crop"
        frame = self.metadata.get("target_coordinate_frame")
        if frame != expected_frame:
            raise ValueError(
                f"Precomputed targets at {self.root} use target_coordinate_frame={frame!r}; "
                f"expected {expected_frame!r}. Regenerate targets with crop-space bbox handling."
            )
        input_size = self.metadata.get("input_size")
        if input_size is None or int(input_size) != int(cfg.input_size):
            raise ValueError(
                f"Precomputed targets at {self.root} use input_size={input_size}; "
                f"expected {cfg.input_size}"
            )
        resize_size = self.metadata.get("preprocess_resize_size")
        if resize_size is None or int(resize_size) != int(PREPROCESS_RESIZE_SIZE):
            raise ValueError(
                f"Precomputed targets at {self.root} use preprocess_resize_size={resize_size}; "
                f"expected {PREPROCESS_RESIZE_SIZE}"
            )

    def set_concept_filter(self, keep_indices: Sequence[int]) -> None:
        # Concept filtering changes the active concept set after precompute;
        # keep a vectorized remap so sparse mask concept ids stay aligned.
        keep = np.asarray(list(keep_indices), dtype=np.int64)
        if keep.ndim != 1:
            raise ValueError("Concept keep indices must be a 1D sequence")
        if keep.size == 0:
            raise ValueError("Concept count filtering removed all concepts")
        if int(keep.min()) < 0 or int(keep.max()) >= self.n_concepts:
            raise ValueError("Concept keep indices are out of bounds for precomputed targets")
        remap = np.full((self.n_concepts,), -1, dtype=np.int64)
        remap[keep] = np.arange(keep.size, dtype=np.int64)
        self.keep_indices = keep
        self.concept_remap = remap

    def get(self, index: int) -> Dict[str, torch.Tensor]:
        # offsets implements a CSR-style layout. Image i owns the sparse masks
        # in concept_ids[offsets[i]:offsets[i + 1]] and the matching rows of
        # mask_targets.
        start = int(self.offsets[index])
        end = int(self.offsets[index + 1])
        global_row = np.asarray(self.global_targets[index], dtype=np.float32)
        if self.keep_indices is not None:
            global_row = global_row[self.keep_indices]
        global_target = torch.from_numpy(np.ascontiguousarray(global_row).copy())
        if end <= start:
            mask_indices = torch.zeros((0,), dtype=torch.long)
            mask_targets = torch.zeros((0, self.mask_h, self.mask_w), dtype=torch.float32)
        else:
            concept_ids = np.asarray(self.concept_ids[start:end], dtype=np.int64)
            masks = np.asarray(self.mask_targets[start:end], dtype=np.float32)
            if self.concept_remap is not None:
                mapped = self.concept_remap[concept_ids]
                valid = mapped >= 0
                concept_ids = mapped[valid]
                masks = masks[valid]
            mask_indices = torch.from_numpy(np.ascontiguousarray(concept_ids).copy())
            mask_targets = torch.from_numpy(np.ascontiguousarray(masks).copy())
        return {
            "global_target": global_target,
            "mask_indices": mask_indices,
            "mask_targets": mask_targets,
        }

    def get_global(self, index: int) -> Dict[str, torch.Tensor]:
        global_row = np.asarray(self.global_targets[index], dtype=np.float32)
        if self.keep_indices is not None:
            global_row = global_row[self.keep_indices]
        return {"global_target": torch.from_numpy(np.ascontiguousarray(global_row).copy())}


def transform_box_for_model_input(
    box: Sequence[float],
    image_size: Tuple[int, int],
    input_size: Optional[int],
    resize_size: int = PREPROCESS_RESIZE_SIZE,
) -> Optional[Tuple[float, float, float, float]]:
    return transform_box_for_resize_center_crop(
        box,
        image_size=image_size,
        input_size=input_size,
        resize_size=resize_size,
    )


def rasterize_box_iou(
    box: Sequence[float],
    image_size: Tuple[int, int],
    input_size: int,
    mask_h: int,
    mask_w: int,
    iou_thresh: float,
) -> Optional[np.ndarray]:
    return _shared_rasterize_box_iou(
        box,
        image_size=image_size,
        mask_h=mask_h,
        mask_w=mask_w,
        iou_thresh=iou_thresh,
        transform="resize_center_crop",
        input_size=input_size,
        resize_size=PREPROCESS_RESIZE_SIZE,
    )


def rasterize_box_soft_occupancy(
    box: Sequence[float],
    image_size: Tuple[int, int],
    input_size: int,
    mask_h: int,
    mask_w: int,
) -> Optional[np.ndarray]:
    return _shared_rasterize_box_soft_occupancy(
        box,
        image_size=image_size,
        mask_h=mask_h,
        mask_w=mask_w,
        transform="resize_center_crop",
        input_size=input_size,
        resize_size=PREPROCESS_RESIZE_SIZE,
    )


def rasterize_box_target(
    box: Sequence[float],
    image_size: Tuple[int, int],
    cfg: Config,
) -> Optional[np.ndarray]:
    return _shared_rasterize_box_target(
        box,
        image_size=image_size,
        target_mode=cfg.spatial_target_mode,
        mask_h=cfg.mask_h,
        mask_w=cfg.mask_w,
        iou_thresh=cfg.patch_iou_thresh,
        transform="resize_center_crop",
        input_size=cfg.input_size,
        resize_size=PREPROCESS_RESIZE_SIZE,
    )


def annotation_entries(sample_annotations: Sequence[Dict[str, Any]]) -> Sequence[Dict[str, Any]]:
    # Generated annotation files are list-shaped with a metadata dict first;
    # tolerate pure concept lists as well.
    if not isinstance(sample_annotations, list):
        return []
    if not sample_annotations:
        return []
    first = sample_annotations[0]
    if isinstance(first, dict) and ("label" in first or "box" in first):
        return sample_annotations
    return sample_annotations[1:]


def build_gdino_targets(
    annotations: Sequence[List[Dict[str, Any]]],
    image_sizes: Sequence[Tuple[int, int]],
    concept_to_idx: Dict[str, int],
    n_concepts: int,
    cfg: Config,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a batch of GDINO targets on the fly.

    This mirrors build_gdino_target_sample, which is used by precompute. Keeping
    both paths on the same rasterization function prevents train-time targets
    from disagreeing with cached targets.
    """
    global_targets = torch.zeros((len(annotations), n_concepts), dtype=torch.float32)
    sparse_indices: List[torch.Tensor] = []
    sparse_masks: List[torch.Tensor] = []
    for sample_idx, sample_annotations in enumerate(annotations):
        scores = np.zeros((n_concepts,), dtype=np.float32)
        mask_dict: Dict[int, np.ndarray] = {}
        entries = annotation_entries(sample_annotations)
        for ann in entries:
            if not isinstance(ann, dict):
                continue
            label = ann.get("label")
            if not isinstance(label, str):
                continue
            concept_idx = concept_to_idx.get(canonicalize_concept_label(label))
            if concept_idx is None:
                continue
            score = float(ann.get("logit", 0.0))
            if score > scores[concept_idx]:
                scores[concept_idx] = score
            if score < cfg.concept_threshold:
                continue
            mask = rasterize_box_target(
                ann.get("box"),
                image_size=image_sizes[sample_idx],
                cfg=cfg,
            )
            if mask is None:
                continue
            existing = mask_dict.get(concept_idx)
            if existing is None:
                mask_dict[concept_idx] = mask
            else:
                np.maximum(existing, mask, out=existing)
        global_targets[sample_idx] = torch.from_numpy((scores > cfg.concept_threshold).astype(np.float32))
        if mask_dict:
            keys = sorted(mask_dict.keys())
            sparse_indices.append(torch.tensor(keys, dtype=torch.long))
            sparse_masks.append(torch.from_numpy(np.stack([mask_dict[k] for k in keys], axis=0)))
        else:
            sparse_indices.append(torch.zeros((0,), dtype=torch.long))
            sparse_masks.append(torch.zeros((0, cfg.mask_h, cfg.mask_w), dtype=torch.float32))

    idx_pad, mask_pad, valid = pad_sparse_targets(
        sparse_indices,
        sparse_masks,
        mask_h=cfg.mask_h,
        mask_w=cfg.mask_w,
    )
    return (
        global_targets.to(device, non_blocking=True),
        idx_pad.to(device, non_blocking=True),
        mask_pad.to(device, non_blocking=True),
        valid.to(device, non_blocking=True),
    )


def build_gdino_global_targets(
    annotations: Sequence[List[Dict[str, Any]]],
    concept_to_idx: Dict[str, int],
    n_concepts: int,
    cfg: Config,
    device: str,
) -> torch.Tensor:
    global_targets = torch.zeros((len(annotations), n_concepts), dtype=torch.float32)
    for sample_idx, sample_annotations in enumerate(annotations):
        scores = np.zeros((n_concepts,), dtype=np.float32)
        for ann in annotation_entries(sample_annotations):
            if not isinstance(ann, dict):
                continue
            label = ann.get("label")
            if not isinstance(label, str):
                continue
            concept_idx = concept_to_idx.get(canonicalize_concept_label(label))
            if concept_idx is None:
                continue
            score = float(ann.get("logit", 0.0))
            if score > scores[concept_idx]:
                scores[concept_idx] = score
        global_targets[sample_idx] = torch.from_numpy((scores > cfg.concept_threshold).astype(np.float32))
    return global_targets.to(device, non_blocking=True)


def batch_targets_to_device(batch: Dict[str, Any], cfg: Config) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _batch_targets_to_device(batch, cfg.device)


def build_gdino_target_sample(
    sample_annotations: Sequence[Dict[str, Any]],
    image_size: Tuple[int, int],
    concept_to_idx: Dict[str, int],
    n_concepts: int,
    cfg: Config,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert one annotation JSON payload into precompute-ready arrays.

    global_target is dense [n_concepts]. concept ids and masks are sparse: only
    concepts with a valid crop-space box above threshold are stored.
    """
    scores = np.zeros((n_concepts,), dtype=np.float32)
    mask_dict: Dict[int, np.ndarray] = {}
    entries = annotation_entries(sample_annotations)
    for ann in entries:
        if not isinstance(ann, dict):
            continue
        label = ann.get("label")
        if not isinstance(label, str):
            continue
        concept_idx = concept_to_idx.get(canonicalize_concept_label(label))
        if concept_idx is None:
            continue
        score = float(ann.get("logit", 0.0))
        if score > scores[concept_idx]:
            scores[concept_idx] = score
        if score < cfg.concept_threshold:
            continue
        mask = rasterize_box_target(
            ann.get("box"),
            image_size=image_size,
            cfg=cfg,
        )
        if mask is None:
            continue
        existing = mask_dict.get(concept_idx)
        if existing is None:
            mask_dict[concept_idx] = mask
        else:
            np.maximum(existing, mask, out=existing)
    global_target = (scores > cfg.concept_threshold).astype(np.uint8)
    if not mask_dict:
        return global_target, np.zeros((0,), dtype=np.int32), np.zeros((0, cfg.mask_h, cfg.mask_w), dtype=np.float32)
    keys = np.asarray(sorted(mask_dict.keys()), dtype=np.int32)
    masks = np.stack([mask_dict[int(key)] for key in keys], axis=0).astype(np.float32, copy=False)
    return global_target, keys, masks


def get_image_size(path: str, input_size: int, min_image_bytes: int) -> Tuple[int, int]:
    try:
        if os.path.getsize(path) < min_image_bytes:
            raise OSError(f"tiny file: {path}")
        with Image.open(path) as image:
            width, height = image.size
        return int(width), int(height)
    except (FileNotFoundError, OSError, UnidentifiedImageError):
        return int(input_size), int(input_size)


def precompute_target_store(
    dataset: SafeImageFolderWithAnnotations,
    output_root: Path,
    cfg: Config,
) -> Dict[str, Any]:
    """Write static GDINO supervision for one split.

    The two-pass design avoids retaining all masks in RAM. Pass 1 writes dense
    global targets and counts sparse masks per image. After counts are known,
    we allocate exact-size memmaps and pass 2 fills concept_ids/mask_targets.
    """
    split_dir = output_root / dataset.split
    split_dir.mkdir(parents=True, exist_ok=True)
    total_examples = len(dataset)
    n_concepts = len(dataset.concepts)
    global_targets_path = split_dir / "global_targets.npy"
    offsets_path = split_dir / "offsets.npy"
    concept_ids_path = split_dir / "concept_ids.npy"
    mask_targets_path = split_dir / "mask_targets.npy"

    counts = np.zeros((total_examples,), dtype=np.int32)
    global_targets = np.lib.format.open_memmap(
        global_targets_path,
        mode="w+",
        dtype=np.uint8,
        shape=(total_examples, n_concepts),
    )
    total_entries = 0
    start_time = time.perf_counter()
    for sample_index in range(total_examples):
        path, _ = dataset.dataset.samples[sample_index]
        annotation_index = dataset.annotation_index_for_row(sample_index)
        image_size = get_image_size(path, dataset.input_size, dataset.min_image_bytes)
        annotations = dataset._load_annotation(annotation_index)
        global_target, concept_ids, _ = build_gdino_target_sample(
            annotations,
            image_size,
            dataset.concept_to_idx,
            n_concepts,
            cfg,
        )
        global_targets[sample_index] = global_target
        counts[sample_index] = int(concept_ids.shape[0])
        total_entries += int(concept_ids.shape[0])
        if (sample_index + 1) % 1000 == 0:
            global_targets.flush()
            elapsed = time.perf_counter() - start_time
            print(
                f"[precompute_targets:{dataset.split}] count_pass n={sample_index + 1}/{total_examples} "
                f"ips={(sample_index + 1) / max(elapsed, 1e-6):.2f}",
                flush=True,
            )
    global_targets.flush()

    offsets = np.zeros((total_examples + 1,), dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    np.save(offsets_path, offsets)
    concept_ids_memmap = np.lib.format.open_memmap(
        concept_ids_path,
        mode="w+",
        dtype=np.int32,
        shape=(total_entries,),
    )
    mask_targets_memmap = np.lib.format.open_memmap(
        mask_targets_path,
        mode="w+",
        dtype=np.float32 if cfg.spatial_target_mode == "soft_box" else np.uint8,
        shape=(total_entries, cfg.mask_h, cfg.mask_w),
    )
    offset = 0
    second_start = time.perf_counter()
    for sample_index in range(total_examples):
        path, _ = dataset.dataset.samples[sample_index]
        annotation_index = dataset.annotation_index_for_row(sample_index)
        image_size = get_image_size(path, dataset.input_size, dataset.min_image_bytes)
        annotations = dataset._load_annotation(annotation_index)
        _, concept_ids, masks = build_gdino_target_sample(
            annotations,
            image_size,
            dataset.concept_to_idx,
            n_concepts,
            cfg,
        )
        count = int(concept_ids.shape[0])
        if count > 0:
            concept_ids_memmap[offset : offset + count] = concept_ids
            mask_targets_memmap[offset : offset + count] = masks
            offset += count
        if (sample_index + 1) % 1000 == 0:
            concept_ids_memmap.flush()
            mask_targets_memmap.flush()
            elapsed = time.perf_counter() - second_start
            print(
                f"[precompute_targets:{dataset.split}] data_pass n={sample_index + 1}/{total_examples} "
                f"ips={(sample_index + 1) / max(elapsed, 1e-6):.2f}",
                flush=True,
            )
    concept_ids_memmap.flush()
    mask_targets_memmap.flush()
    metadata = {
        "split": dataset.split,
        "n_examples": total_examples,
        "n_concepts": n_concepts,
        "mask_h": cfg.mask_h,
        "mask_w": cfg.mask_w,
        "input_size": cfg.input_size,
        "preprocess_resize_size": PREPROCESS_RESIZE_SIZE,
        "target_coordinate_frame": "resize_short_edge_then_center_crop",
        "total_entries": int(total_entries),
        "global_targets_path": str(global_targets_path),
        "offsets_path": str(offsets_path),
        "concept_ids_path": str(concept_ids_path),
        "mask_targets_path": str(mask_targets_path),
        "elapsed_sec": time.perf_counter() - start_time,
    }
    (split_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return metadata
