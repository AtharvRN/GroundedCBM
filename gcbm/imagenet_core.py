import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gcbm.spatial_targets import (  # noqa: E402
    PREPROCESS_RESIZE_SIZE,
    normalize_box,
    rasterize_box_iou as _shared_rasterize_box_iou,
    rasterize_box_soft_occupancy as _shared_rasterize_box_soft_occupancy,
    rasterize_box_target as _shared_rasterize_box_target,
    resize_short_edge_size,
    transform_box_for_resize_center_crop,
)


# Keep target rasterization in the same coordinate frame as the image tensor
# transform: Resize(shorter side=256) followed by CenterCrop(input_size).


IMAGENET_LABEL_ALIASES = {
    "website": "a web page",
    "beer bottle": "a bottle with a long neck",
    "wine bottle": "a bottle with a long neck",
    "soda bottle": "a glass or plastic bottle",
    "ski": "a pair of skis",
    "metal nail": "nails",
}


@dataclass
class Config:
    mode: str
    train_root: str
    train_manifest: str
    annotation_dir: str
    concept_file: str
    val_root: str
    save_dir: str
    run_name: str
    reuse_run_dir: str
    feature_dir: str
    precomputed_target_dir: str
    persist_feature_copy: bool
    max_train_images: int
    max_val_images: int
    val_split: float
    epochs: int
    batch_size: int
    workers: int
    prefetch_factor: int
    persistent_workers: bool
    pin_memory: bool
    device: str
    amp: str
    channels_last: bool
    tf32: bool
    cudnn_benchmark: bool
    seed: int
    min_image_bytes: int
    input_size: int
    resnet50_weights: str
    train_random_transforms: bool
    mask_h: int
    mask_w: int
    patch_iou_thresh: float
    concept_threshold: float
    spatial_target_mode: str
    spatial_loss_mode: str
    filter_concepts_by_count: bool
    concept_min_count: int
    concept_min_frequency: float
    concept_max_frequency: float
    optimizer: str
    lr: float
    weight_decay: float
    momentum: float
    global_pos_weight: float
    patch_pos_weight: float
    loss_global_w: float
    loss_mask_w: float
    branch_arch: str
    spatial_branch_mode: str
    spatial_stage: str
    residual_alpha: float
    profile_steps: int
    warmup_steps: int
    log_every: int
    save_every: int
    skip_final_layer: bool
    final_layer_type: str
    saga_batch_size: int
    saga_workers: int
    saga_prefetch_factor: int
    saga_step_size: float
    saga_lam: float
    saga_n_iters: int
    saga_verbose_every: int
    dense_lr: float
    dense_n_iters: int
    feature_storage_dtype: str
    saga_table_device: str
    vlg_init_path: str
    vlg_concepts_path: str
    freeze_global_head: bool
    scheduler: str
    print_config: bool
    # Old ImageNet checkpoints did not store this field and used avg pooling.
    residual_spatial_pooling: str = "avg"
    learn_spatial_residual_scale: bool = False
    eval_every: int = 1
    feature_batch_size: int = 256
    feature_workers: int = 4
    feature_prefetch_factor: int = 2


def configure_runtime(cfg: Config) -> None:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = cfg.tf32
    torch.backends.cudnn.allow_tf32 = cfg.tf32
    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    try:
        torch.set_float32_matmul_precision("high")
    except AttributeError:
        pass


def amp_dtype(name: str) -> Optional[torch.dtype]:
    if name == "none":
        return None
    if name == "bf16":
        return torch.bfloat16
    return torch.float16


def autocast_context(cfg: Config):
    dtype = amp_dtype(cfg.amp)
    if dtype is None or not str(cfg.device).startswith("cuda"):
        return torch.autocast(device_type="cpu", enabled=False)
    return torch.autocast(device_type="cuda", dtype=dtype)


def reset_cuda_peak_stats_if_needed(cfg: Config) -> None:
    if str(cfg.device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def cuda_peak_stats_mb(cfg: Config) -> Dict[str, float]:
    if str(cfg.device).startswith("cuda") and torch.cuda.is_available():
        return {
            "max_memory_allocated_mb": float(torch.cuda.max_memory_allocated() / (1024 ** 2)),
            "max_memory_reserved_mb": float(torch.cuda.max_memory_reserved() / (1024 ** 2)),
        }
    return {
        "max_memory_allocated_mb": 0.0,
        "max_memory_reserved_mb": 0.0,
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


class SafeImageFolderWithAnnotations(Dataset):
    def __init__(
        self,
        root: str,
        annotation_dir: str,
        concepts: Sequence[str],
        input_size: int,
        min_image_bytes: int,
        split: str,
        manifest: str = "",
        train_random_transforms: bool = False,
    ) -> None:
        self.root = root
        self.annotation_dir = annotation_dir
        self.input_size = int(input_size)
        self.min_image_bytes = int(min_image_bytes)
        self.split = split
        self.train_random_transforms = bool(train_random_transforms)
        self.concepts = list(concepts)
        self.concept_to_idx = {name: idx for idx, name in enumerate(self.concepts)}
        self.sample_indices: Optional[List[int]] = None
        self.annotation_indices: Optional[List[int]] = None
        if manifest:
            self.dataset = self._load_manifest(manifest, split)
        else:
            self.dataset = ImageFolder(root=root, loader=self._safe_loader, transform=self._transform(split))
        self.precomputed_targets: Optional[PrecomputedTargetStore] = None

    def _load_manifest(self, manifest: str, split: str) -> Any:
        samples: List[Tuple[str, int]] = []
        sample_indices: List[int] = []
        annotation_indices: List[int] = []
        class_names: Dict[int, str] = {}
        with open(manifest, "r") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                path = str(payload["path"])
                class_id = int(payload["class_id"])
                sample_index = int(payload.get("sample_index", len(samples)))
                annotation_index = int(payload.get("annotation_index", sample_index))
                samples.append((path, class_id))
                sample_indices.append(sample_index)
                annotation_indices.append(annotation_index)
                class_names[class_id] = str(payload.get("class_name", class_id))
        if not samples:
            raise ValueError(f"Manifest has no samples: {manifest}")
        max_class_id = max(class_names)
        classes = [str(idx) for idx in range(max_class_id + 1)]
        for class_id, class_name in class_names.items():
            classes[class_id] = class_name

        class ManifestDataset:
            def __len__(self) -> int:
                return len(self.samples)

        dataset = ManifestDataset()
        dataset.samples = samples
        dataset.classes = classes
        dataset.transform = self._transform(split)
        self.sample_indices = sample_indices
        self.annotation_indices = annotation_indices
        return dataset

    def attach_precomputed_targets(self, root: str, cfg: Optional[Config] = None) -> None:
        """Attach on-disk GDINO supervision generated by precompute_target_store.

        The store is normally indexed by dataset position. Manifest subsets can
        also point into a larger precomputed store via sample_index.
        """
        if not root:
            return
        target_dir = Path(root) / self.split
        if not target_dir.is_dir():
            raise FileNotFoundError(f"Missing precomputed target directory: {target_dir}")
        self.precomputed_targets = PrecomputedTargetStore(target_dir)
        if self.sample_indices is None and len(self.precomputed_targets) != len(self.dataset):
            raise ValueError(
                f"Precomputed targets at {target_dir} have {len(self.precomputed_targets)} entries, "
                f"expected {len(self.dataset)}"
            )
        if self.sample_indices is not None:
            max_index = max(self.sample_indices) if self.sample_indices else -1
            if max_index >= len(self.precomputed_targets):
                raise ValueError(
                    f"Manifest references sample_index={max_index}, but precomputed targets at "
                    f"{target_dir} have only {len(self.precomputed_targets)} entries"
                )
        if self.precomputed_targets.n_concepts != len(self.concepts):
            raise ValueError(
                f"Precomputed targets at {target_dir} have {self.precomputed_targets.n_concepts} concepts, "
                f"expected {len(self.concepts)}"
            )
        if cfg is not None:
            self.precomputed_targets.validate_target_geometry(cfg)

    def apply_concept_filter(self, keep_indices: Sequence[int]) -> None:
        keep = [int(idx) for idx in keep_indices]
        self.concepts = [self.concepts[idx] for idx in keep]
        self.concept_to_idx = {name: idx for idx, name in enumerate(self.concepts)}
        if self.precomputed_targets is not None:
            self.precomputed_targets.set_concept_filter(keep)

    def _transform(self, split: str) -> transforms.Compose:
        normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
        if split == "train" and self.train_random_transforms:
            return transforms.Compose(
                [
                    transforms.RandomResizedCrop(self.input_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
        return transforms.Compose(
            [
                transforms.Resize(PREPROCESS_RESIZE_SIZE),
                transforms.CenterCrop(self.input_size),
                transforms.ToTensor(),
                normalize,
            ]
        )

    def _safe_loader(self, path: str) -> Image.Image:
        try:
            if os.path.getsize(path) < self.min_image_bytes:
                raise OSError(f"tiny file: {path}")
            with Image.open(path) as image:
                return image.convert("RGB")
        except (FileNotFoundError, OSError, UnidentifiedImageError):
            return Image.new("RGB", (self.input_size, self.input_size), color=0)

    def _annotation_path(self, sample_index: int) -> Path:
        split_dir = "imagenet_train" if self.split == "train" else "imagenet_val"
        return Path(self.annotation_dir) / split_dir / f"{sample_index}.json"

    def _load_annotation(self, sample_index: int) -> List[Dict[str, Any]]:
        path = self._annotation_path(sample_index)
        if not path.exists():
            return []
        try:
            with path.open("r") as handle:
                payload = json.load(handle)
        except Exception:
            return []
        if isinstance(payload, list):
            return payload
        return payload.get("concepts", [])

    def annotation_index_for_row(self, row_index: int) -> int:
        if self.annotation_indices is not None:
            return int(self.annotation_indices[row_index])
        if self.sample_indices is not None:
            return int(self.sample_indices[row_index])
        return int(row_index)

    def __len__(self) -> int:
        return len(self.dataset.samples)

    def __getitem__(self, index: int):
        path, class_id = self.dataset.samples[index]
        sample_index = int(self.sample_indices[index]) if self.sample_indices is not None else int(index)
        annotation_index = self.annotation_index_for_row(index)
        with self._safe_loader(path) as raw_image:
            image_size = (int(raw_image.size[0]), int(raw_image.size[1]))
            image = self.dataset.transform(raw_image) if self.dataset.transform is not None else raw_image
        item = {
            "image": image,
            "class_id": int(class_id),
            "sample_index": sample_index,
            "image_size": image_size,
        }
        if self.precomputed_targets is not None:
            item.update(self.precomputed_targets.get(sample_index))
        else:
            item["annotation"] = self._load_annotation(annotation_index)
        return item


class DatasetView(Dataset):
    def __init__(self, base_dataset: SafeImageFolderWithAnnotations, indices: Sequence[int]) -> None:
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.concepts = base_dataset.concepts
        self.concept_to_idx = base_dataset.concept_to_idx

    def refresh_concepts(self) -> None:
        self.concepts = self.base_dataset.concepts
        self.concept_to_idx = self.base_dataset.concept_to_idx

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        return self.base_dataset[self.indices[index]]


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


def split_train_val(
    dataset: SafeImageFolderWithAnnotations,
    val_split: float,
    max_train_images: int,
    max_val_images: int,
    seed: int,
) -> Tuple[DatasetView, DatasetView]:
    total = len(dataset)
    indices = list(range(total))
    generator = random.Random(seed)
    indices = select_subset_indices(
        dataset,
        indices,
        max_images=max_train_images,
        seed=seed,
        stratify=True,
    )
    n_val = int(round(float(val_split) * len(indices)))
    n_val = min(max(n_val, 1), max(len(indices) - 1, 1))
    generator.shuffle(indices)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    if max_val_images > 0:
        val_indices = select_subset_indices(
            dataset,
            val_indices,
            max_images=max_val_images,
            seed=seed + 1,
            stratify=True,
        )
    return DatasetView(dataset, train_indices), DatasetView(dataset, val_indices)


def select_subset_indices(
    dataset: SafeImageFolderWithAnnotations,
    indices: Sequence[int],
    *,
    max_images: int,
    seed: int,
    stratify: bool,
) -> List[int]:
    selected = list(indices)
    if max_images <= 0 or len(selected) <= max_images:
        return selected

    generator = random.Random(seed)
    if not stratify:
        generator.shuffle(selected)
        return selected[:max_images]

    shuffled = list(selected)
    generator.shuffle(shuffled)
    class_to_indices: Dict[int, List[int]] = {}
    for sample_index in shuffled:
        _, class_id = dataset.dataset.samples[sample_index]
        class_to_indices.setdefault(int(class_id), []).append(int(sample_index))

    class_ids = list(class_to_indices)
    generator.shuffle(class_ids)
    per_class = max_images // len(class_ids)
    remainder = max_images % len(class_ids)

    chosen: List[int] = []
    chosen_set: set[int] = set()
    for class_position, class_id in enumerate(class_ids):
        want = per_class + (1 if class_position < remainder else 0)
        if want <= 0:
            continue
        class_choices = class_to_indices[class_id][:want]
        chosen.extend(class_choices)
        chosen_set.update(class_choices)

    if len(chosen) < max_images:
        for sample_index in shuffled:
            if sample_index in chosen_set:
                continue
            chosen.append(sample_index)
            chosen_set.add(sample_index)
            if len(chosen) >= max_images:
                break

    generator.shuffle(chosen)
    return chosen[:max_images]


def unwrap_dataset_view(dataset: Dataset) -> Tuple[SafeImageFolderWithAnnotations, List[int]]:
    if isinstance(dataset, DatasetView):
        return dataset.base_dataset, list(dataset.indices)
    if isinstance(dataset, SafeImageFolderWithAnnotations):
        return dataset, list(range(len(dataset)))
    raise TypeError(f"Unsupported dataset type for concept filtering: {type(dataset).__name__}")


def refresh_dataset_concepts(dataset: Dataset) -> None:
    if isinstance(dataset, DatasetView):
        dataset.refresh_concepts()


def count_concept_targets(dataset: Dataset, cfg: Config) -> Tuple[np.ndarray, int]:
    base_dataset, indices = unwrap_dataset_view(dataset)
    n_examples = len(indices)
    n_concepts = len(base_dataset.concepts)
    counts = np.zeros((n_concepts,), dtype=np.int64)
    if base_dataset.precomputed_targets is not None:
        targets = base_dataset.precomputed_targets.global_targets
        chunk_size = 4096
        for start in range(0, n_examples, chunk_size):
            chunk_indices = sorted(indices[start : start + chunk_size])
            counts += np.asarray(targets[chunk_indices], dtype=np.int64).sum(axis=0)
            if (start + len(chunk_indices)) % 50000 == 0:
                print(
                    f"[concept_filter] counted {start + len(chunk_indices)}/{n_examples} precomputed targets",
                    flush=True,
                )
        return counts, n_examples

    start_time = time.perf_counter()
    for position, sample_index in enumerate(indices, start=1):
        path, _ = base_dataset.dataset.samples[sample_index]
        annotation_index = base_dataset.annotation_index_for_row(sample_index)
        image_size = get_image_size(path, base_dataset.input_size, base_dataset.min_image_bytes)
        annotations = base_dataset._load_annotation(annotation_index)
        global_target, _, _ = build_gdino_target_sample(
            annotations,
            image_size,
            base_dataset.concept_to_idx,
            n_concepts,
            cfg,
        )
        counts += global_target.astype(np.int64, copy=False)
        if position % 50000 == 0:
            elapsed = time.perf_counter() - start_time
            print(
                f"[concept_filter] counted {position}/{n_examples} annotation targets "
                f"ips={position / max(elapsed, 1e-6):.2f}",
                flush=True,
            )
    return counts, n_examples


def apply_count_concept_filter(
    cfg: Config,
    train_dataset: Dataset,
    all_datasets: Sequence[Dataset],
) -> Optional[Dict[str, Any]]:
    if not cfg.filter_concepts_by_count:
        return None
    counts, n_examples = count_concept_targets(train_dataset, cfg)
    frequencies = counts.astype(np.float64) / max(int(n_examples), 1)
    keep_mask = (
        (counts >= int(cfg.concept_min_count))
        & (frequencies >= float(cfg.concept_min_frequency))
        & (frequencies <= float(cfg.concept_max_frequency))
    )
    keep_indices = np.flatnonzero(keep_mask).astype(np.int64)
    if keep_indices.size == 0:
        raise RuntimeError("Concept count filtering removed all concepts")

    seen: set[int] = set()
    for dataset in all_datasets:
        base_dataset, _ = unwrap_dataset_view(dataset)
        ident = id(base_dataset)
        if ident in seen:
            continue
        base_dataset.apply_concept_filter(keep_indices.tolist())
        seen.add(ident)
    for dataset in all_datasets:
        refresh_dataset_concepts(dataset)

    removed_indices = np.flatnonzero(~keep_mask).astype(np.int64)
    summary = {
        "enabled": True,
        "n_examples": int(n_examples),
        "original_n_concepts": int(counts.shape[0]),
        "kept_n_concepts": int(keep_indices.size),
        "removed_n_concepts": int(removed_indices.size),
        "min_count": int(cfg.concept_min_count),
        "min_frequency": float(cfg.concept_min_frequency),
        "max_frequency": float(cfg.concept_max_frequency),
        "keep_indices": keep_indices.tolist(),
        "removed_indices": removed_indices.tolist(),
        "kept_min_count": int(counts[keep_indices].min()),
        "kept_max_count": int(counts[keep_indices].max()),
    }
    print(
        "[concept_filter] kept "
        f"{summary['kept_n_concepts']}/{summary['original_n_concepts']} concepts "
        f"(removed {summary['removed_n_concepts']})",
        flush=True,
    )
    return summary


def collate_batch(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    payload = {
        "images": torch.stack([item["image"] for item in batch], dim=0),
        "class_ids": torch.tensor([item["class_id"] for item in batch], dtype=torch.long),
        "sample_indices": torch.tensor([item["sample_index"] for item in batch], dtype=torch.long),
        "image_sizes": [item["image_size"] for item in batch],
    }
    if "global_target" not in batch[0]:
        payload["annotations"] = [item["annotation"] for item in batch]
        return payload

    payload["global_targets"] = torch.stack([item["global_target"] for item in batch], dim=0)
    max_k = max((int(item["mask_indices"].numel()) for item in batch), default=0)
    if max_k == 0:
        payload["mask_indices"] = torch.full((len(batch), 1), -1, dtype=torch.long)
        payload["mask_targets"] = torch.zeros((len(batch), 1, batch[0]["mask_targets"].shape[-2], batch[0]["mask_targets"].shape[-1]), dtype=torch.float32)
        payload["mask_valid"] = torch.zeros((len(batch), 1), dtype=torch.bool)
        return payload

    mask_h = int(batch[0]["mask_targets"].shape[-2])
    mask_w = int(batch[0]["mask_targets"].shape[-1])
    idx_pad = torch.full((len(batch), max_k), -1, dtype=torch.long)
    mask_pad = torch.zeros((len(batch), max_k, mask_h, mask_w), dtype=torch.float32)
    valid = torch.zeros((len(batch), max_k), dtype=torch.bool)
    for batch_index, item in enumerate(batch):
        count = int(item["mask_indices"].numel())
        if count == 0:
            continue
        idx_pad[batch_index, :count] = item["mask_indices"]
        mask_pad[batch_index, :count] = item["mask_targets"]
        valid[batch_index, :count] = True
    payload["mask_indices"] = idx_pad
    payload["mask_targets"] = mask_pad
    payload["mask_valid"] = valid
    return payload


def build_loader(
    dataset: Dataset,
    cfg: Config,
    shuffle: bool,
    drop_last: bool,
    *,
    batch_size: Optional[int] = None,
    workers: Optional[int] = None,
    pin_memory: Optional[bool] = None,
    persistent_workers: Optional[bool] = None,
    prefetch_factor: Optional[int] = None,
) -> DataLoader:
    effective_batch_size = int(cfg.batch_size if batch_size is None else batch_size)
    effective_workers = int(cfg.workers if workers is None else workers)
    effective_pin_memory = cfg.pin_memory if pin_memory is None else bool(pin_memory)
    kwargs: Dict[str, Any] = {
        "dataset": dataset,
        "batch_size": effective_batch_size,
        "shuffle": shuffle,
        "num_workers": effective_workers,
        "pin_memory": effective_pin_memory,
        "collate_fn": collate_batch,
        "drop_last": drop_last,
    }
    if effective_workers > 0:
        kwargs["persistent_workers"] = (
            cfg.persistent_workers if persistent_workers is None else bool(persistent_workers)
        )
        kwargs["prefetch_factor"] = max(
            1,
            int(cfg.prefetch_factor if prefetch_factor is None else prefetch_factor),
        )
    return DataLoader(**kwargs)

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

    max_k = max((tensor.numel() for tensor in sparse_indices), default=0)
    if max_k == 0:
        idx_pad = torch.full((len(annotations), 1), -1, dtype=torch.long)
        mask_pad = torch.zeros((len(annotations), 1, cfg.mask_h, cfg.mask_w), dtype=torch.float32)
        valid = torch.zeros((len(annotations), 1), dtype=torch.bool)
    else:
        idx_pad = torch.full((len(annotations), max_k), -1, dtype=torch.long)
        mask_pad = torch.zeros((len(annotations), max_k, cfg.mask_h, cfg.mask_w), dtype=torch.float32)
        valid = torch.zeros((len(annotations), max_k), dtype=torch.bool)
        for batch_idx, (indices, masks) in enumerate(zip(sparse_indices, sparse_masks)):
            if indices.numel() == 0:
                continue
            idx_pad[batch_idx, : indices.numel()] = indices
            mask_pad[batch_idx, : indices.numel()] = masks
            valid[batch_idx, : indices.numel()] = True

    return (
        global_targets.to(device, non_blocking=True),
        idx_pad.to(device, non_blocking=True),
        mask_pad.to(device, non_blocking=True),
        valid.to(device, non_blocking=True),
    )


def batch_targets_to_device(batch: Dict[str, Any], cfg: Config) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        batch["global_targets"].to(cfg.device, non_blocking=True),
        batch["mask_indices"].to(cfg.device, non_blocking=True),
        batch["mask_targets"].to(cfg.device, non_blocking=True),
        batch["mask_valid"].to(cfg.device, non_blocking=True),
    )


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


def prepare_images(images: torch.Tensor, cfg: Config) -> torch.Tensor:
    if cfg.channels_last:
        images = images.contiguous(memory_format=torch.channels_last)
    return images.to(cfg.device, non_blocking=cfg.pin_memory)


def make_optimizer(head: nn.Module, cfg: Config) -> torch.optim.Optimizer:
    parameters = [parameter for parameter in head.parameters() if parameter.requires_grad]
    print(f"[training] trainable parameters={sum(p.numel() for p in parameters)}", flush=True)
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(parameters, lr=cfg.lr, weight_decay=cfg.weight_decay)
    return torch.optim.SGD(
        parameters,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        momentum=cfg.momentum,
    )


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: Config,
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    if cfg.scheduler == "none":
        return None
    if cfg.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(int(cfg.epochs), 1),
            eta_min=1e-6,
        )
    raise ValueError(f"Unsupported scheduler: {cfg.scheduler}")


def build_run_dir(cfg: Config) -> Path:
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    name = cfg.run_name or f"savlg_imagenet_standalone_{timestamp}"
    run_dir = Path(cfg.save_dir) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_checkpoint(
    run_dir: Path,
    head: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    cfg: Config,
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "head": head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": asdict(cfg),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        },
        run_dir / f"checkpoint_epoch_{epoch:03d}.pt",
    )
    torch.save(head.state_dict(), run_dir / "concept_head_latest.pt")
    payload = {
        "epoch": epoch,
        "train": train_metrics,
        "val": val_metrics,
    }
    with (run_dir / "metrics.jsonl").open("a") as handle:
        handle.write(json.dumps(payload) + "\n")
