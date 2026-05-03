import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import ResNet50_Weights, resnet50

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from glm_saga.elasticnet import glm_saga


# Keep target rasterization in the same coordinate frame as the image tensor
# transform: Resize(shorter side=256) followed by CenterCrop(input_size).
PREPROCESS_RESIZE_SIZE = 256


IMAGENET_LABEL_ALIASES = {
    "website": "a web page",
    "beer bottle": "a bottle with a long neck",
    "wine bottle": "a bottle with a long neck",
    "soda bottle": "a glass or plastic bottle",
    "ski": "a pair of skis",
    "metal nail": "nails",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone ImageNet SAVLG trainer optimized for A10 throughput."
    )
    parser.add_argument(
        "--mode",
        choices=["train", "profile", "glm_only", "dense_only", "precompute_targets"],
        default="train",
    )
    parser.add_argument("--train_root", default="")
    parser.add_argument(
        "--train_manifest",
        default="",
        help="Optional JSONL manifest with path/class_id/sample_index entries to avoid ImageFolder full-tree scans.",
    )
    parser.add_argument("--annotation_dir", default="")
    parser.add_argument("--concept_file", default="concept_files/imagenet_filtered.txt")
    parser.add_argument("--val_root", default="")
    parser.add_argument("--save_dir", default="saved_models/imagenet_standalone")
    parser.add_argument("--run_name", default="")
    parser.add_argument("--reuse_run_dir", default="")
    parser.add_argument("--feature_dir", default="")
    parser.add_argument("--precomputed_target_dir", default="")
    parser.add_argument("--persist_feature_copy", action="store_true")
    parser.add_argument("--max_train_images", type=int, default=0)
    parser.add_argument("--max_val_images", type=int, default=0)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--persistent_workers", action="store_true", default=True)
    parser.add_argument("--disable_persistent_workers", action="store_true")
    parser.add_argument("--pin_memory", action="store_true", default=True)
    parser.add_argument("--disable_pin_memory", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", choices=["fp16", "bf16", "none"], default="fp16")
    parser.add_argument("--channels_last", action="store_true", default=True)
    parser.add_argument("--disable_channels_last", action="store_true")
    parser.add_argument("--tf32", action="store_true", default=True)
    parser.add_argument("--disable_tf32", action="store_true")
    parser.add_argument("--cudnn_benchmark", action="store_true", default=True)
    parser.add_argument("--disable_cudnn_benchmark", action="store_true")
    parser.add_argument("--seed", type=int, default=6885)
    parser.add_argument("--min_image_bytes", type=int, default=2048)
    parser.add_argument("--input_size", type=int, default=224)
    transform_group = parser.add_mutually_exclusive_group()
    transform_group.add_argument(
        "--enable_train_random_transforms",
        action="store_true",
        help="Use RandomResizedCrop+RandomHorizontalFlip for the train split. Disabled by default because static bbox masks are not transform-aware.",
    )
    transform_group.add_argument(
        "--disable_train_random_transforms",
        action="store_true",
        help="Use deterministic Resize+CenterCrop for the train split. This is the default and is kept as an explicit job-level guard.",
    )
    parser.add_argument("--mask_h", type=int, default=7)
    parser.add_argument("--mask_w", type=int, default=7)
    parser.add_argument("--patch_iou_thresh", type=float, default=0.5)
    parser.add_argument("--concept_threshold", type=float, default=0.15)
    parser.add_argument(
        "--spatial_target_mode",
        choices=["hard_iou", "soft_box"],
        default="soft_box",
        help="How box supervision is rasterized into spatial targets.",
    )
    parser.add_argument(
        "--spatial_loss_mode",
        choices=["bce", "soft_align"],
        default="soft_align",
        help="Spatial alignment loss: hard/soft BCE or KL alignment to target spatial distribution.",
    )
    parser.add_argument(
        "--filter_concepts_by_count",
        action="store_true",
        help="Filter concepts by train-set target frequency before building the concept head.",
    )
    parser.add_argument(
        "--concept_min_count",
        type=int,
        default=1,
        help="Minimum train-set positive count when --filter_concepts_by_count is enabled.",
    )
    parser.add_argument(
        "--concept_min_frequency",
        type=float,
        default=0.0,
        help="Minimum train-set positive fraction when --filter_concepts_by_count is enabled.",
    )
    parser.add_argument(
        "--concept_max_frequency",
        type=float,
        default=1.0,
        help="Maximum train-set positive fraction when --filter_concepts_by_count is enabled.",
    )
    parser.add_argument("--optimizer", choices=["sgd", "adamw"], default="sgd")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--global_pos_weight", type=float, default=1.0)
    parser.add_argument("--patch_pos_weight", type=float, default=1.0)
    parser.add_argument("--loss_global_w", type=float, default=1.0)
    parser.add_argument("--loss_mask_w", type=float, default=1.0)
    parser.add_argument("--loss_dice_w", type=float, default=0.0)
    parser.add_argument("--branch_arch", choices=["shared", "dual"], default="shared")
    parser.add_argument(
        "--spatial_branch_mode", choices=["shared_stage", "multiscale_conv45"], default="shared_stage"
    )
    parser.add_argument("--spatial_stage", choices=["conv4", "conv5"], default="conv5")
    parser.add_argument("--residual_alpha", type=float, default=0.8)
    parser.add_argument(
        "--residual_spatial_pooling",
        choices=["avg", "lse"],
        default="lse",
        help="Pooling for residual spatial logits. lse matches the CUB SAVLG residual branch.",
    )
    parser.add_argument(
        "--learn_spatial_residual_scale",
        action="store_true",
        help="Learn a positive exp(log_scale) multiplier on the spatial residual branch.",
    )
    parser.add_argument("--profile_steps", type=int, default=20)
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--skip_final_layer", action="store_true")
    parser.add_argument(
        "--final_layer_type",
        choices=["sparse", "dense"],
        default="sparse",
        help="Final classifier to train after CBL feature extraction in train mode.",
    )
    parser.add_argument("--saga_batch_size", type=int, default=512)
    parser.add_argument("--saga_workers", type=int, default=0)
    parser.add_argument("--saga_prefetch_factor", type=int, default=2)
    parser.add_argument("--saga_step_size", type=float, default=0.1)
    parser.add_argument("--saga_lam", type=float, default=5e-4)
    parser.add_argument("--saga_n_iters", type=int, default=80)
    parser.add_argument("--saga_verbose_every", type=int, default=10)
    parser.add_argument("--dense_lr", type=float, default=1e-3)
    parser.add_argument("--dense_n_iters", type=int, default=20)
    parser.add_argument("--feature_storage_dtype", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--saga_table_device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--vlg_init_path", default="")
    parser.add_argument("--vlg_concepts_path", default="")
    parser.add_argument("--freeze_global_head", action="store_true")
    parser.add_argument("--scheduler", choices=["none", "cosine"], default="none")
    parser.add_argument("--print_config", action="store_true")
    return parser.parse_args()


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
    loss_dice_w: float
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


def build_config(args: argparse.Namespace) -> Config:
    return Config(
        mode=args.mode,
        train_root=args.train_root,
        train_manifest=args.train_manifest,
        annotation_dir=args.annotation_dir,
        concept_file=args.concept_file,
        val_root=args.val_root,
        save_dir=args.save_dir,
        run_name=args.run_name,
        reuse_run_dir=args.reuse_run_dir,
        feature_dir=args.feature_dir,
        precomputed_target_dir=args.precomputed_target_dir,
        persist_feature_copy=bool(args.persist_feature_copy),
        max_train_images=args.max_train_images,
        max_val_images=args.max_val_images,
        val_split=args.val_split,
        epochs=args.epochs,
        batch_size=args.batch_size,
        workers=args.workers,
        prefetch_factor=max(1, int(args.prefetch_factor)),
        persistent_workers=bool(args.persistent_workers and not args.disable_persistent_workers),
        pin_memory=bool(args.pin_memory and not args.disable_pin_memory),
        device=args.device,
        amp=args.amp,
        channels_last=bool(args.channels_last and not args.disable_channels_last),
        tf32=bool(args.tf32 and not args.disable_tf32),
        cudnn_benchmark=bool(args.cudnn_benchmark and not args.disable_cudnn_benchmark),
        seed=args.seed,
        min_image_bytes=args.min_image_bytes,
        input_size=args.input_size,
        train_random_transforms=bool(args.enable_train_random_transforms),
        mask_h=args.mask_h,
        mask_w=args.mask_w,
        patch_iou_thresh=args.patch_iou_thresh,
        concept_threshold=args.concept_threshold,
        spatial_target_mode=args.spatial_target_mode,
        spatial_loss_mode=args.spatial_loss_mode,
        filter_concepts_by_count=bool(args.filter_concepts_by_count),
        concept_min_count=max(0, int(args.concept_min_count)),
        concept_min_frequency=float(args.concept_min_frequency),
        concept_max_frequency=float(args.concept_max_frequency),
        optimizer=args.optimizer,
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        global_pos_weight=args.global_pos_weight,
        patch_pos_weight=args.patch_pos_weight,
        loss_global_w=args.loss_global_w,
        loss_mask_w=args.loss_mask_w,
        loss_dice_w=args.loss_dice_w,
        branch_arch=args.branch_arch,
        spatial_branch_mode=args.spatial_branch_mode,
        spatial_stage=args.spatial_stage,
        residual_alpha=args.residual_alpha,
        profile_steps=args.profile_steps,
        warmup_steps=args.warmup_steps,
        log_every=args.log_every,
        save_every=args.save_every,
        skip_final_layer=bool(args.skip_final_layer),
        final_layer_type=args.final_layer_type,
        saga_batch_size=args.saga_batch_size,
        saga_workers=max(0, int(args.saga_workers)),
        saga_prefetch_factor=max(1, int(args.saga_prefetch_factor)),
        saga_step_size=args.saga_step_size,
        saga_lam=args.saga_lam,
        saga_n_iters=args.saga_n_iters,
        saga_verbose_every=max(1, int(args.saga_verbose_every)),
        dense_lr=args.dense_lr,
        dense_n_iters=max(1, int(args.dense_n_iters)),
        feature_storage_dtype=args.feature_storage_dtype,
        saga_table_device=args.saga_table_device,
        vlg_init_path=str(args.vlg_init_path or ""),
        vlg_concepts_path=str(args.vlg_concepts_path or ""),
        freeze_global_head=bool(args.freeze_global_head),
        scheduler=str(args.scheduler or "none"),
        print_config=bool(args.print_config),
        residual_spatial_pooling=str(args.residual_spatial_pooling),
        learn_spatial_residual_scale=bool(args.learn_spatial_residual_scale),
    )


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


def validate_config(cfg: Config) -> None:
    if cfg.mode in {"train", "profile", "precompute_targets"}:
        if not cfg.train_root:
            raise ValueError("--train_root is required for train/profile mode")
        if not cfg.annotation_dir:
            raise ValueError("--annotation_dir is required for train/profile mode")
    if cfg.mode in {"glm_only", "dense_only"} and not cfg.reuse_run_dir:
        raise ValueError("--reuse_run_dir is required for glm_only/dense_only mode")
    if cfg.mode == "precompute_targets" and not cfg.precomputed_target_dir:
        raise ValueError("--precomputed_target_dir is required for precompute_targets mode")
    if not 0.0 <= cfg.concept_min_frequency <= 1.0:
        raise ValueError("--concept_min_frequency must be in [0, 1]")
    if not 0.0 <= cfg.concept_max_frequency <= 1.0:
        raise ValueError("--concept_max_frequency must be in [0, 1]")
    if cfg.concept_min_frequency > cfg.concept_max_frequency:
        raise ValueError("--concept_min_frequency cannot exceed --concept_max_frequency")
    if cfg.freeze_global_head and not cfg.vlg_init_path:
        raise ValueError("--freeze_global_head requires --vlg_init_path")
    if cfg.vlg_init_path and not cfg.vlg_concepts_path:
        raise ValueError("--vlg_concepts_path is required when --vlg_init_path is set")
    if getattr(cfg, "residual_spatial_pooling", "avg") not in {"avg", "lse"}:
        raise ValueError("--residual_spatial_pooling must be one of: avg, lse")


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
        if manifest:
            self.dataset = self._load_manifest(manifest, split)
        else:
            self.dataset = ImageFolder(root=root, loader=self._safe_loader, transform=self._transform(split))
        self.precomputed_targets: Optional[PrecomputedTargetStore] = None

    def _load_manifest(self, manifest: str, split: str) -> Any:
        samples: List[Tuple[str, int]] = []
        sample_indices: List[int] = []
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
                samples.append((path, class_id))
                sample_indices.append(sample_index)
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
        return dataset

    def attach_precomputed_targets(self, root: str, cfg: Optional[Config] = None) -> None:
        """Attach on-disk GDINO supervision generated by precompute_target_store.

        The store is indexed by dataset position, so length/concept/geometry
        checks are intentionally strict. A mismatch here means masks may no
        longer correspond to the image tensors used by training.
        """
        if not root:
            return
        target_dir = Path(root) / self.split
        if not target_dir.is_dir():
            raise FileNotFoundError(f"Missing precomputed target directory: {target_dir}")
        self.precomputed_targets = PrecomputedTargetStore(target_dir)
        if len(self.precomputed_targets) != len(self.dataset):
            raise ValueError(
                f"Precomputed targets at {target_dir} have {len(self.precomputed_targets)} entries, "
                f"expected {len(self.dataset)}"
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

    def __len__(self) -> int:
        return len(self.dataset.samples)

    def __getitem__(self, index: int):
        path, class_id = self.dataset.samples[index]
        sample_index = int(self.sample_indices[index]) if self.sample_indices is not None else int(index)
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
            item.update(self.precomputed_targets.get(index))
        else:
            item["annotation"] = self._load_annotation(sample_index)
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
        annotation_index = (
            int(base_dataset.sample_indices[sample_index])
            if base_dataset.sample_indices is not None
            else int(sample_index)
        )
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


class IndexedTensorDataset(Dataset):
    def __init__(self, features: torch.Tensor, targets: torch.Tensor) -> None:
        self.features = features
        self.targets = targets

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int):
        return self.features[index], self.targets[index], int(index)


class MemmapFeatureDataset(Dataset):
    def __init__(
        self,
        feature_path: Path,
        target_path: Path,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
        include_index: bool = False,
    ) -> None:
        self.features = np.load(feature_path, mmap_mode="r")
        self.targets = np.load(target_path, mmap_mode="r")
        self.mean = None if mean is None else np.asarray(mean, dtype=np.float32)
        self.std = None if std is None else np.asarray(std, dtype=np.float32)
        self.include_index = include_index

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int):
        feature = np.asarray(self.features[index], dtype=np.float32)
        if self.mean is not None and self.std is not None:
            feature = (feature - self.mean) / self.std
        tensor = torch.from_numpy(np.ascontiguousarray(feature))
        target = int(self.targets[index])
        if self.include_index:
            return tensor, target, int(index)
        return tensor, target


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


def feature_storage_dtype(cfg: Config) -> np.dtype:
    if cfg.feature_storage_dtype == "fp32":
        return np.float32
    return np.float16


def normalize_box(box: Sequence[float], image_size: Tuple[int, int]) -> Optional[Tuple[float, float, float, float]]:
    pixel_box = _box_to_original_pixels(box, image_size=image_size)
    if pixel_box is None:
        return None
    width, height = image_size
    x1, y1, x2, y2 = pixel_box
    return x1 / width, y1 / height, x2 / width, y2 / height


def _box_to_original_pixels(
    box: Sequence[float],
    image_size: Tuple[int, int],
) -> Optional[Tuple[float, float, float, float]]:
    # Real GDINO annotations are saved as original-image pixel xyxy boxes, but
    # smoke-test or external annotations may already be normalized to [0, 1].
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    x1, y1, x2, y2 = [float(v) for v in box]
    width, height = image_size
    if width <= 0 or height <= 0:
        return None
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) > 1.5:
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
    else:
        x1, x2 = sorted((x1 * width, x2 * width))
        y1, y2 = sorted((y1 * height, y2 * height))
    x1, x2 = float(np.clip(x1, 0.0, width)), float(np.clip(x2, 0.0, width))
    y1, y2 = float(np.clip(y1, 0.0, height)), float(np.clip(y2, 0.0, height))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def resize_short_edge_size(
    image_size: Tuple[int, int],
    resize_size: int = PREPROCESS_RESIZE_SIZE,
) -> Tuple[int, int]:
    # Match torchvision.transforms.Resize(int): preserve aspect ratio and set
    # the shorter edge to resize_size.
    width, height = image_size
    if width <= 0 or height <= 0:
        return int(resize_size), int(resize_size)
    if width == height:
        return int(resize_size), int(resize_size)
    if width < height:
        return int(resize_size), int(resize_size * height / width)
    return int(resize_size * width / height), int(resize_size)


def transform_box_for_model_input(
    box: Sequence[float],
    image_size: Tuple[int, int],
    input_size: Optional[int],
    resize_size: int = PREPROCESS_RESIZE_SIZE,
) -> Optional[Tuple[float, float, float, float]]:
    # Apply the same deterministic geometry as the image transform before
    # rasterizing, so each mask cell corresponds to the model's spatial map.
    pixel_box = _box_to_original_pixels(box, image_size=image_size)
    if pixel_box is None:
        return None
    crop_size = int(input_size or resize_size)
    width, height = image_size
    resized_width, resized_height = resize_short_edge_size(image_size, resize_size=resize_size)
    scale_x = resized_width / float(width)
    scale_y = resized_height / float(height)
    x1, y1, x2, y2 = pixel_box
    x1 *= scale_x
    x2 *= scale_x
    y1 *= scale_y
    y2 *= scale_y

    # Match torchvision CenterCrop exactly for deterministic precompute.
    crop_left = max(int(round((resized_width - crop_size) / 2.0)), 0)
    crop_top = max(int(round((resized_height - crop_size) / 2.0)), 0)
    x1 -= crop_left
    x2 -= crop_left
    y1 -= crop_top
    y2 -= crop_top

    x1 = float(np.clip(x1, 0.0, crop_size))
    x2 = float(np.clip(x2, 0.0, crop_size))
    y1 = float(np.clip(y1, 0.0, crop_size))
    y2 = float(np.clip(y2, 0.0, crop_size))
    # If center-cropping removes the annotated region entirely, do not create a
    # spatial target for that concept on this image.
    if x2 <= x1 or y2 <= y1:
        return None
    return x1 / crop_size, y1 / crop_size, x2 / crop_size, y2 / crop_size


def rasterize_box_iou(
    box: Sequence[float],
    image_size: Tuple[int, int],
    input_size: int,
    mask_h: int,
    mask_w: int,
    iou_thresh: float,
) -> Optional[np.ndarray]:
    norm = transform_box_for_model_input(box, image_size=image_size, input_size=input_size)
    if norm is None:
        return None
    x1, y1, x2, y2 = norm
    box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if box_area <= 0.0:
        return None
    mask = np.zeros((mask_h, mask_w), dtype=np.float32)
    patch_area = 1.0 / float(mask_h * mask_w)
    for r in range(mask_h):
        py1 = r / float(mask_h)
        py2 = (r + 1) / float(mask_h)
        for c in range(mask_w):
            px1 = c / float(mask_w)
            px2 = (c + 1) / float(mask_w)
            ix1 = max(px1, x1)
            iy1 = max(py1, y1)
            ix2 = min(px2, x2)
            iy2 = min(py2, y2)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            if inter <= 0.0:
                continue
            union = patch_area + box_area - inter
            if union > 0.0 and inter / union > iou_thresh:
                mask[r, c] = 1.0
    return mask


def rasterize_box_soft_occupancy(
    box: Sequence[float],
    image_size: Tuple[int, int],
    input_size: int,
    mask_h: int,
    mask_w: int,
) -> Optional[np.ndarray]:
    norm = transform_box_for_model_input(box, image_size=image_size, input_size=input_size)
    if norm is None:
        return None
    x1, y1, x2, y2 = norm
    box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if box_area <= 0.0:
        return None
    mask = np.zeros((mask_h, mask_w), dtype=np.float32)
    patch_area = 1.0 / float(mask_h * mask_w)
    for r in range(mask_h):
        py1 = r / float(mask_h)
        py2 = (r + 1) / float(mask_h)
        for c in range(mask_w):
            px1 = c / float(mask_w)
            px2 = (c + 1) / float(mask_w)
            ix1 = max(px1, x1)
            iy1 = max(py1, y1)
            ix2 = min(px2, x2)
            iy2 = min(py2, y2)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            if inter > 0.0:
                mask[r, c] = float(np.clip(inter / patch_area, 0.0, 1.0))
    return mask


def rasterize_box_target(
    box: Sequence[float],
    image_size: Tuple[int, int],
    cfg: Config,
) -> Optional[np.ndarray]:
    if cfg.spatial_target_mode == "hard_iou":
        return rasterize_box_iou(
            box,
            image_size=image_size,
            input_size=cfg.input_size,
            mask_h=cfg.mask_h,
            mask_w=cfg.mask_w,
            iou_thresh=cfg.patch_iou_thresh,
        )
    if cfg.spatial_target_mode == "soft_box":
        return rasterize_box_soft_occupancy(
            box,
            image_size=image_size,
            input_size=cfg.input_size,
            mask_h=cfg.mask_h,
            mask_w=cfg.mask_w,
        )
    raise ValueError(f"Unsupported spatial_target_mode: {cfg.spatial_target_mode}")


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
        annotation_index = (
            int(dataset.sample_indices[sample_index])
            if dataset.sample_indices is not None
            else sample_index
        )
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
        annotation_index = (
            int(dataset.sample_indices[sample_index])
            if dataset.sample_indices is not None
            else sample_index
        )
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


class ResNet50Conv45(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        conv4 = self.layer3(x)
        conv5 = self.layer4(conv4)
        return {"conv4": conv4, "conv5": conv5}


class SharedConceptHead(nn.Module):
    def __init__(self, n_concepts: int, spatial_stage: str) -> None:
        super().__init__()
        in_channels = 1024 if spatial_stage == "conv4" else 2048
        self.spatial_stage = spatial_stage
        self.spatial = nn.Conv2d(in_channels, n_concepts, kernel_size=1, bias=True)

    def forward(self, feats: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        spatial_maps = self.spatial(feats[self.spatial_stage])
        pooled = F.adaptive_avg_pool2d(spatial_maps, 1).flatten(1)
        return {
            "global_logits": pooled,
            "spatial_logits": torch.zeros_like(pooled),
            "spatial_maps": spatial_maps,
            "final_logits": pooled,
        }


def pool_residual_spatial_logits(spatial_maps: torch.Tensor, pooling: str) -> torch.Tensor:
    flat = spatial_maps.flatten(2)
    if pooling == "avg":
        return flat.mean(dim=-1)
    if pooling == "lse":
        # Match CUB SAVLG's normalized LSE pooling: constant maps keep the
        # same logit scale, while localized peaks still influence the residual.
        num_patches = flat.shape[-1]
        return torch.logsumexp(flat, dim=-1) - math.log(max(num_patches, 1))
    raise ValueError(f"Unsupported residual spatial pooling mode: {pooling}")


class DualBranchConceptHead(nn.Module):
    def __init__(
        self,
        n_concepts: int,
        spatial_stage: str,
        residual_alpha: float,
        residual_spatial_pooling: str,
        learn_spatial_residual_scale: bool = False,
    ) -> None:
        super().__init__()
        in_channels = 1024 if spatial_stage == "conv4" else 2048
        self.spatial_stage = spatial_stage
        self.residual_alpha = residual_alpha
        self.residual_spatial_pooling = residual_spatial_pooling
        self.log_spatial_scale = (
            nn.Parameter(torch.zeros(())) if learn_spatial_residual_scale else None
        )
        self.global_head = nn.Linear(2048, n_concepts, bias=True)
        self.spatial = nn.Conv2d(in_channels, n_concepts, kernel_size=1, bias=True)

    def forward(self, feats: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        global_logits = self.global_head(feats["conv5"].mean(dim=(2, 3)))
        spatial_maps = self.spatial(feats[self.spatial_stage])
        spatial_logits = pool_residual_spatial_logits(spatial_maps, self.residual_spatial_pooling)
        spatial_scale = 1.0 if self.log_spatial_scale is None else torch.exp(self.log_spatial_scale)
        final_logits = global_logits + self.residual_alpha * spatial_scale * spatial_logits
        return {
            "global_logits": global_logits,
            "spatial_logits": spatial_logits,
            "spatial_maps": spatial_maps,
            "final_logits": final_logits,
        }


class MultiScaleDualBranchConceptHead(nn.Module):
    def __init__(
        self,
        n_concepts: int,
        residual_alpha: float,
        residual_spatial_pooling: str,
        learn_spatial_residual_scale: bool = False,
        fusion_dim: int = 2048,
    ) -> None:
        super().__init__()
        self.residual_alpha = residual_alpha
        self.residual_spatial_pooling = residual_spatial_pooling
        self.log_spatial_scale = (
            nn.Parameter(torch.zeros(())) if learn_spatial_residual_scale else None
        )
        self.global_head = nn.Linear(2048, n_concepts, bias=True)
        self.conv4_proj = nn.Conv2d(1024, fusion_dim, kernel_size=1, bias=False)
        self.conv5_proj = nn.Conv2d(2048, fusion_dim, kernel_size=1, bias=False)
        self.spatial = nn.Conv2d(fusion_dim, n_concepts, kernel_size=1, bias=True)

    def forward(self, feats: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        global_logits = self.global_head(feats["conv5"].mean(dim=(2, 3)))
        conv5_up = F.interpolate(
            self.conv5_proj(feats["conv5"]),
            size=feats["conv4"].shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        fused = F.relu(self.conv4_proj(feats["conv4"]) + conv5_up, inplace=False)
        spatial_maps = self.spatial(fused)
        spatial_logits = pool_residual_spatial_logits(spatial_maps, self.residual_spatial_pooling)
        spatial_scale = 1.0 if self.log_spatial_scale is None else torch.exp(self.log_spatial_scale)
        final_logits = global_logits + self.residual_alpha * spatial_scale * spatial_logits
        return {
            "global_logits": global_logits,
            "spatial_logits": spatial_logits,
            "spatial_maps": spatial_maps,
            "final_logits": final_logits,
        }


def build_model(cfg: Config, n_concepts: int) -> Tuple[nn.Module, nn.Module]:
    backbone = ResNet50Conv45().to(cfg.device)
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    backbone.eval()
    if cfg.channels_last:
        backbone.to(memory_format=torch.channels_last)
    if cfg.spatial_branch_mode == "multiscale_conv45":
        if cfg.branch_arch != "dual":
            raise ValueError("multiscale_conv45 requires branch_arch=dual")
        head = MultiScaleDualBranchConceptHead(
            n_concepts=n_concepts,
            residual_alpha=cfg.residual_alpha,
            residual_spatial_pooling=getattr(cfg, "residual_spatial_pooling", "avg"),
            learn_spatial_residual_scale=bool(getattr(cfg, "learn_spatial_residual_scale", False)),
        )
    elif cfg.branch_arch == "dual":
        head = DualBranchConceptHead(
            n_concepts=n_concepts,
            spatial_stage=cfg.spatial_stage,
            residual_alpha=cfg.residual_alpha,
            residual_spatial_pooling=getattr(cfg, "residual_spatial_pooling", "avg"),
            learn_spatial_residual_scale=bool(getattr(cfg, "learn_spatial_residual_scale", False)),
        )
    else:
        head = SharedConceptHead(n_concepts=n_concepts, spatial_stage=cfg.spatial_stage)
    head = head.to(cfg.device)
    if cfg.channels_last:
        head.to(memory_format=torch.channels_last)
    return backbone, head


def init_global_head_from_vlg(head: nn.Module, cfg: Config, concepts: Sequence[str]) -> None:
    if not cfg.vlg_init_path:
        return
    if not hasattr(head, "global_head"):
        print(
            f"[vlg_init] skipping: head type {type(head).__name__} has no global_head",
            flush=True,
        )
        return

    vlg_state = torch.load(cfg.vlg_init_path, map_location="cpu")
    if isinstance(vlg_state, dict) and "state_dict" in vlg_state and isinstance(vlg_state["state_dict"], dict):
        vlg_state = vlg_state["state_dict"]
    weight = vlg_state.get("model.0.weight")
    bias = vlg_state.get("model.0.bias")
    if weight is None or bias is None:
        raise KeyError(f"Could not find VLG weights in {cfg.vlg_init_path}")

    vlg_concepts = load_concepts(cfg.vlg_concepts_path)
    if len(vlg_concepts) != int(weight.shape[0]):
        raise ValueError(
            f"VLG concept count mismatch: {len(vlg_concepts)} concepts for weight rows {int(weight.shape[0])}"
        )
    vlg_concept_to_idx = {concept: idx for idx, concept in enumerate(vlg_concepts)}
    target_head = head.global_head
    if tuple(weight.shape) != tuple(target_head.weight.shape):
        if int(weight.shape[1]) != int(target_head.weight.shape[1]):
            raise ValueError(
                f"VLG init feature dim mismatch: {tuple(weight.shape)} vs {tuple(target_head.weight.shape)}"
            )
    matched = 0
    with torch.no_grad():
        for our_idx, concept in enumerate(concepts):
            vlg_idx = vlg_concept_to_idx.get(concept)
            if vlg_idx is None:
                continue
            target_head.weight[our_idx].copy_(weight[vlg_idx])
            target_head.bias[our_idx].copy_(bias[vlg_idx])
            matched += 1
    print(f"[vlg_init] matched {matched}/{len(concepts)} concepts from {cfg.vlg_init_path}", flush=True)

    if cfg.freeze_global_head:
        for parameter in target_head.parameters():
            parameter.requires_grad = False
        print("[vlg_init] global head frozen", flush=True)


def prepare_images(images: torch.Tensor, cfg: Config) -> torch.Tensor:
    if cfg.channels_last:
        images = images.contiguous(memory_format=torch.channels_last)
    return images.to(cfg.device, non_blocking=cfg.pin_memory)


def compute_losses(
    outputs: Dict[str, torch.Tensor],
    global_targets: torch.Tensor,
    mask_indices: torch.Tensor,
    mask_targets: torch.Tensor,
    mask_valid: torch.Tensor,
    cfg: Config,
) -> Dict[str, torch.Tensor]:
    final_logits = outputs["final_logits"]
    spatial_maps = F.interpolate(
        outputs["spatial_maps"],
        size=mask_targets.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )

    global_loss_raw = F.binary_cross_entropy_with_logits(final_logits, global_targets, reduction="none")
    global_pos_w = torch.where(
        global_targets > 0.5,
        torch.full_like(global_targets, float(cfg.global_pos_weight)),
        torch.ones_like(global_targets),
    )
    loss_global = (global_loss_raw * global_pos_w).sum() / torch.clamp(global_pos_w.sum(), min=1.0)

    per_sample_mask_losses: List[torch.Tensor] = []
    per_sample_dice_losses: List[torch.Tensor] = []
    for batch_index in range(spatial_maps.shape[0]):
        valid = mask_valid[batch_index]
        if not bool(valid.any()):
            continue
        concept_ids = mask_indices[batch_index][valid]
        pred = spatial_maps[batch_index].index_select(0, concept_ids)
        tgt = mask_targets[batch_index][valid].to(pred.dtype)

        if cfg.spatial_loss_mode == "bce":
            bce_raw = F.binary_cross_entropy_with_logits(pred, tgt, reduction="none")
            patch_pos_w = torch.where(
                tgt > 0.5,
                torch.full_like(tgt, float(cfg.patch_pos_weight)),
                torch.ones_like(tgt),
            )
            patch_pos_w_flat = patch_pos_w.flatten(1)
            per_concept_mask = (bce_raw.flatten(1) * patch_pos_w_flat).sum(dim=1) / torch.clamp(
                patch_pos_w_flat.sum(dim=1),
                min=1.0,
            )
        elif cfg.spatial_loss_mode == "soft_align":
            pred_flat = pred.flatten(1).float()
            target_mass = mask_targets[batch_index][valid].flatten(1).float().clamp(min=0.0)
            target_mass_sum = target_mass.sum(dim=1, keepdim=True)
            valid_targets = target_mass_sum.squeeze(1) > 0.0
            per_concept_mask = torch.zeros((pred_flat.shape[0],), device=pred.device, dtype=torch.float32)
            if bool(valid_targets.any()):
                target_dist = torch.zeros_like(target_mass)
                target_dist[valid_targets] = target_mass[valid_targets] / torch.clamp(
                    target_mass_sum[valid_targets],
                    min=1e-6,
                )
                pred_log_dist = F.log_softmax(pred_flat[valid_targets], dim=1)
                per_concept_mask[valid_targets] = F.kl_div(
                    pred_log_dist,
                    target_dist[valid_targets],
                    reduction="none",
                ).sum(dim=1)
        else:
            raise ValueError(f"Unsupported spatial_loss_mode: {cfg.spatial_loss_mode}")
        per_sample_mask_losses.append(per_concept_mask.mean())

        if cfg.loss_dice_w > 0.0:
            pred_prob = torch.sigmoid(pred).flatten(1)
            tgt_flat = tgt.flatten(1)
            intersection = (pred_prob * tgt_flat).sum(dim=1)
            denom = pred_prob.sum(dim=1) + tgt_flat.sum(dim=1)
            per_concept_dice = 1.0 - ((2.0 * intersection + 1e-6) / (denom + 1e-6))
            per_sample_dice_losses.append(per_concept_dice.mean())

    loss_mask = (
        torch.stack(per_sample_mask_losses).mean() if per_sample_mask_losses else spatial_maps.sum() * 0.0
    )
    loss_dice = (
        torch.stack(per_sample_dice_losses).mean() if per_sample_dice_losses else spatial_maps.sum() * 0.0
    )
    total = cfg.loss_global_w * loss_global + cfg.loss_mask_w * loss_mask + cfg.loss_dice_w * loss_dice
    return {
        "total": total,
        "global": loss_global.detach(),
        "mask": loss_mask.detach(),
        "dice": loss_dice.detach(),
    }


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


def train_one_epoch(
    backbone: nn.Module,
    head: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    cfg: Config,
    concept_to_idx: Dict[str, int],
    n_concepts: int,
    epoch: int,
) -> Dict[str, float]:
    head.train()
    totals = {"total": 0.0, "global": 0.0, "mask": 0.0, "dice": 0.0, "count": 0}
    start_time = time.perf_counter()
    reset_cuda_peak_stats_if_needed(cfg)
    for step, batch in enumerate(loader, start=1):
        images = prepare_images(batch["images"], cfg)
        if "global_targets" in batch:
            global_targets, idx_pad, mask_pad, valid_pad = batch_targets_to_device(batch, cfg)
        else:
            global_targets, idx_pad, mask_pad, valid_pad = build_gdino_targets(
                batch["annotations"],
                batch["image_sizes"],
                concept_to_idx,
                n_concepts,
                cfg,
                cfg.device,
            )
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            feats = backbone(images)
        with autocast_context(cfg):
            outputs = head(feats)
            losses = compute_losses(outputs, global_targets, idx_pad, mask_pad, valid_pad, cfg)
            loss = losses["total"]
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_size = int(images.shape[0])
        totals["total"] += float(loss.detach().item()) * batch_size
        totals["global"] += float(losses["global"].item()) * batch_size
        totals["mask"] += float(losses["mask"].item()) * batch_size
        totals["dice"] += float(losses["dice"].item()) * batch_size
        totals["count"] += batch_size
        if step % cfg.log_every == 0:
            elapsed = time.perf_counter() - start_time
            images_per_second = totals["count"] / max(elapsed, 1e-6)
            print(
                f"[train] epoch={epoch} step={step}/{len(loader)} "
                f"loss={totals['total']/totals['count']:.4f} "
                f"global={totals['global']/totals['count']:.4f} "
                f"mask={totals['mask']/totals['count']:.4f} "
                f"ips={images_per_second:.2f}",
                flush=True,
            )
    count = max(totals["count"], 1)
    elapsed = time.perf_counter() - start_time
    metrics = {
        "loss": totals["total"] / count,
        "loss_global": totals["global"] / count,
        "loss_mask": totals["mask"] / count,
        "loss_dice": totals["dice"] / count,
        "images_per_second": totals["count"] / max(elapsed, 1e-6),
        "elapsed_sec": elapsed,
    }
    metrics.update(cuda_peak_stats_mb(cfg))
    return metrics


@torch.no_grad()
def evaluate_one_epoch(
    backbone: nn.Module,
    head: nn.Module,
    loader: DataLoader,
    cfg: Config,
    concept_to_idx: Dict[str, int],
    n_concepts: int,
    split_name: str,
) -> Dict[str, float]:
    head.eval()
    totals = {"total": 0.0, "global": 0.0, "mask": 0.0, "dice": 0.0, "count": 0}
    start_time = time.perf_counter()
    reset_cuda_peak_stats_if_needed(cfg)
    for step, batch in enumerate(loader, start=1):
        images = prepare_images(batch["images"], cfg)
        if "global_targets" in batch:
            global_targets, idx_pad, mask_pad, valid_pad = batch_targets_to_device(batch, cfg)
        else:
            global_targets, idx_pad, mask_pad, valid_pad = build_gdino_targets(
                batch["annotations"],
                batch["image_sizes"],
                concept_to_idx,
                n_concepts,
                cfg,
                cfg.device,
            )
        with autocast_context(cfg):
            feats = backbone(images)
            outputs = head(feats)
            losses = compute_losses(outputs, global_targets, idx_pad, mask_pad, valid_pad, cfg)
        batch_size = int(images.shape[0])
        totals["total"] += float(losses["total"].item()) * batch_size
        totals["global"] += float(losses["global"].item()) * batch_size
        totals["mask"] += float(losses["mask"].item()) * batch_size
        totals["dice"] += float(losses["dice"].item()) * batch_size
        totals["count"] += batch_size
        if step % cfg.log_every == 0:
            elapsed = time.perf_counter() - start_time
            images_per_second = totals["count"] / max(elapsed, 1e-6)
            print(
                f"[{split_name}] step={step}/{len(loader)} "
                f"loss={totals['total']/totals['count']:.4f} "
                f"ips={images_per_second:.2f}",
                flush=True,
            )
    count = max(totals["count"], 1)
    elapsed = time.perf_counter() - start_time
    metrics = {
        "loss": totals["total"] / count,
        "loss_global": totals["global"] / count,
        "loss_mask": totals["mask"] / count,
        "loss_dice": totals["dice"] / count,
        "images_per_second": totals["count"] / max(elapsed, 1e-6),
        "elapsed_sec": elapsed,
    }
    metrics.update(cuda_peak_stats_mb(cfg))
    return metrics


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


@torch.no_grad()
def extract_concept_features_to_memmap(
    backbone: nn.Module,
    head: nn.Module,
    loader: DataLoader,
    cfg: Config,
    split_name: str,
    output_dir: Path,
) -> Tuple[Path, Path, Dict[str, Any]]:
    head.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    total_examples = len(loader.dataset)
    target_path = output_dir / f"{split_name}_targets.npy"
    target_memmap = np.lib.format.open_memmap(target_path, mode="w+", dtype=np.int64, shape=(total_examples,))
    feature_path: Optional[Path] = None
    feature_memmap: Optional[np.memmap] = None
    offset = 0
    start_time = time.perf_counter()
    reset_cuda_peak_stats_if_needed(cfg)
    for step, batch in enumerate(loader, start=1):
        images = prepare_images(batch["images"], cfg)
        with autocast_context(cfg):
            feats = backbone(images)
            outputs = head(feats)
        batch_features = outputs["final_logits"].detach().float().cpu().numpy()
        batch_targets = batch["class_ids"].detach().cpu().numpy().astype(np.int64, copy=False)
        batch_size = int(batch_features.shape[0])
        if feature_memmap is None:
            feature_path = output_dir / f"{split_name}_features.npy"
            feature_memmap = np.lib.format.open_memmap(
                feature_path,
                mode="w+",
                dtype=feature_storage_dtype(cfg),
                shape=(total_examples, int(batch_features.shape[1])),
            )
        feature_memmap[offset : offset + batch_size] = batch_features.astype(feature_memmap.dtype, copy=False)
        target_memmap[offset : offset + batch_size] = batch_targets
        offset += batch_size
        if step % 10 == 0:
            feature_memmap.flush()
            target_memmap.flush()
        del batch_features, batch_targets, feats, outputs, images
        if step % cfg.log_every == 0:
            elapsed = time.perf_counter() - start_time
            print(
                f"[{split_name}_features] step={step}/{len(loader)} "
                f"n={offset} ips={offset / max(elapsed, 1e-6):.2f}",
                flush=True,
            )
    if feature_memmap is None or feature_path is None:
        raise RuntimeError(f"No features extracted for split {split_name}")
    feature_memmap.flush()
    target_memmap.flush()
    elapsed = time.perf_counter() - start_time
    summary = {
        "stage": f"{split_name}_feature_extraction_summary",
        "n_examples": offset,
        "n_features": int(feature_memmap.shape[1]),
        "images_per_second": offset / max(elapsed, 1e-6),
        "elapsed_sec": elapsed,
        "feature_path": str(feature_path),
        "target_path": str(target_path),
        **cuda_peak_stats_mb(cfg),
    }
    print(json.dumps(summary), flush=True)
    return feature_path, target_path, summary


def compute_feature_stats_memmap(
    feature_path: Path,
    cfg: Config,
    chunk_size: int = 8192,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    features = np.load(feature_path, mmap_mode="r")
    n_examples, n_features = int(features.shape[0]), int(features.shape[1])
    start_time = time.perf_counter()
    sum_vec = np.zeros((n_features,), dtype=np.float64)
    sum_sq_vec = np.zeros((n_features,), dtype=np.float64)
    for start in range(0, n_examples, chunk_size):
        end = min(start + chunk_size, n_examples)
        batch = np.asarray(features[start:end], dtype=np.float32)
        sum_vec += batch.sum(axis=0, dtype=np.float64)
        sum_sq_vec += np.square(batch, dtype=np.float32).sum(axis=0, dtype=np.float64)
    mean = sum_vec / max(n_examples, 1)
    if n_examples > 1:
        var = (sum_sq_vec - (sum_vec * sum_vec) / n_examples) / (n_examples - 1)
    else:
        var = np.ones_like(mean)
    var = np.maximum(var, 1e-6)
    std = np.sqrt(var).astype(np.float32)
    mean = mean.astype(np.float32)
    summary = {
        "stage": "train_feature_normalization_summary",
        "n_examples": n_examples,
        "n_features": n_features,
        "elapsed_sec": time.perf_counter() - start_time,
    }
    return torch.from_numpy(mean), torch.from_numpy(std), summary


def copy_feature_artifacts_to_persist(
    source_dir: Path,
    persist_dir: Path,
    filenames: Sequence[str],
) -> Dict[str, Any]:
    persist_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()
    total_bytes = 0
    copied = []
    for name in filenames:
        src = source_dir / name
        dst = persist_dir / name
        if not src.exists():
            raise FileNotFoundError(f"Missing feature artifact for persistent copy: {src}")
        shutil.copyfile(src, dst)
        total_bytes += int(dst.stat().st_size)
        copied.append(str(dst))
    return {
        "stage": "feature_persist_copy_summary",
        "elapsed_sec": time.perf_counter() - start_time,
        "total_bytes": total_bytes,
        "files": copied,
    }


def topk_accuracy(logits: torch.Tensor, targets: torch.Tensor, k: int) -> float:
    k = min(k, int(logits.shape[1]))
    topk = logits.topk(k, dim=1).indices
    correct = topk.eq(targets.unsqueeze(1)).any(dim=1)
    return float(correct.float().mean().item())


@torch.no_grad()
def evaluate_final_layer(
    linear: nn.Linear,
    loader: DataLoader,
    device: str,
) -> Dict[str, float]:
    linear.eval()
    total_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    total_examples = 0
    for batch in loader:
        features, targets = batch[0].to(device), batch[1].to(device)
        logits = linear(features)
        batch_size = int(targets.shape[0])
        total_loss += float(F.cross_entropy(logits, targets, reduction="sum").item())
        total_top1 += topk_accuracy(logits, targets, k=1) * batch_size
        total_top5 += topk_accuracy(logits, targets, k=5) * batch_size
        total_examples += batch_size
    count = max(total_examples, 1)
    return {
        "loss": total_loss / count,
        "top1": total_top1 / count,
        "top5": total_top5 / count,
        "n": total_examples,
    }


def train_sparse_final_layer(
    train_feature_path: Path,
    train_target_path: Path,
    val_feature_path: Path,
    val_target_path: Path,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
    cfg: Config,
    n_classes: int,
    run_dir: Path,
) -> Dict[str, Any]:
    feature_mean_np = feature_mean.cpu().numpy()
    feature_std_np = feature_std.cpu().numpy()
    train_dataset = MemmapFeatureDataset(
        train_feature_path,
        train_target_path,
        mean=feature_mean_np,
        std=feature_std_np,
        include_index=True,
    )
    train_eval_dataset = MemmapFeatureDataset(
        train_feature_path,
        train_target_path,
        mean=feature_mean_np,
        std=feature_std_np,
        include_index=False,
    )
    val_dataset = MemmapFeatureDataset(
        val_feature_path,
        val_target_path,
        mean=feature_mean_np,
        std=feature_std_np,
        include_index=False,
    )
    train_loader_kwargs: Dict[str, Any] = {
        "batch_size": cfg.saga_batch_size,
        "shuffle": True,
        "num_workers": cfg.saga_workers,
        "pin_memory": cfg.pin_memory,
        "drop_last": False,
    }
    eval_loader_kwargs: Dict[str, Any] = {
        "batch_size": cfg.saga_batch_size,
        "shuffle": False,
        "num_workers": cfg.saga_workers,
        "pin_memory": cfg.pin_memory,
        "drop_last": False,
    }
    if cfg.saga_workers > 0:
        train_loader_kwargs["persistent_workers"] = True
        train_loader_kwargs["prefetch_factor"] = cfg.saga_prefetch_factor
        eval_loader_kwargs["persistent_workers"] = True
        eval_loader_kwargs["prefetch_factor"] = cfg.saga_prefetch_factor

    train_loader = DataLoader(
        train_dataset,
        **train_loader_kwargs,
    )
    train_eval_loader = DataLoader(
        train_eval_dataset,
        **eval_loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        **eval_loader_kwargs,
    )

    linear = nn.Linear(int(train_dataset.features.shape[1]), int(n_classes), bias=True).to(cfg.device)
    linear.weight.data.zero_()
    linear.bias.data.zero_()

    metadata = {"max_reg": {"nongrouped": cfg.saga_lam}}
    reset_cuda_peak_stats_if_needed(cfg)
    start_time = time.perf_counter()
    output = glm_saga(
        linear,
        train_loader,
        cfg.saga_step_size,
        cfg.saga_n_iters,
        0.99,
        table_device=cfg.saga_table_device,
        epsilon=1,
        k=1,
        val_loader=val_loader,
        do_zero=False,
        metadata=metadata,
        n_ex=len(train_dataset),
        n_classes=n_classes,
        verbose=cfg.saga_verbose_every,
    )
    best = output["best"]
    linear.load_state_dict({"weight": best["weight"], "bias": best["bias"]})

    train_metrics = evaluate_final_layer(linear, train_eval_loader, cfg.device)
    val_metrics = evaluate_final_layer(linear, val_loader, cfg.device)

    payload = {
        "best": {
            "lam": float(best["lam"]),
            "lr": float(best["lr"]),
            "alpha": float(best["alpha"]),
            "time": float(best["time"]),
            "metrics": best["metrics"],
        },
        "train": train_metrics,
        "val": val_metrics,
        "nnz": int((best["weight"].abs() > 1e-5).sum().item()),
        "total": int(best["weight"].numel()),
        "elapsed_sec": time.perf_counter() - start_time,
    }
    payload.update(cuda_peak_stats_mb(cfg))

    torch.save(
        {
            "weight": best["weight"],
            "bias": best["bias"],
        },
        run_dir / "final_layer_glm_saga.pt",
    )
    (run_dir / "final_layer_summary.json").write_text(json.dumps(payload, indent=2))
    return payload


def train_dense_final_layer(
    train_feature_path: Path,
    train_target_path: Path,
    val_feature_path: Path,
    val_target_path: Path,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
    cfg: Config,
    n_classes: int,
    run_dir: Path,
) -> Dict[str, Any]:
    feature_mean_np = feature_mean.cpu().numpy()
    feature_std_np = feature_std.cpu().numpy()
    train_dataset = MemmapFeatureDataset(
        train_feature_path,
        train_target_path,
        mean=feature_mean_np,
        std=feature_std_np,
        include_index=False,
    )
    val_dataset = MemmapFeatureDataset(
        val_feature_path,
        val_target_path,
        mean=feature_mean_np,
        std=feature_std_np,
        include_index=False,
    )
    train_loader_kwargs: Dict[str, Any] = {
        "batch_size": cfg.saga_batch_size,
        "shuffle": True,
        "num_workers": cfg.saga_workers,
        "pin_memory": cfg.pin_memory,
        "drop_last": False,
    }
    eval_loader_kwargs: Dict[str, Any] = {
        "batch_size": cfg.saga_batch_size,
        "shuffle": False,
        "num_workers": cfg.saga_workers,
        "pin_memory": cfg.pin_memory,
        "drop_last": False,
    }
    if cfg.saga_workers > 0:
        train_loader_kwargs["persistent_workers"] = True
        train_loader_kwargs["prefetch_factor"] = cfg.saga_prefetch_factor
        eval_loader_kwargs["persistent_workers"] = True
        eval_loader_kwargs["prefetch_factor"] = cfg.saga_prefetch_factor

    train_loader = DataLoader(train_dataset, **train_loader_kwargs)
    val_loader = DataLoader(val_dataset, **eval_loader_kwargs)

    linear = nn.Linear(int(train_dataset.features.shape[1]), int(n_classes), bias=True).to(cfg.device)
    optimizer = torch.optim.Adam(linear.parameters(), lr=cfg.dense_lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    best_val_loss = float("inf")
    best_state = None
    history: List[Dict[str, Any]] = []
    reset_cuda_peak_stats_if_needed(cfg)
    start_time = time.perf_counter()

    for epoch_idx in range(cfg.dense_n_iters):
        linear.train()
        total_train_loss = 0.0
        total_examples = 0
        for batch in train_loader:
            features, targets = batch[0].to(cfg.device), batch[1].to(cfg.device)
            optimizer.zero_grad(set_to_none=True)
            logits = linear(features)
            loss = F.cross_entropy(logits, targets, reduction="mean")
            loss.backward()
            optimizer.step()
            batch_size = int(targets.shape[0])
            total_train_loss += float(loss.item()) * batch_size
            total_examples += batch_size

        scheduler.step()
        train_metrics = evaluate_final_layer(linear, train_loader, cfg.device)
        val_metrics = evaluate_final_layer(linear, val_loader, cfg.device)
        epoch_payload = {
            "epoch": epoch_idx + 1,
            "train": train_metrics,
            "val": val_metrics,
            "lr": float(scheduler.get_last_lr()[0]),
        }
        history.append(epoch_payload)
        print(
            f"[dense_final] epoch={epoch_idx + 1} "
            f"train_top1={train_metrics['top1']:.4f} "
            f"val_top1={val_metrics['top1']:.4f} "
            f"val_loss={val_metrics['loss']:.4f}"
        )
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = {
                "weight": linear.weight.detach().cpu().clone(),
                "bias": linear.bias.detach().cpu().clone(),
                "epoch": epoch_idx + 1,
                "train": train_metrics,
                "val": val_metrics,
            }

    assert best_state is not None
    payload = {
        "best_epoch": int(best_state["epoch"]),
        "best_val_loss": float(best_val_loss),
        "train": best_state["train"],
        "val": best_state["val"],
        "history": history,
        "nnz": int((best_state["weight"].abs() > 1e-5).sum().item()),
        "total": int(best_state["weight"].numel()),
        "elapsed_sec": time.perf_counter() - start_time,
        "dense_lr": float(cfg.dense_lr),
        "dense_n_iters": int(cfg.dense_n_iters),
    }
    payload.update(cuda_peak_stats_mb(cfg))

    torch.save(
        {
            "weight": best_state["weight"],
            "bias": best_state["bias"],
            "epoch": best_state["epoch"],
        },
        run_dir / "final_layer_dense.pt",
    )
    (run_dir / "final_layer_dense_summary.json").write_text(json.dumps(payload, indent=2))
    return payload


def profile_pipeline(cfg: Config) -> Dict[str, Any]:
    concepts = load_run_concepts(cfg)
    dataset = SafeImageFolderWithAnnotations(
        root=cfg.train_root,
        annotation_dir=cfg.annotation_dir,
        concepts=concepts,
        input_size=cfg.input_size,
        min_image_bytes=cfg.min_image_bytes,
        split="train",
        manifest=cfg.train_manifest,
        train_random_transforms=cfg.train_random_transforms,
    )
    dataset.attach_precomputed_targets(cfg.precomputed_target_dir, cfg)
    indices = select_subset_indices(
        dataset,
        list(range(len(dataset))),
        max_images=cfg.max_train_images,
        seed=cfg.seed,
        stratify=True,
    )
    dataset_view = DatasetView(dataset, indices)
    concept_filter_summary = apply_count_concept_filter(cfg, dataset_view, [dataset_view])
    concepts = list(dataset_view.concepts)
    loader = build_loader(dataset_view, cfg, shuffle=False, drop_last=False)
    backbone, head = build_model(cfg, n_concepts=len(concepts))
    init_global_head_from_vlg(head, cfg, concepts)
    iterator = iter(loader)
    data_seconds = 0.0
    h2d_seconds = 0.0
    target_seconds = 0.0
    forward_seconds = 0.0
    profiled_steps = 0
    first_shapes: Optional[Dict[str, Any]] = None
    total_steps = cfg.warmup_steps + cfg.profile_steps
    for step in range(total_steps):
        t0 = time.perf_counter()
        batch = next(iterator)
        t1 = time.perf_counter()
        images = prepare_images(batch["images"], cfg)
        t2 = time.perf_counter()
        if "global_targets" in batch:
            global_targets, idx_pad, mask_pad, valid_pad = batch_targets_to_device(batch, cfg)
        else:
            global_targets, idx_pad, mask_pad, valid_pad = build_gdino_targets(
                batch["annotations"],
                batch["image_sizes"],
                dataset.concept_to_idx,
                len(concepts),
                cfg,
                cfg.device,
            )
        t3 = time.perf_counter()
        with torch.no_grad():
            with autocast_context(cfg):
                feats = backbone(images)
                outputs = head(feats)
                _ = compute_losses(outputs, global_targets, idx_pad, mask_pad, valid_pad, cfg)
        if str(cfg.device).startswith("cuda"):
            torch.cuda.synchronize()
        t4 = time.perf_counter()
        if first_shapes is None:
            first_shapes = {
                "images": list(batch["images"].shape),
                "conv4": list(feats["conv4"].shape),
                "conv5": list(feats["conv5"].shape),
                "final_logits": list(outputs["final_logits"].shape),
                "idx_pad": list(idx_pad.shape),
                "mask_pad": list(mask_pad.shape),
            }
        if step >= cfg.warmup_steps:
            profiled_steps += 1
            data_seconds += t1 - t0
            h2d_seconds += t2 - t1
            target_seconds += t3 - t2
            forward_seconds += t4 - t3
    total_images = profiled_steps * cfg.batch_size
    total_seconds = data_seconds + h2d_seconds + target_seconds + forward_seconds
    return {
        "config": asdict(cfg),
        "n_concepts": len(concepts),
        "concept_filter": concept_filter_summary,
        "profiled_steps": profiled_steps,
        "images_profiled": total_images,
        "timing_seconds": {
            "data": data_seconds,
            "h2d": h2d_seconds,
            "targets": target_seconds,
            "forward_plus_loss": forward_seconds,
            "total": total_seconds,
        },
        "throughput": {
            "images_per_second_total": total_images / max(total_seconds, 1e-6),
            "images_per_second_compute": total_images / max(forward_seconds, 1e-6),
        },
        "shapes": first_shapes,
    }


def run_precompute_targets(cfg: Config) -> Dict[str, Any]:
    concepts = load_concepts(cfg.concept_file)
    train_dataset = SafeImageFolderWithAnnotations(
        root=cfg.train_root,
        annotation_dir=cfg.annotation_dir,
        concepts=concepts,
        input_size=cfg.input_size,
        min_image_bytes=cfg.min_image_bytes,
        split="train",
        manifest=cfg.train_manifest,
        train_random_transforms=cfg.train_random_transforms,
    )
    val_dataset: Optional[SafeImageFolderWithAnnotations] = None
    if cfg.val_root:
        val_dataset = SafeImageFolderWithAnnotations(
            root=cfg.val_root,
            annotation_dir=cfg.annotation_dir,
            concepts=concepts,
            input_size=cfg.input_size,
            min_image_bytes=cfg.min_image_bytes,
            split="val",
            train_random_transforms=cfg.train_random_transforms,
        )
    datasets: List[Dataset] = [train_dataset]
    if val_dataset is not None:
        datasets.append(val_dataset)
    concept_filter_summary = apply_count_concept_filter(cfg, train_dataset, datasets)

    output_root = Path(cfg.precomputed_target_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    train_summary = precompute_target_store(train_dataset, output_root, cfg)
    result: Dict[str, Any] = {
        "mode": "precompute_targets",
        "output_root": str(output_root),
        "concept_filter": concept_filter_summary,
        "n_concepts": len(train_dataset.concepts),
        "train": train_summary,
    }
    if val_dataset is not None:
        result["val"] = precompute_target_store(val_dataset, output_root, cfg)
    (output_root / "precompute_summary.json").write_text(json.dumps(result, indent=2))
    (output_root / "concepts.txt").write_text("\n".join(train_dataset.concepts))
    if concept_filter_summary is not None:
        (output_root / "concept_filter_summary.json").write_text(json.dumps(concept_filter_summary, indent=2))
    return result


def run_training(cfg: Config) -> Dict[str, Any]:
    concepts = load_run_concepts(cfg)
    train_dataset_full = SafeImageFolderWithAnnotations(
        root=cfg.train_root,
        annotation_dir=cfg.annotation_dir,
        concepts=concepts,
        input_size=cfg.input_size,
        min_image_bytes=cfg.min_image_bytes,
        split="train",
        manifest=cfg.train_manifest,
        train_random_transforms=cfg.train_random_transforms,
    )
    train_dataset_full.attach_precomputed_targets(cfg.precomputed_target_dir, cfg)
    if cfg.val_root:
        val_dataset_full = SafeImageFolderWithAnnotations(
            root=cfg.val_root,
            annotation_dir=cfg.annotation_dir,
            concepts=concepts,
            input_size=cfg.input_size,
            min_image_bytes=cfg.min_image_bytes,
            split="val",
            train_random_transforms=cfg.train_random_transforms,
        )
        val_dataset_full.attach_precomputed_targets(cfg.precomputed_target_dir, cfg)
        train_indices = select_subset_indices(
            train_dataset_full,
            list(range(len(train_dataset_full))),
            max_images=cfg.max_train_images,
            seed=cfg.seed,
            stratify=True,
        )
        val_indices = select_subset_indices(
            val_dataset_full,
            list(range(len(val_dataset_full))),
            max_images=cfg.max_val_images,
            seed=cfg.seed + 1,
            stratify=True,
        )
        train_dataset = DatasetView(train_dataset_full, train_indices)
        val_dataset = DatasetView(val_dataset_full, val_indices)
    else:
        train_dataset, val_dataset = split_train_val(
            train_dataset_full,
            val_split=cfg.val_split,
            max_train_images=cfg.max_train_images,
            max_val_images=cfg.max_val_images,
            seed=cfg.seed,
        )

    concept_filter_summary = apply_count_concept_filter(cfg, train_dataset, [train_dataset, val_dataset])
    concepts = list(train_dataset.concepts)
    train_loader = build_loader(train_dataset, cfg, shuffle=True, drop_last=True)
    val_loader = build_loader(val_dataset, cfg, shuffle=False, drop_last=False)
    backbone, head = build_model(cfg, n_concepts=len(concepts))
    init_global_head_from_vlg(head, cfg, concepts)
    optimizer = make_optimizer(head, cfg)
    scheduler = make_scheduler(optimizer, cfg)
    scaler = None
    if cfg.amp == "fp16" and str(cfg.device).startswith("cuda"):
        scaler = torch.cuda.amp.GradScaler()
    run_dir = build_run_dir(cfg)
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    (run_dir / "concepts.txt").write_text("\n".join(concepts))
    if concept_filter_summary is not None:
        (run_dir / "concept_filter_summary.json").write_text(json.dumps(concept_filter_summary, indent=2))

    best_val = float("inf")
    best_path = run_dir / "concept_head_best.pt"
    history: List[Dict[str, Any]] = []
    for epoch in range(1, cfg.epochs + 1):
        train_metrics = train_one_epoch(
            backbone,
            head,
            train_loader,
            optimizer,
            scaler,
            cfg,
            concept_to_idx=train_dataset.concept_to_idx,
            n_concepts=len(concepts),
            epoch=epoch,
        )
        val_metrics = evaluate_one_epoch(
            backbone,
            head,
            val_loader,
            cfg,
            concept_to_idx=val_dataset.concept_to_idx,
            n_concepts=len(concepts),
            split_name="val",
        )
        summary = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        if scheduler is not None:
            summary["lr"] = float(scheduler.get_last_lr()[0])
        history.append(summary)
        epoch_message = (
            f"[epoch] {epoch} "
            f"train_loss={train_metrics['loss']:.4f} train_ips={train_metrics['images_per_second']:.2f} "
            f"val_loss={val_metrics['loss']:.4f} val_ips={val_metrics['images_per_second']:.2f}"
        )
        if scheduler is not None:
            epoch_message += f" lr={scheduler.get_last_lr()[0]:.6f}"
        print(
            epoch_message,
            flush=True,
        )
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(head.state_dict(), best_path)
        if epoch % cfg.save_every == 0:
            save_checkpoint(run_dir, head, optimizer, epoch, cfg, train_metrics, val_metrics)
        if scheduler is not None:
            scheduler.step()

    final_layer_summary: Optional[Dict[str, Any]] = None
    if not cfg.skip_final_layer:
        feature_dir = Path(cfg.feature_dir).resolve() if cfg.feature_dir else (run_dir / "features")
        persist_feature_dir = run_dir / "features"
        feature_batch_size = max(64, min(cfg.batch_size, 256))
        feature_workers = max(1, min(cfg.workers, 4))
        feature_prefetch = max(1, min(cfg.prefetch_factor, 2))
        feature_train_loader = build_loader(
            train_dataset,
            cfg,
            shuffle=False,
            drop_last=False,
            batch_size=feature_batch_size,
            workers=feature_workers,
            pin_memory=False,
            persistent_workers=False,
            prefetch_factor=feature_prefetch,
        )
        feature_val_loader = build_loader(
            val_dataset,
            cfg,
            shuffle=False,
            drop_last=False,
            batch_size=feature_batch_size,
            workers=feature_workers,
            pin_memory=False,
            persistent_workers=False,
            prefetch_factor=feature_prefetch,
        )
        train_feature_path, train_target_path, train_extract_summary = extract_concept_features_to_memmap(
            backbone, head, feature_train_loader, cfg, split_name="train", output_dir=feature_dir
        )
        val_feature_path, val_target_path, val_extract_summary = extract_concept_features_to_memmap(
            backbone, head, feature_val_loader, cfg, split_name="val", output_dir=feature_dir
        )
        persist_copy_summary: Optional[Dict[str, Any]] = None
        if cfg.persist_feature_copy and feature_dir != persist_feature_dir:
            persist_copy_summary = copy_feature_artifacts_to_persist(
                feature_dir,
                persist_feature_dir,
                [
                    train_feature_path.name,
                    train_target_path.name,
                    val_feature_path.name,
                    val_target_path.name,
                ],
            )
            print(json.dumps(persist_copy_summary), flush=True)
        feature_mean, feature_std, norm_summary = compute_feature_stats_memmap(
            train_feature_path, cfg
        )
        normalization_payload = {
            "mean": feature_mean,
            "std": feature_std,
            "train_extraction": train_extract_summary,
            "val_extraction": val_extract_summary,
            "normalization": norm_summary,
        }
        torch.save(
            normalization_payload,
            run_dir / "final_layer_normalization.pt",
        )
        final_layer_fn = train_dense_final_layer if cfg.final_layer_type == "dense" else train_sparse_final_layer
        final_layer_summary = final_layer_fn(
            train_feature_path=train_feature_path,
            train_target_path=train_target_path,
            val_feature_path=val_feature_path,
            val_target_path=val_target_path,
            feature_mean=feature_mean,
            feature_std=feature_std,
            cfg=cfg,
            n_classes=len(train_dataset_full.dataset.classes),
            run_dir=run_dir,
        )
        final_layer_summary["type"] = cfg.final_layer_type
        final_layer_summary["feature_extraction"] = {
            "train": train_extract_summary,
            "val": val_extract_summary,
            "normalization": norm_summary,
        }
        if persist_copy_summary is not None:
            final_layer_summary["feature_extraction"]["persist_copy"] = persist_copy_summary
        (run_dir / "final_layer_summary.json").write_text(json.dumps(final_layer_summary, indent=2))
        print(
            f"[final_layer:{cfg.final_layer_type}] "
            f"train_top1={final_layer_summary['train']['top1']:.4f} "
            f"val_top1={final_layer_summary['val']['top1']:.4f} "
            f"sparsity={final_layer_summary['nnz']}/{final_layer_summary['total']}",
            flush=True,
        )

    result = {
        "run_dir": str(run_dir),
        "best_val_loss": best_val,
        "history": history,
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "n_concepts": len(concepts),
        "concept_filter": concept_filter_summary,
        "final_layer": final_layer_summary,
    }
    (run_dir / "summary.json").write_text(json.dumps(result, indent=2))
    return result


def infer_n_classes_from_targets(*target_paths: Path) -> int:
    max_class_id = -1
    for target_path in target_paths:
        targets = np.load(target_path, mmap_mode="r")
        if int(targets.shape[0]) == 0:
            continue
        max_class_id = max(max_class_id, int(np.asarray(targets).max()))
    if max_class_id < 0:
        raise RuntimeError("Could not infer class count from target files")
    return max_class_id + 1


def resolve_reuse_run_context(
    cfg: Config,
) -> Tuple[
    Path,
    Path,
    Path,
    Path,
    Path,
    Path,
    torch.Tensor,
    torch.Tensor,
    Dict[str, Any],
    int,
]:
    source_run_dir = Path(cfg.reuse_run_dir).resolve()
    if not source_run_dir.is_dir():
        raise FileNotFoundError(f"Missing reuse_run_dir: {source_run_dir}")
    original_source_run_dir = source_run_dir
    original_source_hint = source_run_dir / "source_run_dir.txt"
    if original_source_hint.exists():
        hinted_source_run_dir = Path(original_source_hint.read_text().strip()).resolve()
        if hinted_source_run_dir.is_dir():
            original_source_run_dir = hinted_source_run_dir

    train_feature_path = source_run_dir / "features" / "train_features.npy"
    train_target_path = source_run_dir / "features" / "train_targets.npy"
    val_feature_path = source_run_dir / "features" / "val_features.npy"
    val_target_path = source_run_dir / "features" / "val_targets.npy"
    for path in (train_feature_path, train_target_path, val_feature_path, val_target_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing feature artifact: {path}")

    normalization_path = source_run_dir / "final_layer_normalization.pt"
    if normalization_path.exists():
        normalization_payload = torch.load(normalization_path, map_location="cpu")
        feature_mean = normalization_payload["mean"].float().cpu()
        feature_std = normalization_payload["std"].float().cpu()
        normalization_summary = normalization_payload.get("normalization", {})
    else:
        feature_mean, feature_std, normalization_summary = compute_feature_stats_memmap(train_feature_path, cfg)

    n_classes = infer_n_classes_from_targets(train_target_path, val_target_path)
    return (
        source_run_dir,
        original_source_run_dir,
        train_feature_path,
        train_target_path,
        val_feature_path,
        val_target_path,
        feature_mean,
        feature_std,
        normalization_summary,
        n_classes,
    )


def initialize_reuse_run_dir(
    cfg: Config,
    original_source_run_dir: Path,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
    normalization_summary: Dict[str, Any],
) -> Path:
    run_dir = build_run_dir(cfg)
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    (run_dir / "source_run_dir.txt").write_text(f"{original_source_run_dir}\n")
    source_concepts = original_source_run_dir / "concepts.txt"
    if source_concepts.exists():
        (run_dir / "concepts.txt").write_text(source_concepts.read_text())
    torch.save(
        {
            "mean": feature_mean,
            "std": feature_std,
            "source_run_dir": str(original_source_run_dir),
            "normalization": normalization_summary,
        },
        run_dir / "final_layer_normalization.pt",
    )
    return run_dir


def run_glm_only(cfg: Config) -> Dict[str, Any]:
    (
        _source_run_dir,
        original_source_run_dir,
        train_feature_path,
        train_target_path,
        val_feature_path,
        val_target_path,
        feature_mean,
        feature_std,
        normalization_summary,
        n_classes,
    ) = resolve_reuse_run_context(cfg)
    run_dir = initialize_reuse_run_dir(
        cfg,
        original_source_run_dir,
        feature_mean,
        feature_std,
        normalization_summary,
    )
    final_layer_summary = train_sparse_final_layer(
        train_feature_path=train_feature_path,
        train_target_path=train_target_path,
        val_feature_path=val_feature_path,
        val_target_path=val_target_path,
        feature_mean=feature_mean,
        feature_std=feature_std,
        cfg=cfg,
        n_classes=n_classes,
        run_dir=run_dir,
    )

    result = {
        "mode": "glm_only",
        "source_run_dir": str(original_source_run_dir),
        "run_dir": str(run_dir),
        "n_classes": n_classes,
        "final_layer": final_layer_summary,
    }
    (run_dir / "summary.json").write_text(json.dumps(result, indent=2))
    return result


def run_dense_only(cfg: Config) -> Dict[str, Any]:
    (
        _source_run_dir,
        original_source_run_dir,
        train_feature_path,
        train_target_path,
        val_feature_path,
        val_target_path,
        feature_mean,
        feature_std,
        normalization_summary,
        n_classes,
    ) = resolve_reuse_run_context(cfg)
    run_dir = initialize_reuse_run_dir(
        cfg,
        original_source_run_dir,
        feature_mean,
        feature_std,
        normalization_summary,
    )
    final_layer_summary = train_dense_final_layer(
        train_feature_path=train_feature_path,
        train_target_path=train_target_path,
        val_feature_path=val_feature_path,
        val_target_path=val_target_path,
        feature_mean=feature_mean,
        feature_std=feature_std,
        cfg=cfg,
        n_classes=n_classes,
        run_dir=run_dir,
    )
    result = {
        "mode": "dense_only",
        "source_run_dir": str(original_source_run_dir),
        "run_dir": str(run_dir),
        "n_classes": n_classes,
        "final_layer": final_layer_summary,
    }
    (run_dir / "summary.json").write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    cfg = build_config(parse_args())
    validate_config(cfg)
    configure_runtime(cfg)
    if cfg.print_config:
        print(json.dumps(asdict(cfg), indent=2, sort_keys=True))
    if cfg.mode == "profile":
        result = profile_pipeline(cfg)
    elif cfg.mode == "glm_only":
        result = run_glm_only(cfg)
    elif cfg.mode == "dense_only":
        result = run_dense_only(cfg)
    elif cfg.mode == "precompute_targets":
        result = run_precompute_targets(cfg)
    else:
        result = run_training(cfg)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
