import hashlib
import json
import math
import os
import re
import struct
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm

from data import utils as data_utils
from data.concept_dataset import get_filtered_concepts_and_counts
from glm_saga.elasticnet import IndexedTensorDataset
from methods.common import build_run_dir, save_args, write_artifacts
from methods.lf import TransformedSubset, subset_targets, use_original_label_free_protocol
from methods.salf import (
    RawSubset,
    SpatialBackbone,
    build_single_spatial_concept_layer,
    build_spatial_concept_layer,
)
from model.cbm import Backbone, BackboneCLIP, ConceptLayer, train_dense_final, train_sparse_final
from model.sam import SAM
from PIL import Image


def create_savlg_splits(args):
    backbone = SpatialBackbone(
        args.backbone,
        device=args.device,
        spatial_stage=getattr(args, "savlg_spatial_stage", "conv5"),
    )
    if use_original_label_free_protocol(args):
        base_train_raw = data_utils.get_data(f"{args.dataset}_train", None)
        base_val_raw = data_utils.get_data(f"{args.dataset}_val", None)
        print(
            f"[create_savlg_splits] raw datasets ready train={len(base_train_raw)} val={len(base_val_raw)}",
            flush=True,
        )
        max_train = int(getattr(args, "max_train_images", 0) or 0)
        train_total = len(base_train_raw)
        if max_train > 0:
            train_total = min(train_total, max_train)
        train_indices = list(range(train_total))
        print(
            f"[create_savlg_splits] train_indices ready n={len(train_indices)}",
            flush=True,
        )
        max_test = int(getattr(args, "max_test_images", 0) or 0)
        val_total = len(base_val_raw)
        if max_test > 0:
            val_total = min(val_total, max_test)
        val_indices = list(range(val_total))
        print(
            f"[create_savlg_splits] val_indices ready n={len(val_indices)}",
            flush=True,
        )
        train_raw = RawSubset(base_train_raw, train_indices)
        print("[create_savlg_splits] train_raw ready", flush=True)
        val_raw = RawSubset(base_val_raw, val_indices)
        print("[create_savlg_splits] val_raw ready", flush=True)
        train_dataset = TransformedSubset(base_train_raw, train_indices, backbone.preprocess)
        print("[create_savlg_splits] train_dataset ready", flush=True)
        val_dataset = TransformedSubset(base_val_raw, val_indices, backbone.preprocess)
        print("[create_savlg_splits] val_dataset ready", flush=True)
        test_dataset = val_dataset
        print("[create_savlg_splits] returning original LF protocol splits", flush=True)
        return train_raw, val_raw, train_dataset, val_dataset, test_dataset, backbone

    base_train_raw = data_utils.get_data(f"{args.dataset}_train", None)
    max_train = int(getattr(args, "max_train_images", 0) or 0)
    total = len(base_train_raw)
    if max_train > 0:
        total = min(total, max_train)
    n_val = int(args.val_split * total)
    n_train = total - n_val
    generator = torch.Generator().manual_seed(args.seed)
    train_subset, val_subset = torch.utils.data.random_split(
        list(range(total)),
        [n_train, n_val],
        generator=generator,
    )
    train_raw = RawSubset(base_train_raw, train_subset.indices)
    val_raw = RawSubset(base_train_raw, val_subset.indices)
    train_dataset = TransformedSubset(base_train_raw, train_subset.indices, backbone.preprocess)
    val_dataset = TransformedSubset(base_train_raw, val_subset.indices, backbone.preprocess)
    if getattr(args, "skip_test_eval", False):
        test_dataset = val_dataset
    else:
        base_test = data_utils.get_data(f"{args.dataset}_val", None)
        max_test = int(getattr(args, "max_test_images", 0) or 0)
        test_total = len(base_test)
        if max_test > 0:
            test_total = min(test_total, max_test)
        test_dataset = TransformedSubset(base_test, list(range(test_total)), backbone.preprocess)
    return train_raw, val_raw, train_dataset, val_dataset, test_dataset, backbone


def _annotation_split_dir(annotation_root: str, dataset: str, split_name: str) -> str:
    direct = os.path.join(annotation_root, f"{dataset}_{split_name}")
    if os.path.isdir(direct):
        return direct
    if split_name == "val":
        alt = os.path.join(annotation_root, f"{dataset}_test")
        if os.path.isdir(alt):
            return alt
    raise FileNotFoundError(
        f"Could not find annotation split directory for dataset={dataset} split={split_name} under {annotation_root}"
    )


def _supervision_cache_path(
    args,
    split_name: str,
    concepts: Sequence[str],
    raw_dataset: Optional[Dataset] = None,
) -> str:
    concept_hash = hashlib.sha1("\n".join(concepts).encode("utf-8")).hexdigest()[:16]
    # Include the specific sample indices in the cache key so that different
    # train/val splits (or smoke subsets) do not overwrite each other.
    #
    # Without this, a small subset run can poison the cache for a full run and
    # later cause hard-to-debug IndexErrors in the dataloader.
    sample_tag = "n_unknown"
    if raw_dataset is not None:
        indices = getattr(raw_dataset, "indices", None)
        if isinstance(indices, (list, tuple)) and indices:
            h = hashlib.sha1()
            # Hash indices incrementally to avoid constructing huge strings.
            for idx in indices:
                try:
                    h.update(struct.pack("<I", int(idx)))
                except struct.error:
                    h.update(str(int(idx)).encode("utf-8") + b",")
            sample_tag = f"idx_{len(indices)}_{h.hexdigest()[:12]}"
        else:
            try:
                sample_tag = f"n_{len(raw_dataset)}"
            except Exception:
                sample_tag = "n_unknown"
    threshold_tag = str(float(getattr(args, "cbl_confidence_threshold", 0.15))).replace(".", "p")
    target_mode = str(getattr(args, "savlg_target_mode", "hard_iou")).lower()
    global_target_mode = _savlg_global_target_mode(args)
    patch_iou_tag = str(float(getattr(args, "patch_iou_thresh", 0.5))).replace(".", "p")
    supervision_source = _savlg_supervision_source(args)
    source_tag = supervision_source
    if supervision_source == "groundedsam2":
        manifest_path = _groundedsam2_manifest_path(args, split_name)
        source_tag = "{}_{}".format(
            supervision_source,
            hashlib.sha1(os.path.abspath(manifest_path).encode("utf-8")).hexdigest()[:12],
        )
    cache_dir = os.path.join(getattr(args, "activation_dir", "saved_activations"), "savlg")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(
        cache_dir,
        f"{args.dataset}_{split_name}_{args.backbone}_{sample_tag}_src_{source_tag}_thr_{threshold_tag}_tm_{target_mode}_gtm_{global_target_mode}_piou_{patch_iou_tag}_mh{int(args.mask_h)}_mw{int(args.mask_w)}_{concept_hash}_supervision.pt",
    )


def _image_size_cache_path(args, split_name: str) -> str:
    cache_dir = os.path.join(getattr(args, "activation_dir", "saved_activations"), "savlg")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{args.dataset}_{split_name}_{args.backbone}_image_sizes.json")


def _savlg_global_target_mode(args) -> str:
    return str(getattr(args, "savlg_global_target_mode", "binary_threshold")).lower()


def _savlg_supervision_source(args) -> str:
    return str(getattr(args, "savlg_supervision_source", "gdino")).lower()


def _savlg_concept_filter_mode(args) -> str:
    return str(getattr(args, "savlg_concept_filter_mode", "spatial_threshold")).lower()


def _savlg_global_concept_loss_weight(args) -> float:
    if getattr(args, "loss_global_concept_w", None) is not None:
        return float(getattr(args, "loss_global_concept_w"))
    if getattr(args, "loss_presence_w", None) is not None:
        return float(getattr(args, "loss_presence_w"))
    return 1.0


def _build_global_concept_targets(global_concept_scores: np.ndarray, args) -> np.ndarray:
    mode = _savlg_global_target_mode(args)
    if mode == "binary_threshold":
        threshold = float(getattr(args, "cbl_confidence_threshold", 0.15))
        return (global_concept_scores > threshold).astype(np.float32)
    if mode == "raw_logit":
        return global_concept_scores.astype(np.float32)
    raise ValueError(f"Unsupported SAVLG global target mode: {mode}")


def _savlg_io_workers(args) -> int:
    workers = int(
        getattr(args, "spatial_num_workers", 0)
        or getattr(args, "num_workers", 0)
        or 0
    )
    if workers <= 0:
        workers = min(16, os.cpu_count() or 1)
    return max(1, workers)


def _load_concepts_file(path: str) -> List[str]:
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def _resolve_manifest_path(path_or_dir: str) -> str:
    resolved = os.path.expanduser(path_or_dir)
    if os.path.isdir(resolved):
        resolved = os.path.join(resolved, "manifest.json")
    return resolved


def _groundedsam2_manifest_path(args, split_name: str) -> str:
    attr = f"savlg_groundedsam2_{split_name}_manifest"
    candidate = str(getattr(args, attr, "") or "").strip()
    if not candidate:
        raise ValueError(
            f"SAVLG supervision source is groundedsam2 but --{attr} was not provided."
        )
    manifest_path = _resolve_manifest_path(candidate)
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(
            f"Could not find GroundedSAM2 manifest for split={split_name}: {manifest_path}"
        )
    return manifest_path


def _parse_manifest_image_id(value) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        if stripped.isdigit():
            return int(stripped)
        match = re.search(r"(\d+)$", stripped)
        if match is not None:
            return int(match.group(1))
    return None


def _normalize_relpath(path: Optional[str]) -> str:
    if not path:
        return ""
    return Path(str(path).replace("\\", "/")).as_posix().lstrip("./").lower()


def _path_tail(path: str, n_parts: int) -> str:
    parts = Path(path).parts
    if not parts:
        return ""
    return "/".join(parts[-n_parts:]).lower()


def _resolve_bundle_artifact_path(manifest_path: str, bundle_path: str) -> str:
    if os.path.isabs(bundle_path):
        return bundle_path
    return os.path.join(os.path.dirname(manifest_path), bundle_path)


def _downsample_binary_mask_target(mask: np.ndarray, args) -> Optional[np.ndarray]:
    if mask.ndim != 2:
        return None
    mask_bool = np.asarray(mask, dtype=np.bool_)
    if not bool(mask_bool.any()):
        return None
    mask_tensor = torch.from_numpy(mask_bool.astype(np.float32))[None, None, ...]
    downsampled = F.interpolate(
        mask_tensor,
        size=(int(args.mask_h), int(args.mask_w)),
        mode="area",
    )[0, 0]
    return downsampled.clamp_(0.0, 1.0).numpy().astype(np.float32)


def _groundedsam2_presence_score(record: Dict[str, object], args) -> float:
    if _savlg_global_target_mode(args) == "binary_threshold":
        return 1.0
    for key in ("annotation_score", "annotation_logit", "sam2_score"):
        value = record.get(key)
        try:
            if value is not None:
                score = float(value)
                if np.isfinite(score):
                    return score
        except (TypeError, ValueError):
            continue
    return 1.0


def _load_groundedsam2_spatial_supervision(
    raw_dataset: Dataset,
    concepts: Sequence[str],
    args,
    split_name: str,
    keep_idx: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, List[Dict[int, np.ndarray]], List[int]]:
    manifest_path = _groundedsam2_manifest_path(args, split_name)
    logger.info(
        "Building SAVLG {} supervision from GroundedSAM2 manifest {}",
        split_name,
        manifest_path,
    )
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    records = manifest.get("records", [])

    concept_to_idx = {concept: idx for idx, concept in enumerate(concepts)}
    row_to_ann_idx = (
        list(raw_dataset.indices)
        if hasattr(raw_dataset, "indices")
        else list(range(len(raw_dataset)))
    )
    image_paths = _subset_image_paths(raw_dataset)

    records_by_idx: Dict[int, List[Dict[str, object]]] = {}
    records_by_relpath: Dict[str, List[Dict[str, object]]] = {}
    records_by_tail2: Dict[str, List[Dict[str, object]]] = {}
    records_by_basename: Dict[str, List[Dict[str, object]]] = {}
    for record in records:
        image_idx = _parse_manifest_image_id(record.get("image_id"))
        if image_idx is not None:
            records_by_idx.setdefault(image_idx, []).append(record)
        relpath = _normalize_relpath(record.get("image_relpath"))
        if relpath:
            records_by_relpath.setdefault(relpath, []).append(record)
            records_by_tail2.setdefault(_path_tail(relpath, 2), []).append(record)
            records_by_basename.setdefault(Path(relpath).name.lower(), []).append(record)

    def _records_for_row(row_idx: int, ann_idx: int) -> List[Dict[str, object]]:
        direct = records_by_idx.get(int(ann_idx))
        if direct:
            return direct
        if image_paths is None:
            return []
        full_path = _normalize_relpath(image_paths[row_idx])
        if not full_path:
            return []
        direct_rel = records_by_relpath.get(full_path)
        if direct_rel:
            return direct_rel
        tail2 = _path_tail(full_path, 2)
        tail2_matches = records_by_tail2.get(tail2, [])
        if len(tail2_matches) == 1:
            return tail2_matches
        basename_matches = records_by_basename.get(Path(full_path).name.lower(), [])
        if len(basename_matches) == 1:
            return basename_matches
        return []

    global_concept_scores = np.zeros((len(raw_dataset), len(concepts)), dtype=np.float32)
    mask_entries: List[Dict[int, np.ndarray]] = [dict() for _ in range(len(raw_dataset))]
    bundle_cache: Dict[str, np.ndarray] = {}
    bundle_failures = 0

    for row_idx, ann_idx in tqdm(
        list(enumerate(row_to_ann_idx)),
        total=len(row_to_ann_idx),
        desc=f"SAVLG {split_name} GroundedSAM2",
    ):
        row_records = _records_for_row(row_idx, ann_idx)
        if not row_records:
            continue
        masks_by_bundle: Dict[str, np.ndarray] = {}
        for record in row_records:
            bundle_npz = record.get("bundle_npz")
            if not isinstance(bundle_npz, str) or not bundle_npz:
                continue
            bundle_path = _resolve_bundle_artifact_path(manifest_path, bundle_npz)
            masks = masks_by_bundle.get(bundle_path)
            if masks is None:
                masks = bundle_cache.get(bundle_path)
                if masks is None:
                    try:
                        with np.load(bundle_path, allow_pickle=False) as payload:
                            masks = payload["masks"]
                    except Exception as exc:
                        bundle_failures += 1
                        logger.warning(
                            "Failed to load GroundedSAM2 bundle {}: {}",
                            bundle_path,
                            exc,
                        )
                        continue
                    bundle_cache[bundle_path] = masks
                masks_by_bundle[bundle_path] = masks

            label = record.get("label")
            if not isinstance(label, str):
                continue
            cidx = concept_to_idx.get(data_utils.canonicalize_concept_label(label))
            if cidx is None:
                continue

            try:
                mask_index = int(record.get("mask_index"))
            except (TypeError, ValueError):
                continue
            if mask_index < 0 or mask_index >= int(masks.shape[0]):
                continue

            mask_area = record.get("mask_area")
            if mask_area is not None:
                try:
                    if float(mask_area) <= 0.0:
                        continue
                except (TypeError, ValueError):
                    pass
            mask_target = _downsample_binary_mask_target(masks[mask_index], args)
            if mask_target is None:
                continue

            global_concept_scores[row_idx, cidx] = max(
                global_concept_scores[row_idx, cidx],
                _groundedsam2_presence_score(record, args),
            )
            existing = mask_entries[row_idx].get(cidx)
            if existing is None:
                mask_entries[row_idx][cidx] = mask_target
            else:
                np.maximum(existing, mask_target, out=existing)

    if bundle_failures > 0:
        logger.warning(
            "GroundedSAM2 supervision skipped {} bundle loads due to NPZ read failures",
            bundle_failures,
        )

    threshold = float(getattr(args, "cbl_confidence_threshold", 0.15))
    if keep_idx is None:
        keep_mask = global_concept_scores.max(axis=0) >= threshold
        if not bool(keep_mask.any()):
            raise RuntimeError("All SAVLG concepts were removed after GroundedSAM2 supervision filtering.")
        keep_idx_array = np.where(keep_mask)[0]
    else:
        keep_idx_array = np.asarray(list(keep_idx), dtype=np.int64)
        if keep_idx_array.size == 0:
            raise RuntimeError("SAVLG keep_idx is empty.")

    filtered_scores = global_concept_scores[:, keep_idx_array]
    global_concept_targets = _build_global_concept_targets(filtered_scores, args)
    old_to_new = {old: new for new, old in enumerate(keep_idx_array.tolist())}
    filtered_entries: List[Dict[int, np.ndarray]] = []
    for entry in mask_entries:
        new_entry = {}
        for old_idx, mask in entry.items():
            if old_idx in old_to_new:
                new_entry[old_to_new[old_idx]] = mask
        filtered_entries.append(new_entry)

    return global_concept_targets, filtered_entries, keep_idx_array.tolist()


def _load_savlg_teacher(args, concepts: Sequence[str]) -> Optional[Dict[str, object]]:
    teacher_path = str(getattr(args, "savlg_teacher_load_path", "") or "").strip()
    if not teacher_path:
        return None
    if not os.path.isdir(teacher_path):
        raise FileNotFoundError(f"SAVLG teacher path does not exist: {teacher_path}")

    with open(os.path.join(teacher_path, "args.txt"), "r") as f:
        teacher_args = json.load(f)

    teacher_concepts = _load_concepts_file(os.path.join(teacher_path, "concepts.txt"))
    teacher_lookup = {concept: idx for idx, concept in enumerate(teacher_concepts)}
    missing = [concept for concept in concepts if concept not in teacher_lookup]
    if missing:
        raise RuntimeError(
            "SAVLG teacher is missing {} student concepts, including: {}".format(
                len(missing),
                ", ".join(missing[:10]),
            )
        )
    aligned_indices = torch.tensor(
        [teacher_lookup[concept] for concept in concepts],
        dtype=torch.long,
        device=args.device,
    )

    backbone_path = os.path.join(teacher_path, "backbone.pt")
    if str(teacher_args["backbone"]).startswith("clip_"):
        if os.path.exists(backbone_path):
            teacher_backbone = BackboneCLIP.from_pretrained(teacher_path, args.device)
        else:
            teacher_backbone = BackboneCLIP(
                teacher_args["backbone"],
                use_penultimate=bool(teacher_args.get("use_clip_penultimate", False)),
                device=args.device,
            )
    else:
        if os.path.exists(backbone_path):
            teacher_backbone = Backbone.from_pretrained(teacher_path, args.device)
        else:
            teacher_backbone = Backbone.from_args(teacher_path, args.device)
    teacher_concept_layer = ConceptLayer.from_pretrained(teacher_path, args.device)

    teacher_backbone.eval()
    for parameter in teacher_backbone.parameters():
        parameter.requires_grad = False
    teacher_concept_layer.eval()
    for parameter in teacher_concept_layer.parameters():
        parameter.requires_grad = False

    logger.info(
        "Loaded SAVLG teacher from {} with {} aligned concepts",
        teacher_path,
        len(concepts),
    )
    return {
        "load_path": teacher_path,
        "backbone": teacher_backbone,
        "concept_layer": teacher_concept_layer,
        "indices": aligned_indices,
    }


def _subset_image_paths(raw_dataset: Dataset) -> Optional[List[str]]:
    base_dataset = getattr(raw_dataset, "base_dataset", None)
    indices = getattr(raw_dataset, "indices", None)
    if base_dataset is None or indices is None:
        return None
    samples = None
    if hasattr(base_dataset, "samples"):
        samples = base_dataset.samples
    elif hasattr(base_dataset, "imgs"):
        samples = base_dataset.imgs
    if samples is None:
        return None
    return [str(samples[idx][0]) for idx in indices]


def _load_or_build_image_sizes(
    raw_dataset: Dataset,
    args,
    split_name: str,
) -> List[Tuple[int, int]]:
    cache_path = _image_size_cache_path(args, split_name)
    if os.path.exists(cache_path) and not getattr(args, "recompute_spatial_sims", False):
        with open(cache_path, "r") as f:
            payload = json.load(f)
        sizes = [tuple(size) for size in payload.get("sizes", [])]
        if len(sizes) == len(raw_dataset):
            logger.info("Loading cached SAVLG image sizes from {}", cache_path)
            return sizes

    image_paths = _subset_image_paths(raw_dataset)
    sizes: List[Tuple[int, int]] = []
    worker_count = _savlg_io_workers(args)
    min_bytes = int(os.environ.get("CBM_MIN_IMAGE_BYTES", "1024"))
    fallback_size = int(os.environ.get("CBM_FALLBACK_IMAGE_SIZE", "224"))
    bad_paths: List[str] = []
    if image_paths is not None:
        logger.info(
            "Building SAVLG image-size cache for {} from {} file paths with {} workers",
            split_name,
            len(image_paths),
            worker_count,
        )
        def _read_size(img_path: str) -> Tuple[int, int]:
            try:
                if min_bytes > 0 and os.path.getsize(img_path) < min_bytes:
                    raise OSError(f"image file too small (<{min_bytes} bytes)")
                with Image.open(img_path) as img:
                    return int(img.size[0]), int(img.size[1])
            except Exception:
                # Keep indexing stable; a tiny handful of corrupt images should not crash a run.
                if len(bad_paths) < 200:
                    bad_paths.append(str(img_path))
                return fallback_size, fallback_size

        with ThreadPoolExecutor(max_workers=worker_count) as ex:
            for size in tqdm(
                ex.map(_read_size, image_paths),
                total=len(image_paths),
                desc=f"SAVLG {split_name} image sizes",
            ):
                sizes.append(size)
    else:
        logger.info(
            "Building SAVLG image-size cache for {} by loading dataset items",
            split_name,
        )
        for row_idx in tqdm(range(len(raw_dataset)), desc=f"SAVLG {split_name} image sizes"):
            pil_img, _ = raw_dataset[row_idx]
            sizes.append((int(pil_img.size[0]), int(pil_img.size[1])))

    with open(cache_path, "w") as f:
        json.dump({"sizes": sizes}, f)
    if bad_paths:
        bad_path_file = cache_path + ".bad_paths.txt"
        try:
            with open(bad_path_file, "w") as f:
                f.write("\n".join(bad_paths) + "\n")
            logger.warning(
                "SAVLG image-size cache saw {} unreadable/tiny images; wrote sample list to {}",
                len(bad_paths),
                bad_path_file,
            )
        except Exception:
            pass
    logger.info("Saved SAVLG image-size cache to {}", cache_path)
    return sizes


def _normalize_box(
    box: Sequence[float],
    image_size: Tuple[int, int],
) -> Optional[Tuple[float, float, float, float]]:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    x1, y1, x2, y2 = [float(v) for v in box]
    w, h = int(image_size[0]), int(image_size[1])
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) > 1.5:
        if w <= 0 or h <= 0:
            return None
        x1, x2 = x1 / w, x2 / w
        y1, y2 = y1 / h, y2 / h
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1 = float(np.clip(x1, 0.0, 1.0))
    x2 = float(np.clip(x2, 0.0, 1.0))
    y1 = float(np.clip(y1, 0.0, 1.0))
    y2 = float(np.clip(y2, 0.0, 1.0))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _rasterize_box_patch_iou(
    box: Sequence[float],
    image_size: Tuple[int, int],
    mask_h: int,
    mask_w: int,
    iou_thresh: float,
) -> Optional[np.ndarray]:
    norm = _normalize_box(box, image_size=image_size)
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
            if union > 0.0 and (inter / union) > float(iou_thresh):
                mask[r, c] = 1.0
    return mask


def _rasterize_box_soft_occupancy(
    box: Sequence[float],
    image_size: Tuple[int, int],
    mask_h: int,
    mask_w: int,
) -> Optional[np.ndarray]:
    norm = _normalize_box(box, image_size=image_size)
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
            mask[r, c] = float(np.clip(inter / patch_area, 0.0, 1.0))
    return mask


def _rasterize_box_target(
    box: Sequence[float],
    image_size: Tuple[int, int],
    args,
) -> Optional[np.ndarray]:
    target_mode = str(getattr(args, "savlg_target_mode", "hard_iou")).lower()
    mask_h = int(args.mask_h)
    mask_w = int(args.mask_w)
    if target_mode == "hard_iou":
        return _rasterize_box_patch_iou(
            box=box,
            image_size=image_size,
            mask_h=mask_h,
            mask_w=mask_w,
            iou_thresh=float(getattr(args, "patch_iou_thresh", 0.5)),
        )
    if target_mode == "soft_box":
        return _rasterize_box_soft_occupancy(
            box=box,
            image_size=image_size,
            mask_h=mask_h,
            mask_w=mask_w,
        )
    raise ValueError(f"Unsupported SAVLG target mode: {target_mode}")


def load_spatial_supervision(
    raw_dataset: Dataset,
    annotation_dir: str,
    concepts: Sequence[str],
    args,
    split_name: str,
    keep_idx: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, List[Dict[int, np.ndarray]], List[int]]:
    cache_path = _supervision_cache_path(args, split_name, concepts, raw_dataset=raw_dataset)
    if os.path.exists(cache_path) and not getattr(args, "recompute_spatial_sims", False):
        logger.info("Loading cached SAVLG supervision from {}", cache_path)
        payload = torch.load(cache_path, weights_only=False)
        cached_keep_idx = [int(x) for x in payload.get("keep_idx", [])]
        cached_global_targets = payload.get("global_concept_targets")
        if cached_global_targets is None and payload.get("presence_scores") is not None:
            cached_global_targets = _build_global_concept_targets(
                np.asarray(payload["presence_scores"], dtype=np.float32),
                args,
            )
        cached_mask_entries = payload.get("mask_entries")
        cache_ok = True
        reason = ""
        if cached_global_targets is None or cached_mask_entries is None:
            cache_ok = False
            reason = "missing fields"
        elif len(cached_global_targets) != len(raw_dataset):
            cache_ok = False
            reason = f"row-count mismatch (cached={len(cached_global_targets)} current={len(raw_dataset)})"
        elif len(cached_mask_entries) != len(raw_dataset):
            cache_ok = False
            reason = f"mask-entry mismatch (cached={len(cached_mask_entries)} current={len(raw_dataset)})"
        elif keep_idx is not None and cached_keep_idx != list(keep_idx):
            cache_ok = False
            reason = f"keep_idx changed (cached={len(cached_keep_idx)} current={len(list(keep_idx))})"
        if cache_ok:
            return cached_global_targets, cached_mask_entries, cached_keep_idx
        logger.info("Ignoring cached SAVLG supervision at {} due to {}", cache_path, reason)

    supervision_source = _savlg_supervision_source(args)
    if supervision_source == "groundedsam2":
        global_concept_targets, filtered_entries, keep_idx_list = _load_groundedsam2_spatial_supervision(
            raw_dataset,
            concepts,
            args,
            split_name,
            keep_idx=keep_idx,
        )
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(
            {
                "global_concept_targets": global_concept_targets,
                "mask_entries": filtered_entries,
                "keep_idx": keep_idx_list,
                "source": supervision_source,
            },
            cache_path,
        )
        logger.info(
            "Saved SAVLG {} supervision cache to {} (kept {}/{})",
            supervision_source,
            cache_path,
            int(len(keep_idx_list)),
            len(concepts),
        )
        return global_concept_targets, filtered_entries, keep_idx_list
    if supervision_source != "gdino":
        raise ValueError(
            f"Unsupported SAVLG supervision source: {supervision_source}. Expected one of ['gdino', 'groundedsam2']."
        )

    threshold = float(getattr(args, "cbl_confidence_threshold", 0.15))
    concept_to_idx = {concept: idx for idx, concept in enumerate(concepts)}
    global_concept_scores = np.zeros((len(raw_dataset), len(concepts)), dtype=np.float32)
    mask_entries: List[Dict[int, np.ndarray]] = [dict() for _ in range(len(raw_dataset))]
    image_sizes = _load_or_build_image_sizes(raw_dataset, args, split_name)
    worker_count = _savlg_io_workers(args)

    row_to_ann_idx = (
        list(raw_dataset.indices)
        if hasattr(raw_dataset, "indices")
        else list(range(len(raw_dataset)))
    )

    logger.info(
        "Building SAVLG supervision for {} from {} (rows={}, concepts={})",
        split_name,
        annotation_dir,
        len(row_to_ann_idx),
        len(concepts),
    )
    def _parse_one(task: Tuple[int, int]):
        row_idx, ann_idx = task
        ann_path = os.path.join(annotation_dir, f"{int(ann_idx)}.json")
        if not os.path.exists(ann_path):
            return row_idx, []
        try:
            with open(ann_path, "r") as f:
                data = json.load(f)
        except Exception:
            return row_idx, []
        parsed = []
        for ann in data[1:]:
            if not isinstance(ann, dict):
                continue
            label = ann.get("label")
            if isinstance(label, str):
                label = data_utils.canonicalize_concept_label(label)
            cidx = concept_to_idx.get(label)
            if cidx is None:
                continue
            score = float(ann.get("logit", 0.0))
            box = ann.get("box")
            parsed.append((cidx, score, box))
        return row_idx, parsed

    tasks = list(enumerate(row_to_ann_idx))
    with ThreadPoolExecutor(max_workers=worker_count) as ex:
        for row_idx, parsed in tqdm(
            ex.map(_parse_one, tasks),
            total=len(tasks),
            desc=f"SAVLG {split_name} annotations",
        ):
            image_size = image_sizes[row_idx]
            for cidx, score, box in parsed:
                if score > global_concept_scores[row_idx, cidx]:
                    global_concept_scores[row_idx, cidx] = score
                if box is None or score < threshold:
                    continue
                box_mask = _rasterize_box_target(box=box, image_size=image_size, args=args)
                if box_mask is None:
                    continue
                existing = mask_entries[row_idx].get(cidx)
                if existing is None:
                    mask_entries[row_idx][cidx] = box_mask
                else:
                    np.maximum(existing, box_mask, out=existing)

    if keep_idx is None:
        keep_mask = global_concept_scores.max(axis=0) >= threshold
        if not bool(keep_mask.any()):
            raise RuntimeError("All SAVLG concepts were removed after annotation thresholding.")
        keep_idx_array = np.where(keep_mask)[0]
    else:
        keep_idx_array = np.asarray(list(keep_idx), dtype=np.int64)
        if keep_idx_array.size == 0:
            raise RuntimeError("SAVLG keep_idx is empty.")
    filtered_entries: List[Dict[int, np.ndarray]] = []
    old_to_new = {old: new for new, old in enumerate(keep_idx_array.tolist())}
    filtered_scores = global_concept_scores[:, keep_idx_array]
    global_concept_targets = _build_global_concept_targets(filtered_scores, args)
    for entry in mask_entries:
        new_entry = {}
        for old_idx, mask in entry.items():
            if old_idx in old_to_new:
                new_entry[old_to_new[old_idx]] = mask
        filtered_entries.append(new_entry)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(
        {
            "global_concept_scores": filtered_scores,
            "global_concept_targets": global_concept_targets,
            "presence_scores": filtered_scores,
            "mask_entries": filtered_entries,
            "keep_idx": keep_idx_array.tolist(),
        },
        cache_path,
    )
    logger.info(
        "Saved SAVLG supervision cache to {} (kept {}/{})",
        cache_path,
        int(len(keep_idx_array)),
        len(concepts),
    )
    return global_concept_targets, filtered_entries, keep_idx_array.tolist()


class SpatialSupervisionDataset(Dataset):
    def __init__(
        self,
        base_dataset: Dataset,
        global_concept_targets: np.ndarray,
        mask_entries: List[Dict[int, np.ndarray]],
        mask_h: int,
        mask_w: int,
    ):
        self.base_dataset = base_dataset
        self.global_concept_targets = global_concept_targets.astype(np.float32)
        self.mask_entries = mask_entries
        self.mask_h = int(mask_h)
        self.mask_w = int(mask_w)
        self.targets = subset_targets(base_dataset.base_dataset, base_dataset.indices) if hasattr(base_dataset, "indices") else subset_targets(base_dataset, range(len(base_dataset)))

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, target = self.base_dataset[idx]
        global_concepts = torch.from_numpy(self.global_concept_targets[idx])
        entry = self.mask_entries[idx]
        if entry:
            keys = sorted(entry.keys())
            concept_indices = torch.tensor(keys, dtype=torch.long)
            mask_stack = torch.from_numpy(np.stack([entry[k] for k in keys], axis=0).astype(np.float32))
        else:
            concept_indices = torch.zeros((0,), dtype=torch.long)
            mask_stack = torch.zeros((0, self.mask_h, self.mask_w), dtype=torch.float32)
        return image, global_concepts, concept_indices, mask_stack, target


class OnTheFlySpatialSupervisionDataset(Dataset):
    """Spatial supervision dataset that parses per-image annotation JSONs on-demand.

    This avoids creating the large (tens of GB) SAVLG supervision cache on ImageNet.
    It trades disk IO and JSON parsing for reduced preprocessing and storage.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        indices: Sequence[int],
        transform,
        annotation_dir: str,
        concepts: Sequence[str],
        args,
    ):
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.transform = transform
        self.annotation_dir = str(annotation_dir)
        self.concepts = list(concepts)
        self.concept_to_idx = {c: i for i, c in enumerate(self.concepts)}
        self.threshold = float(getattr(args, "cbl_confidence_threshold", 0.15))
        self.mask_h = int(getattr(args, "mask_h", 7))
        self.mask_w = int(getattr(args, "mask_w", 7))
        self.args = args
        # Provide targets for downstream metrics (mirrors TransformedSubset/RawSubset behavior).
        self.targets = subset_targets(base_dataset, self.indices) if hasattr(base_dataset, "__len__") else None

    def __len__(self):
        return len(self.indices)

    def _ann_path(self, ann_idx: int) -> str:
        return os.path.join(self.annotation_dir, f"{int(ann_idx)}.json")

    def _parse_annotations(self, ann_idx: int):
        ann_path = self._ann_path(ann_idx)
        if not os.path.exists(ann_path):
            return []
        try:
            with open(ann_path, "r") as f:
                data = json.load(f)
        except Exception:
            return []
        if not isinstance(data, list):
            return []
        parsed = []
        for ann in data[1:]:
            if not isinstance(ann, dict):
                continue
            label = ann.get("label")
            if isinstance(label, str):
                label = data_utils.canonicalize_concept_label(label)
            cidx = self.concept_to_idx.get(label)
            if cidx is None:
                continue
            score = float(ann.get("logit", 0.0))
            box = ann.get("box")
            parsed.append((cidx, score, box))
        return parsed

    def __getitem__(self, idx):
        base_idx = int(self.indices[idx])
        image, target = self.base_dataset[base_idx]
        image_size = (int(image.size[0]), int(image.size[1])) if hasattr(image, "size") else None
        if self.transform is not None:
            image = self.transform(image)

        # Build per-image global concept scores + sparse masks for local supervision.
        global_scores = np.zeros((len(self.concepts),), dtype=np.float32)
        mask_dict: Dict[int, np.ndarray] = {}
        for cidx, score, box in self._parse_annotations(base_idx):
            if score > global_scores[cidx]:
                global_scores[cidx] = score
            if box is None or score < self.threshold or image_size is None:
                continue
            box_mask = _rasterize_box_target(box=box, image_size=image_size, args=self.args)
            if box_mask is None:
                continue
            existing = mask_dict.get(cidx)
            if existing is None:
                mask_dict[cidx] = box_mask
            else:
                np.maximum(existing, box_mask, out=existing)

        global_targets = _build_global_concept_targets(global_scores[None, :], self.args)[0]
        global_concepts = torch.from_numpy(global_targets.astype(np.float32))

        if mask_dict:
            keys = sorted(mask_dict.keys())
            concept_indices = torch.tensor(keys, dtype=torch.long)
            mask_stack = torch.from_numpy(
                np.stack([mask_dict[k] for k in keys], axis=0).astype(np.float32)
            )
        else:
            concept_indices = torch.zeros((0,), dtype=torch.long)
            mask_stack = torch.zeros((0, self.mask_h, self.mask_w), dtype=torch.float32)

        return image, global_concepts, concept_indices, mask_stack, int(target)


class CachedSpatialSupervisionDataset(Dataset):
    def __init__(
        self,
        cached_feats,
        labels: torch.Tensor,
        global_concept_targets: np.ndarray,
        mask_entries: List[Dict[int, np.ndarray]],
        mask_h: int,
        mask_w: int,
    ):
        self.cached_feats = cached_feats
        self.labels = labels.long()
        self.global_concept_targets = global_concept_targets.astype(np.float32)
        self.mask_entries = mask_entries
        self.mask_h = int(mask_h)
        self.mask_w = int(mask_w)

    def __len__(self):
        if isinstance(self.cached_feats, dict):
            first_key = next(iter(self.cached_feats))
            return int(self.cached_feats[first_key].shape[0])
        return int(self.cached_feats.shape[0])

    def __getitem__(self, idx):
        if isinstance(self.cached_feats, dict):
            feat_item = {
                key: value[idx]
                for key, value in self.cached_feats.items()
            }
        else:
            feat_item = self.cached_feats[idx]
        target = self.labels[idx]
        global_concepts = torch.from_numpy(self.global_concept_targets[idx])
        entry = self.mask_entries[idx]
        if entry:
            keys = sorted(entry.keys())
            concept_indices = torch.tensor(keys, dtype=torch.long)
            mask_stack = torch.from_numpy(np.stack([entry[k] for k in keys], axis=0).astype(np.float32))
        else:
            concept_indices = torch.zeros((0,), dtype=torch.long)
            mask_stack = torch.zeros((0, self.mask_h, self.mask_w), dtype=torch.float32)
        return feat_item, global_concepts, concept_indices, mask_stack, target


class CachedFeatureLabelDataset(Dataset):
    def __init__(self, cached_feats, labels: torch.Tensor):
        self.cached_feats = cached_feats
        self.labels = labels.long()

    def __len__(self):
        if isinstance(self.cached_feats, dict):
            first_key = next(iter(self.cached_feats))
            return int(self.cached_feats[first_key].shape[0])
        return int(self.cached_feats.shape[0])

    def __getitem__(self, idx):
        if isinstance(self.cached_feats, dict):
            feat_item = {key: value[idx] for key, value in self.cached_feats.items()}
        else:
            feat_item = self.cached_feats[idx]
        return feat_item, self.labels[idx]


def collate_spatial_batch(batch):
    images, global_concepts, c_idx, c_mask, labels = zip(*batch)
    if isinstance(images[0], dict):
        images = {
            key: torch.stack([sample[key] for sample in images], dim=0)
            for key in images[0].keys()
        }
    else:
        images = torch.stack(images, dim=0)
    global_concepts = torch.stack(global_concepts, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)

    max_k = max(x.numel() for x in c_idx)
    bsz = len(batch)
    if max_k == 0:
        idx_pad = torch.full((bsz, 1), -1, dtype=torch.long)
        mask_pad = torch.zeros((bsz, 1, c_mask[0].shape[-2] if c_mask else 1, c_mask[0].shape[-1] if c_mask else 1), dtype=torch.float32)
        valid = torch.zeros((bsz, 1), dtype=torch.bool)
    else:
        mask_h = c_mask[0].shape[-2]
        mask_w = c_mask[0].shape[-1]
        idx_pad = torch.full((bsz, max_k), -1, dtype=torch.long)
        mask_pad = torch.zeros((bsz, max_k, mask_h, mask_w), dtype=torch.float32)
        valid = torch.zeros((bsz, max_k), dtype=torch.bool)
        for i, (idx_i, mask_i) in enumerate(zip(c_idx, c_mask)):
            k = idx_i.numel()
            if k == 0:
                continue
            idx_pad[i, :k] = idx_i
            mask_pad[i, :k] = mask_i
            valid[i, :k] = True
    return images, global_concepts, idx_pad, mask_pad, valid, labels


def _savlg_feature_cache_enabled(args) -> bool:
    return bool(
        getattr(args, "use_activation_cache", False)
        and not bool(getattr(args, "cbl_finetune", False))
        and float(getattr(args, "crop_to_concept_prob", 0.0)) == 0.0
    )


def _savlg_feature_cache_path(
    args,
    base_dataset: Dataset,
    split_name: str,
) -> str:
    cache_dir = os.path.join(
        getattr(args, "activation_dir", "saved_activations"),
        "savlg_feature_cache",
    )
    os.makedirs(cache_dir, exist_ok=True)
    if hasattr(base_dataset, "base_dataset") and hasattr(base_dataset, "indices"):
        root_dataset = base_dataset.base_dataset
        sample_indices = list(base_dataset.indices)
    else:
        root_dataset = base_dataset
        sample_indices = list(range(len(base_dataset)))
    dataset_name = getattr(root_dataset, "dataset_name", args.dataset)
    split_suffix = getattr(root_dataset, "split_suffix", split_name)
    sample_hash = hashlib.sha1(
        ",".join(map(str, sample_indices)).encode("utf-8")
    ).hexdigest()[:16]
    preprocess_repr = repr(getattr(base_dataset, "preprocess", None))
    preprocess_hash = hashlib.sha1(preprocess_repr.encode("utf-8")).hexdigest()[:16]
    metadata = {
        "dataset": dataset_name,
        "split": split_suffix,
        "backbone": args.backbone,
        "feature_layer": args.feature_layer,
        "spatial_stage": getattr(args, "savlg_spatial_stage", "conv5"),
        "branch_arch": getattr(args, "savlg_branch_arch", "shared"),
        "spatial_branch_mode": getattr(args, "savlg_spatial_branch_mode", "shared_stage"),
        "global_head_mode": getattr(args, "savlg_global_head_mode", "spatial_pool"),
        "sample_hash": sample_hash,
        "preprocess_hash": preprocess_hash,
        "cache_tag": split_name,
    }
    digest = hashlib.sha1(
        json.dumps(metadata, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return os.path.join(cache_dir, f"{dataset_name}_{split_suffix}_{digest}.pt")


def get_or_create_savlg_feature_cache(
    args,
    backbone: SpatialBackbone,
    dataset: Dataset,
    split_name: str,
):
    cache_path = _savlg_feature_cache_path(args, dataset, split_name)
    if os.path.exists(cache_path):
        logger.info("Loading cached SAVLG backbone features from {}", cache_path)
        return torch.load(cache_path, weights_only=False)

    logger.info("Caching SAVLG backbone features to {}", cache_path)
    cache_loader_kwargs = {
        "batch_size": args.cbl_batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": True,
    }
    if int(args.num_workers) > 0:
        cache_loader_kwargs["persistent_workers"] = True
    loader = DataLoader(dataset, **cache_loader_kwargs)
    cached_labels = []
    if savlg_uses_multiscale_branch(args) or savlg_uses_split_stage_dual_branch(args):
        feat_store: Dict[str, List[torch.Tensor]] = {}
    else:
        feat_store = {"__single__": []}
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f"SAVLG feature cache ({split_name})"):
            images = images.to(args.device)
            feats = forward_savlg_backbone(backbone, images, args)
            if isinstance(feats, dict):
                for key, value in feats.items():
                    feat_store.setdefault(key, []).append(value.detach().cpu())
            else:
                feat_store["__single__"].append(feats.detach().cpu())
            cached_labels.append(labels.cpu())
    cached = {
        "feats": (
            {key: torch.cat(value, dim=0) for key, value in feat_store.items()}
            if "__single__" not in feat_store
            else torch.cat(feat_store["__single__"], dim=0)
        ),
        "labels": torch.cat(cached_labels, dim=0),
    }
    torch.save(cached, cache_path)
    return cached


def build_savlg_feature_cache_in_memory(
    args,
    backbone: SpatialBackbone,
    dataset: Dataset,
    split_name: str,
):
    logger.info(
        "Building SAVLG backbone features in memory for deterministic training ({})",
        split_name,
    )
    cache_loader_kwargs = {
        "batch_size": args.cbl_batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": True,
    }
    if int(args.num_workers) > 0:
        cache_loader_kwargs["persistent_workers"] = True
    loader = DataLoader(dataset, **cache_loader_kwargs)
    cached_labels = []
    if savlg_uses_multiscale_branch(args) or savlg_uses_split_stage_dual_branch(args):
        feat_store: Dict[str, List[torch.Tensor]] = {}
    else:
        feat_store = {"__single__": []}
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f"SAVLG feature cache ({split_name})"):
            images = images.to(args.device)
            feats = forward_savlg_backbone(backbone, images, args)
            if isinstance(feats, dict):
                for key, value in feats.items():
                    feat_store.setdefault(key, []).append(value.detach().cpu())
            else:
                feat_store["__single__"].append(feats.detach().cpu())
            cached_labels.append(labels.cpu())
    return {
        "feats": (
            {key: torch.cat(value, dim=0) for key, value in feat_store.items()}
            if "__single__" not in feat_store
            else torch.cat(feat_store["__single__"], dim=0)
        ),
        "labels": torch.cat(cached_labels, dim=0),
    }


def _savlg_batch_already_features(batch_input) -> bool:
    if isinstance(batch_input, dict):
        return True
    if not isinstance(batch_input, torch.Tensor):
        return False
    return batch_input.ndim == 4 and int(batch_input.shape[1]) != 3


def _move_savlg_feats_to_device(feats, device: str):
    if isinstance(feats, dict):
        return {key: value.to(device, non_blocking=True) for key, value in feats.items()}
    return feats.to(device, non_blocking=True)


class SAVLGCBM(nn.Module):
    def __init__(self, backbone: SpatialBackbone, concept_layer: nn.Module, args):
        super().__init__()
        self.backbone = backbone
        self.concept_layer = concept_layer
        self.args = args

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = forward_savlg_backbone(self.backbone, x, self.args)
        global_outputs, spatial_maps = forward_savlg_concept_layer(self.concept_layer, feats)
        _, _, final_logits = compute_savlg_concept_logits(
            global_outputs,
            spatial_maps,
            self.args,
            concept_layer=self.concept_layer,
        )
        return final_logits, spatial_maps


def savlg_uses_multiscale_branch(args) -> bool:
    return str(getattr(args, "savlg_spatial_branch_mode", "shared_stage")).lower() == "multiscale_conv45"


def savlg_uses_split_stage_dual_branch(args) -> bool:
    if savlg_uses_multiscale_branch(args):
        return False
    if not savlg_uses_vlg_global_head(args):
        return False
    if str(getattr(args, "savlg_branch_arch", "shared")).lower() != "dual":
        return False
    return str(getattr(args, "savlg_spatial_stage", "conv5")).lower() != "conv5"


def savlg_uses_vlg_global_head(args) -> bool:
    return str(getattr(args, "savlg_global_head_mode", "spatial_pool")).lower() == "vlg_linear"


def build_savlg_global_head(args, in_features: int, n_concepts: int) -> nn.Module:
    if savlg_uses_vlg_global_head(args):
        # Match the original VLG-CBM concept path when savlg_global_hidden_layers=0:
        # GAP over conv5 features followed by a linear concept layer. Optionally
        # extend this with hidden Linear->BN->ReLU->Linear blocks for ablations.
        num_hidden = max(0, int(getattr(args, "savlg_global_hidden_layers", 0)))
        use_bn = bool(getattr(args, "savlg_global_use_batchnorm", False))
        hidden_dim = int(getattr(args, "savlg_global_hidden_dim", n_concepts) or n_concepts)
        hidden_dim = max(1, hidden_dim)
        if num_hidden <= 0:
            return ConceptLayer(
                in_features=in_features,
                out_features=n_concepts,
                num_hidden=0,
                bias=True,
                device=args.device,
            )
        layers: List[nn.Module] = [nn.Linear(in_features, hidden_dim, bias=True)]
        for hidden_idx in range(num_hidden):
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            out_dim = n_concepts if hidden_idx == num_hidden - 1 else hidden_dim
            layers.append(nn.Linear(hidden_dim, out_dim, bias=True))
            hidden_dim = out_dim
        model = nn.Sequential(*layers).to(args.device)
        logger.info(model)
        return model
    return build_single_spatial_concept_layer(args, in_features, n_concepts)


def apply_savlg_global_head(global_layer: nn.Module, feats, args) -> torch.Tensor:
    if isinstance(feats, dict):
        # Dual/multiscale branches keep the global path on conv5 features.
        feats = feats["conv5"]
    if savlg_uses_vlg_global_head(args):
        pooled_feats = feats.mean(dim=[2, 3])
        return global_layer(pooled_feats)
    return global_layer(feats)


class DualBranchMixedConceptLayer(nn.Module):
    def __init__(
        self,
        global_layer: nn.Module,
        spatial_layer: nn.Module,
        args,
        spatial_stage: Optional[str] = None,
    ):
        super().__init__()
        self.global_layer = global_layer
        self.spatial_layer = spatial_layer
        self.args = args
        self.spatial_stage = str(spatial_stage or getattr(args, "savlg_spatial_stage", "conv5")).lower()
        if getattr(args, "savlg_learnable_alpha", False):
            init_alpha = max(1e-4, float(getattr(args, "savlg_residual_spatial_alpha", 0.1)))
            self.log_spatial_scale = nn.Parameter(torch.tensor(math.log(init_alpha)))

    def _spatial_feats(self, x):
        if isinstance(x, dict):
            return x[self.spatial_stage]
        return x

    def forward(self, x) -> torch.Tensor:
        return self.spatial_layer(self._spatial_feats(x))

    def forward_global(self, x):
        return apply_savlg_global_head(self.global_layer, x, self.args)

    def forward_spatial(self, x) -> torch.Tensor:
        return self.spatial_layer(self._spatial_feats(x))

    def forward_both(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward_global(x), self.forward_spatial(x)


class MultiScaleSAVLGConceptLayer(nn.Module):
    def __init__(self, args, backbone: SpatialBackbone, n_concepts: int):
        super().__init__()
        if str(getattr(args, "savlg_branch_arch", "shared")).lower() != "dual":
            raise ValueError("Multi-scale SAVLG spatial fusion requires savlg_branch_arch='dual'.")
        if str(getattr(args, "savlg_spatial_stage", "conv5")).lower() != "conv5":
            raise ValueError("Multi-scale SAVLG spatial fusion keeps the global branch on conv5.")

        conv4_dim = backbone.get_stage_dim("conv4")
        conv5_dim = backbone.get_stage_dim("conv5")
        fusion_dim = int(getattr(args, "savlg_multiscale_fusion_dim", conv5_dim) or conv5_dim)

        self.global_layer = build_savlg_global_head(args, conv5_dim, n_concepts)
        self.conv4_proj = nn.Conv2d(conv4_dim, fusion_dim, kernel_size=1, bias=False).to(args.device)
        self.conv5_proj = nn.Conv2d(conv5_dim, fusion_dim, kernel_size=1, bias=False).to(args.device)
        self.spatial_layer = build_single_spatial_concept_layer(args, fusion_dim, n_concepts)
        self.args = args
        if getattr(args, "savlg_learnable_alpha", False):
            init_alpha = max(1e-4, float(getattr(args, "savlg_residual_spatial_alpha", 0.1)))
            self.log_spatial_scale = nn.Parameter(torch.tensor(math.log(init_alpha)))

    def _fuse_spatial_features(self, feats) -> torch.Tensor:
        if not isinstance(feats, dict):
            raise TypeError("MultiScaleSAVLGConceptLayer expects a stage dict with conv4 and conv5 features.")
        conv4 = feats["conv4"]
        conv5 = feats["conv5"]
        conv5_up = F.interpolate(
            self.conv5_proj(conv5),
            size=conv4.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        fused = self.conv4_proj(conv4) + conv5_up
        return F.relu(fused, inplace=False)

    def forward(self, feats) -> torch.Tensor:
        return self.spatial_layer(self._fuse_spatial_features(feats))

    def forward_global(self, feats) -> torch.Tensor:
        return apply_savlg_global_head(self.global_layer, feats, self.args)

    def forward_spatial(self, feats) -> torch.Tensor:
        return self.spatial_layer(self._fuse_spatial_features(feats))

    def forward_both(self, feats) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward_global(feats), self.forward_spatial(feats)


def build_savlg_concept_layer(args, backbone: SpatialBackbone, n_concepts: int) -> nn.Module:
    if savlg_uses_multiscale_branch(args):
        return MultiScaleSAVLGConceptLayer(args, backbone, n_concepts).to(args.device)
    if savlg_uses_vlg_global_head(args):
        branch_arch = str(getattr(args, "savlg_branch_arch", "shared")).lower()
        if branch_arch != "dual":
            raise ValueError("savlg_global_head_mode='vlg_linear' requires savlg_branch_arch='dual'.")
        spatial_stage = str(getattr(args, "savlg_spatial_stage", "conv5")).lower()
        return DualBranchMixedConceptLayer(
            global_layer=build_savlg_global_head(args, backbone.get_stage_dim("conv5"), n_concepts),
            spatial_layer=build_single_spatial_concept_layer(
                args,
                backbone.get_stage_dim(spatial_stage),
                n_concepts,
            ),
            args=args,
            spatial_stage=spatial_stage,
        ).to(args.device)
    return build_spatial_concept_layer(args, backbone.output_dim, n_concepts)


def _collect_linear_layers(module: nn.Module) -> List[nn.Linear]:
    return [submodule for submodule in module.modules() if isinstance(submodule, nn.Linear)]


def _collect_pointwise_conv_layers(module: nn.Module) -> List[nn.Conv2d]:
    return [
        submodule
        for submodule in module.modules()
        if isinstance(submodule, nn.Conv2d) and tuple(submodule.kernel_size) == (1, 1)
    ]


def maybe_initialize_savlg_from_vlg(
    args,
    concept_layer: nn.Module,
    concepts: Sequence[str],
) -> None:
    init_path = str(getattr(args, "savlg_init_from_vlg_path", "") or "").strip()
    if not init_path:
        return
    if not os.path.isdir(init_path):
        raise FileNotFoundError(
            f"SAVLG VLG-initialization path does not exist: {init_path}"
        )

    vlg_concepts = data_utils.get_concepts(os.path.join(init_path, "concepts.txt"))
    vlg_concept_to_idx = {concept: idx for idx, concept in enumerate(vlg_concepts)}
    matched_pairs = [
        (target_idx, vlg_concept_to_idx[concept])
        for target_idx, concept in enumerate(concepts)
        if concept in vlg_concept_to_idx
    ]
    if not matched_pairs:
        logger.warning(
            "SAVLG VLG warm-start skipped: no overlapping concepts between current run and {}",
            init_path,
        )
        return

    vlg_cbl = ConceptLayer.from_pretrained(init_path, device=args.device)
    source_linears = _collect_linear_layers(vlg_cbl)
    if len(source_linears) != 1:
        logger.warning(
            "SAVLG VLG warm-start expects a single-linear VLG concept layer, found {} linear layers in {}. Skipping.",
            len(source_linears),
            init_path,
        )
        return
    source_linear = source_linears[0]

    target_global = getattr(concept_layer, "global_layer", None)
    if target_global is not None:
        target_linears = _collect_linear_layers(target_global)
        if len(target_linears) == 1:
            target_linear = target_linears[0]
            if (
                target_linear.in_features == source_linear.in_features
                and target_linear.out_features == len(concepts)
            ):
                with torch.no_grad():
                    for target_idx, source_idx in matched_pairs:
                        target_linear.weight[target_idx].copy_(source_linear.weight[source_idx])
                        if target_linear.bias is not None and source_linear.bias is not None:
                            target_linear.bias[target_idx].copy_(source_linear.bias[source_idx])
                logger.info(
                    "Initialized SAVLG global head from VLG checkpoint {} for {}/{} concepts.",
                    init_path,
                    len(matched_pairs),
                    len(concepts),
                )
            else:
                logger.warning(
                    "SAVLG VLG warm-start skipped for global head due to shape mismatch: target Linear({}, {}) vs source Linear({}, {}).",
                    target_linear.in_features,
                    target_linear.out_features,
                    source_linear.in_features,
                    source_linear.out_features,
                )
        else:
            logger.warning(
                "SAVLG VLG warm-start skipped for global head: expected one target linear layer, found {}.",
                len(target_linears),
            )

    if not bool(getattr(args, "savlg_init_spatial_from_vlg", False)):
        return

    target_spatial = getattr(concept_layer, "spatial_layer", None)
    if target_spatial is None:
        return
    target_convs = _collect_pointwise_conv_layers(target_spatial)
    if len(target_convs) != 1:
        logger.warning(
            "SAVLG VLG warm-start skipped for spatial head: expected one pointwise conv layer, found {}.",
            len(target_convs),
        )
        return
    target_conv = target_convs[0]
    if (
        target_conv.in_channels != source_linear.in_features
        or target_conv.out_channels != len(concepts)
    ):
        logger.warning(
            "SAVLG VLG warm-start skipped for spatial head due to shape mismatch: target Conv({}, {}) vs source Linear({}, {}).",
            target_conv.in_channels,
            target_conv.out_channels,
            source_linear.in_features,
            source_linear.out_features,
        )
        return
    with torch.no_grad():
        for target_idx, source_idx in matched_pairs:
            target_conv.weight[target_idx, :, 0, 0].copy_(source_linear.weight[source_idx])
            if target_conv.bias is not None and source_linear.bias is not None:
                target_conv.bias[target_idx].copy_(source_linear.bias[source_idx])
    logger.info(
        "Initialized SAVLG spatial 1x1 head from VLG checkpoint {} for {}/{} concepts.",
        init_path,
        len(matched_pairs),
        len(concepts),
    )


def maybe_freeze_savlg_global_head(args, concept_layer: nn.Module) -> None:
    if not bool(getattr(args, "savlg_freeze_global_head", False)):
        return
    global_layer = getattr(concept_layer, "global_layer", None)
    if global_layer is None:
        logger.warning(
            "Requested SAVLG global-head freeze, but concept layer has no global_layer attribute. Skipping."
        )
        return
    num_params = 0
    for parameter in global_layer.parameters():
        parameter.requires_grad = False
        num_params += parameter.numel()
    global_layer.eval()
    logger.info("Froze SAVLG global head ({} parameters).", num_params)


def forward_savlg_backbone(
    backbone: SpatialBackbone,
    images: torch.Tensor,
    args,
):
    if savlg_uses_multiscale_branch(args):
        return backbone.forward_multistage(images, ("conv4", "conv5"))
    if savlg_uses_split_stage_dual_branch(args):
        spatial_stage = str(getattr(args, "savlg_spatial_stage", "conv5")).lower()
        requested = ("conv5", spatial_stage) if spatial_stage != "conv5" else ("conv5",)
        return backbone.forward_multistage(images, requested)
    return backbone(images)


def forward_savlg_concept_layer(
    concept_layer: nn.Module,
    feats,
) -> Tuple[torch.Tensor, torch.Tensor]:
    forward_both = getattr(concept_layer, "forward_both", None)
    if callable(forward_both):
        global_maps, spatial_maps = forward_both(feats)
        return global_maps, spatial_maps
    spatial_maps = concept_layer(feats)
    return spatial_maps, spatial_maps


def pool_global_concept_outputs(outputs: torch.Tensor, args) -> torch.Tensor:
    if outputs.ndim == 2:
        return outputs
    return pool_concept_maps(outputs, args)


def pool_concept_maps(maps: torch.Tensor, args) -> torch.Tensor:
    pooling = str(getattr(args, "savlg_pooling", "avg")).lower()
    if pooling == "avg":
        return F.adaptive_avg_pool2d(maps, 1).flatten(1)
    if pooling != "topk":
        raise ValueError(f"Unsupported SAVLG pooling mode: {pooling}")

    flat = maps.flatten(2)
    num_patches = flat.shape[-1]
    topk_fraction = float(getattr(args, "savlg_topk_fraction", 0.2))
    topk_fraction = min(max(topk_fraction, 0.0), 1.0)
    k = max(1, int(math.ceil(num_patches * topk_fraction)))
    values, _ = flat.topk(k=k, dim=-1)
    return values.mean(dim=-1)


def pool_local_mil_logits(map_logits: torch.Tensor, args) -> torch.Tensor:
    pooling = str(getattr(args, "savlg_local_pooling", "lse")).lower()
    flat = map_logits.flatten(2)
    if pooling == "lse":
        temperature = float(getattr(args, "savlg_mil_temperature", 1.0))
        temperature = max(temperature, 1e-6)
        num_patches = flat.shape[-1]
        pooled = temperature * torch.logsumexp(flat / temperature, dim=-1)
        return pooled - temperature * math.log(max(num_patches, 1))
    if pooling == "topk":
        num_patches = flat.shape[-1]
        topk_fraction = float(getattr(args, "savlg_mil_topk_fraction", 0.2))
        topk_fraction = min(max(topk_fraction, 0.0), 1.0)
        k = max(1, int(math.ceil(num_patches * topk_fraction)))
        values, _ = flat.topk(k=k, dim=-1)
        return values.mean(dim=-1)
    raise ValueError(f"Unsupported SAVLG local MIL pooling mode: {pooling}")


def savlg_residual_coupling_enabled(args) -> bool:
    return abs(float(getattr(args, "savlg_residual_spatial_alpha", 0.0))) > 0.0


def pool_residual_spatial_logits(map_logits: torch.Tensor, args) -> torch.Tensor:
    pooling = str(getattr(args, "savlg_residual_spatial_pooling", "lse")).lower()
    flat = map_logits.flatten(2)
    if pooling == "lse":
        temperature = float(getattr(args, "savlg_mil_temperature", 1.0))
        temperature = max(temperature, 1e-6)
        num_patches = flat.shape[-1]
        pooled = temperature * torch.logsumexp(flat / temperature, dim=-1)
        return pooled - temperature * math.log(max(num_patches, 1))
    if pooling == "avg":
        return flat.mean(dim=-1)
    if pooling == "topk":
        num_patches = flat.shape[-1]
        topk_fraction = float(
            getattr(
                args,
                "savlg_residual_topk_fraction",
                getattr(args, "savlg_mil_topk_fraction", 0.2),
            )
        )
        topk_fraction = min(max(topk_fraction, 0.0), 1.0)
        k = max(1, int(math.ceil(num_patches * topk_fraction)))
        values, _ = flat.topk(k=k, dim=-1)
        return values.mean(dim=-1)
    raise ValueError(
        f"Unsupported SAVLG residual spatial pooling mode: {pooling}. "
        "Supported modes are lse, avg, and topk."
    )


def compute_savlg_concept_logits(
    global_outputs: torch.Tensor,
    spatial_maps: torch.Tensor,
    args,
    concept_layer: Optional[nn.Module] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    global_logits = pool_global_concept_outputs(global_outputs, args)
    if savlg_residual_coupling_enabled(args):
        spatial_logits = pool_residual_spatial_logits(spatial_maps, args)
        if concept_layer is not None and hasattr(concept_layer, "log_spatial_scale"):
            alpha = concept_layer.log_spatial_scale.exp()
        else:
            alpha = float(getattr(args, "savlg_residual_spatial_alpha", 0.0))
        final_logits = global_logits + alpha * spatial_logits
        return global_logits, spatial_logits, final_logits
    spatial_logits = torch.zeros_like(global_logits)
    return global_logits, spatial_logits, global_logits


def pool_spatial_teacher_logits(map_logits: torch.Tensor, args) -> torch.Tensor:
    if bool(getattr(args, "savlg_use_local_mil", False)):
        return pool_local_mil_logits(map_logits, args)
    return pool_concept_maps(map_logits, args)


def compute_local_trust_weights(
    global_concept_targets: torch.Tensor,
    args,
) -> torch.Tensor:
    mode = str(getattr(args, "savlg_local_weight_mode", "uniform")).lower()
    if mode == "uniform":
        return torch.ones_like(global_concept_targets)
    if mode != "confidence":
        raise ValueError(f"Unsupported SAVLG local weighting mode: {mode}")

    threshold = float(getattr(args, "cbl_confidence_threshold", 0.15))
    denom = max(1.0 - threshold, 1e-6)
    floor = float(getattr(args, "savlg_local_weight_floor", 0.25))
    floor = min(max(floor, 0.0), 1.0)
    power = max(float(getattr(args, "savlg_local_weight_power", 1.0)), 1e-6)
    normalized = ((global_concept_targets - threshold) / denom).clamp(0.0, 1.0)
    return floor + (1.0 - floor) * normalized.pow(power)


def compute_global_spatial_consistency_loss(
    pooled_logits: torch.Tensor,
    spatial_teacher_logits: torch.Tensor,
    global_concept_targets: torch.Tensor,
    local_trust_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    teacher_probs = torch.sigmoid(spatial_teacher_logits.detach())
    consistency = F.binary_cross_entropy_with_logits(
        pooled_logits,
        teacher_probs,
        reduction="none",
    )
    pair_weights = (global_concept_targets > 0.0).to(consistency.dtype)
    if local_trust_weights is not None:
        pair_weights = pair_weights * local_trust_weights.to(consistency.dtype)
    return (consistency * pair_weights).sum() / torch.clamp(pair_weights.sum(), min=1.0)


def _soft_box_distribution_targets(mask_targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    target_mass = mask_targets.clamp(min=0.0)
    target_mass_sum = target_mass.sum(dim=1, keepdim=True)
    valid = target_mass_sum.squeeze(1) > 0.0
    target_dist = torch.zeros_like(target_mass)
    if bool(valid.any()):
        target_dist[valid] = target_mass[valid] / torch.clamp(
            target_mass_sum[valid],
            min=1e-6,
        )
    return target_dist, valid


def _outside_mass_penalty(
    map_logits: torch.Tensor,
    mask_targets: torch.Tensor,
) -> torch.Tensor:
    pred_dist = F.softmax(map_logits, dim=1)
    outside_weight = 1.0 - mask_targets.clamp(0.0, 1.0)
    return (pred_dist * outside_weight).sum(dim=1)


def _coverage_penalty(
    map_logits: torch.Tensor,
    mask_targets: torch.Tensor,
) -> torch.Tensor:
    pred_prob = torch.sigmoid(map_logits)
    target_mass = mask_targets.clamp(min=0.0)
    target_mass_sum = target_mass.sum(dim=1)
    covered_mass = (pred_prob * target_mass).sum(dim=1)
    coverage = covered_mass / torch.clamp(target_mass_sum, min=1e-6)
    return 1.0 - coverage


def _absent_topk_penalty(
    map_logits: torch.Tensor,
    global_concept_targets: torch.Tensor,
    topk_fraction: float,
) -> torch.Tensor:
    flat_probs = torch.sigmoid(map_logits).flatten(2)
    num_locations = flat_probs.shape[-1]
    k = max(1, int(math.ceil(float(topk_fraction) * float(num_locations))))
    topk_vals = flat_probs.topk(k, dim=-1).values.mean(dim=-1)
    absent_mask = global_concept_targets <= 0.5
    if not bool(absent_mask.any()):
        return map_logits.sum() * 0.0
    return topk_vals[absent_mask].mean()


def compute_spatial_losses(
    pooled_logits: torch.Tensor,
    map_logits: torch.Tensor,
    global_concept_targets: torch.Tensor,
    mask_indices: torch.Tensor,
    mask_targets: torch.Tensor,
    mask_valid: torch.Tensor,
    global_bce_pos_weight: float = 1.0,
    patch_bce_pos_weight: float = 1.0,
    loss_dice_w: float = 0.0,
    local_mil_logits: Optional[torch.Tensor] = None,
    local_bce_pos_weight: float = 1.0,
    local_trust_weights: Optional[torch.Tensor] = None,
    local_loss_mode: str = "bce",
    outside_penalty_w: float = 0.0,
    coverage_w: float = 0.0,
    absent_topk_w: float = 0.0,
    absent_topk_fraction: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    global_concept_bce = F.binary_cross_entropy_with_logits(
        pooled_logits, global_concept_targets, reduction="none"
    )
    global_pos_w = torch.where(
        global_concept_targets > 0.5,
        torch.full_like(global_concept_targets, float(global_bce_pos_weight)),
        torch.ones_like(global_concept_targets),
    )
    loss_global_concept = (global_concept_bce * global_pos_w).sum() / torch.clamp(
        global_pos_w.sum(), min=1.0
    )
    map_logits = F.interpolate(
        map_logits,
        size=mask_targets.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    batch_bce = []
    batch_dice = []
    batch_outside = []
    batch_coverage = []
    local_loss_mode = str(local_loss_mode).lower()
    for b in range(map_logits.shape[0]):
        valid = mask_valid[b]
        if not bool(valid.any()):
            continue
        cidx = mask_indices[b][valid]
        pred = map_logits[b].index_select(0, cidx)
        tgt = mask_targets[b][valid].to(pred.dtype)
        concept_weights = torch.ones((pred.shape[0],), device=pred.device, dtype=pred.dtype)
        if local_trust_weights is not None:
            concept_weights = local_trust_weights[b].index_select(0, cidx).to(pred.dtype)
        if local_loss_mode == "bce":
            bce_raw = F.binary_cross_entropy_with_logits(pred, tgt, reduction="none")
            patch_pos_w = torch.where(
                tgt > 0.5,
                torch.full_like(tgt, float(patch_bce_pos_weight)),
                torch.ones_like(tgt),
            )
            patch_pos_w_flat = patch_pos_w.flatten(1)
            per_concept_mask_loss = (bce_raw.flatten(1) * patch_pos_w_flat).sum(dim=1) / torch.clamp(
                patch_pos_w_flat.sum(dim=1),
                min=1.0,
            )
        elif local_loss_mode == "containment":
            pred_prob = torch.sigmoid(pred).flatten(1)
            tgt_flat = tgt.flatten(1).clamp(min=0.0)
            inside_mass = (pred_prob * tgt_flat).sum(dim=1)
            total_mass = pred_prob.sum(dim=1)
            per_concept_mask_loss = 1.0 - (inside_mass / (total_mass + 1e-6))
        elif local_loss_mode == "soft_align":
            pred_flat = pred.flatten(1)
            tgt_flat = tgt.flatten(1).clamp(min=0.0)
            tgt_dist, valid_targets = _soft_box_distribution_targets(tgt_flat)
            per_concept_mask_loss = torch.zeros(
                (pred_flat.shape[0],),
                device=pred.device,
                dtype=pred.dtype,
            )
            if bool(valid_targets.any()):
                pred_log_dist = F.log_softmax(pred_flat[valid_targets], dim=1)
                per_concept_mask_loss[valid_targets] = F.kl_div(
                    pred_log_dist,
                    tgt_dist[valid_targets],
                    reduction="none",
                ).sum(dim=1)
        else:
            raise ValueError(f"Unsupported SAVLG local loss mode: {local_loss_mode}")
        mask_loss = (per_concept_mask_loss * concept_weights).sum() / torch.clamp(
            concept_weights.sum(),
            min=1.0,
        )
        batch_bce.append(mask_loss)
        if outside_penalty_w > 0.0:
            outside_loss = (
                _outside_mass_penalty(pred.flatten(1), tgt.flatten(1).to(pred.dtype)) * concept_weights
            ).sum() / torch.clamp(
                concept_weights.sum(),
                min=1.0,
            )
            batch_outside.append(outside_loss)
        if coverage_w > 0.0:
            coverage_loss = (
                _coverage_penalty(pred.flatten(1), tgt.flatten(1).to(pred.dtype)) * concept_weights
            ).sum() / torch.clamp(
                concept_weights.sum(),
                min=1.0,
            )
            batch_coverage.append(coverage_loss)

        if loss_dice_w > 0.0:
            pred_prob = torch.sigmoid(pred)
            pred_flat = pred_prob.flatten(1)
            tgt_flat = tgt.flatten(1)
            intersection = (pred_flat * tgt_flat).sum(dim=1)
            denom = pred_flat.sum(dim=1) + tgt_flat.sum(dim=1)
            per_concept_dice = 1.0 - ((2.0 * intersection + 1e-6) / (denom + 1e-6))
            batch_dice.append(
                (per_concept_dice * concept_weights).sum()
                / torch.clamp(concept_weights.sum(), min=1.0)
            )
    if batch_bce:
        loss_mask = torch.stack(batch_bce).mean()
    else:
        loss_mask = map_logits.sum() * 0.0
    if batch_dice:
        loss_dice = torch.stack(batch_dice).mean()
    else:
        loss_dice = map_logits.sum() * 0.0
    if batch_outside:
        loss_outside = torch.stack(batch_outside).mean()
    else:
        loss_outside = map_logits.sum() * 0.0
    if batch_coverage:
        loss_coverage = torch.stack(batch_coverage).mean()
    else:
        loss_coverage = map_logits.sum() * 0.0
    if absent_topk_w > 0.0:
        loss_absent_topk = _absent_topk_penalty(
            map_logits,
            global_concept_targets,
            topk_fraction=absent_topk_fraction,
        )
    else:
        loss_absent_topk = map_logits.sum() * 0.0
    if local_mil_logits is not None:
        local_bce = F.binary_cross_entropy_with_logits(
            local_mil_logits, global_concept_targets, reduction="none"
        )
        local_pos_w = torch.where(
            global_concept_targets > 0.5,
            torch.full_like(global_concept_targets, float(local_bce_pos_weight)),
            torch.ones_like(global_concept_targets),
        )
        local_pair_weights = torch.ones_like(local_bce)
        if local_trust_weights is not None:
            local_pair_weights = torch.where(
                global_concept_targets > 0.0,
                local_trust_weights.to(local_bce.dtype),
                torch.ones_like(local_bce),
            )
        local_weight = local_pos_w * local_pair_weights
        loss_local_mil = (local_bce * local_weight).sum() / torch.clamp(
            local_weight.sum(), min=1.0
        )
    else:
        loss_local_mil = map_logits.sum() * 0.0
    return (
        loss_global_concept,
        loss_mask,
        loss_dice,
        loss_local_mil,
        loss_outside,
        loss_coverage,
        loss_absent_topk,
    )


def compute_refinement_loss(
    map_logits: torch.Tensor,
    mask_indices: torch.Tensor,
    mask_targets: torch.Tensor,
    mask_valid: torch.Tensor,
    patch_bce_pos_weight: float = 1.0,
    local_trust_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    resized_logits = F.interpolate(
        map_logits,
        size=mask_targets.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    batch_losses = []
    for b in range(resized_logits.shape[0]):
        valid = mask_valid[b]
        if not bool(valid.any()):
            continue
        cidx = mask_indices[b][valid]
        pred = resized_logits[b].index_select(0, cidx)
        tgt = mask_targets[b][valid].to(pred.dtype)
        concept_weights = torch.ones((pred.shape[0],), device=pred.device, dtype=pred.dtype)
        if local_trust_weights is not None:
            concept_weights = local_trust_weights[b].index_select(0, cidx).to(pred.dtype)

        flat_pred = pred.detach().flatten(1)
        flat_tgt = tgt.flatten(1)
        support = flat_tgt > 0
        if not bool(support.any()):
            continue

        masked_scores = flat_pred.masked_fill(~support, float("-inf"))
        top_idx = masked_scores.argmax(dim=1)
        missing_support = ~support.any(dim=1)
        if bool(missing_support.any()):
            top_idx[missing_support] = flat_pred[missing_support].argmax(dim=1)

        pseudo = torch.zeros_like(flat_tgt)
        pseudo.scatter_(1, top_idx.unsqueeze(1), 1.0)
        pseudo = pseudo.view_as(tgt)

        bce_raw = F.binary_cross_entropy_with_logits(pred, pseudo, reduction="none")
        patch_pos_w = torch.where(
            pseudo > 0.5,
            torch.full_like(pseudo, float(patch_bce_pos_weight)),
            torch.ones_like(pseudo),
        )
        patch_pos_w_flat = patch_pos_w.flatten(1)
        per_concept_loss = (bce_raw.flatten(1) * patch_pos_w_flat).sum(dim=1) / torch.clamp(
            patch_pos_w_flat.sum(dim=1),
            min=1.0,
        )
        batch_losses.append(
            (per_concept_loss * concept_weights).sum()
            / torch.clamp(concept_weights.sum(), min=1.0)
        )

    if batch_losses:
        return torch.stack(batch_losses).mean()
    return map_logits.sum() * 0.0


def train_concept_head(
    args,
    backbone: SpatialBackbone,
    concept_layer: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    teacher: Optional[Dict[str, object]] = None,
) -> nn.Module:
    backbone.eval()
    for parameter in backbone.parameters():
        parameter.requires_grad = False

    maybe_freeze_savlg_global_head(args, concept_layer)

    trainable_params = [parameter for parameter in concept_layer.parameters() if parameter.requires_grad]
    if not trainable_params:
        raise RuntimeError("SAVLG concept-head training has no trainable parameters.")

    if args.cbl_optimizer == "adam":
        base_optimizer_cls = torch.optim.Adam
        optimizer_kwargs = dict(
            lr=args.cbl_lr,
            weight_decay=args.cbl_weight_decay,
        )
    elif args.cbl_optimizer == "sgd":
        base_optimizer_cls = torch.optim.SGD
        optimizer_kwargs = dict(
            lr=args.cbl_lr,
            weight_decay=args.cbl_weight_decay,
            momentum=0.9,
        )
    else:
        raise ValueError(f"Unsupported SAVLG optimizer: {args.cbl_optimizer}")
    if bool(getattr(args, "cbl_use_sam", False)):
        optimizer = SAM(
            trainable_params,
            base_optimizer_cls=base_optimizer_cls,
            rho=float(getattr(args, "cbl_sam_rho", 0.05)),
            adaptive=bool(getattr(args, "cbl_sam_adaptive", False)),
            **optimizer_kwargs,
        )
    else:
        optimizer = base_optimizer_cls(trainable_params, **optimizer_kwargs)
    scheduler = None
    if args.cbl_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, int(args.cbl_epochs)),
        )
    best_loss = float("inf")
    best_state = None
    early_stop_patience = int(getattr(args, "cbl_early_stop_patience", 0))
    min_epochs = max(0, int(getattr(args, "cbl_min_epochs", 0)))
    min_delta = float(getattr(args, "cbl_min_delta", 0.0))
    epochs_without_improvement = 0
    distill_weight = float(getattr(args, "savlg_distill_w", 0.0))
    refine_weight = float(getattr(args, "savlg_refine_w", 0.0))
    refine_warmup_epochs = int(getattr(args, "savlg_refine_warmup_epochs", 0))
    consistency_weight = float(getattr(args, "savlg_global_spatial_consistency_w", 0.0))
    consistency_warmup_epochs = int(getattr(args, "savlg_global_spatial_consistency_warmup_epochs", 0))
    consistency_enabled = (
        consistency_weight > 0.0
        and str(getattr(args, "savlg_branch_arch", "shared")).lower() == "dual"
    )
    global_concept_loss_weight = _savlg_global_concept_loss_weight(args)

    for epoch in range(int(args.cbl_epochs)):
        concept_layer.train()
        running = 0.0
        for images, global_concepts, idx_pad, mask_pad, valid_pad, _ in tqdm(
            train_loader, desc=f"SAVLG CBL epoch {epoch + 1}"
        ):
            global_concepts = global_concepts.to(args.device)
            idx_pad = idx_pad.to(args.device)
            mask_pad = mask_pad.to(args.device)
            valid_pad = valid_pad.to(args.device)

            def compute_train_loss():
                if _savlg_batch_already_features(images):
                    feats = _move_savlg_feats_to_device(images, args.device)
                    images_for_teacher = None
                else:
                    batch_images = images.to(args.device)
                    feats = forward_savlg_backbone(backbone, batch_images, args)
                    images_for_teacher = batch_images
                global_outputs, spatial_maps = forward_savlg_concept_layer(concept_layer, feats)
                _, _, final_logits = compute_savlg_concept_logits(
                    global_outputs,
                    spatial_maps,
                    args,
                    concept_layer=concept_layer,
                )
                local_trust_weights = compute_local_trust_weights(global_concepts, args)
                local_mil_logits = None
                if bool(getattr(args, "savlg_use_local_mil", False)):
                    local_mil_logits = pool_local_mil_logits(spatial_maps, args)
                spatial_teacher_logits = None
                if consistency_enabled:
                    spatial_teacher_logits = (
                        local_mil_logits
                        if local_mil_logits is not None
                        else pool_spatial_teacher_logits(spatial_maps, args)
                    )
                (
                    loss_global_concept,
                    loss_mask,
                    loss_dice,
                    loss_local_mil,
                    loss_outside,
                    loss_coverage,
                    loss_absent_topk,
                ) = compute_spatial_losses(
                    final_logits,
                    spatial_maps,
                    global_concepts,
                    idx_pad,
                    mask_pad,
                    valid_pad,
                    global_bce_pos_weight=float(getattr(args, "global_bce_pos_weight", 1.0)),
                    patch_bce_pos_weight=float(getattr(args, "patch_bce_pos_weight", 1.0)),
                    loss_dice_w=float(getattr(args, "loss_dice_w", 0.0)),
                    local_mil_logits=local_mil_logits,
                    local_bce_pos_weight=float(getattr(args, "local_bce_pos_weight", 1.0)),
                    local_trust_weights=local_trust_weights,
                    local_loss_mode=str(getattr(args, "savlg_local_loss_mode", "bce")),
                    outside_penalty_w=float(getattr(args, "savlg_outside_penalty_w", 0.0)),
                    coverage_w=float(getattr(args, "savlg_coverage_w", 0.0)),
                    absent_topk_w=float(getattr(args, "savlg_absent_topk_w", 0.0)),
                    absent_topk_fraction=float(getattr(args, "savlg_absent_topk_fraction", 0.1)),
                )
                loss_refine = spatial_maps.sum() * 0.0
                if refine_weight > 0.0 and epoch >= refine_warmup_epochs:
                    loss_refine = compute_refinement_loss(
                        spatial_maps,
                        idx_pad,
                        mask_pad,
                        valid_pad,
                        patch_bce_pos_weight=float(getattr(args, "patch_bce_pos_weight", 1.0)),
                        local_trust_weights=local_trust_weights,
                    )
                loss_consistency = final_logits.sum() * 0.0
                if consistency_enabled and epoch >= consistency_warmup_epochs:
                    loss_consistency = compute_global_spatial_consistency_loss(
                        final_logits,
                        spatial_teacher_logits,
                        global_concepts,
                        local_trust_weights=local_trust_weights,
                    )
                loss_distill = final_logits.sum() * 0.0
                if teacher is not None and distill_weight > 0.0:
                    if images_for_teacher is None:
                        raise RuntimeError(
                            "SAVLG teacher distillation is not supported with cached feature batches."
                        )
                    with torch.no_grad():
                        teacher_feats = teacher["backbone"](images_for_teacher)
                        teacher_logits = teacher["concept_layer"](teacher_feats).index_select(
                            1, teacher["indices"]
                        )
                        teacher_probs = torch.sigmoid(teacher_logits)
                    loss_distill = F.binary_cross_entropy_with_logits(
                        final_logits, teacher_probs, reduction="mean"
                    )
                return (
                    global_concept_loss_weight * loss_global_concept
                    + float(getattr(args, "loss_mask_w", 1.0)) * loss_mask
                    + float(getattr(args, "loss_dice_w", 0.0)) * loss_dice
                    + float(getattr(args, "loss_local_mil_w", 0.0)) * loss_local_mil
                    + float(getattr(args, "savlg_outside_penalty_w", 0.0)) * loss_outside
                    + float(getattr(args, "savlg_coverage_w", 0.0)) * loss_coverage
                    + float(getattr(args, "savlg_absent_topk_w", 0.0)) * loss_absent_topk
                    + refine_weight * loss_refine
                    + consistency_weight * loss_consistency
                    + distill_weight * loss_distill
                )

            loss = compute_train_loss()
            optimizer.zero_grad()
            loss.backward()
            if bool(getattr(args, "cbl_use_sam", False)):
                optimizer.first_step(zero_grad=True)
                second_loss = compute_train_loss()
                second_loss.backward()
                optimizer.second_step(zero_grad=True)
            else:
                optimizer.step()
            running += float(loss.item()) * global_concepts.size(0)

        train_loss = running / max(len(train_loader.dataset), 1)
        concept_layer.eval()
        with torch.no_grad():
            val_running = 0.0
            for images, global_concepts, idx_pad, mask_pad, valid_pad, _ in val_loader:
                if _savlg_batch_already_features(images):
                    feats = _move_savlg_feats_to_device(images, args.device)
                    images_for_teacher = None
                else:
                    images = images.to(args.device)
                    feats = forward_savlg_backbone(backbone, images, args)
                    images_for_teacher = images
                global_concepts = global_concepts.to(args.device)
                idx_pad = idx_pad.to(args.device)
                mask_pad = mask_pad.to(args.device)
                valid_pad = valid_pad.to(args.device)
                global_outputs, spatial_maps = forward_savlg_concept_layer(concept_layer, feats)
                _, _, final_logits = compute_savlg_concept_logits(
                    global_outputs,
                    spatial_maps,
                    args,
                    concept_layer=concept_layer,
                )
                local_trust_weights = compute_local_trust_weights(global_concepts, args)
                local_mil_logits = None
                if bool(getattr(args, "savlg_use_local_mil", False)):
                    local_mil_logits = pool_local_mil_logits(spatial_maps, args)
                spatial_teacher_logits = None
                if consistency_enabled:
                    spatial_teacher_logits = (
                        local_mil_logits
                        if local_mil_logits is not None
                        else pool_spatial_teacher_logits(spatial_maps, args)
                    )
                (
                    loss_global_concept,
                    loss_mask,
                    loss_dice,
                    loss_local_mil,
                    loss_outside,
                    loss_coverage,
                    loss_absent_topk,
                ) = compute_spatial_losses(
                    final_logits,
                    spatial_maps,
                    global_concepts,
                    idx_pad,
                    mask_pad,
                    valid_pad,
                    global_bce_pos_weight=float(getattr(args, "global_bce_pos_weight", 1.0)),
                    patch_bce_pos_weight=float(getattr(args, "patch_bce_pos_weight", 1.0)),
                    loss_dice_w=float(getattr(args, "loss_dice_w", 0.0)),
                    local_mil_logits=local_mil_logits,
                    local_bce_pos_weight=float(getattr(args, "local_bce_pos_weight", 1.0)),
                    local_trust_weights=local_trust_weights,
                    local_loss_mode=str(getattr(args, "savlg_local_loss_mode", "bce")),
                    outside_penalty_w=float(getattr(args, "savlg_outside_penalty_w", 0.0)),
                    coverage_w=float(getattr(args, "savlg_coverage_w", 0.0)),
                    absent_topk_w=float(getattr(args, "savlg_absent_topk_w", 0.0)),
                    absent_topk_fraction=float(getattr(args, "savlg_absent_topk_fraction", 0.1)),
                )
                loss_refine = spatial_maps.sum() * 0.0
                if refine_weight > 0.0 and epoch >= refine_warmup_epochs:
                    loss_refine = compute_refinement_loss(
                        spatial_maps,
                        idx_pad,
                        mask_pad,
                        valid_pad,
                        patch_bce_pos_weight=float(getattr(args, "patch_bce_pos_weight", 1.0)),
                        local_trust_weights=local_trust_weights,
                    )
                loss_consistency = final_logits.sum() * 0.0
                if consistency_enabled and epoch >= consistency_warmup_epochs:
                    loss_consistency = compute_global_spatial_consistency_loss(
                        final_logits,
                        spatial_teacher_logits,
                        global_concepts,
                        local_trust_weights=local_trust_weights,
                    )
                loss_distill = final_logits.sum() * 0.0
                if teacher is not None and distill_weight > 0.0:
                    if images_for_teacher is None:
                        raise RuntimeError(
                            "SAVLG teacher distillation is not supported with cached feature batches."
                        )
                    teacher_feats = teacher["backbone"](images_for_teacher)
                    teacher_logits = teacher["concept_layer"](teacher_feats).index_select(
                        1, teacher["indices"]
                    )
                    teacher_probs = torch.sigmoid(teacher_logits)
                    loss_distill = F.binary_cross_entropy_with_logits(
                        final_logits, teacher_probs, reduction="mean"
                    )
                val_loss = (
                    global_concept_loss_weight * loss_global_concept
                    + float(getattr(args, "loss_mask_w", 1.0)) * loss_mask
                    + float(getattr(args, "loss_dice_w", 0.0)) * loss_dice
                    + float(getattr(args, "loss_local_mil_w", 0.0)) * loss_local_mil
                    + float(getattr(args, "savlg_outside_penalty_w", 0.0)) * loss_outside
                    + float(getattr(args, "savlg_coverage_w", 0.0)) * loss_coverage
                    + float(getattr(args, "savlg_absent_topk_w", 0.0)) * loss_absent_topk
                    + refine_weight * loss_refine
                    + consistency_weight * loss_consistency
                    + distill_weight * loss_distill
                )
                val_running += float(val_loss.item()) * global_concepts.size(0)
            val_loss = val_running / max(len(val_loader.dataset), 1)
        if scheduler is not None:
            scheduler.step()
        improved = val_loss < (best_loss - min_delta)
        if improved:
            best_loss = val_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in concept_layer.state_dict().items()
            }
            epochs_without_improvement = 0
        elif (epoch + 1) >= min_epochs:
            epochs_without_improvement += 1
        alpha_str = ""
        if hasattr(concept_layer, "log_spatial_scale"):
            alpha_str = f" alpha={concept_layer.log_spatial_scale.exp().item():.4f}"
        logger.info(
            "[SAVLG CBL] epoch={} train_loss={:.6f} val_loss={:.6f} best_val={:.6f}{}",
            epoch,
            train_loss,
            val_loss,
            best_loss,
            alpha_str,
        )
        if (
            early_stop_patience > 0
            and (epoch + 1) >= min_epochs
            and epochs_without_improvement >= early_stop_patience
        ):
            logger.info(
                "[SAVLG CBL] early stop at epoch={} after {} epochs without >= {:.6f} val improvement",
                epoch,
                epochs_without_improvement,
                min_delta,
            )
            break
        concept_layer.train()

    if best_state is not None:
        concept_layer.load_state_dict(best_state, strict=True)
    return concept_layer


def extract_global_concepts(
    args,
    backbone: SpatialBackbone,
    concept_layer: nn.Module,
    loader: DataLoader,
) -> Tuple[torch.Tensor, torch.Tensor]:
    backbone.eval()
    concept_layer.eval()
    concept_features = []
    labels = []
    with torch.no_grad():
        for images, target in tqdm(loader, desc="SAVLG concept extraction"):
            if _savlg_batch_already_features(images):
                feats = _move_savlg_feats_to_device(images, args.device)
            else:
                images = images.to(args.device)
                feats = forward_savlg_backbone(backbone, images, args)
            global_outputs, spatial_maps = forward_savlg_concept_layer(concept_layer, feats)
            _, _, final_logits = compute_savlg_concept_logits(
                global_outputs,
                spatial_maps,
                args,
                concept_layer=concept_layer,
            )
            concept_features.append(final_logits.cpu())
            labels.append(target)
    return torch.cat(concept_features, dim=0), torch.cat(labels, dim=0)


def evaluate_savlg_accuracy(
    args,
    backbone: SpatialBackbone,
    concept_layer: nn.Module,
    mean: torch.Tensor,
    std: torch.Tensor,
    final_layer: nn.Module,
    dataset: Dataset,
) -> float:
    loader = DataLoader(
        dataset,
        batch_size=args.cbl_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    correct = 0
    total = 0
    with torch.no_grad():
        for images, target in tqdm(loader, desc="SAVLG eval", leave=False):
            if _savlg_batch_already_features(images):
                feats = _move_savlg_feats_to_device(images, args.device)
            else:
                images = images.to(args.device)
                feats = forward_savlg_backbone(backbone, images, args)
            global_outputs, spatial_maps = forward_savlg_concept_layer(concept_layer, feats)
            _, _, final_logits = compute_savlg_concept_logits(
                global_outputs,
                spatial_maps,
                args,
                concept_layer=concept_layer,
            )
            final_logits = (final_logits - mean.to(args.device)) / std.to(args.device)
            pred = final_layer(final_logits).argmax(dim=-1).cpu()
            correct += int((pred == target).sum().item())
            total += int(target.numel())
    return correct / max(total, 1)


def train_savlg_cbm(args):
    save_dir = build_run_dir(args.save_dir, args.dataset, args.model_name)
    logger.add(
        os.path.join(save_dir, "train.log"),
        format="{time} {level} {message}",
        level="DEBUG",
    )
    logger.info("Saving SAVLG-CBM model to {}", save_dir)
    save_args(args, save_dir)

    classes = data_utils.get_classes(args.dataset)
    raw_concepts = data_utils.get_concepts(args.concept_set, args.filter_set)
    train_raw, val_raw, train_dataset, val_dataset, test_dataset, backbone = create_savlg_splits(args)
    supervision_source = _savlg_supervision_source(args)
    if supervision_source == "gdino":
        train_ann_dir = _annotation_split_dir(args.annotation_dir, args.dataset, "train")
        # When we use a train/val split from the *train* images, the spatial supervision
        # must come from the train annotations for both splits (ConceptDataset does the same).
        if use_original_label_free_protocol(args):
            val_ann_dir = _annotation_split_dir(args.annotation_dir, args.dataset, "val")
        else:
            val_ann_dir = train_ann_dir
    else:
        train_ann_dir = args.annotation_dir
        val_ann_dir = args.annotation_dir

    filter_mode = _savlg_concept_filter_mode(args)
    if filter_mode == "vlg_global":
        logger.info("Filtering SAVLG concepts with VLG concept-dataset path")
        filtered_concepts, _, _ = get_filtered_concepts_and_counts(
            args.dataset,
            raw_concepts,
            preprocess=backbone.preprocess,
            val_split=args.val_split,
            batch_size=args.cbl_batch_size,
            num_workers=args.num_workers,
            confidence_threshold=args.cbl_confidence_threshold,
            label_dir=args.annotation_dir,
            use_allones=args.allones_concept,
            seed=args.seed,
        )
        filtered_concept_set = set(filtered_concepts)
        keep_idx = [idx for idx, concept in enumerate(raw_concepts) if concept in filtered_concept_set]
        if not keep_idx:
            raise RuntimeError("VLG-style SAVLG concept filtering removed all concepts.")
    elif filter_mode == "spatial_threshold":
        # The default behavior scans every annotation JSON to determine which
        # concepts survive thresholding, then precomputes dense supervision
        # tensors. For ImageNet this is extremely expensive.
        #
        # When streaming supervision, keep all provided concepts and let the
        # per-sample dataset build targets on-demand.
        keep_idx = None if not bool(getattr(args, "savlg_stream_supervision", False)) else list(range(len(raw_concepts)))
    else:
        raise ValueError(
            f"Unsupported SAVLG concept filter mode: {filter_mode}. Expected one of ['spatial_threshold', 'vlg_global']."
        )

    stream_supervision = bool(getattr(args, "savlg_stream_supervision", False))
    if stream_supervision and _savlg_supervision_source(args) != "gdino":
        raise ValueError("--savlg_stream_supervision currently supports only --savlg_supervision_source=gdino.")

    if stream_supervision:
        if keep_idx is None:
            keep_idx = list(range(len(raw_concepts)))
        concepts = [raw_concepts[i] for i in keep_idx]
        # Stream per-image targets/masks from the annotation JSONs during training.
        train_supervision_ds = OnTheFlySpatialSupervisionDataset(
            train_raw.base_dataset,
            train_raw.indices,
            backbone.preprocess,
            train_ann_dir,
            concepts,
            args,
        )
        val_supervision_ds = OnTheFlySpatialSupervisionDataset(
            val_raw.base_dataset,
            val_raw.indices,
            backbone.preprocess,
            val_ann_dir,
            concepts,
            args,
        )
        train_global_concepts = None
        train_mask_entries = None
        val_global_concepts = None
        val_mask_entries = None
    else:
        train_global_concepts, train_mask_entries, keep_idx = load_spatial_supervision(
            train_raw, train_ann_dir, raw_concepts, args, "train", keep_idx=keep_idx
        )
        concepts = [raw_concepts[i] for i in keep_idx]
        val_global_concepts, val_mask_entries, _ = load_spatial_supervision(
            val_raw, val_ann_dir, raw_concepts, args, "val", keep_idx=keep_idx
        )

        train_supervision_ds = SpatialSupervisionDataset(
            train_dataset,
            train_global_concepts,
            train_mask_entries,
            args.mask_h,
            args.mask_w,
        )
        val_supervision_ds = SpatialSupervisionDataset(
            val_dataset,
            val_global_concepts,
            val_mask_entries,
            args.mask_h,
            args.mask_w,
        )
    if (not stream_supervision) and _savlg_feature_cache_enabled(args):
        logger.info(
            "Using in-memory SAVLG backbone features for deterministic training because crop_to_concept_prob == 0."
        )
        train_cached = build_savlg_feature_cache_in_memory(
            args, backbone, train_dataset, "train"
        )
        val_cached = build_savlg_feature_cache_in_memory(
            args, backbone, val_dataset, "val"
        )
        train_supervision_ds = CachedSpatialSupervisionDataset(
            train_cached["feats"],
            train_cached["labels"],
            train_global_concepts,
            train_mask_entries,
            args.mask_h,
            args.mask_w,
        )
        val_supervision_ds = CachedSpatialSupervisionDataset(
            val_cached["feats"],
            val_cached["labels"],
            val_global_concepts,
            val_mask_entries,
            args.mask_h,
            args.mask_w,
        )
    supervision_loader_kwargs = {
        "batch_size": args.cbl_batch_size,
        "num_workers": args.num_workers,
        "collate_fn": collate_spatial_batch,
        "pin_memory": True,
    }
    if int(args.num_workers) > 0:
        supervision_loader_kwargs["persistent_workers"] = True
    train_supervision_loader = DataLoader(
        train_supervision_ds,
        shuffle=True,
        **supervision_loader_kwargs,
    )
    val_supervision_loader = DataLoader(
        val_supervision_ds,
        shuffle=False,
        **supervision_loader_kwargs,
    )

    teacher = _load_savlg_teacher(args, concepts)

    concept_layer = build_savlg_concept_layer(args, backbone, len(concepts))
    maybe_initialize_savlg_from_vlg(args, concept_layer, concepts)
    concept_layer = train_concept_head(
        args,
        backbone,
        concept_layer,
        train_supervision_loader,
        val_supervision_loader,
        teacher=teacher,
    )

    if getattr(args, "cbl_only", False):
        with open(os.path.join(save_dir, "concepts.txt"), "w") as f:
            f.write("\n".join(concepts))
        torch.save(concept_layer.state_dict(), os.path.join(save_dir, "concept_layer.pt"))
        logger.info("cbl_only=True — saved concept_layer.pt, skipping sparse final layer")
    else:
        if _savlg_feature_cache_enabled(args):
            cached_loader_kwargs = {
                "batch_size": args.cbl_batch_size,
                "shuffle": False,
                "num_workers": args.num_workers,
                "pin_memory": True,
            }
            if int(args.num_workers) > 0:
                cached_loader_kwargs["persistent_workers"] = True
            train_loader = DataLoader(
                CachedFeatureLabelDataset(train_cached["feats"], train_cached["labels"]),
                **cached_loader_kwargs,
            )
            val_loader = DataLoader(
                CachedFeatureLabelDataset(val_cached["feats"], val_cached["labels"]),
                **cached_loader_kwargs,
            )
        else:
            train_loader = DataLoader(
                train_dataset, batch_size=args.cbl_batch_size, shuffle=False, num_workers=args.num_workers
            )
            val_loader = DataLoader(
                val_dataset, batch_size=args.cbl_batch_size, shuffle=False, num_workers=args.num_workers
            )
        train_concepts, train_labels = extract_global_concepts(args, backbone, concept_layer, train_loader)
        val_concepts, val_labels = extract_global_concepts(args, backbone, concept_layer, val_loader)

        train_mean = train_concepts.mean(dim=0, keepdim=True)
        train_std = torch.clamp(train_concepts.std(dim=0, keepdim=True), min=1e-6)
        train_concepts = (train_concepts - train_mean) / train_std
        val_concepts = (val_concepts - train_mean) / train_std

        train_final_loader = DataLoader(
            IndexedTensorDataset(train_concepts, train_labels),
            batch_size=args.saga_batch_size,
            shuffle=True,
        )
        val_final_loader = DataLoader(
            TensorDataset(val_concepts, val_labels),
            batch_size=args.saga_batch_size,
            shuffle=False,
        )
        final_layer = nn.Linear(len(concepts), len(classes)).to(args.device)
        final_layer.weight.data.zero_()
        final_layer.bias.data.zero_()
        if args.dense:
            output_proj = train_dense_final(
                final_layer,
                train_final_loader,
                val_final_loader,
                args.saga_n_iters,
                args.dense_lr,
                device=args.device,
            )
        else:
            output_proj = train_sparse_final(
                final_layer,
                train_final_loader,
                val_final_loader,
                args.saga_n_iters,
                args.saga_lam,
                step_size=args.saga_step_size,
                device=args.device,
            )

        W_g = output_proj["path"][0]["weight"]
        b_g = output_proj["path"][0]["bias"]
        final_layer.load_state_dict({"weight": W_g, "bias": b_g})

        if getattr(args, "skip_train_val_eval", False):
            train_accuracy = None
            val_accuracy = None
        else:
            train_accuracy = evaluate_savlg_accuracy(
                args, backbone, concept_layer, train_mean, train_std, final_layer, train_dataset
            )
            val_accuracy = evaluate_savlg_accuracy(
                args, backbone, concept_layer, train_mean, train_std, final_layer, val_dataset
            )
        if getattr(args, "skip_test_eval", False):
            test_accuracy = None
        else:
            test_accuracy = evaluate_savlg_accuracy(
                args, backbone, concept_layer, train_mean, train_std, final_layer, test_dataset
            )

        with open(os.path.join(save_dir, "concepts.txt"), "w") as f:
            f.write("\n".join(concepts))
        torch.save(concept_layer.state_dict(), os.path.join(save_dir, "concept_layer.pt"))
        torch.save(W_g, os.path.join(save_dir, "W_g.pt"))
        torch.save(b_g, os.path.join(save_dir, "b_g.pt"))
        torch.save(train_mean, os.path.join(save_dir, "proj_mean.pt"))
        torch.save(train_std, os.path.join(save_dir, "proj_std.pt"))

        test_metrics = {"accuracy": test_accuracy}
        metrics_to_write = [("test_metrics.json", test_metrics)]
        if not getattr(args, "skip_train_val_eval", False):
            metrics_to_write = [
                ("train_metrics.json", {"accuracy": train_accuracy}),
                ("val_metrics.json", {"accuracy": val_accuracy}),
                ("test_metrics.json", test_metrics),
            ]
        for filename, payload in metrics_to_write:
            with open(os.path.join(save_dir, filename), "w") as f:
                json.dump(payload, f, indent=2)

        path0 = output_proj["path"][0]
        metrics_payload = {
            key: float(path0[key]) for key in ("lam", "lr", "alpha", "time")
        }
        metrics_payload["metrics"] = path0["metrics"]
        nnz = int((W_g.abs() > 1e-5).sum().item())
        total = int(W_g.numel())
        metrics_payload["sparsity"] = {
            "Non-zero weights": nnz,
            "Total weights": total,
            "Percentage non-zero": nnz / max(total, 1),
        }
        with open(os.path.join(save_dir, "metrics.txt"), "w") as f:
            json.dump(metrics_payload, f, indent=2)

    method_log = {
        "cbm_variant": "savlg_cbm",
        "cbl_only": bool(getattr(args, "cbl_only", False)),
        "annotation_dir": args.annotation_dir,
        "annotation_threshold": float(getattr(args, "cbl_confidence_threshold", 0.15)),
        "concept_filter_mode": _savlg_concept_filter_mode(args),
        "mask_h": int(args.mask_h),
        "mask_w": int(args.mask_w),
        "concept_bottleneck_layer": {
            "type": args.cbl_type,
            "branch_arch": str(getattr(args, "savlg_branch_arch", "shared")),
            "global_head_mode": str(getattr(args, "savlg_global_head_mode", "spatial_pool")),
            "global_hidden_layers": int(getattr(args, "savlg_global_hidden_layers", 0)),
            "spatial_branch_mode": str(getattr(args, "savlg_spatial_branch_mode", "shared_stage")),
            "hidden_layers": args.cbl_hidden_layers if args.cbl_type == "mlp" else 0,
            "use_batchnorm": bool(args.cbl_use_batchnorm) if args.cbl_type == "mlp" else False,
        },
        "spatial_losses": {
            "loss_global_concept_w": _savlg_global_concept_loss_weight(args),
            "loss_mask_w": float(getattr(args, "loss_mask_w", 1.0)),
            "loss_dice_w": float(getattr(args, "loss_dice_w", 0.0)),
            "loss_local_mil_w": float(getattr(args, "loss_local_mil_w", 0.0)),
            "outside_penalty_w": float(getattr(args, "savlg_outside_penalty_w", 0.0)),
            "coverage_w": float(getattr(args, "savlg_coverage_w", 0.0)),
            "absent_topk_w": float(getattr(args, "savlg_absent_topk_w", 0.0)),
            "absent_topk_fraction": float(getattr(args, "savlg_absent_topk_fraction", 0.1)),
            "global_bce_pos_weight": float(getattr(args, "global_bce_pos_weight", 1.0)),
            "patch_bce_pos_weight": float(getattr(args, "patch_bce_pos_weight", 1.0)),
            "local_bce_pos_weight": float(getattr(args, "local_bce_pos_weight", 1.0)),
            "global_target_mode": _savlg_global_target_mode(args),
            "target_mode": str(getattr(args, "savlg_target_mode", "hard_iou")),
            "local_loss_mode": str(getattr(args, "savlg_local_loss_mode", "bce")),
            "patch_iou_thresh": float(getattr(args, "patch_iou_thresh", 0.5)),
        },
        "pooling": {
            "mode": str(getattr(args, "savlg_pooling", "avg")),
            "topk_fraction": float(getattr(args, "savlg_topk_fraction", 0.2)),
            "local_mil_enabled": bool(getattr(args, "savlg_use_local_mil", False)),
            "local_mil_pooling": str(getattr(args, "savlg_local_pooling", "lse")),
            "local_mil_temperature": float(getattr(args, "savlg_mil_temperature", 1.0)),
            "local_mil_topk_fraction": float(getattr(args, "savlg_mil_topk_fraction", 0.2)),
        },
        "residual_spatial_coupling": {
            "alpha": float(getattr(args, "savlg_residual_spatial_alpha", 0.0)),
            "pooling": str(getattr(args, "savlg_residual_spatial_pooling", "lse")),
            "enabled": savlg_residual_coupling_enabled(args),
        },
        "selective_local_weighting": {
            "mode": str(getattr(args, "savlg_local_weight_mode", "uniform")),
            "floor": float(getattr(args, "savlg_local_weight_floor", 0.25)),
            "power": float(getattr(args, "savlg_local_weight_power", 1.0)),
            "enabled": str(getattr(args, "savlg_local_weight_mode", "uniform")).lower() != "uniform",
        },
        "teacher_distillation": {
            "teacher_load_path": str(getattr(args, "savlg_teacher_load_path", "") or ""),
            "distill_w": float(getattr(args, "savlg_distill_w", 0.0)),
            "enabled": teacher is not None and float(getattr(args, "savlg_distill_w", 0.0)) > 0.0,
        },
        "vlg_warm_start": {
            "init_path": str(getattr(args, "savlg_init_from_vlg_path", "") or ""),
            "init_spatial": bool(getattr(args, "savlg_init_spatial_from_vlg", False)),
            "freeze_global_head": bool(getattr(args, "savlg_freeze_global_head", False)),
            "enabled": bool(str(getattr(args, "savlg_init_from_vlg_path", "") or "").strip()),
        },
        "global_spatial_consistency": {
            "consistency_w": float(getattr(args, "savlg_global_spatial_consistency_w", 0.0)),
            "warmup_epochs": int(getattr(args, "savlg_global_spatial_consistency_warmup_epochs", 0)),
            "enabled": (
                float(getattr(args, "savlg_global_spatial_consistency_w", 0.0)) > 0.0
                and str(getattr(args, "savlg_branch_arch", "shared")).lower() == "dual"
            ),
            "positive_only": True,
            "teacher_source": (
                "local_mil_logits"
                if bool(getattr(args, "savlg_use_local_mil", False))
                else "pooled_spatial_maps"
            ),
        },
        "oicr_refinement": {
            "refine_w": float(getattr(args, "savlg_refine_w", 0.0)),
            "warmup_epochs": int(getattr(args, "savlg_refine_warmup_epochs", 0)),
            "box_anchored_top_patch": bool(float(getattr(args, "savlg_refine_w", 0.0)) > 0.0),
        },
        "supervision_cache_paths": {
            "train": _supervision_cache_path(args, "train", concepts),
            "val": _supervision_cache_path(args, "val", concepts),
        },
        "spatial_backbone": {
            "stage": str(getattr(args, "savlg_spatial_stage", "conv5")),
            "global_head_mode": str(getattr(args, "savlg_global_head_mode", "spatial_pool")),
            "global_hidden_layers": int(getattr(args, "savlg_global_hidden_layers", 0)),
            "spatial_branch_mode": str(getattr(args, "savlg_spatial_branch_mode", "shared_stage")),
            "multiscale_enabled": savlg_uses_multiscale_branch(args),
        },
    }
    with open(os.path.join(save_dir, "method_log.json"), "w") as f:
        json.dump(method_log, f, indent=2)

    if getattr(args, "cbl_only", False):
        write_artifacts(
            save_dir,
            {
                "model_name": args.model_name,
                "dataset": args.dataset,
                "backbone": args.backbone,
                "concept_layer_format": "concept_layer.pt",
                "supervision_cache_format": ["*_supervision.pt"],
                "cbl_only": True,
            },
        )
        logger.info("SAVLG-CBM cbl_only training complete — saved to {}", save_dir)
        return save_dir

    write_artifacts(
        save_dir,
        {
            "model_name": args.model_name,
            "dataset": args.dataset,
            "backbone": args.backbone,
            "concept_layer_format": "concept_layer.pt",
            "normalization_format": ["proj_mean.pt", "proj_std.pt"],
            "final_layer_format": ["W_g.pt", "b_g.pt"],
            "supervision_cache_format": ["*_supervision.pt"],
            "sparse_eval_style": "salf_compatible",
        },
    )
    def _fmt_acc(value: Optional[float]) -> str:
        if value is None:
            return "skipped"
        try:
            return f"{float(value):.4f}"
        except Exception:
            return str(value)

    if getattr(args, "skip_train_val_eval", False):
        logger.info("SAVLG-CBM test accuracy={}", _fmt_acc(test_accuracy))
    else:
        logger.info(
            "SAVLG-CBM train accuracy={} val accuracy={} test accuracy={}",
            _fmt_acc(train_accuracy),
            _fmt_acc(val_accuracy),
            _fmt_acc(test_accuracy),
        )
    return save_dir
