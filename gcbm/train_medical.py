from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from gcbm.features import standardize_from_train
from gcbm.losses import sgcbm_concept_losses, weighted_concept_bce
from gcbm.medical_annotations import (
    LazyMedicalTargetDataset,
    MedicalTargetDataset,
    build_medical_presence_targets,
    build_medical_targets,
    calibrate_presence,
    load_concepts,
    medical_target_collate,
    path_match_keys,
)
from gcbm.medical_data import (
    default_mimic_paths,
    get_medical_transforms,
    infer_chexpert_img_root,
    load_chexpert_dataset,
    load_mimic_cxr_dataset,
    medical_collate,
    medical_labels,
)
from gcbm.medical_metrics import compute_medical_metrics
from gcbm.sparse import threshold_weight_truncation
from gcbm.sg_model import DualBranchConceptLayer


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train medical VLG-CBM or SG-CBM on CheXpert/MIMIC-CXR.")
    parser.add_argument("--model_name", choices=["vlg_cbm", "sgcbm", "sg_cbm", "savlg_cbm"], default="sgcbm", help="Medical CBM variant.")
    parser.add_argument("--dataset", choices=["chexpert", "mimic"], required=True, help="Medical dataset to train on.")
    parser.add_argument("--data_dir", required=True, help="Dataset root containing CheXpert CSVs or MIMIC-CXR-JPG files/CSVs.")
    parser.add_argument("--concept_file", required=True, help="Text or JSON concept bank.")
    parser.add_argument("--train_annotation_dir", default="", help="Grounded concept annotations for train split.")
    parser.add_argument("--val_annotation_dir", default="", help="Grounded concept annotations for validation split.")
    parser.add_argument("--train_concept_cache", default="", help="VLG-CBM train concept-cache .npz from generated annotations.")
    parser.add_argument("--val_concept_cache", default="", help="VLG-CBM validation concept-cache .npz from generated annotations.")
    parser.add_argument("--train_presence_cache", default="", help="Optional lightweight train concept-presence target cache shared by VLG-CBM and SG-CBM.")
    parser.add_argument("--val_presence_cache", default="", help="Optional lightweight validation concept-presence target cache shared by VLG-CBM and SG-CBM.")
    parser.add_argument("--precomputed_target_dir", default="", help="Optional ImageNet-style medical target-cache root. Expects split caches like train_targets.pt / val_targets.pt and optional train_presence.pt / val_presence.pt.")
    parser.add_argument("--save_dir", default="artifacts/medical", help="Directory for run outputs.")
    parser.add_argument("--run_name", default="", help="Optional run directory name.")
    parser.add_argument("--train_csv", default="", help="CheXpert train CSV override.")
    parser.add_argument("--val_csv", default="", help="CheXpert validation CSV override.")
    parser.add_argument("--img_root", default="", help="Image root override.")
    parser.add_argument("--mimic_label_csv", default="", help="MIMIC CheXpert-label CSV override.")
    parser.add_argument("--mimic_split_csv", default="", help="MIMIC split CSV override.")
    parser.add_argument("--mimic_metadata_csv", default="", help="MIMIC metadata CSV override.")
    parser.add_argument("--label_subset", choices=["all", "competition", "pathology"], default="all", help="CheXpert-style label subset.")
    parser.add_argument("--uncertain_strategy", choices=["ones", "zeros", "ignore"], default="ones", help="How to map uncertain labels.")
    parser.add_argument("--frontal_only", action=argparse.BooleanOptionalAction, default=True, help="Use only frontal/AP/PA images.")
    parser.add_argument("--backbone", choices=["densenet121", "resnet50"], default="densenet121", help="Spatial backbone.")
    parser.add_argument("--backbone_ckpt", default="", help="Optional classifier checkpoint; classifier weights are ignored.")
    parser.add_argument("--concept_head_ckpt", default="", help="Optional trained concept-head checkpoint; use with --epochs 0 to train only the final layer.")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True, help="Use torchvision pretrained weights if no checkpoint is supplied.")
    parser.add_argument("--img_size", type=int, default=224, help="Input crop size.")
    parser.add_argument("--resize_size", type=int, default=256, help="Resize short edge before center crop.")
    parser.add_argument("--mask_h", type=int, default=14, help="Spatial target height.")
    parser.add_argument("--mask_w", type=int, default=14, help="Spatial target width.")
    parser.add_argument("--target_mode", choices=["soft_box", "hard_iou"], default="soft_box", help="Box-to-mask target mode.")
    parser.add_argument("--patch_iou_thresh", type=float, default=0.5, help="Patch IoU threshold for hard_iou targets.")
    parser.add_argument("--concept_threshold", type=float, default=0.70, help="Positive concept confidence threshold.")
    parser.add_argument("--neg_threshold", type=float, default=0.02, help="Soft target lower calibration threshold.")
    parser.add_argument("--presence_mode", choices=["binary", "soft"], default="binary", help="Global concept target mode; binary matches the upstream VLG-CBM cache.")
    parser.add_argument("--min_concept_freq", type=float, default=0.01, help="Drop concepts below this train-set frequency.")
    parser.add_argument("--max_concept_freq", type=float, default=0.99, help="Drop concepts above this train-set frequency.")
    parser.add_argument("--global_pos_weight", type=float, default=1.0, help="Positive weight for global concept BCE.")
    parser.add_argument("--loss_global_w", type=float, default=1.0, help="Global concept BCE loss weight.")
    parser.add_argument("--loss_mask_w", type=float, default=1.0, help="Spatial soft-align KL loss weight.")
    parser.add_argument("--residual_alpha", type=float, default=0.2, help="Spatial residual logit coupling.")
    parser.add_argument("--epochs", type=int, default=10, help="Concept-layer training epochs.")
    parser.add_argument("--early_stop_patience", type=int, default=0, help="Stop CBL training after this many non-improving validation epochs; 0 disables early stopping.")
    parser.add_argument("--early_stop_min_delta", type=float, default=0.0, help="Minimum validation-loss improvement required to reset early stopping.")
    parser.add_argument("--batch_size", type=int, default=64, help="Concept-layer batch size.")
    parser.add_argument("--extract_batch_size", type=int, default=0, help="Feature-extraction batch size; defaults to --batch_size.")
    parser.add_argument("--extract_chunk_size", type=int, default=10000, help="Rows per resumable concept-extraction chunk.")
    parser.add_argument("--final_batch_size", type=int, default=256, help="Final-layer batch size.")
    parser.add_argument("--lr", type=float, default=5e-4, help="Concept-layer learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Concept-layer weight decay.")
    parser.add_argument("--final_lr", type=float, default=1e-3, help="Dense final-layer learning rate.")
    parser.add_argument("--final_epochs", type=int, default=100, help="Dense final-layer epochs.")
    parser.add_argument("--use_saga", action="store_true", help="Train sparse final layer with GLM-SAGA instead of dense Adam.")
    parser.add_argument("--saga_lam", type=float, default=1e-3, help="GLM-SAGA regularization.")
    parser.add_argument("--saga_iters", type=int, default=1000, help="GLM-SAGA epochs.")
    parser.add_argument("--saga_max_lr", type=float, default=0.1, help="GLM-SAGA max learning rate.")
    parser.add_argument("--nec_values", default="", help="Comma-separated NEC values for thresholded sparse-head evaluation.")
    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader workers.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for precision/recall/F1.")
    parser.add_argument("--device", default="cuda", help="Torch device.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--max_train_images", type=int, default=0, help="Optional train subset for smoke tests.")
    parser.add_argument("--max_val_images", type=int, default=0, help="Optional validation subset for smoke tests.")
    parser.add_argument("--train_target_cache", default="", help="Optional cache for train SG-CBM targets.")
    parser.add_argument("--val_target_cache", default="", help="Optional cache for val SG-CBM targets.")
    parser.add_argument("--allow_annotation_index_fallback", action="store_true", help="Allow row-index fallback when annotation paths do not match.")
    return parser.parse_args(argv)


def normalize_model_name(name: str) -> str:
    normalized = str(name).lower().replace("-", "_")
    if normalized in {"sgcbm", "sg_cbm", "savlg_cbm"}:
        return "sgcbm"
    if normalized == "vlg_cbm":
        return "vlg_cbm"
    raise ValueError(f"Unsupported medical model_name: {name}")


def parse_nec_values(raw: str) -> list[int]:
    if not raw:
        return []
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if any(value <= 0 for value in values):
        raise ValueError("--nec_values must contain positive integers")
    return values


class MedicalBackbone(nn.Module):
    def __init__(self, name: str, *, pretrained: bool, checkpoint: str = "") -> None:
        super().__init__()
        from torchvision import models

        self.name = name
        if name == "densenet121":
            weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.densenet121(weights=weights)
            self.encoder = model.features
            self.feature_dim = 1024
        elif name == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.resnet50(weights=weights)
            self.encoder = nn.Sequential(*list(model.children())[:-2])
            self.feature_dim = 2048
        else:
            raise ValueError(f"Unsupported medical backbone: {name}")
        if checkpoint:
            self._load_checkpoint(checkpoint)

    def _load_checkpoint(self, path: str) -> None:
        import inspect
        import sys

        # Some older medical checkpoints were saved in environments whose
        # pickles reference NumPy 2.x's numpy._core module path.
        if "numpy._core" not in sys.modules:
            sys.modules["numpy._core"] = np.core
        if "numpy._core.multiarray" not in sys.modules:
            sys.modules["numpy._core.multiarray"] = np.core.multiarray

        load_kwargs = {"map_location": "cpu"}
        if "weights_only" in inspect.signature(torch.load).parameters:
            load_kwargs["weights_only"] = False
        payload = torch.load(path, **load_kwargs)
        state = payload.get("model_state_dict", payload.get("state_dict", payload.get("model", payload))) if isinstance(payload, dict) else payload
        state = self._encoder_state_from_checkpoint(state)
        missing, unexpected = self.load_state_dict(state, strict=False)
        print(f"[medical] loaded backbone checkpoint: missing={len(missing)} unexpected={len(unexpected)}", flush=True)

    def _encoder_state_from_checkpoint(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        cleaned: Dict[str, torch.Tensor] = {}
        for raw_key, value in state.items():
            key = raw_key.removeprefix("module.")
            if any(part in key for part in ("classifier", "classification_head", "fc.")):
                continue
            for prefix in ("backbone.encoder.", "model.encoder.", "encoder."):
                if key.startswith(prefix):
                    cleaned["encoder." + key[len(prefix) :]] = value
                    break
            else:
                if self.name == "densenet121" and key.startswith("backbone.features."):
                    cleaned["encoder." + key[len("backbone.features.") :]] = value
                elif self.name == "densenet121" and key.startswith("features."):
                    cleaned["encoder." + key[len("features.") :]] = value
                elif self.name == "resnet50":
                    mapped_key = self._resnet_encoder_key(key)
                    if mapped_key is not None:
                        cleaned[mapped_key] = value
                elif key.startswith("encoder."):
                    cleaned[key] = value
        return cleaned

    @staticmethod
    def _resnet_encoder_key(key: str) -> str | None:
        mapping = {"conv1.": "encoder.0.", "bn1.": "encoder.1.", "layer1.": "encoder.4.", "layer2.": "encoder.5.", "layer3.": "encoder.6.", "layer4.": "encoder.7."}
        for prefix, replacement in mapping.items():
            if key.startswith(prefix):
                return replacement + key[len(prefix) :]
        return key if key.startswith("encoder.") else None

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(images)
        if self.name == "densenet121":
            feats = F.relu(feats, inplace=False)
        return feats


class GlobalConceptLayer(nn.Module):
    """VLG-CBM concept layer: global average pooled backbone features -> concepts."""

    def __init__(self, in_features: int, n_concepts: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(in_features, n_concepts)

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        pooled = self.pool(features).flatten(1)
        return {"final_logits": self.linear(pooled)}


def build_datasets(args: argparse.Namespace, labels: list[str]):
    train_tf = get_medical_transforms(args.img_size, train=False)
    val_tf = get_medical_transforms(args.img_size, train=False)
    data_dir = Path(args.data_dir)
    if args.dataset == "chexpert":
        img_root = Path(args.img_root) if args.img_root else infer_chexpert_img_root(data_dir)
        train_csv = Path(args.train_csv) if args.train_csv else data_dir / "train.csv"
        val_csv = Path(args.val_csv) if args.val_csv else data_dir / "valid.csv"
        train_ds = load_chexpert_dataset(train_csv, img_root=img_root, labels=labels, transform=train_tf, uncertain_strategy=args.uncertain_strategy, frontal_only=args.frontal_only)
        val_ds = load_chexpert_dataset(val_csv, img_root=img_root, labels=labels, transform=val_tf, uncertain_strategy=args.uncertain_strategy, frontal_only=args.frontal_only)
        return train_ds, val_ds

    paths = default_mimic_paths(data_dir)
    label_csv = Path(args.mimic_label_csv) if args.mimic_label_csv else paths["label_csv"]
    split_csv = Path(args.mimic_split_csv) if args.mimic_split_csv else paths["split_csv"]
    metadata_csv = Path(args.mimic_metadata_csv) if args.mimic_metadata_csv else paths["metadata_csv"]
    train_ds = load_mimic_cxr_dataset(label_csv, img_root=Path(args.img_root) if args.img_root else data_dir, split="train", split_csv=split_csv, metadata_csv=metadata_csv, labels=labels, transform=train_tf, uncertain_strategy=args.uncertain_strategy, frontal_only=args.frontal_only)
    val_ds = load_mimic_cxr_dataset(label_csv, img_root=Path(args.img_root) if args.img_root else data_dir, split="validate", split_csv=split_csv, metadata_csv=metadata_csv, labels=labels, transform=val_tf, uncertain_strategy=args.uncertain_strategy, frontal_only=args.frontal_only)
    return train_ds, val_ds


class MedicalSubset(torch.utils.data.Subset):
    def get_image_path(self, index: int) -> str:
        return self.dataset.get_image_path(self.indices[index])  # type: ignore[attr-defined]

    def get_image_size(self, index: int):
        return self.dataset.get_image_size(self.indices[index])  # type: ignore[attr-defined]


def maybe_subset(dataset, max_images: int):
    if max_images <= 0 or max_images >= len(dataset):
        return dataset
    return MedicalSubset(dataset, list(range(int(max_images))))


def get_or_build_targets(dataset, args: argparse.Namespace, concepts: list[str], annotation_dir: str, cache_path: str) -> Dict[str, Any]:
    if cache_path and Path(cache_path).exists():
        return _torch_load(cache_path)
    payload = build_medical_targets(
        dataset,
        annotation_dir=annotation_dir,
        concepts=concepts,
        mask_h=args.mask_h,
        mask_w=args.mask_w,
        concept_threshold=args.concept_threshold,
        neg_threshold=args.neg_threshold,
        presence_mode=args.presence_mode,
        target_mode=args.target_mode,
        input_size=args.img_size,
        resize_size=args.resize_size,
        patch_iou_thresh=args.patch_iou_thresh,
        allow_index_fallback=args.allow_annotation_index_fallback,
        num_workers=args.num_workers,
    )
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, cache_path)
    return payload


def _torch_load(path: str | Path) -> Any:
    import inspect

    load_kwargs = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kwargs["weights_only"] = False
    return torch.load(path, **load_kwargs)


def _concept_hash(concepts: Sequence[str]) -> str:
    digest = hashlib.md5()
    for concept in concepts:
        digest.update(str(concept).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()[:12]


def default_presence_cache_path(annotation_dir: str, *, n_rows: int, concepts: Sequence[str], threshold: float) -> str:
    if not annotation_dir:
        return ""
    return str(
        Path(annotation_dir)
        / f"presence_cache_{int(n_rows)}_{len(concepts)}_{float(threshold):.2f}_{_concept_hash(concepts)}.pt"
    )


def _existing_cache_path(*candidates: Path) -> str:
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return ""


def resolve_precomputed_cache_paths(root: str) -> Dict[str, str]:
    if not root:
        return {}
    base = Path(root)
    return {
        "train_target_store": str(base / "train") if (base / "train" / "metadata.json").exists() else "",
        "val_target_store": str(base / "val") if (base / "val" / "metadata.json").exists() else "",
        "train_target_cache": _existing_cache_path(base / "train_targets.pt", base / "train.pt"),
        "val_target_cache": _existing_cache_path(base / "val_targets.pt", base / "valid_targets.pt", base / "val.pt", base / "valid.pt"),
        "train_presence_cache": _existing_cache_path(base / "train_presence.pt"),
        "val_presence_cache": _existing_cache_path(base / "val_presence.pt", base / "valid_presence.pt"),
    }


def load_presence_cache_targets(
    dataset,
    *,
    cache_path: str,
    concepts: list[str],
    concept_threshold: float,
    neg_threshold: float,
    presence_mode: str,
) -> Dict[str, Any]:
    payload = _torch_load(cache_path)
    if int(payload.get("num_images", -1)) != len(dataset):
        raise ValueError(f"Presence cache image count mismatch for {cache_path}: {payload.get('num_images')} vs {len(dataset)}")
    cache_concepts = [str(item) for item in payload.get("concepts", concepts)]
    concept_to_col = {concept: idx for idx, concept in enumerate(cache_concepts)}
    missing = [concept for concept in concepts if concept not in concept_to_col]
    if missing:
        raise ValueError(f"Presence cache is missing {len(missing)} requested concepts; first missing={missing[:5]}")
    columns = torch.tensor([concept_to_col[concept] for concept in concepts], dtype=torch.long)
    loaded = dict(payload)
    if "presence_scores" in loaded:
        presence_scores = loaded["presence_scores"].float()[:, columns]
        if presence_mode == "binary":
            global_targets = (presence_scores >= float(concept_threshold)).float()
        elif presence_mode == "soft":
            global_targets = torch.from_numpy(
                calibrate_presence(
                    presence_scores.numpy(),
                    pos_thresh=float(concept_threshold),
                    neg_thresh=float(neg_threshold),
                    mode="linear",
                )
            )
        else:
            raise ValueError(f"Unsupported presence_mode: {presence_mode}")
        loaded["presence_scores"] = presence_scores
        loaded["global_targets"] = global_targets
    else:
        cached_threshold = payload.get("concept_threshold")
        if cached_threshold is not None and abs(float(cached_threshold) - float(concept_threshold)) > 1e-8:
            print(
                f"[medical targets] warning: cache {cache_path} has no presence_scores; "
                f"reusing cached global_targets from threshold={cached_threshold} for requested threshold={concept_threshold}",
                flush=True,
            )
        loaded["global_targets"] = loaded["global_targets"].float()[:, columns]
    loaded["num_concepts"] = len(concepts)
    loaded["concept_threshold"] = float(concept_threshold)
    loaded["neg_threshold"] = float(neg_threshold)
    loaded["presence_mode"] = str(presence_mode)
    loaded["cache_path"] = str(cache_path)
    return loaded


def get_or_build_presence_targets(
    dataset,
    args: argparse.Namespace,
    concepts: list[str],
    annotation_dir: str,
    cache_path: str,
) -> Dict[str, Any]:
    resolved_cache = cache_path or default_presence_cache_path(
        annotation_dir,
        n_rows=len(dataset),
        concepts=concepts,
        threshold=args.concept_threshold,
    )
    if resolved_cache and Path(resolved_cache).exists():
        print(f"[medical targets] loading presence cache {resolved_cache}", flush=True)
        return load_presence_cache_targets(
            dataset,
            cache_path=resolved_cache,
            concepts=concepts,
            concept_threshold=args.concept_threshold,
            neg_threshold=args.neg_threshold,
            presence_mode=args.presence_mode,
        )
    payload = build_medical_presence_targets(
        dataset,
        annotation_dir=annotation_dir,
        concepts=concepts,
        concept_threshold=args.concept_threshold,
        allow_index_fallback=args.allow_annotation_index_fallback,
        num_workers=args.num_workers,
    )
    payload = {
        **payload,
        "concepts": list(concepts),
        "concept_threshold": float(args.concept_threshold),
        "annotation_dir": str(annotation_dir),
    }
    if resolved_cache:
        Path(resolved_cache).parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, resolved_cache)
        print(f"[medical targets] saved presence cache {resolved_cache}", flush=True)
    return payload


def _unique_cache_path_index(paths: Sequence[object]) -> Dict[str, int]:
    buckets: Dict[str, list[int]] = {}
    for index, raw_path in enumerate(paths):
        for key in path_match_keys(str(raw_path)):
            buckets.setdefault(key, []).append(index)
    return {key: rows[0] for key, rows in buckets.items() if len(rows) == 1}


def find_concept_cache(annotation_dir: str, *, n_rows: int, n_concepts: int, threshold: float) -> str:
    if not annotation_dir:
        return ""
    directory = Path(annotation_dir)
    exact = sorted(directory.glob(f"concept_cache_{n_rows}_{n_concepts}_{float(threshold):.2f}_*.npz"))
    if exact:
        return str(exact[0])
    candidates = sorted(directory.glob(f"concept_cache_*_{n_concepts}_{float(threshold):.2f}_*.npz"))
    return str(candidates[0]) if candidates else ""


def load_concept_cache_targets(
    dataset,
    *,
    cache_path: str,
    concepts: list[str],
    allow_index_fallback: bool,
) -> Dict[str, Any]:
    if not cache_path:
        raise ValueError("VLG-CBM requires --train_concept_cache/--val_concept_cache or matching concept_cache_*.npz files in annotation dirs.")
    payload = np.load(cache_path, allow_pickle=True)
    matrix = np.asarray(payload["concept_matrix"], dtype=np.float32)
    if "concepts" in payload.files:
        cache_concepts = [str(item) for item in payload["concepts"].tolist()]
    else:
        if matrix.shape[1] != len(concepts):
            raise ValueError(
                f"Concept cache {cache_path} does not store concept names and has {matrix.shape[1]} columns, "
                f"but --concept_file has {len(concepts)} concepts."
            )
        cache_concepts = list(concepts)
    concept_to_col = {concept: idx for idx, concept in enumerate(cache_concepts)}
    missing = [concept for concept in concepts if concept not in concept_to_col]
    if missing:
        raise ValueError(f"Concept cache is missing {len(missing)} requested concepts; first missing={missing[:5]}")
    matrix = matrix[:, [concept_to_col[concept] for concept in concepts]]

    cache_paths = payload["image_paths"] if "image_paths" in payload.files else []
    aligned = np.zeros((len(dataset), len(concepts)), dtype=np.float32)
    matched = 0
    if len(cache_paths):
        cache_index = _unique_cache_path_index(cache_paths)
        for row in range(len(dataset)):
            image_path = getattr(dataset, "get_image_path")(row)
            for key in path_match_keys(image_path):
                cache_row = cache_index.get(key)
                if cache_row is not None:
                    aligned[row] = matrix[cache_row]
                    matched += 1
                    break
    if matched == 0 and allow_index_fallback and len(matrix) >= len(dataset):
        aligned[:] = matrix[: len(dataset)]
        matched = len(dataset)
    unmatched = len(dataset) - matched
    if matched == 0:
        raise ValueError(f"No concept-cache rows matched dataset paths for {cache_path}")
    return {
        "global_targets": torch.from_numpy(aligned),
        "matched_annotations": int(matched),
        "unmatched_annotations": int(unmatched),
        "num_concepts": len(concepts),
        "num_images": len(dataset),
        "cache_path": str(cache_path),
    }


def concept_frequency_filter_indices(
    concepts: list[str],
    train_targets: Dict[str, Any],
    *,
    min_freq: float,
    max_freq: float,
) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    train_matrix = train_targets["global_targets"].float()
    frequencies = (train_matrix > 0).float().mean(dim=0)
    keep = (frequencies >= float(min_freq)) & (frequencies <= float(max_freq))
    if int(keep.sum()) == 0:
        raise ValueError("Concept frequency filtering removed every concept")
    kept_indices = torch.nonzero(keep, as_tuple=False).flatten()
    filtered_concepts = [concepts[int(index)] for index in kept_indices]
    print(f"[medical] filtered concepts: {len(concepts)} -> {len(filtered_concepts)}", flush=True)
    return filtered_concepts, kept_indices, frequencies


def filter_target_payload(targets: Dict[str, Any], kept_indices: torch.Tensor, num_old_concepts: int) -> Dict[str, Any]:
    kept_indices = kept_indices.long().cpu()
    filtered = {**targets, "global_targets": targets["global_targets"][:, kept_indices], "num_concepts": int(len(kept_indices))}
    if "presence_scores" in targets:
        filtered["presence_scores"] = targets["presence_scores"][:, kept_indices]
    if "mask_indices" not in targets or "mask_targets" not in targets:
        return filtered

    old_to_new = torch.full((int(num_old_concepts),), -1, dtype=torch.long)
    old_to_new[kept_indices] = torch.arange(len(kept_indices), dtype=torch.long)
    indices = targets["mask_indices"]
    masks = targets["mask_targets"]
    if isinstance(indices, torch.Tensor):
        valid = targets.get("mask_valid")
        if valid is None:
            raise ValueError("Padded mask target payload requires mask_valid for concept filtering")
        remapped = torch.zeros_like(indices, dtype=torch.long)
        remapped_valid = valid.bool().clone()
        for row in range(indices.shape[0]):
            row_old = indices[row].long()
            row_new = torch.full_like(row_old, -1)
            valid_old = (row_old >= 0) & (row_old < int(num_old_concepts))
            row_new[valid_old] = old_to_new[row_old[valid_old]]
            keep = remapped_valid[row] & valid_old & (row_new >= 0)
            remapped[row][keep] = row_new[keep]
            remapped_valid[row] = keep
        filtered["mask_indices"] = remapped
        filtered["mask_valid"] = remapped_valid
        return filtered

    remapped_indices = []
    remapped_masks = []
    for row_indices, row_masks in zip(indices, masks):
        row_indices = row_indices.long().cpu()
        if row_indices.numel() == 0:
            remapped_indices.append(row_indices)
            remapped_masks.append(row_masks.float().cpu())
            continue
        row_new = torch.full_like(row_indices, -1)
        valid_old = (row_indices >= 0) & (row_indices < int(num_old_concepts))
        row_new[valid_old] = old_to_new[row_indices[valid_old]]
        keep = valid_old & (row_new >= 0)
        remapped_indices.append(row_new[keep].long().cpu())
        remapped_masks.append(row_masks[keep].float().cpu())
    filtered["mask_indices"] = remapped_indices
    filtered["mask_targets"] = remapped_masks
    return filtered


def apply_concept_frequency_filter(
    concepts: list[str],
    train_targets: Dict[str, Any],
    val_targets: Dict[str, Any],
    *,
    min_freq: float,
    max_freq: float,
) -> tuple[list[str], Dict[str, Any], Dict[str, Any], torch.Tensor, torch.Tensor]:
    original_count = len(concepts)
    filtered_concepts, kept_indices, frequencies = concept_frequency_filter_indices(
        concepts,
        train_targets,
        min_freq=min_freq,
        max_freq=max_freq,
    )
    train_targets = filter_target_payload(train_targets, kept_indices, original_count)
    val_targets = filter_target_payload(val_targets, kept_indices, original_count)
    return filtered_concepts, train_targets, val_targets, kept_indices, frequencies


def build_filter_reference_targets(train_ds, args: argparse.Namespace, concepts: list[str]) -> Dict[str, Any]:
    cache_path = args.train_concept_cache or find_concept_cache(
        args.train_annotation_dir,
        n_rows=len(train_ds),
        n_concepts=len(concepts),
        threshold=args.concept_threshold,
    )
    if cache_path:
        return load_concept_cache_targets(
            train_ds,
            cache_path=cache_path,
            concepts=concepts,
            allow_index_fallback=args.allow_annotation_index_fallback,
        )
    return get_or_build_presence_targets(
        train_ds,
        args,
        concepts,
        annotation_dir=args.train_annotation_dir,
        cache_path=args.train_presence_cache,
    )


class MedicalConceptDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, target_payload: Dict[str, Any]) -> None:
        self.base_dataset = base_dataset
        self.global_targets = target_payload["global_targets"].float()
        if len(base_dataset) != int(self.global_targets.shape[0]):
            raise ValueError("base_dataset and concept targets length mismatch")

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = self.base_dataset[index]
        return {**item, "global_targets": self.global_targets[index]}


def medical_concept_collate(batch) -> Dict[str, Any]:
    items = list(batch)
    return {
        "image": torch.stack([item["image"] for item in items]),
        "label": torch.stack([item["target"] for item in items]),
        "global_targets": torch.stack([item["global_targets"] for item in items]),
        "sample_id": [str(item["sample_id"]) for item in items],
        "image_path": [str(item["image_path"]) for item in items],
        "index": torch.tensor([int(item["index"]) for item in items], dtype=torch.long),
    }


def build_head(backbone: MedicalBackbone, n_concepts: int, args: argparse.Namespace) -> nn.Module:
    if normalize_model_name(args.model_name) == "vlg_cbm":
        return GlobalConceptLayer(backbone.feature_dim, n_concepts)
    global_layer = nn.Linear(backbone.feature_dim, n_concepts)
    spatial_layer = nn.Conv2d(backbone.feature_dim, n_concepts, kernel_size=1, bias=False)
    return DualBranchConceptLayer(
        global_layer,
        spatial_layer,
        residual_alpha=args.residual_alpha,
        residual_spatial_pooling="lse",
    )


def compute_losses(outputs: Dict[str, torch.Tensor], batch: Dict[str, Any], args: argparse.Namespace, device: str) -> Dict[str, torch.Tensor]:
    if normalize_model_name(args.model_name) == "vlg_cbm" or args.loss_mask_w == 0.0:
        loss_global = weighted_concept_bce(outputs["final_logits"], batch["global_targets"].to(device), pos_weight=args.global_pos_weight)
        loss_mask = outputs["final_logits"].sum() * 0.0
    else:
        loss_global, loss_mask = sgcbm_concept_losses(
            outputs["final_logits"],
            outputs["spatial_maps"],
            batch["global_targets"].to(device, non_blocking=True),
            batch["mask_indices"].to(device, non_blocking=True),
            batch["mask_targets"].to(device, non_blocking=True),
            batch["mask_valid"].to(device, non_blocking=True),
            global_pos_weight=args.global_pos_weight,
        )
    return {
        "total": args.loss_global_w * loss_global + args.loss_mask_w * loss_mask,
        "global": loss_global.detach(),
        "mask": loss_mask.detach(),
    }


def train_one_epoch(backbone: MedicalBackbone, head: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, args: argparse.Namespace) -> Dict[str, float]:
    backbone.eval()
    head.train()
    totals = {"loss": 0.0, "global": 0.0, "mask": 0.0}
    count = 0
    for batch in tqdm(loader, desc="medical cbl train", leave=False):
        images = batch["image"].to(args.device, non_blocking=True)
        outputs = head(backbone(images))
        losses = compute_losses(outputs, batch, args, args.device)
        optimizer.zero_grad()
        losses["total"].backward()
        optimizer.step()
        n = int(images.shape[0])
        totals["loss"] += float(losses["total"].detach().cpu()) * n
        totals["global"] += float(losses["global"].cpu()) * n
        totals["mask"] += float(losses["mask"].cpu()) * n
        count += n
    return {key: value / max(count, 1) for key, value in totals.items()}


@torch.no_grad()
def evaluate_cbl(backbone: MedicalBackbone, head: nn.Module, loader: DataLoader, args: argparse.Namespace) -> Dict[str, float]:
    backbone.eval()
    head.eval()
    totals = {"loss": 0.0, "global": 0.0, "mask": 0.0}
    count = 0
    for batch in tqdm(loader, desc="medical cbl val", leave=False):
        images = batch["image"].to(args.device, non_blocking=True)
        outputs = head(backbone(images))
        losses = compute_losses(outputs, batch, args, args.device)
        n = int(images.shape[0])
        totals["loss"] += float(losses["total"].detach().cpu()) * n
        totals["global"] += float(losses["global"].cpu()) * n
        totals["mask"] += float(losses["mask"].cpu()) * n
        count += n
    return {key: value / max(count, 1) for key, value in totals.items()}


@torch.no_grad()
def extract_concepts(backbone: MedicalBackbone, head: nn.Module, loader: DataLoader, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    backbone.eval()
    head.eval()
    concept_chunks = []
    label_chunks = []
    for batch in tqdm(loader, desc="extract medical concepts"):
        images = batch["image"].to(device, non_blocking=True)
        outputs = head(backbone(images))
        concept_chunks.append(outputs["final_logits"].cpu())
        label_chunks.append(batch["target"].cpu())
    return torch.cat(concept_chunks), torch.cat(label_chunks)


@torch.no_grad()
def extract_concepts_resumable(
    backbone: MedicalBackbone,
    head: nn.Module,
    dataset,
    args: argparse.Namespace,
    *,
    run_dir: Path,
    split: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    final_cache = run_dir / f"raw_concept_cache_{split}.pt"
    if final_cache.exists():
        payload = _torch_load(final_cache)
        return payload["concepts"].float(), payload["targets"].float()

    chunk_size = max(1, int(args.extract_chunk_size))
    batch_size = max(1, int(args.extract_batch_size or args.batch_size))
    cache_dir = run_dir / "extract_cache" / split
    cache_dir.mkdir(parents=True, exist_ok=True)
    chunk_paths: list[Path] = []
    backbone.eval()
    head.eval()

    for start in range(0, len(dataset), chunk_size):
        stop = min(start + chunk_size, len(dataset))
        chunk_path = cache_dir / f"chunk_{start:06d}_{stop:06d}.pt"
        chunk_paths.append(chunk_path)
        if chunk_path.exists():
            print(f"[medical extract] {split} chunk exists {chunk_path}", flush=True)
            continue
        subset = torch.utils.data.Subset(dataset, list(range(start, stop)))
        workers = min(int(args.num_workers), 4)
        loader_kwargs: Dict[str, Any] = {
            "batch_size": batch_size,
            "shuffle": False,
            "num_workers": workers,
            "pin_memory": True,
            "collate_fn": medical_collate,
        }
        if workers > 0:
            loader_kwargs["prefetch_factor"] = 1
        loader = DataLoader(subset, **loader_kwargs)
        concept_chunks: list[torch.Tensor] = []
        label_chunks: list[torch.Tensor] = []
        for batch in tqdm(loader, desc=f"extract {split} concepts {start}:{stop}", leave=False):
            images = batch["image"].to(args.device, non_blocking=True)
            outputs = head(backbone(images))
            concept_chunks.append(outputs["final_logits"].cpu())
            label_chunks.append(batch["target"].cpu())
        payload = {"concepts": torch.cat(concept_chunks), "targets": torch.cat(label_chunks), "start": start, "stop": stop}
        tmp_path = chunk_path.with_suffix(chunk_path.suffix + ".tmp")
        torch.save(payload, tmp_path)
        tmp_path.replace(chunk_path)
        print(f"[medical extract] wrote {split} chunk {chunk_path}", flush=True)

    concepts = torch.cat([_torch_load(path)["concepts"].float() for path in chunk_paths], dim=0)
    labels = torch.cat([_torch_load(path)["targets"].float() for path in chunk_paths], dim=0)
    torch.save({"concepts": concepts.cpu(), "targets": labels.cpu(), "num_rows": len(dataset)}, final_cache)
    print(f"[medical extract] wrote final raw cache {final_cache}", flush=True)
    return concepts, labels


def save_concept_cache(run_dir: Path, split: str, concepts: torch.Tensor, labels: torch.Tensor, args: argparse.Namespace) -> None:
    if args.dataset == "chexpert":
        split_csv = args.train_csv if split == "train" else args.val_csv
        if not split_csv:
            split_csv = str(Path(args.data_dir) / ("valid.csv" if split == "valid" else "train.csv"))
        label_csv = None
    else:
        split_csv = args.mimic_split_csv or None
        label_csv = args.mimic_label_csv or None
    meta = {
        "split": split,
        "data_dir": args.data_dir,
        "img_root": args.img_root or None,
        "split_csv": split_csv,
        "label_csv": label_csv,
        "metadata_csv": args.mimic_metadata_csv or None,
        "path_col": None,
        "num_concepts": int(concepts.shape[1]),
    }
    torch.save({"meta": meta, "concepts": concepts.cpu(), "targets": labels.cpu()}, run_dir / f"concept_cache_{split}.pt")


def train_dense_final(train_x: torch.Tensor, train_y: torch.Tensor, val_x: torch.Tensor, val_y: torch.Tensor, args: argparse.Namespace) -> Tuple[torch.Tensor, torch.Tensor]:
    linear = nn.Linear(train_x.shape[1], train_y.shape[1]).to(args.device)
    optimizer = torch.optim.AdamW(linear.parameters(), lr=args.final_lr)
    criterion = nn.BCEWithLogitsLoss()
    best = None
    best_loss = float("inf")
    train_ds = torch.utils.data.TensorDataset(train_x.float(), train_y.float())
    train_loader = DataLoader(train_ds, batch_size=args.final_batch_size, shuffle=True)
    for _epoch in tqdm(range(args.final_epochs), desc="dense final"):
        linear.train()
        for features, labels in train_loader:
            features = features.to(args.device)
            labels = labels.to(args.device)
            optimizer.zero_grad()
            loss = criterion(linear(features), labels)
            loss.backward()
            optimizer.step()
        linear.eval()
        with torch.no_grad():
            val_loss = float(criterion(linear(val_x.to(args.device)), val_y.to(args.device)).cpu())
        if val_loss < best_loss:
            best_loss = val_loss
            best = {key: value.detach().cpu().clone() for key, value in linear.state_dict().items()}
    if best is not None:
        linear.load_state_dict(best)
    return linear.weight.detach().cpu(), linear.bias.detach().cpu()


def train_saga_final(train_x: torch.Tensor, train_y: torch.Tensor, val_x: torch.Tensor, val_y: torch.Tensor, args: argparse.Namespace) -> Tuple[torch.Tensor, torch.Tensor]:
    from glm_saga.elasticnet import IndexedTensorDataset, glm_saga

    train_loader = DataLoader(IndexedTensorDataset(train_x.float(), train_y.float()), batch_size=args.final_batch_size, shuffle=True)
    val_loader = DataLoader(torch.utils.data.TensorDataset(val_x.float(), val_y.float()), batch_size=args.final_batch_size, shuffle=False)
    linear = nn.Linear(train_x.shape[1], train_y.shape[1]).to(args.device)
    linear.weight.data.zero_()
    linear.bias.data.zero_()
    output = glm_saga(
        linear,
        train_loader,
        max_lr=args.saga_max_lr,
        nepochs=args.saga_iters,
        alpha=0.99,
        epsilon=1.0,
        k=1,
        val_loader=val_loader,
        do_zero=False,
        metadata={"max_reg": {"nongrouped": args.saga_lam}},
        n_ex=len(train_x),
        n_classes=train_y.shape[1],
        family="multilabel",
    )
    return output["path"][0]["weight"].cpu(), output["path"][0]["bias"].cpu()


def evaluate_nec_weights(
    val_z: torch.Tensor,
    val_labels: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    labels: list[str],
    concepts: list[str],
    args: argparse.Namespace,
    run_dir: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for nec in parse_nec_values(args.nec_values):
        target_sparsity = min(float(nec) / max(len(concepts), 1), 1.0)
        sparse_weight = threshold_weight_truncation(weight, target_sparsity)
        logits = val_z @ sparse_weight.T + bias
        probs = torch.sigmoid(logits).numpy()
        metrics = compute_medical_metrics(val_labels.numpy(), probs, labels, threshold=args.threshold)
        torch.save(sparse_weight.cpu(), run_dir / f"W_g@NEC={nec}.pt")
        torch.save(bias.cpu(), run_dir / f"b_g@NEC={nec}.pt")
        nnz = int((sparse_weight != 0).sum().item())
        rows.append({"NEC": int(nec), "nnz": nnz, "mean_auroc": float(metrics["mean_auroc"]), "mAP": float(metrics["mAP"])})
    if rows:
        with (run_dir / "nec_metrics.json").open("w", encoding="utf-8") as handle:
            json.dump({"metrics": rows, "num_concepts": len(concepts), "labels": labels}, handle, indent=2)
    return rows


def extract_loader_kwargs(args: argparse.Namespace, *, shuffle: bool = False) -> Dict[str, Any]:
    workers = min(int(args.num_workers), 4)
    kwargs: Dict[str, Any] = {
        "batch_size": int(args.batch_size),
        "shuffle": shuffle,
        "num_workers": workers,
        "pin_memory": True,
        "collate_fn": medical_collate,
    }
    if workers > 0:
        kwargs["prefetch_factor"] = 1
    return kwargs


def train_medical_cbm(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.device = str(device)
    precomputed_cache_paths = resolve_precomputed_cache_paths(args.precomputed_target_dir)
    args.train_target_cache = args.train_target_cache or precomputed_cache_paths.get("train_target_cache", "")
    args.val_target_cache = args.val_target_cache or precomputed_cache_paths.get("val_target_cache", "")
    args.train_presence_cache = args.train_presence_cache or precomputed_cache_paths.get("train_presence_cache", "")
    args.val_presence_cache = args.val_presence_cache or precomputed_cache_paths.get("val_presence_cache", "")
    label_subset = args.label_subset
    model_name = normalize_model_name(args.model_name)
    labels = medical_labels(args.dataset, competition=label_subset == "competition", pathology=label_subset == "pathology")
    concepts = load_concepts(args.concept_file)
    run_name = args.run_name or f"{args.dataset}_{model_name}"
    run_dir = Path(args.save_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds = build_datasets(args, labels)
    train_ds = maybe_subset(train_ds, args.max_train_images)
    val_ds = maybe_subset(val_ds, args.max_val_images)
    concept_filter: Dict[str, Any] = {
        "original_num_concepts": len(concepts),
        "min_concept_freq": float(args.min_concept_freq),
        "max_concept_freq": float(args.max_concept_freq),
        "concept_threshold": float(args.concept_threshold),
        "presence_mode": args.presence_mode,
    }
    if args.precomputed_target_dir:
        concept_filter["precomputed_target_dir"] = str(Path(args.precomputed_target_dir))
    if model_name == "vlg_cbm":
        train_cache = args.train_concept_cache or find_concept_cache(args.train_annotation_dir, n_rows=len(train_ds), n_concepts=len(concepts), threshold=args.concept_threshold)
        val_cache = args.val_concept_cache or find_concept_cache(args.val_annotation_dir, n_rows=len(val_ds), n_concepts=len(concepts), threshold=args.concept_threshold)
        if train_cache and val_cache:
            train_targets = load_concept_cache_targets(train_ds, cache_path=train_cache, concepts=concepts, allow_index_fallback=args.allow_annotation_index_fallback)
            val_targets = load_concept_cache_targets(val_ds, cache_path=val_cache, concepts=concepts, allow_index_fallback=args.allow_annotation_index_fallback)
        elif args.train_target_cache and args.val_target_cache:
            train_targets = get_or_build_targets(train_ds, args, concepts, args.train_annotation_dir, args.train_target_cache)
            val_targets = get_or_build_targets(val_ds, args, concepts, args.val_annotation_dir, args.val_target_cache)
        else:
            if not args.train_annotation_dir or not args.val_annotation_dir:
                raise ValueError("VLG-CBM requires concept caches, precomputed targets, or --train_annotation_dir/--val_annotation_dir JSON annotations")
            train_targets = get_or_build_presence_targets(
                train_ds,
                args,
                concepts,
                annotation_dir=args.train_annotation_dir,
                cache_path=args.train_presence_cache,
            )
            val_targets = get_or_build_presence_targets(
                val_ds,
                args,
                concepts,
                annotation_dir=args.val_annotation_dir,
                cache_path=args.val_presence_cache,
            )
        concepts, train_targets, val_targets, kept_indices, frequencies = apply_concept_frequency_filter(concepts, train_targets, val_targets, min_freq=args.min_concept_freq, max_freq=args.max_concept_freq)
        concept_filter.update({"kept_indices": kept_indices.tolist(), "train_frequencies": frequencies[kept_indices].tolist(), "filtered_num_concepts": len(concepts)})
        train_target_ds = MedicalConceptDataset(train_ds, train_targets)
        val_target_ds = MedicalConceptDataset(val_ds, val_targets)
        collate_fn = medical_concept_collate
    else:
        if not args.train_annotation_dir or not args.val_annotation_dir:
            raise ValueError("SG-CBM requires --train_annotation_dir and --val_annotation_dir")
        if args.train_target_cache or args.val_target_cache:
            train_targets = get_or_build_targets(train_ds, args, concepts, args.train_annotation_dir, args.train_target_cache)
            val_targets = get_or_build_targets(val_ds, args, concepts, args.val_annotation_dir, args.val_target_cache)
            concepts, train_targets, val_targets, kept_indices, frequencies = apply_concept_frequency_filter(concepts, train_targets, val_targets, min_freq=args.min_concept_freq, max_freq=args.max_concept_freq)
            concept_filter.update({"kept_indices": kept_indices.tolist(), "train_frequencies": frequencies[kept_indices].tolist(), "filtered_num_concepts": len(concepts)})
            train_target_ds = MedicalTargetDataset(train_ds, train_targets)
            val_target_ds = MedicalTargetDataset(val_ds, val_targets)
        else:
            filter_reference_targets = build_filter_reference_targets(train_ds, args, concepts)
            concepts, kept_indices, frequencies = concept_frequency_filter_indices(
                concepts,
                filter_reference_targets,
                min_freq=args.min_concept_freq,
                max_freq=args.max_concept_freq,
            )
            concept_filter.update({"kept_indices": kept_indices.tolist(), "train_frequencies": frequencies[kept_indices].tolist(), "filtered_num_concepts": len(concepts)})
            train_targets = {"target_mode": "lazy", "annotation_dir": args.train_annotation_dir, "num_concepts": len(concepts), "num_images": len(train_ds)}
            val_targets = {"target_mode": "lazy", "annotation_dir": args.val_annotation_dir, "num_concepts": len(concepts), "num_images": len(val_ds)}
            train_target_ds = LazyMedicalTargetDataset(
                train_ds,
                annotation_dir=args.train_annotation_dir,
                concepts=concepts,
                mask_h=args.mask_h,
                mask_w=args.mask_w,
                concept_threshold=args.concept_threshold,
                neg_threshold=args.neg_threshold,
                presence_mode=args.presence_mode,
                target_mode=args.target_mode,
                input_size=args.img_size,
                resize_size=args.resize_size,
                patch_iou_thresh=args.patch_iou_thresh,
                allow_index_fallback=args.allow_annotation_index_fallback,
            )
            val_target_ds = LazyMedicalTargetDataset(
                val_ds,
                annotation_dir=args.val_annotation_dir,
                concepts=concepts,
                mask_h=args.mask_h,
                mask_w=args.mask_w,
                concept_threshold=args.concept_threshold,
                neg_threshold=args.neg_threshold,
                presence_mode=args.presence_mode,
                target_mode=args.target_mode,
                input_size=args.img_size,
                resize_size=args.resize_size,
                patch_iou_thresh=args.patch_iou_thresh,
                allow_index_fallback=args.allow_annotation_index_fallback,
            )
        collate_fn = medical_target_collate
    train_loader = DataLoader(train_target_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_target_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)

    backbone = MedicalBackbone(args.backbone, pretrained=args.pretrained and not args.backbone_ckpt, checkpoint=args.backbone_ckpt).to(device)
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    head = build_head(backbone, len(concepts), args).to(device)
    if args.concept_head_ckpt:
        state = _torch_load(args.concept_head_ckpt)
        head.load_state_dict(state)
        print(f"[medical] loaded concept head checkpoint: {args.concept_head_ckpt}", flush=True)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state = None
    best_val = float("inf")
    stale_epochs = 0
    metrics_path = run_dir / "metrics.jsonl"
    if args.epochs <= 0:
        if not args.concept_head_ckpt:
            raise ValueError("--epochs 0 requires --concept_head_ckpt")
        print("[medical] skipping CBL training because --epochs <= 0", flush=True)
        torch.save(head.state_dict(), run_dir / "concept_head_best.pt")
    else:
        for epoch in range(1, args.epochs + 1):
            train_metrics = train_one_epoch(backbone, head, train_loader, optimizer, args)
            val_metrics = evaluate_cbl(backbone, head, val_loader, args)
            print(f"[medical] epoch={epoch} train={train_metrics} val={val_metrics}", flush=True)
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"epoch": epoch, "train": train_metrics, "val": val_metrics}) + "\n")
            improved = val_metrics["loss"] < (best_val - float(args.early_stop_min_delta))
            if improved:
                best_val = val_metrics["loss"]
                stale_epochs = 0
                best_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
                torch.save(best_state, run_dir / "concept_head_best.pt")
            else:
                stale_epochs += 1
            if int(args.early_stop_patience) > 0 and stale_epochs >= int(args.early_stop_patience):
                print(f"[medical] early stopping at epoch={epoch} best_val_loss={best_val:.6f}", flush=True)
                break
    if best_state is not None:
        head.load_state_dict(best_state)
    torch.save(head.state_dict(), run_dir / "concept_head_final.pt")

    train_concepts, train_labels = extract_concepts_resumable(backbone, head, train_ds, args, run_dir=run_dir, split="train")
    val_concepts, val_labels = extract_concepts_resumable(backbone, head, val_ds, args, run_dir=run_dir, split="valid")
    train_z, val_z, mean, std = standardize_from_train(train_concepts, val_concepts, unbiased=False)
    save_concept_cache(run_dir, "train", train_z, train_labels, args)
    save_concept_cache(run_dir, "valid", val_z, val_labels, args)
    W, b = train_saga_final(train_z, train_labels, val_z, val_labels, args) if args.use_saga else train_dense_final(train_z, train_labels, val_z, val_labels, args)
    logits = val_z @ W.T + b
    probs = torch.sigmoid(logits).numpy()
    final_metrics = compute_medical_metrics(val_labels.numpy(), probs, labels, threshold=args.threshold)

    torch.save(W, run_dir / "W_g.pt")
    torch.save(b, run_dir / "b_g.pt")
    nec_rows = evaluate_nec_weights(val_z, val_labels, W, b, labels, concepts, args, run_dir)
    torch.save(mean, run_dir / "concept_mean.pt")
    torch.save(std, run_dir / "concept_std.pt")
    (run_dir / "concepts.txt").write_text("\n".join(concepts), encoding="utf-8")
    (run_dir / "labels.txt").write_text("\n".join(labels), encoding="utf-8")
    with (run_dir / "concept_filter.json").open("w", encoding="utf-8") as handle:
        json.dump(concept_filter, handle, indent=2)
    with (run_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump({**vars(args), "labels": labels, "num_concepts": len(concepts), "concept_filter": {k: v for k, v in concept_filter.items() if k != "train_frequencies"}, "train_target_summary": {k: v for k, v in train_targets.items() if isinstance(v, (int, float, str))}, "val_target_summary": {k: v for k, v in val_targets.items() if isinstance(v, (int, float, str))}}, handle, indent=2)
    with (run_dir / "val_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(final_metrics, handle, indent=2)
    print(f"[medical] validation mean AUROC={final_metrics['mean_auroc']:.4f} mAP={final_metrics['mAP']:.4f}", flush=True)
    if nec_rows:
        print(f"[medical] NEC metrics={nec_rows}", flush=True)
    print(f"[medical] saved to {run_dir}", flush=True)


def main(argv: Sequence[str] | None = None) -> None:
    train_medical_cbm(parse_args(argv))


if __name__ == "__main__":
    main()
