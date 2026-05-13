"""Simplified ImageNet SG-CBM CBL trainer.

This entrypoint intentionally keeps only the path we use for paper-scale
ImageNet CBL training:
- precomputed GDINO targets are used when available, otherwise GDINO targets are
  built on the fly from annotations,
- shared, dual, and multiscale dual heads are selectable by CLI,
- spatial supervision is soft-align KL,
- sparse GLM or dense final-layer training is optional after CBL training.

It reuses the dataset, precompute store, and model primitives from the
standalone driver so we do not duplicate the fragile ImageNet annotation/cache
logic here.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gcbm.losses import sgcbm_concept_losses  # noqa: E402
from gcbm.imagenet_core import (  # noqa: E402
    Config,
    DatasetView,
    SafeImageFolderWithAnnotations,
    apply_count_concept_filter,
    autocast_context,
    batch_targets_to_device,
    build_loader,
    build_gdino_targets,
    build_model,
    build_run_dir,
    compute_feature_stats_memmap,
    configure_runtime,
    cuda_peak_stats_mb,
    extract_concept_features_to_memmap,
    init_global_head_from_vlg,
    load_run_concepts,
    make_optimizer,
    make_scheduler,
    prepare_images,
    reset_cuda_peak_stats_if_needed,
    save_checkpoint,
    select_subset_indices,
    split_train_val,
    train_dense_final_layer,
    train_sparse_final_layer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the SG-CBM concept layer on ImageNet.")
    parser.add_argument("--train_root", required=True)
    parser.add_argument(
        "--train_manifest",
        default="",
        help="Optional JSONL manifest with path/class_id/sample_index entries.",
    )
    parser.add_argument("--annotation_dir", required=True)
    parser.add_argument(
        "--precomputed_target_dir",
        default="",
        help="Optional GDINO target cache. If omitted, targets are generated from annotations on the fly.",
    )
    parser.add_argument("--concept_file", default="concept_files/imagenet_filtered.txt")
    parser.add_argument("--val_root", default="")
    parser.add_argument("--save_dir", default="artifacts/imagenet")
    parser.add_argument("--run_name", default="")
    parser.add_argument("--feature_dir", default="", help="Optional directory for extracted GLM features.")
    parser.add_argument("--max_train_images", type=int, default=0)
    parser.add_argument("--max_val_images", type=int, default=0)
    parser.add_argument("--val_split", type=float, default=0.1)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--persistent_workers", action="store_true", default=True)
    parser.add_argument("--disable_persistent_workers", action="store_true")
    parser.add_argument("--pin_memory", action="store_true", default=True)
    parser.add_argument("--disable_pin_memory", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--resnet50_weights",
        choices=["v1", "v2"],
        default="v2",
        help="Torchvision ResNet-50 ImageNet weights. Use v1 for checkpoints compatible with original VLG-CBM.",
    )
    parser.add_argument("--amp", choices=["fp16", "bf16", "none"], default="fp16")
    parser.add_argument("--channels_last", action="store_true", default=True)
    parser.add_argument("--disable_channels_last", action="store_true")
    parser.add_argument("--tf32", action="store_true", default=True)
    parser.add_argument("--disable_tf32", action="store_true")
    parser.add_argument("--cudnn_benchmark", action="store_true", default=True)
    parser.add_argument("--disable_cudnn_benchmark", action="store_true")
    parser.add_argument("--seed", type=int, default=6885)

    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--mask_h", type=int, default=14)
    parser.add_argument("--mask_w", type=int, default=14)
    parser.add_argument("--min_image_bytes", type=int, default=2048)
    parser.add_argument("--patch_iou_thresh", type=float, default=0.5)
    parser.add_argument("--concept_threshold", type=float, default=0.15)
    parser.add_argument(
        "--spatial_target_mode",
        choices=["hard_iou", "soft_box"],
        default="soft_box",
        help="Rasterization used for on-the-fly or precomputed GDINO boxes.",
    )

    parser.add_argument("--filter_concepts_by_count", action="store_true")
    parser.add_argument("--concept_min_count", type=int, default=1)
    parser.add_argument("--concept_min_frequency", type=float, default=0.0)
    parser.add_argument("--concept_max_frequency", type=float, default=1.0)

    parser.add_argument("--optimizer", choices=["sgd", "adamw"], default="adamw")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--scheduler", choices=["none", "cosine"], default="cosine")

    parser.add_argument("--global_pos_weight", type=float, default=100.0)
    parser.add_argument("--loss_global_w", type=float, default=1.0)
    parser.add_argument("--loss_mask_w", type=float, default=1.0)

    parser.add_argument(
        "--branch_arch",
        choices=["shared", "dual"],
        default="dual",
        help="Concept-head architecture. Use dual for SG-CBM residual spatial logits.",
    )
    parser.add_argument(
        "--spatial_branch_mode",
        choices=["shared_stage", "multiscale_conv45"],
        default="multiscale_conv45",
        help="Spatial branch source. multiscale_conv45 is the SG-CBM paper/default path.",
    )
    parser.add_argument("--spatial_stage", choices=["conv4", "conv5"], default="conv5")
    parser.add_argument("--residual_alpha", type=float, default=0.2)
    parser.add_argument("--residual_spatial_pooling", choices=["avg", "lse"], default="lse")
    parser.add_argument(
        "--learn_spatial_residual_scale",
        action="store_true",
        help="Learn a positive exp(log_scale) multiplier on the spatial residual branch.",
    )

    parser.add_argument("--vlg_init_path", default="")
    parser.add_argument("--vlg_concepts_path", default="")
    parser.add_argument("--freeze_global_head", action="store_true")

    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument(
        "--eval_every",
        type=int,
        default=1,
        help="Run validation every N epochs. Set 0 to skip CBL validation for fastest timing runs.",
    )
    parser.add_argument("--print_config", action="store_true")

    parser.add_argument("--train_glm_after_cbl", action="store_true")
    parser.add_argument(
        "--train_final_layer_after_cbl",
        action="store_true",
        help="Train the selected final classifier after CBL. --train_glm_after_cbl is kept as a sparse alias.",
    )
    parser.add_argument(
        "--final_layer_type",
        choices=["sparse", "dense"],
        default="sparse",
        help="Final classifier trained after CBL when enabled.",
    )
    parser.add_argument("--feature_batch_size", type=int, default=256)
    parser.add_argument("--feature_workers", type=int, default=4)
    parser.add_argument("--feature_prefetch_factor", type=int, default=2)
    parser.add_argument("--saga_lam", type=float, default=5e-4)
    parser.add_argument("--saga_n_iters", type=int, default=200)
    parser.add_argument("--saga_step_size", type=float, default=0.02)
    parser.add_argument("--saga_batch_size", type=int, default=4096)
    parser.add_argument("--saga_workers", type=int, default=0)
    parser.add_argument("--saga_prefetch_factor", type=int, default=2)
    parser.add_argument("--saga_verbose_every", type=int, default=20)
    parser.add_argument("--saga_table_device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--dense_lr", type=float, default=1e-3)
    parser.add_argument("--dense_n_iters", type=int, default=20)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    """Populate the full standalone Config schema for eval compatibility."""
    return Config(
        mode="train",
        train_root=args.train_root,
        train_manifest=args.train_manifest,
        annotation_dir=args.annotation_dir,
        concept_file=args.concept_file,
        val_root=args.val_root,
        save_dir=args.save_dir,
        run_name=args.run_name,
        reuse_run_dir="",
        feature_dir=args.feature_dir,
        precomputed_target_dir=args.precomputed_target_dir,
        persist_feature_copy=False,
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
        resnet50_weights=str(args.resnet50_weights),
        train_random_transforms=False,
        mask_h=args.mask_h,
        mask_w=args.mask_w,
        patch_iou_thresh=float(args.patch_iou_thresh),
        concept_threshold=float(args.concept_threshold),
        spatial_target_mode=str(args.spatial_target_mode),
        spatial_loss_mode="soft_align",
        filter_concepts_by_count=bool(args.filter_concepts_by_count),
        concept_min_count=max(0, int(args.concept_min_count)),
        concept_min_frequency=float(args.concept_min_frequency),
        concept_max_frequency=float(args.concept_max_frequency),
        optimizer=args.optimizer,
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        global_pos_weight=args.global_pos_weight,
        patch_pos_weight=1.0,
        loss_global_w=args.loss_global_w,
        loss_mask_w=args.loss_mask_w,
        branch_arch=str(args.branch_arch),
        spatial_branch_mode=args.spatial_branch_mode,
        spatial_stage=args.spatial_stage,
        residual_alpha=args.residual_alpha,
        profile_steps=0,
        warmup_steps=0,
        log_every=max(1, int(args.log_every)),
        save_every=max(1, int(args.save_every)),
        eval_every=max(0, int(args.eval_every)),
        skip_final_layer=not bool(args.train_glm_after_cbl or args.train_final_layer_after_cbl),
        final_layer_type="sparse" if bool(args.train_glm_after_cbl) else str(args.final_layer_type),
        saga_batch_size=max(1, int(args.saga_batch_size)),
        saga_workers=max(0, int(args.saga_workers)),
        saga_prefetch_factor=max(1, int(args.saga_prefetch_factor)),
        saga_step_size=float(args.saga_step_size),
        saga_lam=float(args.saga_lam),
        saga_n_iters=max(1, int(args.saga_n_iters)),
        saga_verbose_every=max(1, int(args.saga_verbose_every)),
        dense_lr=float(args.dense_lr),
        dense_n_iters=max(1, int(args.dense_n_iters)),
        feature_storage_dtype="fp16",
        saga_table_device=str(args.saga_table_device),
        vlg_init_path=str(args.vlg_init_path or ""),
        vlg_concepts_path=str(args.vlg_concepts_path or ""),
        freeze_global_head=bool(args.freeze_global_head),
        scheduler=str(args.scheduler or "none"),
        print_config=bool(args.print_config),
        residual_spatial_pooling=str(args.residual_spatial_pooling),
        learn_spatial_residual_scale=bool(args.learn_spatial_residual_scale),
        feature_batch_size=max(1, int(args.feature_batch_size)),
        feature_workers=max(0, int(args.feature_workers)),
        feature_prefetch_factor=max(1, int(args.feature_prefetch_factor)),
    )


def validate_config(cfg: Config) -> None:
    if not cfg.train_root:
        raise ValueError("--train_root is required")
    if not cfg.annotation_dir:
        raise ValueError("--annotation_dir is required")
    if cfg.spatial_branch_mode == "multiscale_conv45" and cfg.branch_arch != "dual":
        raise ValueError("--spatial_branch_mode=multiscale_conv45 requires --branch_arch=dual")
    if cfg.spatial_loss_mode != "soft_align":
        raise ValueError("This release trainer only supports soft-align KL spatial loss")
    if cfg.freeze_global_head and not cfg.vlg_init_path:
        raise ValueError("--freeze_global_head requires --vlg_init_path")
    if cfg.vlg_init_path and not cfg.vlg_concepts_path:
        raise ValueError("--vlg_concepts_path is required when --vlg_init_path is set")
    if cfg.residual_spatial_pooling not in {"avg", "lse"}:
        raise ValueError("--residual_spatial_pooling must be one of: avg, lse")
    if not 0.0 <= cfg.concept_min_frequency <= 1.0:
        raise ValueError("--concept_min_frequency must be in [0, 1]")
    if not 0.0 <= cfg.concept_max_frequency <= 1.0:
        raise ValueError("--concept_max_frequency must be in [0, 1]")
    if cfg.concept_min_frequency > cfg.concept_max_frequency:
        raise ValueError("--concept_min_frequency cannot exceed --concept_max_frequency")


def compute_cbl_losses(
    outputs: Dict[str, torch.Tensor],
    global_targets: torch.Tensor,
    mask_indices: torch.Tensor,
    mask_targets: torch.Tensor,
    mask_valid: torch.Tensor,
    cfg: Config,
) -> Dict[str, torch.Tensor]:
    """Global weighted BCE plus soft-align KL over precomputed target masks."""
    loss_global, loss_mask = sgcbm_concept_losses(
        outputs["final_logits"],
        outputs["spatial_maps"],
        global_targets,
        mask_indices,
        mask_targets,
        mask_valid,
        global_pos_weight=float(cfg.global_pos_weight),
    )
    total = cfg.loss_global_w * loss_global + cfg.loss_mask_w * loss_mask
    return {
        "total": total,
        "global": loss_global.detach(),
        "mask": loss_mask.detach(),
    }


def batch_cbl_targets(
    batch: Dict[str, Any],
    dataset: DatasetView,
    cfg: Config,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if "global_targets" in batch:
        return batch_targets_to_device(batch, cfg)
    return build_gdino_targets(
        batch["annotations"],
        batch["image_sizes"],
        dataset.concept_to_idx,
        len(dataset.concepts),
        cfg,
        cfg.device,
    )


def train_one_epoch(
    backbone: nn.Module,
    head: nn.Module,
    loader: torch.utils.data.DataLoader,
    dataset: DatasetView,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    cfg: Config,
    epoch: int,
) -> Dict[str, float]:
    head.train()
    totals = {"total": 0.0, "global": 0.0, "mask": 0.0, "count": 0}
    start_time = time.perf_counter()
    reset_cuda_peak_stats_if_needed(cfg)

    for step, batch in enumerate(loader, start=1):
        images = prepare_images(batch["images"], cfg)
        global_targets, idx_pad, mask_pad, valid_pad = batch_cbl_targets(batch, dataset, cfg)

        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad(), autocast_context(cfg):
            feats = backbone(images)
        with autocast_context(cfg):
            outputs = head(feats)
            losses = compute_cbl_losses(outputs, global_targets, idx_pad, mask_pad, valid_pad, cfg)
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
        totals["count"] += batch_size

        if step % cfg.log_every == 0:
            elapsed = time.perf_counter() - start_time
            ips = totals["count"] / max(elapsed, 1e-6)
            print(
                f"[train] epoch={epoch} step={step}/{len(loader)} "
                f"loss={totals['total']/totals['count']:.4f} "
                f"global={totals['global']/totals['count']:.4f} "
                f"mask={totals['mask']/totals['count']:.4f} ips={ips:.2f}",
                flush=True,
            )

    count = max(totals["count"], 1)
    elapsed = time.perf_counter() - start_time
    metrics = {
        "loss": totals["total"] / count,
        "loss_global": totals["global"] / count,
        "loss_mask": totals["mask"] / count,
        "images_per_second": totals["count"] / max(elapsed, 1e-6),
        "elapsed_sec": elapsed,
    }
    metrics.update(cuda_peak_stats_mb(cfg))
    return metrics


@torch.no_grad()
def evaluate_one_epoch(
    backbone: nn.Module,
    head: nn.Module,
    loader: torch.utils.data.DataLoader,
    dataset: DatasetView,
    cfg: Config,
    split_name: str,
) -> Dict[str, float]:
    head.eval()
    totals = {"total": 0.0, "global": 0.0, "mask": 0.0, "count": 0}
    start_time = time.perf_counter()
    reset_cuda_peak_stats_if_needed(cfg)

    for batch in loader:
        images = prepare_images(batch["images"], cfg)
        global_targets, idx_pad, mask_pad, valid_pad = batch_cbl_targets(batch, dataset, cfg)
        with torch.inference_mode(), autocast_context(cfg):
            feats = backbone(images)
        with autocast_context(cfg):
            outputs = head(feats)
            losses = compute_cbl_losses(outputs, global_targets, idx_pad, mask_pad, valid_pad, cfg)

        batch_size = int(images.shape[0])
        totals["total"] += float(losses["total"].item()) * batch_size
        totals["global"] += float(losses["global"].item()) * batch_size
        totals["mask"] += float(losses["mask"].item()) * batch_size
        totals["count"] += batch_size

    count = max(totals["count"], 1)
    elapsed = time.perf_counter() - start_time
    metrics = {
        "loss": totals["total"] / count,
        "loss_global": totals["global"] / count,
        "loss_mask": totals["mask"] / count,
        "images_per_second": totals["count"] / max(elapsed, 1e-6),
        "elapsed_sec": elapsed,
    }
    metrics.update(cuda_peak_stats_mb(cfg))
    print(
        f"[{split_name}] loss={metrics['loss']:.4f} "
        f"global={metrics['loss_global']:.4f} mask={metrics['loss_mask']:.4f} "
        f"ips={metrics['images_per_second']:.2f}",
        flush=True,
    )
    return metrics


def build_datasets(cfg: Config) -> Tuple[DatasetView, DatasetView, Optional[Dict[str, Any]]]:
    concepts = load_run_concepts(cfg)

    def attach_targets_if_present(dataset: SafeImageFolderWithAnnotations) -> None:
        if not cfg.precomputed_target_dir:
            return
        target_dir = Path(cfg.precomputed_target_dir) / dataset.split
        if not target_dir.is_dir():
            print(
                f"[targets] no precomputed targets for split={dataset.split} at {target_dir}; "
                "using on-the-fly GDINO targets",
                flush=True,
            )
            return
        dataset.attach_precomputed_targets(cfg.precomputed_target_dir, cfg)
        print(f"[targets] using precomputed targets from {target_dir}", flush=True)

    train_dataset_full = SafeImageFolderWithAnnotations(
        root=cfg.train_root,
        annotation_dir=cfg.annotation_dir,
        concepts=concepts,
        input_size=cfg.input_size,
        min_image_bytes=cfg.min_image_bytes,
        split="train",
        manifest=cfg.train_manifest,
        train_random_transforms=False,
    )
    attach_targets_if_present(train_dataset_full)

    if cfg.val_root:
        val_dataset_full = SafeImageFolderWithAnnotations(
            root=cfg.val_root,
            annotation_dir=cfg.annotation_dir,
            concepts=concepts,
            input_size=cfg.input_size,
            min_image_bytes=cfg.min_image_bytes,
            split="val",
            train_random_transforms=False,
        )
        attach_targets_if_present(val_dataset_full)
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
    return train_dataset, val_dataset, concept_filter_summary


def infer_num_classes(dataset: DatasetView) -> int:
    classes = getattr(dataset.base_dataset.dataset, "classes", None)
    if classes is not None:
        return int(len(classes))
    return int(max(dataset.base_dataset.dataset.targets)) + 1


def train_final_layer_after_cbl(
    backbone: nn.Module,
    head: nn.Module,
    train_dataset: DatasetView,
    val_dataset: DatasetView,
    cfg: Config,
    run_dir: Path,
    best_path: Path,
) -> Dict[str, Any]:
    print(f"[final_layer:{cfg.final_layer_type}] loading best CBL head from {best_path}", flush=True)
    head.load_state_dict(torch.load(best_path, map_location=cfg.device))

    feature_dir = Path(cfg.feature_dir).resolve() if cfg.feature_dir else (run_dir / "features")
    feature_batch_size = int(getattr(cfg, "feature_batch_size", 256))
    feature_workers = int(getattr(cfg, "feature_workers", 4))
    feature_prefetch = int(getattr(cfg, "feature_prefetch_factor", 2))
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
    feature_mean, feature_std, norm_summary = compute_feature_stats_memmap(train_feature_path, cfg)
    torch.save(
        {
            "mean": feature_mean,
            "std": feature_std,
            "train_extraction": train_extract_summary,
            "val_extraction": val_extract_summary,
            "normalization": norm_summary,
            "source_concept_head": str(best_path),
        },
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
        n_classes=infer_num_classes(train_dataset),
        run_dir=run_dir,
    )
    final_layer_summary["type"] = cfg.final_layer_type
    final_layer_summary["feature_extraction"] = {
        "train": train_extract_summary,
        "val": val_extract_summary,
        "normalization": norm_summary,
        "source_concept_head": str(best_path),
    }
    (run_dir / "final_layer_summary.json").write_text(json.dumps(final_layer_summary, indent=2))
    print(
        f"[final_layer:{cfg.final_layer_type}] train_top1={final_layer_summary['train']['top1']:.4f} "
        f"val_top1={final_layer_summary['val']['top1']:.4f} "
        f"sparsity={final_layer_summary['nnz']}/{final_layer_summary['total']}",
        flush=True,
    )
    return final_layer_summary


def run_training(cfg: Config) -> Dict[str, Any]:
    train_dataset, val_dataset, concept_filter_summary = build_datasets(cfg)
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
        train_metrics = train_one_epoch(backbone, head, train_loader, train_dataset, optimizer, scaler, cfg, epoch)
        eval_every = int(getattr(cfg, "eval_every", 1))
        run_val = eval_every > 0 and (epoch % eval_every == 0 or epoch == cfg.epochs)
        if run_val:
            val_metrics = evaluate_one_epoch(backbone, head, val_loader, val_dataset, cfg, split_name="val")
        else:
            val_metrics = {
                "loss": float("inf"),
                "loss_global": float("nan"),
                "loss_mask": float("nan"),
                "images_per_second": 0.0,
                "elapsed_sec": 0.0,
                "skipped": True,
            }
        summary = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        if scheduler is not None:
            summary["lr"] = float(scheduler.get_last_lr()[0])
        history.append(summary)

        epoch_message = (
            f"[epoch] {epoch} "
            f"train_loss={train_metrics['loss']:.4f} train_ips={train_metrics['images_per_second']:.2f} "
        )
        if run_val:
            epoch_message += f"val_loss={val_metrics['loss']:.4f} val_ips={val_metrics['images_per_second']:.2f}"
        else:
            epoch_message += "val=skipped"
        if scheduler is not None:
            epoch_message += f" lr={scheduler.get_last_lr()[0]:.6f}"
        print(epoch_message, flush=True)

        if run_val and val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(head.state_dict(), best_path)
        elif not run_val:
            # With validation disabled, "best" means the latest CBL head.
            torch.save(head.state_dict(), best_path)
        if epoch % cfg.save_every == 0:
            save_checkpoint(run_dir, head, optimizer, epoch, cfg, train_metrics, val_metrics)
        if scheduler is not None:
            scheduler.step()

    final_layer_summary: Optional[Dict[str, Any]] = None
    if not cfg.skip_final_layer:
        final_layer_summary = train_final_layer_after_cbl(
            backbone=backbone,
            head=head,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            cfg=cfg,
            run_dir=run_dir,
            best_path=best_path,
        )

    result = {
        "mode": "train",
        "script": "train_savlg_imagenet.py",
        "run_dir": str(run_dir),
        "n_concepts": len(concepts),
        "history": history,
        "best_val_loss": best_val,
        "best_checkpoint": str(best_path),
        "final_layer": final_layer_summary,
    }
    (run_dir / "summary.json").write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    cfg = build_config(parse_args())
    validate_config(cfg)
    configure_runtime(cfg)
    if cfg.print_config:
        print(json.dumps(asdict(cfg), indent=2, sort_keys=True), flush=True)
    result = run_training(cfg)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
