from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from gcbm.imagenet_core import (
    autocast_context,
    cuda_peak_stats_mb,
    prepare_images,
    reset_cuda_peak_stats_if_needed,
)
from glm_saga.elasticnet import glm_saga


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


def feature_storage_dtype(cfg: Any) -> np.dtype:
    if cfg.feature_storage_dtype == "fp32":
        return np.float32
    return np.float16


@torch.no_grad()
def extract_concept_features_to_memmap(
    backbone: nn.Module,
    head: nn.Module,
    loader: DataLoader,
    cfg: Any,
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
    cfg: Any,
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


def _final_layer_loader_kwargs(cfg: Any, *, shuffle: bool) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "batch_size": cfg.saga_batch_size,
        "shuffle": shuffle,
        "num_workers": cfg.saga_workers,
        "pin_memory": cfg.pin_memory,
        "drop_last": False,
    }
    if cfg.saga_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = cfg.saga_prefetch_factor
    return kwargs


def train_sparse_final_layer(
    train_feature_path: Path,
    train_target_path: Path,
    val_feature_path: Path,
    val_target_path: Path,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
    cfg: Any,
    n_classes: int,
    run_dir: Path,
) -> Dict[str, Any]:
    feature_mean_np = feature_mean.cpu().numpy()
    feature_std_np = feature_std.cpu().numpy()
    train_dataset = MemmapFeatureDataset(train_feature_path, train_target_path, mean=feature_mean_np, std=feature_std_np, include_index=True)
    train_eval_dataset = MemmapFeatureDataset(train_feature_path, train_target_path, mean=feature_mean_np, std=feature_std_np, include_index=False)
    val_dataset = MemmapFeatureDataset(val_feature_path, val_target_path, mean=feature_mean_np, std=feature_std_np, include_index=False)

    train_loader = DataLoader(train_dataset, **_final_layer_loader_kwargs(cfg, shuffle=True))
    train_eval_loader = DataLoader(train_eval_dataset, **_final_layer_loader_kwargs(cfg, shuffle=False))
    val_loader = DataLoader(val_dataset, **_final_layer_loader_kwargs(cfg, shuffle=False))

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
        **cuda_peak_stats_mb(cfg),
    }
    torch.save({"weight": best["weight"], "bias": best["bias"]}, run_dir / "final_layer_glm_saga.pt")
    (run_dir / "final_layer_summary.json").write_text(json.dumps(payload, indent=2))
    return payload


def train_dense_final_layer(
    train_feature_path: Path,
    train_target_path: Path,
    val_feature_path: Path,
    val_target_path: Path,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
    cfg: Any,
    n_classes: int,
    run_dir: Path,
) -> Dict[str, Any]:
    feature_mean_np = feature_mean.cpu().numpy()
    feature_std_np = feature_std.cpu().numpy()
    train_dataset = MemmapFeatureDataset(train_feature_path, train_target_path, mean=feature_mean_np, std=feature_std_np, include_index=False)
    val_dataset = MemmapFeatureDataset(val_feature_path, val_target_path, mean=feature_mean_np, std=feature_std_np, include_index=False)
    train_loader = DataLoader(train_dataset, **_final_layer_loader_kwargs(cfg, shuffle=True))
    val_loader = DataLoader(val_dataset, **_final_layer_loader_kwargs(cfg, shuffle=False))

    linear = nn.Linear(int(train_dataset.features.shape[1]), int(n_classes), bias=True).to(cfg.device)
    optimizer = torch.optim.Adam(linear.parameters(), lr=cfg.dense_lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    best_val_loss = float("inf")
    best_state = None
    history: list[Dict[str, Any]] = []
    reset_cuda_peak_stats_if_needed(cfg)
    start_time = time.perf_counter()
    for epoch_idx in range(cfg.dense_n_iters):
        linear.train()
        for batch in train_loader:
            features, targets = batch[0].to(cfg.device), batch[1].to(cfg.device)
            optimizer.zero_grad(set_to_none=True)
            logits = linear(features)
            loss = F.cross_entropy(logits, targets, reduction="mean")
            loss.backward()
            optimizer.step()

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
        **cuda_peak_stats_mb(cfg),
    }
    torch.save(
        {"weight": best_state["weight"], "bias": best_state["bias"], "epoch": best_state["epoch"]},
        run_dir / "final_layer_dense.pt",
    )
    (run_dir / "final_layer_dense_summary.json").write_text(json.dumps(payload, indent=2))
    return payload
