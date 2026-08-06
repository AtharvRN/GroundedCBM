#!/usr/bin/env python3
"""Train a dense linear classifier from cached concept-feature memmaps."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset


class CachedFeatureDataset(Dataset):
    def __init__(
        self,
        feature_path: Path,
        target_path: Path,
        *,
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.features = np.load(feature_path, mmap_mode="r")
        self.targets = np.load(target_path, mmap_mode="r")
        if len(self.features) != len(self.targets):
            raise ValueError(f"feature/target length mismatch: {len(self.features)} vs {len(self.targets)}")
        self.mean = mean.astype(np.float32, copy=False) if mean is not None else None
        self.std = std.astype(np.float32, copy=False) if std is not None else None
        self.dtype = dtype

    def __len__(self) -> int:
        return int(self.targets.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = np.asarray(self.features[index], dtype=np.float32)
        if self.mean is not None and self.std is not None:
            x = (x - self.mean) / self.std
        y = int(self.targets[index])
        return torch.as_tensor(x, dtype=self.dtype), torch.tensor(y, dtype=torch.long)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_features", required=True, type=Path)
    parser.add_argument("--train_targets", required=True, type=Path)
    parser.add_argument("--val_features", required=True, type=Path)
    parser.add_argument("--val_targets", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--num_classes", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--scheduler_gamma", type=float, default=0.95)
    parser.add_argument("--stats_chunk_rows", type=int, default=65536)
    parser.add_argument("--cache_device", choices=["none", "cuda"], default="none")
    parser.add_argument("--cache_chunk_rows", type=int, default=65536)
    parser.add_argument("--eval_train", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=100)
    return parser.parse_args()


def compute_stats(feature_path: Path, chunk_rows: int) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    features = np.load(feature_path, mmap_mode="r")
    n_rows, n_features = int(features.shape[0]), int(features.shape[1])
    sum_x = np.zeros(n_features, dtype=np.float64)
    sum_x2 = np.zeros(n_features, dtype=np.float64)
    start = time.perf_counter()
    for start_row in range(0, n_rows, int(chunk_rows)):
        end_row = min(start_row + int(chunk_rows), n_rows)
        chunk = np.asarray(features[start_row:end_row], dtype=np.float32)
        sum_x += chunk.sum(axis=0, dtype=np.float64)
        sum_x2 += np.square(chunk, dtype=np.float32).sum(axis=0, dtype=np.float64)
    mean = (sum_x / max(n_rows, 1)).astype(np.float32)
    var = (sum_x2 / max(n_rows, 1)) - np.square(mean.astype(np.float64))
    std = np.sqrt(np.maximum(var, 1e-12)).astype(np.float32)
    std = np.maximum(std, 1e-6)
    summary = {
        "n_examples": n_rows,
        "n_features": n_features,
        "elapsed_sec": time.perf_counter() - start,
        "chunk_rows": int(chunk_rows),
    }
    return mean, std, summary


def topk_accuracy(logits: torch.Tensor, targets: torch.Tensor, k: int) -> float:
    k = min(k, int(logits.shape[1]))
    pred = logits.topk(k, dim=1).indices
    return float(pred.eq(targets.unsqueeze(1)).any(dim=1).float().mean().item())


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> Dict[str, Any]:
    model.eval()
    loss_sum = 0.0
    top1_sum = 0.0
    top5_sum = 0.0
    n = 0
    for features, targets in loader:
        features = features.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(features)
        batch_size = int(targets.shape[0])
        loss_sum += float(F.cross_entropy(logits, targets, reduction="sum").item())
        top1_sum += topk_accuracy(logits, targets, 1) * batch_size
        top5_sum += topk_accuracy(logits, targets, 5) * batch_size
        n += batch_size
    return {
        "loss": loss_sum / max(n, 1),
        "top1": top1_sum / max(n, 1),
        "top5": top5_sum / max(n, 1),
        "n": n,
    }


def load_normalized_feature_tensor(
    feature_path: Path,
    target_path: Path,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    device: str,
    chunk_rows: int,
) -> tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    features = np.load(feature_path, mmap_mode="r")
    targets_np = np.load(target_path, mmap_mode="r")
    n_rows, n_features = int(features.shape[0]), int(features.shape[1])
    start = time.perf_counter()
    tensor = torch.empty((n_rows, n_features), dtype=torch.float32, device=device)
    mean_t = torch.from_numpy(mean).to(device=device, dtype=torch.float32)
    std_t = torch.from_numpy(std).to(device=device, dtype=torch.float32)
    for start_row in range(0, n_rows, int(chunk_rows)):
        end_row = min(start_row + int(chunk_rows), n_rows)
        chunk = torch.from_numpy(np.asarray(features[start_row:end_row], dtype=np.float32)).to(device=device)
        tensor[start_row:end_row].copy_((chunk - mean_t) / std_t)
        print(f"[dense-cache] loaded {feature_path.name}: n={end_row}/{n_rows}", flush=True)
    targets = torch.from_numpy(np.asarray(targets_np, dtype=np.int64)).to(device=device)
    return tensor, targets, {
        "feature_path": str(feature_path),
        "target_path": str(target_path),
        "n_examples": n_rows,
        "n_features": n_features,
        "device": device,
        "elapsed_sec": time.perf_counter() - start,
    }


@torch.no_grad()
def evaluate_tensor_features(
    model: nn.Module,
    features: torch.Tensor,
    targets: torch.Tensor,
    batch_size: int,
) -> Dict[str, Any]:
    model.eval()
    loss_sum = 0.0
    top1_sum = 0.0
    top5_sum = 0.0
    n_rows = int(targets.shape[0])
    for start_row in range(0, n_rows, int(batch_size)):
        end_row = min(start_row + int(batch_size), n_rows)
        logits = model(features[start_row:end_row])
        batch_targets = targets[start_row:end_row]
        batch_size_actual = int(batch_targets.shape[0])
        loss_sum += float(F.cross_entropy(logits, batch_targets, reduction="sum").item())
        top1_sum += topk_accuracy(logits, batch_targets, 1) * batch_size_actual
        top5_sum += topk_accuracy(logits, batch_targets, 5) * batch_size_actual
    return {
        "loss": loss_sum / max(n_rows, 1),
        "top1": top1_sum / max(n_rows, 1),
        "top5": top5_sum / max(n_rows, 1),
        "n": n_rows,
    }


def train_dense_cached_on_device(
    args: argparse.Namespace,
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    val_features: torch.Tensor,
    val_targets: torch.Tensor,
    n_classes: int,
    n_features: int,
    cache_summaries: Dict[str, Any],
    stats_summary: Dict[str, Any],
) -> Dict[str, Any]:
    model = nn.Linear(n_features, n_classes).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=float(args.scheduler_gamma))

    best: Dict[str, Any] | None = None
    best_state: Dict[str, torch.Tensor] | None = None
    history = []
    start = time.perf_counter()
    n_rows = int(train_targets.shape[0])
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        loss_sum = 0.0
        seen = 0
        epoch_start = time.perf_counter()
        order = torch.randperm(n_rows, device=args.device)
        for step, start_idx in enumerate(range(0, n_rows, int(args.batch_size)), start=1):
            batch_idx = order[start_idx : min(start_idx + int(args.batch_size), n_rows)]
            logits = model(train_features.index_select(0, batch_idx))
            targets = train_targets.index_select(0, batch_idx)
            loss = F.cross_entropy(logits, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_size_actual = int(targets.shape[0])
            loss_sum += float(loss.item()) * batch_size_actual
            seen += batch_size_actual
            if step % max(1, int(args.log_every)) == 0:
                elapsed = time.perf_counter() - epoch_start
                print(
                    f"[dense-cache:gpu] epoch={epoch} step={step} "
                    f"loss={loss_sum / max(seen, 1):.4f} ips={seen / max(elapsed, 1e-6):.2f}",
                    flush=True,
                )
        scheduler.step()
        val_metrics = evaluate_tensor_features(model, val_features, val_targets, int(args.batch_size))
        train_metrics = (
            evaluate_tensor_features(model, train_features, train_targets, int(args.batch_size))
            if args.eval_train
            else {"loss": loss_sum / max(seen, 1), "top1": None, "top5": None, "n": seen}
        )
        epoch_payload = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "lr": float(scheduler.get_last_lr()[0]),
            "train_epoch_sec": time.perf_counter() - epoch_start,
        }
        history.append(epoch_payload)
        print(json.dumps({"dense_epoch": epoch_payload}, indent=2), flush=True)
        if best is None or float(val_metrics["loss"]) < float(best["val"]["loss"]):
            best = epoch_payload
            best_state = {
                "weight": model.weight.detach().cpu().clone(),
                "bias": model.bias.detach().cpu().clone(),
            }
            torch.save({"weight": best_state["weight"], "bias": best_state["bias"], "epoch": epoch}, args.output_dir / "final_layer_dense.pt")

    assert best is not None and best_state is not None
    return {
        "train_features": str(args.train_features),
        "train_targets": str(args.train_targets),
        "val_features": str(args.val_features),
        "val_targets": str(args.val_targets),
        "output_dir": str(args.output_dir),
        "n_features": n_features,
        "n_classes": n_classes,
        "config": {
            "batch_size": int(args.batch_size),
            "workers": int(args.workers),
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "scheduler_gamma": float(args.scheduler_gamma),
            "seed": int(args.seed),
            "eval_train": bool(args.eval_train),
            "cache_device": str(args.cache_device),
            "cache_chunk_rows": int(args.cache_chunk_rows),
        },
        "normalization": stats_summary,
        "cache": cache_summaries,
        "best": best,
        "history": history,
        "nnz": int((best_state["weight"].abs() > 1e-5).sum().item()),
        "total": int(best_state["weight"].numel()),
        "elapsed_sec": time.perf_counter() - start,
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_targets = np.load(args.train_targets, mmap_mode="r")
    n_classes = int(args.num_classes) if int(args.num_classes) > 0 else int(np.max(train_targets)) + 1
    mean, std, stats_summary = compute_stats(args.train_features, int(args.stats_chunk_rows))

    torch.save({"mean": torch.from_numpy(mean), "std": torch.from_numpy(std), "normalization": stats_summary}, args.output_dir / "final_layer_normalization.pt")

    if args.cache_device == "cuda":
        train_features, train_targets_t, train_cache = load_normalized_feature_tensor(
            args.train_features,
            args.train_targets,
            mean,
            std,
            device=args.device,
            chunk_rows=int(args.cache_chunk_rows),
        )
        val_features, val_targets_t, val_cache = load_normalized_feature_tensor(
            args.val_features,
            args.val_targets,
            mean,
            std,
            device=args.device,
            chunk_rows=int(args.cache_chunk_rows),
        )
        payload = train_dense_cached_on_device(
            args,
            train_features,
            train_targets_t,
            val_features,
            val_targets_t,
            n_classes,
            int(train_features.shape[1]),
            {"train": train_cache, "val": val_cache},
            stats_summary,
        )
        (args.output_dir / "final_layer_dense_summary.json").write_text(json.dumps(payload, indent=2))
        print(json.dumps(payload, indent=2), flush=True)
        return

    train_dataset = CachedFeatureDataset(args.train_features, args.train_targets, mean=mean, std=std)
    val_dataset = CachedFeatureDataset(args.val_features, args.val_targets, mean=mean, std=std)
    loader_kwargs = {
        "batch_size": int(args.batch_size),
        "num_workers": int(args.workers),
        "pin_memory": True,
        "drop_last": False,
    }
    if int(args.workers) > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    train_eval_loader = DataLoader(train_dataset, shuffle=False, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    n_features = int(train_dataset.features.shape[1])
    model = nn.Linear(n_features, n_classes).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=float(args.scheduler_gamma))

    best: Dict[str, Any] | None = None
    best_state: Dict[str, torch.Tensor] | None = None
    history = []
    start = time.perf_counter()
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        loss_sum = 0.0
        seen = 0
        epoch_start = time.perf_counter()
        for step, (features, targets) in enumerate(train_loader, start=1):
            features = features.to(args.device, non_blocking=True)
            targets = targets.to(args.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            optimizer.step()
            batch_size = int(targets.shape[0])
            loss_sum += float(loss.item()) * batch_size
            seen += batch_size
            if step % max(1, int(args.log_every)) == 0:
                elapsed = time.perf_counter() - epoch_start
                print(f"[dense-cache] epoch={epoch} step={step}/{len(train_loader)} loss={loss_sum / max(seen, 1):.4f} ips={seen / max(elapsed, 1e-6):.2f}", flush=True)
        scheduler.step()
        val_metrics = evaluate(model, val_loader, args.device)
        train_metrics = evaluate(model, train_eval_loader, args.device) if args.eval_train else {"loss": loss_sum / max(seen, 1), "top1": None, "top5": None, "n": seen}
        epoch_payload = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "lr": float(scheduler.get_last_lr()[0]),
            "train_epoch_sec": time.perf_counter() - epoch_start,
        }
        history.append(epoch_payload)
        print(json.dumps({"dense_epoch": epoch_payload}, indent=2), flush=True)
        if best is None or float(val_metrics["loss"]) < float(best["val"]["loss"]):
            best = epoch_payload
            best_state = {
                "weight": model.weight.detach().cpu().clone(),
                "bias": model.bias.detach().cpu().clone(),
            }
            torch.save({"weight": best_state["weight"], "bias": best_state["bias"], "epoch": epoch}, args.output_dir / "final_layer_dense.pt")

    assert best is not None and best_state is not None
    payload = {
        "train_features": str(args.train_features),
        "train_targets": str(args.train_targets),
        "val_features": str(args.val_features),
        "val_targets": str(args.val_targets),
        "output_dir": str(args.output_dir),
        "n_features": n_features,
        "n_classes": n_classes,
        "config": {
            "batch_size": int(args.batch_size),
            "workers": int(args.workers),
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "scheduler_gamma": float(args.scheduler_gamma),
            "seed": int(args.seed),
            "eval_train": bool(args.eval_train),
            "cache_device": str(args.cache_device),
            "cache_chunk_rows": int(args.cache_chunk_rows),
        },
        "normalization": stats_summary,
        "best": best,
        "history": history,
        "nnz": int((best_state["weight"].abs() > 1e-5).sum().item()),
        "total": int(best_state["weight"].numel()),
        "elapsed_sec": time.perf_counter() - start,
    }
    (args.output_dir / "final_layer_dense_summary.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
