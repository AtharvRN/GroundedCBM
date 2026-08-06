#!/usr/bin/env python3
"""Train a ResNet-50 from scratch on PartImageNet++ manifest splits."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image, ImageFile, UnidentifiedImageError
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

try:
    from torchvision import transforms
    from torchvision.models import resnet50
except RuntimeError as exc:
    if "torchvision::nms" not in str(exc):
        raise
    _tv_lib = torch.library.Library("torchvision", "DEF")
    _tv_lib.define("nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor")
    from torchvision import transforms
from torchvision.models import resnet50


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_manifest", required=True, type=Path)
    parser.add_argument("--val_manifest", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=512, help="Per-process batch size.")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--lr", type=float, default=0.2)
    parser.add_argument("--lr_schedule", choices=("cosine", "constant"), default="cosine")
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--randaugment_num_ops", type=int, default=0)
    parser.add_argument("--randaugment_magnitude", type=int, default=9)
    parser.add_argument("--random_erasing_prob", type=float, default=0.0)
    parser.add_argument("--mixup_alpha", type=float, default=0.0)
    parser.add_argument("--cutmix_alpha", type=float, default=0.0)
    parser.add_argument("--mix_prob", type=float, default=1.0)
    parser.add_argument("--cutmix_switch_prob", type=float, default=0.5)
    parser.add_argument("--ema_decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--channels_last", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", default="", type=Path)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--log_every", type=int, default=50)
    return parser.parse_args()


def distributed_info() -> tuple[bool, int, int, int]:
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return False, 0, 1, 0
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return True, rank, world_size, local_rank


def is_main(rank: int) -> bool:
    return rank == 0


def safe_pil_loader(path: str) -> Image.Image:
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    try:
        with Image.open(path) as image:
            return image.convert("RGB")
    except (UnidentifiedImageError, OSError, FileNotFoundError):
        return Image.new("RGB", (224, 224), color=(0, 0, 0))


class PartImageNetPPManifestDataset(torch.utils.data.Dataset):
    def __init__(self, manifest: Path, transform, class_to_idx: Dict[str, int] | None = None):
        self.manifest = manifest.resolve()
        self.transform = transform
        self.rows: List[Dict[str, Any]] = []
        with self.manifest.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    self.rows.append(json.loads(line))
        if not self.rows:
            raise RuntimeError(f"empty manifest: {self.manifest}")

        class_meta: Dict[str, str] = {}
        for row in self.rows:
            wnid = str(row["wnid"])
            class_meta.setdefault(wnid, str(row.get("object_name") or wnid))
        if class_to_idx is None:
            self.wnids = sorted(class_meta)
            self.class_to_idx = {wnid: idx for idx, wnid in enumerate(self.wnids)}
        else:
            self.class_to_idx = dict(class_to_idx)
            self.wnids = [wnid for wnid, _ in sorted(self.class_to_idx.items(), key=lambda kv: kv[1])]
        missing = sorted({str(row["wnid"]) for row in self.rows} - set(self.class_to_idx))
        if missing:
            raise ValueError(f"manifest contains wnids absent from class_to_idx: {missing[:10]}")
        self.classes = [class_meta.get(wnid, wnid) for wnid in self.wnids]
        self.targets = [self.class_to_idx[str(row["wnid"])] for row in self.rows]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        image = safe_pil_loader(str(row["image"]))
        return self.transform(image), self.targets[idx]


def build_transforms(args: argparse.Namespace) -> tuple[Any, Any]:
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    train_ops: List[Any] = [
        transforms.RandomResizedCrop(224, scale=(0.08, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
    ]
    if int(args.randaugment_num_ops) > 0:
        train_ops.append(
            transforms.RandAugment(
                num_ops=int(args.randaugment_num_ops),
                magnitude=int(args.randaugment_magnitude),
            )
        )
    train_ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    if float(args.random_erasing_prob) > 0.0:
        train_ops.append(transforms.RandomErasing(p=float(args.random_erasing_prob)))
    train_tf = transforms.Compose(train_ops)
    val_tf = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return train_tf, val_tf


def seed_everything(seed: int, rank: int) -> None:
    seed = int(seed) + int(rank)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_lr(base_lr: float, min_lr: float, epoch_float: float, epochs: int, warmup_epochs: int) -> float:
    if warmup_epochs > 0 and epoch_float < warmup_epochs:
        return base_lr * float(epoch_float + 1e-8) / float(warmup_epochs)
    progress = (epoch_float - warmup_epochs) / max(1.0, float(epochs - warmup_epochs))
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * min(1.0, progress)))


def compute_lr(args: argparse.Namespace, epoch_float: float) -> float:
    if args.lr_schedule == "constant":
        return float(args.lr)
    return cosine_lr(float(args.lr), float(args.min_lr), epoch_float, int(args.epochs), int(args.warmup_epochs))


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def mix_batch(images: torch.Tensor, targets: torch.Tensor, args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor, float]:
    mixup_alpha = float(args.mixup_alpha)
    cutmix_alpha = float(args.cutmix_alpha)
    if (mixup_alpha <= 0.0 and cutmix_alpha <= 0.0) or float(torch.rand(()).item()) > float(args.mix_prob):
        return targets, targets, 1.0

    perm = torch.randperm(images.shape[0], device=images.device)
    use_cutmix = cutmix_alpha > 0.0 and (mixup_alpha <= 0.0 or float(torch.rand(()).item()) < float(args.cutmix_switch_prob))
    alpha = cutmix_alpha if use_cutmix else mixup_alpha
    lam = float(torch.distributions.Beta(alpha, alpha).sample().item())
    if not use_cutmix:
        paired_images = images[perm]
        images.mul_(lam).add_(paired_images, alpha=1.0 - lam)
        return targets, targets[perm], lam

    height, width = images.shape[-2:]
    cut_ratio = math.sqrt(1.0 - lam)
    cut_h = max(1, int(height * cut_ratio))
    cut_w = max(1, int(width * cut_ratio))
    center_y = int(torch.randint(height, ()).item())
    center_x = int(torch.randint(width, ()).item())
    y1 = max(0, center_y - cut_h // 2)
    y2 = min(height, center_y + cut_h // 2)
    x1 = max(0, center_x - cut_w // 2)
    x2 = min(width, center_x + cut_w // 2)
    images[:, :, y1:y2, x1:x2] = images[perm, :, y1:y2, x1:x2]
    lam = 1.0 - float((y2 - y1) * (x2 - x1)) / float(height * width)
    return targets, targets[perm], lam


@torch.no_grad()
def update_ema(ema_model: nn.Module, model: nn.Module, decay: float, updates: int) -> None:
    source = model.module if hasattr(model, "module") else model
    decay = min(float(decay), (1.0 + float(updates)) / (10.0 + float(updates)))
    for ema_param, param in zip(ema_model.parameters(), source.parameters()):
        ema_param.mul_(decay).add_(param, alpha=1.0 - decay)
    for ema_buffer, buffer in zip(ema_model.buffers(), source.buffers()):
        if torch.is_floating_point(ema_buffer):
            ema_buffer.mul_(decay).add_(buffer, alpha=1.0 - decay)
        else:
            ema_buffer.copy_(buffer)


def reduce_sum(value: torch.Tensor, distributed: bool) -> torch.Tensor:
    if distributed:
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


def accuracy_counts(logits: torch.Tensor, targets: torch.Tensor) -> tuple[int, int, int]:
    with torch.no_grad():
        top1 = logits.argmax(dim=1)
        top5 = logits.topk(k=min(5, logits.shape[1]), dim=1).indices
        c1 = int(top1.eq(targets).sum().item())
        c5 = int(top5.eq(targets[:, None]).any(dim=1).sum().item())
        n = int(targets.numel())
    return c1, c5, n


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, distributed: bool) -> Dict[str, float]:
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    totals = torch.zeros(4, dtype=torch.float64, device=device)
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True).long()
            logits = model(images)
            loss = loss_fn(logits, targets)
            c1, c5, n = accuracy_counts(logits, targets)
            totals += torch.tensor([float(loss.item() * n), float(c1), float(c5), float(n)], device=device)
    totals = reduce_sum(totals, distributed)
    n = max(float(totals[3].item()), 1.0)
    return {
        "loss": float(totals[0].item() / n),
        "top1": float(totals[1].item() / n),
        "top5": float(totals[2].item() / n),
        "n": int(totals[3].item()),
    }


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    best_top1: float,
    args: argparse.Namespace,
    class_to_idx: Dict[str, int],
    ema_model: nn.Module | None = None,
) -> None:
    raw_model = model.module if hasattr(model, "module") else model
    evaluation_model = ema_model if ema_model is not None else raw_model
    payload = {
        "epoch": int(epoch),
        "model": evaluation_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "best_top1": float(best_top1),
        "args": vars(args),
        "class_to_idx": class_to_idx,
    }
    if ema_model is not None:
        payload["base_model"] = raw_model.state_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
) -> tuple[int, float]:
    payload = torch.load(path, map_location=device)
    raw_model = model.module if hasattr(model, "module") else model
    raw_model.load_state_dict(payload.get("base_model", payload["model"]))
    optimizer.load_state_dict(payload["optimizer"])
    scaler.load_state_dict(payload.get("scaler", {}))
    return int(payload["epoch"]) + 1, float(payload.get("best_top1", 0.0))


def main() -> None:
    args = parse_args()
    distributed, rank, world_size, local_rank = distributed_info()
    seed_everything(args.seed, rank)

    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        device = torch.device(f"cuda:{local_rank}" if distributed else "cuda")
    else:
        device = torch.device("cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_tf, val_tf = build_transforms(args)
    train_dataset = PartImageNetPPManifestDataset(args.train_manifest, train_tf)
    val_dataset = PartImageNetPPManifestDataset(args.val_manifest, val_tf, class_to_idx=train_dataset.class_to_idx)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if distributed else None

    loader_kwargs = {
        "num_workers": int(args.workers),
        "pin_memory": device.type == "cuda",
        "persistent_workers": int(args.workers) > 0,
    }
    if int(args.workers) > 0:
        loader_kwargs["prefetch_factor"] = 2
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(args.batch_size),
        shuffle=train_sampler is None,
        sampler=train_sampler,
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        sampler=val_sampler,
        drop_last=False,
        **loader_kwargs,
    )

    model = resnet50(weights=None, num_classes=len(train_dataset.class_to_idx)).to(device)
    if bool(args.channels_last):
        model = model.to(memory_format=torch.channels_last)
    if distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    ema_model = None
    if float(args.ema_decay) > 0.0:
        raw_model = model.module if hasattr(model, "module") else model
        ema_model = copy.deepcopy(raw_model).eval()
        for parameter in ema_model.parameters():
            parameter.requires_grad_(False)

    loss_fn = nn.CrossEntropyLoss(label_smoothing=float(args.label_smoothing))
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(args.lr),
        momentum=float(args.momentum),
        weight_decay=float(args.weight_decay),
        nesterov=True,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp) and device.type == "cuda")

    start_epoch = 0
    best_top1 = 0.0
    if args.resume and args.resume.is_file():
        start_epoch, best_top1 = load_checkpoint(args.resume, model=model, optimizer=optimizer, scaler=scaler, device=device)
    ema_updates = start_epoch * max(len(train_loader), 1)

    if is_main(rank):
        config = {
            "args": jsonable(vars(args)),
            "distributed": distributed,
            "world_size": world_size,
            "train_len": len(train_dataset),
            "val_len": len(val_dataset),
            "classes": len(train_dataset.class_to_idx),
            "model": "torchvision.resnet50",
            "weights": None,
        }
        (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        (args.output_dir / "class_to_idx.json").write_text(json.dumps(train_dataset.class_to_idx, indent=2) + "\n", encoding="utf-8")

    metrics_path = args.output_dir / "metrics.jsonl"
    for epoch in range(start_epoch, int(args.epochs)):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        epoch_start = time.perf_counter()
        totals = torch.zeros(4, dtype=torch.float64, device=device)
        iterable: Iterable[Any] = train_loader
        if is_main(rank):
            iterable = tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.epochs}", dynamic_ncols=True)
        for step, (images, targets) in enumerate(iterable):
            step_frac = epoch + step / max(len(train_loader), 1)
            lr = compute_lr(args, step_frac)
            set_lr(optimizer, lr)
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True).long()
            if bool(args.channels_last):
                images = images.contiguous(memory_format=torch.channels_last)
            targets_a, targets_b, mix_lambda = mix_batch(images, targets, args)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=bool(args.amp) and device.type == "cuda"):
                logits = model(images)
                loss = mix_lambda * loss_fn(logits, targets_a) + (1.0 - mix_lambda) * loss_fn(logits, targets_b)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if ema_model is not None:
                ema_updates += 1
                update_ema(ema_model, model, float(args.ema_decay), ema_updates)

            c1, c5, n = accuracy_counts(logits.detach(), targets)
            totals += torch.tensor([float(loss.item() * n), float(c1), float(c5), float(n)], device=device)
            if is_main(rank) and int(args.log_every) > 0 and (step + 1) % int(args.log_every) == 0:
                seen = int(totals[3].item())
                elapsed = time.perf_counter() - epoch_start
                print(
                    f"[train] epoch={epoch + 1} step={step + 1}/{len(train_loader)} "
                    f"lr={lr:.6g} loss={totals[0].item() / max(seen, 1):.4f} "
                    f"top1={totals[1].item() / max(seen, 1):.4f} "
                    f"top5={totals[2].item() / max(seen, 1):.4f} ips={seen * world_size / max(elapsed, 1e-6):.2f}",
                    flush=True,
                )

        totals = reduce_sum(totals, distributed)
        train_n = max(float(totals[3].item()), 1.0)
        row: Dict[str, Any] = {
            "epoch": epoch + 1,
            "train_loss": float(totals[0].item() / train_n),
            "train_top1": float(totals[1].item() / train_n),
            "train_top5": float(totals[2].item() / train_n),
            "train_n": int(totals[3].item()),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "elapsed_sec": float(time.perf_counter() - epoch_start),
        }
        if (epoch + 1) % int(args.eval_every) == 0:
            val_metrics = evaluate(ema_model if ema_model is not None else model, val_loader, device, distributed)
            row.update({f"val_{key}": value for key, value in val_metrics.items()})
            if float(val_metrics["top1"]) > best_top1:
                best_top1 = float(val_metrics["top1"])
                if is_main(rank):
                    save_checkpoint(
                        args.output_dir / "checkpoint_best.pt",
                        model=model,
                        optimizer=optimizer,
                        scaler=scaler,
                        epoch=epoch,
                        best_top1=best_top1,
                        args=args,
                        class_to_idx=train_dataset.class_to_idx,
                        ema_model=ema_model,
                    )
        row["best_val_top1"] = best_top1
        if is_main(rank):
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
            save_checkpoint(
                args.output_dir / "checkpoint_latest.pt",
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                best_top1=best_top1,
                args=args,
                class_to_idx=train_dataset.class_to_idx,
                ema_model=ema_model,
            )
            if int(args.save_every) > 0 and (epoch + 1) % int(args.save_every) == 0:
                save_checkpoint(
                    args.output_dir / f"checkpoint_epoch_{epoch + 1:03d}.pt",
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    epoch=epoch,
                    best_top1=best_top1,
                    args=args,
                    class_to_idx=train_dataset.class_to_idx,
                    ema_model=ema_model,
                )

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
