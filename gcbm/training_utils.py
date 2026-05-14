from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn

from gcbm.imagenet_config import Config


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
