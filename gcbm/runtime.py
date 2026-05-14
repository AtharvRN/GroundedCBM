from __future__ import annotations

import random
from typing import Dict, Optional

import numpy as np
import torch

from gcbm.imagenet_config import Config


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


