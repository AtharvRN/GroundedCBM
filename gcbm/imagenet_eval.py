from __future__ import annotations

import argparse
import json
import re
from dataclasses import fields
from pathlib import Path

from gcbm.imagenet_config import Config


VAL_RE = re.compile(r"ILSVRC2012_val_(\d{8})\.JPEG$")


def resolve_source_run_dir(artifact_dir: Path) -> Path:
    source_run_file = artifact_dir / "source_run_dir.txt"
    if source_run_file.exists():
        source_run_dir = Path(source_run_file.read_text().strip()).resolve()
        if source_run_dir.is_dir():
            return source_run_dir
    return artifact_dir


def load_run_config(config_dir: Path, args: argparse.Namespace) -> Config:
    payload = json.loads((config_dir / "config.json").read_text())
    payload.setdefault("feature_storage_dtype", "fp16")
    payload.setdefault("saga_table_device", "cpu")
    payload.setdefault("dense_lr", 1e-3)
    payload.setdefault("dense_n_iters", 20)
    payload.setdefault("train_random_transforms", True)
    payload.setdefault("learn_spatial_residual_scale", False)
    payload["device"] = args.device
    payload["batch_size"] = int(args.batch_size)
    payload["workers"] = int(args.workers)
    payload["prefetch_factor"] = int(args.prefetch_factor)
    payload["persistent_workers"] = bool(args.persistent_workers)
    payload["pin_memory"] = bool(args.pin_memory)
    payload["skip_final_layer"] = True
    payload["print_config"] = False
    valid_fields = {field.name for field in fields(Config)}
    return Config(**{key: value for key, value in payload.items() if key in valid_fields})
