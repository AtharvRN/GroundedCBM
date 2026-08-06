#!/usr/bin/env python3
"""Extract PartImageNet++ SALF concept features for the fast GLM path runner."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluations.sparse_utils import build_nec_feature_set


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_run_dir", required=True, type=Path)
    parser.add_argument("--artifact_dir", required=True, type=Path)
    parser.add_argument("--train_manifest", required=True, type=Path)
    parser.add_argument("--val_manifest", required=True, type=Path)
    parser.add_argument("--cbl_batch_size", type=int, default=256)
    parser.add_argument("--saga_batch_size", type=int, default=4096)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--feature_dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--copy_artifacts", action="store_true")
    return parser.parse_args()


def require_file(path: Path, label: str) -> Path:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def link_or_copy(src: Path, dst: Path, *, copy: bool) -> None:
    if dst.exists() or dst.is_symlink():
        return
    if copy:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src)


def prepare_artifact_dir(source_run_dir: Path, artifact_dir: Path, *, copy: bool) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    for name in ("args.txt", "concepts.txt", "concept_layer.pt", "proj_mean.pt", "proj_std.pt"):
        src = source_run_dir / name
        if not src.is_file():
            raise FileNotFoundError(f"missing required SALF artifact: {src}")
        link_or_copy(src, artifact_dir / name, copy=copy)
    (artifact_dir / "source_run_dir.txt").write_text(str(source_run_dir) + "\n", encoding="utf-8")


def save_array(path: Path, tensor: torch.Tensor, *, dtype: str | None = None) -> None:
    array = tensor.detach().cpu().numpy()
    if dtype is not None:
        array = array.astype(dtype, copy=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def main() -> None:
    args = parse_args()
    source_run_dir = args.source_run_dir.resolve()
    artifact_dir = args.artifact_dir.resolve()
    train_manifest = require_file(args.train_manifest, "train manifest")
    val_manifest = require_file(args.val_manifest, "val manifest")
    if not source_run_dir.is_dir():
        raise FileNotFoundError(f"source run directory not found: {source_run_dir}")

    prepare_artifact_dir(source_run_dir, artifact_dir, copy=bool(args.copy_artifacts))
    os.environ["PARTIMAGENETPP_TRAIN_MANIFEST"] = str(train_manifest)
    os.environ["PARTIMAGENETPP_VAL_MANIFEST"] = str(val_manifest)

    feature_set, run_args = build_nec_feature_set(
        str(artifact_dir),
        "salf_cbm",
        cbl_batch_size=int(args.cbl_batch_size),
        saga_batch_size=int(args.saga_batch_size),
        num_workers=int(args.num_workers),
        max_images=(int(args.max_images) if int(args.max_images) > 0 else None),
    )

    feature_dir = artifact_dir / "features"
    save_array(feature_dir / "train_features.npy", feature_set.train_features, dtype=args.feature_dtype)
    save_array(feature_dir / "train_targets.npy", feature_set.train_labels.long())
    save_array(feature_dir / "val_features.npy", feature_set.test_features, dtype=args.feature_dtype)
    save_array(feature_dir / "val_targets.npy", feature_set.test_labels.long())

    n_features = int(feature_set.train_features.shape[1])
    torch.save(
        {
            "mean": torch.zeros(n_features, dtype=torch.float32),
            "std": torch.ones(n_features, dtype=torch.float32),
        },
        artifact_dir / "final_layer_normalization.pt",
    )

    summary = {
        "source_run_dir": str(source_run_dir),
        "artifact_dir": str(artifact_dir),
        "feature_dir": str(feature_dir),
        "train_manifest": str(train_manifest),
        "val_manifest": str(val_manifest),
        "backbone": getattr(run_args, "backbone", None),
        "lf_clip_name": getattr(run_args, "lf_clip_name", None),
        "concepts": int(len(feature_set.concepts)),
        "classes": int(len(feature_set.classes)),
        "train_shape": list(feature_set.train_features.shape),
        "val_shape": list(feature_set.test_features.shape),
        "feature_dtype": args.feature_dtype,
    }
    (artifact_dir / "feature_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
