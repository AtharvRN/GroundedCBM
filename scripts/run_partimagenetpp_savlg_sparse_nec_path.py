#!/usr/bin/env python3
"""Train and evaluate PartImageNet++ SG-CBM sparse heads at fixed NEC levels."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluations.sparse_utils import DEFAULT_MEASURE_LEVEL, train_sparse_nec_from_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_run_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--train_manifest", required=True, type=Path)
    parser.add_argument("--val_manifest", required=True, type=Path)
    parser.add_argument("--result_json", required=True, type=Path)
    parser.add_argument("--lam_max", type=float, default=0.1)
    parser.add_argument("--n_iters", type=int, default=100)
    parser.add_argument("--max_glm_steps", type=int, default=150)
    parser.add_argument("--cbl_batch_size", type=int, default=512)
    parser.add_argument("--saga_batch_size", type=int, default=4096)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--cache_features_device", choices=["none", "cuda"], default="none")
    parser.add_argument("--copy_artifacts", action="store_true")
    return parser.parse_args()


def require_file(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def prepare_output_dir(source_run_dir: Path, output_dir: Path, copy_artifacts: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("args.txt", "concepts.txt", "concept_layer.pt", "proj_mean.pt", "proj_std.pt"):
        source = require_file(source_run_dir / name, f"SG-CBM artifact {name}")
        destination = output_dir / name
        if destination.exists() or destination.is_symlink():
            continue
        if copy_artifacts:
            shutil.copy2(source, destination)
        else:
            destination.symlink_to(source)
    (output_dir / "source_run_dir.txt").write_text(f"{source_run_dir}\n", encoding="utf-8")


def summarize_heads(output_dir: Path, accuracies: list[float]) -> list[dict[str, Any]]:
    import torch

    rows = []
    for nec, accuracy in zip(DEFAULT_MEASURE_LEVEL, accuracies):
        weight_path = output_dir / f"W_g@NEC={int(nec)}.pt"
        if not weight_path.is_file():
            continue
        weight = torch.load(weight_path, map_location="cpu")
        nnz = int((weight.abs() > 1e-5).sum().item())
        total = int(weight.numel())
        rows.append(
            {
                "nec": int(nec),
                "top1": float(accuracy),
                "nnz": nnz,
                "total": total,
                "effective_nec": float(nnz / max(int(weight.shape[0]), 1)),
                "weight_sparsity": float(1.0 - nnz / max(total, 1)),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    source_run_dir = args.source_run_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not source_run_dir.is_dir():
        raise FileNotFoundError(f"source SG-CBM run directory not found: {source_run_dir}")
    train_manifest = require_file(args.train_manifest, "train manifest")
    val_manifest = require_file(args.val_manifest, "val manifest")
    prepare_output_dir(source_run_dir, output_dir, bool(args.copy_artifacts))

    os.environ["PARTIMAGENETPP_TRAIN_MANIFEST"] = str(train_manifest)
    os.environ["PARTIMAGENETPP_VAL_MANIFEST"] = str(val_manifest)
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    started_at = time.perf_counter()
    accuracies, features, run_args = train_sparse_nec_from_checkpoint(
        str(output_dir),
        "savlg_cbm",
        lam_max=float(args.lam_max),
        n_iters=int(args.n_iters),
        max_glm_steps=int(args.max_glm_steps),
        cbl_batch_size=int(args.cbl_batch_size),
        saga_batch_size=int(args.saga_batch_size),
        num_workers=int(args.num_workers),
        cache_features_device=str(args.cache_features_device),
    )
    summary = {
        "model_name": "savlg_cbm",
        "dataset": "partimagenetpp",
        "source_run_dir": str(source_run_dir),
        "output_dir": str(output_dir),
        "train_manifest": str(train_manifest),
        "val_manifest": str(val_manifest),
        "started_utc": started,
        "elapsed_seconds": float(time.perf_counter() - started_at),
        "backbone": getattr(run_args, "backbone", None),
        "backbone_checkpoint": getattr(run_args, "backbone_checkpoint", None),
        "concepts": len(features.concepts),
        "classes": len(features.classes),
        "train_images": int(features.train_labels.numel()),
        "val_images": int(features.val_labels.numel()) if features.val_labels is not None else 0,
        "test_images": int(features.test_labels.numel()),
        "cache_features_device": str(args.cache_features_device),
        "metrics": summarize_heads(output_dir, accuracies),
    }
    args.result_json.parent.mkdir(parents=True, exist_ok=True)
    args.result_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
