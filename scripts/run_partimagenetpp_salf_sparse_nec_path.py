#!/usr/bin/env python3
"""Run path-trained sparse NEC heads for a PartImageNet++ SALF-CBM checkpoint."""

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
    parser.add_argument(
        "--path_max_nec",
        type=int,
        default=0,
        help=(
            "Optional raw-path NEC ceiling. Set to the total concept count to disable "
            "the usual stop at the largest reported NEC."
        ),
    )
    parser.add_argument("--cbl_batch_size", type=int, default=256)
    parser.add_argument("--saga_batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--cache_features_device", choices=["none", "cuda"], default="none")
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--copy_artifacts", action="store_true")
    return parser.parse_args()


def validate_file(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def prepare_output_dir(source_run_dir: Path, output_dir: Path, *, copy_artifacts: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    required = ("args.txt", "concepts.txt", "concept_layer.pt", "proj_mean.pt", "proj_std.pt")
    for name in required:
        src = source_run_dir / name
        if not src.is_file():
            raise FileNotFoundError(f"missing required SALF artifact: {src}")
        dst = output_dir / name
        if dst.exists() or dst.is_symlink():
            continue
        if copy_artifacts:
            shutil.copy2(src, dst)
        else:
            dst.symlink_to(src)
    (output_dir / "source_run_dir.txt").write_text(str(source_run_dir) + "\n", encoding="utf-8")


def read_metrics_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    import pandas as pd

    rows = []
    for row in pd.read_csv(path).to_dict("records"):
        rows.append(
            {
                "nec_raw": float(row.get("NEC", float("nan"))),
                "accuracy_raw": float(row.get("Accuracy", float("nan"))),
            }
        )
    return rows


def summarize_saved_heads(output_dir: Path, accs: list[float]) -> list[dict[str, Any]]:
    import torch

    rows = []
    for nec, acc in zip(DEFAULT_MEASURE_LEVEL, accs):
        weight_path = output_dir / f"W_g@NEC={int(nec)}.pt"
        bias_path = output_dir / f"b_g@NEC={int(nec)}.pt"
        if not weight_path.is_file() or not bias_path.is_file():
            continue
        weight = torch.load(weight_path, map_location="cpu")
        nnz = int((weight.abs() > 1e-5).sum().item())
        total = int(weight.numel())
        rows.append(
            {
                "nec": int(nec),
                "top1": float(acc),
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
    train_manifest = validate_file(args.train_manifest, "train manifest")
    val_manifest = validate_file(args.val_manifest, "val manifest")
    if not source_run_dir.is_dir():
        raise FileNotFoundError(f"source run directory not found: {source_run_dir}")

    prepare_output_dir(source_run_dir, output_dir, copy_artifacts=bool(args.copy_artifacts))
    os.environ["PARTIMAGENETPP_TRAIN_MANIFEST"] = str(train_manifest)
    os.environ["PARTIMAGENETPP_VAL_MANIFEST"] = str(val_manifest)

    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    start_time = time.perf_counter()
    accs, feature_set, run_args = train_sparse_nec_from_checkpoint(
        str(output_dir),
        "salf_cbm",
        lam_max=float(args.lam_max),
        n_iters=int(args.n_iters),
        max_glm_steps=int(args.max_glm_steps),
        path_max_nec=(int(args.path_max_nec) if int(args.path_max_nec) > 0 else None),
        cbl_batch_size=int(args.cbl_batch_size),
        saga_batch_size=int(args.saga_batch_size),
        num_workers=int(args.num_workers),
        max_images=(int(args.max_images) if int(args.max_images) > 0 else None),
        cache_features_device=str(args.cache_features_device),
    )
    elapsed = time.perf_counter() - start_time

    summary = {
        "model_name": "salf_cbm",
        "dataset": "partimagenetpp",
        "source_run_dir": str(source_run_dir),
        "output_dir": str(output_dir),
        "train_manifest": str(train_manifest),
        "val_manifest": str(val_manifest),
        "started_utc": started,
        "elapsed_seconds": float(elapsed),
        "backbone": getattr(run_args, "backbone", None),
        "lf_clip_name": getattr(run_args, "lf_clip_name", None),
        "concepts": int(len(feature_set.concepts)),
        "classes": int(len(feature_set.classes)),
        "train_images": int(feature_set.train_labels.numel()),
        "val_images": int(feature_set.test_labels.numel()),
        "lam_max": float(args.lam_max),
        "n_iters": int(args.n_iters),
        "max_glm_steps": int(args.max_glm_steps),
        "path_max_nec": int(args.path_max_nec),
        "cbl_batch_size": int(args.cbl_batch_size),
        "saga_batch_size": int(args.saga_batch_size),
        "num_workers": int(args.num_workers),
        "cache_features_device": str(args.cache_features_device),
        "max_images": int(args.max_images),
        "metrics": summarize_saved_heads(output_dir, accs),
        "path_metrics": read_metrics_csv(output_dir / "metrics.csv"),
    }
    args.result_json.parent.mkdir(parents=True, exist_ok=True)
    args.result_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
