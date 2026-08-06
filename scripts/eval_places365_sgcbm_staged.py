#!/usr/bin/env python3
"""Train/evaluate a final Places365 classifier for a staged SG-CBM concept head."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gcbm.imagenet_config import Config  # noqa: E402
from gcbm.imagenet_data import build_loader  # noqa: E402
from gcbm.imagenet_final_layers import (  # noqa: E402
    compute_feature_stats_memmap,
    extract_concept_features_to_memmap,
    train_dense_final_layer,
    train_sparse_final_layer,
)
from gcbm.imagenet_models import build_model  # noqa: E402
from gcbm.runtime import configure_runtime  # noqa: E402
from gcbm.train_imagenet import build_datasets, infer_num_classes  # noqa: E402


def validate_manifest_paths(manifest: str, *, expected_rows: int = 0, label: str = "manifest") -> None:
    path = Path(manifest)
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    missing: list[str] = []
    rows = 0
    with path.open("r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows += 1
            payload = json.loads(line)
            image_path = Path(payload["path"])
            if not image_path.is_file():
                missing.append(str(image_path))
                if len(missing) >= 10:
                    break
    if missing:
        raise FileNotFoundError(f"{label} contains missing image paths; first={missing[0]}")
    if expected_rows and rows != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} rows in {label}, found {rows}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate staged Places365 SG-CBM on the actual val split.")
    parser.add_argument("--run_dir", required=True, type=Path, help="Directory containing config.json and concept_head_best.pt.")
    parser.add_argument("--train_manifest", default="", help="Training manifest used for final-layer training.")
    parser.add_argument("--val_manifest", required=True, help="Actual Places365 val manifest.")
    parser.add_argument("--val_root", required=True, help="Non-empty val root; manifest paths may be absolute.")
    parser.add_argument("--output_dir", default="", type=Path)
    parser.add_argument("--final_layer_type", choices=["dense", "sparse"], default="dense")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--feature_batch_size", type=int, default=512)
    parser.add_argument("--feature_workers", type=int, default=4)
    parser.add_argument("--feature_prefetch_factor", type=int, default=2)
    parser.add_argument("--saga_batch_size", type=int, default=4096)
    parser.add_argument("--saga_workers", type=int, default=0)
    parser.add_argument("--saga_lam", type=float, default=5e-4)
    parser.add_argument("--saga_n_iters", type=int, default=200)
    parser.add_argument("--saga_step_size", type=float, default=0.02)
    parser.add_argument("--saga_table_device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--dense_lr", type=float, default=1e-3)
    parser.add_argument("--dense_n_iters", type=int, default=20)
    parser.add_argument("--max_train_images", type=int, default=0)
    parser.add_argument("--max_val_images", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=20)
    return parser.parse_args()


def load_config(run_dir: Path, args: argparse.Namespace, output_dir: Path) -> Config:
    payload = json.loads((run_dir / "config.json").read_text())
    cfg = Config(**payload)
    cfg.mode = "train"
    cfg.device = args.device
    cfg.train_manifest = args.train_manifest or cfg.train_manifest
    cfg.val_manifest = str(args.val_manifest)
    cfg.val_root = str(args.val_root)
    cfg.max_train_images = int(args.max_train_images)
    cfg.max_val_images = int(args.max_val_images)
    cfg.val_split = 0.0
    cfg.save_dir = str(output_dir.parent)
    cfg.run_name = output_dir.name
    cfg.feature_dir = str(output_dir / "features")
    cfg.skip_final_layer = False
    cfg.final_layer_type = str(args.final_layer_type)
    cfg.feature_batch_size = int(args.feature_batch_size)
    cfg.feature_workers = int(args.feature_workers)
    cfg.feature_prefetch_factor = int(args.feature_prefetch_factor)
    cfg.saga_batch_size = int(args.saga_batch_size)
    cfg.saga_workers = int(args.saga_workers)
    cfg.saga_lam = float(args.saga_lam)
    cfg.saga_n_iters = int(args.saga_n_iters)
    cfg.saga_step_size = float(args.saga_step_size)
    cfg.saga_table_device = str(args.saga_table_device)
    cfg.dense_lr = float(args.dense_lr)
    cfg.dense_n_iters = int(args.dense_n_iters)
    cfg.log_every = int(args.log_every)
    cfg.eval_every = 0
    return cfg


def main() -> None:
    args = parse_args()
    source_run_dir = args.run_dir.resolve()
    if not (source_run_dir / "config.json").is_file():
        raise FileNotFoundError(f"Missing config.json in {source_run_dir}")
    best_path = source_run_dir / "concept_head_best.pt"
    if not best_path.is_file():
        raise FileNotFoundError(f"Missing concept_head_best.pt in {source_run_dir}")

    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_dir = (
        args.output_dir.resolve()
        if str(args.output_dir)
        else source_run_dir / f"actual_val_{args.final_layer_type}_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(source_run_dir, args, output_dir)
    configure_runtime(cfg)
    validate_manifest_paths(cfg.val_manifest, expected_rows=36500, label="Places365 val manifest")
    (output_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    start = time.perf_counter()
    train_dataset, val_dataset, concept_filter_summary = build_datasets(cfg)
    concepts = list(train_dataset.concepts)
    (output_dir / "concepts.txt").write_text("\n".join(concepts))
    if concept_filter_summary is not None:
        (output_dir / "concept_filter_summary.json").write_text(json.dumps(concept_filter_summary, indent=2))

    backbone, head = build_model(cfg, n_concepts=len(concepts))
    head.load_state_dict(torch.load(best_path, map_location=cfg.device))
    backbone.eval()
    head.eval()

    feature_dir = Path(cfg.feature_dir)
    feature_train_loader = build_loader(
        train_dataset,
        cfg,
        shuffle=False,
        drop_last=False,
        batch_size=cfg.feature_batch_size,
        workers=cfg.feature_workers,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=cfg.feature_prefetch_factor,
    )
    feature_val_loader = build_loader(
        val_dataset,
        cfg,
        shuffle=False,
        drop_last=False,
        batch_size=cfg.feature_batch_size,
        workers=cfg.feature_workers,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=cfg.feature_prefetch_factor,
    )
    train_feature_path, train_target_path, train_extract_summary = extract_concept_features_to_memmap(
        backbone,
        head,
        feature_train_loader,
        cfg,
        split_name="train",
        output_dir=feature_dir,
    )
    val_feature_path, val_target_path, val_extract_summary = extract_concept_features_to_memmap(
        backbone,
        head,
        feature_val_loader,
        cfg,
        split_name="val",
        output_dir=feature_dir,
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
        output_dir / "final_layer_normalization.pt",
    )

    final_layer_fn = train_dense_final_layer if cfg.final_layer_type == "dense" else train_sparse_final_layer
    final_layer_summary: Dict[str, Any] = final_layer_fn(
        train_feature_path=train_feature_path,
        train_target_path=train_target_path,
        val_feature_path=val_feature_path,
        val_target_path=val_target_path,
        feature_mean=feature_mean,
        feature_std=feature_std,
        cfg=cfg,
        n_classes=infer_num_classes(train_dataset),
        run_dir=output_dir,
    )
    final_layer_summary["type"] = cfg.final_layer_type
    final_layer_summary["feature_extraction"] = {
        "train": train_extract_summary,
        "val": val_extract_summary,
        "normalization": norm_summary,
        "source_concept_head": str(best_path),
    }
    payload = {
        "source_run_dir": str(source_run_dir),
        "output_dir": str(output_dir),
        "n_concepts": len(concepts),
        "n_classes": infer_num_classes(train_dataset),
        "final_layer": final_layer_summary,
        "elapsed_sec": time.perf_counter() - start,
    }
    (output_dir / "actual_val_summary.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
