#!/usr/bin/env python3
"""Re-evaluate a saved Places365 SG-CBM head using a clean validation manifest.

This is intended for fixing evaluation-only issues: it reuses an existing train
concept-feature memmap and recomputes only validation concept features from the
saved CBL checkpoint.
"""

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
from gcbm.imagenet_data import DatasetView, SafeImageFolderWithAnnotations, build_loader  # noqa: E402
from gcbm.imagenet_final_layers import (  # noqa: E402
    compute_feature_stats_memmap,
    extract_concept_features_to_memmap,
    train_sparse_final_layer,
)
from gcbm.imagenet_models import build_model  # noqa: E402
from gcbm.runtime import configure_runtime  # noqa: E402
from gcbm.train_imagenet import infer_num_classes  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-evaluate Places365 SG-CBM checkpoint on clean val images.")
    parser.add_argument("--run_dir", required=True, type=Path)
    parser.add_argument("--train_features", required=True, type=Path)
    parser.add_argument("--train_targets", required=True, type=Path)
    parser.add_argument("--val_manifest", required=True, type=Path)
    parser.add_argument("--val_root", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--feature_batch_size", type=int, default=512)
    parser.add_argument("--feature_workers", type=int, default=8)
    parser.add_argument("--feature_prefetch_factor", type=int, default=2)
    parser.add_argument("--saga_batch_size", type=int, default=512)
    parser.add_argument("--saga_workers", type=int, default=0)
    parser.add_argument("--saga_lam", type=float, default=0.0007)
    parser.add_argument("--saga_n_iters", type=int, default=100)
    parser.add_argument("--saga_step_size", type=float, default=0.1)
    parser.add_argument("--saga_table_device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--log_every", type=int, default=20)
    return parser.parse_args()


def read_concepts(path: Path) -> list[str]:
    concepts = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not concepts:
        raise ValueError(f"No concepts found in {path}")
    return concepts


def validate_manifest_paths(manifest: Path, *, expected_rows: int) -> int:
    missing: list[str] = []
    rows = 0
    with manifest.open("r") as handle:
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
        raise FileNotFoundError(f"Validation manifest contains missing paths; first={missing[0]}")
    if rows != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} Places365 validation rows, found {rows}")
    return rows


def load_config(run_dir: Path, args: argparse.Namespace, output_dir: Path, concepts_path: Path) -> Config:
    payload = json.loads((run_dir / "config.json").read_text())
    cfg = Config(**payload)
    cfg.mode = "train"
    cfg.device = args.device
    cfg.concept_file = str(concepts_path)
    cfg.val_root = str(args.val_root)
    cfg.val_manifest = str(args.val_manifest)
    cfg.save_dir = str(output_dir)
    cfg.run_name = output_dir.name
    cfg.feature_dir = str(output_dir / "features")
    cfg.precomputed_target_dir = ""
    cfg.max_train_images = 0
    cfg.max_val_images = 0
    cfg.val_split = 0.0
    cfg.eval_every = 0
    cfg.log_every = int(args.log_every)
    cfg.feature_batch_size = int(args.feature_batch_size)
    cfg.feature_workers = int(args.feature_workers)
    cfg.feature_prefetch_factor = int(args.feature_prefetch_factor)
    cfg.saga_batch_size = int(args.saga_batch_size)
    cfg.saga_workers = int(args.saga_workers)
    cfg.saga_lam = float(args.saga_lam)
    cfg.saga_n_iters = int(args.saga_n_iters)
    cfg.saga_step_size = float(args.saga_step_size)
    cfg.saga_table_device = str(args.saga_table_device)
    cfg.final_layer_type = "sparse"
    return cfg


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    best_path = run_dir / "concept_head_best.pt"
    concepts_path = run_dir / "concepts.txt"
    if not best_path.is_file():
        raise FileNotFoundError(f"Missing concept_head_best.pt: {best_path}")
    if not concepts_path.is_file():
        raise FileNotFoundError(f"Missing concepts.txt: {concepts_path}")
    if not args.train_features.is_file():
        raise FileNotFoundError(f"Missing train features: {args.train_features}")
    if not args.train_targets.is_file():
        raise FileNotFoundError(f"Missing train targets: {args.train_targets}")

    start = time.perf_counter()
    val_rows = validate_manifest_paths(args.val_manifest, expected_rows=36500)
    concepts = read_concepts(concepts_path)
    cfg = load_config(run_dir, args, output_dir, concepts_path)
    configure_runtime(cfg)
    (output_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    val_full = SafeImageFolderWithAnnotations(
        root=str(args.val_root),
        annotation_dir="",
        concepts=concepts,
        input_size=cfg.input_size,
        min_image_bytes=cfg.min_image_bytes,
        split="val",
        manifest=str(args.val_manifest),
        train_random_transforms=False,
    )
    val_dataset = DatasetView(val_full, list(range(len(val_full))))

    backbone, head = build_model(cfg, n_concepts=len(concepts))
    head.load_state_dict(torch.load(best_path, map_location=cfg.device))
    backbone.eval()
    head.eval()

    feature_dir = Path(cfg.feature_dir)
    val_loader = build_loader(
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
    val_feature_path, val_target_path, val_extract_summary = extract_concept_features_to_memmap(
        backbone,
        head,
        val_loader,
        cfg,
        split_name="val",
        output_dir=feature_dir,
    )
    feature_mean, feature_std, norm_summary = compute_feature_stats_memmap(args.train_features, cfg)
    torch.save(
        {
            "mean": feature_mean,
            "std": feature_std,
            "train_feature_path": str(args.train_features),
            "train_target_path": str(args.train_targets),
            "val_extraction": val_extract_summary,
            "normalization": norm_summary,
            "source_concept_head": str(best_path),
        },
        output_dir / "final_layer_normalization.pt",
    )

    final_layer_summary: Dict[str, Any] = train_sparse_final_layer(
        train_feature_path=args.train_features,
        train_target_path=args.train_targets,
        val_feature_path=val_feature_path,
        val_target_path=val_target_path,
        feature_mean=feature_mean,
        feature_std=feature_std,
        cfg=cfg,
        n_classes=infer_num_classes(val_dataset),
        run_dir=output_dir,
    )
    final_layer_summary["type"] = "sparse"
    final_layer_summary["feature_extraction"] = {
        "train": {
            "feature_path": str(args.train_features),
            "target_path": str(args.train_targets),
        },
        "val": val_extract_summary,
        "normalization": norm_summary,
        "source_concept_head": str(best_path),
    }

    payload = {
        "source_run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "n_concepts": len(concepts),
        "n_classes": infer_num_classes(val_dataset),
        "val_rows_validated": val_rows,
        "final_layer": final_layer_summary,
        "elapsed_sec": time.perf_counter() - start,
    }
    (output_dir / "clean_val_summary.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
