from __future__ import annotations

import argparse
import inspect
import json
import os
import shutil
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import Subset

from gcbm.medical_annotations import load_concepts
from gcbm.medical_data import medical_labels
from gcbm.medical_target_store import write_target_store_from_payload, write_target_store_from_shards
from gcbm.train_medical import build_datasets, get_or_build_presence_targets, get_or_build_targets, maybe_subset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute medical SG-CBM target caches from grounded annotation JSONs.")
    parser.add_argument("--dataset", choices=["chexpert", "mimic"], required=True, help="Medical dataset.")
    parser.add_argument("--data_dir", required=True, help="Dataset root.")
    parser.add_argument("--concept_file", required=True, help="Text or JSON concept bank.")
    parser.add_argument("--train_annotation_dir", required=True, help="Train grounded annotation directory.")
    parser.add_argument("--val_annotation_dir", required=True, help="Validation grounded annotation directory.")
    parser.add_argument("--output_dir", default="", help="Optional ImageNet-style cache root. Writes train_targets.pt / val_targets.pt and optional presence caches.")
    parser.add_argument("--train_output", default="", help="Output .pt cache for train SG-CBM targets.")
    parser.add_argument("--val_output", default="", help="Output .pt cache for validation SG-CBM targets.")
    parser.add_argument("--train_presence_output", default="", help="Optional lightweight train presence cache.")
    parser.add_argument("--val_presence_output", default="", help="Optional lightweight validation presence cache.")
    parser.add_argument("--train_csv", default="", help="CheXpert train CSV override.")
    parser.add_argument("--val_csv", default="", help="CheXpert validation CSV override.")
    parser.add_argument("--img_root", default="", help="Image root override.")
    parser.add_argument("--mimic_label_csv", default="", help="MIMIC label CSV override.")
    parser.add_argument("--mimic_split_csv", default="", help="MIMIC split CSV override.")
    parser.add_argument("--mimic_metadata_csv", default="", help="MIMIC metadata CSV override.")
    parser.add_argument("--label_subset", choices=["all", "competition", "pathology"], default="all", help="CheXpert-style labels.")
    parser.add_argument("--uncertain_strategy", choices=["ones", "zeros", "ignore"], default="ones", help="Uncertain-label mapping.")
    parser.add_argument("--frontal_only", action=argparse.BooleanOptionalAction, default=True, help="Use frontal/AP/PA images only.")
    parser.add_argument("--img_size", type=int, default=224, help="Input crop size.")
    parser.add_argument("--resize_size", type=int, default=256, help="Resize short edge before center crop.")
    parser.add_argument("--mask_h", type=int, default=14, help="Spatial target height.")
    parser.add_argument("--mask_w", type=int, default=14, help="Spatial target width.")
    parser.add_argument("--target_mode", choices=["soft_box", "hard_iou"], default="soft_box", help="Box-to-mask target mode.")
    parser.add_argument("--patch_iou_thresh", type=float, default=0.5, help="Patch IoU threshold for hard_iou targets.")
    parser.add_argument("--concept_threshold", type=float, default=0.70, help="Positive concept confidence threshold.")
    parser.add_argument("--neg_threshold", type=float, default=0.02, help="Soft target lower calibration threshold.")
    parser.add_argument("--presence_mode", choices=["binary", "soft"], default="binary", help="Global target mode.")
    parser.add_argument("--num_workers", type=int, default=16, help="Annotation parsing workers.")
    parser.add_argument("--max_train_images", type=int, default=0, help="Optional train subset.")
    parser.add_argument("--max_val_images", type=int, default=0, help="Optional validation subset.")
    parser.add_argument("--train_shard_size", type=int, default=10000, help="Rows per train target shard; 0 disables sharding.")
    parser.add_argument("--train_shard_dir", default="", help="Directory for resumable train target shards.")
    parser.add_argument(
        "--store_format",
        choices=["pt", "memmap", "both"],
        default="pt",
        help="Target-cache format. memmap writes ImageNet-style split stores without merging full train targets in RAM.",
    )
    parser.add_argument("--allow_annotation_index_fallback", action="store_true", help="Allow row-index fallback when paths do not match.")
    return parser.parse_args()


def payload_summary(payload: dict) -> dict:
    return {
        key: value
        for key, value in payload.items()
        if isinstance(value, (int, float, str, bool))
    }


def resolve_output_paths(args: argparse.Namespace) -> None:
    if args.output_dir:
        root = Path(args.output_dir)
        args.train_output = args.train_output or str(root / "train_targets.pt")
        args.val_output = args.val_output or str(root / "val_targets.pt")
        args.train_presence_output = args.train_presence_output or str(root / "train_presence.pt")
        args.val_presence_output = args.val_presence_output or str(root / "val_presence.pt")
    if not args.train_output or not args.val_output:
        raise ValueError("Provide --output_dir or both --train_output and --val_output")


def torch_load_cpu(path: str | Path) -> Any:
    kwargs: dict[str, Any] = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False
    return torch.load(path, **kwargs)


class IndexedSubset(Subset):
    def get_image_path(self, index: int) -> str:
        return self.dataset.get_image_path(self.indices[index])  # type: ignore[attr-defined]

    def get_image_size(self, index: int):
        return self.dataset.get_image_size(self.indices[index])  # type: ignore[attr-defined]


def _copy_or_link_annotation(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def prepare_local_annotation_dir(annotation_dir: str, row_indices: Sequence[int], out_dir: Path) -> Path:
    """Create local-indexed annotation links so existing index-aligned parser can shard safely."""
    out_dir.mkdir(parents=True, exist_ok=True)
    source_dir = Path(annotation_dir)
    for local_idx, global_idx in enumerate(row_indices):
        src = source_dir / f"{int(global_idx)}.json"
        if src.exists():
            _copy_or_link_annotation(src, out_dir / f"{local_idx}.json")
    return out_dir


def merge_target_shards(shard_paths: Sequence[Path], output_path: str) -> dict[str, Any]:
    payloads = [torch_load_cpu(path) for path in shard_paths]
    if not payloads:
        raise ValueError("No target shards to merge")
    merged = {
        "global_targets": torch.cat([payload["global_targets"].float() for payload in payloads], dim=0),
        "presence_scores": torch.cat([payload["presence_scores"].float() for payload in payloads], dim=0),
        "mask_indices": [item for payload in payloads for item in payload["mask_indices"]],
        "mask_targets": [item for payload in payloads for item in payload["mask_targets"]],
        "matched_annotations": sum(int(payload.get("matched_annotations", 0)) for payload in payloads),
        "unmatched_annotations": sum(int(payload.get("unmatched_annotations", 0)) for payload in payloads),
        "num_concepts": int(payloads[0]["num_concepts"]),
        "num_images": sum(int(payload["num_images"]) for payload in payloads),
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, output_path)
    print(f"[medical targets] merged {len(shard_paths)} train shards -> {output_path}", flush=True)
    return merged


def get_or_build_train_target_shards(dataset, args: argparse.Namespace, concepts: list[str]) -> list[Path]:
    output_path = Path(args.train_output)
    shard_size = int(args.train_shard_size)
    if shard_size <= 0 or len(dataset) <= shard_size:
        get_or_build_targets(dataset, args, concepts, args.train_annotation_dir, args.train_output)
        return [output_path]

    shard_root = Path(args.train_shard_dir) if args.train_shard_dir else output_path.with_suffix(output_path.suffix + ".shards")
    if output_path.exists() and not shard_root.exists():
        return [output_path]
    shard_root.mkdir(parents=True, exist_ok=True)
    shard_paths: list[Path] = []
    for shard_id, start in enumerate(range(0, len(dataset), shard_size)):
        stop = min(start + shard_size, len(dataset))
        shard_path = shard_root / f"targets_{shard_id:05d}_{start:06d}_{stop:06d}.pt"
        shard_paths.append(shard_path)
        if shard_path.exists():
            print(f"[medical targets] shard exists {shard_path}", flush=True)
            continue
        indices = list(range(start, stop))
        shard_ds = IndexedSubset(dataset, indices)
        shard_annotation_dir = prepare_local_annotation_dir(args.train_annotation_dir, indices, shard_root / f"annotations_{shard_id:05d}_{start:06d}_{stop:06d}")
        shard_args = argparse.Namespace(**vars(args))
        shard_args.allow_annotation_index_fallback = False
        get_or_build_targets(shard_ds, shard_args, concepts, str(shard_annotation_dir), str(shard_path))
        print(f"[medical targets] wrote train shard {shard_path}", flush=True)

    return shard_paths


def get_or_build_train_targets_sharded(dataset, args: argparse.Namespace, concepts: list[str]) -> dict[str, Any]:
    output_path = Path(args.train_output)
    if output_path.exists():
        return torch_load_cpu(output_path)
    shard_paths = get_or_build_train_target_shards(dataset, args, concepts)
    return merge_target_shards(shard_paths, args.train_output)


def main() -> None:
    args = parse_args()
    resolve_output_paths(args)
    labels = medical_labels(
        args.dataset,
        competition=args.label_subset == "competition",
        pathology=args.label_subset == "pathology",
    )
    concepts = load_concepts(args.concept_file)
    train_ds, val_ds = build_datasets(args, labels)
    train_ds = maybe_subset(train_ds, args.max_train_images)
    val_ds = maybe_subset(val_ds, args.max_val_images)

    if args.train_presence_output:
        get_or_build_presence_targets(train_ds, args, concepts, args.train_annotation_dir, args.train_presence_output)
    if args.val_presence_output:
        get_or_build_presence_targets(val_ds, args, concepts, args.val_annotation_dir, args.val_presence_output)

    write_pt = args.store_format in {"pt", "both"}
    write_memmap = args.store_format in {"memmap", "both"}
    train_payload = None
    val_payload = None

    if write_memmap:
        train_shards = get_or_build_train_target_shards(train_ds, args, concepts)
        train_store_dir = Path(args.output_dir) / "train" if args.output_dir else Path(args.train_output).with_suffix("") / "train"
        val_store_dir = Path(args.output_dir) / "val" if args.output_dir else Path(args.val_output).with_suffix("") / "val"
        train_store_meta = write_target_store_from_shards(
            train_shards,
            train_store_dir,
            split="train",
            concepts=concepts,
            extra_metadata={
                "concept_threshold": float(args.concept_threshold),
                "neg_threshold": float(args.neg_threshold),
                "presence_mode": str(args.presence_mode),
                "target_mode": str(args.target_mode),
                "mask_h": int(args.mask_h),
                "mask_w": int(args.mask_w),
            },
        )
        val_payload = get_or_build_targets(val_ds, args, concepts, args.val_annotation_dir, args.val_output)
        val_store_meta = write_target_store_from_payload(
            val_payload,
            val_store_dir,
            split="val",
            concepts=concepts,
            extra_metadata={
                "concept_threshold": float(args.concept_threshold),
                "neg_threshold": float(args.neg_threshold),
                "presence_mode": str(args.presence_mode),
                "target_mode": str(args.target_mode),
                "mask_h": int(args.mask_h),
                "mask_w": int(args.mask_w),
            },
        )
        print(f"[medical targets] wrote train memmap store {train_store_dir}", flush=True)
        print(f"[medical targets] wrote val memmap store {val_store_dir}", flush=True)

    if write_pt:
        train_payload = get_or_build_train_targets_sharded(train_ds, args, concepts)
        if val_payload is None:
            val_payload = get_or_build_targets(val_ds, args, concepts, args.val_annotation_dir, args.val_output)
    elif val_payload is None:
        val_payload = get_or_build_targets(val_ds, args, concepts, args.val_annotation_dir, args.val_output)

    metadata = {
        "dataset": args.dataset,
        "concept_file": args.concept_file,
        "store_format": args.store_format,
        "num_concepts": len(concepts),
        "num_train": len(train_ds),
        "num_val": len(val_ds),
        "train_output": args.train_output,
        "val_output": args.val_output,
        "train_summary": payload_summary(train_payload) if train_payload is not None else {},
        "val_summary": payload_summary(val_payload),
    }
    if write_memmap:
        metadata["train_store"] = train_store_meta
        metadata["val_store"] = val_store_meta
    if args.output_dir:
        output_root = Path(args.output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "concepts.txt").write_text("\n".join(concepts), encoding="utf-8")
        meta_path = output_root / "metadata.json"
    else:
        meta_path = Path(args.train_output).with_suffix(".metadata.json")
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[medical targets] wrote metadata {meta_path}", flush=True)
    if Path(args.train_output).exists():
        print(f"[medical targets] train cache size={Path(args.train_output).stat().st_size / (1024 ** 3):.2f} GiB", flush=True)
    if Path(args.val_output).exists():
        print(f"[medical targets] val cache size={Path(args.val_output).stat().st_size / (1024 ** 2):.2f} MiB", flush=True)


if __name__ == "__main__":
    main()
