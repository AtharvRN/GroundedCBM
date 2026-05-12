#!/usr/bin/env python3
"""Precompute a part-aligned CUB annotation cache for localization eval."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gcbm.evaluate_savlg_cub_parts import (
    build_dataset_image_ids,
    canonicalize_concept_label,
    load_images_index,
    load_mapping,
    load_part_locs,
    load_parts,
    preload_mapped_gt_concepts,
    resolve_base_index,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a compact JSON cache mapping CUB dataset indices to the "
            "concept/part annotations used by SG-CBM CUB part localization."
        )
    )
    parser.add_argument("--load_path", required=True, help="SG-CBM/SAVLG run directory containing args.txt and concepts.txt.")
    parser.add_argument("--annotation_dir", required=True, help="Directory containing cub_train/cub_val annotation JSONs.")
    parser.add_argument("--cub_root", required=True, help="Official CUB_200_2011 root with images.txt and parts/.")
    parser.add_argument("--mapping_json", required=True, help="Concept-to-CUB-part mapping JSON.")
    parser.add_argument("--output", required=True, help="Output cache JSON path.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max_images", type=int, default=None, help="Optional cap on processed images.")
    return parser.parse_args()


def main() -> None:
    args_ns = parse_args()
    from gcbm.savlg_eval_common import _load_args, _load_concepts
    from methods.savlg import create_savlg_splits

    args = _load_args(args_ns.load_path, args_ns.device, args_ns.annotation_dir)
    if getattr(args, "skip_test_eval", False):
        args.skip_test_eval = False
    _, _, _, _, test_dataset, _backbone = create_savlg_splits(args)
    if args_ns.max_images is not None:
        keep = min(args_ns.max_images, len(test_dataset))
        test_dataset = torch.utils.data.Subset(test_dataset, list(range(keep)))

    cub_root = Path(args_ns.cub_root)
    images_index = load_images_index(cub_root / "images.txt")
    part_names = load_parts(cub_root / "parts" / "parts.txt")
    part_locs = load_part_locs(cub_root / "parts" / "part_locs.txt", part_names)
    concept_to_parts = load_mapping(Path(args_ns.mapping_json))
    concepts = _load_concepts(args_ns.load_path, args)
    concept_to_idx = {canonicalize_concept_label(name): idx for idx, name in enumerate(concepts)}

    ann_split_dir = Path(args.annotation_dir) / f"{args.dataset}_test"
    if not ann_split_dir.is_dir():
        ann_split_dir = Path(args.annotation_dir) / f"{args.dataset}_val"

    dataset_base_indices = [resolve_base_index(test_dataset, i) for i in range(len(test_dataset))]
    image_ids_by_ds_idx = build_dataset_image_ids(test_dataset, images_index)
    image_part_names = {img_id: set(parts.keys()) for img_id, parts in part_locs.items()}
    items_by_base_idx = preload_mapped_gt_concepts(
        ann_split_dir=ann_split_dir,
        dataset_base_indices=dataset_base_indices,
        image_ids_by_ds_idx=image_ids_by_ds_idx,
        image_part_names_by_id=image_part_names,
        concept_to_parts=concept_to_parts,
        concept_to_idx=concept_to_idx,
    )

    ds_to_base_idx = {int(ds_idx): int(resolve_base_index(test_dataset, ds_idx)) for ds_idx in range(len(test_dataset))}
    base_to_image_id = {
        base_idx: image_ids_by_ds_idx[ds_idx]
        for ds_idx, base_idx in ds_to_base_idx.items()
        if ds_idx in image_ids_by_ds_idx
    }

    payload = {
        "meta": {
            "load_path": args_ns.load_path,
            "annotation_dir": args_ns.annotation_dir,
            "cub_root": args_ns.cub_root,
            "mapping_json": args_ns.mapping_json,
            "dataset": args.dataset,
            "split": "test",
            "num_dataset_images": len(test_dataset),
            "num_cached_images": len(items_by_base_idx),
            "num_cached_instances": int(sum(len(v) for v in items_by_base_idx.values())),
        },
        "items_by_base_idx": {
            str(base_idx): [{"label": label, "exact_parts": exact_parts} for label, exact_parts in items]
            for base_idx, items in sorted(items_by_base_idx.items())
        },
        "image_id_by_base_idx": {str(base_idx): int(image_id) for base_idx, image_id in sorted(base_to_image_id.items())},
    }

    out = Path(args_ns.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(out),
                "num_cached_images": payload["meta"]["num_cached_images"],
                "num_cached_instances": payload["meta"]["num_cached_instances"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
