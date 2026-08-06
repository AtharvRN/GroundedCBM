#!/usr/bin/env python3
"""Build a GroundingDINO manifest for ImageNet val using PartImageNet++ parts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scipy.io import loadmat

def load_imagenet_id_to_wnid(meta_mat: Path) -> dict[int, str]:
    meta = loadmat(str(meta_mat), squeeze_me=True, struct_as_record=False)
    rows = meta["synsets"]
    id_to_wnid: dict[int, str] = {}
    for row in rows:
        imagenet_id = int(row.ILSVRC2012_ID)
        wnid = str(row.WNID)
        id_to_wnid[imagenet_id] = wnid
    return id_to_wnid


def load_val_labels(path: Path) -> list[int]:
    return [int(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_class_parts(path: Path) -> tuple[dict[str, list[str]], dict[str, dict[str, Any]]]:
    class_meta = json.loads(path.read_text(encoding="utf-8"))
    class_parts = {
        wnid: sorted(set(str(part) for part in payload.get("generic_parts", [])))
        for wnid, payload in class_meta.items()
    }
    return class_parts, class_meta


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--class_parts", type=Path, required=True)
    parser.add_argument("--imagenet_val_root", type=Path, required=True)
    parser.add_argument("--val_ground_truth", type=Path, required=True)
    parser.add_argument("--meta_mat", type=Path, required=True)
    parser.add_argument("--out_root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--summary_out", type=Path, required=True)
    parser.add_argument("--allow_missing_images", action="store_true")
    parser.add_argument("--skip_image_check", action="store_true")
    args = parser.parse_args()

    id_to_wnid = load_imagenet_id_to_wnid(args.meta_mat)
    labels = load_val_labels(args.val_ground_truth)
    class_parts, class_meta = load_class_parts(args.class_parts)

    rows: list[dict[str, Any]] = []
    missing_images: list[str] = []
    missing_parts: list[str] = []
    for index, imagenet_id in enumerate(labels, start=1):
        wnid = id_to_wnid[imagenet_id]
        parts = class_parts.get(wnid, [])
        if not parts:
            missing_parts.append(wnid)
            continue
        image_name = f"ILSVRC2012_val_{index:08d}.JPEG"
        image_path = args.imagenet_val_root / image_name
        if not args.skip_image_check and not image_path.is_file():
            missing_images.append(str(image_path))
            if not args.allow_missing_images:
                continue
        out_path = args.out_root / wnid / f"{Path(image_name).stem}.json"
        rows.append(
            {
                "image": str(image_path),
                "out": str(out_path),
                "file_name": image_name,
                "wnid": wnid,
                "image_id": index,
                "labels": parts,
                "object_name": class_meta[wnid]["object_name"],
                "split": "imagenet_val",
            }
        )

    summary = {
        "num_rows": len(rows),
        "num_val_labels": len(labels),
        "num_classes": len(set(row["wnid"] for row in rows)),
        "num_missing_images": len(missing_images),
        "missing_image_examples": missing_images[:20],
        "num_missing_part_classes": len(set(missing_parts)),
        "missing_part_classes": sorted(set(missing_parts))[:20],
    }
    write_jsonl(args.manifest, rows)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    print(f"manifest={args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
