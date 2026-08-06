#!/usr/bin/env python3
"""Build GroundingDINO manifests for PartImageNet++ generic part concepts.

PartImageNet++ category names are class-specific, e.g. ``bald eagle wing``.
For SG-CBM concept experiments we want the class-agnostic concept ``wing``.
This script strips the ImageNet object prefix, writes a generic part concept
bank, and emits a JSONL manifest where each image is queried only for parts
valid for its ImageNet class.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _norm(text: str) -> str:
    return " ".join(text.replace("_", " ").lower().strip().split())


def generic_part_name(full_part_name: str, object_name: str) -> str:
    full = _norm(full_part_name)
    obj = _norm(object_name)
    prefix = f"{obj} "
    if full.startswith(prefix):
        return full[len(prefix) :].strip()
    # Fallback for rare metadata mismatches. Most PartImageNet++ labels are
    # object-prefixed; taking the final token avoids leaking class names.
    return full.split()[-1]


def load_category_names(root: Path) -> dict[str, dict[str, Any]]:
    path = root / "category_name.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    by_wnid: dict[str, dict[str, Any]] = {}
    for row in rows:
        wnid = Path(row["file name"]).stem
        object_name = _norm(row["object name"])
        full_parts = [_norm(part) for part in row["part name"]]
        generic_parts = [generic_part_name(part, object_name) for part in full_parts]
        by_wnid[wnid] = {
            "wnid": wnid,
            "object_name": object_name,
            "full_parts": full_parts,
            "generic_parts": sorted(set(generic_parts)),
            "full_to_generic": dict(zip(full_parts, generic_parts)),
        }
    return by_wnid


def load_class_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def class_generic_parts(class_payload: dict[str, Any], class_meta: dict[str, Any]) -> list[str]:
    full_to_generic = class_meta["full_to_generic"]
    parts: list[str] = []
    for category in class_payload.get("categories", []):
        full = _norm(category["name"])
        parts.append(full_to_generic.get(full, generic_part_name(full, class_meta["object_name"])))
    return sorted(set(parts))


def iter_manifest_rows(
    partimagenet_root: Path,
    imagenet_train_root: Path,
    out_root: Path,
    require_images: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    class_meta = load_category_names(partimagenet_root)
    rows: list[dict[str, Any]] = []
    missing_images: list[str] = []
    concept_counter: Counter[str] = Counter()
    class_counts: dict[str, int] = {}

    for class_json_path in sorted((partimagenet_root / "json").glob("*.json")):
        wnid = class_json_path.stem
        if wnid not in class_meta:
            raise KeyError(f"{wnid} is missing from category_name.json")
        payload = load_class_json(class_json_path)
        parts = class_generic_parts(payload, class_meta[wnid])
        if not parts:
            continue
        for part in parts:
            concept_counter[part] += 1

        class_counts[wnid] = len(payload.get("images", []))
        for image in payload.get("images", []):
            rel = Path(image["file_name"])
            image_path = imagenet_train_root / rel
            if require_images and not image_path.is_file():
                missing_images.append(str(image_path))
                continue
            out_path = out_root / wnid / f"{rel.stem}.json"
            rows.append(
                {
                    "image": str(image_path),
                    "out": str(out_path),
                    "file_name": str(rel),
                    "wnid": wnid,
                    "image_id": image.get("id"),
                    "width": image.get("width"),
                    "height": image.get("height"),
                    "labels": parts,
                    "object_name": class_meta[wnid]["object_name"],
                }
            )

    summary = {
        "num_rows": len(rows),
        "num_classes": len(class_counts),
        "num_generic_concepts": len(concept_counter),
        "num_missing_images": len(missing_images),
        "missing_image_examples": missing_images[:20],
        "images_per_class": class_counts,
        "concept_class_frequency": dict(sorted(concept_counter.items())),
    }
    return rows, summary


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partimagenet_root", type=Path, required=True)
    parser.add_argument("--imagenet_train_root", type=Path, required=True)
    parser.add_argument("--out_root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--concepts_out", type=Path, required=True)
    parser.add_argument("--class_parts_out", type=Path, required=True)
    parser.add_argument("--summary_out", type=Path, required=True)
    parser.add_argument("--allow_missing_images", action="store_true")
    args = parser.parse_args()

    rows, summary = iter_manifest_rows(
        args.partimagenet_root,
        args.imagenet_train_root,
        args.out_root,
        require_images=not args.allow_missing_images,
    )
    class_meta = load_category_names(args.partimagenet_root)
    concepts = sorted(summary["concept_class_frequency"].keys())

    write_jsonl(args.manifest, rows)
    args.concepts_out.parent.mkdir(parents=True, exist_ok=True)
    args.concepts_out.write_text("\n".join(concepts) + "\n", encoding="utf-8")
    args.class_parts_out.parent.mkdir(parents=True, exist_ok=True)
    args.class_parts_out.write_text(json.dumps(class_meta, indent=2, sort_keys=True), encoding="utf-8")
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps({k: summary[k] for k in ("num_rows", "num_classes", "num_generic_concepts", "num_missing_images")}))
    print(f"manifest={args.manifest}")
    print(f"concepts={args.concepts_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
