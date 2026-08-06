#!/usr/bin/env python3
"""Materialize PartImageNet++ human segmentation polygons for a manifest.

PartImageNet++ stores one COCO annotation file per ImageNet class.  This
script selects the annotations for the manifest images, converts the
class-specific category name (for example ``tench fin``) to the experiment's
generic part name (``fin``), and writes JSONL in manifest order.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--gt_boxes_jsonl",
        type=Path,
        required=True,
        help="Existing validated generic human-box payload, in the same manifest order.",
    )
    parser.add_argument("--source_json_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary_out", type=Path, required=True)
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def xywh_to_xyxy(box: List[float]) -> List[float]:
    x, y, width, height = (float(value) for value in box)
    return [x, y, x + width, y + height]


def generic_concept_for_annotation(annotation: Dict[str, Any], box_row: Dict[str, Any]) -> str:
    """Use the validated box payload as the canonical class-specific-to-generic map."""
    target = xywh_to_xyxy(annotation["bbox"])
    best: tuple[float, str] | None = None
    for raw_concept, boxes in box_row.get("boxes", {}).items():
        for box in boxes:
            error = max(abs(float(actual) - expected) for actual, expected in zip(box, target))
            if best is None or error < best[0]:
                best = (error, str(raw_concept))
    if best is None or best[0] > 1e-3:
        raise RuntimeError(
            f"Could not match source annotation bbox={target} to validated generic boxes for "
            f"{box_row.get('file_name')}; best_error={None if best is None else best[0]}"
        )
    return best[1]


def norm(text: str) -> str:
    return " ".join(str(text).replace("_", " ").lower().strip().split())


def generic_part_name(full_part_name: str, object_name: str) -> str:
    full = norm(full_part_name)
    obj = norm(object_name)
    prefix = f"{obj} "
    if full.startswith(prefix):
        return full[len(prefix) :].strip()
    # This fallback matches the generic-part manifest preparation.
    return full.split()[-1]


def is_polygon_segmentation(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(polygon, list) and len(polygon) >= 6 and len(polygon) % 2 == 0 for polygon in value)
    )


def main() -> None:
    args = parse_args()
    rows = list(iter_jsonl(args.manifest))
    box_rows = list(iter_jsonl(args.gt_boxes_jsonl))
    if not rows:
        raise RuntimeError(f"Manifest is empty: {args.manifest}")
    if len(rows) != len(box_rows):
        raise RuntimeError(f"Manifest/box row count mismatch: {len(rows)} vs {len(box_rows)}")
    for index, (row, box_row) in enumerate(zip(rows, box_rows)):
        if str(row["file_name"]) != str(box_row.get("file_name")):
            raise RuntimeError(f"Manifest/box file_name mismatch at row {index}")

    by_wnid: Dict[str, List[tuple[int, Dict[str, Any]]]] = defaultdict(list)
    for row_index, row in enumerate(rows):
        by_wnid[str(row["wnid"])].append((row_index, row))

    output_rows: List[Dict[str, Any] | None] = [None] * len(rows)
    missing_images: List[str] = []
    unsupported_segmentations = 0
    annotation_count = 0
    polygon_count = 0
    concept_count: Counter[str] = Counter()

    for wnid, class_rows in sorted(by_wnid.items()):
        source_path = args.source_json_dir / f"{wnid}.json"
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing PartImageNet++ source annotation: {source_path}")
        payload = json.loads(source_path.read_text(encoding="utf-8"))
        image_by_id = {int(image["id"]): image for image in payload.get("images", [])}
        category_by_id = {int(category["id"]): str(category["name"]) for category in payload.get("categories", [])}
        annotations_by_image: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for annotation in payload.get("annotations", []):
            annotations_by_image[int(annotation["image_id"])].append(annotation)

        for row_index, row in class_rows:
            image_id = int(row["image_id"])
            image = image_by_id.get(image_id)
            if image is None:
                missing_images.append(f"{wnid}:{image_id}")
                continue
            segmentations: Dict[str, List[List[List[float]]]] = defaultdict(list)
            box_row = box_rows[row_index]
            for annotation in annotations_by_image.get(image_id, []):
                segmentation = annotation.get("segmentation")
                if not is_polygon_segmentation(segmentation):
                    unsupported_segmentations += 1
                    continue
                if int(annotation["category_id"]) not in category_by_id:
                    raise KeyError(f"{wnid}:{image_id} references unknown category {annotation['category_id']}")
                concept = generic_concept_for_annotation(annotation, box_row)
                polygons = [[float(value) for value in polygon] for polygon in segmentation]
                segmentations[concept].append(polygons)
                annotation_count += 1
                polygon_count += len(polygons)
                concept_count[concept] += 1

            output_rows[row_index] = {
                "row_index": row_index,
                "file_name": str(row["file_name"]),
                "wnid": wnid,
                "image_id": image_id,
                "width": int(image["width"]),
                "height": int(image["height"]),
                "object_name": str(row.get("object_name", "")),
                "manifest_labels": list(row.get("labels", [])),
                "segmentations": dict(sorted(segmentations.items())),
            }

    if missing_images:
        raise RuntimeError(
            f"{len(missing_images)} manifest images were not found in source JSON; examples={missing_images[:10]}"
        )
    if any(row is None for row in output_rows):
        raise RuntimeError("Internal error: incomplete materialized output")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary = {
        "manifest": str(args.manifest),
        "gt_boxes_jsonl": str(args.gt_boxes_jsonl),
        "source_json_dir": str(args.source_json_dir),
        "output": str(args.output),
        "rows": len(output_rows),
        "images_with_segmentations": sum(bool(row["segmentations"]) for row in output_rows if row is not None),
        "annotation_instances": annotation_count,
        "polygon_count": polygon_count,
        "generic_concepts": len(concept_count),
        "concept_annotation_counts": dict(sorted(concept_count.items())),
        "unsupported_nonpolygon_segmentations": unsupported_segmentations,
    }
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
