#!/usr/bin/env python3
"""Audit PartImageNet++ human part polygons after ImageNet evaluation preprocessing."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from eval_partimagenetpp_gtbox_localization import rasterize_polygon_union  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize polygon integrity and target sizes for PartImageNet++ segmentation GT."
    )
    parser.add_argument("--segments-jsonl", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--map-size", type=int, default=224)
    return parser.parse_args()


def polygon_area(polygon: list[float]) -> float:
    points = list(zip(polygon[::2], polygon[1::2]))
    return abs(
        sum(
            x1 * y2 - x2 * y1
            for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1])
        )
        / 2.0
    )


def canonicalize(text: str) -> str:
    return " ".join(str(text).lower().replace("_", " ").split())


def main() -> None:
    args = parse_args()
    map_size = int(args.map_size)
    if map_size <= 0:
        raise ValueError("--map-size must be positive")

    rows = 0
    rows_with_targets = 0
    targets = 0
    identity_targets = 0
    polygon_count = 0
    invalid_vertex_encoding = 0
    zero_area_polygons = 0
    out_of_bounds_vertices = 0
    areas: list[int] = []
    labels: Counter[str] = Counter()

    with Path(args.segments_jsonl).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row: dict[str, Any] = json.loads(line)
            rows += 1
            width, height = int(row["width"]), int(row["height"])
            object_name = canonicalize(row.get("object_name", ""))
            has_target = False
            for label, instances in row.get("segmentations", {}).items():
                if canonicalize(label) == object_name:
                    identity_targets += 1
                for instance in instances:
                    for polygon in instance:
                        polygon_count += 1
                        if len(polygon) < 6 or len(polygon) % 2:
                            invalid_vertex_encoding += 1
                            continue
                        if polygon_area(polygon) == 0:
                            zero_area_polygons += 1
                        for x, y in zip(polygon[::2], polygon[1::2]):
                            if x < 0 or x > width or y < 0 or y > height:
                                out_of_bounds_vertices += 1

                mask = rasterize_polygon_union(
                    instances, image_size=(width, height), map_h=map_size, map_w=map_size
                )
                if mask.any():
                    has_target = True
                    targets += 1
                    areas.append(int(mask.sum()))
                    labels[str(label)] += 1
            rows_with_targets += int(has_target)

    target_areas = np.asarray(areas, dtype=np.int64)
    if not len(target_areas):
        raise RuntimeError("No target masks survive preprocessing")
    total_pixels = map_size * map_size
    payload = {
        "segments_jsonl": str(Path(args.segments_jsonl)),
        "map_size": map_size,
        "rows": rows,
        "rows_with_surviving_targets": rows_with_targets,
        "targets_with_surviving_masks": targets,
        "identity_concept_targets_before_filtering": identity_targets,
        "polygon_integrity": {
            "polygons": polygon_count,
            "invalid_vertex_encoding": invalid_vertex_encoding,
            "zero_area_polygons": zero_area_polygons,
            "out_of_bounds_vertices": out_of_bounds_vertices,
        },
        "target_area_pixels": {
            "min": int(target_areas.min()),
            "p01": float(np.quantile(target_areas, 0.01)),
            "p05": float(np.quantile(target_areas, 0.05)),
            "p10": float(np.quantile(target_areas, 0.10)),
            "median": float(np.median(target_areas)),
            "p90": float(np.quantile(target_areas, 0.90)),
            "max": int(target_areas.max()),
        },
        "target_area_fraction": {
            "at_most_1_pixel": float((target_areas <= 1).mean()),
            "at_most_4_pixels": float((target_areas <= 4).mean()),
            "below_0_1_percent": float((target_areas < 0.001 * total_pixels).mean()),
            "below_1_percent": float((target_areas < 0.01 * total_pixels).mean()),
            "above_50_percent": float((target_areas > 0.5 * total_pixels).mean()),
            "above_90_percent": float((target_areas > 0.9 * total_pixels).mean()),
        },
        "top_labels": labels.most_common(30),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
