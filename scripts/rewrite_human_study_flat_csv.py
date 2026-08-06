from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Sequence


FIELDNAMES = [
    "trial_index",
    "filename",
    "method",
    "variant",
    "nec",
    "true_class_id_0based",
    "true_class_name",
    "pred_class_id_0based",
    "pred_class_name",
    "correct",
    "rank",
    "concept_index",
    "concept",
    "activation",
    "class_weight",
    "contribution",
    "map_shape",
    "max_cell",
    "max_value",
    "region_quantile",
    "region_threshold",
    "region_area_cells",
    "box_xyxy_224",
    "highlight_path",
    "crop_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rewrite flat CSV/common-correct CSVs from human-study JSON artifacts.")
    parser.add_argument("--artifact_root", required=True)
    parser.add_argument("--manifest_csv", default="")
    return parser.parse_args()


def json_field(value: Any) -> str:
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value)
    return "" if value is None else str(value)


def write_flat_csv(rows: Sequence[Dict[str, Any]], artifact_root: Path) -> None:
    with (artifact_root / "top5_concepts_flat.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            for concept in row["top_concepts"]:
                flat = {key: "" for key in FIELDNAMES}
                for key in FIELDNAMES:
                    if key in row:
                        flat[key] = json_field(row[key])
                for key in FIELDNAMES:
                    if key in concept:
                        flat[key] = json_field(concept[key])
                writer.writerow(flat)


def write_common_csvs(summary: Dict[str, Any], artifact_root: Path, manifest_csv: Path) -> None:
    if not manifest_csv:
        return
    with manifest_csv.open("r", newline="", encoding="utf-8") as handle:
        manifest_rows = list(csv.DictReader(handle))
    if not manifest_rows:
        return
    by_trial = {int(row["trial_index"]): row for row in manifest_rows}
    for key, indices in summary.get("common_correct_trial_indices", {}).items():
        if "salf_official__AND__sg_seed1234_nec" not in key:
            continue
        out_path = artifact_root / f"{key}_trials.csv"
        with out_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0].keys()))
            writer.writeheader()
            for idx in indices:
                writer.writerow(by_trial[int(idx)])


def main() -> None:
    args = parse_args()
    artifact_root = Path(args.artifact_root)
    rows = json.loads((artifact_root / "top5_concepts_and_native_regions.json").read_text(encoding="utf-8"))
    summary = json.loads((artifact_root / "summary.json").read_text(encoding="utf-8"))
    write_flat_csv(rows, artifact_root)
    write_common_csvs(summary, artifact_root, Path(args.manifest_csv) if args.manifest_csv else Path())
    print(
        json.dumps(
            {
                "artifact_root": str(artifact_root),
                "model_image_rows": len(rows),
                "flat_rows": sum(len(row["top_concepts"]) for row in rows),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
