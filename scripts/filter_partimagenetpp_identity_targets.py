#!/usr/bin/env python3
"""Remove generic targets that are identical to the image's object class.

PartImageNet++ includes a few whole-object categories (for example, a category
named ``bakery`` for a bakery image).  Those are valid source annotations but
are not valid generic *part* targets after class-specific labels are collapsed.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary_out", type=Path, required=True)
    parser.add_argument(
        "--target_field",
        choices=["segmentations", "boxes"],
        default="segmentations",
    )
    return parser.parse_args()


def normalize(value: Any) -> str:
    return " ".join(str(value).lower().replace("_", " ").split())


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    args = parse_args()
    rows = list(iter_jsonl(args.input))
    if not rows:
        raise RuntimeError(f"Empty input: {args.input}")

    removed_by_concept: Counter[str] = Counter()
    input_targets = 0
    kept_targets = 0
    rows_changed = 0
    for row in rows:
        targets = row.get(args.target_field, {})
        if not isinstance(targets, dict):
            raise TypeError(f"Expected {args.target_field} mapping at row={row.get('row_index')}")
        input_targets += len(targets)
        object_name = normalize(row.get("object_name", ""))
        kept = {
            concept: value
            for concept, value in targets.items()
            if normalize(concept) != object_name
        }
        removed = set(targets) - set(kept)
        if removed:
            rows_changed += 1
            removed_by_concept.update(str(concept) for concept in removed)
        kept_targets += len(kept)
        row[args.target_field] = kept

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "target_field": args.target_field,
        "rows": len(rows),
        "rows_changed": rows_changed,
        "input_image_concept_targets": input_targets,
        "kept_image_concept_targets": kept_targets,
        "removed_image_concept_targets": input_targets - kept_targets,
        "removed_concepts": len(removed_by_concept),
        "removed_by_concept": dict(sorted(removed_by_concept.items())),
    }
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
