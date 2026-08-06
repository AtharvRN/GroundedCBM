#!/usr/bin/env python3
"""Materialize PartImageNet++ GDINO JSONL annotations into split-local JSON files."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any, Iterable

from tqdm import tqdm


def read_manifest_index(path: Path) -> dict[str, int]:
    mapping: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            file_name = str(row.get("file_name") or "")
            if not file_name:
                raise ValueError(f"Manifest row {idx} in {path} has no file_name")
            if file_name in mapping:
                raise ValueError(f"Duplicate file_name in {path}: {file_name}")
            mapping[file_name] = idx
    return mapping


def iter_jsonl(path: Path) -> Iterable[Any]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_annotation(path: Path, payload: Any, force: bool) -> bool:
    if path.exists() and not force:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_manifest", type=Path, required=True)
    parser.add_argument("--val_manifest", type=Path, required=True)
    parser.add_argument("--gdino_jsonl", type=Path, nargs="+", required=True)
    parser.add_argument("--out_root", type=Path, required=True)
    parser.add_argument("--dataset", default="partimagenetpp")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    train_index = read_manifest_index(args.train_manifest)
    val_index = read_manifest_index(args.val_manifest)
    if set(train_index).intersection(val_index):
        raise ValueError("Train and val manifests overlap by file_name")

    split_indexes = {
        "train": train_index,
        "val": val_index,
    }
    counts = {"train": 0, "val": 0, "unmatched": 0, "written": 0}
    seen = {"train": set(), "val": set()}

    for source in args.gdino_jsonl:
        for payload in tqdm(iter_jsonl(source), desc=f"materialize {source.name}"):
            if not isinstance(payload, list) or not payload:
                raise ValueError(f"Expected non-empty list annotation payload in {source}")
            meta = payload[0]
            file_name = str(meta.get("file_name") or "")
            matched = False
            for split_name, index in split_indexes.items():
                sample_idx = index.get(file_name)
                if sample_idx is None:
                    continue
                matched = True
                seen[split_name].add(file_name)
                out_path = args.out_root / f"{args.dataset}_{split_name}" / f"{sample_idx}.json"
                if write_annotation(out_path, payload, args.force):
                    counts["written"] += 1
                counts[split_name] += 1
                break
            if not matched:
                counts["unmatched"] += 1

    missing = {
        "train": len(train_index) - len(seen["train"]),
        "val": len(val_index) - len(seen["val"]),
    }
    summary = {
        "train_manifest": str(args.train_manifest),
        "val_manifest": str(args.val_manifest),
        "gdino_jsonl": [str(path) for path in args.gdino_jsonl],
        "out_root": str(args.out_root),
        "counts": counts,
        "missing": missing,
    }
    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "materialize_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if missing["train"] or missing["val"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
