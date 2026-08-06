#!/usr/bin/env python3
"""Split a PartImageNet++ manifest into the paper's 9:1 per-class train/val split."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--train_out", type=Path, required=True)
    parser.add_argument("--val_out", type=Path, required=True)
    parser.add_argument("--summary_out", type=Path, required=True)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mode",
        choices=["tail", "shuffle"],
        default="tail",
        help="tail preserves dataset order and holds out the final val_fraction per class; shuffle samples per class deterministically.",
    )
    args = parser.parse_args()

    rows = read_rows(args.manifest)
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_class[str(row["wnid"])].append(row)

    rng = random.Random(args.seed)
    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    per_class_counts: dict[str, dict[str, int]] = {}

    for wnid in sorted(by_class):
        class_rows = list(by_class[wnid])
        if args.mode == "shuffle":
            rng.shuffle(class_rows)
        n_val = int(round(len(class_rows) * args.val_fraction))
        if args.val_fraction > 0 and n_val == 0 and len(class_rows) > 1:
            n_val = 1
        n_val = min(max(n_val, 0), len(class_rows))
        class_train = class_rows[:-n_val] if n_val else class_rows
        class_val = class_rows[-n_val:] if n_val else []
        train_rows.extend(class_train)
        val_rows.extend(class_val)
        per_class_counts[wnid] = {"train": len(class_train), "val": len(class_val)}

    write_rows(args.train_out, train_rows)
    write_rows(args.val_out, val_rows)
    summary = {
        "source_manifest": str(args.manifest),
        "mode": args.mode,
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "num_classes": len(by_class),
        "num_source_rows": len(rows),
        "num_train_rows": len(train_rows),
        "num_val_rows": len(val_rows),
        "per_class_counts": per_class_counts,
    }
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in summary if k != "per_class_counts"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
