#!/usr/bin/env python3
"""Convert PartImageNet++ manifests to the GCBM ImageNet-style JSONL schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _iter_rows(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _build_class_map(paths: List[Path]) -> Dict[str, int]:
    wnids = sorted({str(row["wnid"]) for path in paths for row in _iter_rows(path)})
    return {wnid: idx for idx, wnid in enumerate(wnids)}


def _convert_manifest(src: Path, dst: Path, class_to_idx: Dict[str, int]) -> Dict[str, Any]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    class_names: Dict[str, str] = {}
    with src.open("r", encoding="utf-8") as in_handle, dst.open("w", encoding="utf-8") as out_handle:
        for row_index, line in enumerate(in_handle):
            if not line.strip():
                continue
            payload = json.loads(line)
            wnid = str(payload["wnid"])
            class_name = str(payload.get("object_name") or wnid)
            class_names.setdefault(wnid, class_name)
            out_payload = dict(payload)
            out_payload["path"] = str(payload.get("path") or payload["image"])
            out_payload["class_id"] = int(class_to_idx[wnid])
            out_payload["class_name"] = class_name
            out_payload["sample_index"] = int(payload.get("sample_index", row_index))
            out_payload["annotation_index"] = int(payload.get("annotation_index", row_index))
            out_handle.write(json.dumps(out_payload, sort_keys=True) + "\n")
            rows_written += 1
    return {
        "source": str(src),
        "output": str(dst),
        "rows": rows_written,
        "classes_present": len(class_names),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_in", type=Path, required=True)
    parser.add_argument("--val_in", type=Path, required=True)
    parser.add_argument("--train_out", type=Path, required=True)
    parser.add_argument("--val_out", type=Path, required=True)
    parser.add_argument("--summary_out", type=Path, default=None)
    args = parser.parse_args()

    class_to_idx = _build_class_map([args.train_in, args.val_in])
    summary = {
        "num_classes": len(class_to_idx),
        "class_to_idx": class_to_idx,
        "train": _convert_manifest(args.train_in, args.train_out, class_to_idx),
        "val": _convert_manifest(args.val_in, args.val_out, class_to_idx),
    }
    if args.summary_out is not None:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("num_classes", "train", "val")}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
