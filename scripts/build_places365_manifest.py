from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_categories(path: Path) -> dict[int, str]:
    categories: dict[int, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2 and parts[-1].lstrip("-").isdigit():
                class_id = int(parts[-1])
                raw_name = " ".join(parts[:-1])
            else:
                class_id = line_number
                raw_name = parts[0]
            categories[class_id] = raw_name.strip("/").replace("/", "_")
    return categories


def write_manifest(
    *,
    split_file: Path,
    image_root: Path,
    categories: dict[int, str],
    output_path: Path,
    val_split: bool,
) -> int:
    count = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with split_file.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for row_index, line in enumerate(src):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            rel_path = parts[0].lstrip("/")
            class_id = int(parts[1]) if len(parts) > 1 else -1
            if val_split and not (image_root / rel_path).exists():
                rel_path = Path(rel_path).name
            image_path = image_root / rel_path
            payload = {
                "path": str(image_path),
                "class_id": class_id,
                "class_name": categories.get(class_id, str(class_id)),
                "sample_index": row_index,
                "annotation_index": row_index,
            }
            dst.write(json.dumps(payload) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Build JSONL manifests for Places365 fast SG-CBM training.")
    parser.add_argument("--places365_root", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--train_image_root", default="")
    parser.add_argument("--val_image_root", default="")
    args = parser.parse_args()

    root = args.places365_root
    train_image_root = Path(args.train_image_root) if args.train_image_root else root / "data_256"
    val_image_root = Path(args.val_image_root) if args.val_image_root else root / "val_256" / "val_256"
    if not val_image_root.is_dir():
        val_image_root = root / "val_256"

    categories = parse_categories(root / "categories_places365.txt")
    train_count = write_manifest(
        split_file=root / "places365_train_standard.txt",
        image_root=train_image_root,
        categories=categories,
        output_path=args.output_dir / "places365_train_manifest.jsonl",
        val_split=False,
    )
    val_count = write_manifest(
        split_file=root / "places365_val.txt",
        image_root=val_image_root,
        categories=categories,
        output_path=args.output_dir / "places365_val_manifest.jsonl",
        val_split=True,
    )
    print(
        json.dumps(
            {
                "train_manifest": str(args.output_dir / "places365_train_manifest.jsonl"),
                "train_count": train_count,
                "val_manifest": str(args.output_dir / "places365_val_manifest.jsonl"),
                "val_count": val_count,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
