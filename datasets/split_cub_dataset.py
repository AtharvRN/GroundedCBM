#!/usr/bin/env python3
"""Create ImageFolder-style CUB train/test directories from CUB_200_2011."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from PIL import Image, UnidentifiedImageError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split CUB_200_2011 into CUB/train and CUB/test directories using the official split file."
    )
    parser.add_argument(
        "--cub_root",
        default="CUB_200_2011",
        help="Directory containing images.txt, train_test_split.txt, and images/.",
    )
    parser.add_argument(
        "--output_root",
        default="CUB",
        help="Output directory. Creates train/ and test/ subdirectories under this path.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove existing output_root before writing the split.",
    )
    return parser.parse_args()


def is_rgb_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            return image.mode == "RGB" or image.convert("RGB").mode == "RGB"
    except (UnidentifiedImageError, OSError):
        return False


def copy_split(cub_root: Path, output_root: Path, overwrite: bool = False) -> tuple[int, int]:
    images_txt = cub_root / "images.txt"
    split_txt = cub_root / "train_test_split.txt"
    images_dir = cub_root / "images"
    if not images_txt.exists() or not split_txt.exists() or not images_dir.is_dir():
        raise FileNotFoundError(
            f"Expected CUB files at {cub_root}: images.txt, train_test_split.txt, images/"
        )

    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"{output_root} already exists; pass --overwrite to replace it")
        shutil.rmtree(output_root)

    train_root = output_root / "train"
    test_root = output_root / "test"
    train_root.mkdir(parents=True)
    test_root.mkdir(parents=True)

    image_lines = images_txt.read_text(encoding="utf-8").splitlines()
    split_lines = split_txt.read_text(encoding="utf-8").splitlines()
    if len(image_lines) != len(split_lines):
        raise ValueError("images.txt and train_test_split.txt have different lengths")

    train_count = 0
    test_count = 0
    for image_line, split_line in zip(image_lines, split_lines):
        _image_id, rel_path = image_line.strip().split(maxsplit=1)
        _split_id, is_train = split_line.strip().split(maxsplit=1)
        src = images_dir / rel_path
        if not is_rgb_image(src):
            continue
        dst_root = train_root if int(is_train) == 1 else test_root
        dst = dst_root / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        if int(is_train) == 1:
            train_count += 1
        else:
            test_count += 1

    return train_count, test_count


def main() -> None:
    args = parse_args()
    train_count, test_count = copy_split(
        Path(args.cub_root),
        Path(args.output_root),
        overwrite=args.overwrite,
    )
    print(f"Created {args.output_root}/train with {train_count} RGB images")
    print(f"Created {args.output_root}/test with {test_count} RGB images")


if __name__ == "__main__":
    main()
