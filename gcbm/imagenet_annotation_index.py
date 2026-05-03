from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from torchvision.datasets import ImageFolder


def resolve_val_annotation_dir(annotation_dir: Path) -> Path:
    if (annotation_dir / "0.json").is_file():
        return annotation_dir.resolve()
    candidate = annotation_dir / "imagenet_val"
    if (candidate / "0.json").is_file():
        return candidate.resolve()
    raise FileNotFoundError(f"Could not find imagenet_val/0.json under {annotation_dir}")


def build_filename_to_annotation_path(
    annotation_val_dir: Path,
    annotation_val_root: Optional[Path],
) -> Optional[Dict[str, Path]]:
    if annotation_val_root is None:
        return None
    root = annotation_val_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"annotation ImageFolder root does not exist: {root}")

    dataset = ImageFolder(str(root))
    mapping: Dict[str, Path] = {}
    for dataset_index, (image_path, _target) in enumerate(dataset.samples):
        ann_path = annotation_val_dir / f"{dataset_index}.json"
        if ann_path.is_file():
            mapping[Path(image_path).name] = ann_path
    if not mapping:
        raise RuntimeError(
            f"no annotation files were mapped from annotation_dir={annotation_val_dir} using annotation_val_root={root}"
        )
    return mapping


def load_annotation_payload(
    annotation_val_dir: Path,
    image_index_1based: int,
    image_name: str,
    filename_to_annotation_path: Optional[Dict[str, Path]] = None,
) -> List[dict]:
    if filename_to_annotation_path is not None:
        path = filename_to_annotation_path.get(Path(image_name).name)
        if path is None:
            return []
    else:
        path = annotation_val_dir / f"{image_index_1based - 1}.json"
        if not path.is_file():
            return []
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        return payload
    return payload.get("concepts", [])
