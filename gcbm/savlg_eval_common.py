import json
import os
from types import SimpleNamespace

import torch

from data import utils as data_utils


def _load_args(load_path: str, device: str, annotation_dir: str | None):
    with open(os.path.join(load_path, "args.txt"), "r") as f:
        payload = json.load(f)
    payload["device"] = device
    if annotation_dir is not None:
        payload["annotation_dir"] = annotation_dir
    return SimpleNamespace(**payload)


def _load_concepts(load_path: str, args) -> list[str]:
    concepts_path = os.path.join(load_path, "concepts.txt")
    if os.path.exists(concepts_path):
        with open(concepts_path, "r") as f:
            return [line.strip() for line in f if line.strip()]
    return data_utils.get_concepts(args.concept_set, getattr(args, "filter_set", None))


def _normalize_map(x: torch.Tensor) -> torch.Tensor:
    x = x.detach().float()
    x = x - x.min()
    denom = x.max().clamp_min(1e-6)
    return x / denom


def _union_boxes(boxes):
    boxes = [b for b in boxes if b is not None]
    if not boxes:
        return None
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    return [int(x1), int(y1), int(x2), int(y2)]
