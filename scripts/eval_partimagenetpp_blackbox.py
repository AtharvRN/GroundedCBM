#!/usr/bin/env python3
"""Evaluate pretrained black-box classifiers on PartImageNet++ manifests."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from PIL import Image, ImageFile, UnidentifiedImageError

try:
    from torchvision.models import ResNet50_Weights, resnet50
except RuntimeError as exc:
    if "torchvision::nms" not in str(exc):
        raise
    _tv_lib = torch.library.Library("torchvision", "DEF")
    _tv_lib.define("nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor")
    from torchvision.models import ResNet50_Weights, resnet50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="resnet50", choices=["resnet50"])
    parser.add_argument("--weights", default="v1", choices=["v1", "v2"])
    parser.add_argument("--partimagenetpp_val_manifest", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=1000)
    return parser.parse_args()


def safe_pil_loader(path: str) -> Image.Image:
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    try:
        with Image.open(path) as img:
            return img.convert("RGB")
    except (UnidentifiedImageError, OSError, FileNotFoundError):
        return Image.new("RGB", (224, 224), color=(0, 0, 0))


class PartImageNetPPManifestDataset(torch.utils.data.Dataset):
    def __init__(self, manifest: Path, transform=None):
        self.manifest = manifest
        self.transform = transform
        self.rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not self.rows:
            raise RuntimeError(f"empty manifest: {manifest}")

        class_meta: Dict[str, str] = {}
        for row in self.rows:
            wnid = str(row["wnid"])
            class_meta.setdefault(wnid, str(row.get("object_name") or wnid))
        self.wnids = sorted(class_meta)
        self.classes = [class_meta[wnid] for wnid in self.wnids]
        self.class_to_idx = {wnid: idx for idx, wnid in enumerate(self.wnids)}
        self.targets = [self.class_to_idx[str(row["wnid"])] for row in self.rows]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        image = safe_pil_loader(str(row["image"]))
        if self.transform is not None:
            image = self.transform(image)
        return image, self.targets[idx]


def build_model_and_preprocess(args: argparse.Namespace):
    if args.model != "resnet50":
        raise ValueError(f"unsupported model: {args.model}")
    weights = ResNet50_Weights.IMAGENET1K_V2 if args.weights == "v2" else ResNet50_Weights.IMAGENET1K_V1
    model = resnet50(weights=weights).to(args.device)
    return model, weights.transforms()


def topk_counts(logits: torch.Tensor, labels: torch.Tensor) -> tuple[int, int]:
    top1 = logits.argmax(dim=1)
    top5 = logits.topk(k=min(5, logits.shape[1]), dim=1).indices
    correct1 = int(top1.eq(labels).sum().item())
    correct5 = int(top5.eq(labels[:, None]).any(dim=1).sum().item())
    return correct1, correct5


def main() -> None:
    args = parse_args()
    manifest = Path(args.partimagenetpp_val_manifest).resolve()
    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    args.device = str(device)

    model, preprocess = build_model_and_preprocess(args)
    model.eval()
    dataset = PartImageNetPPManifestDataset(manifest, transform=preprocess)
    if int(args.max_images) > 0:
        dataset = Subset(dataset, list(range(min(int(args.max_images), len(dataset)))))

    loader_kwargs: Dict[str, Any] = {
        "dataset": dataset,
        "batch_size": int(args.batch_size),
        "shuffle": False,
        "num_workers": int(args.num_workers),
        "pin_memory": device.type == "cuda",
        "drop_last": False,
    }
    if int(args.num_workers) > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    loader = DataLoader(**loader_kwargs)

    correct1 = 0
    correct5 = 0
    n = 0
    start = time.perf_counter()
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f"{args.model}-{args.weights}"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()
            logits = model(images)
            c1, c5 = topk_counts(logits, labels)
            correct1 += c1
            correct5 += c5
            n += int(labels.numel())
            if int(args.log_every) > 0 and n % int(args.log_every) == 0:
                elapsed = time.perf_counter() - start
                print(
                    f"[blackbox] n={n} top1={correct1 / max(n, 1):.4f} "
                    f"top5={correct5 / max(n, 1):.4f} ips={n / max(elapsed, 1e-6):.2f}",
                    flush=True,
                )

    elapsed = time.perf_counter() - start
    classes = getattr(dataset, "classes", getattr(getattr(dataset, "dataset", None), "classes", []))
    payload = {
        "dataset": "partimagenetpp_val",
        "model": args.model,
        "weights": args.weights,
        "manifest": str(manifest),
        "n": int(n),
        "classes": int(len(classes)),
        "top1": float(correct1 / max(n, 1)),
        "top5": float(correct5 / max(n, 1)),
        "correct1": int(correct1),
        "correct5": int(correct5),
        "elapsed_sec": float(elapsed),
        "images_per_sec": float(n / max(elapsed, 1e-6)),
        "device": str(device),
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
