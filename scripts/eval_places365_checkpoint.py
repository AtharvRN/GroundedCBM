#!/usr/bin/env python
"""Direct Places365 classifier eval for a saved VLG-CBM checkpoint.

This evaluates backbone -> CBL -> normalization -> final layer without
requiring concept annotations and without retraining the sparse layer.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.cbm import Backbone, ConceptLayer, FinalLayer, NormalizationLayer


def load_concept_layer_from_weights(checkpoint: Path, encoder_dim: int, device: str):
    state_dict = torch.load(checkpoint / "cbl.pt", map_location=device)
    out_features = int(state_dict["model.0.weight"].shape[0])
    hidden_keys = [key for key in state_dict if key.startswith("model.2.")]
    num_hidden = 1 if hidden_keys else 0
    model = ConceptLayer(
        encoder_dim,
        out_features,
        num_hidden=num_hidden,
        device=device,
    )
    model.load_state_dict(state_dict)
    return model


def load_final_layer_from_weights(checkpoint: Path, device: str):
    state_dict = torch.load(checkpoint / "final.pt", map_location=device)
    out_features, in_features = state_dict["weight"].shape
    model = FinalLayer(int(in_features), int(out_features), device=device)
    model.load_state_dict(state_dict)
    return model


class Places365ValDataset(Dataset):
    def __init__(self, root: Path, transform, image_dir: Path | None = None):
        self.root = root
        self.transform = transform
        list_path = root / "places365_val.txt"
        image_dir = image_dir or (root / "val_256")
        if not list_path.is_file():
            raise FileNotFoundError(f"Missing Places365 val filelist: {list_path}")

        self.rows = []
        for line in list_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rel_path, label = line.split()[:2]
            self.rows.append((image_dir / rel_path, int(label)))

        print(
            f"[places365] rows={len(self.rows)} image_dir={image_dir}",
            flush=True,
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        image_path, label = self.rows[idx]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
        return self.transform(image), label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--places365-root", required=True, type=Path)
    parser.add_argument("--image-dir", type=Path, default=None)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    with (args.checkpoint / "args.txt").open("r", encoding="utf-8") as handle:
        ckpt_args = json.load(handle)

    print(f"[places365] checkpoint={args.checkpoint}", flush=True)
    print(f"[places365] root={args.places365_root}", flush=True)
    print(
        f"[places365] backbone={ckpt_args['backbone']} feature_layer={ckpt_args['feature_layer']} device={device}",
        flush=True,
    )

    backbone = Backbone(ckpt_args["backbone"], ckpt_args["feature_layer"], device=device).eval()
    cbl = load_concept_layer_from_weights(
        args.checkpoint,
        encoder_dim=backbone.output_dim,
        device=device,
    ).eval()
    normalization = NormalizationLayer.from_pretrained(str(args.checkpoint), device=device).eval()
    final = load_final_layer_from_weights(args.checkpoint, device=device).eval()

    dataset = Places365ValDataset(args.places365_root, backbone.preprocess, args.image_dir)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    top1 = 0
    top5 = 0
    total = 0
    start = time.time()
    with torch.no_grad():
        for step, (images, labels) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            features = backbone(images)
            logits = final(normalization(cbl(features)))
            preds = logits.topk(5, dim=1).indices
            total += labels.numel()
            top1 += (preds[:, 0] == labels).sum().item()
            top5 += (preds == labels[:, None]).any(dim=1).sum().item()
            if step == 1 or step % 10 == 0:
                elapsed = time.time() - start
                print(
                    f"[places365] step={step}/{len(loader)} n={total} "
                    f"top1={top1 / total:.6f} top5={top5 / total:.6f} "
                    f"ips={total / max(elapsed, 1e-9):.1f}",
                    flush=True,
                )

    elapsed = time.time() - start
    result = {
        "checkpoint": str(args.checkpoint),
        "places365_root": str(args.places365_root),
        "n": total,
        "top1": top1 / total,
        "top5": top5 / total,
        "elapsed_sec": elapsed,
        "images_per_second": total / elapsed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("[places365] result=" + json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
