#!/usr/bin/env python3
"""Evaluate saved sparse final-layer weights on cached concept features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature_dir", required=True, type=Path)
    parser.add_argument("--weight_dir", required=True, type=Path)
    parser.add_argument("--normalization", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--nec_values", default="5,10,15,20,25,30")
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def parse_nec_values(raw: str) -> List[int]:
    return [int(item.strip()) for item in str(raw).split(",") if item.strip()]


def load_memmap(path: Path) -> np.memmap:
    shape_path = path.with_suffix(path.suffix + ".shape.json")
    if shape_path.exists():
        shape = tuple(json.loads(shape_path.read_text())["shape"])
        dtype = np.dtype(json.loads(shape_path.read_text()).get("dtype", "float16"))
        return np.memmap(path, mode="r", dtype=dtype, shape=shape)
    array = np.load(path, mmap_mode="r")
    return array


def topk_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> tuple[int, int]:
    pred1 = logits.argmax(dim=1)
    top5 = logits.topk(k=min(5, logits.shape[1]), dim=1).indices
    correct1 = int(pred1.eq(targets).sum().item())
    correct5 = int(top5.eq(targets[:, None]).any(dim=1).sum().item())
    return correct1, correct5


def main() -> None:
    args = parse_args()
    feature_dir = args.feature_dir.resolve()
    weight_dir = args.weight_dir.resolve()
    features = load_memmap(feature_dir / "val_features.npy")
    targets = load_memmap(feature_dir / "val_targets.npy")
    norm = torch.load(args.normalization, map_location="cpu")
    mean = norm["mean"].float()
    std = norm["std"].float().clamp_min(1e-6)
    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    mean = mean.to(device)
    std = std.to(device)

    results: List[Dict[str, Any]] = []
    n = int(features.shape[0])
    for nec in parse_nec_values(args.nec_values):
        weight = torch.load(weight_dir / f"W_g@NEC={nec}.pt", map_location="cpu").float().to(device)
        bias = torch.load(weight_dir / f"b_g@NEC={nec}.pt", map_location="cpu").float().to(device)
        correct1 = 0
        correct5 = 0
        for start in range(0, n, int(args.batch_size)):
            end = min(start + int(args.batch_size), n)
            x = torch.from_numpy(np.asarray(features[start:end])).float().to(device)
            y = torch.from_numpy(np.asarray(targets[start:end])).long().to(device)
            x = (x - mean) / std
            logits = x @ weight.t() + bias
            c1, c5 = topk_accuracy(logits, y)
            correct1 += c1
            correct5 += c5
        nnz = int((weight.abs() > 1e-5).sum().item())
        total = int(weight.numel())
        results.append(
            {
                "nec": int(nec),
                "n": n,
                "top1": float(correct1 / max(n, 1)),
                "top5": float(correct5 / max(n, 1)),
                "nnz": nnz,
                "total": total,
                "weight_sparsity": float(1.0 - nnz / max(total, 1)),
                "effective_nec": float(nnz / max(weight.shape[0], 1)),
            }
        )

    payload = {
        "feature_dir": str(feature_dir),
        "weight_dir": str(weight_dir),
        "normalization": str(args.normalization.resolve()),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
