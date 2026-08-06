#!/usr/bin/env python3
"""Evaluate PartImageNet++ ImageNet-classification accuracy at NEC truncations."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gcbm.sparse import threshold_weight_truncation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", required=True, choices=["salf_cbm", "vlg_cbm"])
    parser.add_argument("--load_path", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--feature_dir", default="", type=Path)
    parser.add_argument("--partimagenetpp_train_manifest", default="")
    parser.add_argument("--partimagenetpp_val_manifest", default="")
    parser.add_argument("--nec_values", default="5,10,15,20,25,30")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=1000)
    parser.add_argument("--save_truncated_weights", action="store_true")
    return parser.parse_args()


def parse_nec_values(raw: str) -> List[int]:
    values = [int(item.strip()) for item in str(raw).split(",") if item.strip()]
    if not values:
        raise ValueError("--nec_values must contain at least one integer")
    return values


def configure_partimagenetpp_env(args: argparse.Namespace) -> None:
    if args.partimagenetpp_train_manifest:
        os.environ["PARTIMAGENETPP_TRAIN_MANIFEST"] = str(Path(args.partimagenetpp_train_manifest).resolve())
    if args.partimagenetpp_val_manifest:
        os.environ["PARTIMAGENETPP_VAL_MANIFEST"] = str(Path(args.partimagenetpp_val_manifest).resolve())


def load_tensor_payload(path: Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, torch.nn.Parameter):
        payload = payload.detach()
    if not torch.is_tensor(payload):
        raise TypeError(f"expected tensor payload at {path}, got {type(payload)!r}")
    return payload.float()


def load_final_layer(load_path: Path) -> Tuple[torch.Tensor, torch.Tensor, str]:
    if (load_path / "W_g.pt").exists() and (load_path / "b_g.pt").exists():
        return load_tensor_payload(load_path / "W_g.pt"), load_tensor_payload(load_path / "b_g.pt"), "W_g.pt"
    for name in ("final_layer_glm_saga.pt", "final_layer_dense.pt"):
        path = load_path / name
        if path.exists():
            payload = torch.load(path, map_location="cpu")
            if isinstance(payload, dict) and "weight" in payload and "bias" in payload:
                return payload["weight"].float(), payload["bias"].float(), name
            if isinstance(payload, dict) and isinstance(payload.get("best"), dict):
                best = payload["best"]
                if "weight" in best and "bias" in best:
                    return best["weight"].float(), best["bias"].float(), name
    raise FileNotFoundError(f"no supported final-layer artifact found under {load_path}")


def load_normalization(load_path: Path) -> Tuple[torch.Tensor, torch.Tensor, str]:
    final_norm = load_path / "final_layer_normalization.pt"
    if final_norm.exists():
        payload = torch.load(final_norm, map_location="cpu")
        return payload["mean"].float().reshape(1, -1), payload["std"].float().reshape(1, -1).clamp_min(1e-6), str(final_norm)
    mean_path = load_path / "proj_mean.pt"
    std_path = load_path / "proj_std.pt"
    if mean_path.exists() and std_path.exists():
        mean = load_tensor_payload(mean_path).reshape(1, -1)
        std = load_tensor_payload(std_path).reshape(1, -1).clamp_min(1e-6)
        return mean, std, f"{mean_path},{std_path}"
    raise FileNotFoundError(f"no supported normalization artifact found under {load_path}")


def load_memmap(path: Path) -> np.ndarray:
    shape_path = path.with_suffix(path.suffix + ".shape.json")
    if shape_path.exists():
        meta = json.loads(shape_path.read_text())
        return np.memmap(path, mode="r", dtype=np.dtype(meta.get("dtype", "float16")), shape=tuple(meta["shape"]))
    return np.load(path, mmap_mode="r")


def load_cached_features(feature_dir: Path, max_images: int) -> Tuple[np.ndarray, np.ndarray]:
    features = load_memmap(feature_dir / "val_features.npy")
    targets = load_memmap(feature_dir / "val_targets.npy")
    if max_images > 0:
        features = features[:max_images]
        targets = targets[:max_images]
    return features, targets


def extract_salf_val_features(args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray]:
    from data import utils as data_utils
    from methods.salf import SpatialBackbone, build_spatial_concept_layer, pool_salf_maps

    run_args = argparse.Namespace(**json.loads((args.load_path / "args.txt").read_text()))
    run_args.device = args.device
    run_args.cbl_batch_size = int(args.batch_size)
    run_args.num_workers = int(args.num_workers)

    concepts = [line.strip() for line in (args.load_path / "concepts.txt").read_text().splitlines() if line.strip()]
    backbone = SpatialBackbone(
        run_args.backbone,
        device=run_args.device,
        checkpoint_path=getattr(run_args, "backbone_checkpoint", ""),
    )
    state_dict = torch.load(args.load_path / "concept_layer.pt", map_location=run_args.device)
    if "weight" in state_dict and state_dict["weight"].ndim == 4:
        concept_layer = torch.nn.Conv2d(
            int(state_dict["weight"].shape[1]),
            int(state_dict["weight"].shape[0]),
            kernel_size=1,
            bias=("bias" in state_dict),
        ).to(run_args.device)
    elif "spatial_layer.weight" in state_dict:
        concept_layer = build_spatial_concept_layer(
            run_args,
            backbone.output_dim,
            int(state_dict["spatial_layer.weight"].shape[0]),
            is_vit=getattr(backbone, "is_vit", False),
        )
    else:
        raise ValueError(f"Unsupported SALF concept layer format at {args.load_path / 'concept_layer.pt'}")
    concept_layer.load_state_dict(state_dict)
    backbone.eval()
    concept_layer.eval()

    dataset = data_utils.get_data("partimagenetpp_val", preprocess=backbone.preprocess)
    if int(args.max_images) > 0:
        dataset = Subset(dataset, list(range(min(int(args.max_images), len(dataset)))))
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=True,
        **({"prefetch_factor": 2, "persistent_workers": True} if int(args.num_workers) > 0 else {}),
    )
    features: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []
    seen = 0
    start = time.perf_counter()
    with torch.no_grad():
        for images, labels in loader:
            maps = concept_layer(backbone(images.to(run_args.device, non_blocking=True)))
            if isinstance(maps, tuple):
                maps = maps[0]
            scores = pool_salf_maps(run_args, maps) if maps.ndim > 2 else maps
            scores = scores.detach().cpu().float()
            if int(scores.shape[1]) > len(concepts):
                scores = scores[:, : len(concepts)]
            features.append(scores)
            targets.append(labels.detach().cpu().long())
            seen += int(labels.numel())
            if int(args.log_every) > 0 and seen % int(args.log_every) == 0:
                elapsed = time.perf_counter() - start
                print(f"[nec:salf_features] n={seen} ips={seen / max(elapsed, 1e-6):.2f}", flush=True)
    return torch.cat(features, dim=0).numpy(), torch.cat(targets, dim=0).numpy()


def extract_vlg_val_features(args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray]:
    import dataclasses

    from torchvision import transforms

    from data import utils as data_utils
    from gcbm.imagenet_config import Config
    from gcbm.imagenet_models import build_model
    from gcbm.runtime import configure_runtime
    from gcbm.training_utils import prepare_images

    payload = json.loads((args.load_path / "config.json").read_text())
    valid_fields = {field.name for field in dataclasses.fields(Config)}
    payload = {key: value for key, value in payload.items() if key in valid_fields}
    payload["device"] = args.device
    cfg = Config(**payload)
    configure_runtime(cfg)
    concepts = [line.strip() for line in (args.load_path / "concepts.txt").read_text().splitlines() if line.strip()]
    backbone, concept_layer = build_model(cfg, n_concepts=len(concepts))
    concept_layer.load_state_dict(torch.load(args.load_path / "concept_head_best.pt", map_location=cfg.device))
    backbone.eval()
    concept_layer.eval()

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(int(cfg.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    dataset = data_utils.get_data("partimagenetpp_val", preprocess=transform)
    if int(args.max_images) > 0:
        dataset = Subset(dataset, list(range(min(int(args.max_images), len(dataset)))))
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=True,
        **({"prefetch_factor": 2, "persistent_workers": True} if int(args.num_workers) > 0 else {}),
    )
    features: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []
    seen = 0
    start = time.perf_counter()
    with torch.no_grad():
        for images, labels in loader:
            outputs = concept_layer(backbone(prepare_images(images, cfg)))
            features.append(outputs["final_logits"].detach().cpu().float())
            targets.append(labels.detach().cpu().long())
            seen += int(labels.numel())
            if int(args.log_every) > 0 and seen % int(args.log_every) == 0:
                elapsed = time.perf_counter() - start
                print(f"[nec:vlg_features] n={seen} ips={seen / max(elapsed, 1e-6):.2f}", flush=True)
    return torch.cat(features, dim=0).numpy(), torch.cat(targets, dim=0).numpy()


def build_sweep(weight: torch.Tensor, bias: torch.Tensor, nec_values: Sequence[int], save_dir: Path | None) -> List[Dict[str, Any]]:
    num_concepts = int(weight.shape[1])
    sweep = []
    for nec in nec_values:
        truncated = threshold_weight_truncation(weight, float(nec) / float(num_concepts))
        nnz = int((truncated.abs() > 1e-5).sum().item())
        if save_dir is not None:
            torch.save(truncated.cpu(), save_dir / f"W_g@NEC={int(nec)}.pt")
            torch.save(bias.cpu(), save_dir / f"b_g@NEC={int(nec)}.pt")
        sweep.append({"nec": int(nec), "weight": truncated, "bias": bias, "nnz": nnz, "total": int(truncated.numel())})
    return sweep


def topk_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> Tuple[int, int]:
    pred1 = logits.argmax(dim=1)
    top5 = logits.topk(k=min(5, logits.shape[1]), dim=1).indices
    return int(pred1.eq(targets).sum().item()), int(top5.eq(targets[:, None]).any(dim=1).sum().item())


def evaluate_features(
    features: np.ndarray,
    targets: np.ndarray,
    sweep: Sequence[Dict[str, Any]],
    mean: torch.Tensor,
    std: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> List[Dict[str, Any]]:
    mean = mean.to(device)
    std = std.to(device)
    results: List[Dict[str, Any]] = []
    n = int(features.shape[0])
    y_all = np.asarray(targets).astype(np.int64)
    for item in sweep:
        weight = item["weight"].to(device).float()
        bias = item["bias"].to(device).float()
        correct1 = 0
        correct5 = 0
        for start in range(0, n, int(batch_size)):
            end = min(start + int(batch_size), n)
            x = torch.from_numpy(np.asarray(features[start:end])).float().to(device)
            y = torch.from_numpy(y_all[start:end]).long().to(device)
            x = (x - mean) / std
            logits = x @ weight.t() + bias
            c1, c5 = topk_accuracy(logits, y)
            correct1 += c1
            correct5 += c5
        results.append(
            {
                "nec": int(item["nec"]),
                "n": n,
                "top1": float(correct1 / max(n, 1)),
                "top5": float(correct5 / max(n, 1)),
                "nnz": int(item["nnz"]),
                "total": int(item["total"]),
                "effective_nec": float(item["nnz"] / max(int(weight.shape[0]), 1)),
                "weight_sparsity": float(1.0 - item["nnz"] / max(int(item["total"]), 1)),
            }
        )
    return results


def main() -> None:
    args = parse_args()
    configure_partimagenetpp_env(args)
    load_path = args.load_path.resolve()
    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")

    weight, bias, final_layer_source = load_final_layer(load_path)
    mean, std, normalization_source = load_normalization(load_path)
    sweep = build_sweep(weight, bias, parse_nec_values(args.nec_values), load_path if args.save_truncated_weights else None)

    feature_dir_raw = str(args.feature_dir or "")
    if feature_dir_raw and feature_dir_raw != ".":
        features, targets = load_cached_features(Path(feature_dir_raw).resolve(), int(args.max_images))
        feature_source = str(Path(feature_dir_raw).resolve())
    elif args.model_name == "salf_cbm":
        features, targets = extract_salf_val_features(args)
        feature_source = "salf_val_forward"
    elif args.model_name == "vlg_cbm":
        features, targets = extract_vlg_val_features(args)
        feature_source = "vlg_val_forward"
    else:
        raise ValueError("--feature_dir is required unless --model_name is salf_cbm or vlg_cbm")

    start = time.perf_counter()
    results = evaluate_features(features, targets, sweep, mean, std, int(args.batch_size), device)
    payload = {
        "dataset": "partimagenetpp",
        "model_name": args.model_name,
        "load_path": str(load_path),
        "feature_source": feature_source,
        "final_layer_source": final_layer_source,
        "normalization_source": normalization_source,
        "nec_values": parse_nec_values(args.nec_values),
        "elapsed_eval_sec": time.perf_counter() - start,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
