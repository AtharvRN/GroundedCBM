import argparse
import json
import math
import sys
import time
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from glm_saga.elasticnet import glm_saga
from gcbm.imagenet_config import Config
from gcbm.imagenet_final_layers import MemmapFeatureDataset
from gcbm.sparse import threshold_weight_truncation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a standalone GLM-SAGA lambda path sweep on saved SG-CBM features.")
    parser.add_argument("--artifact_dir", required=True)
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--saga_batch_size", type=int, default=512)
    parser.add_argument("--saga_workers", type=int, default=4)
    parser.add_argument("--saga_prefetch_factor", type=int, default=2)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--step_size", type=float, default=0.1)
    parser.add_argument("--n_iters", type=int, default=30)
    parser.add_argument("--lam_max", type=float, default=0.01)
    parser.add_argument("--max_glm_steps", type=int, default=50)
    parser.add_argument("--epsilon", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=0.99)
    parser.add_argument("--tol", type=float, default=1e-4, help="Solver convergence tolerance for GLM-SAGA.")
    parser.add_argument("--table_device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--verbose_every", type=int, default=1)
    parser.add_argument(
        "--eval_every",
        type=int,
        default=0,
        help="If > 0, run a full validation pass every N outer SAGA iterations and log val loss/acc.",
    )
    parser.add_argument("--nec_values", default="5,10,15,20,25,30")
    parser.add_argument("--skip_train_eval", action="store_true", help="Skip the full train-set metric pass after each lambda.")
    parser.add_argument("--skip_val_eval", action="store_true", help="Skip the full val-set metric pass after each lambda.")
    parser.add_argument(
        "--max_sparsity",
        type=float,
        default=None,
        help="Stop the lambda path once nnz/total exceeds this fraction.",
    )
    parser.add_argument(
        "--cache_features_device",
        choices=["none", "cuda"],
        default="none",
        help="Load normalized feature tensors once onto the selected device and bypass DataLoader/memmap per epoch.",
    )
    parser.add_argument(
        "--cache_chunk_rows",
        type=int,
        default=8192,
        help="Rows per CPU staging chunk when --cache_features_device is enabled.",
    )
    return parser.parse_args()


def parse_nec_values(raw: str) -> List[int]:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    if not values:
        raise ValueError("--nec_values must contain at least one integer")
    return values


def link_source_artifacts(source_run_dir: Path, output_dir: Path) -> None:
    for name in ("config.json", "concepts.txt", "concept_head_best.pt", "final_layer_normalization.pt"):
        source = source_run_dir / name
        target = output_dir / name
        if source.exists() and not target.exists():
            target.symlink_to(source)


def resolve_source_run_dir(artifact_dir: Path) -> Path:
    def existing_dir(value: str) -> Path | None:
        candidate = Path(value).resolve()
        if candidate.is_dir():
            return candidate
        return None

    source_run_file = artifact_dir / "source_run_dir.txt"
    if source_run_file.exists():
        source_run_dir = existing_dir(source_run_file.read_text().strip())
        if source_run_dir is not None:
            return source_run_dir
    return artifact_dir


def load_config(config_dir: Path, device: str) -> Config:
    payload = json.loads((config_dir / "config.json").read_text())
    payload.setdefault("feature_storage_dtype", "fp16")
    payload.setdefault("saga_table_device", "cpu")
    payload.setdefault("dense_lr", 1e-3)
    payload.setdefault("dense_n_iters", 20)
    payload.setdefault("train_random_transforms", True)
    payload.setdefault("learn_spatial_residual_scale", False)
    payload["device"] = device
    payload["skip_final_layer"] = True
    payload["print_config"] = False
    valid_fields = {field.name for field in fields(Config)}
    return Config(**{key: value for key, value in payload.items() if key in valid_fields})


def infer_n_classes(*target_paths: Path) -> int:
    max_class_id = -1
    for target_path in target_paths:
        targets = np.load(target_path, mmap_mode="r")
        if int(targets.shape[0]) == 0:
            continue
        max_class_id = max(max_class_id, int(np.max(targets).item()))
    if max_class_id < 0:
        raise RuntimeError("Could not infer class count from target files")
    return max_class_id + 1


def build_loader(
    feature_path: Path,
    target_path: Path,
    *,
    batch_size: int,
    workers: int,
    pin_memory: bool,
    prefetch_factor: int,
    mean,
    std,
    include_index: bool,
    shuffle: bool,
) -> DataLoader:
    dataset = MemmapFeatureDataset(
        feature_path=feature_path,
        target_path=target_path,
        mean=mean,
        std=std,
        include_index=include_index,
    )
    kwargs: Dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": workers,
        "pin_memory": pin_memory,
        "drop_last": False,
    }
    if workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = max(1, int(prefetch_factor))
    return DataLoader(**kwargs)


class TensorFeatureDataset:
    def __init__(self, features: torch.Tensor, targets: torch.Tensor) -> None:
        self.features = features
        self.targets = targets

    def __len__(self) -> int:
        return int(self.features.shape[0])


class TensorBatchLoader:
    def __init__(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        *,
        batch_size: int,
        include_index: bool,
        shuffle: bool,
    ) -> None:
        self.dataset = TensorFeatureDataset(features, targets)
        self.batch_size = int(batch_size)
        self.include_index = include_index
        self.shuffle = shuffle

    def __len__(self) -> int:
        n = len(self.dataset)
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        features = self.dataset.features
        targets = self.dataset.targets
        n = len(self.dataset)
        if self.shuffle:
            order = torch.randperm(n, device=features.device)
            for start in range(0, n, self.batch_size):
                idx = order[start : start + self.batch_size]
                if self.include_index:
                    yield features[idx], targets[idx], idx
                else:
                    yield features[idx], targets[idx]
            return

        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            if self.include_index:
                idx = torch.arange(start, end, device=features.device)
                yield features[start:end], targets[start:end], idx
            else:
                yield features[start:end], targets[start:end]


def load_cached_feature_tensors(
    feature_path: Path,
    target_path: Path,
    *,
    mean: np.ndarray,
    std: np.ndarray,
    device: str,
    chunk_rows: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    features_np = np.load(feature_path, mmap_mode="r")
    targets_np = np.load(target_path, mmap_mode="r")
    features = torch.empty(features_np.shape, dtype=torch.float32, device=device)
    chunk_rows = max(1, int(chunk_rows))
    for start in range(0, int(features_np.shape[0]), chunk_rows):
        end = min(start + chunk_rows, int(features_np.shape[0]))
        chunk = np.asarray(features_np[start:end], dtype=np.float32)
        chunk = (chunk - mean) / std
        features[start:end].copy_(torch.from_numpy(np.ascontiguousarray(chunk)).to(device, non_blocking=True))
    targets = torch.from_numpy(np.array(targets_np, dtype=np.int64, copy=True)).to(device, non_blocking=True)
    return features, targets


def select_path_points_for_nec(path: Sequence[Dict[str, Any]], n_concepts: int, nec_values: Sequence[int]) -> List[Dict[str, Any]]:
    sparsities = [float((params["weight"].abs() > 1e-5).float().mean().item()) for params in path]
    selections: List[Dict[str, Any]] = []
    for nec in nec_values:
        target_sparsity = float(nec) / float(n_concepts)
        selected_idx = len(path) - 1
        for idx, sparsity in enumerate(sparsities):
            if sparsity >= target_sparsity:
                selected_idx = idx
                break
        params = path[selected_idx]
        weight = params["weight"]
        nnz = int((weight.abs() > 1e-5).sum().item())
        selections.append(
            {
                "nec": int(nec),
                "target_sparsity": target_sparsity,
                "path_index": int(selected_idx),
                "lambda": float(params["lam"]),
                "lr": float(params["lr"]),
                "nnz": nnz,
                "total": int(weight.numel()),
                "weight_sparsity": 1.0 - (nnz / max(int(weight.numel()), 1)),
                "metrics": params["metrics"],
            }
        )
    return selections


def save_thresholded_nec_weights(path: Sequence[Dict[str, Any]], selections: Sequence[Dict[str, Any]], output_dir: Path) -> List[Dict[str, Any]]:
    updated = []
    for item in selections:
        params = path[int(item["path_index"])]
        weight = threshold_weight_truncation(params["weight"], float(item["target_sparsity"]))
        bias = params["bias"]
        nnz = int((weight.abs() > 1e-5).sum().item())
        torch.save(weight.cpu(), output_dir / f"W_g@NEC={item['nec']}.pt")
        torch.save(bias.cpu(), output_dir / f"b_g@NEC={item['nec']}.pt")
        item = dict(item)
        item["truncated_nnz"] = nnz
        item["truncated_weight_sparsity"] = 1.0 - (nnz / max(int(weight.numel()), 1))
        updated.append(item)
    return updated


def serializable_path(path: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    payload = []
    for idx, params in enumerate(path):
        weight = params["weight"]
        nnz = int((weight.abs() > 1e-5).sum().item())
        payload.append(
            {
                "index": idx,
                "lambda": float(params["lam"]),
                "lr": float(params["lr"]),
                "alpha": float(params["alpha"]),
                "time": float(params["time"]),
                "nnz": nnz,
                "total": int(weight.numel()),
                "weight_sparsity": 1.0 - (nnz / max(int(weight.numel()), 1)),
                "metrics": params["metrics"],
            }
        )
    return payload


def main() -> None:
    args = parse_args()
    artifact_dir = Path(args.artifact_dir).resolve()
    source_run_dir = resolve_source_run_dir(artifact_dir)
    output_dir = Path(args.output_dir).resolve() if args.output_dir else artifact_dir / f"glm_path_sweep_lam{str(args.lam_max).replace('.', 'p')}_k{args.max_glm_steps}"
    output_dir.mkdir(parents=True, exist_ok=True)
    link_source_artifacts(source_run_dir, output_dir)

    cfg = load_config(source_run_dir, args.device)
    normalization_path = artifact_dir / "final_layer_normalization.pt"
    if not normalization_path.exists():
        normalization_path = source_run_dir / "final_layer_normalization.pt"
    normalization_payload = torch.load(normalization_path, map_location="cpu")
    feature_mean = normalization_payload["mean"].float().cpu().numpy()
    feature_std = normalization_payload["std"].float().cpu().numpy()

    feature_root = artifact_dir / "features"
    if not feature_root.is_dir():
        feature_root = source_run_dir / "features"
    train_feature_path = feature_root / "train_features.npy"
    train_target_path = feature_root / "train_targets.npy"
    val_feature_path = feature_root / "val_features.npy"
    val_target_path = feature_root / "val_targets.npy"
    for path in (train_feature_path, train_target_path, val_feature_path, val_target_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing feature artifact: {path}")

    if args.cache_features_device == "cuda":
        cache_device = args.device
        print(f"loading normalized train features to {cache_device}...")
        train_features, train_targets = load_cached_feature_tensors(
            train_feature_path,
            train_target_path,
            mean=feature_mean,
            std=feature_std,
            device=cache_device,
            chunk_rows=args.cache_chunk_rows,
        )
        train_loader = TensorBatchLoader(
            train_features,
            train_targets,
            batch_size=args.saga_batch_size,
            include_index=True,
            shuffle=True,
        )
        if args.skip_val_eval:
            val_loader = None
        else:
            print(f"loading normalized val features to {cache_device}...")
            val_features, val_targets = load_cached_feature_tensors(
                val_feature_path,
                val_target_path,
                mean=feature_mean,
                std=feature_std,
                device=cache_device,
                chunk_rows=args.cache_chunk_rows,
            )
            val_loader = TensorBatchLoader(
                val_features,
                val_targets,
                batch_size=args.saga_batch_size,
                include_index=False,
                shuffle=False,
            )
    else:
        train_loader = build_loader(
            train_feature_path,
            train_target_path,
            batch_size=args.saga_batch_size,
            workers=args.saga_workers,
            pin_memory=args.pin_memory,
            prefetch_factor=args.saga_prefetch_factor,
            mean=feature_mean,
            std=feature_std,
            include_index=True,
            shuffle=True,
        )
        val_loader = build_loader(
            val_feature_path,
            val_target_path,
            batch_size=args.saga_batch_size,
            workers=args.saga_workers,
            pin_memory=args.pin_memory,
            prefetch_factor=args.saga_prefetch_factor,
            mean=feature_mean,
            std=feature_std,
            include_index=False,
            shuffle=False,
        )

    n_features = int(train_loader.dataset.features.shape[1])
    n_classes = infer_n_classes(train_target_path, val_target_path)
    linear = torch.nn.Linear(n_features, n_classes, bias=True).to(args.device)
    linear.weight.data.zero_()
    linear.bias.data.zero_()

    metadata = {"max_reg": {"nongrouped": args.lam_max}}
    start = time.perf_counter()
    output = glm_saga(
        linear,
        train_loader,
        args.step_size,
        args.n_iters,
        args.alpha,
        table_device=args.table_device,
        tol=args.tol,
        epsilon=args.epsilon,
        k=args.max_glm_steps,
        val_loader=val_loader,
        do_zero=False,
        metadata=metadata,
        n_ex=len(train_loader.dataset),
        n_classes=n_classes,
        verbose=args.verbose_every,
        eval_every=(args.eval_every if args.eval_every > 0 else None),
        max_sparsity=args.max_sparsity,
        eval_train=not args.skip_train_eval,
        eval_val=not args.skip_val_eval,
        eval_test=False,
    )
    elapsed = time.perf_counter() - start

    path = output["path"]
    best = output["best"]
    nec_values = parse_nec_values(args.nec_values)
    nec_selection = select_path_points_for_nec(path, n_features, nec_values)
    nec_selection = save_thresholded_nec_weights(path, nec_selection, output_dir)

    torch.save({"path": path, "best": best}, output_dir / "glm_path.pt")

    payload = {
        "artifact_dir": str(artifact_dir),
        "source_run_dir": str(source_run_dir),
        "feature_root": str(feature_root),
        "output_dir": str(output_dir),
        "n_features": n_features,
        "n_classes": n_classes,
        "config": {
            "saga_batch_size": args.saga_batch_size,
            "saga_workers": args.saga_workers,
            "saga_prefetch_factor": args.saga_prefetch_factor,
            "step_size": args.step_size,
            "n_iters": args.n_iters,
            "lam_max": args.lam_max,
            "max_glm_steps": args.max_glm_steps,
            "epsilon": args.epsilon,
            "alpha": args.alpha,
            "tol": args.tol,
            "table_device": args.table_device,
            "verbose_every": args.verbose_every,
            "eval_every": args.eval_every,
            "pin_memory": bool(args.pin_memory),
            "skip_train_eval": bool(args.skip_train_eval),
            "skip_val_eval": bool(args.skip_val_eval),
            "max_sparsity": args.max_sparsity,
            "cache_features_device": args.cache_features_device,
            "cache_chunk_rows": args.cache_chunk_rows,
        },
        "elapsed_sec": elapsed,
        "best": {
            "lambda": float(best["lam"]),
            "lr": float(best["lr"]),
            "alpha": float(best["alpha"]),
            "time": float(best["time"]),
            "metrics": best["metrics"],
        },
        "path": serializable_path(path),
        "nec_selection": nec_selection,
    }
    (output_dir / "source_run_dir.txt").write_text(f"{source_run_dir}\n")
    (output_dir / "glm_path_metrics.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
