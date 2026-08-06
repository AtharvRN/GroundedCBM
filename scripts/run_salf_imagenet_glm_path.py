import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data import utils as data_utils
from glm_saga.elasticnet import glm_saga
from scripts.eval_salf_imagenet_nec_tar import SoftmaxPooling2D, build_showandtell_transform
from scripts.run_savlg_imagenet_standalone_glm_path import (
    build_loader,
    load_cached_feature_tensors,
    select_path_points_for_nec,
    serializable_path,
)

try:
    from transformers import ResNetForImageClassification
except ImportError:
    ResNetForImageClassification = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a GLM-SAGA lambda path sweep on official SALF ImageNet concept features.")
    parser.add_argument("--load_dir", required=True, help="SALF checkpoint dir with W_c.pt, proj_mean.pt, proj_std.pt, concepts.txt.")
    parser.add_argument("--train_root", required=True, help="ImageNet train root organized as ImageFolder.")
    parser.add_argument("--val_root", required=True, help="ImageNet val root organized as ImageFolder.")
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--backbone",
        choices=["resnet50_imagenet", "resnet50"],
        default="resnet50_imagenet",
        help="Use resnet50_imagenet for the official SALF-CBM released checkpoint protocol.",
    )
    parser.add_argument("--feature_batch_size", type=int, default=256)
    parser.add_argument("--feature_workers", type=int, default=8)
    parser.add_argument("--feature_prefetch_factor", type=int, default=2)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--map_size", default="12,12")
    parser.add_argument("--pooling", choices=["avg", "softmax"], default="softmax")
    parser.add_argument("--feature_dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--max_train_images", type=int, default=0)
    parser.add_argument("--max_val_images", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--saga_batch_size", type=int, default=512)
    parser.add_argument("--saga_workers", type=int, default=4)
    parser.add_argument("--saga_prefetch_factor", type=int, default=2)
    parser.add_argument("--step_size", type=float, default=0.01)
    parser.add_argument("--n_iters", type=int, default=500)
    parser.add_argument("--lam_max", type=float, default=0.002)
    parser.add_argument("--max_glm_steps", type=int, default=100)
    parser.add_argument("--epsilon", type=float, default=1e-2)
    parser.add_argument("--alpha", type=float, default=0.99)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--table_device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--verbose_every", type=int, default=10)
    parser.add_argument("--eval_every", type=int, default=0)
    parser.add_argument("--nec_values", default="5,10,15,20,25,30")
    parser.add_argument("--skip_train_eval", action="store_true")
    parser.add_argument("--skip_val_eval", action="store_true")
    parser.add_argument("--max_sparsity", type=float, default=None)
    parser.add_argument("--cache_features_device", choices=["none", "cuda"], default="none")
    parser.add_argument("--cache_chunk_rows", type=int, default=8192)
    parser.add_argument("--log_every", type=int, default=2048)
    return parser.parse_args()


def parse_map_size(raw: str) -> Tuple[int, int]:
    parts = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if len(parts) != 2:
        raise ValueError("--map_size must be H,W")
    return parts[0], parts[1]


def parse_nec_values(raw: str) -> List[int]:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    if not values:
        raise ValueError("--nec_values must contain at least one integer")
    return values


def maybe_subset(dataset: Dataset, max_items: int, seed: int) -> Dataset:
    if int(max_items) <= 0 or len(dataset) <= int(max_items):
        return dataset
    generator = torch.Generator().manual_seed(int(seed))
    indices = torch.randperm(len(dataset), generator=generator)[: int(max_items)].tolist()
    return Subset(dataset, indices)


def build_imagefolder(root: Path, max_items: int, seed: int, transform) -> Dataset:
    dataset = datasets.ImageFolder(
        root=str(root),
        transform=transform,
        loader=data_utils._safe_imagenet_pil_loader,
    )
    return maybe_subset(dataset, max_items, seed)


def build_feature_loader(dataset: Dataset, args: argparse.Namespace) -> DataLoader:
    kwargs: Dict[str, Any] = {
        "dataset": dataset,
        "batch_size": int(args.feature_batch_size),
        "shuffle": False,
        "num_workers": int(args.feature_workers),
        "pin_memory": bool(args.pin_memory),
        "drop_last": False,
    }
    if int(args.feature_workers) > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = max(1, int(args.feature_prefetch_factor))
    return DataLoader(**kwargs)


def open_memmap_array(path: Path, shape: Sequence[int], dtype: np.dtype):
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(str(path), mode="w+", dtype=dtype, shape=tuple(shape))


def build_salf_backbone(backbone_name: str, device: str):
    if backbone_name == "resnet50_imagenet":
        if ResNetForImageClassification is None:
            raise ImportError(
                "transformers is required for --backbone resnet50_imagenet "
                "(official SALF-CBM ImageNet checkpoint protocol)."
            )
        target_model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50").to(device)
        target_model.eval()
        backbone = torch.nn.Sequential(*list(target_model.resnet.children())[:-1]).to(device).eval()

        def forward(images: torch.Tensor) -> torch.Tensor:
            return backbone(images).last_hidden_state.float()

        return forward

    target_model, _ = data_utils.get_target_model("resnet50", device)
    target_model.eval()
    return torch.nn.Sequential(*list(target_model.children())[:-2]).to(device).eval()


def load_salf_projection(
    load_dir: Path,
    device: str,
    map_size: Tuple[int, int],
    pooling: str,
):
    w_c = torch.load(load_dir / "W_c.pt", map_location=device).float()
    if w_c.ndim == 2:
        w_c = w_c[:, :, None, None]
    if w_c.ndim != 4:
        raise ValueError(f"expected SALF W_c to have 2 or 4 dims, got shape={tuple(w_c.shape)}")
    proj_mean = torch.load(load_dir / "proj_mean.pt", map_location="cpu").float().flatten().numpy()
    proj_std = torch.load(load_dir / "proj_std.pt", map_location="cpu").float().flatten().clamp_min(1e-6).numpy()
    pool_layer = SoftmaxPooling2D(map_size).to(device) if pooling == "softmax" else None
    return w_c, proj_mean, proj_std, pool_layer


def extract_feature_memmap(
    *,
    name: str,
    dataset: Dataset,
    loader: DataLoader,
    backbone: torch.nn.Module,
    w_c: torch.Tensor,
    pool_layer: torch.nn.Module | None,
    map_size: Tuple[int, int],
    output_dir: Path,
    dtype: str,
    device: str,
    log_every: int,
) -> Tuple[Path, Path, Dict[str, Any]]:
    n_rows = len(dataset)
    n_features = int(w_c.shape[0])
    feature_dtype = np.float16 if dtype == "float16" else np.float32
    feature_path = output_dir / f"{name}_features.npy"
    target_path = output_dir / f"{name}_targets.npy"
    feature_mm = open_memmap_array(feature_path, (n_rows, n_features), feature_dtype)
    target_mm = open_memmap_array(target_path, (n_rows,), np.int64)

    written = 0
    start = time.perf_counter()
    with torch.no_grad():
        for images, targets in loader:
            batch = images.to(device, non_blocking=True)
            feats = backbone(batch).float()
            if tuple(feats.shape[-2:]) != map_size:
                feats = F.interpolate(feats, size=map_size, mode="bilinear", align_corners=False)
            maps = F.conv2d(feats, w_c)
            if pool_layer is not None:
                pooled = pool_layer(maps).flatten(1)
            else:
                pooled = F.adaptive_avg_pool2d(maps, 1).flatten(1)
            batch_np = pooled.detach().cpu().numpy().astype(feature_dtype, copy=False)
            target_np = targets.detach().cpu().numpy().astype(np.int64, copy=False)
            batch_size = int(batch_np.shape[0])
            feature_mm[written : written + batch_size] = batch_np
            target_mm[written : written + batch_size] = target_np
            written += batch_size
            if written % max(int(log_every), 1) == 0 or written == n_rows:
                elapsed = time.perf_counter() - start
                print(
                    f"[salf-glm] extracted {name}: n={written}/{n_rows} "
                    f"ips={written / max(elapsed, 1e-6):.2f}",
                    flush=True,
                )
    feature_mm.flush()
    target_mm.flush()
    elapsed = time.perf_counter() - start
    return (
        feature_path,
        target_path,
        {
            "split": name,
            "rows": int(written),
            "n_features": int(n_features),
            "feature_dtype": str(dtype),
            "elapsed_sec": float(elapsed),
            "images_per_second": float(written / max(elapsed, 1e-6)),
        },
    )


def main() -> None:
    args = parse_args()
    load_dir = Path(args.load_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else load_dir / f"glm_sweep_lam{str(args.lam_max).replace('.', 'p')}_k{args.max_glm_steps}_{args.pooling}"
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_root = output_dir / "features"
    map_size = parse_map_size(args.map_size)

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    transform = build_showandtell_transform() if args.backbone == "resnet50_imagenet" else data_utils.get_resnet_imagenet_preprocess()
    train_dataset = build_imagefolder(Path(args.train_root).resolve(), int(args.max_train_images), int(args.seed), transform)
    val_dataset = build_imagefolder(Path(args.val_root).resolve(), int(args.max_val_images), int(args.seed), transform)
    train_loader = build_feature_loader(train_dataset, args)
    val_loader = build_feature_loader(val_dataset, args)

    if not hasattr(train_dataset, "dataset") and hasattr(train_dataset, "classes"):
        n_classes = len(train_dataset.classes)
    elif hasattr(train_dataset, "dataset") and hasattr(train_dataset.dataset, "classes"):
        n_classes = len(train_dataset.dataset.classes)
    else:
        raise RuntimeError("Could not infer class count from ImageFolder dataset")

    backbone = build_salf_backbone(args.backbone, args.device)
    w_c, proj_mean, proj_std, pool_layer = load_salf_projection(load_dir, args.device, map_size, args.pooling)

    train_feature_path, train_target_path, train_extract = extract_feature_memmap(
        name="train",
        dataset=train_dataset,
        loader=train_loader,
        backbone=backbone,
        w_c=w_c,
        pool_layer=pool_layer,
        map_size=map_size,
        output_dir=feature_root,
        dtype=args.feature_dtype,
        device=args.device,
        log_every=args.log_every,
    )
    val_feature_path, val_target_path, val_extract = extract_feature_memmap(
        name="val",
        dataset=val_dataset,
        loader=val_loader,
        backbone=backbone,
        w_c=w_c,
        pool_layer=pool_layer,
        map_size=map_size,
        output_dir=feature_root,
        dtype=args.feature_dtype,
        device=args.device,
        log_every=args.log_every,
    )

    normalization_payload = {
        "mean": torch.from_numpy(proj_mean).float(),
        "std": torch.from_numpy(proj_std).float(),
        "source_run_dir": str(load_dir),
        "pooling": args.pooling,
        "map_size": list(map_size),
    }
    torch.save(normalization_payload, output_dir / "final_layer_normalization.pt")

    if args.cache_features_device == "cuda":
        cache_device = args.device
        print(f"[salf-glm] loading normalized train features to {cache_device}", flush=True)
        train_features, train_targets = load_cached_feature_tensors(
            train_feature_path,
            train_target_path,
            mean=proj_mean,
            std=proj_std,
            device=cache_device,
            chunk_rows=args.cache_chunk_rows,
        )
        print(f"[salf-glm] loading normalized val features to {cache_device}", flush=True)
        val_features, val_targets = load_cached_feature_tensors(
            val_feature_path,
            val_target_path,
            mean=proj_mean,
            std=proj_std,
            device=cache_device,
            chunk_rows=args.cache_chunk_rows,
        )
        from scripts.run_savlg_imagenet_standalone_glm_path import TensorBatchLoader

        train_loader_glm = TensorBatchLoader(
            train_features,
            train_targets,
            batch_size=args.saga_batch_size,
            include_index=True,
            shuffle=True,
        )
        val_loader_glm = TensorBatchLoader(
            val_features,
            val_targets,
            batch_size=args.saga_batch_size,
            include_index=False,
            shuffle=False,
        )
    else:
        train_loader_glm = build_loader(
            train_feature_path,
            train_target_path,
            batch_size=args.saga_batch_size,
            workers=args.saga_workers,
            pin_memory=args.pin_memory,
            prefetch_factor=args.saga_prefetch_factor,
            mean=proj_mean,
            std=proj_std,
            include_index=True,
            shuffle=True,
        )
        val_loader_glm = build_loader(
            val_feature_path,
            val_target_path,
            batch_size=args.saga_batch_size,
            workers=args.saga_workers,
            pin_memory=args.pin_memory,
            prefetch_factor=args.saga_prefetch_factor,
            mean=proj_mean,
            std=proj_std,
            include_index=False,
            shuffle=False,
        )

    n_features = int(w_c.shape[0])
    linear = torch.nn.Linear(n_features, int(n_classes), bias=True).to(args.device)
    linear.weight.data.zero_()
    linear.bias.data.zero_()

    metadata = {"max_reg": {"nongrouped": args.lam_max}}
    start = time.perf_counter()
    output = glm_saga(
        linear,
        train_loader_glm,
        args.step_size,
        args.n_iters,
        args.alpha,
        table_device=args.table_device,
        tol=args.tol,
        epsilon=args.epsilon,
        k=args.max_glm_steps,
        val_loader=val_loader_glm,
        do_zero=False,
        metadata=metadata,
        n_ex=len(train_loader_glm.dataset),
        n_classes=int(n_classes),
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
    for item in nec_selection:
        params = path[item["path_index"]]
        torch.save(params["weight"].cpu(), output_dir / f"W_g@NEC={item['nec']}.pt")
        torch.save(params["bias"].cpu(), output_dir / f"b_g@NEC={item['nec']}.pt")

    torch.save({"path": path, "best": best}, output_dir / "glm_path.pt")
    (output_dir / "source_run_dir.txt").write_text(f"{load_dir}\n")

    payload = {
        "load_dir": str(load_dir),
        "train_root": str(Path(args.train_root).resolve()),
        "val_root": str(Path(args.val_root).resolve()),
        "output_dir": str(output_dir),
        "feature_root": str(feature_root),
        "n_features": int(n_features),
        "n_classes": int(n_classes),
        "pooling": args.pooling,
        "map_size": list(map_size),
        "backbone": args.backbone,
        "train_extraction": train_extract,
        "val_extraction": val_extract,
        "config": {
            "feature_batch_size": args.feature_batch_size,
            "backbone": args.backbone,
            "feature_workers": args.feature_workers,
            "feature_prefetch_factor": args.feature_prefetch_factor,
            "feature_dtype": args.feature_dtype,
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
            "max_train_images": int(args.max_train_images),
            "max_val_images": int(args.max_val_images),
            "seed": int(args.seed),
        },
        "elapsed_sec": float(elapsed),
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
    (output_dir / "glm_path_metrics.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
