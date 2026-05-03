import argparse
import json
import sys
import tarfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gcbm.imagenet_core import (
    Config,
    amp_dtype,
    build_model,
    configure_runtime,
    cuda_peak_stats_mb,
    prepare_images,
)


VAL_RE = "ILSVRC2012_val_"


def weight_truncation(weight: torch.Tensor, sparsity: float) -> torch.Tensor:
    """Keep the top-k concept weights for each output class.

    NEC is defined per class, not globally over the full class-by-concept
    matrix. A global top-k can starve low-magnitude classes at small NEC.
    """
    num_concepts = int(weight.shape[1])
    k = int(round(float(sparsity) * num_concepts))
    if k <= 0:
        return torch.zeros_like(weight)
    if k >= num_concepts:
        return weight.clone().detach()
    topk_idx = weight.abs().topk(k=k, dim=1).indices
    sparse_weight = torch.zeros_like(weight)
    sparse_weight.scatter_(1, topk_idx, weight.gather(1, topk_idx))
    return sparse_weight


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate G-CBM NEC sweeps on ImageNet.")
    parser.add_argument("--artifact_dir", required=True)
    parser.add_argument("--val_tar", default="")
    parser.add_argument("--devkit_dir", default="")
    parser.add_argument("--val_root", default="")
    parser.add_argument("--output_json", default="")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log_every", type=int, default=1000)
    parser.add_argument("--nec_values", default="5,10,15,20,25,30")
    parser.add_argument("--save_truncated_weights", action="store_true")
    parser.add_argument("--inference_alpha_override", type=float, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
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
    metrics_file = artifact_dir / "glm_path_metrics.json"
    if metrics_file.exists():
        payload = json.loads(metrics_file.read_text())
        source_value = payload.get("source_run_dir", "")
        if source_value:
            source_run_dir = existing_dir(str(source_value))
            if source_run_dir is not None:
                return source_run_dir
    return artifact_dir


def load_run_config(config_dir: Path, args: argparse.Namespace) -> Config:
    payload = json.loads((config_dir / "config.json").read_text())
    payload.setdefault("feature_storage_dtype", "fp16")
    payload.setdefault("saga_table_device", "cpu")
    payload.setdefault("dense_lr", 1e-3)
    payload.setdefault("dense_n_iters", 20)
    payload.setdefault("train_random_transforms", True)
    payload.setdefault("learn_spatial_residual_scale", False)
    payload["device"] = args.device
    payload["workers"] = args.workers
    payload["batch_size"] = args.batch_size
    payload["prefetch_factor"] = args.prefetch_factor
    payload["skip_final_layer"] = True
    payload["print_config"] = False
    return Config(**payload)


def resolve_final_layer_path(artifact_dir: Path) -> Path:
    dense_path = artifact_dir / "final_layer_dense.pt"
    if dense_path.exists():
        return dense_path
    glm_path = artifact_dir / "final_layer_glm_saga.pt"
    if glm_path.exists():
        return glm_path
    path_sweep = artifact_dir / "glm_path.pt"
    if path_sweep.exists():
        return path_sweep
    raise FileNotFoundError(f"no final layer artifact found under {artifact_dir}")


def load_final_layer_payload(final_layer_path: Path) -> Dict[str, torch.Tensor]:
    payload = torch.load(final_layer_path, map_location="cpu")
    if isinstance(payload, dict) and "weight" in payload and "bias" in payload:
        return {"weight": payload["weight"].float(), "bias": payload["bias"].float()}
    if isinstance(payload, dict) and "best" in payload:
        best = payload["best"]
        if isinstance(best, dict) and "weight" in best and "bias" in best:
            return {"weight": best["weight"].float(), "bias": best["bias"].float()}
    raise ValueError(f"could not load weight/bias from final layer artifact: {final_layer_path}")


def load_val_targets(devkit_dir: Path) -> List[int]:
    payload = loadmat(devkit_dir / "data" / "meta.mat", squeeze_me=True, struct_as_record=False)
    synsets = payload["synsets"]
    id_to_wnid: Dict[int, str] = {}
    for syn in synsets:
        ilsvrc_id = int(syn.ILSVRC2012_ID)
        if 1 <= ilsvrc_id <= 1000 and int(syn.num_children) == 0:
            id_to_wnid[ilsvrc_id] = str(syn.WNID)
    if len(id_to_wnid) != 1000:
        raise RuntimeError(f"expected 1000 leaf synsets, got {len(id_to_wnid)}")
    wnids = sorted(id_to_wnid.values())
    class_to_idx = {wnid: idx for idx, wnid in enumerate(wnids)}
    labels: List[int] = []
    with (devkit_dir / "data" / "ILSVRC2012_validation_ground_truth.txt").open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                labels.append(class_to_idx[id_to_wnid[int(line)]])
    if len(labels) != 50000:
        raise RuntimeError(f"expected 50000 validation labels, got {len(labels)}")
    return labels


def build_transform(input_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def iter_tar_samples(val_tar: Path, targets: Sequence[int], transform: transforms.Compose):
    seen = 0
    with tarfile.open(val_tar, "r|") as tf:
        for member in tf:
            if not member.isfile():
                continue
            image_name = Path(member.name).name
            if not (image_name.startswith(VAL_RE) and image_name.endswith(".JPEG")):
                continue
            image_idx = int(image_name[-13:-5])
            target = int(targets[image_idx - 1])
            handle = tf.extractfile(member)
            if handle is None:
                raise FileNotFoundError(member.name)
            with Image.open(handle) as image:
                tensor = transform(image.convert("RGB"))
            seen += 1
            yield tensor, target, member.name
    if seen != 50000:
        raise RuntimeError(f"expected 50000 val images in tar, found {seen}")


def build_val_loader(val_root: Path, cfg: Config) -> DataLoader:
    dataset = ImageFolder(root=str(val_root), transform=build_transform(cfg.input_size))
    kwargs: Dict[str, Any] = {
        "dataset": dataset,
        "batch_size": cfg.batch_size,
        "shuffle": False,
        "num_workers": cfg.workers,
        "pin_memory": cfg.pin_memory,
    }
    if cfg.workers > 0:
        kwargs["persistent_workers"] = cfg.persistent_workers
        kwargs["prefetch_factor"] = cfg.prefetch_factor
    return DataLoader(**kwargs)


def build_weight_sweep(weight: torch.Tensor, bias: torch.Tensor, nec_values: Sequence[int], save_dir: Path | None):
    num_concepts = int(weight.shape[1])
    sweep = []
    for nec in nec_values:
        target_sparsity = float(nec) / float(num_concepts)
        truncated = weight_truncation(weight, target_sparsity)
        nnz = int((truncated.abs() > 1e-5).sum().item())
        if save_dir is not None:
            torch.save(truncated.cpu(), save_dir / f"W_g@NEC={int(nec)}.pt")
            torch.save(bias.cpu(), save_dir / f"b_g@NEC={int(nec)}.pt")
        sweep.append(
            {
                "nec": int(nec),
                "weight": truncated,
                "bias": bias,
                "nnz": nnz,
                "total": int(truncated.numel()),
                "weight_sparsity": 1.0 - (nnz / max(int(truncated.numel()), 1)),
            }
        )
    return sweep


def load_saved_weight_sweep(artifact_dir: Path, nec_values: Sequence[int]) -> List[Dict[str, Any]] | None:
    sweep = []
    for nec in nec_values:
        weight_path = artifact_dir / f"W_g@NEC={int(nec)}.pt"
        bias_path = artifact_dir / f"b_g@NEC={int(nec)}.pt"
        if not (weight_path.exists() and bias_path.exists()):
            return None
        weight = torch.load(weight_path, map_location="cpu").float()
        bias = torch.load(bias_path, map_location="cpu").float()
        nnz = int((weight.abs() > 1e-5).sum().item())
        sweep.append(
            {
                "nec": int(nec),
                "weight": weight,
                "bias": bias,
                "nnz": nnz,
                "total": int(weight.numel()),
                "weight_sparsity": 1.0 - (nnz / max(int(weight.numel()), 1)),
            }
        )
    return sweep


def _evaluate_logits(
    concept_logits: torch.Tensor,
    targets: torch.Tensor,
    stacked_weights: torch.Tensor,
    stacked_biases: torch.Tensor,
    top1: torch.Tensor,
    top5: torch.Tensor,
) -> None:
    logits = torch.einsum("bc,koc->kbo", concept_logits, stacked_weights) + stacked_biases[:, None, :]
    pred1 = logits.argmax(dim=-1)
    pred5 = logits.topk(k=min(5, logits.shape[-1]), dim=-1).indices
    match1 = pred1.eq(targets.unsqueeze(0))
    match5 = pred5.eq(targets.view(1, -1, 1)).any(dim=-1)
    top1.add_(match1.sum(dim=1).cpu())
    top5.add_(match5.sum(dim=1).cpu())


def evaluate_tar(
    val_tar: Path,
    targets: Sequence[int],
    transform: transforms.Compose,
    backbone: torch.nn.Module,
    head: torch.nn.Module,
    sweep: Sequence[Dict[str, Any]],
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
    cfg: Config,
    log_every: int,
    max_samples: int | None,
) -> Dict[str, Any]:
    device = cfg.device
    stacked_weights = torch.stack([item["weight"].to(device).float() for item in sweep], dim=0)
    stacked_biases = torch.stack([item["bias"].to(device).float() for item in sweep], dim=0)
    top1 = torch.zeros(len(sweep), dtype=torch.long)
    top5 = torch.zeros(len(sweep), dtype=torch.long)
    total = 0
    next_log = max(int(log_every), 1)
    start = time.perf_counter()
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    images: List[torch.Tensor] = []
    batch_targets: List[int] = []

    def flush(last_name: str | None) -> None:
        nonlocal next_log, total
        if not images:
            return
        batch = torch.stack(images, dim=0)
        batch = prepare_images(batch, cfg)
        target_tensor = torch.tensor(batch_targets, dtype=torch.long, device=device)
        with torch.no_grad():
            with torch.autocast(
                device_type="cuda",
                dtype=amp_dtype(cfg.amp),
                enabled=(str(device).startswith("cuda") and amp_dtype(cfg.amp) is not None),
            ):
                feats = backbone(batch)
                outputs = head(feats)
                concept_logits = outputs["final_logits"].float()
                concept_logits = (concept_logits - feature_mean) / feature_std
        _evaluate_logits(concept_logits, target_tensor, stacked_weights, stacked_biases, top1, top5)
        total += len(images)
        if total >= next_log:
            elapsed = time.perf_counter() - start
            print(f"[nec-eval] n={total} ips={total / max(elapsed, 1e-6):.2f} last={last_name}", flush=True)
            while next_log <= total:
                next_log += max(int(log_every), 1)
        images.clear()
        batch_targets.clear()

    for image, target, name in iter_tar_samples(val_tar, targets, transform):
        images.append(image)
        batch_targets.append(int(target))
        if len(images) >= cfg.batch_size:
            flush(name)
        if max_samples is not None and total + len(images) >= int(max_samples):
            break
    flush(None)

    elapsed = time.perf_counter() - start
    results = []
    for idx, item in enumerate(sweep):
        results.append(
            {
                "nec": item["nec"],
                "nnz": item["nnz"],
                "total": item["total"],
                "weight_sparsity": item["weight_sparsity"],
                "top1": float(top1[idx].item() / max(total, 1)),
                "top5": float(top5[idx].item() / max(total, 1)),
            }
        )
    payload: Dict[str, Any] = {
        "n": total,
        "elapsed_sec": elapsed,
        "images_per_second": total / max(elapsed, 1e-6),
        "results": results,
    }
    payload.update(cuda_peak_stats_mb(cfg))
    return payload


def evaluate_root(
    loader: DataLoader,
    backbone: torch.nn.Module,
    head: torch.nn.Module,
    sweep: Sequence[Dict[str, Any]],
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
    cfg: Config,
    log_every: int,
    max_samples: int | None,
) -> Dict[str, Any]:
    device = cfg.device
    stacked_weights = torch.stack([item["weight"].to(device).float() for item in sweep], dim=0)
    stacked_biases = torch.stack([item["bias"].to(device).float() for item in sweep], dim=0)
    top1 = torch.zeros(len(sweep), dtype=torch.long)
    top5 = torch.zeros(len(sweep), dtype=torch.long)
    total = 0
    start = time.perf_counter()
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        for step, (images, targets) in enumerate(loader, start=1):
            images = prepare_images(images, cfg)
            target_tensor = targets.to(device, non_blocking=cfg.pin_memory)
            with torch.autocast(
                device_type="cuda",
                dtype=amp_dtype(cfg.amp),
                enabled=(str(device).startswith("cuda") and amp_dtype(cfg.amp) is not None),
            ):
                feats = backbone(images)
                outputs = head(feats)
                concept_logits = outputs["final_logits"].float()
                concept_logits = (concept_logits - feature_mean) / feature_std
            _evaluate_logits(concept_logits, target_tensor, stacked_weights, stacked_biases, top1, top5)
            total += int(target_tensor.numel())
            if step % 50 == 0 or total % log_every == 0:
                elapsed = time.perf_counter() - start
                print(f"[nec-eval] step={step}/{len(loader)} n={total} ips={total / max(elapsed, 1e-6):.2f}", flush=True)
            if max_samples is not None and total >= int(max_samples):
                break

    elapsed = time.perf_counter() - start
    results = []
    for idx, item in enumerate(sweep):
        results.append(
            {
                "nec": item["nec"],
                "nnz": item["nnz"],
                "total": item["total"],
                "weight_sparsity": item["weight_sparsity"],
                "top1": float(top1[idx].item() / max(total, 1)),
                "top5": float(top5[idx].item() / max(total, 1)),
            }
        )
    payload: Dict[str, Any] = {
        "n": total,
        "elapsed_sec": elapsed,
        "images_per_second": total / max(elapsed, 1e-6),
        "results": results,
    }
    payload.update(cuda_peak_stats_mb(cfg))
    return payload


def main() -> None:
    args = parse_args()
    artifact_dir = Path(args.artifact_dir).resolve()
    source_run_dir = resolve_source_run_dir(artifact_dir)
    output_json = Path(args.output_json).resolve() if args.output_json else artifact_dir / "nec_eval.json"
    nec_values = parse_nec_values(args.nec_values)

    cfg = load_run_config(source_run_dir, args)
    if args.inference_alpha_override is not None:
        cfg.residual_alpha = float(args.inference_alpha_override)
    configure_runtime(cfg)

    concepts = [line.strip() for line in (source_run_dir / "concepts.txt").read_text().splitlines() if line.strip()]
    backbone, head = build_model(cfg, n_concepts=len(concepts))
    head.load_state_dict(torch.load(source_run_dir / "concept_head_best.pt", map_location=cfg.device))
    if args.inference_alpha_override is not None and hasattr(head, "residual_alpha"):
        head.residual_alpha = float(args.inference_alpha_override)

    normalization_payload = torch.load(artifact_dir / "final_layer_normalization.pt", map_location="cpu")
    feature_mean = normalization_payload["mean"].to(cfg.device).float()
    feature_std = normalization_payload["std"].to(cfg.device).float().clamp_min(1e-6)
    sweep = None if args.save_truncated_weights else load_saved_weight_sweep(artifact_dir, nec_values)
    final_layer_path: Path | None = None
    if sweep is None:
        final_layer_path = resolve_final_layer_path(artifact_dir)
        linear_payload = load_final_layer_payload(final_layer_path)
        sweep = build_weight_sweep(
            linear_payload["weight"],
            linear_payload["bias"],
            nec_values,
            artifact_dir if args.save_truncated_weights else None,
        )

    payload: Dict[str, Any] = {
        "artifact_dir": str(artifact_dir),
        "final_layer_path": str(final_layer_path) if final_layer_path is not None else "",
        "source_run_dir": str(source_run_dir),
        "nec_values": nec_values,
        "inference_alpha_override": args.inference_alpha_override,
        "max_samples": args.max_samples,
    }

    if args.val_tar:
        if not args.devkit_dir:
            raise ValueError("--devkit_dir is required with --val_tar")
        val_tar = Path(args.val_tar).resolve()
        devkit_dir = Path(args.devkit_dir).resolve()
        targets = load_val_targets(devkit_dir)
        payload["val_tar"] = str(val_tar)
        payload["devkit_dir"] = str(devkit_dir)
        payload["metrics"] = evaluate_tar(
            val_tar,
            targets,
            build_transform(cfg.input_size),
            backbone,
            head,
            sweep,
            feature_mean,
            feature_std,
            cfg,
            args.log_every,
            args.max_samples,
        )
    elif args.val_root:
        val_root = Path(args.val_root).resolve()
        payload["val_root"] = str(val_root)
        payload["metrics"] = evaluate_root(
            build_val_loader(val_root, cfg),
            backbone,
            head,
            sweep,
            feature_mean,
            feature_std,
            cfg,
            args.log_every,
            args.max_samples,
        )
    else:
        raise ValueError("Provide either --val_tar/--devkit_dir or --val_root")

    output_json.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
