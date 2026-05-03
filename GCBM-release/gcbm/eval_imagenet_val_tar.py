import argparse
import json
import re
import sys
import tarfile
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from scipy.io import loadmat
from torchvision import transforms

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


VAL_RE = re.compile(r"ILSVRC2012_val_(\d{8})\.JPEG$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate standalone SAVLG-CBM on ImageNet val tar.")
    parser.add_argument("--artifact_dir", required=True)
    parser.add_argument("--val_tar", required=True)
    parser.add_argument("--devkit_dir", required=True)
    parser.add_argument("--output_json", default="")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log_every", type=int, default=5000)
    return parser.parse_args()


def resolve_source_run_dir(artifact_dir: Path) -> Path:
    source_run_file = artifact_dir / "source_run_dir.txt"
    if source_run_file.exists():
        source_run_dir = Path(source_run_file.read_text().strip()).resolve()
        if source_run_dir.is_dir():
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
    payload["batch_size"] = args.batch_size
    payload["workers"] = 0
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
    raise FileNotFoundError(f"no final layer artifact found under {artifact_dir}")


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
    labels = []
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


def iter_tar_samples(val_tar: Path, targets: List[int], transform: transforms.Compose):
    seen = 0
    with tarfile.open(val_tar, "r|") as tf:
        print(f"[eval] streaming tar members from {val_tar}", flush=True)
        for member in tf:
            if not member.isfile():
                continue
            match = VAL_RE.search(Path(member.name).name)
            if match is None:
                continue
            image_idx = int(match.group(1))
            target = targets[image_idx - 1]
            handle = tf.extractfile(member)
            if handle is None:
                raise FileNotFoundError(member.name)
            with Image.open(handle) as image:
                image = image.convert("RGB")
                tensor = transform(image)
            seen += 1
            yield tensor, target, member.name
    if seen != 50000:
        raise RuntimeError(f"expected 50000 val images in tar, found {seen}")


def flush_batch(
    images: List[torch.Tensor],
    targets: List[int],
    backbone: torch.nn.Module,
    head: torch.nn.Module,
    final_layer: torch.nn.Linear,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
    cfg: Config,
) -> Tuple[int, int]:
    batch = torch.stack(images, dim=0)
    batch = prepare_images(batch, cfg)
    target_tensor = torch.tensor(targets, dtype=torch.long, device=cfg.device)
    with torch.no_grad():
        with torch.autocast(
            device_type="cuda",
            dtype=amp_dtype(cfg.amp),
            enabled=(str(cfg.device).startswith("cuda") and amp_dtype(cfg.amp) is not None),
        ):
            feats = backbone(batch)
            outputs = head(feats)
            concept_logits = outputs["final_logits"].float()
            concept_logits = (concept_logits - feature_mean) / feature_std
            logits = final_layer(concept_logits)
    pred1 = logits.argmax(dim=-1)
    pred5 = logits.topk(k=min(5, logits.shape[1]), dim=-1).indices
    top1 = int((pred1 == target_tensor).sum().item())
    top5 = int((pred5 == target_tensor[:, None]).any(dim=1).sum().item())
    return top1, top5


def main() -> None:
    args = parse_args()
    artifact_dir = Path(args.artifact_dir).resolve()
    val_tar = Path(args.val_tar).resolve()
    devkit_dir = Path(args.devkit_dir).resolve()
    output_json = Path(args.output_json).resolve() if args.output_json else artifact_dir / "full_val_eval_from_tar.json"
    source_run_dir = resolve_source_run_dir(artifact_dir)

    cfg = load_run_config(source_run_dir, args)
    configure_runtime(cfg)
    print(f"[eval] artifact_dir={artifact_dir}", flush=True)
    print(f"[eval] source_run_dir={source_run_dir}", flush=True)
    print(f"[eval] val_tar={val_tar}", flush=True)
    print(f"[eval] devkit_dir={devkit_dir}", flush=True)

    concepts = [line.strip() for line in (source_run_dir / "concepts.txt").read_text().splitlines() if line.strip()]
    backbone, head = build_model(cfg, n_concepts=len(concepts))
    head.load_state_dict(torch.load(source_run_dir / "concept_head_best.pt", map_location=cfg.device))
    print(f"[eval] loaded concept head n_concepts={len(concepts)}", flush=True)

    final_layer_path = resolve_final_layer_path(artifact_dir)
    linear_payload = torch.load(final_layer_path, map_location="cpu")
    normalization_payload = torch.load(artifact_dir / "final_layer_normalization.pt", map_location="cpu")
    feature_mean = normalization_payload["mean"].to(cfg.device).float()
    feature_std = normalization_payload["std"].to(cfg.device).float().clamp_min(1e-6)
    final_layer = torch.nn.Linear(feature_mean.numel(), linear_payload["weight"].shape[0], bias=True).to(cfg.device)
    final_layer.load_state_dict(
        {
            "weight": linear_payload["weight"].to(cfg.device).float(),
            "bias": linear_payload["bias"].to(cfg.device).float(),
        }
    )

    targets = load_val_targets(devkit_dir)
    print(f"[eval] loaded {len(targets)} validation labels", flush=True)
    transform = build_transform(cfg.input_size)
    if str(cfg.device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    top1 = 0
    top5 = 0
    total = 0
    images: List[torch.Tensor] = []
    batch_targets: List[int] = []
    for image_tensor, target, name in iter_tar_samples(val_tar, targets, transform):
        images.append(image_tensor)
        batch_targets.append(target)
        if len(images) >= cfg.batch_size:
            batch_top1, batch_top5 = flush_batch(
                images,
                batch_targets,
                backbone,
                head,
                final_layer,
                feature_mean,
                feature_std,
                cfg,
            )
            top1 += batch_top1
            top5 += batch_top5
            total += len(images)
            if total % args.log_every == 0:
                elapsed = time.perf_counter() - start
                print(f"[eval] n={total} ips={total / max(elapsed, 1e-6):.2f} last={name}", flush=True)
            images.clear()
            batch_targets.clear()
    if images:
        batch_top1, batch_top5 = flush_batch(
            images,
            batch_targets,
            backbone,
            head,
            final_layer,
            feature_mean,
            feature_std,
            cfg,
        )
        top1 += batch_top1
        top5 += batch_top5
        total += len(images)
    elapsed = time.perf_counter() - start
    payload: Dict[str, Any] = {
        "artifact_dir": str(artifact_dir),
        "final_layer_path": str(final_layer_path),
        "val_tar": str(val_tar),
        "metrics": {
            "n": total,
            "top1": top1 / max(total, 1),
            "top5": top5 / max(total, 1),
            "elapsed_sec": elapsed,
            "images_per_second": total / max(elapsed, 1e-6),
        },
    }
    payload["metrics"].update(cuda_peak_stats_mb(cfg))
    output_json.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
