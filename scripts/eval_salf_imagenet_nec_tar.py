import argparse
import json
import sys
import tarfile
import time
from pathlib import Path
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F
from PIL import Image
from scipy.io import loadmat
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data import utils as data_utils
import clip

try:
    from transformers import ResNetForImageClassification
except ImportError:
    ResNetForImageClassification = None


VAL_PREFIX = "ILSVRC2012_val_"


class SoftmaxPooling2D(torch.nn.Module):
    def __init__(self, kernel_size: tuple[int, int]) -> None:
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, _h, _w = x.shape
        kh, kw = self.kernel_size
        patches = F.unfold(x, kernel_size=(kh, kw), stride=(kh, kw))
        patches = patches.view(n, c, kh * kw, -1)
        weights = F.softmax(patches, dim=2)
        return (patches * weights).sum(dim=2).view(n, c, 1, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate pretrained SALF-CBM ImageNet weights on val tar.")
    parser.add_argument("--load_dir", required=True, help="Directory containing W_c.pt, W_g.pt, b_g.pt, proj_mean.pt, proj_std.pt.")
    parser.add_argument("--weights_dir", default="", help="Optional directory containing exact W_g@NEC=*.pt / b_g@NEC=*.pt.")
    parser.add_argument("--val_tar", required=True)
    parser.add_argument("--devkit_dir", required=True)
    parser.add_argument("--output_json", default="")
    parser.add_argument("--backbone", default="resnet50_imagenet")
    parser.add_argument("--feature_layer", default="layer4")
    parser.add_argument("--label_order", choices=["sorted_wnid", "ilsvrc_id"], default="ilsvrc_id")
    parser.add_argument("--map_size", default="12,12")
    parser.add_argument("--pooling", choices=["avg", "softmax"], default="avg")
    parser.add_argument("--nec_values", default="5,10,15,20,25,30")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log_every", type=int, default=5000)
    parser.add_argument("--max_images", type=int, default=0, help="Optional smoke-test limit; 0 evaluates all 50000 images.")
    parser.add_argument("--save_truncated_weights", action="store_true")
    parser.add_argument("--weight_source", choices=["auto", "truncate", "exact"], default="auto")
    return parser.parse_args()


def parse_nec_values(raw: str) -> List[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("--nec_values must contain at least one integer")
    return values


def parse_map_size(raw: str) -> tuple[int, int]:
    parts = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if len(parts) != 2:
        raise ValueError("--map_size must be H,W")
    return parts[0], parts[1]


def load_val_targets(devkit_dir: Path, label_order: str) -> List[int]:
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
                raw_id = int(line)
                if label_order == "ilsvrc_id":
                    labels.append(raw_id - 1)
                else:
                    labels.append(class_to_idx[id_to_wnid[raw_id]])
    if len(labels) != 50000:
        raise RuntimeError(f"expected 50000 validation labels, got {len(labels)}")
    return labels


def build_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def build_showandtell_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def iter_tar_samples(val_tar: Path, targets: Sequence[int], transform: transforms.Compose):
    seen = 0
    with tarfile.open(val_tar, "r|*") as tf:
        for member in tf:
            if not member.isfile():
                continue
            image_name = Path(member.name).name
            if not (image_name.startswith(VAL_PREFIX) and image_name.endswith(".JPEG")):
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


def truncate_per_class(weight: torch.Tensor, nec: int) -> torch.Tensor:
    k = max(0, min(int(nec), int(weight.shape[1])))
    if k == 0:
        return torch.zeros_like(weight)
    if k == int(weight.shape[1]):
        return weight.clone().detach()
    topk_idx = weight.abs().topk(k=k, dim=1).indices
    sparse_weight = torch.zeros_like(weight)
    sparse_weight.scatter_(1, topk_idx, weight.gather(1, topk_idx))
    return sparse_weight


def build_clip_rn50_spatial_backbone(device: str):
    model, preprocess = clip.load("RN50", device=device)
    visual = model.visual.float().eval()

    def forward(images: torch.Tensor) -> torch.Tensor:
        x = images.to(device)
        x = x.type(visual.conv1.weight.dtype)
        x = visual.relu1(visual.bn1(visual.conv1(x)))
        x = visual.relu2(visual.bn2(visual.conv2(x)))
        x = visual.relu3(visual.bn3(visual.conv3(x)))
        x = visual.avgpool(x)
        x = visual.layer1(x)
        x = visual.layer2(x)
        x = visual.layer3(x)
        x = visual.layer4(x)
        return x.float()

    return forward, preprocess


def build_backbone(backbone_name: str, feature_layer: str, device: str):
    if backbone_name == "clip_RN50":
        return build_clip_rn50_spatial_backbone(device)
    if backbone_name == "resnet50_imagenet":
        if ResNetForImageClassification is None:
            raise ImportError(
                "transformers is required for --backbone resnet50_imagenet. "
                "Use --backbone resnet50 to evaluate with torchvision ResNet50."
            )
        target_model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50").to(device)
        target_model.eval()
        backbone = torch.nn.Sequential(*list(target_model.resnet.children())[:-1]).to(device).eval()

        def forward(images: torch.Tensor) -> torch.Tensor:
            return backbone(images).last_hidden_state.float()

        return forward, build_showandtell_transform()

    model, _ = data_utils.get_target_model(backbone_name, device)
    feature_vals: Dict[torch.device, torch.Tensor] = {}

    def hook(_module, _input, output):
        feature_vals[output.device] = output

    target_module = model
    for part in feature_layer.split("."):
        target_module = getattr(target_module, part)
    target_module.register_forward_hook(hook)
    model.eval()

    def forward(images: torch.Tensor) -> torch.Tensor:
        out = model(images)
        return feature_vals[out.device].float()

    return forward, build_transform()


def evaluate(args: argparse.Namespace) -> Dict[str, object]:
    load_dir = Path(args.load_dir).resolve()
    weights_dir = Path(args.weights_dir).resolve() if args.weights_dir else load_dir
    device = args.device

    w_c = torch.load(load_dir / "W_c.pt", map_location="cpu").float()
    weight = torch.load(load_dir / "W_g.pt", map_location="cpu").float()
    bias = torch.load(load_dir / "b_g.pt", map_location="cpu").float()
    proj_mean = torch.load(load_dir / "proj_mean.pt", map_location=device).float().flatten()
    proj_std = torch.load(load_dir / "proj_std.pt", map_location=device).float().flatten().clamp_min(1e-6)
    map_size = parse_map_size(args.map_size)
    pool_layer = SoftmaxPooling2D(map_size).to(device) if args.pooling == "softmax" else None

    nec_values = parse_nec_values(args.nec_values)
    sparse_items = []
    for nec in nec_values:
        exact_weight_path = weights_dir / f"W_g@NEC={int(nec)}.pt"
        exact_bias_path = weights_dir / f"b_g@NEC={int(nec)}.pt"
        use_exact = False
        if args.weight_source == "exact":
            if not exact_weight_path.exists() or not exact_bias_path.exists():
                raise FileNotFoundError(
                    f"Missing exact NEC weights for NEC={nec}: {exact_weight_path} / {exact_bias_path}"
                )
            use_exact = True
        elif args.weight_source == "auto":
            use_exact = exact_weight_path.exists() and exact_bias_path.exists()

        if use_exact:
            truncated = torch.load(exact_weight_path, map_location="cpu").float()
            sparse_bias = torch.load(exact_bias_path, map_location="cpu").float()
        else:
            truncated = truncate_per_class(weight, nec)
            sparse_bias = bias
            if args.save_truncated_weights:
                torch.save(truncated.cpu(), weights_dir / f"W_g@NEC={int(nec)}.per_class.pt")
                torch.save(bias.cpu(), weights_dir / f"b_g@NEC={int(nec)}.per_class.pt")
        sparse_items.append(
            {
                "nec": int(nec),
                "weight": truncated,
                "bias": sparse_bias,
                "nnz": int((truncated.abs() > 1e-8).sum().item()),
                "total": int(truncated.numel()),
                "weight_source": "exact" if use_exact else "truncate",
            }
        )

    dense_weight = weight.to(device)
    dense_bias = bias.to(device)
    sparse_weights = torch.stack([item["weight"].to(device) for item in sparse_items], dim=0)
    sparse_biases = torch.stack([item["bias"].to(device) for item in sparse_items], dim=0)
    if w_c.ndim == 2:
        w_c = w_c[:, :, None, None]
    if w_c.ndim != 4:
        raise ValueError(f"expected SALF W_c to have 2 or 4 dims, got shape={tuple(w_c.shape)}")
    w_c = w_c.to(device)
    w_c_linear = w_c.squeeze(-1).squeeze(-1) if w_c.ndim == 4 else w_c

    backbone, transform = build_backbone(args.backbone, args.feature_layer, device)
    targets = load_val_targets(Path(args.devkit_dir).resolve(), args.label_order)

    dense_top1 = dense_top5 = 0
    sparse_top1 = torch.zeros(len(sparse_items), dtype=torch.long)
    sparse_top5 = torch.zeros(len(sparse_items), dtype=torch.long)
    total = 0
    next_log = max(int(args.log_every), 1)
    start = time.perf_counter()
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    images: List[torch.Tensor] = []
    labels: List[int] = []

    def flush(last_name: str | None) -> None:
        nonlocal dense_top1, dense_top5, next_log, total
        if not images:
            return
        batch = torch.stack(images, dim=0).to(device, non_blocking=True)
        target = torch.tensor(labels, dtype=torch.long, device=device)
        with torch.no_grad():
            feats = backbone(batch)
            if feats.ndim == 4 and w_c.ndim == 4:
                if tuple(feats.shape[-2:]) != map_size:
                    feats = F.interpolate(feats, size=map_size, mode="bilinear", align_corners=False)
                maps = F.conv2d(feats, w_c)
                if pool_layer is not None:
                    concepts = pool_layer(maps).flatten(1)
                else:
                    concepts = F.adaptive_avg_pool2d(maps, 1).flatten(1)
            elif feats.ndim == 4:
                pooled_feats = F.adaptive_avg_pool2d(feats, 1).flatten(1)
                concepts = F.linear(pooled_feats, w_c_linear)
            else:
                concepts = F.linear(torch.flatten(feats, 1), w_c_linear)
            concepts = (concepts - proj_mean) / proj_std

            dense_logits = F.linear(concepts, dense_weight, dense_bias)
            dense_top1 += int(dense_logits.argmax(dim=1).eq(target).sum().item())
            dense_top5 += int(dense_logits.topk(k=5, dim=1).indices.eq(target[:, None]).any(dim=1).sum().item())

            sparse_logits = torch.einsum("bc,koc->kbo", concepts, sparse_weights) + sparse_biases[:, None, :]
            pred1 = sparse_logits.argmax(dim=-1)
            pred5 = sparse_logits.topk(k=5, dim=-1).indices
            sparse_top1.add_(pred1.eq(target.unsqueeze(0)).sum(dim=1).cpu())
            sparse_top5.add_(pred5.eq(target.view(1, -1, 1)).any(dim=-1).sum(dim=1).cpu())

        total += len(images)
        if total >= next_log:
            elapsed = time.perf_counter() - start
            print(f"[salf-imagenet-eval] n={total} ips={total / max(elapsed, 1e-6):.2f} last={last_name}", flush=True)
            while next_log <= total:
                next_log += max(int(args.log_every), 1)
        images.clear()
        labels.clear()

    for image, target, name in iter_tar_samples(Path(args.val_tar).resolve(), targets, transform):
        images.append(image)
        labels.append(int(target))
        if len(images) >= int(args.batch_size):
            flush(name)
        if int(args.max_images) > 0 and total + len(images) >= int(args.max_images):
            flush(name)
            break
    flush(None)

    elapsed = time.perf_counter() - start
    nec_results = []
    for idx, item in enumerate(sparse_items):
        nec_results.append(
            {
                "nec": item["nec"],
                "nnz": item["nnz"],
                "total": item["total"],
                "weight_sparsity": 1.0 - item["nnz"] / max(item["total"], 1),
                "weight_source": item["weight_source"],
                "top1": float(sparse_top1[idx].item() / max(total, 1)),
                "top5": float(sparse_top5[idx].item() / max(total, 1)),
            }
        )

    avg_acc = float(sum(item["top1"] for item in nec_results) / max(len(nec_results), 1))
    payload: Dict[str, object] = {
        "load_dir": str(load_dir),
        "val_tar": str(Path(args.val_tar).resolve()),
        "devkit_dir": str(Path(args.devkit_dir).resolve()),
        "backbone": args.backbone,
        "feature_layer": args.feature_layer,
        "label_order": args.label_order,
        "map_size": list(map_size),
        "pooling": args.pooling,
        "weights_dir": str(weights_dir),
        "weight_source": args.weight_source,
        "n_concepts": int(weight.shape[1]),
        "n": total,
        "elapsed_sec": elapsed,
        "images_per_second": total / max(elapsed, 1e-6),
        "dense": {
            "top1": float(dense_top1 / max(total, 1)),
            "top5": float(dense_top5 / max(total, 1)),
            "nnz": int((weight.abs() > 1e-8).sum().item()),
            "total": int(weight.numel()),
            "weight_sparsity": 1.0 - int((weight.abs() > 1e-8).sum().item()) / max(int(weight.numel()), 1),
        },
        "nec_values": nec_values,
        "nec_results": nec_results,
        "acc_at_nec5": next((item["top1"] for item in nec_results if item["nec"] == 5), None),
        "avg_acc": avg_acc,
    }
    if str(device).startswith("cuda") and torch.cuda.is_available():
        payload["max_memory_allocated_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024)
        payload["max_memory_reserved_mb"] = torch.cuda.max_memory_reserved() / (1024 * 1024)
    return payload


def main() -> None:
    args = parse_args()
    payload = evaluate(args)
    output_json = Path(args.output_json).resolve() if args.output_json else Path(args.load_dir).resolve() / "salf_imagenet_eval.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
