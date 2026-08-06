from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import torch
import torch.nn.functional as F
from PIL import Image, ImageChops, ImageEnhance, ImageFilter, ImageOps
from scipy.io import loadmat
from torchvision import transforms


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gcbm.imagenet_config import Config
from gcbm.imagenet_models import build_model
from gcbm.imagenet_nec import load_run_config as _load_sg_run_config
from gcbm.training_utils import prepare_images
from scripts.eval_salf_imagenet_nec_tar import SoftmaxPooling2D, build_backbone


DEFAULT_SG_RUNS = [
    "/workspace/sgcbm_imagenet_runs/sgcbm_imagenet_v1_conv45_10ep_seed1234",
    "/workspace/sgcbm_imagenet_runs/sgcbm_imagenet_v1_conv45_10ep_seed2024",
    "/workspace/sgcbm_imagenet_runs/sgcbm_imagenet_v1_conv45_10ep_seed6885",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate top-concept and native-region-highlight artifacts for the ImageNet rebuttal study."
    )
    parser.add_argument("--manifest_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--sg_run", action="append", default=[], help="SG-CBM run dir. Can be repeated.")
    parser.add_argument("--sg_nec_values", default="5,30")
    parser.add_argument("--salf_dir", default="/workspace/salf-cbm_models/imagenet")
    parser.add_argument("--imagenet_classes", default="/workspace/GroundedCBM/concept_files/imagenet_classes.txt")
    parser.add_argument("--devkit_dir", default="/workspace/imagenet_eval_devkit")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument(
        "--region_quantile",
        type=float,
        default=0.85,
        help="Quantile threshold on the native concept map used to define the high-activation region.",
    )
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--skip_images", action="store_true", help="Only write JSON/CSV, no highlight JPEGs.")
    return parser.parse_args()


def parse_nec_values(raw: str) -> List[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("--sg_nec_values must contain at least one integer")
    return values


def read_manifest(path: Path, max_images: int = 0) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            row = dict(row)
            row["trial_index"] = int(row["trial_index"])
            row["sample_index_0based"] = int(row["sample_index_0based"])
            row["val_index_1based"] = int(row["val_index_1based"])
            row["devkit_label_1based"] = int(row["devkit_label_1based"])
            row["class_id_0based"] = int(row["class_id_0based"])
            rows.append(row)
            if max_images and len(rows) >= int(max_images):
                break
    return rows


def load_class_names(path: Path) -> List[str]:
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(names) != 1000:
        raise ValueError(f"Expected 1000 ImageNet classes in {path}, found {len(names)}")
    return names


def load_devkit_words(devkit_dir: Path) -> Dict[str, str]:
    payload = loadmat(devkit_dir / "data" / "meta.mat", squeeze_me=True, struct_as_record=False)
    synsets = payload["synsets"]
    return {
        str(syn.WNID): str(syn.words)
        for syn in synsets
        if 1 <= int(syn.ILSVRC2012_ID) <= 1000 and int(syn.num_children) == 0
    }


def pil_transform(size: int = 224) -> transforms.Compose:
    return transforms.Compose([transforms.Resize(256), transforms.CenterCrop(size)])


def tensor_transform(size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def salf_tensor_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def safe_slug(text: str, max_len: int = 72) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    text = re.sub(r"_+", "_", text).strip("_")
    return (text or "concept")[:max_len]


def find_sg_sweep_dir(run_dir: Path, nec_values: Sequence[int]) -> Path:
    required = [f"W_g@NEC={int(nec)}.pt" for nec in nec_values] + [f"b_g@NEC={int(nec)}.pt" for nec in nec_values]
    candidates = [run_dir, *[path for path in run_dir.iterdir() if path.is_dir()]]
    matches = [path for path in candidates if all((path / name).exists() for name in required)]
    if not matches:
        raise FileNotFoundError(f"Could not find SG NEC weights {required} under {run_dir}")
    return sorted(matches, key=lambda path: path.stat().st_mtime, reverse=True)[0]


def load_sg_cfg(run_dir: Path, args: argparse.Namespace) -> Config:
    class Args:
        pass

    proxy = Args()
    proxy.device = args.device
    proxy.workers = 0
    proxy.batch_size = args.batch_size
    proxy.prefetch_factor = 2
    cfg = _load_sg_run_config(run_dir, proxy)
    cfg.batch_size = int(args.batch_size)
    cfg.workers = 0
    cfg.pin_memory = True
    cfg.persistent_workers = False
    cfg.print_config = False
    return cfg


def load_sg_model(run_dir: Path, nec_values: Sequence[int], args: argparse.Namespace) -> Dict[str, Any]:
    cfg = load_sg_cfg(run_dir, args)
    concepts = [line.strip() for line in (run_dir / "concepts.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    backbone, head = build_model(cfg, n_concepts=len(concepts))
    head.load_state_dict(torch.load(run_dir / "concept_head_best.pt", map_location=cfg.device))
    backbone.eval()
    head.eval()
    norm = torch.load(run_dir / "final_layer_normalization.pt", map_location="cpu")
    feature_mean = norm["mean"].to(cfg.device).float()
    feature_std = norm["std"].to(cfg.device).float().clamp_min(1e-6)
    sweep_dir = find_sg_sweep_dir(run_dir, nec_values)
    heads = {}
    for nec in nec_values:
        heads[int(nec)] = {
            "weight": torch.load(sweep_dir / f"W_g@NEC={int(nec)}.pt", map_location=cfg.device).float(),
            "bias": torch.load(sweep_dir / f"b_g@NEC={int(nec)}.pt", map_location=cfg.device).float(),
        }
    return {
        "cfg": cfg,
        "concepts": concepts,
        "backbone": backbone,
        "head": head,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "heads": heads,
        "sweep_dir": sweep_dir,
    }


def load_salf_model(salf_dir: Path, args: argparse.Namespace) -> Dict[str, Any]:
    device = args.device
    w_c = torch.load(salf_dir / "W_c.pt", map_location="cpu").float()
    if w_c.ndim == 2:
        w_c = w_c[:, :, None, None]
    weight = torch.load(salf_dir / "W_g.pt", map_location=device).float()
    bias = torch.load(salf_dir / "b_g.pt", map_location=device).float()
    proj_mean = torch.load(salf_dir / "proj_mean.pt", map_location=device).float().flatten()
    proj_std = torch.load(salf_dir / "proj_std.pt", map_location=device).float().flatten().clamp_min(1e-6)
    concepts = [line.strip() for line in (salf_dir / "concepts.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    backbone, _ = build_backbone("resnet50_imagenet", "layer4", device)
    return {
        "concepts": concepts,
        "backbone": backbone,
        "w_c": w_c.to(device),
        "weight": weight,
        "bias": bias,
        "proj_mean": proj_mean,
        "proj_std": proj_std,
        "pool_layer": SoftmaxPooling2D((12, 12)).to(device),
    }


def top_concepts_for_prediction(
    concepts: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    concept_names: Sequence[str],
    top_k: int,
) -> List[Dict[str, Any]]:
    logits = F.linear(concepts, weight, bias)
    preds = logits.argmax(dim=1)
    rows: List[Dict[str, Any]] = []
    for batch_idx in range(concepts.shape[0]):
        pred = int(preds[batch_idx].item())
        contributions = concepts[batch_idx] * weight[pred]
        positive = (concepts[batch_idx] > 0) & (weight[pred] > 0) & (contributions > 0)
        candidate_indices = torch.nonzero(positive, as_tuple=False).flatten()
        if candidate_indices.numel() > 0:
            candidate_scores = contributions[candidate_indices]
            order = torch.topk(candidate_scores, k=min(top_k, candidate_scores.numel())).indices
            top = candidate_indices[order].tolist()
        else:
            top = []
        rows.append(
            {
                "pred_class_id_0based": pred,
                "pred_logit": float(logits[batch_idx, pred].item()),
                "top_concepts": [
                    {
                        "rank": rank,
                        "concept_index": int(idx),
                        "concept": concept_names[int(idx)],
                        "activation": float(concepts[batch_idx, int(idx)].item()),
                        "class_weight": float(weight[pred, int(idx)].item()),
                        "contribution": float(contributions[int(idx)].item()),
                    }
                    for rank, idx in enumerate(top, start=1)
                ],
            }
        )
    return rows


def _component_containing_peak(mask: torch.Tensor, peak_row: int, peak_col: int) -> torch.Tensor:
    h, w = int(mask.shape[0]), int(mask.shape[1])
    if not bool(mask[peak_row, peak_col].item()):
        mask = mask.clone()
        mask[peak_row, peak_col] = True
    visited = torch.zeros_like(mask, dtype=torch.bool)
    component = torch.zeros_like(mask, dtype=torch.bool)
    stack = [(int(peak_row), int(peak_col))]
    visited[peak_row, peak_col] = True
    while stack:
        row, col = stack.pop()
        if not bool(mask[row, col].item()):
            continue
        component[row, col] = True
        for next_row, next_col in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            if 0 <= next_row < h and 0 <= next_col < w and not bool(visited[next_row, next_col].item()):
                visited[next_row, next_col] = True
                stack.append((next_row, next_col))
    return component


def activation_region(native_map: torch.Tensor, image_size: int, region_quantile: float) -> Dict[str, Any]:
    spatial = native_map.detach().float().cpu()
    h, w = int(spatial.shape[-2]), int(spatial.shape[-1])
    spatial = torch.nan_to_num(spatial, nan=-float("inf"))
    flat_idx = int(spatial.reshape(-1).argmax().item())
    row, col = divmod(flat_idx, w)
    finite = torch.isfinite(spatial)
    valid = spatial[finite]
    if valid.numel() == 0:
        valid = torch.zeros(1)
        spatial = torch.zeros_like(spatial)
    q = min(max(float(region_quantile), 0.0), 1.0)
    threshold = float(torch.quantile(valid, q).item())
    mask = spatial >= threshold
    component = _component_containing_peak(mask, row, col)
    if int(component.sum().item()) == 0:
        component[row, col] = True

    mask_image = Image.fromarray((component.numpy().astype("uint8") * 255), mode="L")
    mask_image = mask_image.resize((image_size, image_size), resample=Image.Resampling.NEAREST)
    bbox = mask_image.getbbox()
    if bbox is None:
        bbox = (0, 0, image_size, image_size)
    return {
        "map_shape": [h, w],
        "max_cell": [row, col],
        "max_value": float(spatial[row, col].item()),
        "region_quantile": q,
        "region_threshold": threshold,
        "region_area_cells": int(component.sum().item()),
        "box_xyxy_224": [int(value) for value in bbox],
        "mask_image": mask_image,
    }


def save_highlight_images(
    image: Image.Image,
    region: Dict[str, Any],
    out_base: Path,
    artifact_root: Path,
) -> Dict[str, str]:
    x0, y0, x1, y1 = [int(v) for v in region["box_xyxy_224"]]
    mask = region["mask_image"]
    rgb = image.convert("RGB")
    dim = ImageEnhance.Brightness(rgb).enhance(0.55)
    bright = ImageEnhance.Contrast(ImageEnhance.Brightness(rgb).enhance(1.35)).enhance(1.08)
    highlighted = Image.composite(bright, dim, mask)
    outline = ImageChops.subtract(mask.filter(ImageFilter.MaxFilter(5)), mask.filter(ImageFilter.MinFilter(5)))
    white = Image.new("RGB", rgb.size, color=(255, 255, 255))
    highlighted = Image.composite(white, highlighted, outline)
    crop = ImageOps.expand(highlighted.crop((x0, y0, x1, y1)), border=3, fill=(255, 255, 255))
    highlight_path = out_base.with_suffix(".highlight.jpg")
    crop_path = out_base.with_suffix(".crop.jpg")
    highlight_path.parent.mkdir(parents=True, exist_ok=True)
    highlighted.save(highlight_path, quality=92)
    crop.save(crop_path, quality=92)
    return {
        "highlight_path": str(highlight_path.relative_to(artifact_root)),
        "crop_path": str(crop_path.relative_to(artifact_root)),
    }


def iter_batches(rows: Sequence[Dict[str, Any]], batch_size: int) -> Iterable[List[Dict[str, Any]]]:
    for start in range(0, len(rows), batch_size):
        yield list(rows[start : start + batch_size])


def load_pil_images(rows: Sequence[Dict[str, Any]], size: int = 224) -> List[Image.Image]:
    to_pil = pil_transform(size)
    images = []
    for row in rows:
        with Image.open(row["path"]) as image:
            images.append(to_pil(image.convert("RGB")))
    return images


def make_output_base(output_dir: Path, row: Dict[str, Any], variant: str, rank: int, concept: str) -> Path:
    return (
        output_dir
        / "native_region_images"
        / variant
        / f"trial_{int(row['trial_index']):03d}_{row['filename'].replace('.JPEG', '')}"
        / f"rank{rank}_{safe_slug(concept)}"
    )


def save_original_images(rows: Sequence[Dict[str, Any]], output_dir: Path, size: int = 224) -> None:
    image_dir = output_dir / "original_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    to_pil = pil_transform(size)
    for row in rows:
        out_path = image_dir / f"trial_{int(row['trial_index']):03d}_{row['filename'].replace('.JPEG', '')}.jpg"
        if out_path.exists():
            continue
        with Image.open(row["path"]) as image:
            to_pil(image.convert("RGB")).save(out_path, quality=92)


def append_sg_results(
    rows: Sequence[Dict[str, Any]],
    model_name: str,
    model: Dict[str, Any],
    class_names: Sequence[str],
    output_dir: Path,
    args: argparse.Namespace,
    results: List[Dict[str, Any]],
) -> None:
    cfg: Config = model["cfg"]
    transform = tensor_transform(cfg.input_size)
    for batch_rows in iter_batches(rows, args.batch_size):
        pil_images = load_pil_images(batch_rows, cfg.input_size)
        tensors = []
        for row in batch_rows:
            with Image.open(row["path"]) as image:
                tensors.append(transform(image.convert("RGB")))
        batch = prepare_images(torch.stack(tensors, dim=0), cfg)
        with torch.no_grad():
            feats = model["backbone"](batch)
            outputs = model["head"](feats)
            concept_logits = outputs["final_logits"].float()
            concept_logits = (concept_logits - model["feature_mean"]) / model["feature_std"]
            spatial_maps = outputs["spatial_maps"].float()
        for nec, head_payload in model["heads"].items():
            top_rows = top_concepts_for_prediction(
                concept_logits,
                head_payload["weight"],
                head_payload["bias"],
                model["concepts"],
                args.top_k,
            )
            variant = f"{model_name}_nec{nec}"
            for batch_idx, row in enumerate(batch_rows):
                pred = int(top_rows[batch_idx]["pred_class_id_0based"])
                item = {
                    "trial_index": row["trial_index"],
                    "filename": row["filename"],
                    "path": row["path"],
                    "true_class_id_0based": row["class_id_0based"],
                    "true_class_name": row["class_name"],
                    "method": "SG-CBM",
                    "variant": variant,
                    "sg_seed": model_name.rsplit("seed", 1)[-1] if "seed" in model_name else "",
                    "nec": int(nec),
                    "pred_class_id_0based": pred,
                    "pred_class_name": class_names[pred],
                    "correct": pred == int(row["class_id_0based"]),
                    "pred_logit": top_rows[batch_idx]["pred_logit"],
                    "top_concepts": [],
                }
                for concept_info in top_rows[batch_idx]["top_concepts"]:
                    concept_idx = int(concept_info["concept_index"])
                    region_info = activation_region(
                        spatial_maps[batch_idx, concept_idx],
                        image_size=cfg.input_size,
                        region_quantile=args.region_quantile,
                    )
                    mask_image = region_info.pop("mask_image")
                    concept_payload = {**concept_info, **region_info}
                    if not args.skip_images:
                        out_base = make_output_base(output_dir, row, variant, int(concept_info["rank"]), concept_info["concept"])
                        region_info["mask_image"] = mask_image
                        concept_payload.update(
                            save_highlight_images(pil_images[batch_idx], region_info, out_base, output_dir)
                        )
                    item["top_concepts"].append(concept_payload)
                results.append(item)


def append_salf_results(
    rows: Sequence[Dict[str, Any]],
    model: Dict[str, Any],
    class_names: Sequence[str],
    output_dir: Path,
    args: argparse.Namespace,
    results: List[Dict[str, Any]],
) -> None:
    device = args.device
    transform = salf_tensor_transform()
    for batch_rows in iter_batches(rows, args.batch_size):
        pil_images = load_pil_images(batch_rows, 224)
        tensors = []
        for row in batch_rows:
            with Image.open(row["path"]) as image:
                tensors.append(transform(image.convert("RGB")))
        batch = torch.stack(tensors, dim=0).to(device, non_blocking=True)
        with torch.no_grad():
            feats = model["backbone"](batch)
            if tuple(feats.shape[-2:]) != (12, 12):
                feats = F.interpolate(feats, size=(12, 12), mode="bilinear", align_corners=False)
            maps = F.conv2d(feats, model["w_c"])
            concepts = model["pool_layer"](maps).flatten(1)
            concepts = (concepts - model["proj_mean"]) / model["proj_std"]
        top_rows = top_concepts_for_prediction(
            concepts,
            model["weight"],
            model["bias"],
            model["concepts"],
            args.top_k,
        )
        variant = "salf_official"
        for batch_idx, row in enumerate(batch_rows):
            pred = int(top_rows[batch_idx]["pred_class_id_0based"])
            item = {
                "trial_index": row["trial_index"],
                "filename": row["filename"],
                "path": row["path"],
                "true_class_id_0based": row["class_id_0based"],
                "true_class_name": row["class_name"],
                "method": "SALF-CBM",
                "variant": variant,
                "sg_seed": "",
                "nec": None,
                "pred_class_id_0based": pred,
                "pred_class_name": class_names[pred],
                "correct": pred == int(row["class_id_0based"]),
                "pred_logit": top_rows[batch_idx]["pred_logit"],
                "top_concepts": [],
            }
            for concept_info in top_rows[batch_idx]["top_concepts"]:
                concept_idx = int(concept_info["concept_index"])
                region_info = activation_region(
                    maps[batch_idx, concept_idx],
                    image_size=224,
                    region_quantile=args.region_quantile,
                )
                mask_image = region_info.pop("mask_image")
                concept_payload = {**concept_info, **region_info}
                if not args.skip_images:
                    out_base = make_output_base(output_dir, row, variant, int(concept_info["rank"]), concept_info["concept"])
                    region_info["mask_image"] = mask_image
                    concept_payload.update(
                        save_highlight_images(pil_images[batch_idx], region_info, out_base, output_dir)
                    )
                item["top_concepts"].append(concept_payload)
            results.append(item)


def write_outputs(results: Sequence[Dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "top5_concepts_and_native_regions.jsonl"
    json_path = output_dir / "top5_concepts_and_native_regions.json"
    flat_csv_path = output_dir / "top5_concepts_flat.csv"
    summary_path = output_dir / "summary.json"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    json_path.write_text(json.dumps(list(results), indent=2), encoding="utf-8")
    with flat_csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "trial_index",
            "filename",
            "method",
            "variant",
            "nec",
            "true_class_id_0based",
            "true_class_name",
            "pred_class_id_0based",
            "pred_class_name",
            "correct",
            "rank",
            "concept_index",
            "concept",
            "activation",
            "class_weight",
            "contribution",
            "map_shape",
            "max_cell",
            "max_value",
            "region_quantile",
            "region_threshold",
            "region_area_cells",
            "box_xyxy_224",
            "highlight_path",
            "crop_path",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            for concept in row["top_concepts"]:
                flat_row = {key: "" for key in fieldnames}
                for key in fieldnames:
                    if key in row:
                        flat_row[key] = row[key]
                for key in fieldnames:
                    if key in concept:
                        flat_row[key] = concept[key]
                flat_row["map_shape"] = json.dumps(concept.get("map_shape", []))
                flat_row["max_cell"] = json.dumps(concept.get("max_cell", []))
                flat_row["box_xyxy_224"] = json.dumps(concept.get("box_xyxy_224", []))
                writer.writerow(
                    flat_row
                )
    variants = sorted({str(row["variant"]) for row in results})
    by_variant = {}
    for variant in variants:
        rows = [row for row in results if row["variant"] == variant]
        by_variant[variant] = {
            "n": len(rows),
            "correct": sum(1 for row in rows if row["correct"]),
            "accuracy_on_sample": sum(1 for row in rows if row["correct"]) / max(len(rows), 1),
        }
    common_correct = {}
    for left in variants:
        for right in variants:
            if left >= right:
                continue
            left_correct = {
                int(row["trial_index"])
                for row in results
                if row["variant"] == left and bool(row["correct"])
            }
            right_correct = {
                int(row["trial_index"])
                for row in results
                if row["variant"] == right and bool(row["correct"])
            }
            common_correct[f"{left}__AND__{right}"] = sorted(left_correct & right_correct)
    summary_path.write_text(
        json.dumps(
            {
                "n_rows": len(results),
                "variants": by_variant,
                "common_correct_trial_indices": common_correct,
                "jsonl": str(jsonl_path),
                "json": str(json_path),
                "flat_csv": str(flat_csv_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"
    rows = read_manifest(Path(args.manifest_csv), args.max_images)
    output_dir = Path(args.output_dir)
    class_names = load_class_names(Path(args.imagenet_classes))
    _ = load_devkit_words(Path(args.devkit_dir))
    sg_runs = [Path(path) for path in (args.sg_run or DEFAULT_SG_RUNS)]
    sg_nec_values = parse_nec_values(args.sg_nec_values)
    results: List[Dict[str, Any]] = []

    print(f"[human-study-artifacts] rows={len(rows)} output_dir={output_dir}", flush=True)
    if not args.skip_images:
        save_original_images(rows, output_dir)
    for run_dir in sg_runs:
        model_name = f"sg_seed{run_dir.name.rsplit('seed', 1)[-1]}"
        print(f"[human-study-artifacts] loading {model_name}: {run_dir}", flush=True)
        sg_model = load_sg_model(run_dir, sg_nec_values, args)
        print(f"[human-study-artifacts] running {model_name} sweep_dir={sg_model['sweep_dir']}", flush=True)
        append_sg_results(rows, model_name, sg_model, class_names, output_dir, args, results)
        del sg_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"[human-study-artifacts] loading SALF: {args.salf_dir}", flush=True)
    salf_model = load_salf_model(Path(args.salf_dir), args)
    append_salf_results(rows, salf_model, class_names, output_dir, args, results)
    write_outputs(results, output_dir)
    print(f"[human-study-artifacts] wrote {len(results)} model-image rows to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
