#!/usr/bin/env python3
"""Generate ChEX grounded concept annotations for CheXpert or MIMIC-CXR."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import tempfile
import types
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from gcbm.medical_annotations import load_concepts  # noqa: E402
from gcbm.medical_data import (  # noqa: E402
    get_medical_transforms,
    infer_chexpert_img_root,
    load_chexpert_dataset,
    load_mimic_cxr_dataset,
    medical_labels,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ChEX annotations for medical SG-CBM training.")
    parser.add_argument("--dataset", choices=["chexpert", "mimic"], required=True, help="Dataset to annotate.")
    parser.add_argument("--data_dir", required=True, help="Dataset root.")
    parser.add_argument("--concept_file", required=True, help="TXT/JSON concept or query file, e.g. concept_files/chexpert_queries_520.txt.")
    parser.add_argument("--output_dir", required=True, help="Directory where 0.json, 1.json, ... will be written.")
    parser.add_argument("--split", choices=["train", "val", "valid", "test"], default="train", help="Dataset split.")
    parser.add_argument("--csv_path", default="", help="CheXpert CSV override.")
    parser.add_argument("--img_root", default="", help="Image root override.")
    parser.add_argument("--mimic_label_csv", default="", help="MIMIC CheXpert-label CSV override.")
    parser.add_argument("--mimic_split_csv", default="", help="MIMIC split CSV override.")
    parser.add_argument("--mimic_metadata_csv", default="", help="MIMIC metadata CSV override.")
    parser.add_argument("--label_subset", choices=["all", "competition", "pathology"], default="all", help="Label subset used only for dataset targets.")
    parser.add_argument("--uncertain_strategy", choices=["ones", "zeros", "ignore"], default="ones", help="How to map uncertain labels in CSVs.")
    parser.add_argument("--frontal_only", action=argparse.BooleanOptionalAction, default=True, help="Use only frontal/AP/PA images.")
    parser.add_argument("--threshold", type=float, default=0.15, help="Detection confidence threshold saved to JSON.")
    parser.add_argument("--image_batch_size", type=int, default=8, help="Number of images to encode/detect together.")
    parser.add_argument("--concept_batch_size", type=int, default=16, help="Number of concepts encoded/detected per image pass.")
    parser.add_argument("--limit_samples", type=int, default=0, help="Optional number of rows to annotate.")
    parser.add_argument("--start_idx", type=int, default=0, help="Start row/output index for sharded or resumed runs.")
    parser.add_argument("--skip_existing", action=argparse.BooleanOptionalAction, default=True, help="Skip existing per-image JSON files.")
    parser.add_argument("--metadata_name", default="metadata.json", help="Metadata filename; useful for multi-process shards sharing one output directory.")
    parser.add_argument("--chex_src", default="/workspace/chex/src", help="Path to ChEX source directory.")
    parser.add_argument("--chex_log_dir", default="/workspace/chex_models", help="ChEX LOG_DIR containing checkpoints.")
    parser.add_argument("--chex_model_name", default="chex_stage3", help="ChEX model registry name.")
    parser.add_argument("--chex_run_name", default="run_0", help="ChEX checkpoint run name.")
    parser.add_argument("--chexzero_model_path", default="", help="Optional CheXzero image/text encoder checkpoint path override.")
    parser.add_argument("--device", default="cuda", help="Torch device.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed for metadata only; row order is not shuffled.")
    return parser.parse_args()


def patch_transformers_detr_compat() -> None:
    try:
        modeling_detr = importlib.import_module("transformers.models.detr.modeling_detr")
    except Exception:
        return
    if hasattr(modeling_detr, "center_to_corners_format"):
        return
    for module_name in ("transformers.image_transforms", "transformers.models.detr.image_processing_detr"):
        try:
            fn = getattr(importlib.import_module(module_name), "center_to_corners_format", None)
        except Exception:
            fn = None
        if fn is not None:
            setattr(modeling_detr, "center_to_corners_format", fn)
            return


def install_chex_inference_stubs() -> None:
    """Stub ChEX training/evaluation imports that are not used for annotation inference."""

    def ensure_module(name: str) -> types.ModuleType:
        if name in sys.modules:
            return sys.modules[name]
        module = types.ModuleType(name)
        sys.modules[name] = module
        return module

    training_metrics = ensure_module("metrics.training_metrics")
    training_metrics.TrainingMetrics = type("TrainingMetrics", (), {"__init__": lambda self, *args, **kwargs: None})

    train_utils = ensure_module("util.train_utils")
    train_utils.EvalConfig = type("EvalConfig", (), {})
    train_utils.Evaluator = type("Evaluator", (), {"__init__": lambda self, *args, **kwargs: None})

    plot_grounding = ensure_module("util.plot_grounding")
    plot_grounding.plot_grounding = lambda *args, **kwargs: None

    plot_utils = ensure_module("util.plot_utils")
    plot_utils.wandb_plot_text = lambda *args, **kwargs: None

    eval_classes = {
        "model.eval.anat_explainer": "AnatomyExplainerEvaluator",
        "model.eval.box_explainer": "BoxExplainerEvaluator",
        "model.eval.pathology_detection": "PathologyDetectionEvaluator",
        "model.eval.report_generation": "ReportEvaluator",
        "model.eval.sentence_grounding": "SentenceGroundingEvaluator",
    }
    for module_name, class_name in eval_classes.items():
        module = ensure_module(module_name)
        setattr(module, class_name, type(class_name, (), {"__init__": lambda self, *args, **kwargs: None}))


def build_dataset(args: argparse.Namespace):
    labels = medical_labels(args.dataset, competition=args.label_subset == "competition", pathology=args.label_subset == "pathology")
    transform = get_medical_transforms(224, train=False)
    data_dir = Path(args.data_dir)
    split = "valid" if args.split == "val" else args.split
    if args.dataset == "chexpert":
        csv_path = Path(args.csv_path) if args.csv_path else data_dir / ("valid.csv" if split == "valid" else f"{split}.csv")
        img_root = Path(args.img_root) if args.img_root else infer_chexpert_img_root(data_dir)
        return load_chexpert_dataset(csv_path, img_root=img_root, labels=labels, transform=transform, uncertain_strategy=args.uncertain_strategy, frontal_only=args.frontal_only)
    return load_mimic_cxr_dataset(
        Path(args.mimic_label_csv),
        img_root=Path(args.img_root) if args.img_root else data_dir,
        split="validate" if split == "valid" else split,
        split_csv=Path(args.mimic_split_csv) if args.mimic_split_csv else None,
        metadata_csv=Path(args.mimic_metadata_csv) if args.mimic_metadata_csv else None,
        labels=labels,
        transform=transform,
        uncertain_strategy=args.uncertain_strategy,
        frontal_only=args.frontal_only,
    )


def preprocess_image_for_chex(image_path: str, target_size: int = 224) -> Tuple[torch.Tensor, Tuple[int, int]]:
    image = Image.open(image_path).convert("L")
    orig_size = image.size
    image = image.resize((target_size, target_size), Image.BILINEAR)
    tensor = torch.from_numpy(np.array(image)).float() / 255.0
    tensor = (tensor - 0.505) / 0.248
    return tensor.unsqueeze(0).repeat(3, 1, 1), orig_size


def preprocess_images_for_chex(image_paths: Sequence[str], target_size: int = 224) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
    tensors: List[torch.Tensor] = []
    sizes: List[Tuple[int, int]] = []
    for image_path in image_paths:
        tensor, size = preprocess_image_for_chex(image_path, target_size=target_size)
        tensors.append(tensor)
        sizes.append(size)
    return torch.stack(tensors, dim=0), sizes


def load_chex_model(args: argparse.Namespace, device: torch.device):
    chex_src = Path(args.chex_src)
    if not chex_src.exists():
        raise FileNotFoundError(f"ChEX source not found: {chex_src}")
    os.environ.setdefault("LOG_DIR", args.chex_log_dir)
    sys.path.insert(0, str(chex_src))
    patch_transformers_detr_compat()
    install_chex_inference_stubs()
    if "wandb" not in sys.modules:
        wandb_mod = types.ModuleType("wandb")

        class _MissingWandbApi:
            def __init__(self, *args, **kwargs):
                raise ImportError("wandb is required only for remote ChEX run lookup; local checkpoint loading does not need it")

        wandb_mod.Api = _MissingWandbApi
        apis_mod = types.ModuleType("wandb.apis")
        public_mod = types.ModuleType("wandb.apis.public")
        public_mod.Run = object
        sys.modules["wandb"] = wandb_mod
        sys.modules["wandb.apis"] = apis_mod
        sys.modules["wandb.apis.public"] = public_mod
    cwd = Path.cwd()
    os.chdir(chex_src)
    try:
        from util.model_utils import ModelRegistry, load_model_by_name, load_model_from_checkpoint, get_model_dir, get_run_dir
        from model.chex import ChEX

        class _SingleClassRegistry:
            module_name = "model"
            model_classes = {"ChEX": ChEX}

            def get_model_class(self, model_cls_name):
                if model_cls_name not in self.model_classes:
                    raise ValueError(f"{model_cls_name} not available in the inference-only ChEX registry")
                return self.model_classes[model_cls_name]

        # The default ChEX registry scans every module under /workspace/chex/src/model,
        # including training-only evaluators and text processing packages. For annotation
        # generation we only need to instantiate the checkpoint's top-level ChEX model.
        ModelRegistry.registries["model"] = _SingleClassRegistry()
        if args.chexzero_model_path:
            chexzero_path = Path(args.chexzero_model_path)
            if not chexzero_path.exists():
                raise FileNotFoundError(f"CheXzero checkpoint override not found: {chexzero_path}")
            model_dir = get_model_dir(args.chex_model_name)
            run_dir = get_run_dir(model_dir, run_name=args.chex_run_name)
            checkpoint_dir = Path(run_dir) / "checkpoints"
            checkpoints = [path for path in checkpoint_dir.iterdir() if path.suffix == ".pth" and not path.name.endswith("_best.pth")]
            checkpoint_path = max(checkpoints, key=lambda path: path.stat().st_mtime)
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            ckpt["config_dict"]["img_encoder"]["model_path"] = str(chexzero_path)
            if "txt_encoder" in ckpt["config_dict"] and "model_path" in ckpt["config_dict"]["txt_encoder"]:
                ckpt["config_dict"]["txt_encoder"]["model_path"] = str(chexzero_path)
            patched_checkpoint = Path(tempfile.gettempdir()) / f"patched_chex_checkpoint_{os.getpid()}.pt"
            torch.save(ckpt, patched_checkpoint)
            try:
                chex_model, _ = load_model_from_checkpoint(str(patched_checkpoint), return_dict=True)
            finally:
                patched_checkpoint.unlink(missing_ok=True)
        else:
            chex_model, _ = load_model_by_name(args.chex_model_name, run_name=args.chex_run_name, load_best=False, return_dict=True)
    finally:
        os.chdir(cwd)
    return chex_model.to(device).eval()


@torch.no_grad()
def encode_concepts(model: Any, concepts: Sequence[str], output_dir: Path, args: argparse.Namespace, device: torch.device) -> torch.Tensor:
    cache_path = output_dir / "concept_features.pt"
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu")
        if payload.get("concepts") == list(concepts) and payload.get("model") == args.chex_model_name and payload.get("run_name") == args.chex_run_name:
            return payload["features"].to(device)
    features = model.txt_encoder.encode_sentences(list(concepts))
    tmp_path = cache_path.with_name(f"{cache_path.stem}_{os.getpid()}{cache_path.suffix}.tmp")
    torch.save({"features": features.cpu(), "concepts": list(concepts), "model": args.chex_model_name, "run_name": args.chex_run_name}, tmp_path)
    os.replace(tmp_path, cache_path)
    return features.to(device)


@torch.no_grad()
def annotate_image(model: Any, image_path: str, concepts: Sequence[str], concept_features: torch.Tensor, args: argparse.Namespace, device: torch.device) -> List[Dict[str, Any]]:
    return annotate_image_batch(model, [image_path], concepts, concept_features, args, device)[0]


@torch.no_grad()
def annotate_image_batch(
    model: Any,
    image_paths: Sequence[str],
    concepts: Sequence[str],
    concept_features: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> List[List[Dict[str, Any]]]:
    image_tensor, sizes = preprocess_images_for_chex(image_paths)
    image_output = model.img_encoder(image_tensor.to(device, non_blocking=True))
    batch_annotations: List[List[Dict[str, Any]]] = [[] for _ in image_paths]
    for start in range(0, len(concepts), int(args.concept_batch_size)):
        end = min(start + int(args.concept_batch_size), len(concepts))
        detector_output = model.detect_prompts(x=image_output, box_prompts_emb=concept_features[start:end], clip_boxes=True)
        if detector_output.multiboxes is None:
            continue
        boxes = detector_output.multiboxes.detach().cpu().numpy()
        weights = detector_output.multiboxes_weights.detach().cpu().numpy()
        for batch_idx, (width, height) in enumerate(sizes):
            for local_idx, concept in enumerate(concepts[start:end]):
                for box, score in zip(boxes[batch_idx, local_idx], weights[batch_idx, local_idx]):
                    if float(score) <= float(args.threshold):
                        continue
                    cx, cy, bw, bh = [float(value) for value in box]
                    batch_annotations[batch_idx].append(
                        {
                            "label": str(concept),
                            "box": [(cx - bw / 2.0) * width, (cy - bh / 2.0) * height, (cx + bw / 2.0) * width, (cy + bh / 2.0) * height],
                            "logit": float(score),
                        }
                    )
    return batch_annotations


def pending_index_batches(dataset, args: argparse.Namespace, output_dir: Path, end_idx: int) -> List[List[int]]:
    batches: List[List[int]] = []
    current: List[int] = []
    for idx in range(int(args.start_idx), end_idx):
        if args.skip_existing and (output_dir / f"{idx}.json").exists():
            continue
        current.append(idx)
        if len(current) >= int(args.image_batch_size):
            batches.append(current)
            current = []
    if current:
        batches.append(current)
    return batches


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    concepts = load_concepts(args.concept_file)
    dataset = build_dataset(args)
    model = load_chex_model(args, device)
    concept_features = encode_concepts(model, concepts, output_dir, args, device)

    end_idx = len(dataset) if args.limit_samples <= 0 else min(len(dataset), int(args.start_idx) + int(args.limit_samples))
    metadata = {
        "generated_at": datetime.now().isoformat(),
        "dataset": args.dataset,
        "split": args.split,
        "data_dir": args.data_dir,
        "csv_path": args.csv_path,
        "img_root": args.img_root,
        "concept_file": args.concept_file,
        "num_concepts": len(concepts),
        "threshold": float(args.threshold),
        "start_idx": int(args.start_idx),
        "end_idx": int(end_idx),
        "row_order_aligned": True,
        "chex_model": args.chex_model_name,
        "chex_run_name": args.chex_run_name,
        "concepts": concepts,
    }
    metadata_path = output_dir / args.metadata_name
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    total_annotations = 0
    images_with_detections = 0
    processed_now = 0
    batches = pending_index_batches(dataset, args, output_dir, end_idx)
    for batch_indices in tqdm(batches, desc="Generating ChEX annotations"):
        image_paths = [dataset.get_image_path(idx) for idx in batch_indices]
        batch_outputs = annotate_image_batch(model, image_paths, concepts, concept_features, args, device)
        for idx, image_path, annotations in zip(batch_indices, image_paths, batch_outputs):
            total_annotations += len(annotations)
            images_with_detections += int(bool(annotations))
            processed_now += 1
            (output_dir / f"{idx}.json").write_text(json.dumps([{"img_path": image_path}] + annotations), encoding="utf-8")

    metadata["processed_now"] = int(processed_now)
    metadata["skipped_existing_now"] = int(end_idx) - int(args.start_idx) - int(processed_now)
    metadata["total_annotations_now"] = int(total_annotations)
    metadata["images_with_detections_now"] = int(images_with_detections)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "processed_now": metadata["processed_now"], "total_annotations_now": total_annotations}, indent=2))


if __name__ == "__main__":
    main()
