import argparse
import json
import re
import sys
import tarfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.patches import Rectangle
from PIL import Image
from scipy.io import loadmat
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data import utils as data_utils
from model.cbm import Backbone, ConceptLayer
from gcbm.imagenet_annotation_index import (
    build_filename_to_annotation_path,
    load_annotation_payload,
    resolve_val_annotation_dir,
)
from gcbm.eval_imagenet_localization_helpers import normalize_maps
from gcbm.eval_imagenet_val_tar import (
    load_run_config,
    resolve_final_layer_path,
    resolve_source_run_dir,
)
from gcbm.imagenet_core import amp_dtype, build_model, configure_runtime, prepare_images


VAL_RE = re.compile(r"ILSVRC2012_val_(\d{8})\.JPEG$")

DEFAULT_GROUP_PATTERNS: Dict[str, List[str]] = {
    "dog": ["dog", "retriever", "terrier", "spaniel", "poodle", "shepherd", "husky", "beagle", "hound"],
    "car": ["car", "cab", "limousine", "jeep", "convertible", "sports car", "racer", "minivan"],
    "bird": ["bird", "partridge", "jay", "magpie", "kite", "eagle", "owl", "sparrow", "finch", "hen", "cock", "duck", "goose"],
    "food": ["pizza", "hotdog", "cheeseburger", "bagel", "pretzel", "trifle", "ice cream", "guacamole"],
    "furniture": ["chair", "sofa", "couch", "table", "lamp", "bookcase", "wardrobe", "cabinet", "desk", "bench"],
}


class SoftmaxPooling2D(torch.nn.Module):
    def __init__(self, kernel_size: Tuple[int, int]) -> None:
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
    parser = argparse.ArgumentParser(description="Render ImageNet G-CBM, SALF-CBM, and VLG-CBM Grad-CAM comparisons.")
    parser.add_argument("--val_tar", required=True)
    parser.add_argument(
        "--val_image_root",
        default="",
        help="Optional extracted ImageNet val ImageFolder root; selected images are read directly when available.",
    )
    parser.add_argument("--devkit_dir", required=True)
    parser.add_argument("--annotation_dir", required=True)
    parser.add_argument("--annotation_val_root", default="")
    parser.add_argument(
        "--annotation_mapping_json",
        default="",
        help="Optional val filename-to-annotation mapping JSON with image_name and annotation_path/file entries.",
    )
    parser.add_argument("--savlg_artifact_dir", required=True)
    parser.add_argument("--salf_dir", default="")
    parser.add_argument("--vlg_load_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image_names", default="")
    parser.add_argument("--groups", default="dog,car,bird,food,furniture")
    parser.add_argument("--concepts_per_image", type=int, default=3)
    parser.add_argument("--concept_manifest_json", default="")
    parser.add_argument("--map_normalization", default="concept_zscore_minmax", choices=["minmax", "sigmoid", "concept_zscore_minmax"])
    parser.add_argument("--paper_clean_labels", action="store_true", help="Use compact paper-facing titles without file names or scores.")
    parser.add_argument(
        "--boxes_on_maps",
        action="store_true",
        help="Also draw GDINO boxes on model heatmap columns; by default boxes are shown only on the original image.",
    )
    parser.add_argument("--savlg_display_name", default="SAVLG native")
    parser.add_argument("--salf_display_name", default="SALF native")
    parser.add_argument("--vlg_display_name", default="VLG-CBM Grad-CAM")
    parser.add_argument("--max_scan_images", type=int, default=50000)
    return parser.parse_args()


def canonicalize_concepts(path: Path) -> List[str]:
    return [data_utils.canonicalize_concept_label(line.strip()) for line in path.read_text().splitlines() if line.strip()]


def normalize_map(x: torch.Tensor) -> torch.Tensor:
    x = x.detach().float()
    x = x - x.min()
    return x / x.max().clamp_min(1e-6)


def normalize_map_with_mode(x: torch.Tensor, mode: str) -> torch.Tensor:
    return normalize_maps(x.detach().float().unsqueeze(0), mode).squeeze(0)


def overlay_heatmap(image_np: np.ndarray, heatmap: torch.Tensor) -> np.ndarray:
    heat = normalize_map(heatmap).cpu().numpy()
    rgba = plt.get_cmap("jet")(heat)[..., :3]
    return np.clip(0.58 * image_np + 0.42 * rgba, 0.0, 1.0)


def add_box(ax, box: Sequence[float], color: str, label: str = "") -> None:
    x1, y1, x2, y2 = [float(v) for v in box]
    ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=2.0))
    if label:
        ax.text(
            x1,
            max(2.0, y1 - 4.0),
            label,
            color=color,
            fontsize=8,
            weight="bold",
            backgroundcolor=(1, 1, 1, 0.68),
        )


def resize_center_crop_box(
    box: Sequence[float],
    original_size: Tuple[int, int],
    resize_short: int = 256,
    crop_size: int = 224,
) -> Optional[List[float]]:
    """Map an original-image xyxy box through Resize(short) + CenterCrop.

    GDINO annotations are stored in original-image pixel coordinates, while the
    rendered images are center-cropped model inputs. Drawing raw boxes directly
    on the crop is wrong, so this mirrors torchvision's validation transform.
    """
    orig_w, orig_h = int(original_size[0]), int(original_size[1])
    if orig_w <= 0 or orig_h <= 0:
        return None
    x1, y1, x2, y2 = [float(v) for v in box]
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.5:
        x1, x2 = x1 * orig_w, x2 * orig_w
        y1, y2 = y1 * orig_h, y2 * orig_h

    if orig_w <= orig_h:
        resized_w = int(resize_short)
        resized_h = int(resize_short * orig_h / orig_w)
    else:
        resized_h = int(resize_short)
        resized_w = int(resize_short * orig_w / orig_h)
    scale_x = resized_w / float(orig_w)
    scale_y = resized_h / float(orig_h)
    crop_left = max((resized_w - crop_size) / 2.0, 0.0)
    crop_top = max((resized_h - crop_size) / 2.0, 0.0)

    mapped = [
        x1 * scale_x - crop_left,
        y1 * scale_y - crop_top,
        x2 * scale_x - crop_left,
        y2 * scale_y - crop_top,
    ]
    clipped = [
        min(max(mapped[0], 0.0), float(crop_size)),
        min(max(mapped[1], 0.0), float(crop_size)),
        min(max(mapped[2], 0.0), float(crop_size)),
        min(max(mapped[3], 0.0), float(crop_size)),
    ]
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        return None
    return clipped


def map_record_to_display_crop(rec: Dict[str, Any], original_size: Tuple[int, int], crop_size: int) -> Dict[str, Any]:
    mapped_boxes = []
    for box in rec.get("boxes", []):
        mapped = resize_center_crop_box(box, original_size, resize_short=256, crop_size=crop_size)
        if mapped is not None:
            mapped_boxes.append(mapped)
    out = dict(rec)
    out["boxes"] = mapped_boxes
    out["original_boxes"] = rec.get("boxes", [])
    return out


def union_boxes(boxes: Sequence[Sequence[float]]) -> Optional[List[float]]:
    valid = [box for box in boxes if isinstance(box, (list, tuple)) and len(box) == 4]
    if not valid:
        return None
    return [
        min(float(box[0]) for box in valid),
        min(float(box[1]) for box in valid),
        max(float(box[2]) for box in valid),
        max(float(box[3]) for box in valid),
    ]


def load_label_words(devkit_dir: Path) -> Dict[int, str]:
    payload = loadmat(devkit_dir / "data" / "meta.mat", squeeze_me=True, struct_as_record=False)
    synsets = payload["synsets"]
    id_to_wnid: Dict[int, str] = {}
    wnid_to_words: Dict[str, str] = {}
    for syn in synsets:
        ilsvrc_id = int(syn.ILSVRC2012_ID)
        if 1 <= ilsvrc_id <= 1000 and int(syn.num_children) == 0:
            wnid = str(syn.WNID)
            id_to_wnid[ilsvrc_id] = wnid
            wnid_to_words[wnid] = str(syn.words)
    wnids = sorted(id_to_wnid.values())
    class_to_idx = {wnid: idx for idx, wnid in enumerate(wnids)}
    idx_to_words = {class_to_idx[wnid]: wnid_to_words[wnid] for wnid in wnids}
    labels: List[int] = []
    with (devkit_dir / "data" / "ILSVRC2012_validation_ground_truth.txt").open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                labels.append(class_to_idx[id_to_wnid[int(line)]])
    return {idx + 1: idx_to_words[label] for idx, label in enumerate(labels)}


def load_annotations(
    annotation_val_dir: Path,
    image_index_1based: int,
    image_name: str,
    filename_to_annotation_path: Optional[Dict[str, Path]] = None,
) -> Dict[str, Dict[str, Any]]:
    payload = load_annotation_payload(
        annotation_val_dir=annotation_val_dir,
        image_index_1based=image_index_1based,
        image_name=image_name,
        filename_to_annotation_path=filename_to_annotation_path,
    )
    entries = payload[1:] if isinstance(payload, list) else payload.get("concepts", [])
    grouped: Dict[str, Dict[str, Any]] = {}
    for ann in entries:
        if not isinstance(ann, dict):
            continue
        label = ann.get("label")
        if isinstance(label, str):
            label = data_utils.canonicalize_concept_label(label)
        if not isinstance(label, str):
            continue
        box = ann.get("box")
        if not isinstance(box, list) or len(box) != 4:
            continue
        rec = grouped.setdefault(label, {"boxes": [], "annotation_logit": float("-inf")})
        rec["boxes"].append([float(v) for v in box])
        rec["annotation_logit"] = max(float(rec["annotation_logit"]), float(ann.get("logit", 0.0)))
    return grouped


def load_filename_to_annotation_mapping(path: str, annotation_val_dir: Path) -> Optional[Dict[str, Path]]:
    if not path:
        return None
    payload = json.loads(Path(path).read_text())
    items = payload.get("items", payload if isinstance(payload, list) else [])
    mapping: Dict[str, Path] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        image_name = item.get("image_name")
        if not isinstance(image_name, str) or not image_name:
            continue
        annotation_path = item.get("annotation_path")
        if isinstance(annotation_path, str) and annotation_path:
            path_obj = Path(annotation_path)
            if not path_obj.is_file():
                path_obj = annotation_val_dir / Path(annotation_path).name
        else:
            annotation_file = item.get("annotation_file")
            if not isinstance(annotation_file, str) or not annotation_file:
                idx = item.get("annotation_index")
                if idx is None:
                    continue
                annotation_file = f"{int(idx)}.json"
            path_obj = annotation_val_dir / annotation_file
        mapping[Path(image_name).name] = path_obj
    if not mapping:
        raise RuntimeError(f"no filename-to-annotation entries loaded from {path}")
    return mapping


def load_filename_to_image_mapping(path: str, val_image_root: Optional[Path]) -> Dict[str, Path]:
    """Load selected-image paths from the mapping JSON or an extracted ImageFolder root.

    This avoids streaming the entire 6GB validation tar just to fetch a handful
    of curated examples. The mapping JSON already records `image_path` when it
    was generated from the extracted val folder; otherwise we build a lightweight
    filename index from `val_image_root`.
    """
    mapping: Dict[str, Path] = {}
    if path:
        raw = json.loads(Path(path).read_text())
        items = raw.get("items", raw if isinstance(raw, list) else [])
        for item in items:
            image_name = str(item.get("image_name", "")).strip()
            image_path = str(item.get("image_path", "")).strip()
            if not image_name or not image_path:
                continue
            path_obj = Path(image_path)
            if path_obj.is_file():
                mapping[Path(image_name).name] = path_obj
    if val_image_root and val_image_root.is_dir():
        for image_path in val_image_root.rglob("*.JPEG"):
            mapping.setdefault(image_path.name, image_path)
    return mapping


def iter_val_images(
    val_tar: Path,
    image_names: Sequence[str],
    filename_to_image_path: Optional[Dict[str, Path]] = None,
) -> Iterable[Tuple[int, str, Image.Image]]:
    requested = [name.strip() for name in image_names if name.strip()]
    remaining = set(requested)
    if filename_to_image_path:
        for image_name in requested:
            path = filename_to_image_path.get(Path(image_name).name)
            if path is None or not path.is_file():
                continue
            match = VAL_RE.search(image_name)
            if match is None:
                continue
            with Image.open(path) as image:
                yield int(match.group(1)), image_name, image.convert("RGB")
            remaining.discard(image_name)
        if not remaining:
            return

    with tarfile.open(val_tar, "r|*") as tf:
        for member in tf:
            if not member.isfile():
                continue
            image_name = Path(member.name).name
            if image_name not in remaining:
                continue
            match = VAL_RE.search(image_name)
            if match is None:
                continue
            image_index = int(match.group(1))
            handle = tf.extractfile(member)
            if handle is None:
                raise FileNotFoundError(member.name)
            with Image.open(handle) as image:
                yield image_index, image_name, image.convert("RGB")
            remaining.remove(image_name)
            if not remaining:
                break
    if remaining:
        raise FileNotFoundError(f"Could not find requested images: {sorted(remaining)}")


class VLGRenderer:
    def __init__(self, load_dir: Path, device: str) -> None:
        self.load_dir = load_dir.resolve()
        self.device = device
        with (self.load_dir / "args.txt").open("r", encoding="utf-8") as handle:
            model_args = json.load(handle)
        self.backbone = Backbone(model_args["backbone"], model_args["feature_layer"], device).eval()
        self.concepts = canonicalize_concepts(self.load_dir / "concepts.txt")
        self.concept_to_idx = {concept: idx for idx, concept in enumerate(self.concepts)}
        self.use_cbl = (self.load_dir / "cbl.pt").is_file()
        if self.use_cbl:
            self.concept_layer = ConceptLayer.from_pretrained(str(self.load_dir), device=device).eval()
            self.w_c = None
            self.proj_mean = None
            self.proj_std = None
        else:
            self.concept_layer = None
            self.w_c = torch.load(self.load_dir / "W_c.pt", map_location=device).float()
            self.proj_mean = torch.load(self.load_dir / "proj_mean.pt", map_location=device).float().flatten()
            self.proj_std = torch.load(self.load_dir / "proj_std.pt", map_location=device).float().flatten().clamp_min(1e-6)
        self.input_size = 224
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(self.input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def concept_map(self, image_tensor: torch.Tensor, concept_name: str) -> Tuple[torch.Tensor, float]:
        batch = image_tensor.unsqueeze(0).to(self.device)
        self.backbone.zero_grad(set_to_none=True)
        if self.concept_layer is not None:
            self.concept_layer.zero_grad(set_to_none=True)
        _ = self.backbone.backbone(batch)
        feats = self.backbone.feature_vals[batch.device]
        pooled = feats.mean(dim=(2, 3))
        concept_idx = self.concept_to_idx[concept_name]
        if self.use_cbl:
            logits = self.concept_layer(pooled).float().squeeze(0)
            target = logits[concept_idx]
            score = float(target.item())
        else:
            concept_logits = ((pooled @ self.w_c.T).squeeze(0) - self.proj_mean) / self.proj_std
            target = concept_logits[concept_idx]
            score = float(target.item())
        grad = torch.autograd.grad(target, feats, retain_graph=False, create_graph=False)[0]
        alpha = grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((alpha * feats).sum(dim=1, keepdim=False)).squeeze(0).detach()
        return cam, score


class SAVLGRenderer:
    def __init__(self, artifact_dir: Path, device: str) -> None:
        self.artifact_dir = artifact_dir.resolve()
        self.source_run_dir = resolve_source_run_dir(self.artifact_dir)
        args_stub = argparse.Namespace(device=device, batch_size=1)
        self.cfg = load_run_config(self.source_run_dir, args_stub)
        self.cfg.batch_size = 1
        self.cfg.workers = 0
        configure_runtime(self.cfg)
        self.concepts = [line.strip() for line in (self.source_run_dir / "concepts.txt").read_text().splitlines() if line.strip()]
        self.concept_to_idx = {concept: idx for idx, concept in enumerate(self.concepts)}
        self.backbone, self.head = build_model(self.cfg, n_concepts=len(self.concepts))
        self.head.load_state_dict(torch.load(self.source_run_dir / "concept_head_best.pt", map_location=self.cfg.device))
        self.backbone.eval()
        self.head.eval()
        weight_payload = torch.load(resolve_final_layer_path(self.artifact_dir), map_location="cpu")
        norm = torch.load(self.artifact_dir / "final_layer_normalization.pt", map_location="cpu")
        self.mean = norm["mean"].to(self.cfg.device).float()
        self.std = norm["std"].to(self.cfg.device).float().clamp_min(1e-6)
        self.weight = weight_payload["weight"].to(self.cfg.device).float()
        self.bias = weight_payload["bias"].to(self.cfg.device).float()
        self.display_size = int(self.cfg.input_size)
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(self.display_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def image_outputs(self, image: Image.Image) -> Dict[str, Any]:
        tensor = self.transform(image).unsqueeze(0)
        batch = prepare_images(tensor, self.cfg)
        with torch.no_grad():
            with torch.autocast(
                device_type="cuda",
                dtype=amp_dtype(self.cfg.amp),
                enabled=(str(self.cfg.device).startswith("cuda") and amp_dtype(self.cfg.amp) is not None),
            ):
                feats = self.backbone(batch)
                outputs = self.head(feats)
                concept_logits = ((outputs["final_logits"].float().squeeze(0) - self.mean) / self.std).float()
                class_logits = F.linear(concept_logits.unsqueeze(0), self.weight, self.bias).squeeze(0)
        pred = int(class_logits.argmax().item())
        return {
            "tensor": tensor.squeeze(0),
            "pred_class": pred,
            "pred_logit": float(class_logits[pred].item()),
            "class_logits": class_logits.detach(),
            "concept_logits": concept_logits.detach(),
            "spatial_maps": outputs["spatial_maps"].float().squeeze(0).detach(),
        }


class SALFRenderer:
    def __init__(self, load_dir: Path, device: str) -> None:
        self.load_dir = load_dir.resolve()
        self.device = device
        self.concepts = canonicalize_concepts(self.load_dir / "concepts.txt")
        self.concept_to_idx = {concept: idx for idx, concept in enumerate(self.concepts)}
        target_model, _ = data_utils.get_target_model("resnet50", device)
        self.backbone = torch.nn.Sequential(*list(target_model.children())[:-2]).to(device).eval()
        w_c = torch.load(self.load_dir / "W_c.pt", map_location=device).float()
        if w_c.ndim == 2:
            w_c = w_c[:, :, None, None]
        if w_c.ndim != 4:
            raise ValueError(f"expected SALF W_c to have 2 or 4 dims, got shape={tuple(w_c.shape)}")
        self.w_c_conv = w_c
        self.mean = torch.load(self.load_dir / "proj_mean.pt", map_location=device).float().flatten()
        self.std = torch.load(self.load_dir / "proj_std.pt", map_location=device).float().flatten().clamp_min(1e-6)
        self.pool = SoftmaxPooling2D((12, 12)).to(device)
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def concept_map(self, image_tensor: torch.Tensor, concept_name: str) -> Tuple[torch.Tensor, float]:
        batch = image_tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            feats = self.backbone(batch).float()
            if tuple(feats.shape[-2:]) != (12, 12):
                feats = F.interpolate(feats, size=(12, 12), mode="bilinear", align_corners=False)
            maps = F.conv2d(feats, self.w_c_conv)
            pooled = self.pool(maps).flatten(1).squeeze(0)
            concept_logits = (pooled - self.mean) / self.std
        concept_idx = self.concept_to_idx[concept_name]
        return maps.squeeze(0).float()[concept_idx].detach(), float(concept_logits[concept_idx].item())


def select_top_concepts(
    grouped: Dict[str, Dict[str, Any]],
    concept_to_idx: Dict[str, int],
    common_concepts: Sequence[str],
    concept_logits: torch.Tensor,
    weight: torch.Tensor,
    pred_class: int,
    top_k: int,
) -> List[Tuple[str, Dict[str, Any], int, float]]:
    rows: List[Tuple[str, Dict[str, Any], int, float]] = []
    fallback_rows: List[Tuple[str, Dict[str, Any], int, float]] = []
    for concept in common_concepts:
        if concept not in grouped:
            continue
        idx = concept_to_idx[concept]
        contrib = float((concept_logits[idx] * weight[pred_class, idx]).item())
        fallback_rows.append((concept, grouped[concept], idx, contrib))
        if contrib <= 0:
            continue
        rows.append((concept, grouped[concept], idx, contrib))
    rows.sort(key=lambda item: item[3], reverse=True)
    if rows:
        return rows[:top_k]
    fallback_rows.sort(key=lambda item: item[3], reverse=True)
    return fallback_rows[:top_k]


def render_figure(
    output_path: Path,
    image_name: str,
    class_label: str,
    image_np: np.ndarray,
    concept_rows: Sequence[Tuple[str, Dict[str, Any], int, float, Dict[str, Any]]],
    *,
    paper_clean_labels: bool,
    savlg_display_name: str,
    salf_display_name: str,
    vlg_display_name: str,
    boxes_on_maps: bool,
) -> None:
    rows = len(concept_rows)
    fig, axes = plt.subplots(rows, 4, figsize=(18, max(3.8 * rows, 4.0)), squeeze=False)
    for row, (concept, rec, _concept_idx, contribution, model_outputs) in enumerate(concept_rows):
        gt_box = union_boxes(rec["boxes"])
        axes[row, 0].imshow(image_np)
        if gt_box is not None:
            add_box(axes[row, 0], gt_box, "#d62728", "GDINO")
        if paper_clean_labels:
            short_class = class_label.split(",")[0].strip()
            axes[row, 0].set_title(f"Class: {short_class}\nConcept: {concept}", fontsize=11)
        else:
            axes[row, 0].set_title(f"{image_name}\n{class_label}\n{concept}")
        axes[row, 0].axis("off")
        axes[row, 1].imshow(model_outputs["savlg_overlay"])
        if boxes_on_maps and gt_box is not None:
            add_box(axes[row, 1], gt_box, "#d62728", "GDINO")
        if paper_clean_labels:
            axes[row, 1].set_title(savlg_display_name, fontsize=11)
        else:
            axes[row, 1].set_title(
                f"{savlg_display_name}\ncontrib={contribution:.2f}, logit={model_outputs['savlg_score']:.2f}",
                fontsize=10,
            )
        axes[row, 1].axis("off")
        axes[row, 2].imshow(model_outputs["salf_overlay"])
        if boxes_on_maps and gt_box is not None:
            add_box(axes[row, 2], gt_box, "#d62728", "GDINO")
        if paper_clean_labels:
            axes[row, 2].set_title(salf_display_name, fontsize=11)
        else:
            axes[row, 2].set_title(f"{salf_display_name}\nlogit={model_outputs['salf_score']:.2f}", fontsize=10)
        axes[row, 2].axis("off")
        axes[row, 3].imshow(model_outputs["vlg_overlay"])
        if boxes_on_maps and gt_box is not None:
            add_box(axes[row, 3], gt_box, "#d62728", "GDINO")
        if paper_clean_labels:
            axes[row, 3].set_title(vlg_display_name, fontsize=11)
        else:
            axes[row, 3].set_title(f"{vlg_display_name}\nlogit={model_outputs['vlg_score']:.2f}", fontsize=10)
        axes[row, 3].axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def choose_images(
    args: argparse.Namespace,
    annotation_val_dir: Path,
    filename_to_annotation_path: Optional[Dict[str, Path]],
    label_words: Dict[int, str],
    common_concepts: Sequence[str],
) -> List[Tuple[str, str]]:
    if args.image_names:
        names = [item.strip() for item in args.image_names.split(",") if item.strip()]
        out = []
        for name in names:
            match = VAL_RE.search(name)
            if match is None:
                raise ValueError(f"Bad image name: {name}")
            idx = int(match.group(1))
            out.append((name, label_words[idx]))
        return out

    groups = [item.strip() for item in args.groups.split(",") if item.strip()]
    found: Dict[str, Tuple[str, str]] = {}
    patterns = {group: DEFAULT_GROUP_PATTERNS[group] for group in groups if group in DEFAULT_GROUP_PATTERNS}
    with tarfile.open(Path(args.val_tar).resolve(), "r|*") as tf:
        for member in tf:
            if len(found) == len(patterns):
                break
            if not member.isfile():
                continue
            image_name = Path(member.name).name
            match = VAL_RE.search(image_name)
            if match is None:
                continue
            image_index = int(match.group(1))
            if image_index > args.max_scan_images:
                break
            label = label_words[image_index]
            grouped = load_annotations(
                annotation_val_dir,
                image_index,
                image_name,
                filename_to_annotation_path=filename_to_annotation_path,
            )
            if len(set(grouped).intersection(common_concepts)) < args.concepts_per_image:
                continue
            label_lower = label.lower()
            for group, pats in patterns.items():
                if group in found:
                    continue
                if any(p in label_lower for p in pats):
                    found[group] = (image_name, label)
                    break
    return [found[group] for group in groups if group in found]


def load_concept_manifest(path: str) -> Dict[str, Dict[str, Any]]:
    if not path:
        return {}
    raw = json.loads(Path(path).read_text())
    if isinstance(raw, dict) and "images" in raw:
        raw = raw["images"]
    manifest: Dict[str, Dict[str, Any]] = {}
    for image_name, payload in raw.items():
        if isinstance(payload, list):
            manifest[image_name] = {"concepts": payload}
        elif isinstance(payload, dict):
            manifest[image_name] = payload
        else:
            raise ValueError(f"Bad manifest entry for {image_name}: {type(payload)}")
    return manifest


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    annotation_val_dir = resolve_val_annotation_dir(Path(args.annotation_dir).resolve())
    annotation_val_root = Path(args.annotation_val_root).resolve() if args.annotation_val_root else None
    val_image_root = Path(args.val_image_root).resolve() if args.val_image_root else None
    filename_to_annotation_path = load_filename_to_annotation_mapping(args.annotation_mapping_json, annotation_val_dir)
    if filename_to_annotation_path is None:
        filename_to_annotation_path = build_filename_to_annotation_path(annotation_val_dir, annotation_val_root)
    filename_to_image_path = load_filename_to_image_mapping(args.annotation_mapping_json, val_image_root)
    label_words = load_label_words(Path(args.devkit_dir).resolve())

    savlg = SAVLGRenderer(Path(args.savlg_artifact_dir), args.device)
    salf = SALFRenderer(Path(args.salf_dir), args.device)
    vlg = VLGRenderer(Path(args.vlg_load_dir), args.device)
    common_concepts = sorted(set(savlg.concepts).intersection(salf.concepts).intersection(vlg.concepts))
    concept_manifest = load_concept_manifest(args.concept_manifest_json)

    selected = choose_images(args, annotation_val_dir, filename_to_annotation_path, label_words, common_concepts)
    selected_names = [name for name, _label in selected]
    label_lookup = {name: label for name, label in selected}
    for image_name, payload in concept_manifest.items():
        if "label" in payload:
            label_lookup[image_name] = str(payload["label"])

    summary: Dict[str, Any] = {
        "val_tar": str(Path(args.val_tar).resolve()),
        "devkit_dir": str(Path(args.devkit_dir).resolve()),
        "annotation_dir": str(annotation_val_dir),
        "val_image_root": str(val_image_root) if val_image_root else "",
        "annotation_mapping_json": str(Path(args.annotation_mapping_json).resolve()) if args.annotation_mapping_json else "",
        "savlg_artifact_dir": str(Path(args.savlg_artifact_dir).resolve()),
        "salf_dir": str(Path(args.salf_dir).resolve()),
        "vlg_load_dir": str(Path(args.vlg_load_dir).resolve()),
        "output_dir": str(output_dir),
        "images": [],
    }

    for image_index, image_name, image in iter_val_images(
        Path(args.val_tar).resolve(),
        selected_names,
        filename_to_image_path=filename_to_image_path,
    ):
        original_size = (int(image.size[0]), int(image.size[1]))
        grouped = load_annotations(
            annotation_val_dir,
            image_index,
            image_name,
            filename_to_annotation_path=filename_to_annotation_path,
        )
        grouped = {k: v for k, v in grouped.items() if k in common_concepts}
        if not grouped:
            continue

        savlg_out = savlg.image_outputs(image)
        manifest_payload = concept_manifest.get(image_name)
        if manifest_payload and manifest_payload.get("concepts"):
            top_rows = []
            for concept in manifest_payload["concepts"]:
                canon = data_utils.canonicalize_concept_label(str(concept))
                if canon not in grouped or canon not in savlg.concept_to_idx:
                    continue
                concept_idx = savlg.concept_to_idx[canon]
                contribution = float((savlg_out["concept_logits"][concept_idx] * savlg.weight[savlg_out["pred_class"], concept_idx]).item())
                top_rows.append((canon, grouped[canon], concept_idx, contribution))
        else:
            top_rows = select_top_concepts(
                grouped,
                savlg.concept_to_idx,
                common_concepts,
                savlg_out["concept_logits"],
                savlg.weight,
                savlg_out["pred_class"],
                args.concepts_per_image,
            )
        if not top_rows:
            continue
        image_np = np.asarray(transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])(image)).astype(np.float32) / 255.0
        concept_rows = []
        for concept, rec, concept_idx, contribution in top_rows:
            display_rec = map_record_to_display_crop(rec, original_size, image_np.shape[0])
            savlg_heat = F.interpolate(
                savlg_out["spatial_maps"][concept_idx].view(1, 1, *savlg_out["spatial_maps"].shape[-2:]),
                size=image_np.shape[:2],
                mode="bilinear",
                align_corners=False,
            ).squeeze()
            savlg_display = normalize_map_with_mode(savlg_heat, args.map_normalization)
            vlg_tensor = vlg.transform(image)
            vlg_cam, vlg_score = vlg.concept_map(vlg_tensor, concept)
            vlg_heat = F.interpolate(vlg_cam.view(1, 1, *vlg_cam.shape[-2:]), size=image_np.shape[:2], mode="bilinear", align_corners=False).squeeze()
            salf_tensor = salf.transform(image)
            salf_map, salf_score = salf.concept_map(salf_tensor, concept)
            salf_heat = F.interpolate(
                salf_map.view(1, 1, *salf_map.shape[-2:]),
                size=image_np.shape[:2],
                mode="bilinear",
                align_corners=False,
            ).squeeze()
            model_outputs = {
                "savlg_overlay": overlay_heatmap(image_np, savlg_display),
                "savlg_score": float(savlg_out["concept_logits"][concept_idx].item()),
                "salf_overlay": overlay_heatmap(image_np, normalize_map_with_mode(salf_heat, args.map_normalization)),
                "salf_score": float(salf_score),
                "vlg_overlay": overlay_heatmap(image_np, normalize_map(vlg_heat)),
                "vlg_score": float(vlg_score),
            }
            concept_rows.append((concept, display_rec, concept_idx, contribution, model_outputs))

        output_path = output_dir / f"{Path(image_name).stem}_savlg_vs_vlg.png"
        render_figure(
            output_path,
            image_name,
            label_lookup.get(image_name, label_words[image_index]),
            image_np,
            concept_rows,
            paper_clean_labels=args.paper_clean_labels,
            savlg_display_name=args.savlg_display_name,
            salf_display_name=args.salf_display_name,
            vlg_display_name=args.vlg_display_name,
            boxes_on_maps=args.boxes_on_maps,
        )
        summary["images"].append(
            {
                "image_index": image_index,
                "image_name": image_name,
                "label": label_lookup.get(image_name, label_words[image_index]),
                "output_path": str(output_path),
                "concepts": [
                    {
                        "concept": concept,
                        "contribution": float(contribution),
                        "savlg_logit": float(model_outputs["savlg_score"]),
                        "salf_logit": float(model_outputs["salf_score"]),
                        "vlg_logit": float(model_outputs["vlg_score"]),
                        "annotation_logit": float(rec["annotation_logit"]),
                        "boxes": rec["boxes"],
                    }
                    for concept, rec, _concept_idx, contribution, model_outputs in concept_rows
                ],
            }
        )

    (output_dir / "paper_spatial_comparison_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
