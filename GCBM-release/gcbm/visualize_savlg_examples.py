import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from data import utils as data_utils
from methods.salf import RawSubset
from model.utils import get_bbox_iou
from methods.savlg import (
    _annotation_split_dir,
    build_savlg_concept_layer,
    collate_spatial_batch,
    compute_savlg_concept_logits,
    create_savlg_splits,
    forward_savlg_backbone,
    forward_savlg_concept_layer,
    load_spatial_supervision,
)


def _load_args(load_path: str, device: str, annotation_dir: str | None):
    with open(os.path.join(load_path, "args.txt"), "r") as f:
        payload = json.load(f)
    payload["device"] = device
    if annotation_dir is not None:
        payload["annotation_dir"] = annotation_dir
    return SimpleNamespace(**payload)


def _load_concepts(load_path: str, args) -> list[str]:
    concepts_path = os.path.join(load_path, "concepts.txt")
    if os.path.exists(concepts_path):
        with open(concepts_path, "r") as f:
            return [line.strip() for line in f if line.strip()]
    return data_utils.get_concepts(args.concept_set, getattr(args, "filter_set", None))


def _normalize_map(x: torch.Tensor) -> torch.Tensor:
    x = x.detach().float()
    x = x - x.min()
    denom = x.max().clamp_min(1e-6)
    return x / denom


def _overlay_heatmap(image_np: np.ndarray, heatmap: torch.Tensor) -> np.ndarray:
    heat = _normalize_map(heatmap).cpu().numpy()
    cmap = plt.get_cmap("jet")
    rgba = cmap(heat)[..., :3]
    overlay = 0.55 * image_np + 0.45 * rgba
    return np.clip(overlay, 0.0, 1.0)


def _parse_indices(indices_arg: str | None) -> list[int] | None:
    if not indices_arg:
        return None
    return [int(x.strip()) for x in indices_arg.split(",") if x.strip()]


def _mask_to_bbox(mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    if ys.size == 0 or xs.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _union_boxes(boxes):
    boxes = [b for b in boxes if b is not None]
    if not boxes:
        return None
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    return [int(x1), int(y1), int(x2), int(y2)]


def _load_gt_boxes(annotation_dir: str, ann_idx: int, concept_name: str):
    ann_path = os.path.join(annotation_dir, f"{int(ann_idx)}.json")
    if not os.path.exists(ann_path):
        return []
    with open(ann_path, "r") as f:
        payload = json.load(f)
    boxes = []
    for ann in payload[1:]:
        if not isinstance(ann, dict):
            continue
        label = ann.get("label")
        if isinstance(label, str):
            label = data_utils.canonicalize_concept_label(label)
        if label != concept_name:
            continue
        box = ann.get("box")
        if isinstance(box, list) and len(box) == 4:
            boxes.append([float(v) for v in box])
    return boxes


def _add_bbox(ax, box, color, label=None, linewidth=2.5):
    if box is None:
        return
    x1, y1, x2, y2 = box
    rect = Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=linewidth)
    ax.add_patch(rect)
    if label:
        ax.text(x1, max(y1 - 4, 2), label, color=color, fontsize=10, weight="bold", backgroundcolor=(1, 1, 1, 0.65))


def _select_examples(args, load_path, top_k, n_images, distinct_pred_classes=False, fixed_indices=None):
    train_raw, val_raw, train_dataset, val_dataset, test_dataset, backbone = create_savlg_splits(args)
    base_test_raw = data_utils.get_data(f"{args.dataset}_val", None)
    test_raw = RawSubset(base_test_raw, list(range(len(base_test_raw))))
    concepts = _load_concepts(load_path, args)
    class_names = data_utils.get_classes(args.dataset)
    n_concepts = len(concepts)

    concept_layer = build_savlg_concept_layer(args, backbone, n_concepts).to(args.device)
    concept_layer.load_state_dict(torch.load(os.path.join(load_path, "concept_layer.pt"), map_location=args.device))
    concept_layer.eval()
    backbone.eval()

    W_g = torch.load(os.path.join(load_path, "W_g.pt"), map_location=args.device).float()
    b_g = torch.load(os.path.join(load_path, "b_g.pt"), map_location=args.device).float()
    mean = torch.load(os.path.join(load_path, "proj_mean.pt"), map_location=args.device).float()
    std = torch.load(os.path.join(load_path, "proj_std.pt"), map_location=args.device).float()

    ann_dir = _annotation_split_dir(args.annotation_dir, args.dataset, "val")
    _, mask_entries, _ = load_spatial_supervision(
        raw_dataset=test_raw,
        annotation_dir=ann_dir,
        concepts=concepts,
        args=args,
        split_name="viz_test",
    )

    chosen = []
    seen_pred_classes = set()
    candidate_indices = fixed_indices if fixed_indices is not None else list(range(len(test_dataset)))
    with torch.no_grad():
        for idx in candidate_indices:
            image_tensor, target = test_dataset[idx]
            entry = mask_entries[idx]
            if not entry:
                continue
            feats = forward_savlg_backbone(backbone, image_tensor.unsqueeze(0).to(args.device), args)
            global_outputs, spatial_maps = forward_savlg_concept_layer(concept_layer, feats)
            global_logits, spatial_logits, final_logits = compute_savlg_concept_logits(global_outputs, spatial_maps, args)
            proj = (final_logits - mean) / std
            class_logits = proj @ W_g.t() + b_g
            pred = int(class_logits.argmax(dim=-1).item())
            if pred != int(target):
                continue
            if distinct_pred_classes and pred in seen_pred_classes:
                continue
            contrib = (W_g[pred] * proj.squeeze(0)).detach().cpu()
            valid_concepts = []
            for cidx in entry.keys():
                score = float(contrib[cidx].item())
                activation = float(final_logits[0, cidx].item())
                if score > 0 and activation > 0:
                    valid_concepts.append((score, cidx))
            if len(valid_concepts) < top_k:
                continue
            valid_concepts.sort(reverse=True)
            chosen.append(
                {
                    "idx": idx,
                    "target": int(target),
                    "pred": pred,
                    "pred_name": class_names[pred],
                    "top_concepts": [cidx for _, cidx in valid_concepts[:top_k]],
                    "top_scores": [float(s) for s, _ in valid_concepts[:top_k]],
                    "image_tensor": image_tensor,
                    "spatial_maps": spatial_maps.squeeze(0).cpu(),
                    "final_logits": final_logits.squeeze(0).cpu(),
                    "entry": entry,
                    "class_logits": class_logits.squeeze(0).cpu(),
                }
            )
            if distinct_pred_classes:
                seen_pred_classes.add(pred)
            if len(chosen) >= n_images:
                break
    return chosen, test_raw, concepts


def _render_examples(
    args,
    load_path,
    output_dir,
    n_images,
    top_k,
    distinct_pred_classes=False,
    fixed_indices=None,
    show_boxes=False,
    heatmap_threshold=0.9,
):
    selected, test_raw, concepts = _select_examples(
        args,
        load_path,
        top_k,
        n_images,
        distinct_pred_classes=distinct_pred_classes,
        fixed_indices=fixed_indices,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    ann_dir = _annotation_split_dir(args.annotation_dir, args.dataset, "val")

    for rank, item in enumerate(selected):
        raw_img, _ = test_raw[item["idx"]]
        image_np = np.asarray(raw_img.convert("RGB"), dtype=np.float32) / 255.0
        img_w, img_h = raw_img.size

        fig, axes = plt.subplots(top_k, 4, figsize=(14, 4.4 * top_k))
        if top_k == 1:
            axes = np.expand_dims(axes, 0)

        title = f"idx={item['idx']} pred={item['pred_name']} target={item['target']}"
        fig.suptitle(title, fontsize=16)

        top_records = []
        for row, (cidx, score) in enumerate(zip(item["top_concepts"], item["top_scores"])):
            concept_name = concepts[cidx]
            native_map = item["spatial_maps"][cidx]
            native_up = F.interpolate(
                native_map.unsqueeze(0).unsqueeze(0),
                size=(img_h, img_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).squeeze(0)
            overlay = _overlay_heatmap(image_np, native_up)

            target_mask = torch.from_numpy(item["entry"][cidx]).float()
            interp_logits = F.interpolate(
                native_map.unsqueeze(0).unsqueeze(0),
                size=target_mask.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).squeeze(0)
            p_k = torch.softmax(interp_logits.flatten(), dim=0).view_as(target_mask)
            q_k = target_mask / target_mask.sum().clamp_min(1e-6)

            gt_boxes = _load_gt_boxes(ann_dir, item["idx"], concept_name)
            gt_box = _union_boxes(gt_boxes)
            pred_box = None
            iou = None
            if show_boxes:
                pred_mask = (_normalize_map(native_up).cpu().numpy() >= float(heatmap_threshold)).astype(np.uint8)
                pred_box = _mask_to_bbox(pred_mask)
                if pred_box is not None and gt_box is not None:
                    iou = float(get_bbox_iou(pred_box, gt_box))

            axes[row, 0].imshow(image_np)
            axes[row, 0].set_title(f"Image\n{concept_name}")
            if show_boxes:
                _add_bbox(axes[row, 0], gt_box, color="#d62728", label="GT")
                _add_bbox(axes[row, 0], pred_box, color="#2ca02c", label="Pred")
            axes[row, 1].imshow(overlay)
            subtitle = f"Native heatmap\ncontrib={score:.3f}"
            if iou is not None:
                subtitle += f"  IoU={iou:.3f}"
            axes[row, 1].set_title(subtitle)
            if show_boxes:
                _add_bbox(axes[row, 1], gt_box, color="#d62728")
                _add_bbox(axes[row, 1], pred_box, color="#2ca02c")
            im2 = axes[row, 2].imshow(p_k.cpu().numpy(), cmap="viridis", vmin=0.0, vmax=float(max(p_k.max().item(), q_k.max().item(), 1e-6)))
            axes[row, 2].set_title(r"Predicted $p_k$")
            im3 = axes[row, 3].imshow(q_k.cpu().numpy(), cmap="viridis", vmin=0.0, vmax=float(max(p_k.max().item(), q_k.max().item(), 1e-6)))
            axes[row, 3].set_title(r"Target $q_k$")
            for ax in axes[row]:
                ax.set_xticks([])
                ax.set_yticks([])

            top_records.append(
                {
                    "concept_index": int(cidx),
                    "concept_name": concept_name,
                    "contribution": float(score),
                    "activation": float(item["final_logits"][cidx].item()),
                    "gt_box": gt_box,
                    "pred_box": pred_box,
                    "iou": iou,
                }
            )

        fig.tight_layout()
        out_path = output_dir / f"example_{rank+1:02d}_idx_{item['idx']}.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        manifest.append(
            {
                "rank": rank + 1,
                "dataset_index": int(item["idx"]),
                "pred_class": item["pred_name"],
                "pred_class_index": int(item["pred"]),
                "target_class_index": int(item["target"]),
                "figure": str(out_path),
                "top_concepts": top_records,
            }
        )

    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--annotation_dir", type=str, default="annotations")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n_images", type=int, default=5)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--distinct_pred_classes", action="store_true")
    parser.add_argument("--indices", type=str, default=None)
    parser.add_argument("--show_boxes", action="store_true")
    parser.add_argument("--heatmap_threshold", type=float, default=0.9)
    args_ns = parser.parse_args()

    model_args = _load_args(args_ns.load_path, args_ns.device, args_ns.annotation_dir)
    _render_examples(
        model_args,
        load_path=args_ns.load_path,
        output_dir=args_ns.output_dir,
        n_images=args_ns.n_images,
        top_k=args_ns.top_k,
        distinct_pred_classes=args_ns.distinct_pred_classes,
        fixed_indices=_parse_indices(args_ns.indices),
        show_boxes=args_ns.show_boxes,
        heatmap_threshold=args_ns.heatmap_threshold,
    )


if __name__ == "__main__":
    main()
