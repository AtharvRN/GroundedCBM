#!/usr/bin/env python3
"""Render high/low PartImageNet++ RMA examples from the human-mask evaluation path."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval_gdino_localization import spatial_distribution_from_map  # noqa: E402
from eval_partimagenetpp_gtbox_localization import (  # noqa: E402
    forward_concept_maps,
    load_gt_rows,
    load_model,
    rasterize_polygon_union,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vlg_path", required=True)
    parser.add_argument("--sg_path", required=True)
    parser.add_argument("--train_manifest", required=True)
    parser.add_argument("--val_manifest", required=True)
    parser.add_argument("--gt_segments_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--evaluation_map_size", type=int, default=14)
    parser.add_argument(
        "--selection_json",
        default="",
        help="Optional existing selection.json; skips the full scan and only renders these examples.",
    )
    return parser.parse_args()


def model_args(path: str, model_name: str, args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        gcbm_path=path,
        model_name=model_name,
        train_manifest=args.train_manifest,
        val_manifest=args.val_manifest,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_images=0,
    )


def resize_maps(maps: torch.Tensor, size: int) -> torch.Tensor:
    if tuple(maps.shape[-2:]) == (size, size):
        return maps
    return F.interpolate(maps, size=(size, size), mode="bilinear", align_corners=False)


def target_pairs(
    row: Dict[str, Any],
    concept_to_idx: Dict[str, int],
    grid_size: int,
) -> tuple[List[int], np.ndarray]:
    width, height = int(row["width"]), int(row["height"])
    indices: List[int] = []
    masks: List[np.ndarray] = []
    for concept, polygons in sorted(row.get("segmentations", {}).items()):
        concept_idx = concept_to_idx.get(str(concept))
        if concept_idx is None:
            continue
        mask = rasterize_polygon_union(polygons, (width, height), grid_size, grid_size)
        if mask.any():
            indices.append(concept_idx)
            masks.append(mask)
    return indices, np.stack(masks) if masks else np.zeros((0, grid_size, grid_size), dtype=bool)


def inverse_imagenet(tensor: torch.Tensor) -> np.ndarray:
    image = tensor.detach().cpu().float().permute(1, 2, 0).numpy()
    image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    return np.clip(image, 0.0, 1.0)


def select_examples(records: Sequence[Dict[str, Any]]) -> List[tuple[str, Dict[str, Any]]]:
    ordered = [
        ("Highest SG RMA", max(records, key=lambda row: row["sg_rma"])),
        ("Lowest SG RMA", min(records, key=lambda row: row["sg_rma"])),
        ("Highest VLG RMA", max(records, key=lambda row: row["vlg_rma"])),
        ("Lowest VLG RMA", min(records, key=lambda row: row["vlg_rma"])),
        ("Largest SG advantage", max(records, key=lambda row: row["sg_rma"] - row["vlg_rma"])),
        ("Largest VLG advantage", max(records, key=lambda row: row["vlg_rma"] - row["sg_rma"])),
    ]
    seen: set[tuple[int, int]] = set()
    unique: List[tuple[str, Dict[str, Any]]] = []
    for label, record in ordered:
        key = (int(record["row_index"]), int(record["concept_index"]))
        if key not in seen:
            seen.add(key)
            unique.append((label, record))
    return unique


def save_panel(
    output: Path,
    label: str,
    record: Dict[str, Any],
    image: np.ndarray,
    gt_mask: np.ndarray,
    vlg_distribution: np.ndarray,
    sg_distribution: np.ndarray,
) -> None:
    def upsample(array: np.ndarray) -> np.ndarray:
        return np.asarray(
            Image.fromarray(array.astype(np.float32), mode="F").resize((224, 224), Image.Resampling.NEAREST)
        )

    gt = upsample(gt_mask.astype(np.float32))
    # RMA is calculated from the spatial-softmax distribution, not from the
    # raw logits. Normalize only for display; each panel still shows its exact
    # probability distribution's relative spatial mass.
    vlg = upsample(vlg_distribution / max(float(vlg_distribution.max()), 1e-12))
    sg = upsample(sg_distribution / max(float(sg_distribution.max()), 1e-12))
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
    axes[0].imshow(image)
    axes[0].set_title("Evaluation-frame image")
    axes[1].imshow(image)
    axes[1].imshow(np.ma.masked_where(~gt.astype(bool), gt), cmap="Reds", alpha=0.60, vmin=0, vmax=1)
    axes[1].set_title("Human generic-part mask")
    axes[2].imshow(image)
    axes[2].imshow(vlg, cmap="magma", alpha=0.58, vmin=0, vmax=1)
    axes[2].set_title(f"VLG RMA distribution\nRMA={record['vlg_rma']:.3f}")
    axes[3].imshow(image)
    axes[3].imshow(sg, cmap="magma", alpha=0.58, vmin=0, vmax=1)
    axes[3].set_title(f"SG RMA distribution\nRMA={record['sg_rma']:.3f}")
    for axis in axes:
        axis.axis("off")
    fig.suptitle(
        f"{label}: {record['concept']} | row={record['row_index']} | "
        f"file={Path(record['file_name']).name}",
        fontsize=12,
    )
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    os.environ["PARTIMAGENETPP_TRAIN_MANIFEST"] = args.train_manifest
    os.environ["PARTIMAGENETPP_VAL_MANIFEST"] = args.val_manifest
    vlg_args, vlg_backbone, vlg_head, concepts, vlg_dataset = load_model(
        model_args(args.vlg_path, "vlg_cbm", args)
    )
    sg_args, sg_backbone, sg_layer, sg_concepts, sg_dataset = load_model(
        model_args(args.sg_path, "savlg_cbm", args)
    )
    if concepts != sg_concepts:
        raise RuntimeError("VLG and SG concept order differs; cannot compare concept maps.")
    gt_rows = load_gt_rows(Path(args.gt_segments_jsonl))
    if len(vlg_dataset) != len(sg_dataset) or len(vlg_dataset) != len(gt_rows):
        raise RuntimeError("Dataset/GT lengths do not match.")

    concept_to_idx = {concept: index for index, concept in enumerate(concepts)}
    records: List[Dict[str, Any]] = []
    if args.selection_json:
        selected = [(str(label), dict(record)) for label, record in json.loads(Path(args.selection_json).read_text())]
    else:
        loader = DataLoader(vlg_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        sg_loader = DataLoader(sg_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        cursor = 0
        with torch.no_grad():
            for (vlg_images, _), (sg_images, _) in zip(loader, sg_loader):
                vlg_maps = resize_maps(
                    forward_concept_maps("vlg_cbm", vlg_backbone, vlg_head, vlg_images.to(args.device), vlg_args),
                    args.evaluation_map_size,
                ).detach().cpu().numpy()
                sg_maps = resize_maps(
                    forward_concept_maps("savlg_cbm", sg_backbone, sg_layer, sg_images.to(args.device), sg_args),
                    args.evaluation_map_size,
                ).detach().cpu().numpy()
                for local_index in range(vlg_maps.shape[0]):
                    row_index = cursor + local_index
                    concept_indices, gt_masks = target_pairs(
                        gt_rows[row_index], concept_to_idx, args.evaluation_map_size
                    )
                    for concept_index, gt_mask in zip(concept_indices, gt_masks):
                        vlg_raw = vlg_maps[local_index, concept_index]
                        sg_raw = sg_maps[local_index, concept_index]
                        records.append(
                            {
                                "row_index": row_index,
                                "concept_index": concept_index,
                                "concept": concepts[concept_index],
                                "file_name": gt_rows[row_index]["file_name"],
                                "vlg_rma": float((spatial_distribution_from_map(vlg_raw) * gt_mask).sum()),
                                "sg_rma": float((spatial_distribution_from_map(sg_raw) * gt_mask).sum()),
                            }
                        )
                cursor += vlg_maps.shape[0]
                print(f"[rma inspect] processed={cursor}/{len(gt_rows)} pairs={len(records)}", flush=True)
        selected = select_examples(records)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, (label, record) in enumerate(selected):
        row_index = int(record["row_index"])
        concept_index = int(record["concept_index"])
        image, _ = vlg_dataset[row_index]
        concept_indices, gt_masks = target_pairs(gt_rows[row_index], concept_to_idx, args.evaluation_map_size)
        target_index = concept_indices.index(concept_index)
        gt_visual = rasterize_polygon_union(
            gt_rows[row_index]["segmentations"][record["concept"]],
            (int(gt_rows[row_index]["width"]), int(gt_rows[row_index]["height"])),
            224,
            224,
        )
        with torch.no_grad():
            vlg_map = resize_maps(
                forward_concept_maps("vlg_cbm", vlg_backbone, vlg_head, image[None].to(args.device), vlg_args),
                args.evaluation_map_size,
            )[0, concept_index].cpu().numpy()
            sg_image, _ = sg_dataset[row_index]
            sg_map = resize_maps(
                forward_concept_maps("savlg_cbm", sg_backbone, sg_layer, sg_image[None].to(args.device), sg_args),
                args.evaluation_map_size,
            )[0, concept_index].cpu().numpy()
        save_panel(
            output_dir / f"{index:02d}_{label.lower().replace(' ', '_')}.png",
            label,
            record,
            inverse_imagenet(image),
            gt_visual,
            spatial_distribution_from_map(vlg_map),
            spatial_distribution_from_map(sg_map),
        )
    (output_dir / "selection.json").write_text(json.dumps(selected, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"pairs_scanned": len(records), "selected": selected}, indent=2), flush=True)


if __name__ == "__main__":
    main()
