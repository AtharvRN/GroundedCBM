#!/usr/bin/env python3
"""Annotate a JSONL manifest with per-image GroundingDINO concept prompts."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable


DEFAULT_GDINO_ROOT = Path(os.environ.get("GROUNDINGDINO_ROOT", "GroundingDINO"))


def iter_chunks(items: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def caption_and_spans(concepts: list[str]) -> tuple[str, list[list[tuple[int, int]]]]:
    pieces: list[str] = []
    spans: list[list[tuple[int, int]]] = []
    pos = 0
    for concept in concepts:
        if pieces:
            pieces.append(" ")
            pos += 1
        start = pos
        pieces.append(concept)
        pos += len(concept)
        spans.append([(start, pos)])
        pieces.append(".")
        pos += 1
    return "".join(pieces), spans


def normalize_labels(labels: list[Any]) -> list[str]:
    normalized = [str(label).lower().strip().strip(".") for label in labels]
    return [label for idx, label in enumerate(normalized) if label and label not in normalized[:idx]]


def load_groundingdino(root: Path):
    sys.path.insert(0, str(root))
    import torch
    from PIL import Image
    from torchvision.ops import box_convert

    import groundingdino.datasets.transforms as transforms
    from groundingdino.models import build_model
    from groundingdino.util.slconfig import SLConfig
    from groundingdino.util.utils import clean_state_dict
    from groundingdino.util.misc import nested_tensor_from_tensor_list
    from groundingdino.util.vl_utils import create_positive_map_from_span

    return {
        "torch": torch,
        "Image": Image,
        "box_convert": box_convert,
        "transforms": transforms,
        "build_model": build_model,
        "SLConfig": SLConfig,
        "clean_state_dict": clean_state_dict,
        "nested_tensor_from_tensor_list": nested_tensor_from_tensor_list,
        "create_positive_map_from_span": create_positive_map_from_span,
    }


def load_model(gdino: dict[str, Any], config: Path, checkpoint: Path, device: str):
    torch = gdino["torch"]
    cfg = gdino["SLConfig"].fromfile(str(config))
    cfg.device = device
    model = gdino["build_model"](cfg)
    ckpt = torch.load(str(checkpoint), map_location="cpu")
    model.load_state_dict(gdino["clean_state_dict"](ckpt["model"]), strict=False)
    return model.eval().to(device)


def load_image(gdino: dict[str, Any], image_path: Path):
    image_pil = gdino["Image"].open(image_path).convert("RGB")
    transform = gdino["transforms"].Compose(
        [
            gdino["transforms"].RandomResize([800], max_size=1333),
            gdino["transforms"].ToTensor(),
            gdino["transforms"].Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_tensor, _ = transform(image_pil, None)
    return image_pil, image_tensor


def detect_chunk(
    gdino: dict[str, Any],
    model,
    image_tensor,
    width: int,
    height: int,
    concepts: list[str],
    box_threshold: float,
    device: str,
) -> list[dict[str, Any]]:
    torch = gdino["torch"]
    caption, spans = caption_and_spans(concepts)
    tokenized = model.tokenizer(caption)
    positive_maps = gdino["create_positive_map_from_span"](tokenized, token_span=spans).to(device)

    with torch.no_grad():
        outputs = model(image_tensor.to(device)[None], captions=[caption])

    logits = outputs["pred_logits"].sigmoid()[0]
    boxes = outputs["pred_boxes"][0]
    phrase_scores = positive_maps @ logits.T

    rows: list[dict[str, Any]] = []
    for concept_idx, concept in enumerate(concepts):
        scores = phrase_scores[concept_idx]
        keep = scores > box_threshold
        if not bool(keep.any()):
            continue
        kept_scores = scores[keep].detach().cpu()
        kept_boxes = boxes[keep].detach().cpu()
        xyxy = gdino["box_convert"](
            boxes=kept_boxes * torch.tensor([width, height, width, height]),
            in_fmt="cxcywh",
            out_fmt="xyxy",
        )
        for score, box in zip(kept_scores.tolist(), xyxy.tolist()):
            rows.append({"label": concept, "logit": round(float(score), 4), "box": [float(x) for x in box]})
    return rows


def detect_batch_chunk(
    gdino: dict[str, Any],
    model,
    image_tensors: list[Any],
    sizes: list[tuple[int, int]],
    concepts: list[str],
    box_threshold: float,
    device: str,
) -> list[list[dict[str, Any]]]:
    torch = gdino["torch"]
    caption, spans = caption_and_spans(concepts)
    tokenized = model.tokenizer(caption)
    positive_maps = gdino["create_positive_map_from_span"](tokenized, token_span=spans).to(device)
    samples = gdino["nested_tensor_from_tensor_list"]([image_tensor.to(device) for image_tensor in image_tensors]).to(device)

    with torch.no_grad():
        outputs = model(samples, captions=[caption] * len(image_tensors))

    batch_rows: list[list[dict[str, Any]]] = []
    for batch_idx, (width, height) in enumerate(sizes):
        logits = outputs["pred_logits"].sigmoid()[batch_idx]
        boxes = outputs["pred_boxes"][batch_idx]
        phrase_scores = positive_maps @ logits.T

        rows: list[dict[str, Any]] = []
        for concept_idx, concept in enumerate(concepts):
            scores = phrase_scores[concept_idx]
            keep = scores > box_threshold
            if not bool(keep.any()):
                continue
            kept_scores = scores[keep].detach().cpu()
            kept_boxes = boxes[keep].detach().cpu()
            xyxy = gdino["box_convert"](
                boxes=kept_boxes * torch.tensor([width, height, width, height]),
                in_fmt="cxcywh",
                out_fmt="xyxy",
            )
            for score, box in zip(kept_scores.tolist(), xyxy.tolist()):
                rows.append({"label": concept, "logit": round(float(score), 4), "box": [float(x) for x in box]})
        batch_rows.append(rows)
    return batch_rows


def annotate_row(
    gdino: dict[str, Any],
    model,
    row: dict[str, Any],
    box_threshold: float,
    chunk_size: int,
    device: str,
) -> list[dict[str, Any]]:
    labels = normalize_labels(row["labels"])
    image_path = Path(row["image"])
    image_pil, image_tensor = load_image(gdino, image_path)
    width, height = image_pil.size

    payload: list[dict[str, Any]] = [
        {
            "img_path": str(image_path),
            "file_name": row.get("file_name"),
            "wnid": row.get("wnid"),
            "object_name": row.get("object_name"),
            "queried_parts": labels,
            "source": "partimagenetpp_gdino_generic_parts",
        }
    ]
    detections: list[dict[str, Any]] = []
    if hasattr(model, "unset_image_tensor"):
        model.unset_image_tensor()
    try:
        for chunk in iter_chunks(labels, chunk_size):
            detections.extend(detect_chunk(gdino, model, image_tensor, width, height, chunk, box_threshold, device))
    finally:
        if hasattr(model, "unset_image_tensor"):
            model.unset_image_tensor()
    detections.sort(key=lambda item: (-float(item["logit"]), str(item["label"])))
    payload.extend(detections)
    return payload


def make_payload(row: dict[str, Any], labels: list[str], detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    image_path = Path(row["image"])
    detections.sort(key=lambda item: (-float(item["logit"]), str(item["label"])))
    return [
        {
            "img_path": str(image_path),
            "file_name": row.get("file_name"),
            "wnid": row.get("wnid"),
            "object_name": row.get("object_name"),
            "queried_parts": labels,
            "source": "partimagenetpp_gdino_generic_parts",
        },
        *detections,
    ]


def annotate_batch_same_labels(
    gdino: dict[str, Any],
    model,
    rows: list[dict[str, Any]],
    box_threshold: float,
    chunk_size: int,
    device: str,
) -> list[list[dict[str, Any]]]:
    if not rows:
        return []
    labels = normalize_labels(rows[0]["labels"])
    loaded = [load_image(gdino, Path(row["image"])) for row in rows]
    image_tensors = [image_tensor for _, image_tensor in loaded]
    sizes = [image_pil.size for image_pil, _ in loaded]

    detections_by_image: list[list[dict[str, Any]]] = [[] for _ in rows]
    if hasattr(model, "unset_image_tensor"):
        model.unset_image_tensor()
    try:
        for chunk in iter_chunks(labels, chunk_size):
            chunk_rows = detect_batch_chunk(gdino, model, image_tensors, sizes, chunk, box_threshold, device)
            for detections, extra in zip(detections_by_image, chunk_rows):
                detections.extend(extra)
    finally:
        if hasattr(model, "unset_image_tensor"):
            model.unset_image_tensor()
    return [make_payload(row, labels, detections) for row, detections in zip(rows, detections_by_image)]


def load_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def shard_rows(rows: list[dict[str, Any]], num_shards: int, shard_index: int) -> list[dict[str, Any]]:
    if num_shards <= 0:
        raise ValueError("--num_shards must be positive")
    if not 0 <= shard_index < num_shards:
        raise ValueError("--shard_index must be in [0, num_shards)")
    return [row for idx, row in enumerate(rows) if idx % num_shards == shard_index]


def iter_label_batches(rows: list[dict[str, Any]], batch_size: int) -> Iterable[list[dict[str, Any]]]:
    batch: list[dict[str, Any]] = []
    batch_key: tuple[str, ...] | None = None
    for row in rows:
        key = tuple(normalize_labels(row["labels"]))
        if batch and (key != batch_key or len(batch) >= batch_size):
            yield batch
            batch = []
        batch.append(row)
        batch_key = key
    if batch:
        yield batch


def main() -> int:
    root = DEFAULT_GDINO_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--groundingdino_root", type=Path, default=root)
    parser.add_argument("--config", type=Path, default=root / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py")
    parser.add_argument("--checkpoint", type=Path, default=root / "groundingdino_swint_ogc.pth")
    parser.add_argument("--box_threshold", type=float, default=0.25)
    parser.add_argument("--chunk_size", type=int, default=48)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--output_jsonl", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress_every", type=int, default=100)
    args = parser.parse_args()

    rows = shard_rows(load_manifest(args.manifest), args.num_shards, args.shard_index)
    if args.max_images > 0:
        rows = rows[: args.max_images]

    gdino = load_groundingdino(args.groundingdino_root)
    model = load_model(gdino, args.config, args.checkpoint, args.device)

    completed = skipped = failed = seen = 0
    if args.output_jsonl is not None:
        pending = rows
        args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        jsonl_mode = "w" if args.overwrite else "x"
        jsonl_handle = args.output_jsonl.open(jsonl_mode, encoding="utf-8")
    else:
        pending = []
        jsonl_handle = None
        for row in rows:
            out_path = Path(row["out"])
            if out_path.exists() and not args.overwrite:
                skipped += 1
            else:
                pending.append(row)

    try:
        next_report = args.progress_every if args.progress_every > 0 else 0
        for batch in iter_label_batches(pending, max(args.batch_size, 1)):
            try:
                if args.batch_size <= 1:
                    payloads = [
                        annotate_row(gdino, model, row, args.box_threshold, args.chunk_size, args.device)
                        for row in batch
                    ]
                else:
                    payloads = annotate_batch_same_labels(gdino, model, batch, args.box_threshold, args.chunk_size, args.device)
                for row, payload in zip(batch, payloads):
                    if jsonl_handle is not None:
                        jsonl_handle.write(json.dumps(payload, separators=(",", ":")) + "\n")
                    else:
                        out_path = Path(row["out"])
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                    completed += 1
                    seen += 1
            except Exception as exc:  # Keep long shards alive; failures are easy to audit later.
                for row in batch:
                    failed += 1
                    seen += 1
                    error_payload = {"row": row, "error": repr(exc)}
                    if jsonl_handle is not None:
                        jsonl_handle.write(json.dumps([error_payload], separators=(",", ":")) + "\n")
                    else:
                        out_path = Path(row["out"])
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        err_path = out_path.with_suffix(".error.json")
                        err_path.write_text(json.dumps(error_payload, indent=2), encoding="utf-8")
            if next_report and seen + skipped >= next_report:
                if jsonl_handle is not None:
                    jsonl_handle.flush()
                print(
                    f"progress shard={args.shard_index}/{args.num_shards} "
                    f"seen={seen + skipped}/{len(rows)} completed={completed} skipped={skipped} failed={failed}",
                    flush=True,
                )
                while next_report and seen + skipped >= next_report:
                    next_report += args.progress_every
    finally:
        if jsonl_handle is not None:
            jsonl_handle.close()
    print(
        f"done shard={args.shard_index}/{args.num_shards} total={len(rows)} "
        f"completed={completed} skipped={skipped} failed={failed}",
        flush=True,
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
