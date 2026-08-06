#!/usr/bin/env python3
"""Evaluate PartImageNet++ GroundingDINO annotations against human GT part boxes."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval_concept_accuracy import (  # noqa: E402
    macro_average_precision,
    precision_at_k,
    safe_average_precision,
    safe_roc_auc,
)
from eval_gdino_localization import rasterize_box_union  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved GDINO PartImageNet++ val annotations.")
    parser.add_argument("--annotation_dir", required=True, help="Directory with val GDINO JSON files named <row_index>.json.")
    parser.add_argument("--gt_boxes_jsonl", required=True, help="Human GT boxes JSONL in val manifest order.")
    parser.add_argument("--concept_file", required=True, help="Concept bank to evaluate against.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--score_thresholds", default="0.1,0.15,0.2,0.3,0.5")
    parser.add_argument("--box_iou_thresholds", default="0.1,0.3,0.5")
    parser.add_argument("--mask_size", type=int, default=224)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=1000)
    return parser.parse_args()


def parse_float_list(raw: str) -> List[float]:
    return [float(x.strip()) for x in str(raw).split(",") if x.strip()]


def canonicalize(text: str) -> str:
    from data import utils as data_utils

    return data_utils.canonicalize_concept_label(str(text))


def load_concepts(path: Path) -> List[str]:
    return [canonicalize(line.strip()) for line in path.read_text().splitlines() if line.strip()]


def load_gt_rows(path: Path, max_images: int = 0) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if max_images > 0 and len(rows) >= max_images:
                    break
    return rows


def parse_gdino_items(payload: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload[1:] if isinstance(row, dict)]
    if isinstance(payload, dict):
        items = payload.get("concepts", payload.get("detections", []))
        if isinstance(items, list):
            return [row for row in items if isinstance(row, dict)]
    return []


def load_gdino_annotations(
    annotation_dir: Path,
    n_images: int,
    concept_to_idx: Dict[str, int],
) -> Tuple[
    np.ndarray,
    Dict[int, Dict[int, List[Tuple[float, List[float]]]]],
    Dict[int, List[Tuple[float, int, List[float]]]],
    Dict[str, Any],
]:
    scores = np.zeros((n_images, len(concept_to_idx)), dtype=np.float32)
    preds_by_image: Dict[int, Dict[int, List[Tuple[float, List[float]]]]] = {}
    preds_by_concept: Dict[int, List[Tuple[float, int, List[float]]]] = defaultdict(list)
    missing_files = 0
    total_detections = 0
    matched_detections = 0
    for row_idx in range(n_images):
        path = annotation_dir / f"{row_idx}.json"
        if not path.is_file():
            missing_files += 1
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            missing_files += 1
            continue
        by_concept: Dict[int, List[Tuple[float, List[float]]]] = defaultdict(list)
        for item in parse_gdino_items(payload):
            total_detections += 1
            concept_idx = concept_to_idx.get(canonicalize(item.get("label", item.get("concept", item.get("name", "")))))
            if concept_idx is None:
                continue
            box = item.get("box")
            if not isinstance(box, list) or len(box) != 4:
                continue
            try:
                score = float(item.get("logit", item.get("score", item.get("confidence", 0.0))))
                pred_box = [float(v) for v in box]
            except Exception:
                continue
            matched_detections += 1
            if score > scores[row_idx, concept_idx]:
                scores[row_idx, concept_idx] = score
            by_concept[concept_idx].append((score, pred_box))
            preds_by_concept[concept_idx].append((score, row_idx, pred_box))
        preds_by_image[row_idx] = dict(by_concept)
    summary = {
        "missing_annotation_files": missing_files,
        "total_detections": total_detections,
        "matched_concept_detections": matched_detections,
    }
    return scores, preds_by_image, dict(preds_by_concept), summary


def build_gt_presence_and_boxes(
    gt_rows: Sequence[Dict[str, Any]],
    concept_to_idx: Dict[str, int],
) -> Tuple[np.ndarray, Dict[int, Dict[int, List[List[float]]]], Dict[int, int], Dict[str, Any]]:
    gt = np.zeros((len(gt_rows), len(concept_to_idx)), dtype=np.float32)
    boxes_by_image: Dict[int, Dict[int, List[List[float]]]] = {}
    gt_count_by_concept: Dict[int, int] = defaultdict(int)
    skipped_boxes = 0
    for row_idx, row in enumerate(gt_rows):
        concept_boxes: Dict[int, List[List[float]]] = defaultdict(list)
        for raw_concept, boxes in (row.get("boxes") or {}).items():
            concept_idx = concept_to_idx.get(canonicalize(raw_concept))
            if concept_idx is None:
                skipped_boxes += len(boxes) if isinstance(boxes, list) else 0
                continue
            if not isinstance(boxes, list):
                continue
            valid_boxes: List[List[float]] = []
            for box in boxes:
                if isinstance(box, list) and len(box) == 4:
                    valid_boxes.append([float(v) for v in box])
            if valid_boxes:
                gt[row_idx, concept_idx] = 1.0
                concept_boxes[concept_idx].extend(valid_boxes)
                gt_count_by_concept[concept_idx] += len(valid_boxes)
        boxes_by_image[row_idx] = dict(concept_boxes)
    summary = {
        "skipped_gt_boxes_not_in_concept_bank": skipped_boxes,
        "gt_positive_pairs": int(gt.sum()),
        "gt_box_instances_in_concept_bank": int(sum(gt_count_by_concept.values())),
    }
    return gt, boxes_by_image, dict(gt_count_by_concept), summary


def confusion_metrics(gt: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    gt_bool = gt.astype(bool)
    pred_bool = pred.astype(bool)
    tp = int((gt_bool & pred_bool).sum())
    fp = int((~gt_bool & pred_bool).sum())
    fn = int((gt_bool & ~pred_bool).sum())
    tn = int((~gt_bool & ~pred_bool).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    accuracy = (tp + tn) / max(tp + fp + fn + tn, 1)
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def point_in_any_box(point: Tuple[float, float], boxes: Sequence[Sequence[float]]) -> bool:
    x, y = point
    return any(float(b[0]) <= x <= float(b[2]) and float(b[1]) <= y <= float(b[3]) for b in boxes)


def detection_average_precision(
    preds_by_concept: Dict[int, List[Tuple[float, int, List[float]]]],
    gt_boxes_by_image: Dict[int, Dict[int, List[List[float]]]],
    concept_indices: Sequence[int],
    iou_threshold: float,
) -> Dict[str, Any]:
    ap_values: List[float] = []
    per_concept: Dict[str, Dict[str, float]] = {}
    for concept_idx in concept_indices:
        num_gt = sum(len(by_concept.get(concept_idx, [])) for by_concept in gt_boxes_by_image.values())
        if num_gt <= 0:
            continue
        detections = list(preds_by_concept.get(concept_idx, []))
        detections.sort(key=lambda row: row[0], reverse=True)
        matched: Dict[int, set[int]] = defaultdict(set)
        tp: List[float] = []
        fp: List[float] = []
        for _score, image_idx, pred_box in detections:
            gt_boxes = gt_boxes_by_image.get(image_idx, {}).get(concept_idx, [])
            best_iou = 0.0
            best_gt = -1
            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_idx in matched[image_idx]:
                    continue
                iou = box_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt = gt_idx
            if best_iou >= float(iou_threshold) and best_gt >= 0:
                matched[image_idx].add(best_gt)
                tp.append(1.0)
                fp.append(0.0)
            else:
                tp.append(0.0)
                fp.append(1.0)
        if not detections:
            ap = 0.0
        else:
            tp_cum = np.cumsum(np.asarray(tp, dtype=np.float64))
            fp_cum = np.cumsum(np.asarray(fp, dtype=np.float64))
            recall = tp_cum / max(float(num_gt), 1.0)
            precision = tp_cum / np.maximum(tp_cum + fp_cum, 1.0)
            recall_prev = np.concatenate([[0.0], recall[:-1]])
            ap = float(np.sum((recall - recall_prev) * precision))
        ap_values.append(ap)
        per_concept[str(concept_idx)] = {"ap": ap, "num_gt": float(num_gt), "num_pred": float(len(detections))}
    return {
        "map": float(np.mean(ap_values)) if ap_values else float("nan"),
        "num_concepts_with_gt": len(ap_values),
        "per_concept": per_concept,
    }


def localization_at_threshold(
    preds_by_image: Dict[int, Dict[int, List[Tuple[float, List[float]]]]],
    gt_boxes_by_image: Dict[int, Dict[int, List[List[float]]]],
    gt_rows: Sequence[Dict[str, Any]],
    score_threshold: float,
    box_iou_thresholds: Sequence[float],
    mask_size: int,
) -> Dict[str, Any]:
    total = 0
    with_pred = 0
    mass_sum = 0.0
    point_sum = 0.0
    mask_iou_sum = 0.0
    box_hits = {str(t): 0 for t in box_iou_thresholds}
    for image_idx, by_concept in gt_boxes_by_image.items():
        row = gt_rows[image_idx]
        image_size = (int(row.get("width") or 0), int(row.get("height") or 0))
        for concept_idx, gt_boxes in by_concept.items():
            total += 1
            pred_items = [
                (score, box)
                for score, box in preds_by_image.get(image_idx, {}).get(concept_idx, [])
                if float(score) >= float(score_threshold)
            ]
            pred_boxes = [box for _score, box in pred_items]
            if pred_boxes:
                with_pred += 1
                top_score, top_box = max(pred_items, key=lambda row: row[0])
                cx = 0.5 * (float(top_box[0]) + float(top_box[2]))
                cy = 0.5 * (float(top_box[1]) + float(top_box[3]))
                point_sum += float(point_in_any_box((cx, cy), gt_boxes))
                max_iou = max(box_iou(pred_box, gt_box) for pred_box in pred_boxes for gt_box in gt_boxes)
            else:
                max_iou = 0.0
            for tau in box_iou_thresholds:
                box_hits[str(tau)] += int(max_iou >= float(tau))

            gt_mask = rasterize_box_union(gt_boxes, image_size=image_size, map_h=mask_size, map_w=mask_size)
            pred_mask = rasterize_box_union(pred_boxes, image_size=image_size, map_h=mask_size, map_w=mask_size)
            inter = np.logical_and(gt_mask, pred_mask).sum()
            union = np.logical_or(gt_mask, pred_mask).sum()
            mask_iou_sum += float(inter / max(union, 1))
            pred_area = int(pred_mask.sum())
            mass_sum += float(inter / pred_area) if pred_area > 0 else 0.0
    denom = max(total, 1)
    return {
        "instances": total,
        "instances_with_prediction": with_pred,
        "prediction_rate": with_pred / denom,
        "mass_in_gt": mass_sum / denom,
        "point_hit": point_sum / denom,
        "mask_iou": mask_iou_sum / denom,
        "box_acc": {str(t): box_hits[str(t)] / denom for t in box_iou_thresholds},
    }


def main() -> None:
    args = parse_args()
    start = time.time()
    score_thresholds = parse_float_list(args.score_thresholds)
    box_iou_thresholds = parse_float_list(args.box_iou_thresholds)
    concepts = load_concepts(Path(args.concept_file))
    concept_to_idx = {concept: idx for idx, concept in enumerate(concepts)}
    gt_rows = load_gt_rows(Path(args.gt_boxes_jsonl), max_images=int(args.max_images))
    print(f"[pinpp gdino gt] loaded gt rows={len(gt_rows)} concepts={len(concepts)}", flush=True)
    gt, gt_boxes_by_image, gt_count_by_concept, gt_summary = build_gt_presence_and_boxes(gt_rows, concept_to_idx)
    scores, preds_by_image, preds_by_concept, pred_summary = load_gdino_annotations(
        Path(args.annotation_dir),
        len(gt_rows),
        concept_to_idx,
    )
    print(
        "[pinpp gdino gt] parsed annotations "
        f"detections={pred_summary['matched_concept_detections']} gt_pairs={gt_summary['gt_positive_pairs']}",
        flush=True,
    )
    flat_gt = gt.reshape(-1)
    flat_scores = scores.reshape(-1)
    threshold_metrics = {str(t): confusion_metrics(gt, scores >= float(t)) for t in score_thresholds}
    concept_metrics = {
        "n_images": int(gt.shape[0]),
        "n_concepts": int(gt.shape[1]),
        "n_pairs": int(gt.size),
        "gt_positive_rate": float(gt.mean()) if gt.size else float("nan"),
        "score_mean": float(scores.mean()) if scores.size else float("nan"),
        "score_std": float(scores.std()) if scores.size else float("nan"),
        "auroc": safe_roc_auc(flat_gt, flat_scores),
        "ap": safe_average_precision(flat_gt, flat_scores),
        "macro_ap": macro_average_precision(gt, scores),
        "p_at_5": precision_at_k(gt, scores, k=5),
        "threshold_metrics": threshold_metrics,
    }
    concept_indices = sorted(gt_count_by_concept)
    localization = {
        str(t): localization_at_threshold(
            preds_by_image,
            gt_boxes_by_image,
            gt_rows,
            score_threshold=float(t),
            box_iou_thresholds=box_iou_thresholds,
            mask_size=int(args.mask_size),
        )
        for t in score_thresholds
    }
    print("[pinpp gdino gt] localization sweep complete", flush=True)
    detection_map = {
        str(t): detection_average_precision(preds_by_concept, gt_boxes_by_image, concept_indices, float(t))
        for t in box_iou_thresholds
    }
    print("[pinpp gdino gt] detector AP complete", flush=True)
    payload = {
        "dataset": "partimagenetpp",
        "method": "GroundingDINO",
        "annotation_dir": str(Path(args.annotation_dir)),
        "gt_boxes_jsonl": str(Path(args.gt_boxes_jsonl)),
        "concept_file": str(Path(args.concept_file)),
        "concept_count": len(concepts),
        "score_thresholds": score_thresholds,
        "box_iou_thresholds": box_iou_thresholds,
        "mask_size": int(args.mask_size),
        "gt_summary": gt_summary,
        "prediction_summary": pred_summary,
        "concept_metrics": concept_metrics,
        "localization_metrics": localization,
        "detection_map": detection_map,
        "elapsed_sec": time.time() - start,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    brief = {
        "concept": {k: concept_metrics[k] for k in ("auroc", "ap", "macro_ap", "p_at_5", "gt_positive_rate")},
        "localization": localization,
        "detection_map": {k: {"map": v["map"], "num_concepts_with_gt": v["num_concepts_with_gt"]} for k, v in detection_map.items()},
        "elapsed_sec": payload["elapsed_sec"],
    }
    print(json.dumps(brief, indent=2, sort_keys=True), flush=True)
    print(f"[pinpp gdino gt] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
