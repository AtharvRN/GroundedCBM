#!/usr/bin/env python3
"""Evaluate G-CBM CUB localization against GDINO pseudo boxes."""

import argparse
import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate G-CBM/SAVLG native spatial maps on CUB using GDINO "
            "annotation boxes as pseudo ground truth."
        )
    )
    parser.add_argument("--gcbm_path", required=True, help="Path to a trained G-CBM/SAVLG run directory.")
    parser.add_argument("--annotation_dir", required=True, help="Directory containing cub_train/cub_val GDINO JSONs.")
    parser.add_argument("--output", required=True, help="Output JSON path.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_images", type=int, default=None, help="Optional smoke-test cap.")
    parser.add_argument(
        "--activation_thresholds",
        default="0.3,0.5,0.7,0.9",
        help="Comma-separated activation thresholds. Use --threshold_mode mean for per-map mean thresholding.",
    )
    parser.add_argument("--box_iou_thresholds", default="0.1,0.3,0.5")
    parser.add_argument(
        "--threshold_mode",
        default="fixed",
        choices=["fixed", "percentile", "mean"],
        help="Interpret thresholds as fixed normalized values, percentiles, or per-map mean thresholding.",
    )
    parser.add_argument(
        "--map_normalization",
        default="concept_zscore_minmax",
        choices=["minmax", "sigmoid", "proj_zscore_minmax", "concept_zscore_minmax"],
        help="Map normalization before thresholding. concept_zscore_minmax uses saved proj_mean/proj_std then min-max.",
    )
    parser.add_argument("--name", default="gcbm", help="Model name used in the output JSON.")
    return parser.parse_known_args()


def main() -> None:
    args, remaining = parse_args()
    activation_thresholds = str(args.activation_thresholds)
    threshold_mode = str(args.threshold_mode)
    if activation_thresholds.strip().lower() in {"mean", "meanthr"}:
        threshold_mode = "mean"
        activation_thresholds = "0.0"
    sys.path.insert(0, str(ROOT))
    sys.argv = [
        "evaluate_native_spatial_maps.py",
        "--load_paths",
        args.gcbm_path,
        "--names",
        args.name,
        "--annotation_dir",
        args.annotation_dir,
        "--dataset",
        "cub",
        "--split",
        "val",
        "--concept_mode",
        "intersection",
        "--map_source",
        "native",
        "--gt_source",
        "gdino_boxes",
        "--eval_subset_mode",
        "gt_present",
        "--gt_cache_max_entries",
        "0",
        "--output",
        args.output,
        "--device",
        args.device,
        "--batch_size",
        str(args.batch_size),
        "--num_workers",
        str(args.num_workers),
        "--activation_thresholds",
        activation_thresholds,
        "--threshold_mode",
        threshold_mode,
        "--box_iou_thresholds",
        args.box_iou_thresholds,
        "--map_normalization",
        args.map_normalization,
        *(["--max_images", str(args.max_images)] if args.max_images is not None else []),
        *remaining,
    ]
    runpy.run_path(str(ROOT / "scripts" / "evaluate_native_spatial_maps.py"), run_name="__main__")


if __name__ == "__main__":
    main()
