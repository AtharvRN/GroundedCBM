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
    parser.add_argument("--activation_thresholds", default="0.3,0.5,0.7,0.9,meanthr")
    parser.add_argument("--box_iou_thresholds", default="0.1,0.3,0.5")
    parser.add_argument(
        "--map_normalization",
        default="concept_zscore_minmax",
        choices=["minmax", "sigmoid", "concept_zscore_minmax"],
    )
    parser.add_argument(
        "--threshold_source",
        default="normalized_map",
        choices=["normalized_map", "pred_dist"],
    )
    parser.add_argument(
        "--compute_distribution_metrics",
        action="store_true",
        help="Also report threshold-free soft-IoU, mass-in-box, and point-hit from spatial softmax maps.",
    )
    return parser.parse_known_args()


def main() -> None:
    args, remaining = parse_args()
    use_meanthr = "meanthr" in str(args.activation_thresholds).lower()
    activation_thresholds = args.activation_thresholds.replace("meanthr", "mean")
    sys.path.insert(0, str(ROOT))
    sys.argv = [
        "evaluate_savlg_native_maps.py",
        "--load_path",
        args.gcbm_path,
        "--annotation_dir",
        args.annotation_dir,
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
        "--box_iou_thresholds",
        args.box_iou_thresholds,
        "--map_normalization",
        args.map_normalization,
        "--threshold_source",
        args.threshold_source,
        *(["--max_images", str(args.max_images)] if args.max_images is not None else []),
        *(["--threshold_protocol", "meanthr"] if use_meanthr else []),
        *(["--compute_distribution_metrics"] if args.compute_distribution_metrics else []),
        *remaining,
    ]
    runpy.run_path(str(ROOT / "gcbm" / "evaluate_savlg_native_maps.py"), run_name="__main__")


if __name__ == "__main__":
    main()
