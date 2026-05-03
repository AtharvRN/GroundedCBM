import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

from PIL import Image, ImageOps


REPO_ROOT = Path(__file__).resolve().parents[1]


# Current high-contribution paper examples.
# Each image path must end in an ImageNet val filename:
# ILSVRC2012_val_XXXXXXXX.JPEG
DEFAULT_EXAMPLES: List[Tuple[str, str]] = [
    ("imagenet_val/ILSVRC2012_val_00035654.JPEG", "wide head"),
    ("imagenet_val/ILSVRC2012_val_00008178.JPEG", "yellow eyes"),
    ("imagenet_val/ILSVRC2012_val_00040242.JPEG", "dark brown body"),
    ("imagenet_val/ILSVRC2012_val_00021983.JPEG", "turtle"),
    ("imagenet_val/ILSVRC2012_val_00042198.JPEG", "used for storing garbage"),
    ("imagenet_val/ILSVRC2012_val_00017693.JPEG", "round face"),
]

# Earlier localization-first examples. Kept here as commented presets so they
# are easy to swap back in without digging through old ranking JSON files.
# DEFAULT_EXAMPLES = [
#     ("imagenet_val/ILSVRC2012_val_00006472.JPEG", "yellow eyes"),
#     ("imagenet_val/ILSVRC2012_val_00014180.JPEG", "chicken"),
#     ("imagenet_val/ILSVRC2012_val_00001138.JPEG", "wide head"),
#     ("imagenet_val/ILSVRC2012_val_00045551.JPEG", "chicken"),
#     ("imagenet_val/ILSVRC2012_val_00008049.JPEG", "yellow eyes"),
#     ("imagenet_val/ILSVRC2012_val_00000052.JPEG", "wide head"),
# ]


def parse_example(raw: str) -> Tuple[str, str]:
    if "::" not in raw:
        raise argparse.ArgumentTypeError("examples must be formatted as IMAGE_PATH::concept name")
    image_path, concept = raw.split("::", 1)
    image_path = image_path.strip()
    concept = concept.strip()
    if not image_path or not concept:
        raise argparse.ArgumentTypeError("both IMAGE_PATH and concept must be non-empty")
    return image_path, concept


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a 2x3 ImageNet spatial-comparison figure for six image/concept pairs."
    )
    parser.add_argument(
        "--example",
        action="append",
        type=parse_example,
        default=None,
        help="Image/concept pair as IMAGE_PATH::concept. Pass exactly six times. Defaults to a relative paper-example set.",
    )
    parser.add_argument("--output_dir", required=True, help="Directory for individual panels and the combined figure.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--val_tar", default="")
    parser.add_argument("--devkit_dir", default="")
    parser.add_argument("--annotation_dir", default="")
    parser.add_argument(
        "--annotation_mapping_json",
        default="",
    )
    parser.add_argument(
        "--savlg_artifact_dir",
        default="",
    )
    parser.add_argument("--salf_dir", default="")
    parser.add_argument("--vlg_load_dir", default="")
    parser.add_argument("--savlg_display_name", default="G-CBM")
    parser.add_argument("--salf_display_name", default="SALF-CBM")
    parser.add_argument("--vlg_display_name", default="VLG-CBM Grad-CAM")
    parser.add_argument("--map_normalization", default="concept_zscore_minmax")
    parser.add_argument(
        "--boxes_on_maps",
        action="store_true",
        help="Draw GDINO boxes on model-map columns too. Default keeps boxes only on original images.",
    )
    parser.add_argument("--combined_name", default="imagenet_spatial_comparison_2x3.png")
    parser.add_argument("--combined_pdf_name", default="imagenet_spatial_comparison_2x3.pdf")
    return parser.parse_args()


def image_name_from_path(image_path: str) -> str:
    name = Path(image_path).name
    if not name.startswith("ILSVRC2012_val_") or not name.endswith(".JPEG"):
        raise ValueError(f"expected an ImageNet val filename, got: {image_path}")
    return name


def write_manifest(output_dir: Path, examples: Sequence[Tuple[str, str]]) -> Path:
    manifest = {
        image_name_from_path(image_path): {"concepts": [concept]}
        for image_path, concept in examples
    }
    manifest_path = output_dir / "six_example_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def compose_grid(output_dir: Path, image_names: Sequence[str], combined_name: str, combined_pdf_name: str) -> None:
    panel_paths = [output_dir / f"{Path(name).stem}_savlg_vs_vlg.png" for name in image_names]
    missing = [str(path) for path in panel_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing rendered panel(s): {missing}")

    panels = [ImageOps.crop(Image.open(path).convert("RGB"), border=12) for path in panel_paths]
    target_w = 1500
    resized = []
    for panel in panels:
        scale = target_w / panel.width
        resized.append(panel.resize((target_w, int(panel.height * scale)), Image.Resampling.LANCZOS))

    cols, rows = 2, 3
    pad_x, pad_y = 52, 46
    row_heights = [max(resized[r * cols + c].height for c in range(cols)) for r in range(rows)]
    canvas_w = cols * target_w + (cols - 1) * pad_x
    canvas_h = sum(row_heights) + (rows - 1) * pad_y
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    for idx, panel in enumerate(resized):
        row, col = divmod(idx, cols)
        x = col * (target_w + pad_x)
        y = sum(row_heights[:row]) + row * pad_y
        canvas.paste(panel, (x, y))

    canvas.save(output_dir / combined_name, dpi=(300, 300))
    canvas.save(output_dir / combined_pdf_name, resolution=300.0)


def main() -> None:
    args = parse_args()
    examples = args.example or DEFAULT_EXAMPLES
    if len(examples) != 6:
        raise ValueError(f"expected exactly six examples, got {len(examples)}")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = write_manifest(output_dir, examples)
    image_names = [image_name_from_path(image_path) for image_path, _concept in examples]
    image_roots = sorted({str(Path(image_path).resolve().parent) for image_path, _concept in examples})
    if len(image_roots) != 1:
        raise ValueError(f"all six images must live under one directory; got {image_roots}")

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "render_imagenet_paper_spatial_comparison.py"),
        "--val_tar",
        args.val_tar,
        "--val_image_root",
        image_roots[0],
        "--devkit_dir",
        args.devkit_dir,
        "--annotation_dir",
        args.annotation_dir,
        "--annotation_mapping_json",
        args.annotation_mapping_json,
        "--savlg_artifact_dir",
        args.savlg_artifact_dir,
        "--salf_dir",
        args.salf_dir,
        "--vlg_load_dir",
        args.vlg_load_dir,
        "--concept_manifest_json",
        str(manifest_path),
        "--image_names",
        ",".join(image_names),
        "--output_dir",
        str(output_dir),
        "--device",
        args.device,
        "--paper_clean_labels",
        "--savlg_display_name",
        args.savlg_display_name,
        "--salf_display_name",
        args.salf_display_name,
        "--vlg_display_name",
        args.vlg_display_name,
        "--map_normalization",
        args.map_normalization,
    ]
    if args.boxes_on_maps:
        cmd.append("--boxes_on_maps")

    subprocess.run(cmd, check=True)
    compose_grid(output_dir, image_names, args.combined_name, args.combined_pdf_name)
    print(f"wrote {output_dir / args.combined_name}")
    print(f"wrote {output_dir / args.combined_pdf_name}")


if __name__ == "__main__":
    main()
