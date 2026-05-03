import argparse
import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Evaluate CUB part localization for a trained G-CBM checkpoint.")
    parser.add_argument("--load_path", required=True)
    parser.add_argument("--annotation_dir", required=True)
    parser.add_argument("--cub_root", required=True)
    parser.add_argument("--mapping_json", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_known_args()


def main() -> None:
    args, remaining = parse_args()
    sys.path.insert(0, str(ROOT))
    sys.argv = [
        "evaluate_savlg_cub_parts.py",
        "--load_path",
        args.load_path,
        "--annotation_dir",
        args.annotation_dir,
        "--cub_root",
        args.cub_root,
        "--mapping_json",
        args.mapping_json,
        "--output",
        args.output,
        *remaining,
    ]
    runpy.run_path(str(ROOT / "gcbm" / "evaluate_savlg_cub_parts.py"), run_name="__main__")


if __name__ == "__main__":
    main()
