import argparse
import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Train sparse GLM/NEC heads for a trained CBM checkpoint."
    )
    parser.add_argument("--dataset", required=True, choices=["cub", "imagenet"])
    parser.add_argument("--load_path", default="", help="CUB CBM run directory.")
    parser.add_argument("--artifact_dir", default="", help="ImageNet SG-CBM artifact directory with extracted features.")
    return parser.parse_known_args()


def main() -> None:
    args, remaining = parse_args()
    sys.path.insert(0, str(ROOT))
    if args.dataset == "cub":
        if not args.load_path:
            raise SystemExit("--load_path is required for --dataset cub")
        sys.argv = ["sparse_evaluation.py", "--load_path", args.load_path, *remaining]
        runpy.run_path(str(ROOT / "sparse_evaluation.py"), run_name="__main__")
        return

    if not args.artifact_dir:
        raise SystemExit("--artifact_dir is required for --dataset imagenet")
    sys.argv = ["run_imagenet_glm_path.py", "--artifact_dir", args.artifact_dir, *remaining]
    runpy.run_path(str(ROOT / "gcbm" / "run_imagenet_glm_path.py"), run_name="__main__")


if __name__ == "__main__":
    main()
