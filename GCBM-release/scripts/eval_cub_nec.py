import argparse
import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Run CUB sparse GLM/NEC evaluation for a trained G-CBM checkpoint.")
    parser.add_argument("--load_path", required=True, help="CUB run directory containing args.txt and saved weights.")
    return parser.parse_known_args()


def main() -> None:
    args, remaining = parse_args()
    sys.path.insert(0, str(ROOT))
    sys.argv = ["sparse_evaluation.py", "--load_path", args.load_path, *remaining]
    runpy.run_path(str(ROOT / "sparse_evaluation.py"), run_name="__main__")


if __name__ == "__main__":
    main()
