import argparse
import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Evaluate concept presence accuracy on the common concept set across multiple CBM checkpoints.")
    parser.add_argument("--load_paths", nargs="+", required=True)
    return parser.parse_known_args()


def main() -> None:
    args, remaining = parse_args()
    sys.path.insert(0, str(ROOT))
    sys.argv = ["eval_concept_accuracy.py", "--load_paths", *args.load_paths, *remaining]
    runpy.run_path(str(ROOT / "gcbm" / "eval_concept_accuracy.py"), run_name="__main__")


if __name__ == "__main__":
    main()
