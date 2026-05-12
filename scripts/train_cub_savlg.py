import argparse
import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Train the CUB SAVLG/SG-CBM checkpoint using the preserved legacy trainer.")
    parser.add_argument(
        "--config",
        default="configs/cub_gcbm.json",
        help="JSON config forwarded to train_cbm.py.",
    )
    return parser.parse_known_args()


def main() -> None:
    args, remaining = parse_args()
    sys.path.insert(0, str(ROOT))
    sys.argv = ["train_cbm.py", "--config", args.config, *remaining]
    runpy.run_path(str(ROOT / "train_cbm.py"), run_name="__main__")


if __name__ == "__main__":
    main()
