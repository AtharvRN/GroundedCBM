import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    sys.argv = ["train_cbm.py", "--dataset", "imagenet", "--model", "sgcbm", *sys.argv[1:]]
    runpy.run_path(str(ROOT / "train_cbm.py"), run_name="__main__")
