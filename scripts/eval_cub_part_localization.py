import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gcbm.evaluate_savlg_cub_parts import main


if __name__ == "__main__":
    main()
