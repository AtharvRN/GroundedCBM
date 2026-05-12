import argparse
import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Run CUB70 localization evaluation for any subset of SG-CBM/SAVLG, SALF, VLG, and LF CBMs.")
    parser.add_argument("--gcbm_path", default=None, help="Path to an SG-CBM/SAVLG run directory. Alias for --savlg_path.")
    parser.add_argument("--savlg_path", default=None)
    parser.add_argument("--salf_path", default=None)
    parser.add_argument("--vlg_path", default=None)
    parser.add_argument("--lf_path", default=None)
    parser.add_argument("--cub70_root", default="datasets/CUB70-PartSegmentationDataset")
    parser.add_argument("--cub_root", default="datasets/CUB")
    parser.add_argument("--mapping_json", default="concept_files/cub_concept_part_mapping.json")
    parser.add_argument("--output", default="results/cub70_localization.json")
    return parser.parse_known_args()


def main() -> None:
    args, remaining = parse_args()
    savlg_path = args.gcbm_path or args.savlg_path
    sys.path.insert(0, str(ROOT))
    sys.argv = [
        "eval_cub70_localization.py",
        "--cub70_root",
        args.cub70_root,
        "--cub_root",
        args.cub_root,
        "--mapping_json",
        args.mapping_json,
        "--output",
        args.output,
        *(
            ["--gcbm_path", savlg_path]
            if savlg_path
            else []
        ),
        *(
            ["--salf_path", args.salf_path]
            if args.salf_path
            else []
        ),
        *(
            ["--vlg_path", args.vlg_path]
            if args.vlg_path
            else []
        ),
        *(
            ["--lf_path", args.lf_path]
            if args.lf_path
            else []
        ),
        *remaining,
    ]
    runpy.run_path(str(ROOT / "gcbm" / "eval_cub70_localization.py"), run_name="__main__")


if __name__ == "__main__":
    main()
