import argparse
import json
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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
        run_cub_nec(args.load_path, remaining)
        return

    if not args.artifact_dir:
        raise SystemExit("--artifact_dir is required for --dataset imagenet")
    sys.argv = ["imagenet_glm_path.py", "--artifact_dir", args.artifact_dir, *remaining]
    runpy.run_path(str(ROOT / "gcbm" / "imagenet_glm_path.py"), run_name="__main__")


def cub_nec_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train sparse GLM/NEC heads for a trained CUB CBM checkpoint.")
    parser.add_argument("--lam", type=float, default=0.1)
    parser.add_argument(
        "--fixed_lam",
        "--fixed-lam",
        action="store_true",
        help="Train/evaluate one sparse head at --lam instead of running the full NEC lambda path.",
    )
    parser.add_argument("--filter", type=float, default=0)
    parser.add_argument("--annotation_dir", type=str, default=None)
    parser.add_argument("--result_file", type=str, default=None)
    parser.add_argument("--lf-cbm", action="store_true")
    parser.add_argument("--n_iters", type=int, default=None)
    parser.add_argument("--max_glm_steps", type=int, default=150)
    parser.add_argument("--cbl_batch_size", type=int, default=None)
    parser.add_argument("--saga_batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--savlg_alpha_override", type=float, default=None)
    parser.add_argument("--disable_activation_cache", action="store_true")
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--savlg_branch_norm_mode", type=str, default="none")
    return parser


def run_cub_nec(load_path: str, remaining: list[str]) -> None:
    import pandas as pd

    from evaluations.sparse_utils import (
        DEFAULT_MEASURE_LEVEL,
        train_fixed_lam_from_checkpoint,
        train_sparse_nec_from_checkpoint,
    )
    from methods.common import load_run_info

    cub_args = cub_nec_parser().parse_args(remaining)
    run_info = load_run_info(load_path)
    model_name = "lf_cbm" if cub_args.lf_cbm else run_info.get("model_name", "vlg_cbm")
    if cub_args.fixed_lam:
        acc, _, _ = train_fixed_lam_from_checkpoint(
            load_path,
            model_name,
            lam=cub_args.lam,
            bot_filter=cub_args.filter,
            annotation_dir=cub_args.annotation_dir,
            n_iters=cub_args.n_iters,
            cbl_batch_size=cub_args.cbl_batch_size,
            saga_batch_size=cub_args.saga_batch_size,
            num_workers=cub_args.num_workers,
            savlg_alpha_override=cub_args.savlg_alpha_override,
            disable_activation_cache=cub_args.disable_activation_cache,
            max_images=cub_args.max_images,
            savlg_branch_norm_mode=cub_args.savlg_branch_norm_mode,
        )
        with open(os.path.join(load_path, "nec_metrics.json"), "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "model_name": model_name,
                    "load_path": load_path,
                    "mode": "fixed_lam",
                    "fixed_lam": float(cub_args.lam),
                    "metrics": {
                        "fixed_lam_acc": float(acc),
                        "nec_avg": float(acc),
                    },
                },
                handle,
                indent=2,
            )
        if cub_args.result_file:
            if os.path.exists(cub_args.result_file):
                df = pd.read_csv(cub_args.result_file)
            else:
                df = pd.DataFrame(columns=["fixed_lam", "fixed_lam_acc"])
            df.loc[len(df.index)] = pd.Series(
                {"fixed_lam": float(cub_args.lam), "fixed_lam_acc": float(acc)}
            )
            df.to_csv(cub_args.result_file, index=False)
        return

    accs, _, _ = train_sparse_nec_from_checkpoint(
        load_path,
        model_name,
        lam_max=cub_args.lam,
        bot_filter=cub_args.filter,
        annotation_dir=cub_args.annotation_dir,
        n_iters=cub_args.n_iters,
        max_glm_steps=cub_args.max_glm_steps,
        cbl_batch_size=cub_args.cbl_batch_size,
        saga_batch_size=cub_args.saga_batch_size,
        num_workers=cub_args.num_workers,
        savlg_alpha_override=cub_args.savlg_alpha_override,
        disable_activation_cache=cub_args.disable_activation_cache,
        max_images=cub_args.max_images,
        savlg_branch_norm_mode=cub_args.savlg_branch_norm_mode,
    )
    nec_rows = []
    for level, acc in zip(DEFAULT_MEASURE_LEVEL, accs):
        nec = int(level)
        if os.path.exists(os.path.join(load_path, f"W_g@NEC={nec}.pt")):
            nec_rows.append({"NEC": nec, "Accuracy": float(acc)})
    with open(os.path.join(load_path, "nec_metrics.json"), "w", encoding="utf-8") as handle:
        json.dump({"model_name": model_name, "load_path": load_path, "metrics": nec_rows}, handle, indent=2)
    if cub_args.result_file:
        if os.path.exists(cub_args.result_file):
            df = pd.read_csv(cub_args.result_file)
        else:
            df = pd.DataFrame(columns=["ACC@5", "AVGACC"])
        df.loc[len(df.index)] = pd.Series({"ACC@5": accs[0], "AVGACC": sum(accs) / len(accs)})
        df.to_csv(cub_args.result_file, index=False)


if __name__ == "__main__":
    main()
