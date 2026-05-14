import os
from argparse import ArgumentParser, Namespace
import json

import pandas as pd

from evaluations.sparse_utils import (
    DEFAULT_MEASURE_LEVEL,
    train_sparse_nec_from_checkpoint,
)
from methods.common import load_run_info


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Run sparse GLM/NEC evaluation for a trained CBM checkpoint.")
    parser.add_argument("--load_path", type=str, required=True)
    parser.add_argument("--lam", type=float, default=0.1)
    parser.add_argument("--filter", type=float, default=0)
    parser.add_argument("--annotation_dir", type=str, default=None)
    parser.add_argument("--result_file", type=str, default=None)
    parser.add_argument("--lf-cbm", action="store_true")
    parser.add_argument("--n_iters", type=int, default=None)
    parser.add_argument("--max_glm_steps", type=int, default=None)
    parser.add_argument("--cbl_batch_size", type=int, default=None)
    parser.add_argument("--saga_batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--savlg_alpha_override", type=float, default=None)
    parser.add_argument("--disable_activation_cache", action="store_true")
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--savlg_branch_norm_mode", type=str, default="none")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_info = load_run_info(args.load_path)
    model_name = "lf_cbm" if args.lf_cbm else run_info.get("model_name", "vlg_cbm")
    accs, _, _ = train_sparse_nec_from_checkpoint(
        args.load_path,
        model_name,
        lam_max=args.lam,
        bot_filter=args.filter,
        annotation_dir=args.annotation_dir,
        n_iters=args.n_iters,
        max_glm_steps=args.max_glm_steps if args.max_glm_steps is not None else 150,
        cbl_batch_size=args.cbl_batch_size,
        saga_batch_size=args.saga_batch_size,
        num_workers=args.num_workers,
        savlg_alpha_override=args.savlg_alpha_override,
        disable_activation_cache=args.disable_activation_cache,
        max_images=args.max_images,
        savlg_branch_norm_mode=args.savlg_branch_norm_mode,
    )
    nec_rows = []
    for level, acc in zip(DEFAULT_MEASURE_LEVEL, accs):
        nec = int(level)
        if os.path.exists(os.path.join(args.load_path, f"W_g@NEC={nec}.pt")):
            nec_rows.append({"NEC": nec, "Accuracy": float(acc)})
    with open(os.path.join(args.load_path, "nec_metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "model_name": model_name,
                "load_path": args.load_path,
                "metrics": nec_rows,
            },
            handle,
            indent=2,
        )
    if args.result_file:
        if os.path.exists(args.result_file):
            df = pd.read_csv(args.result_file)
        else:
            df = pd.DataFrame(columns=["ACC@5", "AVGACC"])
        row = pd.Series({"ACC@5": accs[0], "AVGACC": sum(accs) / len(accs)})
        df.loc[len(df.index)] = row
        df.to_csv(args.result_file, index=False)


if __name__ == "__main__":
    main()
