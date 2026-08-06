import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluations.sparse_utils import build_nec_feature_set
from gcbm.ncc import ncc_counts_for_batch


def parse_values(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("expected at least one comma-separated value")
    return values


def load_cub_sweep(load_path: Path, nec_values: list[int]) -> list[dict[str, Any]]:
    sweep = []
    for nec in nec_values:
        weight_path = load_path / f"W_g@NEC={nec}.pt"
        bias_path = load_path / f"b_g@NEC={nec}.pt"
        if not weight_path.exists() or not bias_path.exists():
            raise FileNotFoundError(f"missing sparse head for NEC={nec}: {weight_path} / {bias_path}")
        weight = torch.load(weight_path, map_location="cpu").float()
        bias = torch.load(bias_path, map_location="cpu").float()
        nnz = int((weight.abs() > 1e-5).sum().item())
        total = int(weight.numel())
        sweep.append(
            {
                "nec": int(nec),
                "weight": weight,
                "bias": bias,
                "nnz": nnz,
                "total": total,
                "weight_sparsity": 1.0 - nnz / max(total, 1),
            }
        )
    return sweep


def evaluate_cub(args: argparse.Namespace) -> dict[str, Any]:
    load_path = Path(args.load_path).resolve()
    feature_set, run_args = build_nec_feature_set(
        str(load_path),
        args.model_name,
        annotation_dir=args.annotation_dir or None,
        cbl_batch_size=args.cbl_batch_size,
        saga_batch_size=args.batch_size,
        num_workers=args.num_workers,
        savlg_alpha_override=args.savlg_alpha_override,
        disable_activation_cache=args.disable_activation_cache,
        max_images=args.max_samples,
        savlg_branch_norm_mode=args.savlg_branch_norm_mode,
    )
    device = torch.device(args.device)
    test_dataset = TensorDataset(feature_set.test_features, feature_set.test_labels)
    loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=str(device).startswith("cuda"),
    )
    sweep = load_cub_sweep(load_path, parse_values(args.nec_values))
    correct = torch.zeros(len(sweep), dtype=torch.long)
    top5 = torch.zeros(len(sweep), dtype=torch.long)
    ncc_sum = torch.zeros(len(sweep), dtype=torch.float64)
    ncc_count = torch.zeros(len(sweep), dtype=torch.long)
    total = 0

    for features, labels in loader:
        features = features.to(device, non_blocking=True).float()
        labels = labels.to(device, non_blocking=True).long()
        total += int(labels.numel())
        for idx, item in enumerate(sweep):
            weight = item["weight"].to(device)
            bias = item["bias"].to(device)
            logits = features @ weight.t() + bias
            pred = logits.argmax(dim=-1)
            correct[idx] += pred.eq(labels).sum().cpu()
            topk = logits.topk(k=min(5, logits.shape[-1]), dim=-1).indices
            top5[idx] += topk.eq(labels[:, None]).any(dim=-1).sum().cpu()
            counts = ncc_counts_for_batch(
                features,
                weight,
                tau=args.tau,
                mode=args.ncc_mode,
                bias=bias,
                targets=labels,
                class_chunk_size=args.class_chunk_size,
            )
            ncc_sum[idx] += counts.double().sum().cpu()
            ncc_count[idx] += int(counts.numel())

    results = []
    for idx, item in enumerate(sweep):
        ncc = float(ncc_sum[idx].item() / max(int(ncc_count[idx].item()), 1))
        results.append(
            {
                "nec": item["nec"],
                "nnz": item["nnz"],
                "total": item["total"],
                "weight_sparsity": item["weight_sparsity"],
                "top1": float(correct[idx].item() / max(total, 1)),
                "top5": float(top5[idx].item() / max(total, 1)),
                "ncc_tau": float(args.tau),
                "ncc_mode": args.ncc_mode,
                "ncc": ncc if math.isfinite(ncc) else None,
                "ncc_count": int(ncc_count[idx].item()),
            }
        )

    return {
        "dataset": "cub",
        "load_path": str(load_path),
        "model_name": args.model_name,
        "run_device": getattr(run_args, "device", ""),
        "max_samples": args.max_samples,
        "metrics": {"n": total, "results": results},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Number of Contributing Concepts for sparse CBM heads.")
    parser.add_argument("--dataset", choices=["cub"], default="cub")
    parser.add_argument("--load_path", required=True)
    parser.add_argument("--model_name", default="sg_cbm")
    parser.add_argument("--annotation_dir", default="")
    parser.add_argument("--nec_values", default="5,10,15,20,25,30")
    parser.add_argument("--tau", type=float, default=0.95)
    parser.add_argument("--ncc_mode", choices=["all_classes", "predicted_class", "target_class"], default="all_classes")
    parser.add_argument("--class_chunk_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--cbl_batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--savlg_alpha_override", type=float, default=None)
    parser.add_argument("--disable_activation_cache", action="store_true")
    parser.add_argument("--savlg_branch_norm_mode", default="none")
    parser.add_argument("--output_json", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = evaluate_cub(args)
    print(json.dumps(payload, indent=2), flush=True)
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
