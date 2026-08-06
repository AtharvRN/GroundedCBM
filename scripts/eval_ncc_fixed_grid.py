import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluations.sparse_utils import build_nec_feature_set, make_feature_loader
from gcbm.ncc import ncc_counts_for_batch
from glm_saga.elasticnet import glm_saga


def parse_float_list(raw: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("expected at least one comma-separated float")
    return values


def parse_int_list(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("expected at least one comma-separated int")
    return values


def evaluate_weight(
    features: torch.Tensor,
    labels: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    tau: float,
    ncc_mode: str,
    batch_size: int,
    class_chunk_size: int,
    device: str,
) -> dict[str, Any]:
    weight = weight.to(device).float()
    bias = bias.to(device).float()
    total = top1 = top5 = ncc_count = 0
    ncc_sum = 0.0
    for start in range(0, int(features.shape[0]), int(batch_size)):
        end = min(start + int(batch_size), int(features.shape[0]))
        x = features[start:end].to(device).float()
        y = labels[start:end].to(device).long()
        logits = x @ weight.t() + bias
        pred = logits.argmax(dim=-1)
        topk = logits.topk(k=min(5, logits.shape[-1]), dim=-1).indices
        top1 += int(pred.eq(y).sum().item())
        top5 += int(topk.eq(y[:, None]).any(dim=-1).sum().item())
        counts = ncc_counts_for_batch(
            x,
            weight,
            tau=tau,
            mode=ncc_mode,
            bias=bias,
            targets=y,
            class_chunk_size=class_chunk_size,
        )
        ncc_sum += float(counts.double().sum().item())
        ncc_count += int(counts.numel())
        total += int(y.numel())
    nnz = int((weight.abs() > 1e-5).sum().item())
    return {
        "n": int(total),
        "top1": float(top1 / max(total, 1)),
        "top5": float(top5 / max(total, 1)),
        "ncc_tau": float(tau),
        "ncc_mode": ncc_mode,
        "ncc": float(ncc_sum / max(ncc_count, 1)),
        "ncc_count": int(ncc_count),
        "nnz": nnz,
        "total": int(weight.numel()),
        "weight_sparsity": 1.0 - nnz / max(int(weight.numel()), 1),
    }


def select_targets(rows: list[dict[str, Any]], targets: list[int]) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: float(row["ncc"]))
    selected = []
    for target in targets:
        row = min(ordered, key=lambda item: abs(float(item["ncc"]) - float(target)))
        out = dict(row)
        out["target_ncc"] = int(target)
        out["target_abs_error"] = abs(float(row["ncc"]) - float(target))
        selected.append(out)
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit fixed GLM-SAGA lambdas and report ACC at NCC targets.")
    parser.add_argument("--load_path", required=True)
    parser.add_argument("--model_name", default="")
    parser.add_argument("--annotation_dir", default="")
    parser.add_argument("--lambdas", default="0.001,0.0005,0.0002,0.0001,0.00005,0.00002,0.00001,0.000005,0.000002,0.000001")
    parser.add_argument("--ncc_targets", default="5,10,15,20,25,30")
    parser.add_argument("--tau", type=float, default=0.95)
    parser.add_argument("--ncc_mode", choices=["all_classes", "predicted_class", "target_class"], default="all_classes")
    parser.add_argument("--n_iters", type=int, default=500)
    parser.add_argument("--step_size", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.99)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--saga_batch_size", type=int, default=512)
    parser.add_argument("--eval_batch_size", type=int, default=512)
    parser.add_argument("--cbl_batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--class_chunk_size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--savlg_alpha_override", type=float, default=None)
    parser.add_argument("--disable_activation_cache", action="store_true")
    parser.add_argument("--savlg_branch_norm_mode", default="none")
    parser.add_argument("--output_json", default="")
    parser.add_argument("--save_dir", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    load_path = Path(args.load_path).resolve()
    model_name = args.model_name
    if not model_name:
        model_name = json.loads((load_path / "args.txt").read_text()).get("model_name", "sg_cbm")

    feature_set, _ = build_nec_feature_set(
        str(load_path),
        model_name,
        annotation_dir=args.annotation_dir or None,
        cbl_batch_size=args.cbl_batch_size,
        saga_batch_size=args.saga_batch_size,
        num_workers=args.num_workers,
        savlg_alpha_override=args.savlg_alpha_override,
        disable_activation_cache=args.disable_activation_cache,
        max_images=args.max_images,
        savlg_branch_norm_mode=args.savlg_branch_norm_mode,
    )
    train_loader = make_feature_loader(
        feature_set.train_features,
        feature_set.train_labels,
        args.saga_batch_size,
        indexed=True,
        shuffle=True,
    )
    num_concepts = len(feature_set.concepts)
    num_classes = len(feature_set.classes)
    rows = []
    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    for lam in parse_float_list(args.lambdas):
        linear = torch.nn.Linear(num_concepts, num_classes).to(args.device)
        linear.weight.data.zero_()
        linear.bias.data.zero_()
        metadata = {"max_reg": {"nongrouped": float(lam)}}
        output = glm_saga(
            linear,
            train_loader,
            args.step_size,
            args.n_iters,
            args.alpha,
            k=1,
            epsilon=1.0,
            do_zero=False,
            metadata=metadata,
            n_ex=len(train_loader.dataset),
            n_classes=num_classes,
            tol=args.tol,
            eval_train=False,
            eval_val=False,
            eval_test=False,
        )
        params = output["path"][0]
        metrics = evaluate_weight(
            feature_set.test_features,
            feature_set.test_labels,
            params["weight"],
            params["bias"],
            tau=args.tau,
            ncc_mode=args.ncc_mode,
            batch_size=args.eval_batch_size,
            class_chunk_size=args.class_chunk_size,
            device=args.device,
        )
        metrics.update({"lambda": float(lam), "fit_lambda": float(params["lam"]), "lr": float(params["lr"])})
        rows.append(metrics)
        if save_dir:
            suffix = str(lam).replace(".", "p").replace("-", "m")
            torch.save(params["weight"].cpu(), save_dir / f"W_g@lambda={suffix}.pt")
            torch.save(params["bias"].cpu(), save_dir / f"b_g@lambda={suffix}.pt")
        print(
            f"lambda={lam:.6g} fit_lambda={float(params['lam']):.6g} top1={metrics['top1']:.4f} "
            f"ncc={metrics['ncc']:.4f} nnz={metrics['nnz']}",
            flush=True,
        )

    targets = parse_int_list(args.ncc_targets)
    payload = {
        "load_path": str(load_path),
        "model_name": model_name,
        "tau": float(args.tau),
        "ncc_mode": args.ncc_mode,
        "lambdas": parse_float_list(args.lambdas),
        "ncc_targets": targets,
        "n_test": int(feature_set.test_labels.numel()),
        "rows": rows,
        "selected": select_targets(rows, targets),
        "elapsed_sec": time.perf_counter() - start,
    }
    print(json.dumps(payload, indent=2), flush=True)
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
