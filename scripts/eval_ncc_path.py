import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluations.sparse_utils import DEFAULT_MEASURE_LEVEL, GLM_STEP_SIZE, build_nec_feature_set, make_feature_loader
from gcbm.ncc import ncc_counts_for_batch
from glm_saga.elasticnet import glm_saga


def parse_targets(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("--ncc_targets must contain at least one value")
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
    total = 0
    top1 = 0
    top5 = 0
    ncc_sum = 0.0
    ncc_count = 0
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
        "n": total,
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


def select_targets(path_metrics: list[dict[str, Any]], targets: list[int]) -> list[dict[str, Any]]:
    ordered = sorted(path_metrics, key=lambda row: float(row["ncc"]))
    selected = []
    for target in targets:
        reached = [row for row in ordered if float(row["ncc"]) >= float(target)]
        if reached:
            row = reached[0]
            target_reached = True
        else:
            row = min(ordered, key=lambda item: abs(float(item["ncc"]) - float(target)))
            target_reached = False
        payload = dict(row)
        payload["target_ncc"] = int(target)
        payload["target_reached"] = target_reached
        selected.append(payload)
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a GLM-SAGA path and report ACC at NCC targets.")
    parser.add_argument("--dataset", choices=["cub"], default="cub")
    parser.add_argument("--load_path", required=True)
    parser.add_argument("--model_name", default="")
    parser.add_argument("--annotation_dir", default="")
    parser.add_argument("--ncc_targets", default="5,10,15,20,25,30")
    parser.add_argument("--tau", type=float, default=0.95)
    parser.add_argument("--ncc_mode", choices=["all_classes", "predicted_class", "target_class"], default="all_classes")
    parser.add_argument("--lam_max", type=float, default=0.001)
    parser.add_argument("--n_iters", type=int, default=4000)
    parser.add_argument("--max_glm_steps", type=int, default=150)
    parser.add_argument("--step_size", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.99)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--max_sparsity", type=float, default=None)
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
    parser.add_argument("--save_path", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    load_path = Path(args.load_path).resolve()
    model_name = args.model_name
    if not model_name:
        run_args = json.loads((load_path / "args.txt").read_text())
        model_name = run_args.get("model_name", "sg_cbm")

    feature_set, run_args = build_nec_feature_set(
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
    val_loader = None
    if feature_set.val_features is not None and feature_set.val_labels is not None:
        val_loader = make_feature_loader(
            feature_set.val_features,
            feature_set.val_labels,
            args.saga_batch_size,
            indexed=False,
            shuffle=False,
        )

    num_concepts = len(feature_set.concepts)
    num_classes = len(feature_set.classes)
    linear = torch.nn.Linear(num_concepts, num_classes).to(args.device)
    linear.weight.data.zero_()
    linear.bias.data.zero_()
    metadata = {"max_reg": {"nongrouped": float(args.lam_max)}}
    epsilon = 1.0 / (GLM_STEP_SIZE ** int(args.max_glm_steps))
    glm_output = glm_saga(
        linear,
        train_loader,
        args.step_size,
        args.n_iters,
        args.alpha,
        k=args.max_glm_steps,
        epsilon=epsilon,
        val_loader=val_loader,
        test_loader=None,
        do_zero=False,
        metadata=metadata,
        n_ex=len(train_loader.dataset),
        n_classes=num_classes,
        tol=args.tol,
        max_sparsity=args.max_sparsity,
        eval_train=False,
        eval_val=False,
        eval_test=False,
    )
    path = glm_output["path"]
    if args.save_path:
        torch.save({"path": path, "metadata": metadata}, args.save_path)

    path_metrics = []
    for idx, params in enumerate(path):
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
        metrics.update(
            {
                "path_index": int(idx),
                "lambda": float(params["lam"]),
                "lr": float(params["lr"]),
            }
        )
        path_metrics.append(metrics)
        print(
            "path_index={idx} lambda={lam:.6g} top1={top1:.4f} ncc={ncc:.4f} nnz={nnz}".format(
                idx=idx,
                lam=float(params["lam"]),
                top1=float(metrics["top1"]),
                ncc=float(metrics["ncc"]),
                nnz=int(metrics["nnz"]),
            ),
            flush=True,
        )

    targets = parse_targets(args.ncc_targets)
    payload = {
        "dataset": args.dataset,
        "load_path": str(load_path),
        "model_name": model_name,
        "ncc_targets": targets,
        "tau": float(args.tau),
        "ncc_mode": args.ncc_mode,
        "glm": {
            "lam_max": float(args.lam_max),
            "n_iters": int(args.n_iters),
            "max_glm_steps": int(args.max_glm_steps),
            "step_size": float(args.step_size),
            "alpha": float(args.alpha),
            "tol": float(args.tol),
            "max_sparsity": args.max_sparsity,
        },
        "n_test": int(feature_set.test_labels.numel()),
        "path_metrics": path_metrics,
        "selected": select_targets(path_metrics, targets),
        "elapsed_sec": time.perf_counter() - start,
    }
    print(json.dumps(payload, indent=2), flush=True)
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
