from __future__ import annotations

import argparse
import inspect
import json
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from glm_saga.elasticnet import IndexedTensorDataset, glm_saga
from gcbm.medical_metrics import compute_medical_metrics
from gcbm.sparse import threshold_weight_truncation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sweep naturally sparse GLM-SAGA NEC heads for cached medical CBM concepts.")
    parser.add_argument("--run_dir", required=True, help="Completed medical CBM run dir containing concept_cache_{train,valid}.pt.")
    parser.add_argument("--output_dir", default="", help="Sweep output dir. Defaults under run_dir.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--saga_max_lr", type=float, default=0.1)
    parser.add_argument("--saga_iters", type=int, default=60, help="SAGA epochs per lambda path point.")
    parser.add_argument(
        "--lam_max",
        type=float,
        default=0.0,
        help="Strongest regularization in the path. If <= 0, compute the data-dependent maximum regularization.",
    )
    parser.add_argument("--epsilon", type=float, default=0.02, help="Weakest lambda as epsilon * lam_max.")
    parser.add_argument("--path_steps", type=int, default=24)
    parser.add_argument("--max_density", type=float, default=0.25, help="Stop GLM path once raw density exceeds this fraction; <=0 disables.")
    parser.add_argument("--alpha", type=float, default=0.99)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--table_device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--verbose_every", type=int, default=1)
    parser.add_argument("--eval_every", type=int, default=0)
    parser.add_argument("--nec_values", default="1,2,3,4,5,7,10,15,20,25,30,40,50")
    parser.add_argument("--selection_metric", choices=["mean_auroc", "mAP"], default="mean_auroc")
    parser.add_argument(
        "--natural_nnz_tolerance",
        type=float,
        default=0.15,
        help="Relative nnz error allowed when selecting a naturally sparse GLM point for NEC; otherwise use truncation fallback.",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for precision/recall/F1 metrics.")
    parser.add_argument("--skip_path_save", action="store_true", help="Do not save full glm_path.pt tensors.")
    return parser


def torch_load(path: Path) -> Any:
    kwargs: dict[str, Any] = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False
    return torch.load(path, **kwargs)


def parse_csv_ints(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one NEC value")
    return values


def read_lines(path: Path, fallback_count: int, prefix: str) -> list[str]:
    if path.exists():
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [f"{prefix}{idx}" for idx in range(fallback_count)]


def load_split_cache(run_dir: Path, split: str) -> tuple[torch.Tensor, torch.Tensor]:
    payload = torch_load(run_dir / f"concept_cache_{split}.pt")
    return payload["concepts"].float(), payload["targets"].float()


@torch.no_grad()
def evaluate_logits(targets: torch.Tensor, logits: torch.Tensor, labels: list[str], threshold: float) -> dict[str, Any]:
    probs = torch.sigmoid(logits).cpu().numpy()
    return compute_medical_metrics(targets.cpu().numpy(), probs, labels, threshold=threshold)


def compact_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        "mean_auroc": float(metrics.get("mean_auroc", float("nan"))),
        "mAP": float(metrics.get("mAP", float("nan"))),
        "macro_f1": float(metrics.get("f1", {}).get("macro", float("nan"))),
        "macro_precision": float(metrics.get("precision", {}).get("macro", float("nan"))),
        "macro_recall": float(metrics.get("recall", {}).get("macro", float("nan"))),
    }


def _score(metrics: dict[str, float], metric_name: str) -> float:
    return float(metrics.get(metric_name, float("-inf")))


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = Path(args.run_dir).resolve()
    lam_label = "auto" if float(args.lam_max) <= 0.0 else str(args.lam_max).replace(".", "p")
    output_dir = Path(args.output_dir).resolve() if args.output_dir else run_dir / (
        f"medical_glm_nec_sweep_lam{lam_label}"
        f"_eps{str(args.epsilon).replace('.', 'p')}_k{args.path_steps}_it{args.saga_iters}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    train_x, train_y = load_split_cache(run_dir, "train")
    val_x, val_y = load_split_cache(run_dir, "valid")
    labels = read_lines(run_dir / "labels.txt", int(train_y.shape[1]), "label_")
    concepts = read_lines(run_dir / "concepts.txt", int(train_x.shape[1]), "concept_")
    nec_values = parse_csv_ints(args.nec_values)

    train_loader = DataLoader(
        IndexedTensorDataset(train_x, train_y),
        batch_size=int(args.batch_size),
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_x, val_y),
        batch_size=int(args.batch_size),
        shuffle=False,
    )

    linear = nn.Linear(train_x.shape[1], train_y.shape[1]).to(args.device)
    linear.weight.data.zero_()
    linear.bias.data.zero_()

    start_time = time.perf_counter()
    metadata = None
    if float(args.lam_max) > 0.0:
        metadata = {"max_reg": {"nongrouped": float(args.lam_max)}}

    output = glm_saga(
        linear,
        train_loader,
        max_lr=float(args.saga_max_lr),
        nepochs=int(args.saga_iters),
        alpha=float(args.alpha),
        table_device=args.table_device,
        tol=float(args.tol),
        epsilon=float(args.epsilon),
        k=int(args.path_steps),
        val_loader=val_loader,
        do_zero=False,
        metadata=metadata,
        n_ex=len(train_x),
        n_classes=int(train_y.shape[1]),
        verbose=int(args.verbose_every),
        eval_every=(int(args.eval_every) if int(args.eval_every) > 0 else None),
        eval_train=False,
        eval_val=True,
        eval_test=False,
        family="multilabel",
        max_sparsity=(float(args.max_density) if float(args.max_density) > 0.0 else None),
    )
    elapsed = time.perf_counter() - start_time

    path_rows: list[dict[str, Any]] = []
    natural_by_nec: dict[int, dict[str, Any]] = {}
    truncated_by_nec: dict[int, dict[str, Any]] = {}
    val_x_device = val_x.to(args.device)
    val_y_cpu = val_y.cpu()

    for path_idx, params in enumerate(output["path"]):
        weight = params["weight"].float()
        bias = params["bias"].float()
        nnz = int((weight.abs() > 1e-5).sum().item())
        total = int(weight.numel())
        logits = val_x_device @ weight.to(args.device).T + bias.to(args.device)
        dense_metrics = compact_metrics(evaluate_logits(val_y_cpu, logits.cpu(), labels, float(args.threshold)))
        raw_item = {
            "path_index": int(path_idx),
            "lambda": float(params["lam"]),
            "lr": float(params["lr"]),
            "nnz": nnz,
            "total": total,
            "density": nnz / max(total, 1),
            "glm_metrics": {key: float(value) for key, value in params["metrics"].items()},
            "dense_metrics": dense_metrics,
        }
        path_row = {
            **raw_item,
            "nec": [],
        }
        for nec in nec_values:
            target_nnz = int(nec) * int(train_y.shape[1])
            target_sparsity = min(float(nec) / max(len(concepts), 1), 1.0)
            nnz_error = abs(nnz - target_nnz)
            relative_nnz_error = nnz_error / max(target_nnz, 1)
            natural_item = {
                "NEC": int(nec),
                "target_nnz": target_nnz,
                "target_sparsity": target_sparsity,
                "nnz": nnz,
                "relative_nnz_error": relative_nnz_error,
                "metrics": dense_metrics,
                "path_index": int(path_idx),
                "lambda": float(params["lam"]),
                "lr": float(params["lr"]),
                "weight": weight.cpu(),
                "bias": bias.cpu(),
            }
            current_natural = natural_by_nec.get(int(nec))
            if current_natural is None:
                natural_by_nec[int(nec)] = natural_item
            else:
                current_error = float(current_natural["relative_nnz_error"])
                current_score = _score(current_natural["metrics"], args.selection_metric)
                candidate_score = _score(natural_item["metrics"], args.selection_metric)
                if relative_nnz_error < current_error or (
                    relative_nnz_error == current_error and candidate_score > current_score
                ):
                    natural_by_nec[int(nec)] = natural_item

            sparse_weight = threshold_weight_truncation(weight, target_sparsity)
            sparse_nnz = int((sparse_weight.abs() > 1e-5).sum().item())
            sparse_logits = val_x_device @ sparse_weight.to(args.device).T + bias.to(args.device)
            metrics = compact_metrics(evaluate_logits(val_y_cpu, sparse_logits.cpu(), labels, float(args.threshold)))
            item = {
                "NEC": int(nec),
                "target_nnz": target_nnz,
                "target_sparsity": target_sparsity,
                "nnz": sparse_nnz,
                "relative_nnz_error": abs(sparse_nnz - target_nnz) / max(target_nnz, 1),
                "metrics": metrics,
                "path_index": int(path_idx),
                "lambda": float(params["lam"]),
                "lr": float(params["lr"]),
                "dense_nnz": nnz,
                "dense_density": nnz / max(total, 1),
                "weight": sparse_weight.cpu(),
                "bias": bias.cpu(),
            }
            path_row["nec"].append({key: value for key, value in item.items() if key not in {"weight", "bias"}})
            current = truncated_by_nec.get(int(nec))
            score = float(metrics[args.selection_metric])
            current_score = float("-inf") if current is None else float(current["metrics"][args.selection_metric])
            if score > current_score:
                truncated_by_nec[int(nec)] = item
        path_rows.append(path_row)

    selected_rows: list[dict[str, Any]] = []
    natural_rows: list[dict[str, Any]] = []
    truncated_rows: list[dict[str, Any]] = []
    for nec in nec_values:
        natural = natural_by_nec[int(nec)]
        truncated = truncated_by_nec[int(nec)]
        use_natural = float(natural["relative_nnz_error"]) <= float(args.natural_nnz_tolerance)
        selected = natural if use_natural else truncated
        selected_mode = "natural_glm" if use_natural else "truncated_fallback"

        torch.save(selected["weight"], output_dir / f"W_g@NEC={int(nec)}.pt")
        torch.save(selected["bias"], output_dir / f"b_g@NEC={int(nec)}.pt")
        torch.save(natural["weight"], output_dir / f"W_g_natural@NEC={int(nec)}.pt")
        torch.save(natural["bias"], output_dir / f"b_g_natural@NEC={int(nec)}.pt")
        torch.save(truncated["weight"], output_dir / f"W_g_truncated@NEC={int(nec)}.pt")
        torch.save(truncated["bias"], output_dir / f"b_g_truncated@NEC={int(nec)}.pt")

        natural_row = {key: value for key, value in natural.items() if key not in {"weight", "bias"}}
        truncated_row = {key: value for key, value in truncated.items() if key not in {"weight", "bias"}}
        selected_row = {
            **{key: value for key, value in selected.items() if key not in {"weight", "bias"}},
            "selection_mode": selected_mode,
            "natural_relative_nnz_error": float(natural["relative_nnz_error"]),
            "truncated_relative_nnz_error": float(truncated["relative_nnz_error"]),
        }
        natural_rows.append(natural_row)
        truncated_rows.append(truncated_row)
        selected_rows.append(selected_row)

    if not args.skip_path_save:
        torch.save({"path": output["path"], "best": output["best"]}, output_dir / "glm_path.pt")
    payload = {
        "dataset": "medical",
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "num_examples": int(train_x.shape[0]),
        "num_concepts": int(train_x.shape[1]),
        "num_labels": int(train_y.shape[1]),
        "selection_metric": str(args.selection_metric),
        "elapsed_sec": float(elapsed),
        "config": vars(args),
        "path": path_rows,
        "best_by_nec": selected_rows,
        "natural_by_nec": natural_rows,
        "truncated_by_nec": truncated_rows,
    }
    (output_dir / "glm_nec_sweep_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output_dir / "labels.txt").write_text("\n".join(labels), encoding="utf-8")
    (output_dir / "concepts.txt").write_text("\n".join(concepts), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "elapsed_sec": elapsed, "best_by_nec": selected_rows}, indent=2))
    return payload


def main(argv: list[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
