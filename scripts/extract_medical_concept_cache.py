from __future__ import annotations

import argparse
import inspect
import json
import sys
import gc
from argparse import Namespace
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gcbm.task_utils import build_task_spec
from methods.savlg import (
    build_savlg_concept_layer,
    create_savlg_splits,
    extract_global_concepts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract standardized medical concept caches from a trained SG-CBM/SAVLG checkpoint."
    )
    parser.add_argument("--run_dir", required=True, help="Completed SG-CBM/SAVLG run directory.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", action="store_true", help="Pin dataloader memory during extraction.")
    return parser.parse_args()


def torch_load(path: Path):
    kwargs = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False
    return torch.load(path, **kwargs)


def load_run_args(run_dir: Path, overrides: argparse.Namespace) -> Namespace:
    args_path = run_dir / "args.txt"
    if not args_path.exists():
        raise FileNotFoundError(f"Missing args file: {args_path}")
    payload = json.loads(args_path.read_text(encoding="utf-8"))
    payload["device"] = overrides.device
    payload["num_workers"] = int(overrides.num_workers)
    payload["cbl_batch_size"] = int(overrides.batch_size)
    payload["skip_test_eval"] = True
    return Namespace(**payload)


def load_concepts(run_dir: Path) -> list[str]:
    concepts_path = run_dir / "concepts.txt"
    if not concepts_path.exists():
        raise FileNotFoundError(f"Missing concepts file: {concepts_path}")
    concepts = [
        line.strip()
        for line in concepts_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not concepts:
        raise ValueError(f"No concepts found in {concepts_path}")
    return concepts


def apply_checkpoint_compatibility(run_args: Namespace, state_dict: dict) -> None:
    """Match old SG-CBM checkpoints that were saved before DenseNet conv4 support."""
    keys = set(state_dict.keys())
    old_dual_branch = {"global_layer.weight", "spatial_layer.weight"}.issubset(keys)
    multiscale = str(getattr(run_args, "savlg_spatial_branch_mode", "")).lower() == "multiscale_conv45"
    if old_dual_branch and multiscale and "conv4_proj.weight" not in keys:
        print(
            "[extract] checkpoint is old single-stage dual-branch SG-CBM; "
            "using shared_stage layer construction for compatibility",
            flush=True,
        )
        run_args.savlg_spatial_branch_mode = "shared_stage"


def save_cache(run_dir: Path, split: str, concepts: torch.Tensor, targets: torch.Tensor) -> None:
    torch.save(
        {
            "meta": {
                "split": split,
                "standardized": True,
                "num_rows": int(concepts.shape[0]),
                "num_concepts": int(concepts.shape[1]),
                "num_labels": int(targets.shape[1]) if targets.ndim > 1 else 1,
            },
            "concepts": concepts.cpu(),
            "targets": targets.cpu(),
        },
        run_dir / f"concept_cache_{split}.pt",
    )


def build_loader(dataset, cli_args: argparse.Namespace) -> DataLoader:
    loader_kwargs = {
        "batch_size": int(cli_args.batch_size),
        "shuffle": False,
        "num_workers": int(cli_args.num_workers),
        "pin_memory": bool(cli_args.pin_memory),
    }
    if int(cli_args.num_workers) > 0:
        loader_kwargs["persistent_workers"] = False
    return DataLoader(dataset, **loader_kwargs)


def main() -> None:
    cli_args = parse_args()
    run_dir = Path(cli_args.run_dir).resolve()
    run_args = load_run_args(run_dir, cli_args)
    concepts = load_concepts(run_dir)
    task = build_task_spec(run_args)

    _, _, train_dataset, val_dataset, _, backbone = create_savlg_splits(run_args)
    state_dict = torch_load(run_dir / "concept_layer.pt")
    apply_checkpoint_compatibility(run_args, state_dict)
    concept_layer = build_savlg_concept_layer(run_args, backbone, len(concepts))
    concept_layer.load_state_dict(state_dict, strict=True)
    concept_layer = concept_layer.to(run_args.device).eval()

    train_loader = build_loader(train_dataset, cli_args)
    train_concepts, train_labels = extract_global_concepts(
        run_args, backbone, concept_layer, train_loader
    )

    mean_path = run_dir / "proj_mean.pt"
    std_path = run_dir / "proj_std.pt"
    if mean_path.exists() and std_path.exists():
        mean = torch_load(mean_path)
        std = torch_load(std_path)
    else:
        mean = train_concepts.mean(dim=0, keepdim=True)
        std = torch.clamp(train_concepts.std(dim=0, keepdim=True), min=1e-6)
        torch.save(mean, mean_path)
        torch.save(std, std_path)

    train_concepts.sub_(mean).div_(std)
    train_shape = list(train_concepts.shape)
    save_cache(run_dir, "train", train_concepts, train_labels)
    del train_loader, train_concepts, train_labels
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    val_loader = build_loader(val_dataset, cli_args)
    val_concepts, val_labels = extract_global_concepts(
        run_args, backbone, concept_layer, val_loader
    )
    val_z = (val_concepts - mean) / std
    save_cache(run_dir, "valid", val_z, val_labels)
    (run_dir / "labels.txt").write_text("\n".join(task.label_names), encoding="utf-8")
    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "train_shape": train_shape,
                "valid_shape": list(val_z.shape),
                "num_labels": len(task.label_names),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
