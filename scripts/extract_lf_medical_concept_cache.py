from __future__ import annotations

import argparse
import inspect
import json
import sys
from argparse import Namespace
from pathlib import Path

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gcbm.task_utils import build_task_spec, load_task_base_dataset
from methods.lf import TransformedSubset
from model.cbm import Backbone, BackboneCLIP

try:
    mp.set_sharing_strategy("file_system")
except RuntimeError:
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract standardized medical LF-CBM concept caches from a completed LF run."
    )
    parser.add_argument("--run_dir", required=True, help="Completed LF-CBM run directory.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", action="store_true")
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
    payload["lf_batch_size"] = int(overrides.batch_size)
    payload["skip_test_eval"] = True
    return Namespace(**payload)


def build_backbone(args: Namespace):
    if str(args.backbone).startswith("clip_"):
        return BackboneCLIP(
            args.backbone,
            use_penultimate=bool(getattr(args, "use_clip_penultimate", False)),
            device=args.device,
        )
    return Backbone(
        args.backbone,
        args.feature_layer,
        args.device,
        checkpoint=getattr(args, "backbone_ckpt", ""),
    )


def create_lf_eval_splits(args: Namespace, backbone) -> tuple[TransformedSubset, TransformedSubset]:
    base_train = load_task_base_dataset(args, "train", transform=None, raw=True)
    max_train = int(getattr(args, "max_train_images", 0) or 0)
    total = min(len(base_train), max_train) if max_train > 0 else len(base_train)
    n_val = int(float(args.val_split) * total)
    if float(args.val_split) > 0 and n_val == 0 and total > 1:
        n_val = 1
    n_train = total - n_val
    generator = torch.Generator().manual_seed(int(args.seed))
    train_subset, val_subset = torch.utils.data.random_split(
        list(range(total)),
        [n_train, n_val],
        generator=generator,
    )
    train_dataset = TransformedSubset(base_train, train_subset.indices, backbone.preprocess)
    val_dataset = TransformedSubset(base_train, val_subset.indices, backbone.preprocess)
    return train_dataset, val_dataset


def extract_split(
    args: Namespace,
    dataset,
    backbone,
    concept_weight: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    weight = concept_weight.to(args.device)
    mean = mean.to(args.device)
    std = std.to(args.device)
    concepts = []
    targets = []
    backbone.eval()
    with torch.no_grad():
        for images, target in tqdm(loader, desc="LF concept extraction"):
            feats = backbone(images.to(args.device))
            z = feats @ weight.T
            z = (z - mean) / std
            concepts.append(z.cpu())
            targets.append(target.detach().cpu())
    return torch.cat(concepts, dim=0), torch.cat(targets, dim=0)


def save_cache(run_dir: Path, split: str, concepts: torch.Tensor, targets: torch.Tensor) -> None:
    torch.save(
        {
            "meta": {
                "split": split,
                "standardized": True,
                "num_rows": int(concepts.shape[0]),
                "num_concepts": int(concepts.shape[1]),
                "num_labels": int(targets.shape[1]) if targets.ndim > 1 else 1,
                "source": "lf_cbm",
            },
            "concepts": concepts.float(),
            "targets": targets.float(),
        },
        run_dir / f"concept_cache_{split}.pt",
    )


def main() -> None:
    cli_args = parse_args()
    run_dir = Path(cli_args.run_dir).resolve()
    run_args = load_run_args(run_dir, cli_args)
    task = build_task_spec(run_args)
    backbone = build_backbone(run_args)
    train_dataset, val_dataset = create_lf_eval_splits(run_args, backbone)

    concept_weight = torch_load(run_dir / "W_c.pt").float()
    mean = torch_load(run_dir / "proj_mean.pt").float()
    std = torch_load(run_dir / "proj_std.pt").float().clamp_min(1e-6)

    train_z, train_y = extract_split(
        run_args,
        train_dataset,
        backbone,
        concept_weight,
        mean,
        std,
        int(cli_args.batch_size),
        int(cli_args.num_workers),
        bool(cli_args.pin_memory),
    )
    save_cache(run_dir, "train", train_z, train_y)
    del train_z, train_y

    val_z, val_y = extract_split(
        run_args,
        val_dataset,
        backbone,
        concept_weight,
        mean,
        std,
        int(cli_args.batch_size),
        int(cli_args.num_workers),
        bool(cli_args.pin_memory),
    )
    save_cache(run_dir, "valid", val_z, val_y)
    (run_dir / "labels.txt").write_text("\n".join(task.label_names), encoding="utf-8")
    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "train_shape": [len(train_dataset), int(concept_weight.shape[0])],
                "valid_shape": [len(val_dataset), int(concept_weight.shape[0])],
                "num_labels": len(task.label_names),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
