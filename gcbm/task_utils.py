from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from data import utils as data_utils
from gcbm.medical_data import (
    default_mimic_paths,
    get_medical_transforms,
    infer_chexpert_img_root,
    load_chexpert_dataset,
    load_mimic_cxr_dataset,
    medical_labels,
)
from gcbm.medical_metrics import compute_medical_metrics
from glm_saga.elasticnet import IndexedTensorDataset, glm_saga
from model.cbm import train_dense_final as train_single_label_dense_final
from model.cbm import train_sparse_final as train_single_label_sparse_final


MEDICAL_DATASETS = {"chexpert", "mimic"}


@dataclass(frozen=True)
class TaskSpec:
    dataset: str
    label_names: list[str]
    multilabel: bool

    @property
    def output_dim(self) -> int:
        return len(self.label_names)


def is_medical_dataset(dataset: str) -> bool:
    return str(dataset).lower() in MEDICAL_DATASETS


def is_multilabel_dataset(dataset: str) -> bool:
    return is_medical_dataset(dataset)


def build_task_spec(args) -> TaskSpec:
    dataset = str(args.dataset).lower()
    if is_medical_dataset(dataset):
        label_subset = getattr(args, "label_subset", "all")
        labels = medical_labels(
            dataset,
            competition=label_subset == "competition",
            pathology=label_subset == "pathology",
        )
        return TaskSpec(dataset=dataset, label_names=labels, multilabel=True)
    return TaskSpec(
        dataset=dataset,
        label_names=data_utils.get_classes(dataset),
        multilabel=False,
    )


def unpack_sample(sample: Any) -> tuple[Any, torch.Tensor]:
    if isinstance(sample, dict):
        return sample["image"], torch.as_tensor(sample["target"])
    image, target = sample
    return image, torch.as_tensor(target)


def dataset_targets_view(base_dataset: Dataset) -> torch.Tensor | None:
    targets = getattr(base_dataset, "targets", None)
    if targets is None:
        return None
    if isinstance(targets, torch.Tensor):
        return targets.detach().cpu()
    return torch.as_tensor(targets)


def subset_targets(base_dataset: Dataset, indices: Iterable[int]) -> torch.Tensor:
    idx_list = list(indices)
    targets = dataset_targets_view(base_dataset)
    if targets is not None:
        return targets[idx_list].clone()
    return torch.stack([unpack_sample(base_dataset[idx])[1] for idx in idx_list], dim=0)


class TupleTransformSubset(Dataset):
    def __init__(self, base_dataset: Dataset, indices: Iterable[int], transform):
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.transform = transform
        self.targets = subset_targets(base_dataset, self.indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        image, target = unpack_sample(self.base_dataset[self.indices[idx]])
        if self.transform is not None:
            image = self.transform(image)
        return image, target


class DualTransformSubset(Dataset):
    def __init__(self, base_dataset: Dataset, indices: Iterable[int], transform_a, transform_b):
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.transform_a = transform_a
        self.transform_b = transform_b
        self.targets = subset_targets(base_dataset, self.indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        image, target = unpack_sample(self.base_dataset[self.indices[idx]])
        image_a = self.transform_a(image) if self.transform_a is not None else image
        image_b = self.transform_b(image) if self.transform_b is not None else image
        return image_a, image_b, target


class RawTupleSubset(Dataset):
    def __init__(self, base_dataset: Dataset, indices: Iterable[int]):
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.targets = subset_targets(base_dataset, self.indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        return unpack_sample(self.base_dataset[self.indices[idx]])


def load_task_base_dataset(args, split: str, *, transform=None, raw: bool = False) -> Dataset:
    dataset = str(args.dataset).lower()
    if not is_medical_dataset(dataset):
        suffix = "train" if split == "train" else "val"
        return data_utils.get_data(f"{dataset}_{suffix}", preprocess=transform)

    label_subset = getattr(args, "label_subset", "all")
    labels = medical_labels(
        dataset,
        competition=label_subset == "competition",
        pathology=label_subset == "pathology",
    )
    if transform is None and not raw:
        transform = get_medical_transforms(int(getattr(args, "img_size", 224)), train=split == "train")
    data_dir = Path(args.data_dir)
    if dataset == "chexpert":
        img_root = Path(args.img_root) if getattr(args, "img_root", "") else infer_chexpert_img_root(data_dir)
        csv_path = Path(args.train_csv) if split == "train" and getattr(args, "train_csv", "") else None
        if csv_path is None:
            csv_path = Path(args.val_csv) if split != "train" and getattr(args, "val_csv", "") else None
        if csv_path is None:
            csv_path = data_dir / ("train.csv" if split == "train" else "valid.csv")
        return load_chexpert_dataset(
            csv_path,
            img_root=img_root,
            labels=labels,
            transform=transform,
            uncertain_strategy=getattr(args, "uncertain_strategy", "ones"),
            frontal_only=bool(getattr(args, "frontal_only", True)),
        )

    paths = default_mimic_paths(data_dir)
    label_csv = Path(args.mimic_label_csv) if getattr(args, "mimic_label_csv", "") else paths["label_csv"]
    split_csv = Path(args.mimic_split_csv) if getattr(args, "mimic_split_csv", "") else paths["split_csv"]
    metadata_csv = Path(args.mimic_metadata_csv) if getattr(args, "mimic_metadata_csv", "") else paths["metadata_csv"]
    return load_mimic_cxr_dataset(
        label_csv,
        img_root=Path(args.img_root) if getattr(args, "img_root", "") else data_dir,
        split="train" if split == "train" else "validate",
        split_csv=split_csv,
        metadata_csv=metadata_csv,
        labels=labels,
        transform=transform,
        uncertain_strategy=getattr(args, "uncertain_strategy", "ones"),
        frontal_only=bool(getattr(args, "frontal_only", True)),
    )


def _train_multilabel_dense_final(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    args,
    output_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    linear = nn.Linear(train_x.shape[1], output_dim).to(args.device)
    optimizer = torch.optim.AdamW(linear.parameters(), lr=getattr(args, "final_lr", getattr(args, "dense_lr", 1e-3)))
    criterion = nn.BCEWithLogitsLoss()
    best = None
    best_loss = float("inf")
    train_ds = TensorDataset(train_x.float(), train_y.float())
    train_loader = DataLoader(
        train_ds,
        batch_size=int(getattr(args, "final_batch_size", getattr(args, "saga_batch_size", 256))),
        shuffle=True,
    )
    final_epochs = int(getattr(args, "final_epochs", getattr(args, "saga_n_iters", 100)))
    for _ in range(final_epochs):
        linear.train()
        for features, labels in train_loader:
            features = features.to(args.device).float()
            labels = labels.to(args.device).float()
            optimizer.zero_grad()
            loss = criterion(linear(features), labels)
            loss.backward()
            optimizer.step()
        linear.eval()
        with torch.no_grad():
            val_loss = float(
                criterion(
                    linear(val_x.to(args.device).float()),
                    val_y.to(args.device).float(),
                ).cpu()
            )
        if val_loss < best_loss:
            best_loss = val_loss
            best = {key: value.detach().cpu().clone() for key, value in linear.state_dict().items()}
    if best is not None:
        linear.load_state_dict(best)
    return linear.weight.detach().cpu(), linear.bias.detach().cpu()


def _train_multilabel_sparse_final(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    args,
    output_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    train_loader = DataLoader(
        IndexedTensorDataset(train_x.float(), train_y.float()),
        batch_size=int(getattr(args, "final_batch_size", getattr(args, "saga_batch_size", 256))),
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_x.float(), val_y.float()),
        batch_size=int(getattr(args, "final_batch_size", getattr(args, "saga_batch_size", 256))),
        shuffle=False,
    )
    linear = nn.Linear(train_x.shape[1], output_dim).to(args.device)
    linear.weight.data.zero_()
    linear.bias.data.zero_()
    output = glm_saga(
        linear,
        train_loader,
        max_lr=float(getattr(args, "saga_max_lr", 0.1)),
        nepochs=int(getattr(args, "saga_iters", getattr(args, "saga_n_iters", 1000))),
        alpha=0.99,
        epsilon=1.0,
        k=1,
        val_loader=val_loader,
        do_zero=False,
        metadata={"max_reg": {"nongrouped": float(getattr(args, "saga_lam", 1e-3))}},
        n_ex=len(train_x),
        n_classes=output_dim,
        family="multilabel",
    )
    return output["path"][0]["weight"].cpu(), output["path"][0]["bias"].cpu()


def train_task_final_layer(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    args,
    task: TaskSpec,
) -> tuple[torch.Tensor, torch.Tensor]:
    if task.multilabel:
        if bool(getattr(args, "use_saga", False)) or not bool(getattr(args, "dense", False)):
            return _train_multilabel_sparse_final(train_x, train_y, val_x, val_y, args, task.output_dim)
        return _train_multilabel_dense_final(train_x, train_y, val_x, val_y, args, task.output_dim)

    train_loader = DataLoader(
        IndexedTensorDataset(train_x.float(), train_y.long()),
        batch_size=int(getattr(args, "saga_batch_size", 512)),
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_x.float(), val_y.long()),
        batch_size=int(getattr(args, "saga_batch_size", 512)),
        shuffle=False,
    )
    final_layer = nn.Linear(train_x.shape[1], task.output_dim).to(args.device)
    final_layer.weight.data.zero_()
    final_layer.bias.data.zero_()
    if bool(getattr(args, "dense", False)):
        output_proj = train_single_label_dense_final(
            final_layer,
            train_loader,
            val_loader,
            int(getattr(args, "saga_n_iters", 2000)),
            float(getattr(args, "dense_lr", 1e-3)),
            device=args.device,
        )
    else:
        output_proj = train_single_label_sparse_final(
            final_layer,
            train_loader,
            val_loader,
            int(getattr(args, "saga_n_iters", 2000)),
            float(getattr(args, "saga_lam", 7e-4)),
            step_size=float(getattr(args, "saga_step_size", 0.1)),
            device=args.device,
        )
    return output_proj["path"][0]["weight"].cpu(), output_proj["path"][0]["bias"].cpu()


def summarize_task_metrics(targets: torch.Tensor, logits: torch.Tensor, task: TaskSpec, *, threshold: float = 0.5) -> dict[str, Any]:
    if task.multilabel:
        probs = torch.sigmoid(logits).cpu().numpy()
        return compute_medical_metrics(
            targets.detach().cpu().numpy(),
            probs,
            task.label_names,
            threshold=threshold,
        )
    pred = logits.argmax(dim=-1).detach().cpu()
    truth = targets.detach().cpu().long()
    accuracy = float((pred == truth).float().mean().item()) if truth.numel() else 0.0
    return {"accuracy": accuracy}


def write_task_metrics(path: str | Path, metrics: dict[str, Any]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
