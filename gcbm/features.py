from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm

from glm_saga.elasticnet import IndexedTensorDataset


class MemmapFeatureDataset(Dataset):
    def __init__(
        self,
        feature_path: Path,
        target_path: Path,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
        include_index: bool = False,
    ) -> None:
        self.features = np.load(feature_path, mmap_mode="r")
        self.targets = np.load(target_path, mmap_mode="r")
        self.mean = None if mean is None else np.asarray(mean, dtype=np.float32)
        self.std = None if std is None else np.asarray(std, dtype=np.float32)
        self.include_index = include_index

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int):
        feature = np.asarray(self.features[index], dtype=np.float32)
        if self.mean is not None and self.std is not None:
            feature = (feature - self.mean) / self.std
        tensor = torch.from_numpy(np.array(feature, dtype=np.float32, copy=True))
        target = int(self.targets[index])
        if self.include_index:
            return tensor, target, int(index)
        return tensor, target


def feature_storage_dtype(cfg: Any) -> np.dtype:
    return np.float32 if getattr(cfg, "feature_storage_dtype", "fp16") == "fp32" else np.float16


def make_feature_loader(
    features: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    *,
    indexed: bool = False,
    shuffle: bool = False,
) -> DataLoader:
    dataset = IndexedTensorDataset(features, labels) if indexed else TensorDataset(features, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def default_labeled_batch(batch: Any) -> tuple[Any, torch.Tensor]:
    """Return model input and class label from common CBM dataloader batches."""
    if not isinstance(batch, (tuple, list)) or len(batch) < 2:
        raise TypeError(f"Expected a tuple/list batch with labels, got {type(batch).__name__}")
    return batch[0], batch[-1]


def extract_labeled_feature_tensors(
    loader: DataLoader,
    feature_fn: Callable[[Any], torch.Tensor | tuple[torch.Tensor, Any]],
    *,
    batch_unpacker: Callable[[Any], tuple[Any, torch.Tensor]] = default_labeled_batch,
    progress: bool = True,
    accuracy_fn: Callable[[Any, torch.Tensor], torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, float | None]:
    """Extract feature and label tensors from a dataloader.

    ``feature_fn`` may return just features or ``(features, aux)`` where
    ``aux`` is passed to ``accuracy_fn``. Features and labels are stored on CPU.
    """
    feature_chunks: list[torch.Tensor] = []
    label_chunks: list[torch.Tensor] = []
    correct = 0.0
    total = 0
    iterator = tqdm(loader) if progress else loader
    with torch.no_grad():
        for batch in iterator:
            inputs, labels = batch_unpacker(batch)
            output = feature_fn(inputs)
            if isinstance(output, tuple):
                features, aux = output
            else:
                features, aux = output, None
            feature_chunks.append(features.detach().cpu())
            label_chunks.append(labels.detach().cpu())
            if accuracy_fn is not None:
                if aux is None:
                    raise ValueError("accuracy_fn requires feature_fn to return (features, aux)")
                batch_correct = accuracy_fn(aux, labels).detach().float().sum().item()
                correct += float(batch_correct)
                total += int(labels.numel())
    accuracy = None if accuracy_fn is None else correct / max(total, 1)
    return torch.cat(feature_chunks, dim=0), torch.cat(label_chunks, dim=0), accuracy


def standardize_from_train(
    train_x: torch.Tensor,
    *others: torch.Tensor,
    unbiased: bool = False,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, ...]:
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True, unbiased=unbiased).clamp_min(eps)
    standardized = [(train_x - mean) / std]
    standardized.extend((x - mean) / std for x in others)
    standardized.extend([mean, std])
    return tuple(standardized)


def compute_feature_stats_memmap(
    feature_path: Path,
    cfg: Any | None = None,
    chunk_size: int = 8192,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    features = np.load(feature_path, mmap_mode="r")
    n_examples, n_features = int(features.shape[0]), int(features.shape[1])
    start_time = time.perf_counter()
    sum_vec = np.zeros((n_features,), dtype=np.float64)
    sum_sq_vec = np.zeros((n_features,), dtype=np.float64)
    for start in range(0, n_examples, chunk_size):
        end = min(start + chunk_size, n_examples)
        batch = np.asarray(features[start:end], dtype=np.float32)
        sum_vec += batch.sum(axis=0, dtype=np.float64)
        sum_sq_vec += np.square(batch, dtype=np.float32).sum(axis=0, dtype=np.float64)
    mean = sum_vec / max(n_examples, 1)
    if n_examples > 1:
        var = (sum_sq_vec - (sum_vec * sum_vec) / n_examples) / (n_examples - 1)
    else:
        var = np.ones_like(mean)
    var = np.maximum(var, 1e-6)
    summary = {
        "stage": "train_feature_normalization_summary",
        "n_examples": n_examples,
        "n_features": n_features,
        "elapsed_sec": time.perf_counter() - start_time,
    }
    return torch.from_numpy(mean.astype(np.float32)), torch.from_numpy(np.sqrt(var).astype(np.float32)), summary
