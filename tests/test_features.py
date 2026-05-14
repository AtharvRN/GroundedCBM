from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from gcbm.features import (
    MemmapFeatureDataset,
    compute_feature_stats_memmap,
    extract_labeled_feature_tensors,
    feature_storage_dtype,
    standardize_from_train,
)


def test_memmap_feature_dataset_reads_and_normalizes(tmp_path):
    features = np.array([[1.0, 3.0], [5.0, 7.0]], dtype=np.float32)
    targets = np.array([2, 4], dtype=np.int64)
    feature_path = tmp_path / "features.npy"
    target_path = tmp_path / "targets.npy"
    np.save(feature_path, features)
    np.save(target_path, targets)
    dataset = MemmapFeatureDataset(feature_path, target_path, mean=np.array([1.0, 1.0]), std=np.array([2.0, 2.0]), include_index=True)
    x, y, idx = dataset[1]
    assert torch.allclose(x, torch.tensor([2.0, 3.0]))
    assert y == 4
    assert idx == 1


def test_compute_feature_stats_memmap_matches_unbiased_std(tmp_path):
    feature_path = tmp_path / "features.npy"
    np.save(feature_path, np.array([[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]], dtype=np.float32))
    mean, std, summary = compute_feature_stats_memmap(feature_path, chunk_size=2)
    assert torch.allclose(mean, torch.tensor([3.0, 6.0]))
    assert torch.allclose(std, torch.tensor([2.0, 4.0]))
    assert summary["n_examples"] == 3


def test_standardize_from_train_clamps_constant_features():
    train = torch.tensor([[1.0, 2.0], [1.0, 4.0]])
    test = torch.tensor([[1.0, 6.0]])
    train_z, test_z, mean, std = standardize_from_train(train, test, unbiased=False)
    assert torch.isfinite(train_z).all()
    assert torch.isfinite(test_z).all()
    assert torch.allclose(mean, torch.tensor([[1.0, 3.0]]))
    assert std[0, 0] >= 1e-6


def test_feature_storage_dtype_defaults_to_fp16():
    assert feature_storage_dtype(SimpleNamespace(feature_storage_dtype="fp32")) == np.float32
    assert feature_storage_dtype(SimpleNamespace()) == np.float16


def test_extract_labeled_feature_tensors_supports_accuracy_aux():
    loader = DataLoader(
        TensorDataset(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([0, 0])),
        batch_size=1,
        shuffle=False,
    )

    def feature_fn(inputs):
        logits = torch.stack([inputs[:, 1], inputs[:, 0]], dim=1)
        return inputs + 1.0, logits

    features, labels, accuracy = extract_labeled_feature_tensors(
        loader,
        feature_fn,
        progress=False,
        accuracy_fn=lambda logits, y: logits.argmax(dim=1) == y,
    )

    assert torch.allclose(features, torch.tensor([[2.0, 3.0], [4.0, 5.0]]))
    assert torch.equal(labels, torch.tensor([0, 0]))
    assert accuracy == 1.0
