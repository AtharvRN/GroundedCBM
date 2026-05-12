import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from gcbm.imagenet_core import topk_accuracy
from gcbm.run_imagenet_glm_path import (
    TensorBatchLoader,
    infer_n_classes,
    parse_nec_values,
    select_path_points_for_nec,
)


class TestGlmNecUtils(unittest.TestCase):
    def test_parse_nec_values_rejects_empty_input(self) -> None:
        self.assertEqual(parse_nec_values("5, 10,20"), [5, 10, 20])
        with self.assertRaises(ValueError):
            parse_nec_values(" , ")

    def test_infer_n_classes_uses_max_label_across_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first = root / "first.npy"
            second = root / "second.npy"
            np.save(first, np.asarray([0, 2, 4], dtype=np.int64))
            np.save(second, np.asarray([1, 7], dtype=np.int64))

            self.assertEqual(infer_n_classes(first, second), 8)

    def test_tensor_batch_loader_preserves_order_and_indices(self) -> None:
        features = torch.arange(20, dtype=torch.float32).view(5, 4)
        targets = torch.arange(5, dtype=torch.long)
        loader = TensorBatchLoader(features, targets, batch_size=2, include_index=True, shuffle=False)

        batches = list(loader)

        self.assertEqual(len(batches), 3)
        first_features, first_targets, first_indices = batches[0]
        torch.testing.assert_close(first_features, features[:2])
        torch.testing.assert_close(first_targets, targets[:2])
        torch.testing.assert_close(first_indices.cpu(), torch.tensor([0, 1]))
        last_features, last_targets, last_indices = batches[-1]
        torch.testing.assert_close(last_features, features[4:])
        torch.testing.assert_close(last_targets, targets[4:])
        torch.testing.assert_close(last_indices.cpu(), torch.tensor([4]))

    def test_select_path_points_for_nec_chooses_first_sufficient_sparsity(self) -> None:
        path = [
            {"weight": torch.tensor([[1.0, 0.0, 0.0, 0.0]]), "lam": 1.0, "lr": 0.1, "metrics": {"acc": 0.1}},
            {"weight": torch.tensor([[1.0, 1.0, 0.0, 0.0]]), "lam": 0.5, "lr": 0.1, "metrics": {"acc": 0.2}},
            {"weight": torch.tensor([[1.0, 1.0, 1.0, 1.0]]), "lam": 0.1, "lr": 0.1, "metrics": {"acc": 0.3}},
        ]

        selected = select_path_points_for_nec(path, n_concepts=4, nec_values=[1, 2, 3])

        self.assertEqual([item["path_index"] for item in selected], [0, 1, 2])
        self.assertEqual([item["nnz"] for item in selected], [1, 2, 4])
        self.assertEqual(selected[1]["metrics"]["acc"], 0.2)

    def test_topk_accuracy_clamps_k_to_class_count(self) -> None:
        logits = torch.tensor([[0.1, 2.0, 0.3], [3.0, 0.1, 0.2]], dtype=torch.float32)
        targets = torch.tensor([1, 2], dtype=torch.long)

        self.assertAlmostEqual(topk_accuracy(logits, targets, k=1), 0.5)
        self.assertAlmostEqual(topk_accuracy(logits, targets, k=10), 1.0)


if __name__ == "__main__":
    unittest.main()
