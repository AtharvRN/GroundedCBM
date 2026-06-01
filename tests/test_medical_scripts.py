from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval_gdino_localization import normalize_box, pred_masks_for_threshold, rasterize_box_union
from scripts.generate_chex_annotations import (
    encode_concepts,
    pending_index_batches,
    preprocess_image_for_chex,
    preprocess_images_for_chex,
)
from scripts.precompute_medical_targets import merge_target_shards, prepare_local_annotation_dir, resolve_output_paths
from gcbm.medical_target_store import MedicalPrecomputedTargetStore, write_target_store_from_shards


class _FakeTxtEncoder:
    def __init__(self, value: float) -> None:
        self.value = float(value)

    def encode_sentences(self, concepts: list[str]) -> torch.Tensor:
        return torch.full((len(concepts), 3), self.value, dtype=torch.float32)


class _FakeModel:
    def __init__(self, value: float) -> None:
        self.txt_encoder = _FakeTxtEncoder(value)


class _DummyDataset:
    def __init__(self, n_rows: int) -> None:
        self.n_rows = int(n_rows)

    def __len__(self) -> int:
        return self.n_rows


class MedicalScriptTests(unittest.TestCase):
    def test_resolve_output_paths_from_output_dir(self) -> None:
        args = argparse.Namespace(
            output_dir="/tmp/medical-cache",
            train_output="",
            val_output="",
            train_presence_output="",
            val_presence_output="",
        )
        resolve_output_paths(args)
        self.assertEqual(args.train_output, "/tmp/medical-cache/train_targets.pt")
        self.assertEqual(args.val_output, "/tmp/medical-cache/val_targets.pt")
        self.assertEqual(args.train_presence_output, "/tmp/medical-cache/train_presence.pt")
        self.assertEqual(args.val_presence_output, "/tmp/medical-cache/val_presence.pt")

    def test_prepare_local_annotation_dir_reindexes_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source"
            source.mkdir()
            (source / "10.json").write_text(json.dumps([{"img_path": "a"}]), encoding="utf-8")
            (source / "42.json").write_text(json.dumps([{"img_path": "b"}]), encoding="utf-8")
            out_dir = prepare_local_annotation_dir(str(source), [10, 42], root / "shard")
            self.assertTrue((out_dir / "0.json").exists())
            self.assertTrue((out_dir / "1.json").exists())

    def test_merge_target_shards_concatenates_expected_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            shard_a = root / "a.pt"
            shard_b = root / "b.pt"
            torch.save(
                {
                    "global_targets": torch.tensor([[1.0, 0.0]], dtype=torch.float32),
                    "presence_scores": torch.tensor([[0.8, 0.1]], dtype=torch.float32),
                    "mask_indices": [torch.tensor([0], dtype=torch.long)],
                    "mask_targets": [torch.ones((1, 2, 2), dtype=torch.float32)],
                    "matched_annotations": 1,
                    "unmatched_annotations": 0,
                    "num_concepts": 2,
                    "num_images": 1,
                },
                shard_a,
            )
            torch.save(
                {
                    "global_targets": torch.tensor([[0.0, 1.0]], dtype=torch.float32),
                    "presence_scores": torch.tensor([[0.2, 0.9]], dtype=torch.float32),
                    "mask_indices": [torch.tensor([1], dtype=torch.long)],
                    "mask_targets": [torch.ones((1, 2, 2), dtype=torch.float32)],
                    "matched_annotations": 1,
                    "unmatched_annotations": 0,
                    "num_concepts": 2,
                    "num_images": 1,
                },
                shard_b,
            )
            merged = merge_target_shards([shard_a, shard_b], str(root / "merged.pt"))
            self.assertEqual(tuple(merged["global_targets"].shape), (2, 2))
            self.assertEqual(merged["matched_annotations"], 2)
            self.assertEqual(len(merged["mask_indices"]), 2)

    def test_target_store_from_shards_streams_expected_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            shard_a = root / "a.pt"
            shard_b = root / "b.pt"
            torch.save(
                {
                    "global_targets": torch.tensor([[1.0, 0.0]], dtype=torch.float32),
                    "presence_scores": torch.tensor([[0.8, 0.1]], dtype=torch.float32),
                    "mask_indices": [torch.tensor([0], dtype=torch.long)],
                    "mask_targets": [torch.ones((1, 2, 2), dtype=torch.float32)],
                    "matched_annotations": 1,
                    "unmatched_annotations": 0,
                    "num_concepts": 2,
                    "num_images": 1,
                },
                shard_a,
            )
            torch.save(
                {
                    "global_targets": torch.tensor([[0.0, 1.0]], dtype=torch.float32),
                    "presence_scores": torch.tensor([[0.2, 0.9]], dtype=torch.float32),
                    "mask_indices": [torch.tensor([1], dtype=torch.long)],
                    "mask_targets": [2 * torch.ones((1, 2, 2), dtype=torch.float32)],
                    "matched_annotations": 1,
                    "unmatched_annotations": 0,
                    "num_concepts": 2,
                    "num_images": 1,
                },
                shard_b,
            )
            store_dir = root / "store" / "train"
            metadata = write_target_store_from_shards([shard_a, shard_b], store_dir, split="train", concepts=["a", "b"])
            self.assertEqual(metadata["n_examples"], 2)
            store = MedicalPrecomputedTargetStore(store_dir)
            self.assertEqual(tuple(store.get(0)["global_targets"].shape), (2,))
            self.assertEqual(store.get(1)["mask_indices"].tolist(), [1])
            self.assertTrue(torch.allclose(store.compute_frequencies(), torch.tensor([0.5, 0.5])))
            store.set_concept_filter([1])
            self.assertEqual(store.get(0)["global_targets"].tolist(), [0.0])
            self.assertEqual(store.get(1)["mask_indices"].tolist(), [0])

    def test_preprocess_image_helpers_produce_expected_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "image.png"
            Image.new("L", (9, 7), color=180).save(image_path)

            tensor, size = preprocess_image_for_chex(str(image_path), target_size=16)
            self.assertEqual(size, (9, 7))
            self.assertEqual(tuple(tensor.shape), (3, 16, 16))

            batch_tensor, sizes = preprocess_images_for_chex([str(image_path), str(image_path)], target_size=12)
            self.assertEqual(tuple(batch_tensor.shape), (2, 3, 12, 12))
            self.assertEqual(sizes, [(9, 7), (9, 7)])

    def test_encode_concepts_uses_on_disk_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            args = SimpleNamespace(chex_model_name="chex_stage3", chex_run_name="run_0")
            first = encode_concepts(_FakeModel(1.0), ["a", "b"], output_dir, args, torch.device("cpu"))
            second = encode_concepts(_FakeModel(5.0), ["a", "b"], output_dir, args, torch.device("cpu"))
            self.assertTrue(torch.equal(first.cpu(), second.cpu()))
            self.assertTrue(torch.allclose(second.cpu(), torch.ones((2, 3))))

    def test_pending_index_batches_skips_existing_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            (output_dir / "1.json").write_text("{}", encoding="utf-8")
            args = SimpleNamespace(start_idx=0, image_batch_size=2, skip_existing=True)
            batches = pending_index_batches(_DummyDataset(5), args, output_dir, end_idx=5)
            self.assertEqual(batches, [[0, 2], [3, 4]])

    def test_localization_helpers_smoke(self) -> None:
        norm = normalize_box([1, 2, 9, 10], (10, 10))
        self.assertEqual(norm, (0.1, 0.2, 0.9, 1.0))

        mask = rasterize_box_union([[1, 1, 9, 9]], (10, 10), map_h=4, map_w=4)
        self.assertEqual(mask.shape, (4, 4))
        self.assertTrue(mask.any())

        score_maps = np.array(
            [
                [[0.1, 0.9], [0.2, 0.8]],
                [[0.6, 0.4], [0.3, 0.7]],
            ],
            dtype=np.float32,
        )
        fixed_masks = pred_masks_for_threshold(score_maps, 0.5, "fixed")
        percentile_masks = pred_masks_for_threshold(score_maps, 50, "percentile")
        self.assertEqual(fixed_masks.shape, score_maps.shape)
        self.assertEqual(percentile_masks.shape, score_maps.shape)
        self.assertTrue(fixed_masks.any())


if __name__ == "__main__":
    unittest.main()
