import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from gcbm.imagenet_core import PrecomputedTargetStore, build_gdino_target_sample, precompute_target_store


class _DummyImageFolder:
    def __init__(self, image_path: str) -> None:
        self.samples = [(image_path, 0)]


class _DummyDataset:
    def __init__(self, image_path: str, annotation_dir: Path, concepts: list[str]) -> None:
        self.dataset = _DummyImageFolder(image_path)
        self.split = "train"
        self.sample_indices = None
        self.concepts = concepts
        self.concept_to_idx = {name: idx for idx, name in enumerate(concepts)}
        self.input_size = 224
        self.min_image_bytes = 1
        self.annotation_dir = annotation_dir

    def __len__(self) -> int:
        return 1

    def _load_annotation(self, sample_index: int):
        path = self.annotation_dir / "imagenet_train" / f"{sample_index}.json"
        return json.loads(path.read_text())


class TestPrecomputeTargets(unittest.TestCase):
    def test_precomputed_store_matches_direct_target_build(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_path = root / "sample.jpg"
            Image.new("RGB", (256, 256), color=(255, 255, 255)).save(image_path)
            ann_dir = root / "annotations" / "imagenet_train"
            ann_dir.mkdir(parents=True)
            annotation = [
                {"image_id": 0},
                {"label": "wide head", "logit": 0.9, "box": [0.25, 0.25, 0.75, 0.75]},
            ]
            (ann_dir / "0.json").write_text(json.dumps(annotation), encoding="utf-8")

            cfg = SimpleNamespace(
                spatial_target_mode="soft_box",
                mask_h=14,
                mask_w=14,
                input_size=224,
                concept_threshold=0.15,
                patch_iou_thresh=0.5,
            )
            dataset = _DummyDataset(str(image_path), root / "annotations", ["wide head"])
            out_root = root / "precomputed"
            precompute_target_store(dataset, out_root, cfg)

            store = PrecomputedTargetStore(out_root / "train")
            stored = store.get(0)
            expected_global, expected_ids, expected_masks = build_gdino_target_sample(
                annotation,
                image_size=(256, 256),
                concept_to_idx=dataset.concept_to_idx,
                n_concepts=len(dataset.concepts),
                cfg=cfg,
            )

            np.testing.assert_array_equal(stored["global_target"].numpy(), expected_global.astype(np.float32))
            np.testing.assert_array_equal(stored["mask_indices"].numpy(), expected_ids.astype(np.int64))
            np.testing.assert_allclose(stored["mask_targets"].numpy(), expected_masks.astype(np.float32))


if __name__ == "__main__":
    unittest.main()
