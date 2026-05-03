import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from gcbm.eval_imagenet_localization import average_precision
from gcbm.eval_imagenet_localization_helpers import finalize_threshold_metrics, normalize_maps, update_threshold_metrics
from gcbm.imagenet_annotation_index import build_filename_to_annotation_path, load_annotation_payload


class TestLocalizationMetrics(unittest.TestCase):
    def test_average_precision_toy_case(self) -> None:
        sorted_tp = np.asarray([True, False, True, False], dtype=bool)
        ap = average_precision(sorted_tp, total_gt=2)
        self.assertAlmostEqual(ap, (1.0 + (2.0 / 3.0)) / 2.0)

    def test_threshold_metrics_perfect_prediction(self) -> None:
        score_maps = torch.tensor([[[0.0, 0.0], [0.0, 1.0]]], dtype=torch.float32)
        pred_masks = score_maps > 0.5
        gt_masks = pred_masks.clone()
        gt_boxes = torch.tensor([[1.0, 1.0, 2.0, 2.0]], dtype=torch.float32)
        gt_box_valid = torch.tensor([True])
        raw = {}
        update_threshold_metrics(
            raw,
            "thr=0.5",
            score_maps=score_maps,
            pred_masks=pred_masks,
            gt_masks=gt_masks,
            gt_boxes=gt_boxes,
            gt_box_valid=gt_box_valid,
            box_iou_thresholds=[0.3, 0.5],
        )
        final = finalize_threshold_metrics(raw, [0.3, 0.5])["thr=0.5"]
        self.assertAlmostEqual(final["mask_iou"], 1.0)
        self.assertAlmostEqual(final["point_hit"], 1.0)
        self.assertAlmostEqual(final["box_acc"]["0.5"], 1.0)

    def test_concept_zscore_minmax_normalization_bounds(self) -> None:
        maps = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float32)
        norm = normalize_maps(maps, "concept_zscore_minmax")
        self.assertGreaterEqual(float(norm.min().item()), 0.0)
        self.assertLessEqual(float(norm.max().item()), 1.0)

    def test_filename_annotation_mapping_uses_image_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_root = root / "val_root" / "n00000001"
            image_root.mkdir(parents=True)
            Image.new("RGB", (4, 4), color=(0, 0, 0)).save(image_root / "ILSVRC2012_val_00000000.JPEG")
            Image.new("RGB", (4, 4), color=(0, 0, 0)).save(image_root / "ILSVRC2012_val_00000001.JPEG")
            ann_root = root / "annotations" / "imagenet_val"
            ann_root.mkdir(parents=True)
            payload = [{"image_id": 1}, {"label": "wide head", "logit": 0.9, "box": [0.1, 0.1, 0.9, 0.9]}]
            (ann_root / "0.json").write_text('[{"image_id": 999}]', encoding="utf-8")
            (ann_root / "1.json").write_text(str(payload).replace("'", '"'), encoding="utf-8")

            mapping = build_filename_to_annotation_path(ann_root, root / "val_root")
            loaded = load_annotation_payload(ann_root, 1, "ILSVRC2012_val_00000001.JPEG", mapping)
            self.assertEqual(loaded[0]["image_id"], 1)


if __name__ == "__main__":
    unittest.main()
