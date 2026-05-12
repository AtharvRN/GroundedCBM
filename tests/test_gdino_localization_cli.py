import importlib.util
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_module():
    spec = importlib.util.spec_from_file_location(
        "eval_gdino_localization", ROOT / "scripts" / "eval_gdino_localization.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestGdinoLocalizationCli(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_module()

    def test_threshold_keys_fixed_and_mean(self) -> None:
        fixed = Namespace(threshold_mode="fixed", activation_thresholds="0.3,0.5")
        mean = Namespace(threshold_mode="mean", activation_thresholds="0.3,0.5")
        self.assertEqual(self.module.threshold_keys(fixed, [0.3, 0.5]), ["0.3", "0.5"])
        self.assertEqual(self.module.threshold_keys(mean, [0.0]), ["mean"])

    def test_shared_metric_update_perfect_mask(self) -> None:
        keys = ["0.5"]
        state = self.module.init_state(keys, [0.3, 0.5])
        score = np.zeros((1, 4, 4), dtype=np.float32)
        score[:, 1:3, 1:3] = 1.0
        gt = score.astype(bool)
        self.module.update_metrics(
            state,
            score_maps=score,
            raw_maps=score,
            gt_masks=gt,
            thresholds=[0.5],
            keys=keys,
            threshold_mode="fixed",
            box_iou_thresholds=[0.3, 0.5],
        )
        out = self.module.finalize(state, keys, [0.3, 0.5])
        self.assertEqual(out["instances"], 1)
        self.assertAlmostEqual(out["threshold_metrics"]["0.5"]["mask_iou"], 1.0)
        self.assertAlmostEqual(out["threshold_metrics"]["0.5"]["box_acc"]["0.5"], 1.0)


if __name__ == "__main__":
    unittest.main()
