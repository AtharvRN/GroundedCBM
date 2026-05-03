import math
import unittest

from gcbm.imagenet_core import transform_box_for_model_input


class TestBboxTransforms(unittest.TestCase):
    def test_center_crop_box_transform(self) -> None:
        box = [0.25, 0.25, 0.75, 0.75]
        got = transform_box_for_model_input(box, image_size=(256, 256), input_size=224)
        self.assertIsNotNone(got)
        x1, y1, x2, y2 = got
        self.assertTrue(math.isclose(x1, 48.0 / 224.0, rel_tol=0.0, abs_tol=1e-6))
        self.assertTrue(math.isclose(y1, 48.0 / 224.0, rel_tol=0.0, abs_tol=1e-6))
        self.assertTrue(math.isclose(x2, 176.0 / 224.0, rel_tol=0.0, abs_tol=1e-6))
        self.assertTrue(math.isclose(y2, 176.0 / 224.0, rel_tol=0.0, abs_tol=1e-6))

    def test_box_dropped_when_crop_removes_it(self) -> None:
        box = [0.0, 0.0, 0.05, 0.05]
        got = transform_box_for_model_input(box, image_size=(256, 256), input_size=224)
        self.assertIsNone(got)


if __name__ == "__main__":
    unittest.main()
