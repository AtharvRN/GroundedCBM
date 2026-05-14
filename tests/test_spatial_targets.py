import numpy as np

from gcbm.spatial_targets import (
    box_to_original_pixels,
    rasterize_box_target,
    transform_box,
)


def test_box_to_original_pixels_accepts_normalized_and_pixel_boxes():
    assert box_to_original_pixels([0.25, 0.5, 0.75, 1.0], (200, 100)) == (50.0, 50.0, 150.0, 100.0)
    assert box_to_original_pixels([150, 90, 50, 10], (200, 100)) == (50.0, 10.0, 150.0, 90.0)


def test_resize_center_crop_transform_clips_box_to_crop():
    box = transform_box([0, 0, 400, 200], (400, 200), transform="resize_center_crop", input_size=224)
    assert np.allclose(box, (0.0, 0.0, 1.0, 1.0))


def test_resize_center_crop_transform_drops_box_outside_crop():
    box = transform_box([0, 0, 10, 10], (400, 200), transform="resize_center_crop", input_size=224)
    assert box is None


def test_soft_box_rasterization_has_expected_occupancy():
    mask = rasterize_box_target(
        [0.25, 0.25, 0.75, 0.75],
        (100, 100),
        target_mode="soft_box",
        mask_h=4,
        mask_w=4,
    )
    assert mask.shape == (4, 4)
    assert np.isclose(mask.sum(), 4.0)
    assert np.all(mask[1:3, 1:3] == 1.0)


def test_hard_iou_rasterization_selects_overlapping_patches():
    mask = rasterize_box_target(
        [0.25, 0.25, 0.75, 0.75],
        (100, 100),
        target_mode="hard_iou",
        mask_h=4,
        mask_w=4,
        iou_thresh=0.05,
    )
    assert mask.sum() == 4.0
    assert np.all(mask[1:3, 1:3] == 1.0)
