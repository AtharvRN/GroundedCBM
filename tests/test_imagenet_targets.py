from types import SimpleNamespace

import numpy as np

from gcbm.imagenet_core import build_gdino_target_sample


def test_build_gdino_target_sample_maps_labels_and_filters_scores():
    cfg = SimpleNamespace(
        concept_threshold=0.5,
        spatial_target_mode="soft_box",
        mask_h=4,
        mask_w=4,
        patch_iou_thresh=0.5,
        input_size=224,
    )
    annotations = [
        {"image": "metadata"},
        {"label": "red wing", "logit": 0.9, "box": [0.0, 0.0, 1.0, 1.0]},
        {"label": "blue head", "logit": 0.4, "box": [0.0, 0.0, 1.0, 1.0]},
        {"label": "unknown", "logit": 1.0, "box": [0.0, 0.0, 1.0, 1.0]},
    ]
    global_target, indices, masks = build_gdino_target_sample(
        annotations,
        image_size=(224, 224),
        concept_to_idx={"red wing": 0, "blue head": 1},
        n_concepts=2,
        cfg=cfg,
    )
    assert global_target.tolist() == [1, 0]
    assert indices.tolist() == [0]
    assert masks.shape == (1, 4, 4)
    assert np.isclose(masks.sum(), 16.0)
