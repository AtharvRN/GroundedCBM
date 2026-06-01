from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.utils import resolve_backbone_feature_layer
from gcbm.clip_utils import CHEXPERT_LF_CLIP_DEFAULT, GENERIC_LF_CLIP_DEFAULT, resolve_lf_clip_name
from gcbm.medical_annotations import build_medical_targets, path_match_keys
from gcbm.medical_data import load_chexpert_dataset, load_mimic_cxr_dataset, medical_labels
from gcbm.medical_metrics import compute_medical_metrics
from gcbm.train_medical import load_presence_cache_targets, resolve_precomputed_cache_paths
from methods.registry import get_train_handler
from model.cbm import Backbone
from train_cbm import _normalize_model_name


def _write_rgb_image(path: Path, color: tuple[int, int, int] = (128, 128, 128), size: tuple[int, int] = (8, 8)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=color).save(path)


def _full_label_row(active: dict[str, float] | None = None) -> dict[str, float]:
    row = {label: 0.0 for label in medical_labels("chexpert")}
    if active:
        row.update(active)
    return row


class DummyDataset:
    def __len__(self) -> int:
        return 2


class DummyDenseNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Conv2d(3, 4, kernel_size=1, bias=False)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class MedicalCoreTests(unittest.TestCase):
    def test_load_chexpert_dataset_filters_frontal_and_builds_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_rel_0 = Path("CheXpert-v1.0-small/train/patient0001/study1/view1.png")
            image_rel_1 = Path("CheXpert-v1.0-small/train/patient0002/study1/view1.png")
            _write_rgb_image(root / image_rel_0)
            _write_rgb_image(root / image_rel_1)

            frame = pd.DataFrame(
                [
                    {"Path": str(image_rel_0), "Frontal/Lateral": "Frontal", **_full_label_row({"Cardiomegaly": 1.0})},
                    {"Path": str(image_rel_1), "Frontal/Lateral": "Lateral", **_full_label_row({"Edema": 1.0})},
                ]
            )
            csv_path = root / "train.csv"
            frame.to_csv(csv_path, index=False)

            dataset = load_chexpert_dataset(
                csv_path,
                img_root=root,
                labels=medical_labels("chexpert"),
                transform=lambda image: torch.from_numpy(np.array(image)).permute(2, 0, 1).float(),
                frontal_only=True,
            )
            self.assertEqual(len(dataset), 1)
            self.assertTrue(dataset.get_image_path(0).endswith(str(image_rel_0)))
            sample = dataset[0]
            self.assertEqual(tuple(sample["image"].shape), (3, 8, 8))
            self.assertEqual(sample["sample_id"], str(image_rel_0))

    def test_load_mimic_dataset_constructs_jpg_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_rel = Path("files/p00/p00000012/s00000034/dicom123.jpg")
            _write_rgb_image(root / image_rel, color=(0, 255, 0))

            labels_frame = pd.DataFrame(
                [
                    {
                        "subject_id": 12,
                        "study_id": 34,
                        "dicom_id": "dicom123",
                        **_full_label_row({"Pleural Effusion": 1.0}),
                    }
                ]
            )
            split_frame = pd.DataFrame([{"subject_id": 12, "study_id": 34, "split": "validate"}])
            metadata_frame = pd.DataFrame(
                [{"subject_id": 12, "study_id": 34, "dicom_id": "dicom123", "ViewPosition": "PA", "Rows": 8, "Columns": 8}]
            )

            label_csv = root / "mimic-cxr-2.0.0-chexpert.csv"
            split_csv = root / "mimic-cxr-2.0.0-split.csv"
            metadata_csv = root / "mimic-cxr-2.0.0-metadata.csv"
            labels_frame.to_csv(label_csv, index=False)
            split_frame.to_csv(split_csv, index=False)
            metadata_frame.to_csv(metadata_csv, index=False)

            dataset = load_mimic_cxr_dataset(
                label_csv,
                img_root=root,
                split="validate",
                split_csv=split_csv,
                metadata_csv=metadata_csv,
                labels=medical_labels("mimic"),
                transform=lambda image: torch.from_numpy(np.array(image)).permute(2, 0, 1).float(),
                frontal_only=True,
            )
            self.assertEqual(len(dataset), 1)
            self.assertTrue(dataset.get_image_path(0).endswith(str(image_rel)))
            self.assertEqual(dataset.get_sample_id(0), "dicom123")

    def test_build_medical_targets_from_synthetic_annotations(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_rel_0 = Path("CheXpert-v1.0-small/valid/patient0001/study1/view1.png")
            image_rel_1 = Path("CheXpert-v1.0-small/valid/patient0002/study1/view1.png")
            _write_rgb_image(root / image_rel_0)
            _write_rgb_image(root / image_rel_1)

            frame = pd.DataFrame(
                [
                    {"Path": str(image_rel_0), "Frontal/Lateral": "Frontal", **_full_label_row()},
                    {"Path": str(image_rel_1), "Frontal/Lateral": "Frontal", **_full_label_row()},
                ]
            )
            csv_path = root / "valid.csv"
            frame.to_csv(csv_path, index=False)
            dataset = load_chexpert_dataset(csv_path, img_root=root, labels=medical_labels("chexpert"), transform=None, frontal_only=True)

            annotation_dir = root / "annotations"
            annotation_dir.mkdir()
            (annotation_dir / "0.json").write_text(
                json.dumps(
                    [
                        {"img_path": dataset.get_image_path(0)},
                        {"label": "opacity", "box": [1, 1, 6, 6], "logit": 0.9},
                    ]
                ),
                encoding="utf-8",
            )
            (annotation_dir / "1.json").write_text(
                json.dumps(
                    [
                        {"img_path": dataset.get_image_path(1)},
                        {"label": "effusion", "box": [2, 2, 7, 7], "logit": 0.8},
                    ]
                ),
                encoding="utf-8",
            )

            payload = build_medical_targets(
                dataset,
                annotation_dir=annotation_dir,
                concepts=["opacity", "effusion"],
                mask_h=4,
                mask_w=4,
                concept_threshold=0.5,
                presence_mode="binary",
                input_size=8,
                resize_size=8,
                num_workers=0,
            )
            self.assertEqual(tuple(payload["global_targets"].shape), (2, 2))
            self.assertEqual(payload["global_targets"].tolist(), [[1.0, 0.0], [0.0, 1.0]])
            self.assertEqual(payload["matched_annotations"], 2)
            self.assertEqual(len(payload["mask_indices"]), 2)
            self.assertEqual(payload["mask_indices"][0].tolist(), [0])
            self.assertEqual(payload["mask_indices"][1].tolist(), [1])
            self.assertEqual(tuple(payload["mask_targets"][0].shape[-2:]), (4, 4))

    def test_presence_cache_targets_rethreshold_from_presence_scores(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "presence.pt"
            torch.save(
                {
                    "num_images": 2,
                    "num_concepts": 2,
                    "concepts": ["a", "b"],
                    "presence_scores": torch.tensor([[0.2, 0.8], [0.6, 0.1]], dtype=torch.float32),
                    "global_targets": torch.tensor([[0.0, 1.0], [0.0, 0.0]], dtype=torch.float32),
                    "concept_threshold": 0.7,
                },
                cache_path,
            )

            payload = load_presence_cache_targets(
                DummyDataset(),
                cache_path=str(cache_path),
                concepts=["a", "b"],
                concept_threshold=0.5,
                neg_threshold=0.2,
                presence_mode="binary",
            )
            self.assertEqual(payload["global_targets"].tolist(), [[0.0, 1.0], [1.0, 0.0]])
            self.assertTrue(
                torch.allclose(
                    payload["presence_scores"],
                    torch.tensor([[0.2, 0.8], [0.6, 0.1]], dtype=torch.float32),
                )
            )

    def test_compute_medical_metrics_reports_macro_values(self) -> None:
        targets = np.array([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=np.float32)
        probabilities = np.array([[0.9, 0.1], [0.2, 0.8], [0.8, 0.7], [0.1, 0.2]], dtype=np.float32)
        metrics = compute_medical_metrics(targets, probabilities, ["a", "b"], threshold=0.5)
        self.assertAlmostEqual(metrics["mean_auroc"], 1.0)
        self.assertAlmostEqual(metrics["mAP"], 1.0)
        self.assertAlmostEqual(metrics["f1"]["micro"], 1.0)

    def test_resolve_precomputed_cache_paths_detects_split_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for name in ("train_targets.pt", "val_targets.pt", "train_presence.pt", "val_presence.pt"):
                (root / name).write_bytes(b"x")
            (root / "train").mkdir()
            (root / "val").mkdir()
            (root / "train" / "metadata.json").write_text("{}", encoding="utf-8")
            (root / "val" / "metadata.json").write_text("{}", encoding="utf-8")
            resolved = resolve_precomputed_cache_paths(str(root))
            self.assertTrue(resolved["train_target_store"].endswith("train"))
            self.assertTrue(resolved["val_target_store"].endswith("val"))
            self.assertTrue(resolved["train_target_cache"].endswith("train_targets.pt"))
            self.assertTrue(resolved["val_target_cache"].endswith("val_targets.pt"))
            self.assertTrue(resolved["train_presence_cache"].endswith("train_presence.pt"))
            self.assertTrue(resolved["val_presence_cache"].endswith("val_presence.pt"))

    def test_path_match_keys_include_suffix_variants(self) -> None:
        path = "/tmp/CheXpert-v1.0-small/train/patient0001/study1/view1.png"
        keys = set(path_match_keys(path))
        self.assertIn("CheXpert-v1.0-small/train/patient0001/study1/view1.png", keys)
        self.assertIn("patient0001/study1/view1.png", keys)

    def test_resolve_backbone_feature_layer_maps_densenet_layer4_to_features(self) -> None:
        self.assertEqual(resolve_backbone_feature_layer("densenet121", "layer4"), "features")
        self.assertEqual(resolve_backbone_feature_layer("densenet121", ""), "features")

    def test_backbone_uses_resolved_densenet_feature_hook(self) -> None:
        dummy_model = DummyDenseNet()
        with mock.patch("model.cbm.data_utils.get_target_model", return_value=(dummy_model, object())):
            backbone = Backbone("densenet121", "layer4", device="cpu")
        self.assertEqual(backbone.feature_layer, "features")
        output = backbone(torch.randn(2, 3, 8, 8))
        self.assertEqual(tuple(output.shape), (2, 4))

    def test_resolve_lf_clip_name_uses_chexpert_default_only_for_chexpert(self) -> None:
        self.assertEqual(resolve_lf_clip_name(None, "chexpert"), CHEXPERT_LF_CLIP_DEFAULT)
        self.assertEqual(resolve_lf_clip_name(None, "mimic"), GENERIC_LF_CLIP_DEFAULT)
        self.assertEqual(resolve_lf_clip_name(None, "cub"), GENERIC_LF_CLIP_DEFAULT)

    def test_resolve_lf_clip_name_supports_chexpert_aliases(self) -> None:
        self.assertEqual(resolve_lf_clip_name("cxrclip", "chexpert"), CHEXPERT_LF_CLIP_DEFAULT)
        self.assertEqual(resolve_lf_clip_name("CXR_CLIP", "chexpert"), CHEXPERT_LF_CLIP_DEFAULT)
        self.assertEqual(resolve_lf_clip_name("biomedclip", "chexpert"), "biomedclip")

    def test_train_cbm_uses_common_registry_for_medical_models(self) -> None:
        train_cbm_source = (REPO_ROOT / "train_cbm.py").read_text(encoding="utf-8")
        self.assertNotIn("parse_medical_args", train_cbm_source)
        self.assertNotIn("train_medical_cbm", train_cbm_source)
        self.assertEqual(get_train_handler("vlg_cbm").__module__, "methods.vlg")
        self.assertEqual(_normalize_model_name("sgcbm"), "savlg_cbm")
        self.assertEqual(_normalize_model_name("sg_cbm"), "savlg_cbm")


if __name__ == "__main__":
    unittest.main()
