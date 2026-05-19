from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

try:
    from torchvision import transforms
except Exception as exc:  # pragma: no cover
    transforms = None
    _TORCHVISION_IMPORT_ERROR = exc


CHEXPERT_LABELS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]

CHEXPERT_COMPETITION_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
]

CHEXPERT_PATHOLOGY_LABELS = [
    label for label in CHEXPERT_LABELS if label not in {"No Finding", "Support Devices"}
]

FRONTAL_VIEW_VALUES = {"ap", "pa", "frontal", "ap portable"}


def medical_labels(label_set: str = "chexpert", *, competition: bool = False, pathology: bool = False) -> List[str]:
    """Return CheXpert-style label names for CheXpert or MIMIC-CXR."""
    if label_set not in {"chexpert", "mimic"}:
        raise ValueError(f"Unsupported medical label set: {label_set}")
    if competition:
        return list(CHEXPERT_COMPETITION_LABELS)
    if pathology:
        return list(CHEXPERT_PATHOLOGY_LABELS)
    return list(CHEXPERT_LABELS)


def process_multilabel_targets(
    frame: pd.DataFrame,
    labels: List[str],
    *,
    uncertain_strategy: str = "ones",
) -> torch.Tensor:
    """Convert CheXpert-style {-1, NaN, 0, 1} label columns to binary targets."""
    missing = [label for label in labels if label not in frame.columns]
    if missing:
        raise ValueError(f"CSV is missing label columns: {missing}")
    values = frame[labels].to_numpy(dtype=np.float32)
    values = np.nan_to_num(values, nan=0.0)
    if uncertain_strategy == "ones":
        values[values == -1.0] = 1.0
    elif uncertain_strategy == "zeros":
        values[values == -1.0] = 0.0
    elif uncertain_strategy == "ignore":
        values[values == -1.0] = 0.0
    else:
        raise ValueError(f"Unsupported uncertain_strategy: {uncertain_strategy}")
    return torch.from_numpy(values.astype(np.float32, copy=False))


def get_medical_transforms(img_size: int = 224, *, train: bool = False):
    """ImageNet-normalized RGB transforms used by the existing CBM backbones."""
    if transforms is None:
        raise ImportError("torchvision is required for medical image transforms") from _TORCHVISION_IMPORT_ERROR
    if train:
        return transforms.Compose(
            [
                transforms.Resize(int(img_size) + 32),
                transforms.RandomCrop(int(img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(int(img_size) + 32),
            transforms.CenterCrop(int(img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def _detect_path_column(frame: pd.DataFrame) -> str:
    candidates = ["Path", "path", "jpg_path", "image_path", "image_path_jpg"]
    for column in candidates:
        if column in frame.columns:
            return column
    for column in frame.columns:
        if "path" in column.lower():
            return column
    raise ValueError("Could not detect an image path column")


def _filter_frontal(frame: pd.DataFrame) -> pd.DataFrame:
    if "Frontal/Lateral" in frame.columns:
        return frame[frame["Frontal/Lateral"].astype(str).str.lower() == "frontal"].reset_index(drop=True)
    if "ViewPosition" in frame.columns:
        view = frame["ViewPosition"].astype(str).str.lower()
        return frame[view.isin(FRONTAL_VIEW_VALUES)].reset_index(drop=True)
    return frame.reset_index(drop=True)


def _mimic_jpg_relpath(row: pd.Series, *, prefix: str = "files") -> str:
    subject_id = str(int(row["subject_id"])).zfill(8)
    study_id = str(int(row["study_id"])).zfill(8)
    dicom_id = str(row["dicom_id"])
    relpath = f"p{subject_id[:2]}/p{subject_id}/s{study_id}/{dicom_id}.jpg"
    return f"{prefix.rstrip('/')}/{relpath}" if prefix else relpath


class MedicalCsvDataset(Dataset):
    """CSV-backed CheXpert-style multilabel chest X-ray dataset.

    Items are dictionaries with stable `sample_id` and `image_path` fields so
    concept annotations can be aligned by identity instead of row order.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        img_root: str | Path,
        labels: List[str],
        transform: Optional[Callable] = None,
        uncertain_strategy: str = "ones",
        path_col: Optional[str] = None,
        sample_id_cols: Tuple[str, ...] = ("dicom_id", "Path", "path"),
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.img_root = Path(img_root)
        self.labels = list(labels)
        self.transform = transform
        self.path_col = path_col or _detect_path_column(self.frame)
        self.targets = process_multilabel_targets(self.frame, self.labels, uncertain_strategy=uncertain_strategy)
        self.sample_ids = [self._sample_id(i, sample_id_cols) for i in range(len(self.frame))]

    def _sample_id(self, index: int, sample_id_cols: Tuple[str, ...]) -> str:
        row = self.frame.iloc[index]
        for column in sample_id_cols:
            if column in self.frame.columns and pd.notna(row[column]):
                return str(row[column])
        return str(row[self.path_col])

    def __len__(self) -> int:
        return len(self.frame)

    def image_path(self, index: int) -> Path:
        raw_path = Path(str(self.frame.iloc[index][self.path_col]))
        return raw_path if raw_path.is_absolute() else self.img_root / raw_path

    def get_image_path(self, index: int) -> str:
        return str(self.image_path(index))

    def get_sample_id(self, index: int) -> str:
        return self.sample_ids[index]

    def get_image_size(self, index: int) -> Tuple[int, int]:
        with Image.open(self.image_path(index)) as image:
            return image.size

    def __getitem__(self, index: int) -> Dict[str, object]:
        path = self.image_path(index)
        image = Image.open(path).convert("RGB")
        image_tensor = self.transform(image) if self.transform is not None else image
        return {
            "image": image_tensor,
            "target": self.targets[index],
            "sample_id": self.sample_ids[index],
            "image_path": str(path),
            "index": int(index),
        }


def load_chexpert_dataset(
    csv_path: str | Path,
    *,
    img_root: str | Path,
    labels: List[str],
    transform: Optional[Callable] = None,
    uncertain_strategy: str = "ones",
    frontal_only: bool = True,
) -> MedicalCsvDataset:
    frame = pd.read_csv(csv_path)
    if frontal_only:
        frame = _filter_frontal(frame)
    return MedicalCsvDataset(
        frame,
        img_root=img_root,
        labels=labels,
        transform=transform,
        uncertain_strategy=uncertain_strategy,
        path_col="Path" if "Path" in frame.columns else None,
        sample_id_cols=("Path", "orig_row_idx"),
    )


def load_mimic_cxr_dataset(
    label_csv: str | Path,
    *,
    img_root: str | Path,
    split: str,
    split_csv: Optional[str | Path] = None,
    metadata_csv: Optional[str | Path] = None,
    labels: List[str],
    transform: Optional[Callable] = None,
    uncertain_strategy: str = "ones",
    frontal_only: bool = True,
    path_col: Optional[str] = None,
) -> MedicalCsvDataset:
    labels_frame = pd.read_csv(label_csv)
    frame = labels_frame
    if split_csv is not None:
        split_frame = pd.read_csv(split_csv)
        split_name = "validate" if str(split).lower() in {"valid", "val"} else str(split).lower()
        frame = frame.merge(split_frame, on=["subject_id", "study_id"], how="inner")
        frame = frame[frame["split"].astype(str).str.lower() == split_name].reset_index(drop=True)
    if metadata_csv is not None:
        metadata = pd.read_csv(metadata_csv)
        keep_cols = ["dicom_id", "subject_id", "study_id", "ViewPosition", "Rows", "Columns"]
        keep_cols = [column for column in keep_cols if column in metadata.columns]
        join_cols = ["dicom_id"] if "dicom_id" in frame.columns and "dicom_id" in metadata.columns else ["subject_id", "study_id"]
        keep_cols = [column for column in keep_cols if column in join_cols or column not in frame.columns]
        frame = frame.merge(metadata[keep_cols].drop_duplicates(join_cols), on=join_cols, how="inner")
    if path_col is None or path_col not in frame.columns:
        if "dicom_id" not in frame.columns:
            raise ValueError("MIMIC path construction requires dicom_id; provide metadata_csv or path_col")
        frame = frame.copy()
        prefix = "files" if (Path(img_root) / "files").is_dir() else ""
        frame["path"] = frame.apply(lambda row: _mimic_jpg_relpath(row, prefix=prefix), axis=1)
        path_col = "path"
    if frontal_only:
        frame = _filter_frontal(frame)
    return MedicalCsvDataset(
        frame,
        img_root=img_root,
        labels=labels,
        transform=transform,
        uncertain_strategy=uncertain_strategy,
        path_col=path_col,
        sample_id_cols=("dicom_id", "path"),
    )


def medical_collate(batch: List[Dict[str, object]]) -> Dict[str, object]:
    return {
        "image": torch.stack([item["image"] for item in batch]),  # type: ignore[arg-type]
        "target": torch.stack([item["target"] for item in batch]),  # type: ignore[arg-type]
        "sample_id": [str(item["sample_id"]) for item in batch],
        "image_path": [str(item["image_path"]) for item in batch],
        "index": torch.tensor([int(item["index"]) for item in batch], dtype=torch.long),
    }


def infer_chexpert_img_root(data_dir: str | Path) -> Path:
    data_path = Path(data_dir)
    if (data_path / "CheXpert-v1.0-small").exists():
        return data_path
    return data_path.parent


def default_mimic_paths(data_dir: str | Path) -> Dict[str, Path]:
    root = Path(data_dir)
    return {
        "label_csv": root / "mimic-cxr-2.0.0-chexpert.csv.gz",
        "split_csv": root / "mimic-cxr-2.0.0-split.csv.gz",
        "metadata_csv": root / "mimic-cxr-2.0.0-metadata.csv.gz",
    }
