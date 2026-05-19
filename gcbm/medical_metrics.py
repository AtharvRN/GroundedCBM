from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score


def sigmoid_np(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-logits))


def _safe_mean(values: List[float]) -> float:
    valid = [value for value in values if not np.isnan(value)]
    return float(np.mean(valid)) if valid else float("nan")


def multilabel_ranking_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    labels: List[str],
) -> Dict[str, Dict[str, float]]:
    """Per-label AUROC/AP plus macro means for multilabel medical prediction."""
    targets = np.asarray(targets)
    probabilities = np.asarray(probabilities)
    if targets.shape != probabilities.shape:
        raise ValueError(f"targets/probabilities shape mismatch: {targets.shape} vs {probabilities.shape}")
    if targets.shape[1] != len(labels):
        raise ValueError("Number of labels does not match prediction width")

    auroc: Dict[str, float] = {}
    ap: Dict[str, float] = {}
    auroc_values: List[float] = []
    ap_values: List[float] = []
    for index, label in enumerate(labels):
        y_true = targets[:, index]
        y_score = probabilities[:, index]
        if np.unique(y_true).size < 2:
            auroc[label] = float("nan")
            ap[label] = float("nan")
            continue
        auc = float(roc_auc_score(y_true, y_score))
        avg_precision = float(average_precision_score(y_true, y_score))
        auroc[label] = auc
        ap[label] = avg_precision
        auroc_values.append(auc)
        ap_values.append(avg_precision)
    auroc["mean"] = _safe_mean(auroc_values)
    ap["mean"] = _safe_mean(ap_values)
    return {"auroc": auroc, "ap": ap}


def threshold_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    labels: List[str],
    *,
    threshold: float = 0.5,
) -> Dict[str, Dict[str, float]]:
    """Precision/recall/F1/support at a fixed probability threshold."""
    targets_bin = (np.asarray(targets) >= 0.5).astype(np.int32)
    preds_bin = (np.asarray(probabilities) >= float(threshold)).astype(np.int32)
    precision, recall, f1, support = precision_recall_fscore_support(
        targets_bin,
        preds_bin,
        average=None,
        zero_division=0,
    )
    micro = precision_recall_fscore_support(targets_bin, preds_bin, average="micro", zero_division=0)
    macro = precision_recall_fscore_support(targets_bin, preds_bin, average="macro", zero_division=0)

    payload = {
        "precision": {label: float(precision[idx]) for idx, label in enumerate(labels)},
        "recall": {label: float(recall[idx]) for idx, label in enumerate(labels)},
        "f1": {label: float(f1[idx]) for idx, label in enumerate(labels)},
        "support": {label: int(support[idx]) for idx, label in enumerate(labels)},
    }
    payload["precision"]["micro"] = float(micro[0])
    payload["recall"]["micro"] = float(micro[1])
    payload["f1"]["micro"] = float(micro[2])
    payload["precision"]["macro"] = float(macro[0])
    payload["recall"]["macro"] = float(macro[1])
    payload["f1"]["macro"] = float(macro[2])
    return payload


def compute_medical_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    labels: List[str],
    *,
    threshold: float = 0.5,
) -> Dict[str, object]:
    """Compute the release metrics for MIMIC-CXR/CheXpert multilabel runs."""
    ranking = multilabel_ranking_metrics(targets, probabilities, labels)
    fixed = threshold_metrics(targets, probabilities, labels, threshold=threshold)
    return {
        **ranking,
        **fixed,
        "mAP": ranking["ap"]["mean"],
        "mean_auroc": ranking["auroc"]["mean"],
        "threshold": float(threshold),
    }

