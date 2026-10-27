"""Validation-loss selection shared by standalone concept-head trainers."""
import math

import torch.nn.functional as F


def loss_grid_targets(masks, maps):
    """Keep full-resolution masks for metrics; use soft occupancy for the loss."""
    if masks.shape[-2:] == maps.shape[-2:]:
        return masks
    if masks.numel() == 0:
        return masks.new_empty((*masks.shape[:2], *maps.shape[-2:]))
    return F.adaptive_avg_pool2d(
        masks.flatten(0, 1)[:, None], maps.shape[-2:]
    ).reshape(*masks.shape[:2], *maps.shape[-2:])


def minimum_loss(history):
    if not history:
        return None
    if any("val_loss" not in row or not math.isfinite(row["val_loss"]) for row in history):
        raise ValueError("Validation-loss history missing or nonfinite; legacy resume requires a new run")
    return min(history, key=lambda row: (row["val_loss"], row["epoch"]))


def stop_for_loss(history, patience, min_epochs):
    if patience < 1 or min_epochs < 1:
        raise ValueError("Validation-loss patience and minimum epochs must be positive")
    best = minimum_loss(history)
    return bool(best and history[-1]["epoch"] >= min_epochs
                and history[-1]["epoch"] - best["epoch"] >= patience)
