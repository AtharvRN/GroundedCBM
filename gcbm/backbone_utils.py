from __future__ import annotations

import inspect
import sys
from typing import Dict

import numpy as np
import torch


def load_backbone_checkpoint(module: torch.nn.Module, path: str, *, backbone_name: str) -> None:
    if not path:
        return
    if "numpy._core" not in sys.modules:
        sys.modules["numpy._core"] = np.core
    if "numpy._core.multiarray" not in sys.modules:
        sys.modules["numpy._core.multiarray"] = np.core.multiarray

    load_kwargs = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kwargs["weights_only"] = False
    payload = torch.load(path, **load_kwargs)
    state = (
        payload.get("model_state_dict", payload.get("state_dict", payload.get("model", payload)))
        if isinstance(payload, dict)
        else payload
    )
    cleaned = _clean_backbone_state(state, backbone_name=backbone_name)
    missing, unexpected = module.load_state_dict(cleaned, strict=False)
    print(
        f"[backbone] loaded checkpoint {path}: missing={len(missing)} unexpected={len(unexpected)}",
        flush=True,
    )


def _clean_backbone_state(state: Dict[str, torch.Tensor], *, backbone_name: str) -> Dict[str, torch.Tensor]:
    cleaned: Dict[str, torch.Tensor] = {}
    for raw_key, value in state.items():
        key = raw_key.removeprefix("module.")
        if any(part in key for part in ("classifier", "classification_head", "fc.")):
            continue
        if backbone_name == "densenet121":
            for prefix in ("backbone.features.", "model.features.", "features."):
                if key.startswith(prefix):
                    cleaned[key[len(prefix) :]] = value
                    break
        elif backbone_name == "resnet50":
            mapped = _resnet_feature_key(key)
            if mapped is not None:
                cleaned[mapped] = value
    return cleaned


def _resnet_feature_key(key: str) -> str | None:
    mapping = {
        "conv1.": "conv1.",
        "bn1.": "bn1.",
        "layer1.": "layer1.",
        "layer2.": "layer2.",
        "layer3.": "layer3.",
        "layer4.": "layer4.",
    }
    for prefix, replacement in mapping.items():
        if key.startswith(prefix):
            return replacement + key[len(prefix) :]
    if key.startswith(("conv1.", "bn1.", "layer1.", "layer2.", "layer3.", "layer4.")):
        return key
    return None
