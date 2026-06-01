from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn

try:
    import clip as openai_clip
except ImportError:  # pragma: no cover
    openai_clip = None

try:
    from open_clip import create_model_from_pretrained, get_tokenizer
except ImportError:  # pragma: no cover
    create_model_from_pretrained = None
    get_tokenizer = None

try:
    from transformers import SiglipModel, SiglipProcessor
except ImportError:  # pragma: no cover
    SiglipModel = None
    SiglipProcessor = None


GENERIC_LF_CLIP_DEFAULT = "clip_RN50"
CHEXPERT_LF_CLIP_DEFAULT = "cxrclip_swint_mcc"
CHEXPERT_DATASET = "chexpert"
CXR_CLIP_MODEL_ID = "StanfordAIMI/XrayCLIP__vit-l-16-siglip-384__webli"
BIOMEDCLIP_MODEL_ID = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
CXR_CLIP_ALIASES = {
    "cxrclip",
    "cxr_clip",
    "clip_cxr",
    "cxr-clip",
    "xrayclip",
    "xray_clip",
    "cxrclip_swint_mcc",
    "CXR_CLIP",
}
BIOMEDCLIP_ALIASES = {
    "biomedclip",
    "biomed_clip",
    "biomed-clip",
    "BIOMEDCLIP",
}


def _unwrap_features(output, *, model: nn.Module | None = None, kind: str | None = None) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output

    for attr in ("image_embeds", "text_embeds", "embeds", "embedding", "embeddings"):
        value = getattr(output, attr, None)
        if isinstance(value, torch.Tensor):
            return value

    pooler = getattr(output, "pooler_output", None)
    if isinstance(pooler, torch.Tensor):
        if model is not None and kind in {"image", "text"}:
            proj = None
            if kind == "image":
                proj = getattr(model, "visual_projection", None) or getattr(model, "vision_projection", None)
            else:
                proj = getattr(model, "text_projection", None)
            if isinstance(proj, nn.Module):
                try:
                    return proj(pooler)
                except Exception:
                    return pooler
        return pooler

    last_hidden = getattr(output, "last_hidden_state", None)
    if isinstance(last_hidden, torch.Tensor):
        return last_hidden[:, 0] if last_hidden.dim() == 3 else last_hidden

    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]

    raise TypeError(f"Unsupported feature output type: {type(output).__name__}")


def is_chexpert_dataset(dataset: str | None) -> bool:
    return str(dataset or "").lower() == CHEXPERT_DATASET


def canonicalize_lf_clip_name(name: str) -> str:
    stripped = str(name).strip()
    lowered = stripped.lower()
    if lowered in {alias.lower() for alias in CXR_CLIP_ALIASES}:
        return CHEXPERT_LF_CLIP_DEFAULT
    if lowered in {alias.lower() for alias in BIOMEDCLIP_ALIASES}:
        return "biomedclip"
    return stripped


def resolve_lf_clip_name(name: str | None, dataset: str | None) -> str:
    if name is not None and str(name).strip():
        return canonicalize_lf_clip_name(str(name))
    if is_chexpert_dataset(dataset):
        return CHEXPERT_LF_CLIP_DEFAULT
    return GENERIC_LF_CLIP_DEFAULT


@dataclass
class AlignmentModelBundle:
    name: str
    model: object
    preprocess: object


class OpenAIClipAdapter:
    def __init__(self, clip_name: str, device: str):
        if openai_clip is None:
            raise RuntimeError("OpenAI CLIP package is not installed.")
        self.name = clip_name
        self.device = device
        self.model, self.preprocess = openai_clip.load(clip_name.replace("clip_", ""), device=device)
        self.model = self.model.float().eval()

    @torch.no_grad()
    def encode_images(self, images) -> torch.Tensor:
        if isinstance(images, torch.Tensor):
            batch = images.to(self.device)
        else:
            batch = torch.stack([self.preprocess(img) for img in list(images)], dim=0).to(self.device)
        return self.model.encode_image(batch).float().cpu()

    @torch.no_grad()
    def encode_texts(self, texts: Sequence[str]) -> torch.Tensor:
        tokens = openai_clip.tokenize([str(text) for text in texts]).to(self.device)
        return self.model.encode_text(tokens).float().cpu()


class BiomedClipAdapter:
    def __init__(self, device: str):
        if create_model_from_pretrained is None or get_tokenizer is None:
            raise RuntimeError("BiomedCLIP requires `open_clip_torch`.")
        self.name = "biomedclip"
        self.device = device
        self.model, self.preprocess = create_model_from_pretrained(BIOMEDCLIP_MODEL_ID)
        self.tokenizer = get_tokenizer(BIOMEDCLIP_MODEL_ID)
        if not hasattr(self.tokenizer, "batch_encode_plus") and hasattr(self.tokenizer, "__call__"):
            try:
                self.tokenizer.batch_encode_plus = self.tokenizer.__call__  # type: ignore[attr-defined]
            except Exception:
                pass
        base_tokenizer = getattr(self.tokenizer, "tokenizer", None)
        if base_tokenizer is not None and not hasattr(base_tokenizer, "batch_encode_plus") and hasattr(base_tokenizer, "__call__"):
            try:
                base_tokenizer.batch_encode_plus = base_tokenizer.__call__  # type: ignore[attr-defined]
            except Exception:
                pass
        self.model = self.model.to(device).eval()

    @torch.no_grad()
    def encode_images(self, images) -> torch.Tensor:
        if isinstance(images, torch.Tensor):
            batch = images.to(self.device)
        else:
            batch = torch.stack([self.preprocess(img) for img in list(images)], dim=0).to(self.device)
        return self.model.encode_image(batch).float().cpu()

    @torch.no_grad()
    def encode_texts(self, texts: Sequence[str]) -> torch.Tensor:
        tokens = self.tokenizer([str(text) for text in texts]).to(self.device)
        return self.model.encode_text(tokens).float().cpu()


class XrayClipAdapter:
    def __init__(self, device: str):
        if SiglipModel is None or SiglipProcessor is None:
            raise RuntimeError("CXR-CLIP requires `transformers` with SigLIP support.")
        self.name = CHEXPERT_LF_CLIP_DEFAULT
        self.device = device
        self.processor = SiglipProcessor.from_pretrained(CXR_CLIP_MODEL_ID)
        is_cuda = getattr(device, "type", None) == "cuda" or str(device).startswith("cuda")
        torch_dtype = torch.float16 if is_cuda else torch.float32
        try:
            self.model = SiglipModel.from_pretrained(CXR_CLIP_MODEL_ID, torch_dtype=torch_dtype)
        except TypeError:
            self.model = SiglipModel.from_pretrained(CXR_CLIP_MODEL_ID)
        self.model = self.model.to(device).eval()
        image_processor = self.processor.image_processor
        mean = getattr(image_processor, "image_mean", [0.5, 0.5, 0.5])
        std = getattr(image_processor, "image_std", [0.5, 0.5, 0.5])
        size = getattr(image_processor, "size", {"height": 384, "width": 384})
        target_h = int(size.get("height") or size.get("shortest_edge") or 384)
        target_w = int(size.get("width") or size.get("shortest_edge") or 384)
        from torchvision.transforms import CenterCrop, Compose, InterpolationMode, Normalize, Resize, ToTensor

        self.preprocess = Compose(
            [
                Resize((target_h, target_w), interpolation=InterpolationMode.BICUBIC),
                CenterCrop((target_h, target_w)),
                ToTensor(),
                Normalize(mean=mean, std=std),
            ]
        )

    @torch.no_grad()
    def encode_images(self, images) -> torch.Tensor:
        if isinstance(images, torch.Tensor):
            pixel_values = images.to(self.device)
        else:
            inputs = self.processor(images=list(images), return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)
        output = self.model.get_image_features(pixel_values=pixel_values.to(next(self.model.parameters()).dtype))
        return _unwrap_features(output, model=self.model, kind="image").float().cpu()

    @torch.no_grad()
    def encode_texts(self, texts: Sequence[str]) -> torch.Tensor:
        inputs = self.processor(text=[str(text) for text in texts], padding=True, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        output = self.model.get_text_features(**inputs)
        return _unwrap_features(output, model=self.model, kind="text").float().cpu()


def load_lf_alignment_model(name: str | None, *, dataset: str | None, device: str) -> AlignmentModelBundle:
    resolved_name = resolve_lf_clip_name(name, dataset)
    lowered = resolved_name.lower()

    if is_chexpert_dataset(dataset):
        if lowered == CHEXPERT_LF_CLIP_DEFAULT:
            adapter = XrayClipAdapter(device=device)
            return AlignmentModelBundle(name=resolved_name, model=adapter, preprocess=adapter.preprocess)
        if lowered == "biomedclip":
            adapter = BiomedClipAdapter(device=device)
            return AlignmentModelBundle(name=resolved_name, model=adapter, preprocess=adapter.preprocess)
        raise ValueError(
            "CheXpert LF/SALF supports only `cxrclip_swint_mcc` and `biomedclip`."
        )

    if lowered.startswith("clip_"):
        adapter = OpenAIClipAdapter(resolved_name, device=device)
        return AlignmentModelBundle(name=resolved_name, model=adapter, preprocess=adapter.preprocess)
    if lowered == "biomedclip":
        adapter = BiomedClipAdapter(device=device)
        return AlignmentModelBundle(name=resolved_name, model=adapter, preprocess=adapter.preprocess)
    raise ValueError(f"Unsupported lf_clip_name={resolved_name}.")
