from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import torch


def _torch_load_cpu(path: str | Path) -> Any:
    load_kwargs: dict[str, Any] = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kwargs["weights_only"] = False
    return torch.load(path, **load_kwargs)


def _compact_masks(payload: Dict[str, Any]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    indices = payload["mask_indices"]
    masks = payload["mask_targets"]
    valid = payload.get("mask_valid")
    if isinstance(indices, torch.Tensor):
        if indices.ndim != 2 or not isinstance(masks, torch.Tensor) or valid is None:
            raise ValueError("Padded medical target payload requires mask_valid")
        compact_indices: list[torch.Tensor] = []
        compact_masks: list[torch.Tensor] = []
        valid = valid.bool()
        for row in range(indices.shape[0]):
            row_valid = valid[row]
            compact_indices.append(indices[row][row_valid].long().cpu())
            compact_masks.append(masks[row][row_valid].float().cpu())
        return compact_indices, compact_masks
    return [item.long().cpu() for item in indices], [item.float().cpu() for item in masks]


def _payload_counts(payload: Dict[str, Any]) -> np.ndarray:
    indices, _ = _compact_masks(payload)
    return np.asarray([int(item.numel()) for item in indices], dtype=np.int64)


def _metadata_from_payload(
    payload: Dict[str, Any],
    *,
    split: str,
    concepts: Sequence[str] | None = None,
    extra_metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    global_targets = payload["global_targets"]
    n_examples = int(global_targets.shape[0])
    n_concepts = int(global_targets.shape[1])
    mask_h = int(payload.get("mask_h", 0) or 0)
    mask_w = int(payload.get("mask_w", 0) or 0)
    if (mask_h <= 0 or mask_w <= 0) and "mask_targets" in payload:
        _, masks = _compact_masks(payload)
        for row_masks in masks:
            if row_masks.numel() > 0:
                mask_h = int(row_masks.shape[-2])
                mask_w = int(row_masks.shape[-1])
                break
    metadata: Dict[str, Any] = {
        "format": "medical_target_store_v1",
        "split": split,
        "n_examples": n_examples,
        "n_concepts": n_concepts,
        "mask_h": mask_h,
        "mask_w": mask_w,
        "has_presence_scores": "presence_scores" in payload,
        "global_dtype": "float32",
        "mask_dtype": "float32",
    }
    for key in (
        "matched_annotations",
        "unmatched_annotations",
        "concept_threshold",
        "neg_threshold",
        "presence_mode",
        "target_mode",
    ):
        if key in payload:
            value = payload[key]
            if isinstance(value, torch.Tensor):
                continue
            metadata[key] = value
    if concepts is not None:
        metadata["concepts"] = [str(item) for item in concepts]
    if extra_metadata:
        metadata.update(extra_metadata)
    return metadata


def _write_payload_into_store(
    payload: Dict[str, Any],
    *,
    global_targets: np.memmap,
    presence_scores: np.memmap | None,
    concept_ids: np.memmap,
    mask_targets: np.memmap,
    offsets: np.ndarray,
    row_offset: int,
    entry_offset: int,
) -> int:
    n_rows = int(payload["global_targets"].shape[0])
    global_targets[row_offset : row_offset + n_rows] = payload["global_targets"].float().cpu().numpy()
    if presence_scores is not None:
        if "presence_scores" not in payload:
            raise ValueError("Cannot write presence_scores store from payload without presence_scores")
        presence_scores[row_offset : row_offset + n_rows] = payload["presence_scores"].float().cpu().numpy()
    indices, masks = _compact_masks(payload)
    cursor = int(entry_offset)
    for local_row, (row_indices, row_masks) in enumerate(zip(indices, masks)):
        start = cursor
        count = int(row_indices.numel())
        if count > 0:
            concept_ids[cursor : cursor + count] = row_indices.to(torch.int32).numpy()
            mask_targets[cursor : cursor + count] = row_masks.float().numpy()
            cursor += count
        expected_end = int(offsets[row_offset + local_row + 1])
        if cursor != expected_end:
            raise RuntimeError(f"Target-store offset mismatch at row {row_offset + local_row}: {cursor} != {expected_end}")
        if start != int(offsets[row_offset + local_row]):
            raise RuntimeError(f"Target-store offset start mismatch at row {row_offset + local_row}")
    return cursor


def write_target_store_from_payload(
    payload: Dict[str, Any],
    root: str | Path,
    *,
    split: str,
    concepts: Sequence[str] | None = None,
    extra_metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    metadata = _metadata_from_payload(payload, split=split, concepts=concepts, extra_metadata=extra_metadata)
    counts = _payload_counts(payload)
    offsets = np.zeros((int(counts.shape[0]) + 1,), dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    np.save(root / "offsets.npy", offsets)
    total_entries = int(offsets[-1])
    global_targets = np.lib.format.open_memmap(
        root / "global_targets.npy",
        mode="w+",
        dtype=np.float32,
        shape=(metadata["n_examples"], metadata["n_concepts"]),
    )
    presence_scores = None
    if metadata["has_presence_scores"]:
        presence_scores = np.lib.format.open_memmap(
            root / "presence_scores.npy",
            mode="w+",
            dtype=np.float32,
            shape=(metadata["n_examples"], metadata["n_concepts"]),
        )
    concept_ids = np.lib.format.open_memmap(root / "concept_ids.npy", mode="w+", dtype=np.int32, shape=(total_entries,))
    mask_targets = np.lib.format.open_memmap(
        root / "mask_targets.npy",
        mode="w+",
        dtype=np.float32,
        shape=(total_entries, int(metadata["mask_h"]), int(metadata["mask_w"])),
    )
    _write_payload_into_store(
        payload,
        global_targets=global_targets,
        presence_scores=presence_scores,
        concept_ids=concept_ids,
        mask_targets=mask_targets,
        offsets=offsets,
        row_offset=0,
        entry_offset=0,
    )
    global_targets.flush()
    if presence_scores is not None:
        presence_scores.flush()
    concept_ids.flush()
    mask_targets.flush()
    metadata["total_entries"] = total_entries
    metadata["global_targets_path"] = str(root / "global_targets.npy")
    metadata["offsets_path"] = str(root / "offsets.npy")
    metadata["concept_ids_path"] = str(root / "concept_ids.npy")
    metadata["mask_targets_path"] = str(root / "mask_targets.npy")
    if presence_scores is not None:
        metadata["presence_scores_path"] = str(root / "presence_scores.npy")
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if concepts is not None:
        (root / "concepts.txt").write_text("\n".join(str(item) for item in concepts) + "\n", encoding="utf-8")
    return metadata


def write_target_store_from_shards(
    shard_paths: Sequence[str | Path],
    root: str | Path,
    *,
    split: str,
    concepts: Sequence[str] | None = None,
    extra_metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    paths = [Path(path) for path in shard_paths]
    if not paths:
        raise ValueError("No target shards provided")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    counts_per_shard: list[np.ndarray] = []
    n_examples = 0
    n_concepts: int | None = None
    mask_h: int | None = None
    mask_w: int | None = None
    has_presence_scores: bool | None = None
    matched = 0
    unmatched = 0
    for path in paths:
        payload = _torch_load_cpu(path)
        metadata = _metadata_from_payload(payload, split=split, concepts=concepts, extra_metadata=extra_metadata)
        if n_concepts is None:
            n_concepts = int(metadata["n_concepts"])
            mask_h = int(metadata["mask_h"])
            mask_w = int(metadata["mask_w"])
            has_presence_scores = bool(metadata["has_presence_scores"])
        elif n_concepts != int(metadata["n_concepts"]):
            raise ValueError(f"Shard {path} has {metadata['n_concepts']} concepts; expected {n_concepts}")
        if mask_h != int(metadata["mask_h"]) or mask_w != int(metadata["mask_w"]):
            raise ValueError(f"Shard {path} mask geometry does not match previous shards")
        if bool(has_presence_scores) != bool(metadata["has_presence_scores"]):
            raise ValueError(f"Shard {path} presence_scores availability differs from previous shards")
        counts = _payload_counts(payload)
        counts_per_shard.append(counts)
        n_examples += int(counts.shape[0])
        matched += int(payload.get("matched_annotations", 0))
        unmatched += int(payload.get("unmatched_annotations", 0))

    assert n_concepts is not None and mask_h is not None and mask_w is not None and has_presence_scores is not None
    counts_all = np.concatenate(counts_per_shard, axis=0)
    offsets = np.zeros((n_examples + 1,), dtype=np.int64)
    np.cumsum(counts_all, out=offsets[1:])
    np.save(root / "offsets.npy", offsets)
    total_entries = int(offsets[-1])
    global_targets = np.lib.format.open_memmap(root / "global_targets.npy", mode="w+", dtype=np.float32, shape=(n_examples, n_concepts))
    presence_scores = None
    if has_presence_scores:
        presence_scores = np.lib.format.open_memmap(root / "presence_scores.npy", mode="w+", dtype=np.float32, shape=(n_examples, n_concepts))
    concept_ids = np.lib.format.open_memmap(root / "concept_ids.npy", mode="w+", dtype=np.int32, shape=(total_entries,))
    mask_targets = np.lib.format.open_memmap(root / "mask_targets.npy", mode="w+", dtype=np.float32, shape=(total_entries, mask_h, mask_w))

    row_offset = 0
    entry_offset = 0
    for path in paths:
        payload = _torch_load_cpu(path)
        entry_offset = _write_payload_into_store(
            payload,
            global_targets=global_targets,
            presence_scores=presence_scores,
            concept_ids=concept_ids,
            mask_targets=mask_targets,
            offsets=offsets,
            row_offset=row_offset,
            entry_offset=entry_offset,
        )
        row_offset += int(payload["global_targets"].shape[0])
    global_targets.flush()
    if presence_scores is not None:
        presence_scores.flush()
    concept_ids.flush()
    mask_targets.flush()
    metadata = {
        "format": "medical_target_store_v1",
        "split": split,
        "n_examples": n_examples,
        "n_concepts": n_concepts,
        "mask_h": mask_h,
        "mask_w": mask_w,
        "has_presence_scores": bool(has_presence_scores),
        "global_dtype": "float32",
        "mask_dtype": "float32",
        "total_entries": total_entries,
        "matched_annotations": matched,
        "unmatched_annotations": unmatched,
        "global_targets_path": str(root / "global_targets.npy"),
        "offsets_path": str(root / "offsets.npy"),
        "concept_ids_path": str(root / "concept_ids.npy"),
        "mask_targets_path": str(root / "mask_targets.npy"),
    }
    if presence_scores is not None:
        metadata["presence_scores_path"] = str(root / "presence_scores.npy")
    if concepts is not None:
        metadata["concepts"] = [str(item) for item in concepts]
        (root / "concepts.txt").write_text("\n".join(str(item) for item in concepts) + "\n", encoding="utf-8")
    if extra_metadata:
        metadata.update(extra_metadata)
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


class MedicalPrecomputedTargetStore:
    """Memory-mapped CheXpert/MIMIC SG-CBM supervision store."""

    def __init__(self, root: str | Path) -> None:
        root = Path(root)
        metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        self.root = root
        self.metadata = metadata
        self.n_examples = int(metadata["n_examples"])
        self.n_concepts = int(metadata["n_concepts"])
        self.mask_h = int(metadata["mask_h"])
        self.mask_w = int(metadata["mask_w"])
        self.global_targets = np.load(root / "global_targets.npy", mmap_mode="r")
        presence_path = root / "presence_scores.npy"
        self.presence_scores = np.load(presence_path, mmap_mode="r") if presence_path.exists() else None
        self.offsets = np.load(root / "offsets.npy", mmap_mode="r")
        self.concept_ids = np.load(root / "concept_ids.npy", mmap_mode="r")
        self.mask_targets = np.load(root / "mask_targets.npy", mmap_mode="r")
        self.keep_indices: np.ndarray | None = None
        self.concept_remap: np.ndarray | None = None

    def __len__(self) -> int:
        return self.n_examples

    def compute_frequencies(self, chunk_size: int = 4096) -> torch.Tensor:
        counts = np.zeros((self.n_concepts,), dtype=np.float64)
        for start in range(0, self.n_examples, int(chunk_size)):
            stop = min(start + int(chunk_size), self.n_examples)
            counts += (np.asarray(self.global_targets[start:stop]) > 0).sum(axis=0)
        return torch.from_numpy((counts / max(self.n_examples, 1)).astype(np.float32))

    def set_concept_filter(self, keep_indices: Sequence[int]) -> None:
        keep = np.asarray([int(idx) for idx in keep_indices], dtype=np.int64)
        if keep.ndim != 1 or keep.size == 0:
            raise ValueError("Concept keep indices must be a non-empty 1D sequence")
        if int(keep.min()) < 0 or int(keep.max()) >= self.n_concepts:
            raise ValueError("Concept keep indices are out of bounds for precomputed medical targets")
        remap = np.full((self.n_concepts,), -1, dtype=np.int64)
        remap[keep] = np.arange(keep.size, dtype=np.int64)
        self.keep_indices = keep
        self.concept_remap = remap

    def get(self, index: int) -> Dict[str, torch.Tensor]:
        index = int(index)
        start = int(self.offsets[index])
        end = int(self.offsets[index + 1])
        global_row = np.asarray(self.global_targets[index], dtype=np.float32)
        if self.keep_indices is not None:
            global_row = global_row[self.keep_indices]
        if end <= start:
            mask_indices = torch.zeros((0,), dtype=torch.long)
            mask_targets = torch.zeros((0, self.mask_h, self.mask_w), dtype=torch.float32)
        else:
            concept_ids = np.asarray(self.concept_ids[start:end], dtype=np.int64)
            masks = np.asarray(self.mask_targets[start:end], dtype=np.float32)
            if self.concept_remap is not None:
                mapped = self.concept_remap[concept_ids]
                valid = mapped >= 0
                concept_ids = mapped[valid]
                masks = masks[valid]
            mask_indices = torch.from_numpy(np.ascontiguousarray(concept_ids).copy()).long()
            mask_targets = torch.from_numpy(np.ascontiguousarray(masks).copy()).float()
        return {
            "global_targets": torch.from_numpy(np.ascontiguousarray(global_row).copy()).float(),
            "mask_indices": mask_indices,
            "mask_targets": mask_targets,
        }
