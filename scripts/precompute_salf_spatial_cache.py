#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F
from loguru import logger
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from gcbm.config import load_flat_config
from methods.lf import get_lf_concepts
from methods.salf import (
    RawSubset,
    _apply_prompt_masks_to_batch,
    _build_prompt_grid_metadata,
    _concept_cache_base,
    _prepare_base_clip_tensors,
    _save_cache_order,
    create_salf_splits,
    infer_clip_input_size,
    pil_collate,
)


def _base_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=str, default=None)
    return parser


def _parser(defaults: dict) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Precompute/assemble sharded SALF prompt-grid spatial caches."
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--mode", choices=["precompute", "assemble"], default="precompute")
    parser.add_argument("--splits", nargs="+", default=["train", "val"], choices=["train", "val"])
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float16")
    parser.add_argument("--force", action="store_true")

    parser.add_argument("--model_name", default=defaults.get("model_name", "salf_cbm"))
    parser.add_argument("--dataset", default=defaults.get("dataset", "chexpert"))
    parser.add_argument("--data_dir", default=defaults.get("data_dir", ""))
    parser.add_argument("--img_root", default=defaults.get("img_root", ""))
    parser.add_argument("--train_csv", default=defaults.get("train_csv", ""))
    parser.add_argument("--val_csv", default=defaults.get("val_csv", ""))
    parser.add_argument("--label_subset", default=defaults.get("label_subset", "all"))
    parser.add_argument("--uncertain_strategy", default=defaults.get("uncertain_strategy", "ones"))
    parser.add_argument("--frontal_only", action=argparse.BooleanOptionalAction, default=defaults.get("frontal_only", True))
    parser.add_argument("--concept_set", default=defaults.get("concept_set", "concept_files/cub_filtered.txt"))
    parser.add_argument("--filter_set", default=defaults.get("filter_set", None))
    parser.add_argument("--backbone", default=defaults.get("backbone", "densenet121"))
    parser.add_argument("--backbone_ckpt", default=defaults.get("backbone_ckpt", ""))
    parser.add_argument("--lf_clip_name", default=defaults.get("lf_clip_name", None))
    parser.add_argument("--device", default=defaults.get("device", "cuda"))
    parser.add_argument("--seed", type=int, default=int(defaults.get("seed", 42)))
    parser.add_argument("--val_split", type=float, default=float(defaults.get("val_split", 0.1)))
    parser.add_argument("--max_train_images", type=int, default=int(defaults.get("max_train_images", 0)))
    parser.add_argument("--max_test_images", type=int, default=int(defaults.get("max_test_images", 0)))
    parser.add_argument("--img_size", type=int, default=int(defaults.get("img_size", 224)))
    parser.add_argument("--resize_size", type=int, default=int(defaults.get("resize_size", 256)))
    parser.add_argument("--grid_h", type=int, default=int(defaults.get("grid_h", 7)))
    parser.add_argument("--grid_w", type=int, default=int(defaults.get("grid_w", 7)))
    parser.add_argument("--prompt_radius", type=int, default=int(defaults.get("prompt_radius", 3)))
    parser.add_argument("--prompt_batch_size", type=int, default=int(defaults.get("prompt_batch_size", 1024)))
    parser.add_argument("--spatial_batch_size", type=int, default=int(defaults.get("spatial_batch_size", 128)))
    parser.add_argument("--spatial_num_workers", type=int, default=int(defaults.get("spatial_num_workers", defaults.get("num_workers", 8))))
    parser.add_argument("--num_workers", type=int, default=int(defaults.get("num_workers", 8)))
    parser.add_argument("--spatial_source", default=defaults.get("spatial_source", "prompt_grid"))
    parser.add_argument("--activation_dir", default=defaults.get("activation_dir", "saved_activations"))
    parser.add_argument("--savlg_spatial_stage", default=defaults.get("savlg_spatial_stage", "conv5"))
    return parser


def parse_args() -> argparse.Namespace:
    base_args, _ = _base_parser().parse_known_args()
    defaults = load_flat_config(base_args.config) if base_args.config else {}
    args = _parser(defaults).parse_args()
    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard_index must be in [0, num_shards)")
    return args


def _slice_bounds(total: int, shard_index: int, num_shards: int) -> tuple[int, int]:
    start = total * shard_index // num_shards
    end = total * (shard_index + 1) // num_shards
    return start, end


def _slice_raw_subset(dataset: Dataset, start: int, end: int) -> Dataset:
    if isinstance(dataset, RawSubset):
        return RawSubset(dataset.base_dataset, dataset.indices[start:end])
    return torch.utils.data.Subset(dataset, range(start, end))


def _shard_dir(cache_base: str) -> Path:
    return Path(cache_base + "_shards")


def _shard_path(cache_base: str, shard_index: int, num_shards: int) -> Path:
    return _shard_dir(cache_base) / f"shard_{shard_index:05d}_of_{num_shards:05d}.pt"


def _compute_shard(
    args: argparse.Namespace,
    raw_dataset: Dataset,
    clip_model,
    clip_preprocess,
    concepts: Sequence[str],
    split_name: str,
) -> None:
    cache_base = _concept_cache_base(args, split_name, concepts)
    shard_path = _shard_path(cache_base, args.shard_index, args.num_shards)
    if shard_path.exists() and not args.force:
        logger.info("Skipping existing shard {}", shard_path)
        return

    total = len(raw_dataset)
    start, end = _slice_bounds(total, args.shard_index, args.num_shards)
    shard_dataset = _slice_raw_subset(raw_dataset, start, end)
    shard_path.parent.mkdir(parents=True, exist_ok=True)

    loader = DataLoader(
        shard_dataset,
        batch_size=int(args.spatial_batch_size),
        shuffle=False,
        num_workers=int(args.spatial_num_workers),
        collate_fn=pil_collate,
        persistent_workers=int(args.spatial_num_workers) > 0,
    )
    with torch.no_grad():
        text_emb = clip_model.encode_texts([str(concept) for concept in concepts]).float()
        text_emb = F.normalize(text_emb, dim=1)

    input_size = infer_clip_input_size(clip_preprocess)
    _, prompt_masks = _build_prompt_grid_metadata(
        image_size=input_size,
        grid_h=int(args.grid_h),
        grid_w=int(args.grid_w),
        radius=int(args.prompt_radius),
    )

    all_p: list[torch.Tensor] = []
    logger.info(
        "Computing SALF {} shard {}/{} rows [{}:{}) with spatial_batch_size={} prompt_batch_size={}",
        split_name,
        args.shard_index,
        args.num_shards,
        start,
        end,
        args.spatial_batch_size,
        args.prompt_batch_size,
    )
    for pil_images, _ in tqdm(loader, desc=f"SALF P {split_name} shard {args.shard_index}"):
        base_images = _prepare_base_clip_tensors(pil_images, clip_preprocess)
        prompted_images = _apply_prompt_masks_to_batch(base_images, prompt_masks, clip_preprocess)
        sims_chunks = []
        for batch_start in range(0, prompted_images.shape[0], int(args.prompt_batch_size)):
            image_tensor = prompted_images[batch_start : batch_start + int(args.prompt_batch_size)].to(args.device)
            with torch.no_grad():
                img_emb = clip_model.encode_images(image_tensor).float()
                img_emb = F.normalize(img_emb, dim=1)
                sim = img_emb @ text_emb.T
            sims_chunks.append(sim.cpu())
        sims = torch.cat(sims_chunks, dim=0)
        sim_maps = sims.view(len(pil_images), int(args.grid_h), int(args.grid_w), -1)
        all_p.append(sim_maps)

    p_shard = torch.cat(all_p, dim=0) if all_p else torch.empty((0, int(args.grid_h), int(args.grid_w), len(concepts)))
    if args.dtype == "float16":
        p_shard = p_shard.half()
    torch.save(
        {
            "P": p_shard,
            "split": split_name,
            "start": start,
            "end": end,
            "total": total,
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
            "dtype": str(p_shard.dtype).replace("torch.", ""),
        },
        shard_path,
    )
    logger.info("Saved SALF spatial shard {}", shard_path)


def _assemble_split(args: argparse.Namespace, concepts: Sequence[str], split_name: str) -> None:
    cache_base = _concept_cache_base(args, split_name, concepts)
    cache_path = Path(cache_base + "_P.pt")
    if cache_path.exists() and not args.force:
        logger.info("Skipping existing assembled cache {}", cache_path)
        return

    payloads = []
    expected_start = 0
    total = None
    for shard_index in range(args.num_shards):
        path = _shard_path(cache_base, shard_index, args.num_shards)
        if not path.exists():
            raise FileNotFoundError(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload["split"] != split_name:
            raise RuntimeError(f"Shard {path} has split={payload['split']} expected {split_name}")
        if int(payload["start"]) != expected_start:
            raise RuntimeError(f"Shard {path} starts at {payload['start']} expected {expected_start}")
        expected_start = int(payload["end"])
        total = int(payload["total"])
        payloads.append(payload["P"])

    if total is None or expected_start != total:
        raise RuntimeError(f"Assembled {expected_start} rows, expected total {total}")
    p_full = torch.cat(payloads, dim=0)
    if p_full.shape[0] != total:
        raise RuntimeError(f"Assembled tensor has {p_full.shape[0]} rows, expected {total}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(p_full, cache_path)
    _save_cache_order(cache_base, total)
    logger.info("Saved assembled SALF spatial cache {} shape={}", cache_path, tuple(p_full.shape))


def main() -> None:
    args = parse_args()
    if args.spatial_source != "prompt_grid":
        raise ValueError("Sharded SALF cache precompute currently supports only prompt_grid.")

    raw_concepts = get_lf_concepts(args)
    if args.mode == "assemble":
        for split_name in args.splits:
            _assemble_split(args, raw_concepts, split_name)
        return

    (
        train_raw,
        val_raw,
        _train_dataset,
        _val_dataset,
        _test_dataset,
        _backbone,
        clip_model,
        clip_preprocess,
    ) = create_salf_splits(args)
    split_to_dataset = {"train": train_raw, "val": val_raw}
    for split_name in args.splits:
        _compute_shard(args, split_to_dataset[split_name], clip_model, clip_preprocess, raw_concepts, split_name)


if __name__ == "__main__":
    main()
