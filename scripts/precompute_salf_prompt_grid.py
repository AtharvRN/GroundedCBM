#!/usr/bin/env python
"""Precompute SALF prompt-grid concept targets, optionally in shards."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import clip
from data import utils as data_utils
from methods.lf import get_lf_concepts
from methods.salf import (
    RawSubset,
    _apply_prompt_masks_to_batch,
    _build_prompt_grid_metadata,
    _concept_cache_base,
    _prepare_base_clip_tensors,
    infer_clip_input_size,
    pil_collate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="partimagenetpp")
    parser.add_argument("--concept_set", required=True)
    parser.add_argument("--backbone", default="clip_RN50")
    parser.add_argument("--lf_clip_name", default="clip_RN50")
    parser.add_argument("--activation_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--max_train_images", type=int, default=0)
    parser.add_argument("--max_test_images", type=int, default=0)
    parser.add_argument("--grid_h", type=int, default=7)
    parser.add_argument("--grid_w", type=int, default=7)
    parser.add_argument("--prompt_radius", type=int, default=32)
    parser.add_argument("--spatial_batch_size", type=int, default=32)
    parser.add_argument("--prompt_batch_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--split", choices=["train", "val"], required=True)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--merge_only", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def build_split_dataset(args: argparse.Namespace):
    explicit_train_manifest = os.environ.get("PARTIMAGENETPP_TRAIN_MANIFEST")
    explicit_val_manifest = os.environ.get("PARTIMAGENETPP_VAL_MANIFEST")
    if args.dataset == "partimagenetpp" and explicit_train_manifest and explicit_val_manifest:
        split_raw = data_utils.get_data(f"{args.dataset}_{args.split}", None)
        max_images = int(args.max_train_images or 0) if args.split == "train" else int(args.max_test_images or 0)
        if max_images > 0:
            return RawSubset(split_raw, list(range(min(len(split_raw), max_images))))
        return split_raw

    base_train_raw = data_utils.get_data(f"{args.dataset}_train", None)
    max_train = int(args.max_train_images or 0)
    total = min(len(base_train_raw), max_train) if max_train > 0 else len(base_train_raw)
    n_val = int(args.val_split * total)
    if args.val_split > 0 and n_val == 0 and total > 1:
        n_val = 1
    n_train = total - n_val
    generator = torch.Generator().manual_seed(args.seed)
    train_subset, val_subset = torch.utils.data.random_split(
        list(range(total)),
        [n_train, n_val],
        generator=generator,
    )
    indices = train_subset.indices if args.split == "train" else val_subset.indices
    return RawSubset(base_train_raw, indices)


def shard_bounds(n: int, shard_id: int, num_shards: int) -> tuple[int, int]:
    if not 0 <= shard_id < num_shards:
        raise ValueError(f"shard_id must be in [0, {num_shards}), got {shard_id}")
    start = (n * shard_id) // num_shards
    end = (n * (shard_id + 1)) // num_shards
    return start, end


def cache_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        dataset=args.dataset,
        backbone=args.backbone,
        lf_clip_name=args.lf_clip_name,
        activation_dir=args.activation_dir,
        spatial_source="prompt_grid",
        grid_h=args.grid_h,
        grid_w=args.grid_w,
        prompt_radius=args.prompt_radius,
    )


def shard_paths(cache_base: str, num_shards: int) -> list[Path]:
    return [Path(f"{cache_base}_shard{i:03d}of{num_shards:03d}_P.pt") for i in range(num_shards)]


def merge_shards(cache_base: str, dataset_len: int, num_shards: int, force: bool) -> None:
    final_path = Path(cache_base + "_P.pt")
    meta_path = Path(cache_base + "_meta.json")
    if final_path.exists() and not force:
        print(f"[salf-precompute] final cache exists, skipping merge: {final_path}", flush=True)
        return
    paths = shard_paths(cache_base, num_shards)
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing shard cache(s):\n" + "\n".join(missing))
    chunks = [torch.load(path, map_location="cpu", weights_only=False) for path in paths]
    merged = torch.cat(chunks, dim=0)
    if int(merged.shape[0]) != int(dataset_len):
        raise RuntimeError(f"Merged length mismatch: {merged.shape[0]} vs {dataset_len}")
    torch.save(merged, final_path)
    with meta_path.open("w") as f:
        json.dump({"dataset_len": dataset_len}, f, indent=2)
    print(f"[salf-precompute] saved merged cache: {final_path}", flush=True)


def compute_shard(
    args: argparse.Namespace,
    raw_dataset,
    concepts: Sequence[str],
    cache_base: str,
) -> None:
    paths = shard_paths(cache_base, args.num_shards)
    out_path = paths[args.shard_id]
    if out_path.exists() and not args.force:
        print(f"[salf-precompute] shard exists, skipping: {out_path}", flush=True)
        return

    clip_name = (args.lf_clip_name or args.backbone).replace("clip_", "")
    clip_model, clip_preprocess = clip.load(clip_name, device=args.device)
    clip_model = clip_model.float().eval()

    start, end = shard_bounds(len(raw_dataset), args.shard_id, args.num_shards)
    shard_dataset = RawSubset(raw_dataset, list(range(start, end)))
    loader = DataLoader(
        shard_dataset,
        batch_size=args.spatial_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=pil_collate,
        persistent_workers=args.num_workers > 0,
    )

    tokens = clip.tokenize([str(concept) for concept in concepts]).to(args.device)
    with torch.no_grad():
        text_emb = clip_model.encode_text(tokens).float()
        text_emb = torch.nn.functional.normalize(text_emb, dim=1)

    input_size = infer_clip_input_size(clip_preprocess)
    _, prompt_masks = _build_prompt_grid_metadata(
        image_size=input_size,
        grid_h=args.grid_h,
        grid_w=args.grid_w,
        radius=args.prompt_radius,
    )

    all_p = []
    print(
        "[salf-precompute] "
        f"split={args.split} shard={args.shard_id}/{args.num_shards} "
        f"range=[{start},{end}) n={len(shard_dataset)} grid={args.grid_h}x{args.grid_w} "
        f"radius={args.prompt_radius} spatial_bs={args.spatial_batch_size} "
        f"prompt_bs={args.prompt_batch_size}",
        flush=True,
    )
    for pil_images, _ in tqdm(loader, desc=f"SALF P {args.split} shard {args.shard_id}"):
        base_images = _prepare_base_clip_tensors(pil_images, clip_preprocess)
        prompted_images = _apply_prompt_masks_to_batch(base_images, prompt_masks, clip_preprocess)
        sims_chunks = []
        for batch_start in range(0, prompted_images.shape[0], args.prompt_batch_size):
            image_tensor = prompted_images[batch_start : batch_start + args.prompt_batch_size].to(
                args.device
            )
            with torch.no_grad():
                img_emb = clip_model.encode_image(image_tensor).float()
                img_emb = torch.nn.functional.normalize(img_emb, dim=1)
                sims_chunks.append((img_emb @ text_emb.T).cpu())
        sims = torch.cat(sims_chunks, dim=0)
        sim_maps = sims.view(len(pil_images), args.grid_h, args.grid_w, -1)
        all_p.append(sim_maps)

    p_shard = torch.cat(all_p, dim=0) if all_p else torch.empty(0, args.grid_h, args.grid_w, len(concepts))
    torch.save(p_shard, out_path)
    print(f"[salf-precompute] saved shard: {out_path} shape={tuple(p_shard.shape)}", flush=True)


def main() -> None:
    args = parse_args()
    os.makedirs(args.activation_dir, exist_ok=True)
    concepts = get_lf_concepts(args)
    raw_dataset = build_split_dataset(args)
    cache_base = _concept_cache_base(cache_args(args), args.split, concepts)
    if args.merge_only:
        merge_shards(cache_base, len(raw_dataset), args.num_shards, args.force)
    else:
        compute_shard(args, raw_dataset, concepts, cache_base)


if __name__ == "__main__":
    main()
