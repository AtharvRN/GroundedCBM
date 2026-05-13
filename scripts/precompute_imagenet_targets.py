import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gcbm.imagenet_core import (  # noqa: E402
    Config,
    SafeImageFolderWithAnnotations,
    load_concepts,
    precompute_target_store,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Precompute ImageNet GDINO supervision for SG-CBM training. "
            "Targets are rasterized after deterministic resize+center-crop preprocessing."
        )
    )
    parser.add_argument("--image_root", required=True, help="ImageFolder root or manifest image root for this split.")
    parser.add_argument(
        "--manifest",
        default="",
        help="Optional JSONL manifest with path, class_id, sample_index, and optional annotation_index.",
    )
    parser.add_argument("--annotation_dir", required=True, help="Directory containing imagenet_train/ or imagenet_val/ JSON files.")
    parser.add_argument("--concept_file", required=True, help="Concept text file, one concept per line.")
    parser.add_argument("--output_dir", required=True, help="Output cache root. The script writes <output_dir>/<split>/ arrays.")
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--mask_h", type=int, default=14)
    parser.add_argument("--mask_w", type=int, default=14)
    parser.add_argument("--min_image_bytes", type=int, default=2048)
    parser.add_argument("--patch_iou_thresh", type=float, default=0.5)
    parser.add_argument("--concept_threshold", type=float, default=0.15)
    parser.add_argument("--spatial_target_mode", choices=["hard_iou", "soft_box"], default="soft_box")
    parser.add_argument("--max_images", type=int, default=0, help="Optional deterministic subset size for smoke tests.")
    parser.add_argument("--seed", type=int, default=6885)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    return Config(
        mode="precompute_targets",
        train_root=args.image_root,
        train_manifest=args.manifest,
        annotation_dir=args.annotation_dir,
        concept_file=args.concept_file,
        val_root="",
        save_dir=args.output_dir,
        run_name="",
        reuse_run_dir="",
        feature_dir="",
        precomputed_target_dir="",
        persist_feature_copy=False,
        max_train_images=0,
        max_val_images=0,
        val_split=0.0,
        epochs=0,
        batch_size=1,
        workers=0,
        prefetch_factor=1,
        persistent_workers=False,
        pin_memory=False,
        device="cpu",
        amp="none",
        channels_last=False,
        tf32=False,
        cudnn_benchmark=False,
        seed=args.seed,
        min_image_bytes=args.min_image_bytes,
        input_size=args.input_size,
        resnet50_weights="v1",
        train_random_transforms=False,
        mask_h=args.mask_h,
        mask_w=args.mask_w,
        patch_iou_thresh=args.patch_iou_thresh,
        concept_threshold=args.concept_threshold,
        spatial_target_mode=args.spatial_target_mode,
        spatial_loss_mode="soft_align",
        filter_concepts_by_count=False,
        concept_min_count=1,
        concept_min_frequency=0.0,
        concept_max_frequency=1.0,
        optimizer="sgd",
        lr=0.0,
        weight_decay=0.0,
        momentum=0.0,
        global_pos_weight=1.0,
        patch_pos_weight=1.0,
        loss_global_w=1.0,
        loss_mask_w=1.0,
        branch_arch="dual",
        spatial_branch_mode="multiscale_conv45",
        spatial_stage="conv5",
        residual_alpha=0.2,
        profile_steps=0,
        warmup_steps=0,
        log_every=1000,
        save_every=1,
        skip_final_layer=True,
        final_layer_type="sparse",
        saga_batch_size=1,
        saga_workers=0,
        saga_prefetch_factor=1,
        saga_step_size=0.0,
        saga_lam=0.0,
        saga_n_iters=1,
        saga_verbose_every=1,
        dense_lr=0.0,
        dense_n_iters=1,
        feature_storage_dtype="fp16",
        saga_table_device="cpu",
        vlg_init_path="",
        vlg_concepts_path="",
        freeze_global_head=False,
        scheduler="none",
        print_config=False,
        residual_spatial_pooling="lse",
        learn_spatial_residual_scale=False,
        eval_every=0,
        feature_batch_size=1,
        feature_workers=0,
        feature_prefetch_factor=1,
    )


def compact_dataset(dataset: SafeImageFolderWithAnnotations, max_images: int, seed: int) -> None:
    if max_images <= 0 or len(dataset) <= max_images:
        return
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    indices = sorted(indices[:max_images])
    dataset.dataset.samples = [dataset.dataset.samples[idx] for idx in indices]
    if dataset.sample_indices is None:
        dataset.annotation_indices = indices
        dataset.sample_indices = list(range(len(indices)))
    else:
        annotation_indices = (
            dataset.annotation_indices
            if dataset.annotation_indices is not None
            else dataset.sample_indices
        )
        dataset.annotation_indices = [int(annotation_indices[idx]) for idx in indices]
        dataset.sample_indices = list(range(len(indices)))


def write_training_manifest(dataset: SafeImageFolderWithAnnotations, output_root: Path) -> Path:
    manifest_path = output_root / f"{dataset.split}_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row_idx, (image_path, class_id) in enumerate(dataset.dataset.samples):
            annotation_index = dataset.annotation_index_for_row(row_idx)
            class_name = str(dataset.dataset.classes[int(class_id)])
            payload: dict[str, Any] = {
                "path": str(image_path),
                "class_id": int(class_id),
                "class_name": class_name,
                "sample_index": row_idx,
                "annotation_index": annotation_index,
            }
            handle.write(json.dumps(payload) + "\n")
    return manifest_path


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    concepts = load_concepts(args.concept_file)
    (output_root / "concepts.txt").write_text("\n".join(concepts) + "\n", encoding="utf-8")
    cfg = build_config(args)
    dataset = SafeImageFolderWithAnnotations(
        root=args.image_root,
        annotation_dir=args.annotation_dir,
        concepts=concepts,
        input_size=args.input_size,
        min_image_bytes=args.min_image_bytes,
        split=args.split,
        manifest=args.manifest,
        train_random_transforms=False,
    )
    compact_dataset(dataset, args.max_images, args.seed)
    manifest_path = write_training_manifest(dataset, output_root)
    metadata = precompute_target_store(dataset, output_root, cfg)
    metadata["training_manifest_path"] = str(manifest_path)
    metadata["source_manifest_path"] = str(args.manifest or "")
    split_metadata_path = output_root / args.split / "metadata.json"
    split_metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_root), "split": args.split, "metadata": metadata}, indent=2))


if __name__ == "__main__":
    main()
