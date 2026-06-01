from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

from gcbm.features import standardize_from_train
from gcbm.medical_annotations import load_concepts
from gcbm.medical_data import medical_labels
from gcbm.medical_metrics import compute_medical_metrics
from gcbm.task_utils import is_medical_dataset
from gcbm.train_medical import (
    MedicalBackbone,
    MedicalConceptDataset,
    apply_concept_frequency_filter,
    build_datasets,
    build_head,
    evaluate_cbl,
    evaluate_nec_weights,
    extract_concepts_resumable,
    find_concept_cache,
    get_or_build_presence_targets,
    get_or_build_targets,
    load_concept_cache_targets,
    maybe_subset,
    medical_concept_collate,
    resolve_precomputed_cache_paths,
    save_concept_cache,
    train_dense_final,
    train_one_epoch,
    train_saga_final,
)


def _medical_vlg_run_dir(args, model_name: str) -> Path:
    run_name = getattr(args, "run_name", "") or f"{args.dataset}_{model_name}"
    run_dir = Path(args.save_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def train_vlg_cbm(args) -> None:
    """Train VLG-CBM through the shared method registry.

    CUB's original VLG implementation still lives in train_cbm.py because it is
    tightly coupled to the legacy concept-dataset helpers. Medical VLG goes
    through this method handler so train_cbm.py no longer dispatches to a
    separate medical trainer.
    """
    if not is_medical_dataset(args.dataset):
        raise ValueError("methods.vlg.train_vlg_cbm currently handles medical datasets only.")

    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.device = str(device)
    args.model_name = "vlg_cbm"

    precomputed_cache_paths = resolve_precomputed_cache_paths(getattr(args, "precomputed_target_dir", ""))
    args.train_target_cache = getattr(args, "train_target_cache", "") or precomputed_cache_paths.get("train_target_cache", "")
    args.val_target_cache = getattr(args, "val_target_cache", "") or precomputed_cache_paths.get("val_target_cache", "")
    args.train_presence_cache = getattr(args, "train_presence_cache", "") or precomputed_cache_paths.get("train_presence_cache", "")
    args.val_presence_cache = getattr(args, "val_presence_cache", "") or precomputed_cache_paths.get("val_presence_cache", "")

    label_subset = getattr(args, "label_subset", "all")
    labels = medical_labels(
        args.dataset,
        competition=label_subset == "competition",
        pathology=label_subset == "pathology",
    )
    concept_file = getattr(args, "concept_file", "") or getattr(args, "concept_set", "")
    if not concept_file:
        raise ValueError("Medical VLG-CBM requires --concept_file or --concept_set")
    concepts = load_concepts(concept_file)
    run_dir = _medical_vlg_run_dir(args, "vlg_cbm")

    train_ds, val_ds = build_datasets(args, labels)
    train_ds = maybe_subset(train_ds, getattr(args, "max_train_images", 0))
    val_ds = maybe_subset(val_ds, getattr(args, "max_val_images", 0))

    concept_filter: Dict[str, Any] = {
        "original_num_concepts": len(concepts),
        "min_concept_freq": float(args.min_concept_freq),
        "max_concept_freq": float(args.max_concept_freq),
        "concept_threshold": float(args.concept_threshold),
        "presence_mode": args.presence_mode,
    }
    if getattr(args, "precomputed_target_dir", ""):
        concept_filter["precomputed_target_dir"] = str(Path(args.precomputed_target_dir))

    train_cache = getattr(args, "train_concept_cache", "") or find_concept_cache(
        args.train_annotation_dir,
        n_rows=len(train_ds),
        n_concepts=len(concepts),
        threshold=args.concept_threshold,
    )
    val_cache = getattr(args, "val_concept_cache", "") or find_concept_cache(
        args.val_annotation_dir,
        n_rows=len(val_ds),
        n_concepts=len(concepts),
        threshold=args.concept_threshold,
    )
    if train_cache and val_cache:
        train_targets = load_concept_cache_targets(
            train_ds,
            cache_path=train_cache,
            concepts=concepts,
            allow_index_fallback=args.allow_annotation_index_fallback,
        )
        val_targets = load_concept_cache_targets(
            val_ds,
            cache_path=val_cache,
            concepts=concepts,
            allow_index_fallback=args.allow_annotation_index_fallback,
        )
    elif args.train_target_cache and args.val_target_cache:
        train_targets = get_or_build_targets(train_ds, args, concepts, args.train_annotation_dir, args.train_target_cache)
        val_targets = get_or_build_targets(val_ds, args, concepts, args.val_annotation_dir, args.val_target_cache)
    else:
        if not args.train_annotation_dir or not args.val_annotation_dir:
            raise ValueError(
                "Medical VLG-CBM requires concept caches, precomputed targets, or train/val annotation dirs"
            )
        train_targets = get_or_build_presence_targets(
            train_ds,
            args,
            concepts,
            annotation_dir=args.train_annotation_dir,
            cache_path=args.train_presence_cache,
        )
        val_targets = get_or_build_presence_targets(
            val_ds,
            args,
            concepts,
            annotation_dir=args.val_annotation_dir,
            cache_path=args.val_presence_cache,
        )

    concepts, train_targets, val_targets, kept_indices, frequencies = apply_concept_frequency_filter(
        concepts,
        train_targets,
        val_targets,
        min_freq=args.min_concept_freq,
        max_freq=args.max_concept_freq,
    )
    concept_filter.update(
        {
            "kept_indices": kept_indices.tolist(),
            "train_frequencies": frequencies[kept_indices].tolist(),
            "filtered_num_concepts": len(concepts),
        }
    )

    train_target_ds = MedicalConceptDataset(train_ds, train_targets)
    val_target_ds = MedicalConceptDataset(val_ds, val_targets)
    train_loader = DataLoader(
        train_target_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=medical_concept_collate,
    )
    val_loader = DataLoader(
        val_target_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=medical_concept_collate,
    )

    backbone = MedicalBackbone(
        args.backbone,
        pretrained=args.pretrained and not args.backbone_ckpt,
        checkpoint=args.backbone_ckpt,
    ).to(device)
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    head = build_head(backbone, len(concepts), args).to(device)
    if getattr(args, "concept_head_ckpt", ""):
        state = torch.load(args.concept_head_ckpt, map_location="cpu")
        head.load_state_dict(state)
        print(f"[medical vlg] loaded concept head checkpoint: {args.concept_head_ckpt}", flush=True)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state = None
    best_val = float("inf")
    stale_epochs = 0
    metrics_path = run_dir / "metrics.jsonl"
    if args.epochs <= 0:
        if not getattr(args, "concept_head_ckpt", ""):
            raise ValueError("--epochs 0 requires --concept_head_ckpt")
        torch.save(head.state_dict(), run_dir / "concept_head_best.pt")
    else:
        for epoch in range(1, args.epochs + 1):
            train_metrics = train_one_epoch(backbone, head, train_loader, optimizer, args)
            val_metrics = evaluate_cbl(backbone, head, val_loader, args)
            print(f"[medical vlg] epoch={epoch} train={train_metrics} val={val_metrics}", flush=True)
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"epoch": epoch, "train": train_metrics, "val": val_metrics}) + "\n")
            improved = val_metrics["loss"] < (best_val - float(args.early_stop_min_delta))
            if improved:
                best_val = val_metrics["loss"]
                stale_epochs = 0
                best_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
                torch.save(best_state, run_dir / "concept_head_best.pt")
            else:
                stale_epochs += 1
            if int(args.early_stop_patience) > 0 and stale_epochs >= int(args.early_stop_patience):
                print(f"[medical vlg] early stopping at epoch={epoch} best_val_loss={best_val:.6f}", flush=True)
                break

    if best_state is not None:
        head.load_state_dict(best_state)
    torch.save(head.state_dict(), run_dir / "concept_head_final.pt")

    train_concepts, train_labels = extract_concepts_resumable(backbone, head, train_ds, args, run_dir=run_dir, split="train")
    val_concepts, val_labels = extract_concepts_resumable(backbone, head, val_ds, args, run_dir=run_dir, split="valid")
    train_z, val_z, mean, std = standardize_from_train(train_concepts, val_concepts, unbiased=False)
    save_concept_cache(run_dir, "train", train_z, train_labels, args)
    save_concept_cache(run_dir, "valid", val_z, val_labels, args)
    W, b = train_saga_final(train_z, train_labels, val_z, val_labels, args) if args.use_saga else train_dense_final(train_z, train_labels, val_z, val_labels, args)
    logits = val_z @ W.T + b
    probs = torch.sigmoid(logits).numpy()
    final_metrics = compute_medical_metrics(val_labels.numpy(), probs, labels, threshold=args.threshold)

    torch.save(W, run_dir / "W_g.pt")
    torch.save(b, run_dir / "b_g.pt")
    nec_rows = evaluate_nec_weights(val_z, val_labels, W, b, labels, concepts, args, run_dir)
    torch.save(mean, run_dir / "concept_mean.pt")
    torch.save(std, run_dir / "concept_std.pt")
    (run_dir / "concepts.txt").write_text("\n".join(concepts), encoding="utf-8")
    (run_dir / "labels.txt").write_text("\n".join(labels), encoding="utf-8")
    with (run_dir / "concept_filter.json").open("w", encoding="utf-8") as handle:
        json.dump(concept_filter, handle, indent=2)
    with (run_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                **vars(args),
                "labels": labels,
                "num_concepts": len(concepts),
                "concept_filter": {k: v for k, v in concept_filter.items() if k != "train_frequencies"},
                "train_target_summary": {k: v for k, v in train_targets.items() if isinstance(v, (int, float, str))},
                "val_target_summary": {k: v for k, v in val_targets.items() if isinstance(v, (int, float, str))},
            },
            handle,
            indent=2,
        )
    with (run_dir / "val_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(final_metrics, handle, indent=2)
    print(f"[medical vlg] validation mean AUROC={final_metrics['mean_auroc']:.4f} mAP={final_metrics['mAP']:.4f}", flush=True)
    if nec_rows:
        print(f"[medical vlg] NEC metrics={nec_rows}", flush=True)
    print(f"[medical vlg] saved to {run_dir}", flush=True)
