"""
Concept accuracy evaluation against GDINO GT labels.

For each (image, concept) pair on a subset of test images:
  GT label  : GDINO annotation score > 0.15 -> present=1, else 0
  Prediction: raw concept logit > 0          -> present=1, else 0
              normalized logit > 0           -> present=1, else 0 (z-scored)

Evaluated on the INTERSECTION of all model concept sets (common concepts only).
Reports per-model: Precision, Recall, F1, Accuracy.
Handles all model types: savlg_cbm, vlg_cbm, lf_cbm, salf_cbm.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))


def compute_metrics(gt: np.ndarray, pred: np.ndarray):
    tp = int(((gt == 1) & (pred == 1)).sum())
    fp = int(((gt == 0) & (pred == 1)).sum())
    fn = int(((gt == 1) & (pred == 0)).sum())
    tn = int(((gt == 0) & (pred == 0)).sum())
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    accuracy = (tp + tn) / (tp + fp + fn + tn + 1e-9)
    return dict(
        precision=precision,
        recall=recall,
        f1=f1,
        accuracy=accuracy,
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
    )


def get_common_concepts(load_paths):
    import data.utils as data_utils

    common = None
    for path in load_paths:
        with open(os.path.join(path, "concepts.txt")) as f:
            cs = set(data_utils.canonicalize_concept_label(c) for c in f.read().strip().split("\n"))
        common = cs if common is None else common & cs
    return sorted(common)


def load_gt_for_common(annotation_dir, indices, common_concepts, threshold=0.15):
    import data.utils as data_utils

    concept_set = set(common_concepts)
    concept_to_col = {c: i for i, c in enumerate(common_concepts)}

    gt = np.zeros((len(indices), len(common_concepts)), dtype=np.float32)
    for row, idx in enumerate(indices):
        ann_path = os.path.join(annotation_dir, f"{idx}.json")
        if not os.path.exists(ann_path):
            continue
        with open(ann_path) as f:
            data_ann = json.load(f)
        for ann in data_ann[1:]:
            if not isinstance(ann, dict):
                continue
            label = ann.get("label", "")
            if isinstance(label, str):
                label = data_utils.canonicalize_concept_label(label)
            if label not in concept_set:
                continue
            score = float(ann.get("logit", 0.0))
            if score > threshold:
                gt[row, concept_to_col[label]] = 1.0
    return torch.from_numpy(gt)


def eval_model(load_dir, common_concepts, image_indices, device_override=None):
    with open(os.path.join(load_dir, "args.txt")) as f:
        args = argparse.Namespace(**json.load(f))
    if device_override:
        args.device = device_override

    with open(os.path.join(load_dir, "concepts.txt")) as f:
        model_concepts = f.read().strip().split("\n")

    model_name = getattr(args, "model_name", "savlg_cbm")
    if not os.path.isdir(args.annotation_dir):
        candidate = "annotations"
        if os.path.isdir(candidate):
            args.annotation_dir = candidate
        else:
            raise FileNotFoundError(
                f"annotation_dir does not exist in saved args and no local fallback was found: {args.annotation_dir}"
            )
    print(
        f"  model={model_name}, n_model_concepts={len(model_concepts)}, n_common={len(common_concepts)}",
        flush=True,
    )

    print("  loading GDINO GT labels...", flush=True)
    ann_dir = args.annotation_dir
    split_specific = os.path.join(ann_dir, f"{args.dataset}_val")
    if os.path.isdir(split_specific):
        ann_dir = split_specific
    gt_targets = load_gt_for_common(
        ann_dir,
        image_indices,
        common_concepts,
        threshold=float(getattr(args, "cbl_confidence_threshold", 0.15)),
    )
    print(
        f"  GT positive rate: {gt_targets.mean():.3f} ({gt_targets.sum():.0f}/{gt_targets.numel()})",
        flush=True,
    )

    import data.utils as data_utils

    concept_to_idx = {data_utils.canonicalize_concept_label(c): i for i, c in enumerate(model_concepts)}
    common_cols = torch.tensor([concept_to_idx[c] for c in common_concepts if c in concept_to_idx], dtype=torch.long)

    print("  extracting concept logits...", flush=True)

    if model_name == "savlg_cbm":
        from methods.savlg import (
            build_savlg_concept_layer,
            compute_savlg_concept_logits,
            create_savlg_splits,
            forward_savlg_backbone,
            forward_savlg_concept_layer,
        )

        if getattr(args, "skip_test_eval", False):
            print(
                "  overriding saved skip_test_eval=True to force evaluation on dataset_val",
                flush=True,
            )
            args.skip_test_eval = False
        _, _, _, _, test_ds, backbone = create_savlg_splits(args)
        concept_layer = build_savlg_concept_layer(args, backbone, len(model_concepts))
        concept_layer.load_state_dict(torch.load(os.path.join(load_dir, "concept_layer.pt"), map_location=args.device))
        concept_layer.eval()
        backbone.eval()
        subset_ds = Subset(test_ds, image_indices)
        loader = DataLoader(subset_ds, batch_size=64, shuffle=False, num_workers=0)
        all_logits = []
        with torch.no_grad():
            for imgs, _ in loader:
                imgs = imgs.to(args.device)
                feats = forward_savlg_backbone(backbone, imgs, args)
                g_out, s_maps = forward_savlg_concept_layer(concept_layer, feats)
                _, _, final = compute_savlg_concept_logits(g_out, s_maps, args)
                all_logits.append(final.cpu())
        logits_full = torch.cat(all_logits, dim=0)
        mean_path = os.path.join(load_dir, "proj_mean.pt")
        std_path = os.path.join(load_dir, "proj_std.pt")
        if os.path.exists(mean_path) and os.path.exists(std_path):
            proj_mean = torch.load(mean_path, map_location="cpu")
            proj_std = torch.load(std_path, map_location="cpu")
            logits_norm_full = (logits_full - proj_mean) / proj_std
        else:
            logits_norm_full = logits_full

    elif model_name == "salf_cbm":
        from methods.salf import SpatialBackbone
        import torch.nn as nn

        backbone = SpatialBackbone(args.backbone, device=args.device)
        backbone.eval()
        sd = torch.load(os.path.join(load_dir, "concept_layer.pt"), map_location=args.device)
        w = sd["weight"]
        in_ch, out_ch = w.shape[1], w.shape[0]
        has_bias = "bias" in sd
        concept_layer = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=has_bias).to(args.device)
        concept_layer.load_state_dict(sd)
        concept_layer.eval()
        base_test = data_utils.get_data(f"{args.dataset}_val", preprocess=backbone.preprocess)
        subset_ds = Subset(base_test, image_indices)
        loader = DataLoader(subset_ds, batch_size=64, shuffle=False, num_workers=0)
        all_logits = []
        with torch.no_grad():
            for imgs, _ in loader:
                imgs = imgs.to(args.device)
                feats = backbone(imgs)
                logits = concept_layer(feats)
                if isinstance(logits, tuple):
                    logits = logits[0]
                if logits.ndim > 2:
                    logits = logits.flatten(2).max(dim=2).values
                all_logits.append(logits.cpu())
        logits_full = torch.cat(all_logits, dim=0)
        mean_path = os.path.join(load_dir, "proj_mean.pt")
        std_path = os.path.join(load_dir, "proj_std.pt")
        if os.path.exists(mean_path) and os.path.exists(std_path):
            proj_mean = torch.load(mean_path, map_location="cpu")
            proj_std = torch.load(std_path, map_location="cpu")
            logits_norm_full = (logits_full - proj_mean) / proj_std
        else:
            logits_norm_full = logits_full

    elif model_name == "lf_cbm":
        from model.cbm import load_cbm

        cbm = load_cbm(load_dir, args.device)
        cbm.eval()
        base_test = data_utils.get_data(f"{args.dataset}_val", preprocess=cbm.preprocess)
        subset_ds = Subset(base_test, image_indices)
        loader = DataLoader(subset_ds, batch_size=64, shuffle=False, num_workers=0)
        all_logits = []
        with torch.no_grad():
            for imgs, _ in loader:
                _, concept_logits = cbm(imgs.to(args.device))
                all_logits.append(concept_logits.cpu())
        logits_full = torch.cat(all_logits, dim=0)
        logits_norm_full = logits_full

    elif model_name in ("vlg_cbm", "cub_cbm"):
        from model.cbm import Backbone, ConceptLayer

        backbone_model = Backbone.from_args(load_dir, device=args.device)
        backbone_model.eval()
        cbl = ConceptLayer.from_pretrained(load_dir, device=args.device)
        cbl.eval()
        norm_mean = torch.load(os.path.join(load_dir, "train_concept_features_mean.pt"), map_location=args.device)
        norm_std = torch.load(os.path.join(load_dir, "train_concept_features_std.pt"), map_location=args.device)
        base_test = data_utils.get_data(f"{args.dataset}_val", preprocess=backbone_model.preprocess)
        subset_ds = Subset(base_test, image_indices)
        loader = DataLoader(subset_ds, batch_size=64, shuffle=False, num_workers=0)
        all_logits = []
        with torch.no_grad():
            for imgs, _ in loader:
                feats = backbone_model(imgs.to(args.device))
                concept_logits_raw = cbl(feats)
                concept_logits = (concept_logits_raw - norm_mean) / norm_std
                all_logits.append(concept_logits.cpu())
        logits_full = torch.cat(all_logits, dim=0)
        logits_norm_full = logits_full

    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    logits = logits_full[:, common_cols]
    logits_norm = logits_norm_full[:, common_cols]

    gt = gt_targets.numpy().flatten()
    pred_raw = (logits.numpy() > 0).flatten().astype(float)
    pred_norm = (logits_norm.numpy() > 0).flatten().astype(float)

    return {
        "raw": compute_metrics(gt, pred_raw),
        "normalized": compute_metrics(gt, pred_norm),
        "n_images": len(image_indices),
        "n_common_concepts": len(common_concepts),
        "gt_positive_rate": float(gt.mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_paths", nargs="+", required=True)
    parser.add_argument("--names", nargs="+", default=None)
    parser.add_argument("--n_images", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    cli = parser.parse_args()

    names = cli.names or [os.path.basename(p) for p in cli.load_paths]

    common_concepts = get_common_concepts(cli.load_paths)
    print(f"Common concept set: {len(common_concepts)} concepts across {len(cli.load_paths)} models")

    import data.utils as data_utils

    with open(os.path.join(cli.load_paths[0], "args.txt")) as f:
        ref_args = argparse.Namespace(**json.load(f))
    base_test = data_utils.get_data(f"{ref_args.dataset}_val", None)
    rng = np.random.RandomState(cli.seed)
    indices = sorted(rng.choice(len(base_test), size=min(cli.n_images, len(base_test)), replace=False).tolist())
    print(f"Evaluating on {len(indices)} images (seed={cli.seed})\n")

    results = {}
    for name, path in zip(names, cli.load_paths):
        print(f"\n=== {name} ===", flush=True)
        results[name] = eval_model(path, common_concepts, indices, cli.device)
        r = results[name]["raw"]
        n = results[name]["normalized"]
        print(
            f"  RAW  (logit>0):   P={r['precision']:.4f}  R={r['recall']:.4f}  F1={r['f1']:.4f}  Acc={r['accuracy']:.4f}",
            flush=True,
        )
        print(
            f"  NORM (logit>mu):  P={n['precision']:.4f}  R={n['recall']:.4f}  F1={n['f1']:.4f}  Acc={n['accuracy']:.4f}",
            flush=True,
        )

    print("\n" + "=" * 90, flush=True)
    print(f"Common concepts: {len(common_concepts)}  |  Images: {len(indices)}", flush=True)
    print(f"{'Model':<20} {'Threshold':<12} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Accuracy':>10}", flush=True)
    print("-" * 90, flush=True)
    for name, res in results.items():
        for thr_key, label in [("raw", "logit>0"), ("normalized", "logit>mu")]:
            m = res[thr_key]
            print(
                f"{name:<20} {label:<12} {m['precision']:>10.4f} {m['recall']:>8.4f} {m['f1']:>8.4f} {m['accuracy']:>10.4f}",
                flush=True,
            )
        print()

    if cli.output:
        os.makedirs(os.path.dirname(cli.output) or ".", exist_ok=True)
        with open(cli.output, "w") as f:
            json.dump(
                {
                    "common_concepts": common_concepts,
                    "n_images": len(indices),
                    "seed": cli.seed,
                    "results": results,
                },
                f,
                indent=2,
            )
        print(f"Saved to {cli.output}", flush=True)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
