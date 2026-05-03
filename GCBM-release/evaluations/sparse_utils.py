import argparse
import json
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import model.utils as utils
import data.utils as data_utils
from data.concept_dataset import get_concept_dataloader, get_final_layer_dataset
from glm_saga.elasticnet import IndexedTensorDataset, glm_saga
from methods.lf import TransformedSubset, use_original_label_free_protocol
from methods.salf import SpatialBackbone, build_spatial_concept_layer
from methods.savlg import (
    build_savlg_concept_layer,
    compute_savlg_concept_logits,
    create_savlg_splits,
    forward_savlg_backbone,
    forward_savlg_concept_layer,
)
from model.cbm import (
    Backbone,
    BackboneCLIP,
    ConceptLayer,
    NormalizationLayer,
    load_cbm,
)

import numpy as np

MAX_GLM_STEP = 150
GLM_STEP_SIZE = 2 ** 0.1
DEFAULT_MEASURE_LEVEL = (5, 10, 15, 20, 25, 30)


def measure_acc(
    num_concepts,
    num_classes,
    num_samples,
    train_loader,
    val_loader,
    test_concept_loader,
    saga_step_size=0.1,
    saga_n_iters=500,
    device="cuda",
    max_lam=0.01,
    measure_level=DEFAULT_MEASURE_LEVEL,
    max_glm_steps=MAX_GLM_STEP,
):
    linear = torch.nn.Linear(num_concepts, num_classes).to(device)
    linear.weight.data.zero_()
    linear.bias.data.zero_()

    ALPHA = 0.99
    metadata = {}
    metadata["max_reg"] = {}
    metadata["max_reg"]["nongrouped"] = max_lam
    # Solve the GLM path
    max_sparsity = measure_level[-1] / num_concepts
    output_proj = glm_saga(linear, train_loader, saga_step_size, saga_n_iters, ALPHA, k=max_glm_steps, epsilon=1 / (GLM_STEP_SIZE ** max_glm_steps),
                    val_loader=val_loader, test_loader=test_concept_loader, do_zero=False, metadata=metadata, n_ex=num_samples, n_classes=num_classes,
                    max_sparsity=max_sparsity, eval_train=False, eval_val=False, eval_test=False)
    path = output_proj['path']
    sparsity_list = [(params['weight'].abs() > 1e-5).float().mean().item() for params in path]

    # Measure accuracy on test set
    final_layer = torch.nn.Linear(num_concepts, num_classes)
    accs = []
    weights = []
    for eff_concept_num in measure_level:
        target_sparsity = eff_concept_num / num_concepts
        # Pick the lam with sparsity closest to target
        for i, sparsity in enumerate(sparsity_list):
            if sparsity >= target_sparsity:
                break
        params = path[i]
        W_g, b_g, lam = params["weight"], params["bias"], params["lam"]
        print(eff_concept_num, lam, sparsity)
        print(
            f"Num of effective concept: {eff_concept_num}. Choose lambda={lam:.6f} with sparsity {sparsity:.4f}"
        )
        W_g_trunc = utils.weight_truncation(W_g, target_sparsity)
        weight_contribs = torch.sum(torch.abs(W_g_trunc), dim=0)
        print(
            "Num concepts with outgoing weights:{}/{}".format(
                torch.sum(weight_contribs > 1e-5), len(weight_contribs)
            )
        )
        print(target_sparsity, (W_g_trunc.abs() > 0).sum())
        final_layer.load_state_dict({"weight": W_g_trunc, "bias": b_g})
        final_layer = final_layer.to(device)
        weights.append((W_g_trunc, b_g))
        # Test final weights
        correct = []
        for x, y in test_concept_loader:
            x, y = x.to(device), y.to(device)
            pred = final_layer(x).argmax(dim=-1)
            correct.append(pred == y)
        correct = torch.cat(correct)
        accs.append(correct.float().mean().item())
        print(f"Test Acc: {correct.float().mean():.4f}")
    print(f"Average acc: {sum(accs) / len(accs):.4f}")
    return path, {NEC: weight for NEC, weight in zip(measure_level, weights)}, accs


def sparsity_acc_test(load_dir, lam_max=0.1, bot_filter=0, anno=None, n_iters=None, max_glm_steps=MAX_GLM_STEP):
    # Load arguments
    with open(os.path.join(load_dir, "args.txt"), "r") as f:
        args = json.load(f)
        args = argparse.Namespace(**args)
    if anno is not None:
        args.annotation_dir = anno
    with open(os.path.join(load_dir, "concepts.txt"), "r") as f:
        concepts = f.read().split("\n")
    classes = data_utils.get_classes(args.dataset)
    if anno is None:
        anno = args.annotation_dir
    # Concept filtering
    filtered_idx = None
    if args.backbone.startswith("clip_"):
        backbone = BackboneCLIP(
            args.backbone, device=args.device, use_penultimate=args.use_clip_penultimate
        )
    else:
        backbone = Backbone(args.backbone, args.feature_layer, args.device)
    if os.path.exists(os.path.join(load_dir, "backbone.pt")):
        ckpt = torch.load(os.path.join(load_dir, "backbone.pt"))
        backbone.backbone.load_state_dict(ckpt)
    cbl = ConceptLayer.from_pretrained(load_dir, args.device)
    train_cbl_loader = get_concept_dataloader(
        args.dataset,
        "train",
        concepts,
        backbone.preprocess,
        batch_size=args.cbl_batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        val_split=args.val_split,
        seed=args.seed,
        label_dir=anno,
    )
    val_cbl_loader = get_concept_dataloader(
        args.dataset,
        "val",
        concepts,
        backbone.preprocess,
        batch_size=args.cbl_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        val_split=args.val_split,
        seed=args.seed,
        label_dir=anno,
    )
    test_cbl_loader = get_concept_dataloader(
        args.dataset,
        "test",
        concepts,
        backbone.preprocess,
        batch_size=args.cbl_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        val_split=None,
        seed=args.seed,
        label_dir=anno,
    )
    # Calculating test features
    train_concept_loader, val_concept_loader, _ = get_final_layer_dataset(
        backbone,
        cbl,
        train_cbl_loader,
        val_cbl_loader,
        save_dir=load_dir,
        load_dir=load_dir
        if os.path.exists(os.path.join(load_dir, "train_concept_features.pt"))
        else None,
        batch_size=args.saga_batch_size,
        filter=filtered_idx,
    )
    normalization = NormalizationLayer.from_pretrained(load_dir, args.device)
    with torch.no_grad():
        test_concept_features = []
        test_concept_labels = []
        for features, _, labels in tqdm(test_cbl_loader):
            features = features.to(args.device)
            concept_logits = normalization(cbl(backbone(features)))
            test_concept_features.append(concept_logits.detach().cpu())
            test_concept_labels.append(labels)
        test_concept_features = torch.cat(test_concept_features, dim=0)
        concept_labels = torch.cat(test_concept_labels, dim=0)
    test_concept_dataset = TensorDataset(test_concept_features, concept_labels)
    test_concept_loader = DataLoader(
        test_concept_dataset, batch_size=args.saga_batch_size, shuffle=False
    )

    path, truncated_weights, accs = measure_acc(
        len(concepts),
        len(classes),
        len(train_concept_loader.dataset),
        train_concept_loader,
        val_concept_loader,
        test_concept_loader,
        saga_step_size=args.saga_step_size,
        saga_n_iters=n_iters if n_iters is not None else args.saga_n_iters,
        device=args.device,
        max_lam=lam_max,
        max_glm_steps=max_glm_steps,
    )
    sparsity_list = [
        (params["weight"].abs() > 1e-5).float().mean().item() for params in path
    ]
    NEC = [len(concepts) * sparsity for sparsity in sparsity_list]
    acc = [params["metrics"].get("acc_test", float("nan")) for params in path]
    df = pd.DataFrame(data={"NEC": NEC, "Accuracy": acc})
    df.to_csv(os.path.join(load_dir, "metrics.csv"))
    # Save truncated weights
    for NEC in truncated_weights:
        W, b = truncated_weights[NEC]
        torch.save(W, os.path.join(load_dir, f"W_g@NEC={NEC:d}.pt"))
        torch.save(b, os.path.join(load_dir, f"b_g@NEC={NEC:d}.pt"))
    return accs


def sparsity_acc_test_lf_cbm(load_dir, lam_max=0.1, n_iters=None, max_glm_steps=MAX_GLM_STEP):
    # Load arguments
    with open(os.path.join(load_dir, "args.txt"), "r") as f:
        args = json.load(f)
        args = argparse.Namespace(**args)
    if not hasattr(args, "batch_size"):
        args.batch_size = getattr(args, "lf_batch_size", getattr(args, "cbl_batch_size", 64))
    if not hasattr(args, "n_iters"):
        args.n_iters = getattr(args, "saga_n_iters", 500)
    if not hasattr(args, "saga_batch_size"):
        args.saga_batch_size = getattr(args, "batch_size", 256)
    with open(os.path.join(load_dir, "concepts.txt"), "r") as f:
        concepts = f.read().split("\n")
    classes = data_utils.get_classes(args.dataset)
    # Concept filtering

    cbm = load_cbm(load_dir, args.device)
    cbm.eval()
    train_dataset = data_utils.get_data(args.dataset + "_train", preprocess=cbm.preprocess)
    test_dataset = data_utils.get_data(args.dataset + "_val", preprocess=cbm.preprocess)
    # Calculating test features
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, num_workers=8, batch_size=args.batch_size
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, shuffle=False, num_workers=8, batch_size=args.batch_size
    )
    with torch.no_grad():
        final_loaders = []
        for i, loader in enumerate([train_dataloader, test_dataloader]):
            concept_features = []
            concept_labels = []
            correct = 0
            for features, labels in tqdm(loader):
                features = features.to(args.device)
                pred, concept_logits = cbm(features)
                concept_features.append(concept_logits.detach().cpu())
                correct += (pred.argmax(dim=-1) == labels.to(args.device)).float().sum()
                concept_labels.append(labels)
            print("Accuracy: ", correct / len(loader.dataset))
            concept_features = torch.cat(concept_features, dim=0)
            concept_labels = torch.cat(concept_labels, dim=0)
            concept_dataset = (
                IndexedTensorDataset(concept_features, concept_labels)
                if i == 0
                else TensorDataset(concept_features, concept_labels)
            )
            concept_loader = DataLoader(
                concept_dataset, batch_size=args.saga_batch_size, shuffle=False
            )
            final_loaders.append(concept_loader)
    train_concept_loader, test_concept_loader = final_loaders
    path, truncated_weights, accs = measure_acc(
        len(concepts),
        len(classes),
        len(train_concept_loader.dataset),
        train_concept_loader,
        None,
        test_concept_loader,
        saga_n_iters=n_iters if n_iters is not None else args.n_iters,
        device=args.device,
        max_lam=lam_max,
        max_glm_steps=max_glm_steps,
    )
    sparsity_list = [
        (params["weight"].abs() > 1e-5).float().mean().item() for params in path
    ]
    NEC = [len(concepts) * sparsity for sparsity in sparsity_list]
    acc = [params["metrics"].get("acc_test", float("nan")) for params in path]
    df = pd.DataFrame(data={"NEC": NEC, "Accuracy": acc})
    df.to_csv(os.path.join(load_dir, "metrics.csv"))
    # Save truncated weights
    for NEC in truncated_weights:
        W, b = truncated_weights[NEC]
        torch.save(W, os.path.join(load_dir, f"W_g@NEC={NEC:d}.pt"))
        torch.save(b, os.path.join(load_dir, f"b_g@NEC={NEC:d}.pt"))
    return accs


def _extract_salf_concepts(backbone, concept_layer, loader, device):
    backbone.eval()
    concept_layer.eval()
    concept_features = []
    concept_labels = []
    with torch.no_grad():
        for images, labels in tqdm(loader):
            images = images.to(device)
            maps = concept_layer(backbone(images))
            pooled = torch.nn.functional.adaptive_avg_pool2d(maps, 1).flatten(1)
            concept_features.append(pooled.cpu())
            concept_labels.append(labels)
    return torch.cat(concept_features, dim=0), torch.cat(concept_labels, dim=0)


def _extract_savlg_concepts(args, backbone, concept_layer, loader):
    backbone.eval()
    concept_layer.eval()
    concept_features = []
    concept_labels = []
    with torch.no_grad():
        for images, labels in tqdm(loader):
            if isinstance(images, dict):
                feats = {
                    key: value.to(args.device, non_blocking=True)
                    for key, value in images.items()
                }
            elif isinstance(images, torch.Tensor) and images.ndim == 4 and int(images.shape[1]) != 3:
                feats = images.to(args.device, non_blocking=True)
            else:
                images = images.to(args.device)
                feats = forward_savlg_backbone(backbone, images, args)
            global_outputs, spatial_maps = forward_savlg_concept_layer(concept_layer, feats)
            _, _, final_logits = compute_savlg_concept_logits(
                global_outputs,
                spatial_maps,
                args,
            )
            concept_features.append(final_logits.cpu())
            concept_labels.append(labels)
    return torch.cat(concept_features, dim=0), torch.cat(concept_labels, dim=0)


def _extract_savlg_concept_components(args, backbone, concept_layer, loader):
    backbone.eval()
    concept_layer.eval()
    global_features = []
    spatial_features = []
    concept_labels = []
    print(
        f"[SAVLG NEC] start component extraction: batches={len(loader)} batch_size={getattr(loader, 'batch_size', 'na')}",
        flush=True,
    )
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(loader)):
            if batch_idx == 0:
                if isinstance(images, dict):
                    shape_msg = {k: tuple(v.shape) for k, v in images.items()}
                elif isinstance(images, torch.Tensor):
                    shape_msg = tuple(images.shape)
                else:
                    shape_msg = type(images).__name__
                print(
                    f"[SAVLG NEC] first batch fetched: labels={tuple(labels.shape)} images={shape_msg}",
                    flush=True,
                )
            if isinstance(images, dict):
                feats = {
                    key: value.to(args.device, non_blocking=True)
                    for key, value in images.items()
                }
            elif isinstance(images, torch.Tensor) and images.ndim == 4 and int(images.shape[1]) != 3:
                feats = images.to(args.device, non_blocking=True)
            else:
                images = images.to(args.device)
                feats = forward_savlg_backbone(backbone, images, args)
            global_outputs, spatial_maps = forward_savlg_concept_layer(concept_layer, feats)
            global_logits, spatial_logits, _ = compute_savlg_concept_logits(
                global_outputs,
                spatial_maps,
                args,
            )
            if batch_idx == 0:
                print(
                    f"[SAVLG NEC] first forward done: global={tuple(global_logits.shape)} spatial={tuple(spatial_logits.shape)}",
                    flush=True,
                )
            global_features.append(global_logits.cpu())
            spatial_features.append(spatial_logits.cpu())
            concept_labels.append(labels)
    return (
        torch.cat(global_features, dim=0),
        torch.cat(spatial_features, dim=0),
        torch.cat(concept_labels, dim=0),
    )


def _savlg_nec_component_cache_path(load_dir: str, split_name: str) -> str:
    return os.path.join(load_dir, f"savlg_nec_{split_name}_components.pt")


def _get_or_create_savlg_nec_components(
    load_dir,
    split_name: str,
    args,
    backbone,
    concept_layer,
    dataset,
):
    # For NEC we only need concept activations, not raw backbone features.
    # Materializing whole-split backbone tensors causes heavy memory/page-cache
    # pressure on the shared pod filesystem. Extract concepts directly once.
    print(
        f"[SAVLG NEC] preparing split={split_name} len={len(dataset)} cbl_batch_size={args.cbl_batch_size} num_workers={args.num_workers}",
        flush=True,
    )
    loader_kwargs = {
        "batch_size": args.cbl_batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": True,
    }
    if int(getattr(args, "num_workers", 0)) > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4
    loader = DataLoader(dataset, **loader_kwargs)

    global_concepts, spatial_concepts, labels = _extract_savlg_concept_components(
        args, backbone, concept_layer, loader
    )
    payload = {
        "global": global_concepts,
        "spatial": spatial_concepts,
        "labels": labels,
    }
    return payload


def _compose_savlg_final_concepts(component_cache, alpha: float):
    return component_cache["global"] + float(alpha) * component_cache["spatial"]


def _zscore_from_train(train_x, val_x, test_x):
    mean = train_x.mean(dim=0)
    std = train_x.std(dim=0, unbiased=False).clamp_min(1e-6)
    return (
        (train_x - mean) / std,
        (val_x - mean) / std,
        (test_x - mean) / std,
        mean,
        std,
    )


def _compose_savlg_final_concepts_with_branch_norm(
    train_components,
    val_components,
    test_components,
    alpha: float,
    branch_norm_mode: str,
):
    mode = str(branch_norm_mode).lower()
    if mode in {"none", "off", ""}:
        return (
            _compose_savlg_final_concepts(train_components, alpha),
            _compose_savlg_final_concepts(val_components, alpha),
            _compose_savlg_final_concepts(test_components, alpha),
        )
    if mode not in {"train_zscore", "zscore_train"}:
        raise ValueError(f"Unsupported SAVLG branch normalization mode: {branch_norm_mode}")

    train_global, val_global, test_global, _, _ = _zscore_from_train(
        train_components["global"],
        val_components["global"],
        test_components["global"],
    )
    train_spatial, val_spatial, test_spatial, _, _ = _zscore_from_train(
        train_components["spatial"],
        val_components["spatial"],
        test_components["spatial"],
    )
    alpha = float(alpha)
    return (
        train_global + alpha * train_spatial,
        val_global + alpha * val_spatial,
        test_global + alpha * test_spatial,
    )


def _subset_component_cache(component_cache, max_images: int | None):
    if max_images is None:
        return component_cache
    keep = min(int(max_images), int(component_cache["labels"].shape[0]))
    return {
        "global": component_cache["global"][:keep],
        "spatial": component_cache["spatial"][:keep],
        "labels": component_cache["labels"][:keep],
    }


def _normalize_savlg_final_concepts(load_dir, train_concepts, val_concepts, test_concepts):
    train_mean = torch.load(
        os.path.join(load_dir, "proj_mean.pt"), map_location="cpu"
    )
    train_std = torch.load(
        os.path.join(load_dir, "proj_std.pt"), map_location="cpu"
    )
    train_concepts = (train_concepts - train_mean) / train_std
    val_concepts = (val_concepts - train_mean) / train_std
    test_concepts = (test_concepts - train_mean) / train_std
    return train_concepts, val_concepts, test_concepts


def sparsity_acc_test_salf_cbm(load_dir, lam_max=0.1, n_iters=None, max_glm_steps=MAX_GLM_STEP):
    with open(os.path.join(load_dir, "args.txt"), "r") as f:
        args = json.load(f)
        args = argparse.Namespace(**args)
    with open(os.path.join(load_dir, "concepts.txt"), "r") as f:
        concepts = f.read().split("\n")
    classes = data_utils.get_classes(args.dataset)

    backbone = SpatialBackbone(args.backbone, device=args.device)
    concept_layer = build_savlg_concept_layer(args, backbone, len(concepts))
    concept_layer.load_state_dict(
        torch.load(os.path.join(load_dir, "concept_layer.pt"), map_location=args.device)
    )

    if use_original_label_free_protocol(args):
        train_dataset = data_utils.get_data(
            f"{args.dataset}_train", preprocess=backbone.preprocess
        )
        test_dataset = data_utils.get_data(
            f"{args.dataset}_val", preprocess=backbone.preprocess
        )
    else:
        base_train = data_utils.get_data(f"{args.dataset}_train", preprocess=None)
        total = len(base_train)
        n_val = int(args.val_split * total)
        n_train = total - n_val
        generator = torch.Generator().manual_seed(args.seed)
        train_subset, _ = torch.utils.data.random_split(
            list(range(total)),
            [n_train, n_val],
            generator=generator,
        )
        train_dataset = TransformedSubset(
            base_train, train_subset.indices, backbone.preprocess
        )
        base_test = data_utils.get_data(f"{args.dataset}_val", preprocess=None)
        test_dataset = TransformedSubset(
            base_test, list(range(len(base_test))), backbone.preprocess
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.cbl_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.cbl_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    train_concepts, train_labels = _extract_salf_concepts(
        backbone, concept_layer, train_loader, args.device
    )
    test_concepts, test_labels = _extract_salf_concepts(
        backbone, concept_layer, test_loader, args.device
    )

    train_mean = torch.load(
        os.path.join(load_dir, "proj_mean.pt"), map_location="cpu"
    )
    train_std = torch.load(
        os.path.join(load_dir, "proj_std.pt"), map_location="cpu"
    )
    train_concepts = (train_concepts - train_mean) / train_std
    test_concepts = (test_concepts - train_mean) / train_std

    train_concept_loader = DataLoader(
        IndexedTensorDataset(train_concepts, train_labels),
        batch_size=args.saga_batch_size,
        shuffle=False,
    )
    test_concept_loader = DataLoader(
        TensorDataset(test_concepts, test_labels),
        batch_size=args.saga_batch_size,
        shuffle=False,
    )

    path, truncated_weights, accs = measure_acc(
        len(concepts),
        len(classes),
        len(train_concept_loader.dataset),
        train_concept_loader,
        None,
        test_concept_loader,
        saga_step_size=args.saga_step_size,
        saga_n_iters=n_iters if n_iters is not None else args.saga_n_iters,
        device=args.device,
        max_lam=lam_max,
        max_glm_steps=max_glm_steps,
    )
    sparsity_list = [
        (params["weight"].abs() > 1e-5).float().mean().item() for params in path
    ]
    NEC = [len(concepts) * sparsity for sparsity in sparsity_list]
    acc = [params["metrics"].get("acc_test", float("nan")) for params in path]
    df = pd.DataFrame(data={"NEC": NEC, "Accuracy": acc})
    df.to_csv(os.path.join(load_dir, "metrics.csv"))
    for NEC in truncated_weights:
        W, b = truncated_weights[NEC]
        torch.save(W, os.path.join(load_dir, f"W_g@NEC={NEC:d}.pt"))
        torch.save(b, os.path.join(load_dir, f"b_g@NEC={NEC:d}.pt"))
    return accs


def sparsity_acc_test_savlg_cbm(
    load_dir,
    lam_max=0.1,
    n_iters=None,
    max_glm_steps=MAX_GLM_STEP,
    cbl_batch_size=None,
    saga_batch_size=None,
    num_workers=None,
    alpha_override=None,
    disable_activation_cache_override=False,
    max_images=None,
    branch_norm_mode="none",
):
    print(f"[SAVLG NEC] loading run from {load_dir}", flush=True)
    with open(os.path.join(load_dir, "args.txt"), "r") as f:
        args = json.load(f)
        args = argparse.Namespace(**args)
    if cbl_batch_size is not None:
        args.cbl_batch_size = cbl_batch_size
    if saga_batch_size is not None:
        args.saga_batch_size = saga_batch_size
    if num_workers is not None:
        args.num_workers = num_workers
    if disable_activation_cache_override:
        args.use_activation_cache = False
        args.disable_activation_cache = True
    if not hasattr(args, "use_activation_cache"):
        args.use_activation_cache = not bool(getattr(args, "disable_activation_cache", False))
    with open(os.path.join(load_dir, "concepts.txt"), "r") as f:
        concepts = f.read().split("\n")
    classes = data_utils.get_classes(args.dataset)

    print("[SAVLG NEC] creating SAVLG splits", flush=True)
    _, _, train_dataset, val_dataset, test_dataset, backbone = create_savlg_splits(args)
    print(
        f"[SAVLG NEC] splits ready: train={len(train_dataset)} val={len(val_dataset)} test={len(test_dataset)}",
        flush=True,
    )
    print("[SAVLG NEC] building concept layer", flush=True)
    concept_layer = build_savlg_concept_layer(args, backbone, len(concepts))
    concept_layer.load_state_dict(
        torch.load(os.path.join(load_dir, "concept_layer.pt"), map_location=args.device)
    )
    print("[SAVLG NEC] concept layer loaded", flush=True)

    train_components = _get_or_create_savlg_nec_components(
        load_dir, "train", args, backbone, concept_layer, train_dataset
    )
    val_components = _get_or_create_savlg_nec_components(
        load_dir, "val", args, backbone, concept_layer, val_dataset
    )
    test_components = _get_or_create_savlg_nec_components(
        load_dir, "test", args, backbone, concept_layer, test_dataset
    )

    train_components = _subset_component_cache(train_components, max_images)
    val_components = _subset_component_cache(val_components, max_images)
    test_components = _subset_component_cache(test_components, max_images)

    alpha = (
        float(alpha_override)
        if alpha_override is not None
        else float(getattr(args, "savlg_residual_spatial_alpha", 0.0))
    )
    train_concepts, val_concepts, test_concepts = _compose_savlg_final_concepts_with_branch_norm(
        train_components,
        val_components,
        test_components,
        alpha,
        branch_norm_mode=branch_norm_mode,
    )
    train_labels = train_components["labels"]
    val_labels = val_components["labels"]
    test_labels = test_components["labels"]

    if str(branch_norm_mode).lower() in {"none", "off", ""}:
        train_concepts, val_concepts, test_concepts = _normalize_savlg_final_concepts(
            load_dir, train_concepts, val_concepts, test_concepts
        )
    else:
        train_concepts, val_concepts, test_concepts, _, _ = _zscore_from_train(
            train_concepts,
            val_concepts,
            test_concepts,
        )

    train_concept_loader = DataLoader(
        IndexedTensorDataset(train_concepts, train_labels),
        batch_size=args.saga_batch_size,
        shuffle=True,
    )
    val_concept_loader = DataLoader(
        TensorDataset(val_concepts, val_labels),
        batch_size=args.saga_batch_size,
        shuffle=False,
    )
    test_concept_loader = DataLoader(
        TensorDataset(test_concepts, test_labels),
        batch_size=args.saga_batch_size,
        shuffle=False,
    )

    path, truncated_weights, accs = measure_acc(
        len(concepts),
        len(classes),
        len(train_concept_loader.dataset),
        train_concept_loader,
        val_concept_loader,
        test_concept_loader,
        saga_step_size=args.saga_step_size,
        saga_n_iters=n_iters if n_iters is not None else args.saga_n_iters,
        device=args.device,
        max_lam=lam_max,
        max_glm_steps=max_glm_steps,
    )
    sparsity_list = [
        (params["weight"].abs() > 1e-5).float().mean().item() for params in path
    ]
    NEC = [len(concepts) * sparsity for sparsity in sparsity_list]
    acc = [params["metrics"].get("acc_test", float("nan")) for params in path]
    df = pd.DataFrame(data={"NEC": NEC, "Accuracy": acc})
    df.to_csv(os.path.join(load_dir, "metrics.csv"))
    for NEC in truncated_weights:
        W, b = truncated_weights[NEC]
        torch.save(W, os.path.join(load_dir, f"W_g@NEC={NEC:d}.pt"))
        torch.save(b, os.path.join(load_dir, f"b_g@NEC={NEC:d}.pt"))
    return accs
