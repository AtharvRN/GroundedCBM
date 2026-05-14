import argparse
import json
import os
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import model.utils as utils
import data.utils as data_utils
from data.concept_dataset import get_concept_dataloader, get_final_layer_dataset
from gcbm.features import make_feature_loader, standardize_from_train
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


@dataclass
class NECFeatureSet:
    """Concept features used by sparse GLM/NEC training and evaluation."""

    concepts: list[str]
    classes: list[str]
    train_features: torch.Tensor
    train_labels: torch.Tensor
    test_features: torch.Tensor
    test_labels: torch.Tensor
    val_features: Optional[torch.Tensor] = None
    val_labels: Optional[torch.Tensor] = None


def _load_run_args(load_dir: str) -> argparse.Namespace:
    with open(os.path.join(load_dir, "args.txt"), "r") as f:
        return argparse.Namespace(**json.load(f))


def _load_run_concepts(load_dir: str) -> list[str]:
    with open(os.path.join(load_dir, "concepts.txt"), "r") as f:
        return f.read().split("\n")


def _apply_common_overrides(
    args: argparse.Namespace,
    *,
    cbl_batch_size=None,
    saga_batch_size=None,
    num_workers=None,
) -> argparse.Namespace:
    if cbl_batch_size is not None:
        args.cbl_batch_size = cbl_batch_size
    if saga_batch_size is not None:
        args.saga_batch_size = saga_batch_size
    if num_workers is not None:
        args.num_workers = num_workers
    return args


def _indexed_dataset_tensors(dataset) -> tuple[torch.Tensor, torch.Tensor]:
    if not hasattr(dataset, "features") or not hasattr(dataset, "targets"):
        raise TypeError(f"Expected IndexedTensorDataset, got {type(dataset).__name__}")
    return dataset.features, dataset.targets


def _tensor_dataset_tensors(dataset) -> tuple[torch.Tensor, torch.Tensor]:
    if not hasattr(dataset, "tensors") or len(dataset.tensors) < 2:
        raise TypeError(f"Expected TensorDataset, got {type(dataset).__name__}")
    return dataset.tensors[0], dataset.tensors[1]


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

    feasible_measure_level = tuple(
        int(level) for level in measure_level if 0 < int(level) <= int(num_concepts)
    )
    if not feasible_measure_level:
        feasible_measure_level = (int(num_concepts),)

    ALPHA = 0.99
    metadata = {}
    metadata["max_reg"] = {}
    metadata["max_reg"]["nongrouped"] = max_lam
    # Solve the GLM path
    max_sparsity = feasible_measure_level[-1] / num_concepts
    output_proj = glm_saga(linear, train_loader, saga_step_size, saga_n_iters, ALPHA, k=max_glm_steps, epsilon=1 / (GLM_STEP_SIZE ** max_glm_steps),
                    val_loader=val_loader, test_loader=test_concept_loader, do_zero=False, metadata=metadata, n_ex=num_samples, n_classes=num_classes,
                    max_sparsity=max_sparsity, eval_train=False, eval_val=False, eval_test=False)
    path = output_proj['path']
    sparsity_list = [(params['weight'].abs() > 1e-5).float().mean().item() for params in path]

    # Measure accuracy on test set
    final_layer = torch.nn.Linear(num_concepts, num_classes)
    accs = []
    weights = []
    for eff_concept_num in feasible_measure_level:
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
    return path, {NEC: weight for NEC, weight in zip(feasible_measure_level, weights)}, accs


def _feature_loader(features, labels, batch_size, *, indexed: bool, shuffle: bool):
    return make_feature_loader(features, labels, batch_size, indexed=indexed, shuffle=shuffle)


def _save_nec_outputs(load_dir: str, concepts: list[str], path, truncated_weights) -> None:
    sparsity_list = [
        (params["weight"].abs() > 1e-5).float().mean().item() for params in path
    ]
    nec_values = [len(concepts) * sparsity for sparsity in sparsity_list]
    acc_values = [params["metrics"].get("acc_test", float("nan")) for params in path]
    pd.DataFrame(data={"NEC": nec_values, "Accuracy": acc_values}).to_csv(
        os.path.join(load_dir, "metrics.csv")
    )
    for nec, (weights, bias) in truncated_weights.items():
        torch.save(weights, os.path.join(load_dir, f"W_g@NEC={nec:d}.pt"))
        torch.save(bias, os.path.join(load_dir, f"b_g@NEC={nec:d}.pt"))


def run_nec_sweep_from_features(
    load_dir: str,
    features: NECFeatureSet,
    *,
    saga_batch_size: int,
    saga_step_size: float,
    saga_n_iters: int,
    device: str,
    lam_max: float = 0.1,
    max_glm_steps: int = MAX_GLM_STEP,
) -> list[float]:
    """Train sparse final layers from already-extracted concept features."""
    train_loader = _feature_loader(
        features.train_features,
        features.train_labels,
        saga_batch_size,
        indexed=True,
        shuffle=True,
    )
    val_loader = None
    if features.val_features is not None and features.val_labels is not None:
        val_loader = _feature_loader(
            features.val_features,
            features.val_labels,
            saga_batch_size,
            indexed=False,
            shuffle=False,
        )
    test_loader = _feature_loader(
        features.test_features,
        features.test_labels,
        saga_batch_size,
        indexed=False,
        shuffle=False,
    )
    path, truncated_weights, accs = measure_acc(
        len(features.concepts),
        len(features.classes),
        len(train_loader.dataset),
        train_loader,
        val_loader,
        test_loader,
        saga_step_size=saga_step_size,
        saga_n_iters=saga_n_iters,
        device=device,
        max_lam=lam_max,
        max_glm_steps=max_glm_steps,
    )
    _save_nec_outputs(load_dir, features.concepts, path, truncated_weights)
    return accs


def extract_vlg_nec_features(
    load_dir,
    bot_filter=0,
    anno=None,
    cbl_batch_size=None,
    saga_batch_size=None,
    num_workers=None,
    max_images=None,
) -> tuple[NECFeatureSet, argparse.Namespace]:
    args = _load_run_args(load_dir)
    if anno is not None:
        args.annotation_dir = anno
    _apply_common_overrides(
        args,
        cbl_batch_size=cbl_batch_size,
        saga_batch_size=saga_batch_size,
        num_workers=num_workers,
    )
    concepts = _load_run_concepts(load_dir)
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
        max_images=max_images or 0,
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
        max_images=max_images or 0,
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
        max_images=max_images or 0,
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
    train_features, train_labels = _indexed_dataset_tensors(train_concept_loader.dataset)
    val_features, val_labels = _tensor_dataset_tensors(val_concept_loader.dataset)
    feature_set = NECFeatureSet(
        concepts=concepts,
        classes=classes,
        train_features=train_features,
        train_labels=train_labels,
        val_features=val_features,
        val_labels=val_labels,
        test_features=test_concept_features,
        test_labels=concept_labels,
    )
    return feature_set, args


def sparsity_acc_test(
    load_dir,
    lam_max=0.1,
    bot_filter=0,
    anno=None,
    n_iters=None,
    max_glm_steps=MAX_GLM_STEP,
    cbl_batch_size=None,
    saga_batch_size=None,
    num_workers=None,
    max_images=None,
):
    feature_set, args = extract_vlg_nec_features(
        load_dir,
        bot_filter=bot_filter,
        anno=anno,
        cbl_batch_size=cbl_batch_size,
        saga_batch_size=saga_batch_size,
        num_workers=num_workers,
        max_images=max_images,
    )
    return run_nec_sweep_from_features(
        load_dir,
        feature_set,
        saga_batch_size=args.saga_batch_size,
        saga_step_size=args.saga_step_size,
        saga_n_iters=n_iters if n_iters is not None else args.saga_n_iters,
        device=args.device,
        lam_max=lam_max,
        max_glm_steps=max_glm_steps,
    )


def extract_lf_nec_features(
    load_dir,
    cbl_batch_size=None,
    saga_batch_size=None,
    num_workers=None,
    max_images=None,
) -> tuple[NECFeatureSet, argparse.Namespace]:
    args = _load_run_args(load_dir)
    if not hasattr(args, "batch_size"):
        args.batch_size = getattr(args, "lf_batch_size", getattr(args, "cbl_batch_size", 64))
    if not hasattr(args, "n_iters"):
        args.n_iters = getattr(args, "saga_n_iters", 500)
    if not hasattr(args, "saga_batch_size"):
        args.saga_batch_size = getattr(args, "batch_size", 256)
    if cbl_batch_size is not None:
        args.batch_size = cbl_batch_size
    if saga_batch_size is not None:
        args.saga_batch_size = saga_batch_size
    if num_workers is not None:
        args.num_workers = num_workers
    concepts = _load_run_concepts(load_dir)
    classes = data_utils.get_classes(args.dataset)
    # Concept filtering

    cbm = load_cbm(load_dir, args.device)
    cbm.eval()
    train_dataset = data_utils.get_data(args.dataset + "_train", preprocess=cbm.preprocess)
    test_dataset = data_utils.get_data(args.dataset + "_val", preprocess=cbm.preprocess)
    if max_images is not None:
        keep_train = min(int(max_images), len(train_dataset))
        keep_test = min(int(max_images), len(test_dataset))
        train_dataset = torch.utils.data.Subset(train_dataset, list(range(keep_train)))
        test_dataset = torch.utils.data.Subset(test_dataset, list(range(keep_test)))
    # Calculating test features
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, num_workers=args.num_workers, batch_size=args.batch_size
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, shuffle=False, num_workers=args.num_workers, batch_size=args.batch_size
    )
    with torch.no_grad():
        final_features = []
        for loader in [train_dataloader, test_dataloader]:
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
            final_features.append((concept_features, concept_labels))
    (train_features, train_labels), (test_features, test_labels) = final_features
    feature_set = NECFeatureSet(
        concepts=concepts,
        classes=classes,
        train_features=train_features,
        train_labels=train_labels,
        test_features=test_features,
        test_labels=test_labels,
    )
    return feature_set, args


def sparsity_acc_test_lf_cbm(
    load_dir,
    lam_max=0.1,
    n_iters=None,
    max_glm_steps=MAX_GLM_STEP,
    cbl_batch_size=None,
    saga_batch_size=None,
    num_workers=None,
    max_images=None,
):
    feature_set, args = extract_lf_nec_features(
        load_dir,
        cbl_batch_size=cbl_batch_size,
        saga_batch_size=saga_batch_size,
        num_workers=num_workers,
        max_images=max_images,
    )
    return run_nec_sweep_from_features(
        load_dir,
        feature_set,
        saga_batch_size=args.saga_batch_size,
        saga_step_size=getattr(args, "saga_step_size", 0.1),
        saga_n_iters=n_iters if n_iters is not None else args.n_iters,
        device=args.device,
        lam_max=lam_max,
        max_glm_steps=max_glm_steps,
    )


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
    train_z, val_z, test_z, mean, std = standardize_from_train(train_x, val_x, test_x, unbiased=False)
    return train_z, val_z, test_z, mean.squeeze(0), std.squeeze(0)


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


def extract_salf_nec_features(
    load_dir,
    cbl_batch_size=None,
    saga_batch_size=None,
    num_workers=None,
    max_images=None,
) -> tuple[NECFeatureSet, argparse.Namespace]:
    args = _load_run_args(load_dir)
    concepts = _load_run_concepts(load_dir)
    _apply_common_overrides(
        args,
        cbl_batch_size=cbl_batch_size,
        saga_batch_size=saga_batch_size,
        num_workers=num_workers,
    )
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
        if max_images is not None:
            keep_train = min(int(max_images), len(train_dataset))
            keep_test = min(int(max_images), len(test_dataset))
            train_dataset = torch.utils.data.Subset(train_dataset, list(range(keep_train)))
            test_dataset = torch.utils.data.Subset(test_dataset, list(range(keep_test)))
    else:
        base_train = data_utils.get_data(f"{args.dataset}_train", preprocess=None)
        total = len(base_train)
        if max_images is not None:
            total = min(total, int(max_images))
        n_val = int(args.val_split * total)
        if args.val_split > 0 and n_val == 0 and total > 1:
            n_val = 1
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
        test_total = len(base_test)
        if max_images is not None:
            test_total = min(test_total, int(max_images))
        test_dataset = TransformedSubset(
            base_test, list(range(test_total)), backbone.preprocess
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

    feature_set = NECFeatureSet(
        concepts=concepts,
        classes=classes,
        train_features=train_concepts,
        train_labels=train_labels,
        test_features=test_concepts,
        test_labels=test_labels,
    )
    return feature_set, args


def sparsity_acc_test_salf_cbm(
    load_dir,
    lam_max=0.1,
    n_iters=None,
    max_glm_steps=MAX_GLM_STEP,
    cbl_batch_size=None,
    saga_batch_size=None,
    num_workers=None,
    max_images=None,
):
    feature_set, args = extract_salf_nec_features(
        load_dir,
        cbl_batch_size=cbl_batch_size,
        saga_batch_size=saga_batch_size,
        num_workers=num_workers,
        max_images=max_images,
    )
    return run_nec_sweep_from_features(
        load_dir,
        feature_set,
        saga_batch_size=args.saga_batch_size,
        saga_step_size=args.saga_step_size,
        saga_n_iters=n_iters if n_iters is not None else args.saga_n_iters,
        device=args.device,
        lam_max=lam_max,
        max_glm_steps=max_glm_steps,
    )


def extract_savlg_nec_features(
    load_dir,
    cbl_batch_size=None,
    saga_batch_size=None,
    num_workers=None,
    alpha_override=None,
    disable_activation_cache_override=False,
    max_images=None,
    branch_norm_mode="none",
) -> tuple[NECFeatureSet, argparse.Namespace]:
    print(f"[SAVLG NEC] loading run from {load_dir}", flush=True)
    args = _load_run_args(load_dir)
    _apply_common_overrides(
        args,
        cbl_batch_size=cbl_batch_size,
        saga_batch_size=saga_batch_size,
        num_workers=num_workers,
    )
    if disable_activation_cache_override:
        args.use_activation_cache = False
        args.disable_activation_cache = True
    if not hasattr(args, "use_activation_cache"):
        args.use_activation_cache = not bool(getattr(args, "disable_activation_cache", False))
    if getattr(args, "skip_test_eval", False):
        print(
            "[SAVLG NEC] overriding saved skip_test_eval=True to force evaluation on dataset_val",
            flush=True,
        )
        args.skip_test_eval = False
    concepts = _load_run_concepts(load_dir)
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

    feature_set = NECFeatureSet(
        concepts=concepts,
        classes=classes,
        train_features=train_concepts,
        train_labels=train_labels,
        val_features=val_concepts,
        val_labels=val_labels,
        test_features=test_concepts,
        test_labels=test_labels,
    )
    return feature_set, args


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
    feature_set, args = extract_savlg_nec_features(
        load_dir,
        cbl_batch_size=cbl_batch_size,
        saga_batch_size=saga_batch_size,
        num_workers=num_workers,
        alpha_override=alpha_override,
        disable_activation_cache_override=disable_activation_cache_override,
        max_images=max_images,
        branch_norm_mode=branch_norm_mode,
    )
    return run_nec_sweep_from_features(
        load_dir,
        feature_set,
        saga_batch_size=args.saga_batch_size,
        saga_step_size=args.saga_step_size,
        saga_n_iters=n_iters if n_iters is not None else args.saga_n_iters,
        device=args.device,
        lam_max=lam_max,
        max_glm_steps=max_glm_steps,
    )


def build_nec_feature_set(
    load_dir: str,
    model_name: str,
    *,
    annotation_dir=None,
    bot_filter=0,
    cbl_batch_size=None,
    saga_batch_size=None,
    num_workers=None,
    savlg_alpha_override=None,
    disable_activation_cache=False,
    max_images=None,
    savlg_branch_norm_mode="none",
) -> tuple[NECFeatureSet, argparse.Namespace]:
    """Extract reusable concept tensors for sparse training or later evaluation."""
    normalized = str(model_name).lower().replace("-", "_")
    if normalized == "vlg_cbm":
        return extract_vlg_nec_features(
            load_dir,
            bot_filter=bot_filter,
            anno=annotation_dir,
            cbl_batch_size=cbl_batch_size,
            saga_batch_size=saga_batch_size,
            num_workers=num_workers,
            max_images=max_images,
        )
    if normalized == "lf_cbm":
        return extract_lf_nec_features(
            load_dir,
            cbl_batch_size=cbl_batch_size,
            saga_batch_size=saga_batch_size,
            num_workers=num_workers,
            max_images=max_images,
        )
    if normalized == "salf_cbm":
        return extract_salf_nec_features(
            load_dir,
            cbl_batch_size=cbl_batch_size,
            saga_batch_size=saga_batch_size,
            num_workers=num_workers,
            max_images=max_images,
        )
    if normalized in {"savlg_cbm", "sg_cbm", "sgcbm"}:
        return extract_savlg_nec_features(
            load_dir,
            cbl_batch_size=cbl_batch_size,
            saga_batch_size=saga_batch_size,
            num_workers=num_workers,
            alpha_override=savlg_alpha_override,
            disable_activation_cache_override=disable_activation_cache,
            max_images=max_images,
            branch_norm_mode=savlg_branch_norm_mode,
        )
    raise NotImplementedError(
        f"Sparse evaluation for model_name={model_name} is not implemented yet."
    )


def train_sparse_nec_from_checkpoint(
    load_dir: str,
    model_name: str,
    *,
    lam_max=0.1,
    bot_filter=0,
    annotation_dir=None,
    n_iters=None,
    max_glm_steps=MAX_GLM_STEP,
    cbl_batch_size=None,
    saga_batch_size=None,
    num_workers=None,
    savlg_alpha_override=None,
    disable_activation_cache=False,
    max_images=None,
    savlg_branch_norm_mode="none",
) -> tuple[list[float], NECFeatureSet, argparse.Namespace]:
    """Extract concept features once, then run the shared sparse NEC sweep."""
    feature_set, args = build_nec_feature_set(
        load_dir,
        model_name,
        annotation_dir=annotation_dir,
        bot_filter=bot_filter,
        cbl_batch_size=cbl_batch_size,
        saga_batch_size=saga_batch_size,
        num_workers=num_workers,
        savlg_alpha_override=savlg_alpha_override,
        disable_activation_cache=disable_activation_cache,
        max_images=max_images,
        savlg_branch_norm_mode=savlg_branch_norm_mode,
    )
    accs = run_nec_sweep_from_features(
        load_dir,
        feature_set,
        saga_batch_size=args.saga_batch_size,
        saga_step_size=getattr(args, "saga_step_size", 0.1),
        saga_n_iters=n_iters if n_iters is not None else getattr(args, "saga_n_iters", 500),
        device=args.device,
        lam_max=lam_max,
        max_glm_steps=max_glm_steps,
    )
    return accs, feature_set, args
