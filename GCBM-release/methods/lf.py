import json
import os
import random
from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm

import clip
from data import utils as data_utils
from glm_saga.elasticnet import IndexedTensorDataset
from methods.common import build_run_dir, save_args, write_artifacts
from model.cbm import Backbone, BackboneCLIP, train_dense_final, train_sparse_final
from model.sam import SAM


def _dataset_targets_view(base_dataset: Dataset):
    targets = getattr(base_dataset, "targets", None)
    if targets is None:
        return None
    if isinstance(targets, torch.Tensor):
        return targets.detach().cpu()
    return torch.as_tensor(list(targets), dtype=torch.long)


def _fast_subset_targets(base_dataset: Dataset, indices: Iterable[int]) -> torch.Tensor:
    idx_list = list(indices)
    targets = _dataset_targets_view(base_dataset)
    if targets is not None:
        return targets[idx_list].to(dtype=torch.long)
    return torch.tensor([base_dataset[idx][1] for idx in idx_list], dtype=torch.long)


class TransformedSubset(Dataset):
    def __init__(self, base_dataset: Dataset, indices: Iterable[int], transform):
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.transform = transform
        self.targets = _fast_subset_targets(base_dataset, self.indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        image, target = self.base_dataset[self.indices[idx]]
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def use_original_label_free_protocol(args) -> bool:
    return bool(getattr(args, "lf_original_protocol", False))


def subset_targets(base_dataset: Dataset, indices: Iterable[int]) -> torch.Tensor:
    return _fast_subset_targets(base_dataset, indices)


def get_lf_concepts(args) -> list[str]:
    return data_utils.get_concepts(args.concept_set, getattr(args, "filter_set", None))


class DualTransformSubset(Dataset):
    def __init__(self, base_dataset: Dataset, indices: Iterable[int], transform_a, transform_b):
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.transform_a = transform_a
        self.transform_b = transform_b

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        image, target = self.base_dataset[self.indices[idx]]
        image_a = self.transform_a(image) if self.transform_a is not None else image
        image_b = self.transform_b(image) if self.transform_b is not None else image
        return image_a, image_b, target


class LFConceptLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_concepts: int,
        num_hidden: int = 1,
        use_batchnorm: bool = False,
    ):
        super().__init__()
        hidden_dim = max(1, input_dim // 2)
        layers = []
        in_dim = input_dim
        for _ in range(max(0, int(num_hidden))):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, n_concepts))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.layers(x)


@dataclass
class ProjectionArtifacts:
    layer: nn.Module
    linear_weight: torch.Tensor | None


def cos_similarity_cubed(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a_norm = F.normalize(a, dim=0)
    b_norm = F.normalize(b, dim=0)
    return ((a_norm * b_norm).sum(dim=0)) ** 3


def make_projection_layer(args, input_dim: int, n_concepts: int) -> nn.Module:
    if args.cbl_type == "linear":
        return nn.Linear(input_dim, n_concepts, bias=False).to(args.device)
    if args.cbl_type == "mlp":
        return LFConceptLayer(
            input_dim,
            n_concepts,
            num_hidden=args.cbl_hidden_layers,
            use_batchnorm=args.cbl_use_batchnorm,
        ).to(args.device)
    raise ValueError(f"Unsupported cbl_type: {args.cbl_type}")


def prune_projection_outputs(args, proj_layer: nn.Module, keep_mask: torch.Tensor, input_dim: int) -> ProjectionArtifacts:
    keep_idx = torch.nonzero(keep_mask.cpu(), as_tuple=False).flatten()
    n_keep = int(keep_idx.numel())
    if n_keep == 0:
        raise ValueError("No concepts remain after interpretability filtering.")

    if args.cbl_type == "linear":
        weight = proj_layer.weight.detach().cpu()[keep_idx].clone()
        pruned = nn.Linear(input_dim, n_keep, bias=False)
        pruned.weight.data.copy_(weight)
        return ProjectionArtifacts(pruned.to(args.device), weight)

    pruned = LFConceptLayer(
        input_dim,
        n_keep,
        num_hidden=args.cbl_hidden_layers,
        use_batchnorm=args.cbl_use_batchnorm,
    )
    old_sd = {k: v.detach().cpu().clone() for k, v in proj_layer.state_dict().items()}
    new_sd = pruned.state_dict()
    block_size = 3 if args.cbl_use_batchnorm else 2
    final_linear_idx = block_size * max(0, int(args.cbl_hidden_layers))
    w_key = f"layers.{final_linear_idx}.weight"
    b_key = f"layers.{final_linear_idx}.bias"
    for key in new_sd.keys():
        if key == w_key:
            new_sd[key] = old_sd[key][keep_idx].clone()
        elif key == b_key:
            new_sd[key] = old_sd[key][keep_idx].clone()
        else:
            new_sd[key] = old_sd[key].clone()
    pruned.load_state_dict(new_sd, strict=True)
    return ProjectionArtifacts(pruned.to(args.device), None)


def train_projection_layer(
    args,
    train_backbone_features: torch.Tensor,
    train_clip_features: torch.Tensor,
    val_backbone_features: torch.Tensor,
    val_clip_features: torch.Tensor,
    n_concepts: int,
) -> nn.Module:
    proj_layer = make_projection_layer(args, train_backbone_features.shape[1], n_concepts)
    if getattr(args, "cbl_use_sam", False):
        optimizer = SAM(
            proj_layer.parameters(),
            base_optimizer_cls=torch.optim.Adam,
            rho=float(getattr(args, "cbl_sam_rho", 0.05)),
            adaptive=bool(getattr(args, "cbl_sam_adaptive", False)),
            lr=args.proj_lr,
        )
    else:
        optimizer = torch.optim.Adam(proj_layer.parameters(), lr=args.proj_lr)
    indices = list(range(len(train_backbone_features)))
    best_val_loss = float("inf")
    best_state = {k: v.detach().cpu().clone() for k, v in proj_layer.state_dict().items()}
    best_step = 0
    no_improve_evals = 0
    batch_size = min(args.proj_batch_size, len(train_backbone_features))
    eval_every = max(1, int(args.proj_eval_every))

    pbar = tqdm(range(args.proj_steps), desc="LF projection", dynamic_ncols=True)
    for step in pbar:
        batch_idx = torch.LongTensor(random.sample(indices, batch_size))
        batch_backbone = train_backbone_features[batch_idx].to(args.device)
        batch_target = train_clip_features[batch_idx].to(args.device)

        def compute_loss():
            proj_out = proj_layer(batch_backbone)
            return -cos_similarity_cubed(proj_out.T, batch_target.T).mean()

        loss = compute_loss()

        optimizer.zero_grad()
        loss.backward()
        if getattr(args, "cbl_use_sam", False):
            optimizer.first_step(zero_grad=True)
            second_loss = compute_loss()
            second_loss.backward()
            optimizer.second_step(zero_grad=True)
        else:
            optimizer.step()

        if step % eval_every == 0 or step == args.proj_steps - 1:
            proj_layer.eval()
            with torch.no_grad():
                val_out = proj_layer(val_backbone_features.to(args.device))
                val_loss = -cos_similarity_cubed(
                    val_out.T,
                    val_clip_features.to(args.device).T,
                ).mean()
                val_loss_value = float(val_loss.item())
            proj_layer.train()

            if step == 0 or val_loss_value < (best_val_loss - args.proj_min_delta):
                best_val_loss = val_loss_value
                best_state = {
                    k: v.detach().cpu().clone() for k, v in proj_layer.state_dict().items()
                }
                best_step = step
                no_improve_evals = 0
            elif val_loss_value > (best_val_loss + args.proj_min_delta):
                no_improve_evals += 1
                if (
                    args.proj_early_stop_patience > 0
                    and step >= args.proj_min_steps_before_early_stop
                    and no_improve_evals > args.proj_early_stop_patience
                ):
                    break

            pbar.set_postfix(
                train_loss=float(loss.item()),
                val_loss=val_loss_value,
                best_val=best_val_loss,
            )

    pbar.close()
    proj_layer.load_state_dict(best_state, strict=True)
    logger.info(f"LF projection best step={best_step} best_val_similarity={-best_val_loss:.4f}")
    return proj_layer


def create_splits(args):
    base_train = data_utils.get_data(f"{args.dataset}_train", None)
    total = len(base_train)
    n_val = int(args.val_split * total)
    n_train = total - n_val
    generator = torch.Generator().manual_seed(args.seed)
    train_subset, val_subset = torch.utils.data.random_split(
        list(range(total)),
        [n_train, n_val],
        generator=generator,
    )

    if args.backbone.startswith("clip_"):
        backbone = BackboneCLIP(
            args.backbone,
            use_penultimate=args.use_clip_penultimate,
            device=args.device,
        )
    else:
        backbone = Backbone(args.backbone, args.feature_layer, args.device)

    clip_model, clip_preprocess = clip.load(
        (args.lf_clip_name or "clip_RN50").replace("clip_", ""),
        device=args.device,
    )
    clip_model = clip_model.float().eval()

    train_dataset = DualTransformSubset(
        base_train,
        train_subset.indices,
        backbone.preprocess,
        clip_preprocess,
    )
    val_dataset = DualTransformSubset(
        base_train,
        val_subset.indices,
        backbone.preprocess,
        clip_preprocess,
    )
    base_test = data_utils.get_data(f"{args.dataset}_val", None)
    test_dataset = TransformedSubset(
        base_test,
        list(range(len(base_test))),
        backbone.preprocess,
    )
    return train_dataset, val_dataset, test_dataset, backbone, clip_model


def compute_dual_features(args, dataset: Dataset, backbone, clip_model):
    loader = DataLoader(
        dataset,
        batch_size=args.lf_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    backbone_features = []
    clip_features = []
    labels = []
    with torch.no_grad():
        for backbone_images, clip_images, target in tqdm(loader, desc="LF features"):
            bb = backbone(backbone_images.to(args.device))
            clip_img = clip_model.encode_image(clip_images.to(args.device)).float()
            backbone_features.append(bb.detach().cpu())
            clip_features.append(clip_img.detach().cpu())
            labels.append(target)
    return (
        torch.cat(backbone_features, dim=0),
        torch.cat(clip_features, dim=0),
        torch.cat(labels, dim=0),
    )


def compute_concept_features(
    args,
    proj_layer: nn.Module,
    backbone_features: torch.Tensor,
) -> torch.Tensor:
    proj_layer.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, backbone_features.shape[0], args.proj_batch_size):
            batch = backbone_features[start : start + args.proj_batch_size].to(args.device)
            chunks.append(proj_layer(batch).cpu())
    return torch.cat(chunks, dim=0)


def evaluate_accuracy(backbone, proj_layer, mean, std, final_layer, dataset, args) -> float:
    loader = DataLoader(
        dataset,
        batch_size=args.lf_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    correct = 0
    total = 0
    with torch.no_grad():
        for images, target in tqdm(loader, desc="LF eval", leave=False):
            features = backbone(images.to(args.device))
            concepts = proj_layer(features)
            concepts = (concepts - mean.to(args.device)) / std.to(args.device)
            logits = final_layer(concepts)
            pred = logits.argmax(dim=-1).cpu()
            correct += int((pred == target).sum().item())
            total += int(target.numel())
    return correct / max(total, 1)


def train_lf_cbm(args):
    save_dir = build_run_dir(args.save_dir, args.dataset, args.model_name)
    logger.add(
        os.path.join(save_dir, "train.log"),
        format="{time} {level} {message}",
        level="DEBUG",
    )
    logger.info(f"Saving LF-CBM model to {save_dir}")
    save_args(args, save_dir)

    classes = data_utils.get_classes(args.dataset)
    raw_concepts = data_utils.get_concepts(args.concept_set, args.filter_set)
    train_dataset, val_dataset, test_dataset, backbone, clip_model = create_splits(args)

    tokens = clip.tokenize([str(concept) for concept in raw_concepts]).to(args.device)
    with torch.no_grad():
        clip_text_features = clip_model.encode_text(tokens).float().cpu()

    train_backbone, train_clip_img, train_labels = compute_dual_features(
        args, train_dataset, backbone, clip_model
    )
    val_backbone, val_clip_img, val_labels = compute_dual_features(
        args, val_dataset, backbone, clip_model
    )

    train_clip_img = F.normalize(train_clip_img, dim=1)
    val_clip_img = F.normalize(val_clip_img, dim=1)
    clip_text_features = F.normalize(clip_text_features, dim=1)

    train_clip_sim = train_clip_img @ clip_text_features.T
    val_clip_sim = val_clip_img @ clip_text_features.T

    highest = torch.mean(torch.topk(train_clip_sim, dim=0, k=min(5, train_clip_sim.shape[0]))[0], dim=0)
    keep_mask = highest > args.clip_cutoff
    concepts = [raw_concepts[i] for i in range(len(raw_concepts)) if keep_mask[i]]
    if not concepts:
        raise RuntimeError(
            "All concepts were removed by LF clip_cutoff. Lower --clip_cutoff."
        )
    clip_text_features = clip_text_features[keep_mask]
    train_clip_sim = train_clip_img @ clip_text_features.T
    val_clip_sim = val_clip_img @ clip_text_features.T

    proj_layer = train_projection_layer(
        args,
        train_backbone,
        train_clip_sim,
        val_backbone,
        val_clip_sim,
        len(concepts),
    )

    with torch.no_grad():
        val_proj = proj_layer(val_backbone.to(args.device))
        interpretable = (
            cos_similarity_cubed(val_proj, val_clip_sim.to(args.device))
            > args.interpretability_cutoff
        )

    concepts = [concepts[i] for i in range(len(concepts)) if interpretable[i].item()]
    if not concepts:
        raise RuntimeError(
            "No concepts remain after LF interpretability filtering. Lower --interpretability_cutoff."
        )
    artifacts = prune_projection_outputs(
        args,
        proj_layer,
        interpretable,
        train_backbone.shape[1],
    )
    proj_layer = artifacts.layer

    train_concepts = compute_concept_features(args, proj_layer, train_backbone)
    val_concepts = compute_concept_features(args, proj_layer, val_backbone)

    train_mean = train_concepts.mean(dim=0, keepdim=True)
    train_std = torch.clamp(train_concepts.std(dim=0, keepdim=True), min=1e-6)
    train_concepts = (train_concepts - train_mean) / train_std
    val_concepts = (val_concepts - train_mean) / train_std

    train_loader = DataLoader(
        IndexedTensorDataset(train_concepts, train_labels),
        batch_size=args.saga_batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_concepts, val_labels),
        batch_size=args.saga_batch_size,
        shuffle=False,
    )

    final_layer = nn.Linear(len(concepts), len(classes)).to(args.device)
    final_layer.weight.data.zero_()
    final_layer.bias.data.zero_()
    if args.dense:
        output_proj = train_dense_final(
            final_layer,
            train_loader,
            val_loader,
            args.saga_n_iters,
            args.dense_lr,
            device=args.device,
        )
    else:
        output_proj = train_sparse_final(
            final_layer,
            train_loader,
            val_loader,
            args.saga_n_iters,
            args.saga_lam,
            step_size=args.saga_step_size,
            device=args.device,
        )

    W_g = output_proj["path"][0]["weight"]
    b_g = output_proj["path"][0]["bias"]
    final_layer.load_state_dict({"weight": W_g, "bias": b_g})

    train_eval_dataset = TransformedSubset(
        data_utils.get_data(f"{args.dataset}_train", None),
        train_dataset.indices,
        backbone.preprocess,
    )
    val_eval_dataset = TransformedSubset(
        data_utils.get_data(f"{args.dataset}_train", None),
        val_dataset.indices,
        backbone.preprocess,
    )

    if getattr(args, "skip_train_val_eval", False):
        train_accuracy = None
        val_accuracy = None
    else:
        train_accuracy = evaluate_accuracy(
            backbone,
            proj_layer,
            train_mean,
            train_std,
            final_layer,
            train_eval_dataset,
            args,
        )
        val_accuracy = evaluate_accuracy(
            backbone,
            proj_layer,
            train_mean,
            train_std,
            final_layer,
            val_eval_dataset,
            args,
        )
    test_accuracy = evaluate_accuracy(
        backbone,
        proj_layer,
        train_mean,
        train_std,
        final_layer,
        test_dataset,
        args,
    )

    with open(os.path.join(save_dir, "concepts.txt"), "w") as f:
        f.write("\n".join(concepts))

    if artifacts.linear_weight is not None:
        torch.save(artifacts.linear_weight, os.path.join(save_dir, "W_c.pt"))
    else:
        torch.save(proj_layer.state_dict(), os.path.join(save_dir, "concept_layer.pt"))
    torch.save(W_g, os.path.join(save_dir, "W_g.pt"))
    torch.save(b_g, os.path.join(save_dir, "b_g.pt"))
    torch.save(train_mean, os.path.join(save_dir, "proj_mean.pt"))
    torch.save(train_std, os.path.join(save_dir, "proj_std.pt"))

    test_metrics = {"accuracy": test_accuracy}
    with open(os.path.join(save_dir, "test_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)
    if not getattr(args, "skip_train_val_eval", False):
        train_metrics = {"accuracy": train_accuracy}
        val_metrics = {"accuracy": val_accuracy}
        with open(os.path.join(save_dir, "train_metrics.json"), "w") as f:
            json.dump(train_metrics, f, indent=2)
        with open(os.path.join(save_dir, "val_metrics.json"), "w") as f:
            json.dump(val_metrics, f, indent=2)

    W_g_np = W_g.detach().cpu().numpy()
    interpretations = {}
    for class_idx, class_name in enumerate(classes):
        weights = W_g_np[class_idx]
        top_pos_idx = np.argsort(weights)[-10:][::-1]
        top_neg_idx = np.argsort(weights)[:10]
        interpretations[class_name] = {
            "positive": [
                (concepts[i], float(weights[i])) for i in top_pos_idx if weights[i] > 0
            ],
            "negative": [
                (concepts[i], float(weights[i])) for i in top_neg_idx if weights[i] < 0
            ],
        }
    with open(os.path.join(save_dir, "interpretations.json"), "w") as f:
        json.dump(interpretations, f, indent=2)

    method_log = {
        "cbm_variant": "lf_cbm",
        "concept_pseudo_label_source": "clip_image_text_cosine",
        "clip_name": args.lf_clip_name,
        "concept_filters": {
            "clip_cutoff": args.clip_cutoff,
            "interpretability_cutoff": args.interpretability_cutoff,
        },
        "concept_bottleneck_layer": {
            "type": args.cbl_type,
            "hidden_layers": args.cbl_hidden_layers if args.cbl_type == "mlp" else 0,
            "use_batchnorm": bool(args.cbl_use_batchnorm) if args.cbl_type == "mlp" else False,
        },
        "sparse_final_layer": {
            "solver": "glm_saga",
            "lam": args.saga_lam,
            "saga_iters": args.saga_n_iters,
            "saga_batch_size": args.saga_batch_size,
        },
    }
    with open(os.path.join(save_dir, "method_log.json"), "w") as f:
        json.dump(method_log, f, indent=2)

    write_artifacts(
        save_dir,
        {
            "model_name": args.model_name,
            "dataset": args.dataset,
            "backbone": args.backbone,
            "concept_layer_format": "W_c.pt" if artifacts.linear_weight is not None else "concept_layer.pt",
            "normalization_format": ["proj_mean.pt", "proj_std.pt"],
            "final_layer_format": ["W_g.pt", "b_g.pt"],
            "sparse_eval_style": "lf_linear_only" if artifacts.linear_weight is not None else "not_yet_supported",
        },
    )
    if getattr(args, "skip_train_val_eval", False):
        logger.info(f"LF-CBM test accuracy={test_accuracy:.4f}")
    else:
        logger.info(
            f"LF-CBM train accuracy={train_accuracy:.4f} val accuracy={val_accuracy:.4f} test accuracy={test_accuracy:.4f}"
        )
    return save_dir
