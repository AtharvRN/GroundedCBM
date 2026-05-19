from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from gcbm.features import standardize_from_train
from gcbm.losses import sgcbm_concept_losses, weighted_concept_bce
from gcbm.medical_annotations import (
    MedicalTargetDataset,
    build_medical_targets,
    load_concepts,
    medical_target_collate,
)
from gcbm.medical_data import (
    default_mimic_paths,
    get_medical_transforms,
    infer_chexpert_img_root,
    load_chexpert_dataset,
    load_mimic_cxr_dataset,
    medical_collate,
    medical_labels,
)
from gcbm.medical_metrics import compute_medical_metrics
from gcbm.sg_model import DualBranchConceptLayer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SG-CBM on CheXpert or MIMIC-CXR.")
    parser.add_argument("--dataset", choices=["chexpert", "mimic"], required=True, help="Medical dataset to train on.")
    parser.add_argument("--data_dir", required=True, help="Dataset root containing CheXpert CSVs or MIMIC-CXR-JPG files/CSVs.")
    parser.add_argument("--concept_file", required=True, help="Text or JSON concept bank.")
    parser.add_argument("--train_annotation_dir", required=True, help="Grounded concept annotations for train split.")
    parser.add_argument("--val_annotation_dir", required=True, help="Grounded concept annotations for validation split.")
    parser.add_argument("--save_dir", default="artifacts/medical", help="Directory for run outputs.")
    parser.add_argument("--run_name", default="", help="Optional run directory name.")
    parser.add_argument("--train_csv", default="", help="CheXpert train CSV override.")
    parser.add_argument("--val_csv", default="", help="CheXpert validation CSV override.")
    parser.add_argument("--img_root", default="", help="Image root override.")
    parser.add_argument("--mimic_label_csv", default="", help="MIMIC CheXpert-label CSV override.")
    parser.add_argument("--mimic_split_csv", default="", help="MIMIC split CSV override.")
    parser.add_argument("--mimic_metadata_csv", default="", help="MIMIC metadata CSV override.")
    parser.add_argument("--label_subset", choices=["all", "competition", "pathology"], default="all", help="CheXpert-style label subset.")
    parser.add_argument("--uncertain_strategy", choices=["ones", "zeros", "ignore"], default="ones", help="How to map uncertain labels.")
    parser.add_argument("--frontal_only", action=argparse.BooleanOptionalAction, default=True, help="Use only frontal/AP/PA images.")
    parser.add_argument("--backbone", choices=["densenet121", "resnet50"], default="densenet121", help="Spatial backbone.")
    parser.add_argument("--backbone_ckpt", default="", help="Optional classifier checkpoint; classifier weights are ignored.")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True, help="Use torchvision pretrained weights if no checkpoint is supplied.")
    parser.add_argument("--img_size", type=int, default=224, help="Input crop size.")
    parser.add_argument("--resize_size", type=int, default=256, help="Resize short edge before center crop.")
    parser.add_argument("--mask_h", type=int, default=14, help="Spatial target height.")
    parser.add_argument("--mask_w", type=int, default=14, help="Spatial target width.")
    parser.add_argument("--target_mode", choices=["soft_box", "hard_iou"], default="soft_box", help="Box-to-mask target mode.")
    parser.add_argument("--patch_iou_thresh", type=float, default=0.5, help="Patch IoU threshold for hard_iou targets.")
    parser.add_argument("--concept_threshold", type=float, default=0.15, help="Positive concept confidence threshold.")
    parser.add_argument("--neg_threshold", type=float, default=0.02, help="Soft target lower calibration threshold.")
    parser.add_argument("--global_pos_weight", type=float, default=1.0, help="Positive weight for global concept BCE.")
    parser.add_argument("--loss_global_w", type=float, default=1.0, help="Global concept BCE loss weight.")
    parser.add_argument("--loss_mask_w", type=float, default=1.0, help="Spatial soft-align KL loss weight.")
    parser.add_argument("--residual_alpha", type=float, default=0.2, help="Spatial residual logit coupling.")
    parser.add_argument("--epochs", type=int, default=10, help="Concept-layer training epochs.")
    parser.add_argument("--batch_size", type=int, default=64, help="Concept-layer batch size.")
    parser.add_argument("--final_batch_size", type=int, default=256, help="Final-layer batch size.")
    parser.add_argument("--lr", type=float, default=5e-4, help="Concept-layer learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Concept-layer weight decay.")
    parser.add_argument("--final_lr", type=float, default=1e-3, help="Dense final-layer learning rate.")
    parser.add_argument("--final_epochs", type=int, default=100, help="Dense final-layer epochs.")
    parser.add_argument("--use_saga", action="store_true", help="Train sparse final layer with GLM-SAGA instead of dense Adam.")
    parser.add_argument("--saga_lam", type=float, default=1e-3, help="GLM-SAGA regularization.")
    parser.add_argument("--saga_iters", type=int, default=1000, help="GLM-SAGA epochs.")
    parser.add_argument("--saga_max_lr", type=float, default=0.1, help="GLM-SAGA max learning rate.")
    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader workers.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for precision/recall/F1.")
    parser.add_argument("--device", default="cuda", help="Torch device.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--max_train_images", type=int, default=0, help="Optional train subset for smoke tests.")
    parser.add_argument("--max_val_images", type=int, default=0, help="Optional validation subset for smoke tests.")
    parser.add_argument("--train_target_cache", default="", help="Optional cache for train SG-CBM targets.")
    parser.add_argument("--val_target_cache", default="", help="Optional cache for val SG-CBM targets.")
    parser.add_argument("--allow_annotation_index_fallback", action="store_true", help="Allow row-index fallback when annotation paths do not match.")
    return parser.parse_args()


class MedicalBackbone(nn.Module):
    def __init__(self, name: str, *, pretrained: bool, checkpoint: str = "") -> None:
        super().__init__()
        from torchvision import models

        self.name = name
        if name == "densenet121":
            weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.densenet121(weights=weights)
            self.encoder = model.features
            self.feature_dim = 1024
        elif name == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.resnet50(weights=weights)
            self.encoder = nn.Sequential(*list(model.children())[:-2])
            self.feature_dim = 2048
        else:
            raise ValueError(f"Unsupported medical backbone: {name}")
        if checkpoint:
            self._load_checkpoint(checkpoint)

    def _load_checkpoint(self, path: str) -> None:
        import inspect

        load_kwargs = {"map_location": "cpu"}
        if "weights_only" in inspect.signature(torch.load).parameters:
            load_kwargs["weights_only"] = False
        payload = torch.load(path, **load_kwargs)
        state = payload.get("model_state_dict", payload.get("model", payload)) if isinstance(payload, dict) else payload
        state = self._encoder_state_from_checkpoint(state)
        missing, unexpected = self.load_state_dict(state, strict=False)
        print(f"[medical] loaded backbone checkpoint: missing={len(missing)} unexpected={len(unexpected)}", flush=True)

    def _encoder_state_from_checkpoint(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        cleaned: Dict[str, torch.Tensor] = {}
        for raw_key, value in state.items():
            key = raw_key.removeprefix("module.")
            if any(part in key for part in ("classifier", "classification_head", "fc.")):
                continue
            for prefix in ("backbone.encoder.", "model.encoder.", "encoder."):
                if key.startswith(prefix):
                    cleaned["encoder." + key[len(prefix) :]] = value
                    break
            else:
                if self.name == "densenet121" and key.startswith("features."):
                    cleaned["encoder." + key[len("features.") :]] = value
                elif self.name == "resnet50":
                    mapped_key = self._resnet_encoder_key(key)
                    if mapped_key is not None:
                        cleaned[mapped_key] = value
                elif key.startswith("encoder."):
                    cleaned[key] = value
        return cleaned

    @staticmethod
    def _resnet_encoder_key(key: str) -> str | None:
        mapping = {"conv1.": "encoder.0.", "bn1.": "encoder.1.", "layer1.": "encoder.4.", "layer2.": "encoder.5.", "layer3.": "encoder.6.", "layer4.": "encoder.7."}
        for prefix, replacement in mapping.items():
            if key.startswith(prefix):
                return replacement + key[len(prefix) :]
        return key if key.startswith("encoder.") else None

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(images)
        if self.name == "densenet121":
            feats = F.relu(feats, inplace=False)
        return feats


def build_datasets(args: argparse.Namespace, labels: list[str]):
    train_tf = get_medical_transforms(args.img_size, train=False)
    val_tf = get_medical_transforms(args.img_size, train=False)
    data_dir = Path(args.data_dir)
    if args.dataset == "chexpert":
        img_root = Path(args.img_root) if args.img_root else infer_chexpert_img_root(data_dir)
        train_csv = Path(args.train_csv) if args.train_csv else data_dir / "train.csv"
        val_csv = Path(args.val_csv) if args.val_csv else data_dir / "valid.csv"
        train_ds = load_chexpert_dataset(train_csv, img_root=img_root, labels=labels, transform=train_tf, uncertain_strategy=args.uncertain_strategy, frontal_only=args.frontal_only)
        val_ds = load_chexpert_dataset(val_csv, img_root=img_root, labels=labels, transform=val_tf, uncertain_strategy=args.uncertain_strategy, frontal_only=args.frontal_only)
        return train_ds, val_ds

    paths = default_mimic_paths(data_dir)
    label_csv = Path(args.mimic_label_csv) if args.mimic_label_csv else paths["label_csv"]
    split_csv = Path(args.mimic_split_csv) if args.mimic_split_csv else paths["split_csv"]
    metadata_csv = Path(args.mimic_metadata_csv) if args.mimic_metadata_csv else paths["metadata_csv"]
    train_ds = load_mimic_cxr_dataset(label_csv, img_root=Path(args.img_root) if args.img_root else data_dir, split="train", split_csv=split_csv, metadata_csv=metadata_csv, labels=labels, transform=train_tf, uncertain_strategy=args.uncertain_strategy, frontal_only=args.frontal_only)
    val_ds = load_mimic_cxr_dataset(label_csv, img_root=Path(args.img_root) if args.img_root else data_dir, split="validate", split_csv=split_csv, metadata_csv=metadata_csv, labels=labels, transform=val_tf, uncertain_strategy=args.uncertain_strategy, frontal_only=args.frontal_only)
    return train_ds, val_ds


class MedicalSubset(torch.utils.data.Subset):
    def get_image_path(self, index: int) -> str:
        return self.dataset.get_image_path(self.indices[index])  # type: ignore[attr-defined]

    def get_image_size(self, index: int):
        return self.dataset.get_image_size(self.indices[index])  # type: ignore[attr-defined]


def maybe_subset(dataset, max_images: int):
    if max_images <= 0 or max_images >= len(dataset):
        return dataset
    return MedicalSubset(dataset, list(range(int(max_images))))


def get_or_build_targets(dataset, args: argparse.Namespace, concepts: list[str], annotation_dir: str, cache_path: str) -> Dict[str, Any]:
    if cache_path and Path(cache_path).exists():
        import inspect

        load_kwargs = {"map_location": "cpu"}
        if "weights_only" in inspect.signature(torch.load).parameters:
            load_kwargs["weights_only"] = False
        return torch.load(cache_path, **load_kwargs)
    payload = build_medical_targets(
        dataset,
        annotation_dir=annotation_dir,
        concepts=concepts,
        mask_h=args.mask_h,
        mask_w=args.mask_w,
        concept_threshold=args.concept_threshold,
        neg_threshold=args.neg_threshold,
        target_mode=args.target_mode,
        input_size=args.img_size,
        resize_size=args.resize_size,
        patch_iou_thresh=args.patch_iou_thresh,
        allow_index_fallback=args.allow_annotation_index_fallback,
    )
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, cache_path)
    return payload


def build_head(backbone: MedicalBackbone, n_concepts: int, args: argparse.Namespace) -> DualBranchConceptLayer:
    global_layer = nn.Linear(backbone.feature_dim, n_concepts)
    spatial_layer = nn.Conv2d(backbone.feature_dim, n_concepts, kernel_size=1, bias=False)
    return DualBranchConceptLayer(
        global_layer,
        spatial_layer,
        residual_alpha=args.residual_alpha,
        residual_spatial_pooling="lse",
    )


def compute_losses(outputs: Dict[str, torch.Tensor], batch: Dict[str, Any], args: argparse.Namespace, device: str) -> Dict[str, torch.Tensor]:
    if args.loss_mask_w == 0.0:
        loss_global = weighted_concept_bce(outputs["final_logits"], batch["global_targets"].to(device), pos_weight=args.global_pos_weight)
        loss_mask = outputs["final_logits"].sum() * 0.0
    else:
        loss_global, loss_mask = sgcbm_concept_losses(
            outputs["final_logits"],
            outputs["spatial_maps"],
            batch["global_targets"].to(device, non_blocking=True),
            batch["mask_indices"].to(device, non_blocking=True),
            batch["mask_targets"].to(device, non_blocking=True),
            batch["mask_valid"].to(device, non_blocking=True),
            global_pos_weight=args.global_pos_weight,
        )
    return {
        "total": args.loss_global_w * loss_global + args.loss_mask_w * loss_mask,
        "global": loss_global.detach(),
        "mask": loss_mask.detach(),
    }


def train_one_epoch(backbone: MedicalBackbone, head: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, args: argparse.Namespace) -> Dict[str, float]:
    backbone.eval()
    head.train()
    totals = {"loss": 0.0, "global": 0.0, "mask": 0.0}
    count = 0
    for batch in tqdm(loader, desc="medical cbl train", leave=False):
        images = batch["image"].to(args.device, non_blocking=True)
        outputs = head(backbone(images))
        losses = compute_losses(outputs, batch, args, args.device)
        optimizer.zero_grad()
        losses["total"].backward()
        optimizer.step()
        n = int(images.shape[0])
        totals["loss"] += float(losses["total"].detach().cpu()) * n
        totals["global"] += float(losses["global"].cpu()) * n
        totals["mask"] += float(losses["mask"].cpu()) * n
        count += n
    return {key: value / max(count, 1) for key, value in totals.items()}


@torch.no_grad()
def evaluate_cbl(backbone: MedicalBackbone, head: nn.Module, loader: DataLoader, args: argparse.Namespace) -> Dict[str, float]:
    backbone.eval()
    head.eval()
    totals = {"loss": 0.0, "global": 0.0, "mask": 0.0}
    count = 0
    for batch in tqdm(loader, desc="medical cbl val", leave=False):
        images = batch["image"].to(args.device, non_blocking=True)
        outputs = head(backbone(images))
        losses = compute_losses(outputs, batch, args, args.device)
        n = int(images.shape[0])
        totals["loss"] += float(losses["total"].detach().cpu()) * n
        totals["global"] += float(losses["global"].cpu()) * n
        totals["mask"] += float(losses["mask"].cpu()) * n
        count += n
    return {key: value / max(count, 1) for key, value in totals.items()}


@torch.no_grad()
def extract_concepts(backbone: MedicalBackbone, head: nn.Module, loader: DataLoader, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    backbone.eval()
    head.eval()
    concept_chunks = []
    label_chunks = []
    for batch in tqdm(loader, desc="extract medical concepts"):
        images = batch["image"].to(device, non_blocking=True)
        outputs = head(backbone(images))
        concept_chunks.append(outputs["final_logits"].cpu())
        label_chunks.append(batch["target"].cpu())
    return torch.cat(concept_chunks), torch.cat(label_chunks)


def train_dense_final(train_x: torch.Tensor, train_y: torch.Tensor, val_x: torch.Tensor, val_y: torch.Tensor, args: argparse.Namespace) -> Tuple[torch.Tensor, torch.Tensor]:
    linear = nn.Linear(train_x.shape[1], train_y.shape[1]).to(args.device)
    optimizer = torch.optim.AdamW(linear.parameters(), lr=args.final_lr)
    criterion = nn.BCEWithLogitsLoss()
    best = None
    best_loss = float("inf")
    train_ds = torch.utils.data.TensorDataset(train_x.float(), train_y.float())
    train_loader = DataLoader(train_ds, batch_size=args.final_batch_size, shuffle=True)
    for _epoch in tqdm(range(args.final_epochs), desc="dense final"):
        linear.train()
        for features, labels in train_loader:
            features = features.to(args.device)
            labels = labels.to(args.device)
            optimizer.zero_grad()
            loss = criterion(linear(features), labels)
            loss.backward()
            optimizer.step()
        linear.eval()
        with torch.no_grad():
            val_loss = float(criterion(linear(val_x.to(args.device)), val_y.to(args.device)).cpu())
        if val_loss < best_loss:
            best_loss = val_loss
            best = {key: value.detach().cpu().clone() for key, value in linear.state_dict().items()}
    if best is not None:
        linear.load_state_dict(best)
    return linear.weight.detach().cpu(), linear.bias.detach().cpu()


def train_saga_final(train_x: torch.Tensor, train_y: torch.Tensor, val_x: torch.Tensor, val_y: torch.Tensor, args: argparse.Namespace) -> Tuple[torch.Tensor, torch.Tensor]:
    from glm_saga.elasticnet import IndexedTensorDataset, glm_saga

    train_loader = DataLoader(IndexedTensorDataset(train_x.float(), train_y.float()), batch_size=args.final_batch_size, shuffle=True)
    val_loader = DataLoader(torch.utils.data.TensorDataset(val_x.float(), val_y.float()), batch_size=args.final_batch_size, shuffle=False)
    linear = nn.Linear(train_x.shape[1], train_y.shape[1]).to(args.device)
    linear.weight.data.zero_()
    linear.bias.data.zero_()
    output = glm_saga(
        linear,
        train_loader,
        max_lr=args.saga_max_lr,
        nepochs=args.saga_iters,
        alpha=0.99,
        epsilon=1.0,
        k=1,
        val_loader=val_loader,
        do_zero=False,
        metadata={"max_reg": {"nongrouped": args.saga_lam}},
        n_ex=len(train_x),
        n_classes=train_y.shape[1],
        family="multilabel",
    )
    return output["path"][0]["weight"].cpu(), output["path"][0]["bias"].cpu()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.device = str(device)
    label_subset = args.label_subset
    labels = medical_labels(args.dataset, competition=label_subset == "competition", pathology=label_subset == "pathology")
    concepts = load_concepts(args.concept_file)
    run_name = args.run_name or f"{args.dataset}_sgcbm"
    run_dir = Path(args.save_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds = build_datasets(args, labels)
    train_ds = maybe_subset(train_ds, args.max_train_images)
    val_ds = maybe_subset(val_ds, args.max_val_images)
    train_targets = get_or_build_targets(train_ds, args, concepts, args.train_annotation_dir, args.train_target_cache)
    val_targets = get_or_build_targets(val_ds, args, concepts, args.val_annotation_dir, args.val_target_cache)
    train_target_ds = MedicalTargetDataset(train_ds, train_targets)
    val_target_ds = MedicalTargetDataset(val_ds, val_targets)
    train_loader = DataLoader(train_target_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn=medical_target_collate)
    val_loader = DataLoader(val_target_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=medical_target_collate)

    backbone = MedicalBackbone(args.backbone, pretrained=args.pretrained and not args.backbone_ckpt, checkpoint=args.backbone_ckpt).to(device)
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    head = build_head(backbone, len(concepts), args).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state = None
    best_val = float("inf")
    metrics_path = run_dir / "metrics.jsonl"
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(backbone, head, train_loader, optimizer, args)
        val_metrics = evaluate_cbl(backbone, head, val_loader, args)
        print(f"[medical] epoch={epoch} train={train_metrics} val={val_metrics}", flush=True)
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"epoch": epoch, "train": train_metrics, "val": val_metrics}) + "\n")
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
            torch.save(best_state, run_dir / "concept_head_best.pt")
    if best_state is not None:
        head.load_state_dict(best_state)
    torch.save(head.state_dict(), run_dir / "concept_head_final.pt")

    train_extract_loader = DataLoader(train_ds, batch_size=args.final_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=medical_collate)
    val_extract_loader = DataLoader(val_ds, batch_size=args.final_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=medical_collate)
    train_concepts, train_labels = extract_concepts(backbone, head, train_extract_loader, args.device)
    val_concepts, val_labels = extract_concepts(backbone, head, val_extract_loader, args.device)
    train_z, val_z, mean, std = standardize_from_train(train_concepts, val_concepts, unbiased=False)
    W, b = train_saga_final(train_z, train_labels, val_z, val_labels, args) if args.use_saga else train_dense_final(train_z, train_labels, val_z, val_labels, args)
    logits = val_z @ W.T + b
    probs = torch.sigmoid(logits).numpy()
    final_metrics = compute_medical_metrics(val_labels.numpy(), probs, labels, threshold=args.threshold)

    torch.save(W, run_dir / "W_g.pt")
    torch.save(b, run_dir / "b_g.pt")
    torch.save(mean, run_dir / "concept_mean.pt")
    torch.save(std, run_dir / "concept_std.pt")
    (run_dir / "concepts.txt").write_text("\n".join(concepts), encoding="utf-8")
    (run_dir / "labels.txt").write_text("\n".join(labels), encoding="utf-8")
    with (run_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump({**vars(args), "labels": labels, "num_concepts": len(concepts), "train_target_summary": {k: v for k, v in train_targets.items() if isinstance(v, (int, float, str))}, "val_target_summary": {k: v for k, v in val_targets.items() if isinstance(v, (int, float, str))}}, handle, indent=2)
    with (run_dir / "val_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(final_metrics, handle, indent=2)
    print(f"[medical] validation mean AUROC={final_metrics['mean_auroc']:.4f} mAP={final_metrics['mAP']:.4f}", flush=True)
    print(f"[medical] saved to {run_dir}", flush=True)


if __name__ == "__main__":
    main()
