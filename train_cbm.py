import argparse
import datetime
import json
import os
import random
import sys

from gcbm.config import (
    config_to_argv,
    load_flat_config,
    model_from_argv_or_config,
    option_value,
    strip_dispatcher_args,
)


IMAGENET_MODEL_ALIASES = {"sgcbm", "sg-cbm", "gcbm", "g-cbm", "savlg", "savlg-cbm"}
IMAGENET_VLG_ALIASES = {"vlg", "vlg-cbm", "vlg_cbm"}
MEDICAL_DATASETS = {"chexpert", "mimic"}
MODEL_CHOICES = ("vlg_cbm", "lf_cbm", "salf_cbm", "savlg_cbm", "sgcbm", "sg_cbm")


def _run_imagenet_training(argv: list[str], config=None) -> None:
    if config is None:
        config = load_flat_config(option_value(argv, "--config"))
    model = model_from_argv_or_config(argv, config)
    if model not in IMAGENET_MODEL_ALIASES and model not in IMAGENET_VLG_ALIASES:
        raise SystemExit("ImageNet training supports SG-CBM and VLG-CBM.")
    if model in IMAGENET_VLG_ALIASES:
        config = {
            **config,
            "branch_arch": "global_only",
            "loss_mask_w": 0.0,
            "residual_alpha": 0.0,
            "run_name": config.get("run_name") or "vlg_cbm_imagenet",
        }

    from gcbm.train_imagenet import main as imagenet_main

    old_argv = sys.argv[:]
    try:
        sys.argv = [
            "train_cbm.py",
            *config_to_argv(config),
            *strip_dispatcher_args(argv),
        ]
        imagenet_main()
    finally:
        sys.argv = old_argv


def _normalize_model_name(name: str) -> str:
    normalized = str(name).lower().replace("-", "_")
    if normalized in {"sgcbm", "sg_cbm", "savlg", "savlg_cbm"}:
        return "savlg_cbm"
    return normalized


def train_cbm_and_save(args):
    import numpy as np
    import torch
    import torch.nn as nn
    from loguru import logger
    from torch.utils.tensorboard import SummaryWriter
    from tqdm import tqdm

    import model.utils as utils
    from data import utils as data_utils
    from data.concept_dataset import (
        get_concept_dataloader,
        get_filtered_concepts_and_counts,
        get_final_layer_dataset,
        get_or_create_backbone_embedding_cache,
    )
    from gcbm.losses import get_loss
    from methods.common import get_model_name, write_artifacts
    from model.cbm import (
        Backbone,
        BackboneCLIP,
        ConceptLayer,
        FinalLayer,
        per_class_accuracy,
        test_model,
        train_cbl,
        train_dense_final,
        train_sparse_final,
    )

    # Setup log directory and logger
    save_dir = "{}/{}_cbm_{}".format(
        args.save_dir,
        args.dataset,
        datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"),
    )
    while os.path.exists(save_dir):
        save_dir += "-1"
    os.makedirs(save_dir)
    logger.add(
        os.path.join(save_dir, "train.log"),
        format="{time} {level} {message}",
        level="DEBUG",
    )
    logger.info("Saving model to {}".format(save_dir))
    with open(os.path.join(save_dir, "args.txt"), "w") as f:
        json.dump(args.__dict__, f, indent=2)
    write_artifacts(
        save_dir,
        {
            "model_name": get_model_name(args),
            "dataset": args.dataset,
            "backbone": args.backbone,
            "concept_layer_format": "cbl.pt",
            "normalization_format": [
                "train_concept_features_mean.pt",
                "train_concept_features_std.pt",
            ],
            "final_layer_format": ["final.pt"],
            "sparse_eval_style": "vlg_upstream",
        },
    )

    # Load classes
    classes = data_utils.get_classes(args.dataset)

    # Load Backbone model
    if args.backbone.startswith("clip_"):
        backbone = BackboneCLIP(
            args.backbone, use_penultimate=args.use_clip_penultimate, device=args.device
        )
    else:
        backbone = Backbone(args.backbone, args.feature_layer, args.device)

    # Remove concepts that are not present in the annotations
    if args.load_dir is None:
        if args.skip_concept_filter:
            logger.info("Skipping concept filtering")
            concepts, concept_counts = data_utils.load_concept_and_count(
                os.path.dirname(args.concept_set), filter_file=args.filter_set
            )
        else:
            # filter concepts
            logger.info("Filtering concepts")
            raw_concepts = data_utils.get_concepts(args.concept_set, args.filter_set)
            (
                concepts,
                concept_counts,
                filtered_concepts,
            ) = get_filtered_concepts_and_counts(
                args.dataset,
                raw_concepts,
                preprocess=backbone.preprocess,
                val_split=args.val_split,
                batch_size=args.cbl_batch_size,
                num_workers=args.num_workers,
                confidence_threshold=args.cbl_confidence_threshold,
                label_dir=args.annotation_dir,
                use_allones=args.allones_concept,
                seed=args.seed,
                max_images=args.max_train_images,
            )

            # save concept counts
            data_utils.save_concept_count(concepts, concept_counts, save_dir)
            data_utils.save_filtered_concepts(filtered_concepts, save_dir)
    else:
        # load concepts set directly from load model
        logger.info("Loading concepts from {}".format(args.load_dir))
        concepts, concept_counts = data_utils.load_concept_and_count(
            args.load_dir, filter_file=args.filter_set
        )

    with open(os.path.join(save_dir, "concepts.txt"), "w") as f:
        f.write(concepts[0])
        for concept in concepts[1:]:
            f.write("\n" + concept)

    # setup tensorboard writer
    tb_writer = SummaryWriter(save_dir)
    activation_cache_dir = (
        args.activation_cache_dir
        if args.activation_cache_dir is not None
        else os.path.join(args.annotation_dir, "_cache", "backbone_embeddings")
    )

    # setup all dataloaders
    augmented_train_cbl_loader = get_concept_dataloader(
        args.dataset,
        "train",
        concepts,
        preprocess=backbone.preprocess,
        val_split=args.val_split,
        batch_size=args.cbl_batch_size,
        num_workers=args.num_workers,
        shuffle=True,  # shuffle for training
        confidence_threshold=args.cbl_confidence_threshold,
        crop_to_concept_prob=args.crop_to_concept_prob,  # crop to concept
        label_dir=args.annotation_dir,
        use_allones=args.allones_concept,
        seed=args.seed,
        max_images=args.max_train_images,
    )
    train_cbl_loader = get_concept_dataloader(
        args.dataset,
        "train",
        concepts,
        preprocess=backbone.preprocess,
        val_split=args.val_split,
        batch_size=args.cbl_batch_size,
        num_workers=args.num_workers,
        shuffle=False,  # no shuffle to match order
        confidence_threshold=args.cbl_confidence_threshold,
        crop_to_concept_prob=0.0,  # no augmentation
        label_dir=args.annotation_dir,
        use_allones=args.allones_concept,
        seed=args.seed,
        max_images=args.max_train_images,
    )  # no shuffle to match labels
    val_cbl_loader = get_concept_dataloader(
        args.dataset,
        "val",
        concepts,
        preprocess=backbone.preprocess,
        val_split=args.val_split,
        batch_size=args.cbl_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        confidence_threshold=args.cbl_confidence_threshold,
        crop_to_concept_prob=0.0,  # no augmentation
        label_dir=args.annotation_dir,
        use_allones=args.allones_concept,
        seed=args.seed,
        max_images=args.max_train_images,
    )
    test_cbl_loader = get_concept_dataloader(
        args.dataset,
        "test",
        concepts,
        preprocess=backbone.preprocess,
        val_split=None,  # not needed
        batch_size=args.cbl_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        confidence_threshold=args.cbl_confidence_threshold,
        crop_to_concept_prob=0.0,  # no augmentation
        label_dir=args.annotation_dir,
        use_allones=args.allones_concept,
        seed=args.seed,
        max_images=args.max_test_images,
    )

    ##############################################
    # CBL training: Train CBL to map backbone features to concept space
    ##############################################
    loss_fn = get_loss(
        args.cbl_loss_type,
        len(concepts),
        len(train_cbl_loader.dataset),
        concept_counts,
        args.cbl_pos_weight,
        args.cbl_auto_weight,
        args.cbl_twoway_tp,
        args.device,
    )

    if args.load_dir is None:
        logger.info("Training CBL")
        cbl = ConceptLayer(
            backbone.output_dim,
            len(concepts),
            num_hidden=args.cbl_hidden_layers,
            device=args.device,
        )
        cached_val_embeddings = None
        cached_val_concepts = None
        use_activation_cache = args.use_activation_cache and not args.cbl_finetune
        if use_activation_cache:
            val_cached = get_or_create_backbone_embedding_cache(
                backbone,
                val_cbl_loader,
                device=args.device,
                cache_dir=activation_cache_dir,
                cache_tag="val",
            )
            cached_val_embeddings = val_cached["embeddings"]
            cached_val_concepts = val_cached["concept_one_hot"]
        cbl, backbone = train_cbl(
            backbone,
            cbl,
            augmented_train_cbl_loader,
            val_cbl_loader,
            args.cbl_epochs,
            loss_fn=loss_fn,
            lr=args.cbl_lr,
            weight_decay=args.cbl_weight_decay,
            concepts=concepts,
            tb_writer=tb_writer,
            device=args.device,
            finetune=args.cbl_finetune,
            optimizer=args.cbl_optimizer,
            scheduler=args.cbl_scheduler,
            backbone_lr=args.cbl_lr * args.cbl_bb_lr_rate,
            data_parallel=args.data_parallel,
            cached_val_embeddings=cached_val_embeddings,
            cached_val_concepts=cached_val_concepts,
        )
    else:
        logger.info("Loading CBL from {}".format(args.load_dir))
        cbl = ConceptLayer.from_pretrained(args.load_dir, args.device)
        if args.backbone.startswith("clip_"):
            raise NotImplementedError(
                "Loading backbone from pretrained model is not supported yet"
            )
        else:
            backbone = Backbone.from_pretrained(args.load_dir, args.device)

    cbl.save_model(save_dir)
    if args.cbl_finetune:
        backbone.save_model(save_dir)

    ##############################################
    # FINAL layer training
    ##############################################
    (
        train_concept_loader,
        val_concept_loader,
        normalization_layer,
    ) = get_final_layer_dataset(
        backbone,
        cbl,
        train_cbl_loader,
        val_cbl_loader,
        save_dir,
        load_dir=args.load_dir,
        batch_size=args.saga_batch_size,
        device=args.device,
        use_activation_cache=args.use_activation_cache and not args.cbl_finetune,
        activation_cache_dir=activation_cache_dir,
    )

    # Make linear model
    final_layer = FinalLayer(len(concepts), len(classes), device=args.device)

    if args.dense:
        logger.info(f"Training dense final layer with lr: {args.dense_lr} ...")
        output_proj = train_dense_final(
            final_layer,
            train_concept_loader,
            val_concept_loader,
            args.saga_n_iters,
            args.dense_lr,
            device=args.device,
        )
    else:
        logger.info(f"Training sparse final layer ...")
        output_proj = train_sparse_final(
            final_layer,
            train_concept_loader,
            val_concept_loader,
            args.saga_n_iters,
            args.saga_lam,
            step_size=args.saga_step_size,
            device=args.device,
        )

    W_g = output_proj["path"][0]["weight"]
    b_g = output_proj["path"][0]["bias"]
    final_layer.load_state_dict({"weight": W_g, "bias": b_g})
    final_layer.save_model(save_dir)

    ##############################################
    #### Test the model on test set ####
    ##############################################
    if getattr(args, "skip_test_eval", False):
        test_accuracy = None
        logger.info("Skipping test evaluation (--skip_test_eval)")
    else:
        test_accuracy = test_model(
            test_cbl_loader, backbone, cbl, normalization_layer, final_layer, args.device
        )
        logger.info(f"Test accuracy: {test_accuracy}")

    ##############################################
    # Store training metadata
    ##############################################
    with open(os.path.join(save_dir, "metrics.txt"), "w") as f:
        out_dict = {}
        out_dict["per_class_accuracies"] = per_class_accuracy(
            nn.Sequential(backbone, cbl, normalization_layer, final_layer).to(
                args.device
            ),
            test_cbl_loader,
            classes,
            device=args.device,
        )

        for key in ("lam", "lr", "alpha", "time"):
            out_dict[key] = float(output_proj["path"][0][key])
        out_dict["metrics"] = output_proj["path"][0]["metrics"]
        out_dict["metrics"]["test_accuracy"] = test_accuracy
        nnz = (W_g.abs() > 1e-5).sum().item()
        total = W_g.numel()
        out_dict["sparsity"] = {
            "Non-zero weights": nnz,
            "Total weights": total,
            "Percentage non-zero": nnz / total,
        }
        out_dict["skip_train_val_eval"] = bool(getattr(args, "skip_train_val_eval", False))
        out_dict["dense_eval_splits"] = ["test"]
        json.dump(out_dict, f, indent=2)

    if test_accuracy is not None:
        utils.write_parameters_tensorboard(
            tb_writer, vars(args), test_accuracy * 100.0, (nnz / total) * 100.0
        )

    ##############################################
    ## Visualize top images for concepts ##
    ##############################################
    if args.visualize_concepts:
        target_layer = data_utils.BACKBONE_VISUALIZATION_TARGET_LAYER[args.backbone]
        os.mkdir(os.path.join(save_dir, "concept_visualization"))
        cbl_with_backbone = nn.Sequential(backbone, cbl).to(args.device)
        concepts_logits = []
        for (images_tensor, _, _) in tqdm(test_cbl_loader):
            images_tensor = images_tensor.to(args.device)
            with torch.no_grad():
                concepts_logits.append(
                    cbl_with_backbone(images_tensor).detach().cpu().numpy()
                )
        concepts_logits = np.concatenate(concepts_logits, axis=0)
        for concept_idx, concept in enumerate(concepts):
            fig = utils.display_top_activated_images(
                concept_idx,
                concepts_logits,
                cbl_with_backbone,
                target_layer,
                test_cbl_loader.dataset,
                transform=backbone.preprocess,
                device=args.device,
                k=10,
            )
            fig.savefig(
                os.path.join(save_dir, "concept_visualization", f"{concept}.png")
            )

    return save_dir


def main():
    argv = sys.argv[1:]
    config = load_flat_config(option_value(argv, "--config"))
    dataset = option_value(argv, "--dataset") or config.get("dataset")
    model = model_from_argv_or_config(argv, config).replace("-", "_")
    if dataset is not None and str(dataset).lower() == "imagenet":
        _run_imagenet_training(argv, config)
        return

    parser = argparse.ArgumentParser(description="Train CBM baselines across CUB, ImageNet, and medical datasets.")
    parser.add_argument("--config", type=str, default=None, help="Flat JSON/YAML config. CLI values override config values.")
    parser.add_argument("--model_name", type=str, default="savlg_cbm", choices=MODEL_CHOICES, help="CBM variant to train. sgcbm is an alias for savlg_cbm.")
    parser.add_argument("--dataset", type=str, default="cub", help="Dataset name. Use cub here; imagenet dispatches to the ImageNet trainer.")
    parser.add_argument("--annotation_dir", type=str, default="outputs", help="GDINO/SALF annotation directory.")
    parser.add_argument("--data_dir", type=str, default="", help="Dataset root for medical datasets.")
    parser.add_argument("--img_root", type=str, default="", help="Optional image root override for medical datasets.")
    parser.add_argument("--train_csv", type=str, default="", help="Optional train CSV override for CheXpert.")
    parser.add_argument("--val_csv", type=str, default="", help="Optional val CSV override for CheXpert.")
    parser.add_argument("--mimic_label_csv", type=str, default="", help="Optional label CSV override for MIMIC-CXR.")
    parser.add_argument("--mimic_split_csv", type=str, default="", help="Optional split CSV override for MIMIC-CXR.")
    parser.add_argument("--mimic_metadata_csv", type=str, default="", help="Optional metadata CSV override for MIMIC-CXR.")
    parser.add_argument("--label_subset", type=str, default="all", choices=["all", "competition", "pathology"], help="Medical multilabel subset.")
    parser.add_argument("--uncertain_strategy", type=str, default="ones", choices=["ones", "zeros", "ignore"], help="How to map uncertain medical labels.")
    parser.add_argument("--frontal_only", action=argparse.BooleanOptionalAction, default=True, help="Use frontal-only images for medical datasets.")
    parser.add_argument("--save_dir", type=str, default="saved_models", help="Output directory for trained runs.")
    parser.add_argument("--load_dir", type=str, default=None, help="Optional existing CBL checkpoint directory.")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device.")
    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader worker count.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--max_train_images", type=int, default=0, help="If >0, limit training images for quick checks.")
    parser.add_argument("--max_val_images", type=int, default=0, help="If >0, limit validation images for medical VLG quick checks.")
    parser.add_argument("--max_test_images", type=int, default=0, help="If >0, limit test images for quick checks.")
    parser.add_argument("--skip_test_eval", action="store_true", help="Skip final test-set evaluation.")
    parser.add_argument("--concept_set", type=str, default="concept_files/cub_filtered.txt", help="Concept list file.")
    parser.add_argument("--concept_file", type=str, default="", help="Medical concept list file; defaults to concept_set when omitted.")
    parser.add_argument("--train_annotation_dir", type=str, default="", help="Medical train grounded annotation directory.")
    parser.add_argument("--val_annotation_dir", type=str, default="", help="Medical validation grounded annotation directory.")
    parser.add_argument("--train_concept_cache", type=str, default="", help="Medical VLG train concept cache.")
    parser.add_argument("--val_concept_cache", type=str, default="", help="Medical VLG validation concept cache.")
    parser.add_argument("--train_presence_cache", type=str, default="", help="Medical train concept-presence cache.")
    parser.add_argument("--val_presence_cache", type=str, default="", help="Medical validation concept-presence cache.")
    parser.add_argument("--train_target_cache", type=str, default="", help="Medical train SG/VLG target cache.")
    parser.add_argument("--val_target_cache", type=str, default="", help="Medical validation SG/VLG target cache.")
    parser.add_argument("--precomputed_target_dir", type=str, default="", help="Medical ImageNet-style target cache root.")
    parser.add_argument("--backbone", type=str, default="resnet50_cub_mm", help="Backbone name.")
    parser.add_argument("--backbone_ckpt", type=str, default="", help="Optional medical backbone checkpoint.")
    parser.add_argument("--concept_head_ckpt", type=str, default="", help="Optional trained medical concept-head checkpoint.")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True, help="Use torchvision pretrained weights if no checkpoint is supplied.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for multilabel metrics.")
    parser.add_argument("--val_split", type=float, default=0.1, help="Train/val split ratio for generic methods.")
    parser.add_argument("--img_size", type=int, default=224, help="Medical image crop size.")
    parser.add_argument("--resize_size", type=int, default=256, help="Medical resize size before crop.")
    parser.add_argument("--cbl_batch_size", type=int, default=32, help="Concept-layer batch size.")
    parser.add_argument("--cbl_epochs", type=int, default=20, help="Concept-layer training epochs.")
    parser.add_argument("--cbl_lr", type=float, default=5e-4, help="Concept-layer learning rate.")
    parser.add_argument("--cbl_confidence_threshold", type=float, default=0.15, help="Concept filtering / supervision threshold.")
    parser.add_argument("--concept_threshold", type=float, default=0.70, help="Medical positive concept confidence threshold.")
    parser.add_argument("--neg_threshold", type=float, default=0.02, help="Medical soft target lower calibration threshold.")
    parser.add_argument("--presence_mode", type=str, default="binary", choices=["binary", "soft"], help="Medical global concept target mode.")
    parser.add_argument("--min_concept_freq", type=float, default=0.01, help="Medical minimum train-set concept frequency.")
    parser.add_argument("--max_concept_freq", type=float, default=0.99, help="Medical maximum train-set concept frequency.")
    parser.add_argument("--mask_h", type=int, default=14, help="SG-CBM spatial supervision mask height.")
    parser.add_argument("--mask_w", type=int, default=14, help="SG-CBM spatial supervision mask width.")
    parser.add_argument("--target_mode", type=str, default="soft_box", choices=["soft_box", "hard_iou"], help="Medical box-to-mask target mode.")
    parser.add_argument("--grid_h", type=int, default=7, help="Spatial grid height for SALF/SAVLG.")
    parser.add_argument("--grid_w", type=int, default=7, help="Spatial grid width for SALF/SAVLG.")
    parser.add_argument("--loss_mask_w", type=float, default=1.0, help="SG-CBM spatial soft-align KL weight.")
    parser.add_argument("--loss_global_w", type=float, default=1.0, help="Global concept loss weight.")
    parser.add_argument("--global_pos_weight", type=float, default=1.0, help="Positive weight for medical global concept BCE.")
    parser.add_argument("--residual_alpha", type=float, default=0.2, help="Medical SG-CBM spatial residual logit coupling.")
    parser.add_argument("--epochs", type=int, default=10, help="Medical VLG concept-layer training epochs.")
    parser.add_argument("--early_stop_patience", type=int, default=0, help="Medical VLG early stopping patience.")
    parser.add_argument("--early_stop_min_delta", type=float, default=0.0, help="Medical VLG early stopping minimum delta.")
    parser.add_argument("--batch_size", type=int, default=64, help="Medical VLG concept-layer batch size.")
    parser.add_argument("--extract_batch_size", type=int, default=0, help="Medical VLG feature-extraction batch size.")
    parser.add_argument("--extract_chunk_size", type=int, default=10000, help="Medical VLG feature-extraction chunk size.")
    parser.add_argument("--final_batch_size", type=int, default=256, help="Medical final-layer batch size.")
    parser.add_argument("--final_epochs", type=int, default=100, help="Medical dense final-layer epochs.")
    parser.add_argument("--final_lr", type=float, default=1e-3, help="Medical dense final-layer learning rate.")
    parser.add_argument("--lr", type=float, default=5e-4, help="Medical VLG concept-layer learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Medical VLG concept-layer weight decay.")
    parser.add_argument("--use_saga", action="store_true", help="Use sparse GLM-SAGA final layer.")
    parser.add_argument("--saga_lam", type=float, default=0.0007, help="GLM-SAGA regularization strength.")
    parser.add_argument("--saga_iters", type=int, default=1000, help="Medical GLM-SAGA epochs.")
    parser.add_argument("--saga_n_iters", type=int, default=2000, help="Generic method GLM-SAGA epochs.")
    parser.add_argument("--saga_batch_size", type=int, default=512, help="GLM-SAGA batch size.")
    parser.add_argument("--saga_max_lr", type=float, default=0.1, help="Medical GLM-SAGA max learning rate.")
    parser.add_argument("--saga_step_size", type=float, default=0.1, help="Single-label GLM-SAGA step size.")
    parser.add_argument("--nec_values", type=str, default="", help="Comma-separated NEC values for sparse medical evaluation.")
    parser.add_argument("--savlg_residual_spatial_alpha", type=float, default=0.2, help="Residual spatial-logit coupling weight.")
    parser.add_argument("--savlg_target_mode", type=str, default="soft_box", choices=["hard_iou", "soft_box"], help="SG-CBM spatial target rasterization.")
    parser.add_argument("--savlg_concept_filter_mode", type=str, default="spatial_threshold", choices=["spatial_threshold", "vlg_global"], help="SAVLG concept filtering mode.")
    parser.add_argument("--savlg_stream_supervision", action="store_true", help="Stream SAVLG supervision from annotation JSONs.")
    parser.add_argument("--disable_activation_cache", action="store_true", help="Disable deterministic activation caching.")
    parser.add_argument("--dense", action="store_true", help="Train a dense final layer instead of sparse SAGA.")
    parser.add_argument("--dense_lr", type=float, default=1e-3, help="Dense final-layer learning rate.")
    parser.add_argument(
        "--lf_clip_name",
        type=str,
        default=None,
        help="Alignment backbone used by LF/SALF. CheXpert supports `cxrclip_swint_mcc` and `biomedclip`; default is CXR-CLIP. CUB/ImageNet default to clip_RN50.",
    )
    parser.add_argument("--clip_cutoff", type=float, default=0.20, help="Concept cutoff for LF/SALF.")
    parser.add_argument("--interpretability_cutoff", type=float, default=0.40, help="Interpretability cutoff for LF.")
    parser.add_argument("--lf_batch_size", type=int, default=64, help="LF feature-extraction batch size.")
    parser.add_argument("--proj_batch_size", type=int, default=512, help="LF projection batch size.")
    parser.add_argument("--proj_steps", type=int, default=20000, help="LF projection optimization steps.")
    parser.add_argument("--proj_eval_every", type=int, default=50, help="LF projection eval frequency.")
    parser.add_argument("--prompt_batch_size", type=int, default=1024, help="SALF prompt-grid batch size.")
    parser.add_argument("--prompt_radius", type=int, default=3, help="SALF prompt-grid radius.")
    parser.add_argument("--spatial_batch_size", type=int, default=128, help="SALF spatial similarity batch size.")
    parser.add_argument("--spatial_num_workers", type=int, default=8, help="SALF spatial similarity worker count.")
    parser.add_argument("--spatial_source", type=str, default="prompt_grid", help="Spatial supervision source for SALF.")
    parser.add_argument("--activation_dir", type=str, default="saved_activations", help="Directory for cached activations and spatial supervision.")
    parser.add_argument("--savlg_branch_arch", type=str, default="dual", help="SGCBM branch architecture.")
    parser.add_argument("--savlg_spatial_stage", type=str, default="conv5", help="Backbone stage used by SGCBM spatial branch.")
    parser.add_argument("--savlg_spatial_branch_mode", type=str, default="multiscale_conv45", help="SGCBM spatial branch feature mode.")
    parser.add_argument("--allow_annotation_index_fallback", action="store_true", help="Allow medical annotation row-index fallback.")
    parser.add_argument("--run_name", type=str, default="", help="Optional run directory name.")
    parser.set_defaults(
        activation_cache_dir=None,
        allones_concept=False,
        cbl_auto_weight=False,
        cbl_bb_lr_rate=1.0,
        cbl_confidence_threshold=0.15,
        cbl_early_stop_patience=0,
        cbl_finetune=False,
        cbl_hidden_dim=0,
        cbl_loss_type="bce",
        cbl_min_delta=0.0,
        cbl_min_epochs=0,
        cbl_hidden_layers=1,
        cbl_optimizer="sgd",
        cbl_pos_weight=1.0,
        cbl_scheduler=None,
        cbl_twoway_tp=4.0,
        cbl_type="linear",
        cbl_use_batchnorm=False,
        cbl_weight_decay=1e-5,
        clip_cutoff=0.20,
        clip_quantile=0.995,
        clip_score_mode="topk",
        clip_topk=500,
        crop_to_concept_prob=0.0,
        data_parallel=False,
        dense_lr=0.001,
        feature_layer="layer4",
        filter_set=None,
        global_bce_pos_weight=1.0,
        grid_h=7,
        grid_w=7,
        interpretability_cutoff=0.40,
        lf_batch_size=64,
        lf_clip_name=None,
        lf_original_protocol=False,
        loss_global_concept_w=None,
        loss_presence_w=None,
        proj_batch_size=512,
        proj_early_stop_patience=0,
        proj_eval_every=50,
        proj_lr=1e-3,
        proj_min_delta=0.0,
        proj_min_steps_before_early_stop=0,
        proj_steps=20000,
        prompt_batch_size=1024,
        prompt_radius=3,
        patch_iou_thresh=0.5,
        recompute_spatial_sims=False,
        saga_batch_size=512,
        saga_lam=0.0007,
        saga_n_iters=2000,
        saga_step_size=0.1,
        savlg_concept_filter_mode="spatial_threshold",
        savlg_freeze_global_head=False,
        savlg_global_head_mode="spatial_pool",
        savlg_global_hidden_dim=0,
        savlg_global_hidden_layers=0,
        savlg_global_use_batchnorm=False,
        savlg_init_from_vlg_path="",
        savlg_init_spatial_from_vlg=False,
        savlg_local_weight_floor=0.25,
        savlg_local_weight_mode="uniform",
        savlg_local_weight_power=1.0,
        savlg_pooling="avg",
        savlg_residual_spatial_pooling="lse",
        savlg_stream_supervision=False,
        savlg_target_transform="original",
        savlg_topk_fraction=0.2,
        skip_concept_filter=False,
        skip_train_val_eval=False,
        spatial_source="prompt_grid",
        spatial_batch_size=128,
        spatial_num_workers=8,
        use_clip_penultimate=False,
        val_split=0.1,
        visualize_concepts=False,
    )

    config_path = option_value(argv, "--config")
    if config_path is not None:
        config_defaults = dict(config)
        if "mask_h" in config_defaults and "grid_h" not in config_defaults:
            config_defaults["grid_h"] = config_defaults["mask_h"]
        if "mask_w" in config_defaults and "grid_w" not in config_defaults:
            config_defaults["grid_w"] = config_defaults["mask_w"]
        parser.set_defaults(**config_defaults)
    
    # run the training
    args = parser.parse_args(argv)

    import numpy as np
    import torch
    try:
        from loguru import logger  # type: ignore
    except Exception:  # pragma: no cover
        class _FallbackLogger:
            def info(self, *args, **kwargs):
                if args:
                    print(str(args[0]).format(*args[1:]))

        logger = _FallbackLogger()  # type: ignore
    from gcbm.clip_utils import resolve_lf_clip_name
    from gcbm.task_utils import is_medical_dataset
    from methods.registry import get_train_handler

    args.model_name = _normalize_model_name(args.model_name)
    if getattr(args, "concept_file", "") and (
        not getattr(args, "concept_set", "") or args.concept_set == parser.get_default("concept_set")
    ):
        args.concept_set = args.concept_file
    if not getattr(args, "concept_file", ""):
        args.concept_file = args.concept_set
    if int(getattr(args, "max_test_images", 0) or 0) <= 0 and int(getattr(args, "max_val_images", 0) or 0) > 0:
        args.max_test_images = args.max_val_images
    args.use_activation_cache = not args.disable_activation_cache
    args.lf_clip_name = resolve_lf_clip_name(
        getattr(args, "lf_clip_name", None),
        getattr(args, "dataset", None),
    )
    logger.info(args)
    
    # set random seed for reproducibility
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.model_name == "vlg_cbm" and not is_medical_dataset(args.dataset):
        _ = train_cbm_and_save(args)
    else:
        train_handler = get_train_handler(args.model_name)
        _ = train_handler(args)


if __name__ == "__main__":
    main()
