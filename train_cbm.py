import argparse
import datetime
import json
import os
import random
import sys

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
from loss import get_loss
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
from methods.common import get_model_name, write_artifacts
from methods.registry import get_train_handler, SUPPORTED_MODELS


class LoggerWriter:
    def __init__(self, level):
        self.level = level

    def write(self, message):
        if message.rstrip() != "":
            logger.log(self.level, message.rstrip())

    def flush(self):
        pass


def train_cbm_and_save(args):
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
            use_sam=args.cbl_use_sam,
            sam_rho=args.cbl_sam_rho,
            sam_adaptive=args.cbl_sam_adaptive,
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
    sys.stdout = LoggerWriter("INFO")
    sys.stderr = LoggerWriter("DEBUG")

    parser = argparse.ArgumentParser(description="Settings for creating CBM")
    parser.add_argument(
        "--model_name",
        type=str,
        default="vlg_cbm",
        choices=SUPPORTED_MODELS,
        help="Which CBM variant to train",
    )
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument(
        "--max_train_images",
        type=int,
        default=0,
        help="If >0, limit the number of training images used (useful for ImageNet smoke tests).",
    )
    parser.add_argument(
        "--max_test_images",
        type=int,
        default=0,
        help="If >0, limit the number of test images used (useful for ImageNet smoke tests).",
    )
    parser.add_argument(
        "--skip_test_eval",
        action="store_true",
        help="Skip test-set evaluation (useful when ImageNet val is not available or for quick smoke tests).",
    )
    parser.add_argument(
        "--concept_set", type=str, default=None, help="path to concept set name"
    )
    parser.add_argument(
        "--filter_set", type=str, default=None, help="path to concept set name"
    )
    parser.add_argument(
        "--val_split", type=float, default=0.1, help="Validation split fraction"
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="clip_RN50",
        help="Which pretrained model to use as backbone",
    )
    parser.add_argument(
        "--feature_layer",
        type=str,
        default="layer4",
        help="Which layer to collect activations from. Should be the name of second to last layer in the model",
    )
    parser.add_argument(
        "--use_clip_penultimate",
        action="store_true",
        help="Whether to use the penultimate layer of the CLIP backbone",
    )
    parser.add_argument(
        "--skip_concept_filter",
        action="store_true",
        help="Whether to skip filtering concepts",
    )
    parser.add_argument(
        "--annotation_dir",
        type=str,
        default="outputs",
        help="where annotations are saved",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="saved_models",
        help="where to save trained models",
    )
    parser.add_argument(
        "--load_dir", type=str, default=None, help="where to load trained models from"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Which device to use"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of workers used for loading data",
    )
    parser.add_argument(
        "--allones_concept",
        action="store_true",
        help="Change concept dataset to ones corresponding to class",
    )
    # arguments for CBL
    parser.add_argument(
        "--crop_to_concept_prob",
        type=float,
        default=0.0,
        help="Probability of cropping to concept granuality",
    )
    parser.add_argument(
        "--cbl_confidence_threshold",
        type=float,
        default=0.15,
        help="Confidence threshold for bouding boxes to use",
    )
    parser.add_argument(
        "--cbl_hidden_layers",
        type=int,
        default=1,
        help="how many hidden layers to use in the projection layer",
    )
    parser.add_argument(
        "--cbl_batch_size",
        type=int,
        default=32,
        help="Batch size used when fitting projection layer",
    )
    parser.add_argument(
        "--cbl_epochs",
        type=int,
        default=20,
        help="how many steps to train the projection layer for",
    )
    parser.add_argument(
        "--cbl_weight_decay",
        type=float,
        default=1e-5,
        help="weight decay for training the projection layer",
    )
    parser.add_argument(
        "--cbl_lr",
        type=float,
        default=5e-4,
        help="learning rate for training the projection layer",
    )
    parser.add_argument(
        "--cbl_loss_type",
        choices=["bce", "twoway"],
        default="bce",
        help="Loss type for training CBL",
    )
    parser.add_argument(
        "--cbl_twoway_tp",
        type=float,
        default=4.0,
        help="TPE hyperparameter for TwoWay CBL loss",
    )
    parser.add_argument(
        "--cbl_pos_weight",
        type=float,
        default=1.0,
        help="loss weight for positive examples",
    )
    parser.add_argument(
        "--cbl_auto_weight",
        action="store_true",
        help="whether to automatically weight positive examples",
    )
    parser.add_argument(
        "--cbl_finetune",
        action="store_true",
        help="Enable finetuning backbone in CBL training",
    )
    parser.add_argument(
        "--cbl_bb_lr_rate",
        type=float,
        default=1,
        help="Rescale the learning rate of backbone during finetuning",
    )
    parser.add_argument(
        "--cbl_optimizer",
        choices=["adam", "sgd"],
        default="sgd",
        help="Optimizer used in CBL training.",
    )
    parser.add_argument(
        "--cbl_use_sam",
        action="store_true",
        help="Wrap the CBL optimizer with Sharpness-Aware Minimization.",
    )
    parser.add_argument(
        "--cbl_sam_rho",
        type=float,
        default=0.05,
        help="Neighborhood size rho for Sharpness-Aware Minimization.",
    )
    parser.add_argument(
        "--cbl_sam_adaptive",
        action="store_true",
        help="Use the adaptive SAM variant that scales perturbations by weight magnitude.",
    )
    parser.add_argument(
        "--cbl_scheduler",
        choices=[None, "cosine"],
        default=None,
        help="Scheduler used in CBL training.",
    )
    # arguments for SAGA
    parser.add_argument(
        "--saga_batch_size",
        type=int,
        default=512,
        help="Batch size used when fitting final layer",
    )
    parser.add_argument(
        "--saga_step_size", type=float, default=0.1, help="Step size for SAGA"
    )
    parser.add_argument(
        "--saga_lam",
        type=float,
        default=0.0007,
        help="Sparsity regularization parameter, higher->more sparse",
    )
    parser.add_argument(
        "--saga_n_iters",
        type=int,
        default=2000,
        help="How many iterations to run the final layer solver for",
    )
    parser.add_argument(
        "--skip_train_val_eval",
        action="store_true",
        help="Skip final train/val accuracy evaluation and only compute test accuracy.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--activation_cache_dir",
        type=str,
        default=None,
        help="Shared directory for cached deterministic backbone embeddings",
    )
    parser.add_argument(
        "--disable_activation_cache",
        action="store_true",
        help="Disable deterministic backbone activation caching",
    )
    parser.add_argument(
        "--dense", action="store_true", help="train with dense or sparse method"
    )
    parser.add_argument(
        "--dense_lr",
        type=float,
        default=0.001,
        help="Learning rate for the dense final layer training",
    )
    parser.add_argument("--data_parallel", action="store_true")
    parser.add_argument(
        "--visualize_concepts", action="store_true", help="Visualize concepts"
    )
    parser.add_argument(
        "--lf_clip_name",
        type=str,
        default="clip_RN50",
        help="CLIP model used for LF-CBM pseudo-label similarities",
    )
    parser.add_argument(
        "--lf_batch_size",
        type=int,
        default=64,
        help="Batch size for LF-CBM backbone/CLIP feature extraction",
    )
    parser.add_argument(
        "--clip_cutoff",
        type=float,
        default=0.20,
        help="LF-CBM concept filter based on top-k CLIP activation",
    )
    parser.add_argument(
        "--interpretability_cutoff",
        type=float,
        default=0.40,
        help="LF-CBM interpretability filter threshold",
    )
    parser.add_argument(
        "--proj_steps",
        type=int,
        default=20000,
        help="LF-CBM projection-layer optimization steps",
    )
    parser.add_argument(
        "--proj_lr",
        type=float,
        default=1e-3,
        help="LF-CBM projection-layer learning rate",
    )
    parser.add_argument(
        "--proj_batch_size",
        type=int,
        default=512,
        help="LF-CBM projection-layer batch size",
    )
    parser.add_argument(
        "--proj_eval_every",
        type=int,
        default=50,
        help="LF-CBM projection-layer validation interval",
    )
    parser.add_argument(
        "--proj_early_stop_patience",
        type=int,
        default=0,
        help="LF-CBM early-stop patience for projection training",
    )
    parser.add_argument(
        "--proj_min_steps_before_early_stop",
        type=int,
        default=0,
        help="LF-CBM warmup before projection early-stop activates",
    )
    parser.add_argument(
        "--proj_min_delta",
        type=float,
        default=0.0,
        help="LF-CBM minimum validation improvement for early-stop reset",
    )
    parser.add_argument(
        "--cbl_type",
        type=str,
        default="linear",
        choices=["linear", "mlp"],
        help="LF-CBM projection type",
    )
    parser.add_argument(
        "--cbl_use_batchnorm",
        action="store_true",
        help="Use BatchNorm in LF-CBM MLP concept bottleneck",
    )
    parser.add_argument(
        "--cbl_hidden_dim",
        type=int,
        default=0,
        help="Hidden dimension for SALF/SAVLG spatial MLP bottlenecks; <=0 uses backbone dim",
    )
    parser.add_argument(
        "--cbl_early_stop_patience",
        type=int,
        default=0,
        help="Early-stop patience for SAVLG/SALF concept-head training; <=0 disables",
    )
    parser.add_argument(
        "--cbl_min_epochs",
        type=int,
        default=0,
        help="Minimum number of concept-head epochs before SAVLG/SALF early-stop can trigger",
    )
    parser.add_argument(
        "--cbl_min_delta",
        type=float,
        default=0.0,
        help="Minimum validation-loss improvement required to reset SAVLG/SALF early-stop patience",
    )
    parser.add_argument(
        "--lf_original_protocol",
        action="store_true",
        help="Use the original LF-style official train/test split semantics when supported",
    )
    parser.add_argument("--grid_h", type=int, default=7, help="Spatial grid height for SALF/SAVLG")
    parser.add_argument("--grid_w", type=int, default=7, help="Spatial grid width for SALF/SAVLG")
    parser.add_argument(
        "--prompt_radius",
        type=int,
        default=3,
        help="Prompt radius in raw-image pixels for SALF prompt-grid target generation",
    )
    parser.add_argument(
        "--spatial_source",
        type=str,
        default="prompt_grid",
        choices=["prompt_grid", "patch_tokens"],
        help="How SALF/SAVLG spatial targets are constructed",
    )
    parser.add_argument(
        "--savlg_spatial_stage",
        type=str,
        default="conv5",
        choices=["conv3", "conv4", "conv5"],
        help="Spatial backbone stage for SALF/SAVLG ResNet18 runs: conv3=28x28x128, conv4=14x14x256, conv5=7x7x512",
    )
    parser.add_argument(
        "--spatial_batch_size",
        type=int,
        default=128,
        help="Dataset batch size used only while building SALF/SAVLG spatial target tensors",
    )
    parser.add_argument(
        "--prompt_batch_size",
        type=int,
        default=1024,
        help="Prompted-image CLIP batch size used while building SALF prompt-grid targets",
    )
    parser.add_argument(
        "--spatial_num_workers",
        type=int,
        default=8,
        help="Dataloader worker count used only while building SALF/SAVLG spatial target tensors",
    )
    parser.add_argument(
        "--activation_dir",
        type=str,
        default="saved_activations",
        help="Directory used for cached LF/SALF/SAVLG intermediate artifacts",
    )
    parser.add_argument(
        "--recompute_spatial_sims",
        action="store_true",
        help="Force recompute cached SALF/SAVLG spatial target tensors",
    )
    parser.add_argument(
        "--loss_global_concept_w",
        type=float,
        default=None,
        help="Weight for SAVLG global concept BCE loss; if omitted, the deprecated --loss_presence_w alias is used when present.",
    )
    parser.add_argument(
        "--loss_presence_w",
        type=float,
        default=None,
        help="Deprecated alias for --loss_global_concept_w.",
    )
    parser.add_argument(
        "--loss_mask_w",
        type=float,
        default=1.0,
        help="Weight for SAVLG local patch-mask BCE loss",
    )
    parser.add_argument(
        "--loss_dice_w",
        type=float,
        default=0.0,
        help="Weight for SAVLG local Dice loss on valid boxed concepts",
    )
    parser.add_argument(
        "--global_bce_pos_weight",
        type=float,
        default=1.0,
        help="Positive-class weight for SAVLG global concept BCE",
    )
    parser.add_argument(
        "--patch_bce_pos_weight",
        type=float,
        default=1.0,
        help="Positive-class weight for SAVLG local patch-mask BCE",
    )
    parser.add_argument(
        "--local_bce_pos_weight",
        type=float,
        default=1.0,
        help="Positive-class weight for SAVLG local MIL BCE",
    )
    parser.add_argument(
        "--savlg_global_target_mode",
        type=str,
        default="binary_threshold",
        choices=["binary_threshold", "raw_logit"],
        help="How SAVLG converts annotation logits into global concept targets; binary_threshold matches VLG concept labels.",
    )
    parser.add_argument(
        "--savlg_concept_filter_mode",
        type=str,
        default="spatial_threshold",
        choices=["spatial_threshold", "vlg_global"],
        help="How SAVLG filters concepts before training. 'vlg_global' reuses the same concept-dataset filtering path as VLG-CBM.",
    )
    parser.add_argument(
        "--loss_local_mil_w",
        type=float,
        default=0.0,
        help="Weight for SAVLG auxiliary local MIL loss",
    )
    parser.add_argument(
        "--savlg_local_weight_mode",
        type=str,
        default="uniform",
        choices=["uniform", "confidence"],
        help="How SAVLG reweights local losses across positive concept-image pairs",
    )
    parser.add_argument(
        "--savlg_local_weight_floor",
        type=float,
        default=0.25,
        help="Minimum local-loss weight retained for weak SAVLG positive pairs when confidence weighting is enabled",
    )
    parser.add_argument(
        "--savlg_local_weight_power",
        type=float,
        default=1.0,
        help="Exponent applied to normalized SAVLG annotation confidence when local confidence weighting is enabled",
    )
    parser.add_argument(
        "--savlg_pooling",
        type=str,
        default="avg",
        choices=["avg", "topk"],
        help="Global pooling mode for SAVLG concept maps",
    )
    parser.add_argument(
        "--savlg_branch_arch",
        type=str,
        default="shared",
        choices=["shared", "dual"],
        help="Whether SAVLG uses one shared spatial concept head or separate global/spatial heads.",
    )
    parser.add_argument(
        "--savlg_global_head_mode",
        type=str,
        default="spatial_pool",
        choices=["spatial_pool", "vlg_linear"],
        help="How SAVLG computes global concept logits; vlg_linear matches the original VLG-CBM path with GAP over conv5 features followed by a linear concept layer.",
    )
    parser.add_argument(
        "--savlg_init_from_vlg_path",
        type=str,
        default="",
        help="Optional VLG-CBM checkpoint used to initialize compatible SAVLG concept heads.",
    )
    parser.add_argument(
        "--savlg_init_spatial_from_vlg",
        action="store_true",
        help="When --savlg_init_from_vlg_path is set, also initialize compatible SAVLG spatial 1x1 concept layers from the VLG concept weights.",
    )
    parser.add_argument(
        "--savlg_freeze_global_head",
        action="store_true",
        help="Freeze the SAVLG global concept head parameters after optional VLG warm-start, so only the spatial branch is trained.",
    )
    parser.add_argument(
        "--no_savlg_freeze_global_head",
        action="store_false",
        dest="savlg_freeze_global_head",
        help="Do not freeze the SAVLG global concept head parameters after optional VLG warm-start.",
    )
    parser.add_argument(
        "--savlg_global_hidden_layers",
        type=int,
        default=0,
        help="Number of hidden ReLU+Linear layers in the VLG-style SAVLG global concept head when --savlg_global_head_mode=vlg_linear.",
    )
    parser.add_argument(
        "--savlg_global_hidden_dim",
        type=int,
        default=0,
        help="Hidden width for the VLG-style SAVLG global concept head when using hidden layers; defaults to num_concepts if 0.",
    )
    parser.add_argument(
        "--savlg_global_use_batchnorm",
        action="store_true",
        help="Use BatchNorm1d between hidden layers in the VLG-style SAVLG global concept head.",
    )
    parser.add_argument(
        "--savlg_spatial_branch_mode",
        type=str,
        default="shared_stage",
        choices=["shared_stage", "multiscale_conv45"],
        help="How the SAVLG spatial branch consumes backbone features; multiscale_conv45 keeps the global branch on conv5 and fuses conv4+conv5 only for localization.",
    )
    parser.add_argument(
        "--savlg_topk_fraction",
        type=float,
        default=0.2,
        help="Patch fraction used when --savlg_pooling=topk",
    )
    parser.add_argument(
        "--savlg_use_local_mil",
        action="store_true",
        help="Enable an auxiliary local MIL objective for SAVLG",
    )
    parser.add_argument(
        "--savlg_local_pooling",
        type=str,
        default="lse",
        choices=["lse", "topk"],
        help="Pooling mode for SAVLG local MIL logits",
    )
    parser.add_argument(
        "--savlg_mil_temperature",
        type=float,
        default=1.0,
        help="Temperature for SAVLG LSE local MIL pooling",
    )
    parser.add_argument(
        "--savlg_mil_topk_fraction",
        type=float,
        default=0.2,
        help="Patch fraction used when --savlg_local_pooling=topk",
    )
    parser.add_argument(
        "--savlg_residual_spatial_alpha",
        type=float,
        default=0.0,
        help="Residual coupling weight in c_final = c_global + alpha * c_spatial for SAVLG concept logits.",
    )
    parser.add_argument(
        "--savlg_residual_spatial_pooling",
        type=str,
        default="lse",
        choices=["lse", "avg", "topk"],
        help="Pooling mode for residual SAVLG spatial logits.",
    )
    parser.add_argument(
        "--savlg_residual_topk_fraction",
        type=float,
        default=0.2,
        help="Patch fraction used when --savlg_residual_spatial_pooling=topk",
    )
    parser.add_argument(
        "--savlg_global_spatial_consistency_w",
        type=float,
        default=0.0,
        help="Weight for detached spatial-to-global SAVLG concept consistency on positive concept-image pairs",
    )
    parser.add_argument(
        "--savlg_global_spatial_consistency_warmup_epochs",
        type=int,
        default=0,
        help="Number of SAVLG concept-head epochs to wait before enabling spatial-to-global consistency",
    )
    parser.add_argument(
        "--savlg_target_mode",
        type=str,
        default="hard_iou",
        choices=["hard_iou", "soft_box"],
        help="How SAVLG rasterizes box supervision into patch targets",
    )
    parser.add_argument(
        "--savlg_supervision_source",
        type=str,
        default="gdino",
        choices=["gdino", "groundedsam2"],
        help="Spatial supervision source used by SAVLG: stored GDINO boxes or cached GroundedSAM2 masks",
    )
    parser.add_argument(
        "--savlg_groundedsam2_train_manifest",
        type=str,
        default=None,
        help="Path to the GroundedSAM2 train manifest.json (or its containing directory) used when --savlg_supervision_source=groundedsam2",
    )
    parser.add_argument(
        "--savlg_groundedsam2_val_manifest",
        type=str,
        default=None,
        help="Path to the GroundedSAM2 val manifest.json (or its containing directory) used when --savlg_supervision_source=groundedsam2",
    )
    parser.add_argument(
        "--savlg_stream_supervision",
        action="store_true",
        help=(
            "Build SAVLG spatial supervision targets on-the-fly inside the dataset __getitem__ "
            "instead of precomputing and saving a large *_supervision.pt cache. "
            "Useful for quick experiments when annotations are on fast local storage."
        ),
    )
    parser.add_argument(
        "--savlg_local_loss_mode",
        type=str,
        default="bce",
        choices=["bce", "containment", "soft_align"],
        help="How SAVLG applies local box supervision once patch targets are built",
    )
    parser.add_argument(
        "--savlg_outside_penalty_w",
        type=float,
        default=0.0,
        help="Weight for penalizing normalized SAVLG spatial activation mass outside the GT box",
    )
    parser.add_argument(
        "--savlg_absent_topk_w",
        type=float,
        default=0.0,
        help="Weight for penalizing top-k spatial activation on concept-absent maps",
    )
    parser.add_argument(
        "--savlg_absent_topk_fraction",
        type=float,
        default=0.1,
        help="Fraction of highest-activation spatial locations to penalize for absent concepts",
    )
    parser.add_argument(
        "--patch_iou_thresh",
        type=float,
        default=0.5,
        help="IoU threshold used for SAVLG hard patch targets",
    )
    parser.add_argument(
        "--savlg_teacher_load_path",
        type=str,
        default=None,
        help="Optional VLG-CBM teacher checkpoint directory for SAVLG distillation",
    )
    parser.add_argument(
        "--savlg_distill_w",
        type=float,
        default=0.0,
        help="Weight for SAVLG teacher-distillation loss on pooled concept logits",
    )
    parser.add_argument(
        "--savlg_refine_w",
        type=float,
        default=0.0,
        help="Weight for SAVLG OICR-style detached patch refinement loss",
    )
    parser.add_argument(
        "--savlg_refine_warmup_epochs",
        type=int,
        default=0,
        help="Number of SAVLG concept-head epochs to train before enabling refinement",
    )
    parser.add_argument(
        "--clip_score_mode",
        type=str,
        default="topk",
        choices=["mean", "topk", "quantile"],
        help="How SALF reduces spatial target tensors to per-concept scores before clip_cutoff",
    )
    parser.add_argument(
        "--clip_topk",
        type=int,
        default=500,
        help="k used when --clip_score_mode=topk",
    )
    parser.add_argument(
        "--clip_quantile",
        type=float,
        default=0.995,
        help="Quantile used when --clip_score_mode=quantile",
    )

    config_parser = argparse.ArgumentParser()
    config_parser.add_argument("--config", type=str, default=None)
    config_arg, remaining_args = config_parser.parse_known_args()
    if config_arg.config is not None:
        with open(config_arg.config, "r") as f:
            config_arg = json.load(f)
        parser.set_defaults(**config_arg)
    
    # run the training
    args = parser.parse_args(remaining_args)
    args.use_activation_cache = not args.disable_activation_cache
    logger.info(args)
    
    # set random seed for reproducibility
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.model_name == "vlg_cbm":
        _ = train_cbm_and_save(args)
    else:
        train_handler = get_train_handler(args.model_name)
        _ = train_handler(args)


if __name__ == "__main__":
    main()
