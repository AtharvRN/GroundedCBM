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
CUB_MODEL_CHOICES = ("vlg_cbm", "lf_cbm", "salf_cbm", "savlg_cbm")


def _run_imagenet_training(argv: list[str], config=None) -> None:
    if config is None:
        config = load_flat_config(option_value(argv, "--config"))
    model = model_from_argv_or_config(argv, config)
    if model not in IMAGENET_MODEL_ALIASES:
        raise SystemExit("ImageNet training in this repository supports SG-CBM only.")

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


def _load_cub_dependencies() -> None:
    global np, torch, nn, logger, SummaryWriter, tqdm
    global utils, data_utils, get_concept_dataloader, get_filtered_concepts_and_counts
    global get_final_layer_dataset, get_or_create_backbone_embedding_cache
    global get_loss, Backbone, BackboneCLIP, ConceptLayer, FinalLayer
    global per_class_accuracy, test_model, train_cbl, train_dense_final, train_sparse_final
    global get_model_name, write_artifacts, get_train_handler, SUPPORTED_MODELS

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
    if dataset is not None and str(dataset).lower() == "imagenet":
        _run_imagenet_training(argv, config)
        return

    parser = argparse.ArgumentParser(description="Train CUB CBM baselines or SG-CBM. Use --dataset imagenet for the ImageNet SG-CBM trainer.")
    parser.add_argument("--model_name", type=str, default="savlg_cbm", choices=CUB_MODEL_CHOICES, help="CUB model to train: savlg_cbm is SG-CBM.")
    parser.add_argument("--dataset", type=str, default="cub", help="Dataset name. Use cub here; imagenet dispatches to the ImageNet trainer.")
    parser.add_argument("--concept_set", type=str, default="concept_files/cub_filtered.txt", help="Concept list file.")
    parser.add_argument("--filter_set", type=str, default=None, help="Optional concept filter file.")
    parser.add_argument("--annotation_dir", type=str, default="outputs", help="GDINO/SALF annotation directory.")
    parser.add_argument("--save_dir", type=str, default="saved_models", help="Output directory for trained runs.")
    parser.add_argument("--load_dir", type=str, default=None, help="Optional existing CBL checkpoint directory.")
    parser.add_argument("--backbone", type=str, default="resnet50_cub_mm", help="Backbone name.")
    parser.add_argument("--feature_layer", type=str, default="layer4", help="Backbone feature layer for VLG/LF style heads.")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device.")
    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader worker count.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--max_train_images", type=int, default=0, help="If >0, limit training images for smoke tests.")
    parser.add_argument("--max_test_images", type=int, default=0, help="If >0, limit test images for smoke tests.")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split fraction.")
    parser.add_argument("--skip_test_eval", action="store_true", help="Skip final test-set evaluation.")
    parser.add_argument("--skip_train_val_eval", action="store_true", help="Skip final train/val accuracy evaluation.")
    parser.add_argument("--skip_concept_filter", action="store_true", help="Use all concepts without annotation-frequency filtering.")
    parser.add_argument("--cbl_batch_size", type=int, default=32, help="Concept-layer batch size.")
    parser.add_argument("--cbl_epochs", type=int, default=20, help="Concept-layer training epochs.")
    parser.add_argument("--cbl_lr", type=float, default=5e-4, help="Concept-layer learning rate.")
    parser.add_argument("--cbl_weight_decay", type=float, default=1e-5, help="Concept-layer weight decay.")
    parser.add_argument("--cbl_hidden_layers", type=int, default=1, help="Hidden layers for CUB concept heads.")
    parser.add_argument("--cbl_optimizer", choices=["adam", "sgd"], default="sgd", help="Concept-layer optimizer.")
    parser.add_argument("--cbl_scheduler", choices=[None, "cosine"], default=None, help="Concept-layer scheduler.")
    parser.add_argument("--saga_batch_size", type=int, default=512, help="Final-layer/SAGA batch size.")
    parser.add_argument("--saga_step_size", type=float, default=0.1, help="SAGA step size.")
    parser.add_argument("--saga_lam", type=float, default=0.0007, help="Sparse final-layer regularization.")
    parser.add_argument("--saga_n_iters", type=int, default=2000, help="Final-layer solver iterations.")
    parser.add_argument("--dense", action="store_true", help="Train a dense final layer instead of sparse SAGA.")
    parser.add_argument("--dense_lr", type=float, default=0.001, help="Dense final-layer learning rate.")
    parser.add_argument("--activation_cache_dir", type=str, default=None, help="Optional shared deterministic backbone embedding cache.")
    parser.add_argument("--disable_activation_cache", action="store_true", help="Disable deterministic activation caching.")
    parser.add_argument("--clip_cutoff", type=float, default=0.20, help="LF/SALF concept filter threshold.")
    parser.add_argument("--grid_h", type=int, default=7, help="SALF spatial grid height.")
    parser.add_argument("--grid_w", type=int, default=7, help="SALF spatial grid width.")
    parser.add_argument("--mask_h", type=int, default=14, help="SG-CBM spatial supervision mask height.")
    parser.add_argument("--mask_w", type=int, default=14, help="SG-CBM spatial supervision mask width.")
    parser.add_argument("--prompt_radius", type=int, default=3, help="SALF prompt radius in raw-image pixels.")
    parser.add_argument("--spatial_source", type=str, default="prompt_grid", choices=["prompt_grid", "patch_tokens"], help="SALF spatial target source.")
    parser.add_argument("--loss_mask_w", type=float, default=1.0, help="SG-CBM spatial soft-align KL weight.")
    parser.add_argument("--savlg_spatial_stage", type=str, default="conv5", choices=["conv3", "conv4", "conv5"], help="Spatial backbone stage.")
    parser.add_argument("--savlg_branch_arch", type=str, default="dual", choices=["shared", "dual"], help="SG-CBM concept-head architecture.")
    parser.add_argument("--savlg_spatial_branch_mode", type=str, default="multiscale_conv45", choices=["shared_stage", "multiscale_conv45"], help="SG-CBM spatial branch source.")
    parser.add_argument("--savlg_residual_spatial_alpha", type=float, default=0.2, help="Residual spatial-logit coupling weight.")
    parser.add_argument("--savlg_residual_spatial_pooling", type=str, default="lse", choices=["lse", "avg"], help="Pooling mode for residual spatial logits.")
    parser.add_argument("--savlg_target_mode", type=str, default="soft_box", choices=["hard_iou", "soft_box"], help="SG-CBM spatial target rasterization.")
    parser.add_argument("--savlg_stream_supervision", action="store_true", help="Build SG-CBM spatial supervision on the fly instead of saving *_supervision.pt caches.")
    parser.add_argument("--patch_iou_thresh", type=float, default=0.5, help="IoU threshold for hard_iou spatial targets.")
    parser.add_argument("--clip_score_mode", type=str, default="topk", choices=["mean", "topk", "quantile"], help="SALF reduction from spatial target maps to concept scores.")
    parser.set_defaults(
        activation_dir="saved_activations",
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
        cbl_pos_weight=1.0,
        cbl_twoway_tp=4.0,
        cbl_type="linear",
        cbl_use_batchnorm=False,
        clip_quantile=0.995,
        clip_topk=500,
        crop_to_concept_prob=0.0,
        data_parallel=False,
        global_bce_pos_weight=1.0,
        interpretability_cutoff=0.40,
        lf_batch_size=64,
        lf_clip_name="clip_RN50",
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
        recompute_spatial_sims=False,
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
        savlg_topk_fraction=0.2,
        spatial_batch_size=128,
        spatial_num_workers=8,
        use_clip_penultimate=False,
        visualize_concepts=False,
    )

    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default=None)
    config_arg, remaining_args = config_parser.parse_known_args()
    if config_arg.config is not None:
        with open(config_arg.config, "r") as f:
            config_arg = json.load(f)
        if "mask_h" in config_arg and "grid_h" not in config_arg:
            config_arg["grid_h"] = config_arg["mask_h"]
        if "mask_w" in config_arg and "grid_w" not in config_arg:
            config_arg["grid_w"] = config_arg["mask_w"]
        parser.set_defaults(**config_arg)
    
    # run the training
    args = parser.parse_args(remaining_args)
    _load_cub_dependencies()
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
