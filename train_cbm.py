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
CUB_MODEL_CHOICES = ("vlg_cbm", "lf_cbm", "salf_cbm", "savlg_cbm")


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
        ConceptCorrRefinerCBL,
        ConceptLayer,
        CosineSimilarityConceptLayer,
        InputGatedRefinerCBL,
        LinearResidualRefinerCBL,
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
        _cbl_type = getattr(args, "cbl_type", "linear")
        if _cbl_type == "cosine_sim":
            cbl = CosineSimilarityConceptLayer(
                backbone.output_dim,
                len(concepts),
                tau=float(getattr(args, "cbl_tau", 20.0)),
                device=args.device,
            )
        elif _cbl_type == "linear_residual_refiner":
            cbl = LinearResidualRefinerCBL(
                backbone.output_dim,
                len(concepts),
                hidden_dim=int(getattr(args, "cbl_residual_hidden_dim", 64)),
                device=args.device,
            )
        elif _cbl_type == "input_gated_refiner":
            cbl = InputGatedRefinerCBL(
                backbone.output_dim,
                len(concepts),
                hidden_dim=int(getattr(args, "cbl_residual_hidden_dim", 64)),
                device=args.device,
            )
        elif _cbl_type == "concept_corr_refiner":
            cbl = ConceptCorrRefinerCBL(
                backbone.output_dim,
                len(concepts),
                corr_rank=int(getattr(args, "cbl_corr_rank", 16)),
                device=args.device,
            )
        else:
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
            early_stop_patience=args.cbl_early_stop_patience,
            min_delta=args.cbl_min_delta,
            min_epochs=args.cbl_min_epochs,
        )
    else:
        logger.info("Loading CBL from {}".format(args.load_dir))
        with open(os.path.join(args.load_dir, "args.txt")) as _f:
            _saved_args = json.load(_f)
        if _saved_args.get("cbl_type") == "cosine_sim":
            cbl = CosineSimilarityConceptLayer.from_pretrained(args.load_dir, args.device)
        elif _saved_args.get("cbl_type") == "linear_residual_refiner":
            cbl = LinearResidualRefinerCBL.from_pretrained(args.load_dir, args.device)
        elif _saved_args.get("cbl_type") == "input_gated_refiner":
            cbl = InputGatedRefinerCBL.from_pretrained(args.load_dir, args.device)
        elif _saved_args.get("cbl_type") == "concept_corr_refiner":
            cbl = ConceptCorrRefinerCBL.from_pretrained(args.load_dir, args.device)
        else:
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
    parser.add_argument("--config", type=str, default=None, help="Flat JSON/YAML config. CLI values override config values.")
    parser.add_argument("--model_name", type=str, default="savlg_cbm", choices=CUB_MODEL_CHOICES, help="CUB model to train: savlg_cbm is SG-CBM.")
    parser.add_argument("--dataset", type=str, default="cub", help="Dataset name. Use cub here; imagenet dispatches to the ImageNet trainer.")
    parser.add_argument("--annotation_dir", type=str, default="outputs", help="GDINO/SALF annotation directory.")
    parser.add_argument("--save_dir", type=str, default="saved_models", help="Output directory for trained runs.")
    parser.add_argument("--load_dir", type=str, default=None, help="Optional existing CBL checkpoint directory.")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device.")
    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader worker count.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--max_train_images", type=int, default=0, help="If >0, limit training images for quick checks.")
    parser.add_argument("--max_test_images", type=int, default=0, help="If >0, limit test images for quick checks.")
    parser.add_argument("--skip_test_eval", action="store_true", help="Skip final test-set evaluation.")
    parser.add_argument("--concept_set", type=str, default="concept_files/cub_filtered.txt", help="Concept list file.")
    parser.add_argument("--backbone", type=str, default="resnet18_cub", help="Backbone name.")
    parser.add_argument("--cbl_batch_size", type=int, default=32, help="Concept-layer batch size.")
    parser.add_argument("--cbl_epochs", type=int, default=75, help="Concept-layer training epochs.")
    parser.add_argument("--cbl_lr", type=float, default=5e-4, help="Concept-layer learning rate.")
    parser.add_argument("--cbl_optimizer", choices=["adam", "sgd"], default="adam", help="Concept-layer optimizer.")
    parser.add_argument("--mask_h", type=int, default=14, help="SG-CBM spatial supervision mask height.")
    parser.add_argument("--mask_w", type=int, default=14, help="SG-CBM spatial supervision mask width.")
    parser.add_argument("--loss_mask_w", type=float, default=0.25, help="SG-CBM spatial soft-align KL weight.")
    parser.add_argument("--loss_global_spatial_align_w", type=float, default=0.0, help="SG-CBM global/spatial concept probability alignment weight.")
    parser.add_argument("--savlg_residual_spatial_alpha", type=float, default=0.1, help="Residual spatial-logit coupling weight.")
    parser.add_argument("--savlg_target_mode", type=str, default="soft_box", choices=["hard_iou", "soft_box"], help="SG-CBM spatial target rasterization.")
    parser.add_argument("--disable_activation_cache", action="store_true", help="Disable deterministic activation caching.")
    parser.add_argument("--dense", action="store_true", help="Train a dense final layer instead of sparse SAGA.")
    parser.set_defaults(
        activation_dir="saved_activations",
        activation_cache_dir=None,
        allones_concept=False,
        cbl_auto_weight=False,
        cbl_bb_lr_rate=1.0,
        cbl_confidence_threshold=0.15,
        cbl_early_stop_patience=8,
        cbl_finetune=False,
        cbl_hidden_dim=0,
        cbl_loss_type="bce",
        cbl_tau=20.0,
        cbl_min_delta=1e-3,
        cbl_min_epochs=15,
        cbl_hidden_layers=1,
        cbl_optimizer="adam",
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
        grid_h=14,
        grid_w=14,
        interpretability_cutoff=0.40,
        lf_batch_size=64,
        lf_clip_name="clip_RN50",
        lf_original_protocol=False,
        loss_global_concept_w=None,
        loss_global_spatial_align_w=0.0,
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
        saga_lam=0.0002,
        saga_n_iters=4000,
        saga_step_size=0.1,
        savlg_branch_arch="dual",
        savlg_concept_filter_mode="spatial_threshold",
        savlg_freeze_global_head=False,
        savlg_global_head_mode="vlg_linear",
        savlg_global_hidden_dim=0,
        savlg_global_hidden_layers=0,
        savlg_global_use_batchnorm=False,
        savlg_init_from_vlg_path="",
        savlg_init_spatial_from_vlg=False,
        savlg_pooling="avg",
        savlg_residual_spatial_pooling="lse",
        savlg_spatial_branch_mode="multiscale_conv45",
        savlg_spatial_stage="conv5",
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
    from loguru import logger
    from methods.registry import get_train_handler

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
