from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Config:
    mode: str
    train_root: str
    train_manifest: str
    annotation_dir: str
    concept_file: str
    val_root: str
    save_dir: str
    run_name: str
    reuse_run_dir: str
    feature_dir: str
    precomputed_target_dir: str
    persist_feature_copy: bool
    max_train_images: int
    max_val_images: int
    val_split: float
    epochs: int
    batch_size: int
    workers: int
    prefetch_factor: int
    persistent_workers: bool
    pin_memory: bool
    device: str
    amp: str
    channels_last: bool
    tf32: bool
    cudnn_benchmark: bool
    seed: int
    min_image_bytes: int
    input_size: int
    resnet50_weights: str
    train_random_transforms: bool
    mask_h: int
    mask_w: int
    patch_iou_thresh: float
    concept_threshold: float
    spatial_target_mode: str
    spatial_loss_mode: str
    filter_concepts_by_count: bool
    concept_min_count: int
    concept_min_frequency: float
    concept_max_frequency: float
    optimizer: str
    lr: float
    weight_decay: float
    momentum: float
    global_pos_weight: float
    patch_pos_weight: float
    loss_global_w: float
    loss_mask_w: float
    branch_arch: str
    spatial_branch_mode: str
    spatial_stage: str
    residual_alpha: float
    profile_steps: int
    warmup_steps: int
    log_every: int
    save_every: int
    skip_final_layer: bool
    final_layer_type: str
    saga_batch_size: int
    saga_workers: int
    saga_prefetch_factor: int
    saga_step_size: float
    saga_lam: float
    saga_n_iters: int
    saga_verbose_every: int
    dense_lr: float
    dense_n_iters: int
    feature_storage_dtype: str
    saga_table_device: str
    vlg_init_path: str
    vlg_concepts_path: str
    freeze_global_head: bool
    scheduler: str
    print_config: bool
    # Old ImageNet checkpoints did not store this field and used avg pooling.
    residual_spatial_pooling: str = "avg"
    learn_spatial_residual_scale: bool = False
    loss_spatial_presence_w: float = 0.0
    loss_global_spatial_align_w: float = 0.0
    eval_every: int = 1
    feature_batch_size: int = 256
    feature_workers: int = 4
    feature_prefetch_factor: int = 2
