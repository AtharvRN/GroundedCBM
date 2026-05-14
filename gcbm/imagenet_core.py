from __future__ import annotations

# Compatibility layer: implementation lives in focused modules.
from gcbm.imagenet_config import Config
from gcbm.runtime import (
    amp_dtype,
    autocast_context,
    configure_runtime,
    cuda_peak_stats_mb,
    reset_cuda_peak_stats_if_needed,
)
from gcbm.imagenet_targets import (
    IMAGENET_LABEL_ALIASES,
    PREPROCESS_RESIZE_SIZE,
    PrecomputedTargetStore,
    annotation_entries,
    batch_targets_to_device,
    build_gdino_target_sample,
    build_gdino_targets,
    canonicalize_concept_label,
    format_concept,
    get_image_size,
    load_concepts,
    load_run_concepts,
    normalize_box,
    precompute_target_store,
    rasterize_box_iou,
    rasterize_box_soft_occupancy,
    rasterize_box_target,
    resize_short_edge_size,
    transform_box_for_model_input,
    transform_box_for_resize_center_crop,
)
from gcbm.imagenet_data import (
    DatasetView,
    SafeImageFolderWithAnnotations,
    apply_count_concept_filter,
    build_loader,
    collate_batch,
    count_concept_targets,
    refresh_dataset_concepts,
    select_subset_indices,
    split_train_val,
    unwrap_dataset_view,
)
from gcbm.training_utils import (
    build_run_dir,
    make_optimizer,
    make_scheduler,
    prepare_images,
    save_checkpoint,
)
