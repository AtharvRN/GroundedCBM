from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np


PREPROCESS_RESIZE_SIZE = 256


def box_to_original_pixels(
    box: Sequence[float],
    image_size: Tuple[int, int],
) -> Optional[Tuple[float, float, float, float]]:
    """Convert xyxy boxes from pixel or normalized coordinates to original pixels."""
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    x1, y1, x2, y2 = [float(value) for value in box]
    width, height = image_size
    if width <= 0 or height <= 0:
        return None
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) > 1.5:
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
    else:
        x1, x2 = sorted((x1 * width, x2 * width))
        y1, y2 = sorted((y1 * height, y2 * height))
    x1, x2 = float(np.clip(x1, 0.0, width)), float(np.clip(x2, 0.0, width))
    y1, y2 = float(np.clip(y1, 0.0, height)), float(np.clip(y2, 0.0, height))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def normalize_box(
    box: Sequence[float],
    image_size: Tuple[int, int],
) -> Optional[Tuple[float, float, float, float]]:
    """Normalize a box in the original image coordinate frame."""
    pixel_box = box_to_original_pixels(box, image_size=image_size)
    if pixel_box is None:
        return None
    width, height = image_size
    x1, y1, x2, y2 = pixel_box
    return x1 / width, y1 / height, x2 / width, y2 / height


def resize_short_edge_size(
    image_size: Tuple[int, int],
    resize_size: int = PREPROCESS_RESIZE_SIZE,
) -> Tuple[int, int]:
    """Match torchvision.transforms.Resize(int) output size."""
    width, height = image_size
    if width <= 0 or height <= 0:
        return int(resize_size), int(resize_size)
    if width == height:
        return int(resize_size), int(resize_size)
    if width < height:
        return int(resize_size), int(resize_size * height / width)
    return int(resize_size * width / height), int(resize_size)


def transform_box_for_resize_center_crop(
    box: Sequence[float],
    image_size: Tuple[int, int],
    input_size: Optional[int],
    resize_size: int = PREPROCESS_RESIZE_SIZE,
) -> Optional[Tuple[float, float, float, float]]:
    """Apply Resize(shorter side) + CenterCrop geometry before normalizing."""
    pixel_box = box_to_original_pixels(box, image_size=image_size)
    if pixel_box is None:
        return None
    crop_size = int(input_size or resize_size)
    width, height = image_size
    resized_width, resized_height = resize_short_edge_size(image_size, resize_size=resize_size)
    scale_x = resized_width / float(width)
    scale_y = resized_height / float(height)
    x1, y1, x2, y2 = pixel_box
    x1 *= scale_x
    x2 *= scale_x
    y1 *= scale_y
    y2 *= scale_y

    crop_left = max(int(round((resized_width - crop_size) / 2.0)), 0)
    crop_top = max(int(round((resized_height - crop_size) / 2.0)), 0)
    x1 -= crop_left
    x2 -= crop_left
    y1 -= crop_top
    y2 -= crop_top

    x1 = float(np.clip(x1, 0.0, crop_size))
    x2 = float(np.clip(x2, 0.0, crop_size))
    y1 = float(np.clip(y1, 0.0, crop_size))
    y2 = float(np.clip(y2, 0.0, crop_size))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1 / crop_size, y1 / crop_size, x2 / crop_size, y2 / crop_size


def transform_box(
    box: Sequence[float],
    image_size: Tuple[int, int],
    *,
    transform: str = "original",
    input_size: Optional[int] = None,
    resize_size: int = PREPROCESS_RESIZE_SIZE,
) -> Optional[Tuple[float, float, float, float]]:
    transform = str(transform or "original").lower()
    if transform in {"original", "none", "identity"}:
        return normalize_box(box, image_size=image_size)
    if transform in {"resize_center_crop", "resize_short_edge_center_crop"}:
        return transform_box_for_resize_center_crop(
            box,
            image_size=image_size,
            input_size=input_size,
            resize_size=resize_size,
        )
    raise ValueError(f"Unsupported box transform: {transform}")


def rasterize_box_iou(
    box: Sequence[float],
    image_size: Tuple[int, int],
    *,
    mask_h: int,
    mask_w: int,
    iou_thresh: float,
    transform: str = "original",
    input_size: Optional[int] = None,
    resize_size: int = PREPROCESS_RESIZE_SIZE,
) -> Optional[np.ndarray]:
    norm = transform_box(
        box,
        image_size=image_size,
        transform=transform,
        input_size=input_size,
        resize_size=resize_size,
    )
    if norm is None:
        return None
    x1, y1, x2, y2 = norm
    box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if box_area <= 0.0:
        return None
    mask = np.zeros((mask_h, mask_w), dtype=np.float32)
    patch_area = 1.0 / float(mask_h * mask_w)
    for row in range(mask_h):
        py1 = row / float(mask_h)
        py2 = (row + 1) / float(mask_h)
        for col in range(mask_w):
            px1 = col / float(mask_w)
            px2 = (col + 1) / float(mask_w)
            ix1 = max(px1, x1)
            iy1 = max(py1, y1)
            ix2 = min(px2, x2)
            iy2 = min(py2, y2)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            if inter <= 0.0:
                continue
            union = patch_area + box_area - inter
            if union > 0.0 and inter / union > iou_thresh:
                mask[row, col] = 1.0
    return mask


def rasterize_box_soft_occupancy(
    box: Sequence[float],
    image_size: Tuple[int, int],
    *,
    mask_h: int,
    mask_w: int,
    transform: str = "original",
    input_size: Optional[int] = None,
    resize_size: int = PREPROCESS_RESIZE_SIZE,
) -> Optional[np.ndarray]:
    norm = transform_box(
        box,
        image_size=image_size,
        transform=transform,
        input_size=input_size,
        resize_size=resize_size,
    )
    if norm is None:
        return None
    x1, y1, x2, y2 = norm
    box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if box_area <= 0.0:
        return None
    mask = np.zeros((mask_h, mask_w), dtype=np.float32)
    patch_area = 1.0 / float(mask_h * mask_w)
    for row in range(mask_h):
        py1 = row / float(mask_h)
        py2 = (row + 1) / float(mask_h)
        for col in range(mask_w):
            px1 = col / float(mask_w)
            px2 = (col + 1) / float(mask_w)
            ix1 = max(px1, x1)
            iy1 = max(py1, y1)
            ix2 = min(px2, x2)
            iy2 = min(py2, y2)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            if inter > 0.0:
                mask[row, col] = float(np.clip(inter / patch_area, 0.0, 1.0))
    return mask


def rasterize_box_target(
    box: Sequence[float],
    image_size: Tuple[int, int],
    *,
    target_mode: str,
    mask_h: int,
    mask_w: int,
    iou_thresh: float = 0.5,
    transform: str = "original",
    input_size: Optional[int] = None,
    resize_size: int = PREPROCESS_RESIZE_SIZE,
) -> Optional[np.ndarray]:
    mode = str(target_mode).lower()
    if mode == "hard_iou":
        return rasterize_box_iou(
            box,
            image_size=image_size,
            mask_h=mask_h,
            mask_w=mask_w,
            iou_thresh=iou_thresh,
            transform=transform,
            input_size=input_size,
            resize_size=resize_size,
        )
    if mode == "soft_box":
        return rasterize_box_soft_occupancy(
            box,
            image_size=image_size,
            mask_h=mask_h,
            mask_w=mask_w,
            transform=transform,
            input_size=input_size,
            resize_size=resize_size,
        )
    raise ValueError(f"Unsupported spatial target mode: {target_mode}")
