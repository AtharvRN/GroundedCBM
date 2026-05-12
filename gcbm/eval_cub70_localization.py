"""
CUB70 Part Localization: SAVLG vs SALF using human-annotated segmentation masks.

Protocol:
1. For each CUB70 test image with part masks:
   - Load GT segmentation mask for each part (beak, wing, body, tail)
   - Get model's spatial concept maps for concepts mapped to that part
   - Select the concept with highest activation (activation-selected)
   - Resize concept map to image size, normalize, threshold
   - Compute IoU between thresholded map and GT mask
2. Report per-part and overall mean IoU, dice, precision, recall.

Both SAVLG and SALF produce native spatial maps — no Grad-CAM needed.
GT masks from CUB70 are completely independent of both models' training signals.
"""
import argparse, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from collections import defaultdict
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

# CUB70 coarse part mapping (GPT part_group → CUB70 mask suffix)
PART_GROUP_TO_CUB70 = {
    "beak":   ["beak"],
    "wing":   ["left_wing", "right_wing"],
    "tail":   ["tail"],
    "back":   ["body"],
    "breast": ["body"],
    "belly":  ["body"],
    "eye":    ["left_eye", "right_eye"],
    "crown":  ["head"],
    "throat": ["neck"],
    "leg":    ["left_leg", "right_leg"],
}

EASY_PARTS = {"beak", "wing", "tail", "body"}
# Map part_group to CUB70 mask name for easy parts
EASY_GROUP_TO_CUB70 = {
    "beak":   ["beak"],
    "wing":   ["left_wing", "right_wing"],
    "tail":   ["tail"],
    "back":   ["body"],
    "breast": ["body"],
    "belly":  ["body"],
}


def format_concept(s: str) -> str:
    s = s.lower()
    for token in ["-", ",", ".", "(", ")"]:
        s = s.replace(token, " ")
    if s.startswith("a "):
        s = s[2:]
    elif s.startswith("an "):
        s = s[3:]
    return " ".join(s.split())


def canonicalize_concept_label(s: str) -> str:
    return format_concept(s)


def load_concept_part_mapping(mapping_path, model_concepts):
    """Returns {concept_idx: part_group} for concepts in the model's concept list."""
    with open(mapping_path, "r", encoding="utf-8") as handle:
        d = json.load(handle)
    concept_to_idx = {canonicalize_concept_label(c): i for i, c in enumerate(model_concepts)}
    mapping = {}  # concept_idx -> part_group
    for entry in d["mappings"]:
        if not entry["keep"] or not entry["part_group"]:
            continue
        cname = canonicalize_concept_label(entry["concept"])
        if cname in concept_to_idx:
            mapping[concept_to_idx[cname]] = entry["part_group"]
    return mapping


def find_cub70_images(cub70_root, cub_root, split="test"):
    """
    Match CUB70 annotated images to CUB dataset.
    Returns list of {cub_class_id, image_stem, image_path, masks: {part: mask_path}}
    """
    from pathlib import Path
    # Load CUB train/test split
    split_file = os.path.join(cub_root, "train_test_split.txt")
    images_file = os.path.join(cub_root, "images.txt")
    test_ids = set()
    with open(split_file, "r", encoding="utf-8") as handle:
        for line in handle:
            img_id, is_train = line.strip().split()
            if (split == "test" and is_train == "0") or (split == "train" and is_train == "1"):
                test_ids.add(int(img_id))

    # Build image_id -> (class_id, image_stem)
    id_to_info = {}
    with open(images_file, "r", encoding="utf-8") as handle:
        for line in handle:
            img_id, rel_path = line.strip().split(" ", 1)
            class_folder = rel_path.split("/")[0]
            class_id = class_folder.split(".")[0]  # "001" from "001.Black_footed_Albatross"
            stem = os.path.splitext(rel_path.split("/")[1])[0]
            id_to_info[int(img_id)] = (class_id, stem, rel_path)

    # Find CUB70 mask directories
    mask_root = os.path.join(cub70_root, "AnnotationMasksPerclass")
    available_classes = set(os.listdir(mask_root)) if os.path.isdir(mask_root) else set()

    results = []
    for img_id in sorted(test_ids):
        if img_id not in id_to_info:
            continue
        class_id, stem, rel_path = id_to_info[img_id]
        class_id_int = str(int(class_id))  # "001" -> "1"
        if class_id_int not in available_classes:
            continue
        # Check which part masks exist
        mask_dir = os.path.join(mask_root, class_id_int)
        masks = {}
        for part in ["beak", "body", "head", "neck", "tail",
                      "left_wing", "right_wing", "left_leg", "right_leg",
                      "left_eye", "right_eye"]:
            mask_path = os.path.join(mask_dir, f"{stem}_{part}.png")
            if os.path.exists(mask_path):
                masks[part] = mask_path
        if masks:
            image_path = os.path.join(cub_root, "images", rel_path)
            results.append({
                "img_id": img_id,
                "class_id": class_id,
                "stem": stem,
                "image_path": image_path,
                "masks": masks,
            })
    return results


def load_gt_mask(mask_path, target_size=None):
    """Load binary mask from PNG, optionally resize."""
    mask = np.array(Image.open(mask_path).convert("L"))
    mask = (mask > 127).astype(np.float32)
    return mask


def get_spatial_maps_savlg(load_dir, image_paths, device):
    """Extract spatial concept maps from SAVLG checkpoint."""
    from methods.savlg import (
        create_savlg_splits, build_savlg_concept_layer,
        forward_savlg_backbone, forward_savlg_concept_layer,
    )
    from methods.salf import SpatialBackbone
    with open(os.path.join(load_dir, "args.txt")) as f:
        args = argparse.Namespace(**json.load(f))
    args.device = device
    if getattr(args, "skip_test_eval", False):
        print(
            "[CUB70 loc] overriding saved skip_test_eval=True to force evaluation on dataset_val",
            flush=True,
        )
        args.skip_test_eval = False
    with open(os.path.join(load_dir, "concepts.txt")) as f:
        concepts = f.read().strip().split("\n")

    _, _, _, _, test_ds, backbone = create_savlg_splits(args)
    cl = build_savlg_concept_layer(args, backbone, len(concepts))
    cl.load_state_dict(torch.load(os.path.join(load_dir, "concept_layer.pt"), map_location=device))
    cl.eval(); backbone.eval()

    return backbone, cl, args, concepts


def get_spatial_maps_salf(load_dir, device):
    """Load SALF concept layer."""
    import torch.nn as nn
    from methods.salf import SpatialBackbone
    with open(os.path.join(load_dir, "args.txt")) as f:
        args = argparse.Namespace(**json.load(f))
    args.device = device
    with open(os.path.join(load_dir, "concepts.txt")) as f:
        concepts = f.read().strip().split("\n")

    backbone = SpatialBackbone(args.backbone, device=device)
    backbone.eval()
    sd = torch.load(os.path.join(load_dir, "concept_layer.pt"), map_location=device)
    w = sd["weight"]
    cl = nn.Conv2d(w.shape[1], w.shape[0], kernel_size=1, bias="bias" in sd).to(device)
    cl.load_state_dict(sd)
    cl.eval()

    return backbone, cl, args, concepts


def get_gradcam_model(load_dir, device):
    """Load VLG or LF model for Grad-CAM."""
    from model.cbm import Backbone, ConceptLayer
    import data.utils as du
    with open(os.path.join(load_dir, "args.txt")) as f:
        args = argparse.Namespace(**json.load(f))
    args.device = device
    with open(os.path.join(load_dir, "concepts.txt")) as f:
        concepts = f.read().strip().split("\n")
    feature_layer = str(getattr(args, "feature_layer", "features.final_pool"))
    backbone = Backbone(args.backbone, feature_layer, device)
    backbone.eval()
    # Load concept layer
    cbl_path = os.path.join(load_dir, "cbl.pt")
    wc_path = os.path.join(load_dir, "W_c.pt")
    cl_path = os.path.join(load_dir, "concept_layer.pt")
    if os.path.exists(cbl_path):
        concept_layer = ConceptLayer.from_pretrained(load_dir, device=device)
    elif os.path.exists(wc_path):
        import torch.nn as nn
        W_c = torch.load(wc_path, map_location=device)
        concept_layer = nn.Linear(W_c.shape[1], W_c.shape[0], bias=False).to(device)
        concept_layer.load_state_dict({"weight": W_c})
    elif os.path.exists(cl_path):
        concept_layer = ConceptLayer.from_pretrained(load_dir, device=device)
    else:
        raise FileNotFoundError(f"No concept layer found in {load_dir}")
    concept_layer.eval()
    test_ds = du.get_data(f"{args.dataset}_val", preprocess=backbone.preprocess)
    return backbone, concept_layer, args, concepts, test_ds


def compute_gradcam_maps(backbone, concept_layer, image_tensor, device, concept_idxs=None):
    """Compute Grad-CAM spatial maps for specified concepts. Returns [C, H, W] and logits [C]."""
    activations = {}
    def fwd_hook(m, i, o):
        activations["feat"] = o
    target = backbone.backbone
    hook_handle = None
    for name, module in reversed(list(target.named_modules())):
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Sequential)):
            if "layer4" in name or "stage4" in name:
                hook_handle = module.register_forward_hook(fwd_hook)
                break
    if hook_handle is None:
        return None, None
    img = image_tensor.unsqueeze(0).to(device)
    backbone.zero_grad(); concept_layer.zero_grad()
    feats = backbone(img)
    logits = concept_layer(feats)
    n_concepts = logits.shape[1]
    if concept_idxs is None:
        concept_idxs = list(range(n_concepts))
    act = activations["feat"].detach()
    h_feat, w_feat = act.shape[2], act.shape[3]
    all_maps = torch.zeros(n_concepts, h_feat, w_feat, device=device)
    for cidx in concept_idxs:
        backbone.zero_grad(); concept_layer.zero_grad()
        logits[0, cidx].backward(retain_graph=True)
        gradients = {}
        for name, module in reversed(list(target.named_modules())):
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Sequential)):
                if "layer4" in name or "stage4" in name:
                    if module.weight is not None and module.weight.grad is not None:
                        pass
                    break
        # Re-register for gradient
        pass
    hook_handle.remove()
    # Simpler approach: use full backward hook
    gradients_store = {}
    def bwd_hook(m, gi, go):
        gradients_store["feat"] = go[0]
    for name, module in reversed(list(target.named_modules())):
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Sequential)):
            if "layer4" in name or "stage4" in name:
                h1 = module.register_forward_hook(fwd_hook)
                h2 = module.register_full_backward_hook(bwd_hook)
                break
    backbone.zero_grad(); concept_layer.zero_grad()
    feats = backbone(img)
    logits = concept_layer(feats)
    act = activations["feat"].detach()
    for cidx in concept_idxs:
        backbone.zero_grad(); concept_layer.zero_grad()
        if "feat" in gradients_store:
            del gradients_store["feat"]
        logits[0, cidx].backward(retain_graph=True)
        if "feat" in gradients_store:
            grad = gradients_store["feat"].detach()
            weights = grad.mean(dim=[2, 3], keepdim=True)
            cam = torch.relu((weights * act).sum(dim=1)).squeeze()
            all_maps[cidx] = cam
    h1.remove(); h2.remove()
    return all_maps, logits.detach().squeeze(0)


def extract_concept_maps(model_type, backbone, concept_layer, args, image_tensor):
    """
    Returns spatial_maps [C, H, W] and final_logits [C] for one image.
    """
    with torch.no_grad():
        if model_type == "savlg":
            from methods.savlg import (
                forward_savlg_backbone, forward_savlg_concept_layer,
                compute_savlg_concept_logits,
            )
            feats = forward_savlg_backbone(backbone, image_tensor.unsqueeze(0), args)
            g_out, s_maps = forward_savlg_concept_layer(concept_layer, feats)
            _, _, final = compute_savlg_concept_logits(g_out, s_maps, args)
            return s_maps.squeeze(0), final.squeeze(0)  # [C, H, W], [C]
        elif model_type == "salf":
            feats = backbone(image_tensor.unsqueeze(0).to(args.device))
            s_maps = concept_layer(feats)
            # Pool for logits
            logits = s_maps.flatten(2).max(dim=2).values.squeeze(0)
            return s_maps.squeeze(0), logits  # [C, H, W], [C]


def normalize_map(concept_map):
    """Z-score then minmax normalize a single concept map to [0,1]."""
    m = concept_map.float()
    mu = m.mean(); std = m.std()
    if std > 1e-6:
        m = (m - mu) / std
    m = m - m.min()
    mx = m.max()
    if mx > 1e-6:
        m = m / mx
    return m


def compute_iou_metrics(pred_mask, gt_mask):
    """Compute IoU, dice, precision, recall between binary masks."""
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    intersection = (pred & gt).sum()
    union = (pred | gt).sum()
    iou = intersection / (union + 1e-9)
    dice = 2 * intersection / (pred.sum() + gt.sum() + 1e-9)
    precision = intersection / (pred.sum() + 1e-9)
    recall = intersection / (gt.sum() + 1e-9)
    return {"iou": float(iou), "dice": float(dice),
            "precision": float(precision), "recall": float(recall)}


def evaluate_model(model_type, load_dir, cub70_images, concept_part_map, device,
                   thresholds=(0.3, 0.5, 0.7, 0.9)):
    """
    Evaluate one model on CUB70 part localization.
    Returns per-part and overall metrics.
    """
    import data.utils as du

    if model_type == "savlg":
        backbone, concept_layer, args, concepts = get_spatial_maps_savlg(load_dir, [], device)
        from methods.savlg import create_savlg_splits
        if getattr(args, "skip_test_eval", False):
            args.skip_test_eval = False
        _, _, _, _, test_ds, _ = create_savlg_splits(args)
    elif model_type == "salf":
        backbone, concept_layer, args, concepts = get_spatial_maps_salf(load_dir, device)
        test_ds = du.get_data(f"{args.dataset}_val", preprocess=backbone.preprocess)
    elif model_type in ("vlg", "lf"):
        backbone, concept_layer, args, concepts, test_ds = get_gradcam_model(load_dir, device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Build part_group -> list of concept indices
    part_to_concepts = defaultdict(list)
    for cidx, part_group in concept_part_map.items():
        part_to_concepts[part_group].append(cidx)

    # Build dataset_index -> cub70_image mapping
    # Match by image stem in the ImageFolder
    # Navigate through wrappers to find the base ImageFolder
    raw_ds = test_ds
    while hasattr(raw_ds, 'dataset'):
        raw_ds = raw_ds.dataset
    while hasattr(raw_ds, 'base_dataset'):
        raw_ds = raw_ds.base_dataset
    img_list = raw_ds.imgs if hasattr(raw_ds, 'imgs') else raw_ds.samples

    # Dict-based matching (O(N) instead of O(N×M))
    stem_to_cub70 = {img["stem"]: img for img in cub70_images}
    idx_to_cub70 = {}
    for ds_idx, (full_path, _) in enumerate(img_list):
        stem = os.path.splitext(os.path.basename(full_path))[0]
        if stem in stem_to_cub70:
            idx_to_cub70[ds_idx] = stem_to_cub70[stem]

    matched_indices = sorted(idx_to_cub70.keys())
    print(f"  Matched {len(matched_indices)} images to CUB70 annotations", flush=True)

    # Pre-load all GT masks into memory (fast — PNG files are small)
    print(f"  Pre-loading GT masks...", flush=True)
    gt_cache = {}  # (ds_idx, part_group) -> gt_mask numpy array
    for ds_idx in matched_indices:
        cub70_img = idx_to_cub70[ds_idx]
        for part_group, cub70_parts in EASY_GROUP_TO_CUB70.items():
            if not part_to_concepts.get(part_group):
                continue
            gt_mask = None
            for cub70_part in cub70_parts:
                if cub70_part in cub70_img["masks"]:
                    m = load_gt_mask(cub70_img["masks"][cub70_part])
                    gt_mask = m if gt_mask is None else np.maximum(gt_mask, m)
            if gt_mask is not None and gt_mask.sum() >= 1:
                report_part = "body" if part_group in ("back", "breast", "belly") else part_group
                gt_cache[(ds_idx, report_part, part_group)] = gt_mask
    print(f"  Cached {len(gt_cache)} (image, part) GT masks", flush=True)

    # Batched inference
    BATCH_SIZE = 64
    all_results = defaultdict(lambda: defaultdict(list))
    oracle_ap_data = defaultdict(list)  # part -> list of (score, iou_at_best_thr)
    n_done = 0

    for batch_start in range(0, len(matched_indices), BATCH_SIZE):
        batch_idxs = matched_indices[batch_start:batch_start + BATCH_SIZE]
        batch_tensors = torch.stack([test_ds[i][0] for i in batch_idxs]).to(device)

        if model_type in ("vlg", "lf"):
            # Grad-CAM: process one image at a time, only for part-relevant concepts
            all_part_cidxs = set()
            for pg in part_to_concepts.values():
                all_part_cidxs.update(pg)
            all_part_cidxs = sorted(all_part_cidxs)
            batch_spatial_list = []
            batch_logits_list = []
            for img_tensor in batch_tensors:
                s_maps, logits = compute_gradcam_maps(
                    backbone, concept_layer, img_tensor, device,
                    concept_idxs=all_part_cidxs,
                )
                batch_spatial_list.append(s_maps)
                batch_logits_list.append(logits)
            batch_spatial = torch.stack(batch_spatial_list)
            batch_logits = torch.stack(batch_logits_list)
        else:
            with torch.no_grad():
                if model_type == "savlg":
                    from methods.savlg import (
                        forward_savlg_backbone, forward_savlg_concept_layer,
                        compute_savlg_concept_logits,
                    )
                    feats = forward_savlg_backbone(backbone, batch_tensors, args)
                    g_out, s_maps = forward_savlg_concept_layer(concept_layer, feats)
                    _, _, final = compute_savlg_concept_logits(g_out, s_maps, args)
                    batch_spatial = s_maps  # [B, C, H, W]
                    batch_logits = final    # [B, C]
                elif model_type == "salf":
                    feats = backbone(batch_tensors)
                    s_maps = concept_layer(feats)
                    batch_spatial = s_maps
                    batch_logits = s_maps.flatten(2).max(dim=2).values

        # Process each image in batch
        for i, ds_idx in enumerate(batch_idxs):
            spatial_maps = batch_spatial[i]  # [C, H, W]
            logits = batch_logits[i]         # [C]

            for part_group, cub70_parts in EASY_GROUP_TO_CUB70.items():
                concept_idxs = part_to_concepts.get(part_group, [])
                if not concept_idxs:
                    continue
                report_part = "body" if part_group in ("back", "breast", "belly") else part_group
                cache_key = (ds_idx, report_part, part_group)
                if cache_key not in gt_cache:
                    continue
                gt_mask = gt_cache[cache_key]

                h, w = gt_mask.shape

                # --- Activation-selected ---
                activations = logits[concept_idxs]
                best_cidx = concept_idxs[activations.argmax().item()]
                cmap = spatial_maps[best_cidx]
                cmap_norm = normalize_map(cmap)
                cmap_resized = F.interpolate(
                    cmap_norm.unsqueeze(0).unsqueeze(0), size=(h, w),
                    mode="bilinear", align_corners=False
                ).squeeze().cpu().numpy()

                for thr in thresholds:
                    pred_mask = (cmap_resized >= thr).astype(np.float32)
                    metrics = compute_iou_metrics(pred_mask, gt_mask)
                    all_results[report_part][thr].append(metrics)
                    all_results["overall"][thr].append(metrics)

                # --- Oracle: pick concept with best IoU against GT ---
                best_oracle_iou = -1
                best_oracle_metrics = None
                best_oracle_thr = None
                best_oracle_cidx = None
                for cidx in concept_idxs:
                    cm = normalize_map(spatial_maps[cidx])
                    cm_resized = F.interpolate(
                        cm.unsqueeze(0).unsqueeze(0), size=(h, w),
                        mode="bilinear", align_corners=False
                    ).squeeze().cpu().numpy()
                    for thr in thresholds:
                        pm = (cm_resized >= thr).astype(np.float32)
                        m = compute_iou_metrics(pm, gt_mask)
                        if m["iou"] > best_oracle_iou:
                            best_oracle_iou = m["iou"]
                            best_oracle_metrics = m
                            best_oracle_thr = thr
                            best_oracle_cidx = cidx
                if best_oracle_metrics is not None:
                    all_results[f"{report_part}_oracle"][best_oracle_thr].append(best_oracle_metrics)
                    all_results["overall_oracle"][best_oracle_thr].append(best_oracle_metrics)
                    oracle_score = float(logits[best_oracle_cidx].item())
                    oracle_ap_data[report_part].append((oracle_score, best_oracle_iou))
                    oracle_ap_data["overall"].append((oracle_score, best_oracle_iou))

        n_done += len(batch_idxs)
        if n_done % 256 == 0 or n_done == len(matched_indices):
            print(f"  [{n_done}/{len(matched_indices)}] images processed", flush=True)

    # Compute oracle mAP at various IoU thresholds
    oracle_map = {}
    for part, entries in oracle_ap_data.items():
        scores = np.array([s for s, _ in entries])
        ious = np.array([iou for _, iou in entries])
        order = np.argsort(-scores)
        part_ap = {}
        for tau in thresholds:
            labels_sorted = (ious[order] >= tau).astype(np.int32)
            num_pos = int(labels_sorted.sum())
            if num_pos == 0:
                part_ap[str(tau)] = 0.0
                continue
            tp = 0.0
            precision_sum = 0.0
            for rank, label in enumerate(labels_sorted, start=1):
                if label == 1:
                    tp += 1.0
                    precision_sum += tp / float(rank)
            part_ap[str(tau)] = float(precision_sum / num_pos)
        oracle_map[part] = part_ap

    return all_results, oracle_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gcbm_path", default=None, help="Path to a G-CBM/SAVLG run directory. Alias for --savlg_path.")
    parser.add_argument("--savlg_path", default=None)
    parser.add_argument("--salf_path", default=None)
    parser.add_argument("--vlg_path", default=None)
    parser.add_argument("--lf_path", default=None)
    parser.add_argument("--cub70_root", default="datasets/CUB70-PartSegmentationDataset")
    parser.add_argument("--cub_root", default="datasets/CUB")
    parser.add_argument("--mapping_json", default="concept_files/cub_concept_part_mapping.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="results/cub70_savlg_vs_salf.json")
    parser.add_argument("--max_images", type=int, default=None)
    cli = parser.parse_args()

    # Check CUB root has images.txt and train_test_split.txt
    cub_root = cli.cub_root
    if not os.path.exists(os.path.join(cub_root, "images.txt")):
        # Try parent
        for candidate in ["datasets/CUB_200_2011",
                          os.path.join(cub_root, "CUB_200_2011")]:
            if os.path.exists(os.path.join(candidate, "images.txt")):
                cub_root = candidate; break

    print("Finding CUB70 annotated test images...", flush=True)
    cub70_images = find_cub70_images(cli.cub70_root, cub_root, split="test")
    if cli.max_images:
        cub70_images = cub70_images[:cli.max_images]
    print(f"  Found {len(cub70_images)} images with CUB70 masks", flush=True)

    results = {}
    models_to_eval = []
    savlg_path = cli.gcbm_path or cli.savlg_path
    if savlg_path:
        models_to_eval.append(("G-CBM", "savlg", savlg_path))
    if cli.salf_path:
        models_to_eval.append(("SALF", "salf", cli.salf_path))
    if cli.vlg_path:
        models_to_eval.append(("VLG", "vlg", cli.vlg_path))
    if cli.lf_path:
        models_to_eval.append(("LF", "lf", cli.lf_path))
    if not models_to_eval:
        print("No model paths provided. Use --savlg_path, --salf_path, --vlg_path, or --lf_path.")
        return
    for model_name, model_type, load_dir in models_to_eval:
        print(f"\n=== {model_name} ===", flush=True)
        with open(os.path.join(load_dir, "concepts.txt")) as f:
            concepts = f.read().strip().split("\n")
        concept_part_map = load_concept_part_mapping(cli.mapping_json, concepts)
        print(f"  {len(concept_part_map)} concepts mapped to parts", flush=True)

        part_results, oracle_map = evaluate_model(model_type, load_dir, cub70_images,
                                                    concept_part_map, cli.device)

        # Summarize
        model_summary = {}
        for part in ["beak", "wing", "tail", "body", "overall",
                      "beak_oracle", "wing_oracle", "tail_oracle", "body_oracle", "overall_oracle"]:
            if part not in part_results:
                continue
            part_summary = {}
            for thr, metrics_list in sorted(part_results[part].items()):
                if not metrics_list:
                    continue
                mean_iou  = np.mean([m["iou"]  for m in metrics_list])
                mean_dice = np.mean([m["dice"] for m in metrics_list])
                mean_prec = np.mean([m["precision"] for m in metrics_list])
                mean_rec  = np.mean([m["recall"] for m in metrics_list])
                part_summary[str(thr)] = {
                    "iou": round(mean_iou, 4), "dice": round(mean_dice, 4),
                    "precision": round(mean_prec, 4), "recall": round(mean_rec, 4),
                    "n_instances": len(metrics_list),
                }
            if part_summary:
                best_thr = max(part_summary, key=lambda t: part_summary[t]["iou"])
                model_summary[part] = {
                    "thresholds": part_summary,
                    "best_iou": part_summary[best_thr]["iou"],
                    "best_thr": best_thr,
                }
        model_summary["oracle_mAP"] = oracle_map
        results[model_name] = model_summary

        # Print
        print(f"\n  {'Part':<18} {'Best IoU':>10} {'Thr':>6} {'Dice':>8} {'N':>6}", flush=True)
        print(f"  {'-'*52}", flush=True)
        for part in ["beak", "wing", "tail", "body", "overall",
                      "beak_oracle", "wing_oracle", "tail_oracle", "body_oracle", "overall_oracle"]:
            if part in model_summary:
                s = model_summary[part]
                bt = s["best_thr"]
                print(f"  {part:<18} {s['best_iou']:>10.4f} {bt:>6} "
                      f"{s['thresholds'][bt]['dice']:>8.4f} "
                      f"{s['thresholds'][bt]['n_instances']:>6}", flush=True)

        print(f"\n  Oracle mAP:", flush=True)
        for part in ["beak", "wing", "tail", "body", "overall"]:
            if part in oracle_map:
                ap_strs = "  ".join(f"@{t}={v:.4f}" for t, v in sorted(oracle_map[part].items()))
                print(f"  {part:<10} {ap_strs}", flush=True)

    # Final comparison
    print("\n" + "="*60, flush=True)
    print("G-CBM vs SALF — CUB70 Part Localization (IoU)", flush=True)
    print(f"{'Part':<10} {'G-CBM IoU':>12} {'SALF IoU':>12} {'Delta':>10}", flush=True)
    print("-"*46, flush=True)
    for part in ["beak", "wing", "tail", "body", "overall"]:
        savlg_iou = results.get("G-CBM", {}).get(part, {}).get("best_iou", 0)
        salf_iou  = results.get("SALF", {}).get(part, {}).get("best_iou", 0)
        delta = savlg_iou - salf_iou
        print(f"{part:<10} {savlg_iou:>12.4f} {salf_iou:>12.4f} {delta:>+10.4f}", flush=True)

    os.makedirs(os.path.dirname(cli.output) or ".", exist_ok=True)
    with open(cli.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {cli.output}", flush=True)


if __name__ == "__main__":
    main()
