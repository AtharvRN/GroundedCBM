import json
import os
import glob
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

# Optional dependency: most training/eval paths do not need matplotlib.
# Keep this import soft so lightweight pods can run without it.
try:
    from matplotlib import pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
    plt = None  # type: ignore
import torch
from pytorchcv.model_provider import get_model as ptcv_get_model
from torchvision import datasets, models, transforms
from tqdm import tqdm
from loguru import logger
import data.data_lp as data_lp
import clip
from PIL import Image, ImageFile, UnidentifiedImageError

# get from the environment variable
DATASET_FOLDER = os.environ.get("DATASET_FOLDER", "datasets")


def get_dataset_roots() -> Dict[str, str]:
    imagenet_root = os.environ.get(
        "IMAGENET_DATASET_ROOT", f"{DATASET_FOLDER}/imagenet/ILSVRC/Data/CLS-LOC"
    )
    cub_root = os.environ.get("CUB_DATASET_ROOT", f"{DATASET_FOLDER}/CUB")
    roots = {
        "imagenet_train": os.environ.get(
            "IMAGENET_TRAIN_ROOT", f"{imagenet_root}/train"
        ),
        "imagenet_val": os.environ.get(
            "IMAGENET_VAL_ROOT", f"{imagenet_root}/ImageNet_val"
        ),
        "cub_train": os.environ.get("CUB_TRAIN_ROOT", f"{cub_root}/train"),
        "cub_val": os.environ.get("CUB_VAL_ROOT", f"{cub_root}/test"),
    }
    broden_root = os.environ.get("BRODEN_ROOT")
    if broden_root:
        roots["broden"] = broden_root
    return roots


DATASET_ROOTS = get_dataset_roots()

LABEL_FILES = {
    "places365": "concept_files/categories_places365_clean.txt",
    "imagenet": "concept_files/imagenet_classes.txt",
    "cifar10": "concept_files/cifar10_classes.txt",
    "cifar100": "concept_files/cifar100_classes.txt",
    "cub": "concept_files/cub_classes.txt",
    "food": "concept_files/food_classes.txt",
    "flower": "concept_files/flower_classes.txt",
    "aircraft": "concept_files/aircraft_classes.txt",
    "dtd": "concept_files/dtd_classes.txt",
}

BACKBONE_ENCODING_DIMENSION = {
    "resnet18_cub": 512,
    "resnet50_cub": 2048,
    "resnet50_cub_mm": 2048,
    "clip_RN50": 1024,
    "clip_RN50_penultimate": 2048,
    "resnet50": 2048,
}

BACKBONE_VISUALIZATION_TARGET_LAYER = {
    "resnet18_cub": "features.stage4.unit2.body.conv2",
    "resnet50_cub": "features.stage4.unit3.body.conv3",
    "resnet50_cub_mm": "layer4.2.conv3",
    "resnet50": "layer4.2.conv3",
}

# Conservative ImageNet annotation aliases for labels that otherwise stay
# unmapped after the repo's standard concept normalization. Only add
# aliases when the target phrase already exists in the current concept bank.
IMAGENET_LABEL_ALIASES = {
    "website": "a web page",
    "beer bottle": "a bottle with a long neck",
    "wine bottle": "a bottle with a long neck",
    "soda bottle": "a glass or plastic bottle",
    "ski": "a pair of skis",
    "metal nail": "nails",
}

MM_RESNET50_CUB_CHECKPOINT_URL = (
    "https://download.openmmlab.com/mmclassification/v0/resnet/"
    "resnet50_8xb8_cub_20220307-57840e60.pth"
)


def get_resnet50_cub_mm_preprocess():
    return transforms.Compose(
        [
            transforms.Resize(600),
            transforms.CenterCrop(448),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[123.675 / 255.0, 116.28 / 255.0, 103.53 / 255.0],
                std=[58.395 / 255.0, 57.12 / 255.0, 57.375 / 255.0],
            ),
        ]
    )


def _strip_prefix_state_dict(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    stripped = {}
    plen = len(prefix)
    for key, value in state_dict.items():
        if key.startswith(prefix):
            stripped[key[plen:]] = value
    return stripped


def load_mm_resnet50_cub_state_dict(cache_dir: str = "~/.cache/torch/hub/checkpoints") -> Dict[str, torch.Tensor]:
    cache_dir = os.path.expanduser(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    filename = os.path.basename(urlparse(MM_RESNET50_CUB_CHECKPOINT_URL).path)
    file_path = os.path.join(cache_dir, filename)
    if not os.path.exists(file_path):
        torch.hub.download_url_to_file(MM_RESNET50_CUB_CHECKPOINT_URL, file_path)
    checkpoint = torch.load(file_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    stripped = _strip_prefix_state_dict(state_dict, "backbone.")
    if not stripped:
        raise RuntimeError(
            f"Failed to extract backbone weights from MMPretrain checkpoint {file_path}."
        )
    return stripped

def get_resnet_imagenet_preprocess():
    target_mean = [0.485, 0.456, 0.406]
    target_std = [0.229, 0.224, 0.225]
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=target_mean, std=target_std),
        ]
    )
    return preprocess


def _safe_imagenet_pil_loader(path: str) -> Image.Image:
    """
    Robust ImageNet loader.

    We occasionally see a handful of tiny/truncated/corrupt JPEGs after large
    extractions. Failing hard in the data loader is worse than returning a
    placeholder for a tiny fraction of samples.
    """
    min_bytes = int(os.environ.get("CBM_MIN_IMAGE_BYTES", "1024"))
    fallback_size = int(os.environ.get("CBM_FALLBACK_IMAGE_SIZE", "224"))

    # Allow loading truncated JPEGs when possible.
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    try:
        if min_bytes > 0:
            if os.path.getsize(path) < min_bytes:
                raise OSError(f"image file too small (<{min_bytes} bytes)")
        with Image.open(path) as img:
            return img.convert("RGB")
    except (UnidentifiedImageError, OSError, FileNotFoundError):
        # Placeholder keeps dataset indexing stable (important for annotation-aligned datasets).
        return Image.new("RGB", (fallback_size, fallback_size), color=(0, 0, 0))


def get_data(dataset_name, preprocess=None):
    if dataset_name == "cifar100_train":
        data = datasets.CIFAR100(
            root=os.path.expanduser(DATASET_FOLDER),
            download=True,
            train=True,
            transform=preprocess,
        )

    elif dataset_name == "cifar100_val":
        data = datasets.CIFAR100(
            root=os.path.expanduser(DATASET_FOLDER),
            download=True,
            train=False,
            transform=preprocess,
        )

    elif dataset_name == "cifar10_train":
        data = datasets.CIFAR10(
            root=os.path.expanduser(DATASET_FOLDER),
            download=True,
            train=True,
            transform=preprocess,
        )

    elif dataset_name == "cifar10_val":
        data = datasets.CIFAR10(
            root=os.path.expanduser(DATASET_FOLDER),
            download=True,
            train=False,
            transform=preprocess,
        )

    elif dataset_name == "places365_train":
        try:
            data = datasets.Places365(
                root=f"{os.path.expanduser(DATASET_FOLDER)}/places365_torch",
                split="train-standard",
                small=True,
                download=True,
                transform=preprocess,
            )
        except RuntimeError:
            data = datasets.Places365(
                root=f"{os.path.expanduser(DATASET_FOLDER)}/places365_torch",
                split="train-standard",
                small=True,
                download=False,
                transform=preprocess,
            )
    elif dataset_name == "places365_val":
        try:
            data = datasets.Places365(
                root=f"{os.path.expanduser(DATASET_FOLDER)}/places365_torch",
                split="val",
                small=True,
                download=True,
                transform=preprocess,
            )
        except RuntimeError:
            data = datasets.Places365(
                root=f"{os.path.expanduser(DATASET_FOLDER)}/places365_torch",
                split="val",
                small=True,
                download=False,
                transform=preprocess,
            )
    elif dataset_name == "food_train":
        data = data_lp.LinearProbeDataset(
            data_path=f"datasets/food",
            split="train",
            transform=preprocess,
            img_ext="",
            cls_names_file=LABEL_FILES["food"],
        )
        data.targets = data.labels
    elif dataset_name == "food_val":
        data = data_lp.LinearProbeDataset(
            data_path=f"datasets/food",
            split="test",
            transform=preprocess,
            img_ext="",
            cls_names_file=LABEL_FILES["food"],
        )
        data.targets = data.labels
    elif dataset_name == "dtd_train":
        data = data_lp.LinearProbeDataset(
            data_path=f"datasets/dtd",
            split="train",
            transform=preprocess,
            img_ext="",
            cls_names_file=LABEL_FILES["dtd"],
        )
        data.targets = data.labels
    elif dataset_name == "dtd_val":
        data = data_lp.LinearProbeDataset(
            data_path=f"datasets/dtd",
            split="test",
            transform=preprocess,
            img_ext="",
            cls_names_file=LABEL_FILES["dtd"],
        )
        data.targets = data.labels
    elif dataset_name == "flower_train":
        data = data_lp.LinearProbeDataset(
            data_path=f"datasets/flower",
            split="train",
            transform=preprocess,
            cls_names_file=LABEL_FILES["flower"],
        )
        data.targets = data.labels
    elif dataset_name == "flower_val":
        data = data_lp.LinearProbeDataset(
            data_path=f"datasets/flower",
            split="test",
            transform=preprocess,
            cls_names_file=LABEL_FILES["flower"],
        )
        data.targets = data.labels
    elif dataset_name == "aircraft_train":
        data = data_lp.LinearProbeDataset(
            data_path=f"datasets/aircraft",
            split="train",
            transform=preprocess,
            cls_names_file=LABEL_FILES["aircraft"],
        )
        data.targets = data.labels
    elif dataset_name == "aircraft_val":
        data = data_lp.LinearProbeDataset(
            data_path=f"datasets/aircraft",
            split="test",
            transform=preprocess,
            cls_names_file=LABEL_FILES["aircraft"],
        )
        data.targets = data.labels
    elif dataset_name in get_dataset_roots().keys():
        dataset_roots = get_dataset_roots()
        print(
            f"[get_data] building ImageFolder for dataset_name={dataset_name} root={dataset_roots[dataset_name]}",
            flush=True,
        )
        if dataset_name.startswith("imagenet"):
            data = datasets.ImageFolder(
                dataset_roots[dataset_name],
                transform=preprocess,
                loader=_safe_imagenet_pil_loader,
            )
        else:
            data = datasets.ImageFolder(dataset_roots[dataset_name], preprocess)
        print(
            f"[get_data] ready ImageFolder for dataset_name={dataset_name} len={len(data)}",
            flush=True,
        )
    elif dataset_name == "imagenet_broden":
        dataset_roots = get_dataset_roots()
        data = torch.utils.data.ConcatDataset(
            [
                datasets.ImageFolder(dataset_roots["imagenet_val"], preprocess),
                datasets.ImageFolder(dataset_roots["broden"], preprocess),
            ]
        )
    return data


def get_targets_only(dataset_name):
    pil_data = get_data(dataset_name)
    return pil_data.targets


def get_target_model(target_name, device):
    print(f"[get_target_model] request target_name={target_name}", flush=True)
    if target_name.startswith("clip_"):
        target_name = target_name[5:]
        model, preprocess = clip.load(target_name, device=device)
        target_model = lambda x: model.encode_image(x).float()

    elif target_name == "resnet18_places":
        target_model = models.resnet18(pretrained=False, num_classes=365).to(device)
        state_dict = torch.load("data/resnet18_places365.pth.tar")["state_dict"]
        new_state_dict = {}
        for key in state_dict:
            if key.startswith("module."):
                new_state_dict[key[7:]] = state_dict[key]
        target_model.load_state_dict(new_state_dict)
        target_model.eval()
        preprocess = get_resnet_imagenet_preprocess()

    elif target_name in {"resnet18_cub", "resnet50_cub"}:
        target_model = _load_ptcv_cub_model_from_local_cache(target_name, device)
        target_model.eval()
        preprocess = get_resnet_imagenet_preprocess()

    elif target_name == "resnet50_cub_mm":
        target_model = models.resnet50(weights=None).to(device)
        missing = target_model.load_state_dict(load_mm_resnet50_cub_state_dict(), strict=False)
        unexpected = list(getattr(missing, "unexpected_keys", []))
        missing_keys = [k for k in getattr(missing, "missing_keys", []) if not k.startswith("fc.")]
        if unexpected or missing_keys:
            raise RuntimeError(
                f"Unexpected MMPretrain ResNet-50 CUB load mismatch. missing={missing_keys} unexpected={unexpected}"
            )
        target_model.eval()
        preprocess = get_resnet50_cub_mm_preprocess()

    elif target_name.endswith("_v2"):
        target_name = target_name[:-3]
        target_name_cap = target_name.replace("resnet", "ResNet")
        weights = eval("models.{}_Weights.IMAGENET1K_V2".format(target_name_cap))
        target_model = eval("models.{}(weights).to(device)".format(target_name))
        target_model.eval()
        preprocess = weights.transforms()

    else:
        target_name_cap = target_name.replace("resnet", "ResNet")
        weights = eval("models.{}_Weights.IMAGENET1K_V1".format(target_name_cap))
        target_model = eval("models.{}(weights=weights).to(device)".format(target_name))
        target_model.eval()
        preprocess = weights.transforms()

    return target_model, preprocess


def _load_ptcv_cub_model_from_local_cache(target_name: str, device):
    print(f"[_load_ptcv_cub_model_from_local_cache] init target_name={target_name}", flush=True)
    model = ptcv_get_model(target_name, pretrained=False).to(device)
    print(f"[_load_ptcv_cub_model_from_local_cache] instantiated target_name={target_name}", flush=True)
    cache_dir = os.path.expanduser("~/.torch/models")
    patterns = [
        os.path.join(cache_dir, f"{target_name}-*.pth"),
        os.path.join(cache_dir, f"{target_name}-*.pt"),
        os.path.join(cache_dir, f"{target_name}-*.pth.zip"),
    ]
    matches = []
    for pattern in patterns:
        matches.extend(sorted(glob.glob(pattern)))
    if matches:
        print(f"[_load_ptcv_cub_model_from_local_cache] loading state from {matches[0]}", flush=True)
        state_dict = torch.load(matches[0], map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict)
        print(f"[_load_ptcv_cub_model_from_local_cache] loaded target_name={target_name}", flush=True)
        return model
    logger.info(
        "No local cached weights found for {} under ~/.torch/models; falling back to pytorchcv pretrained load.",
        target_name,
    )
    return ptcv_get_model(target_name, pretrained=True).to(device)


def format_concept(s):
    # replace - with ' '
    # replace , with ' '
    # only one space between words
    s = s.lower()
    s = s.replace("-", " ")
    s = s.replace(",", " ")
    s = s.replace(".", " ")
    s = s.replace("(", " ")
    s = s.replace(")", " ")
    if s[:2] == "a ":
        s = s[2:]
    elif s[:3] == "an ":
        s = s[3:]

    # remove trailing and leading spaces
    s = " ".join(s.split())
    return s


def canonicalize_concept_label(s: str) -> str:
    normalized = format_concept(s)
    aliased = IMAGENET_LABEL_ALIASES.get(normalized, normalized)
    return format_concept(aliased)

def get_classes(dataset_name):
    with open(LABEL_FILES[dataset_name], "r") as f:
        classes = f.read().split("\n")
    return classes


def get_concepts(concept_file: str, filter_file:Optional[str]=None) -> List[str]:
    with open(concept_file) as f:
        concepts: List[str] = f.read().split("\n")

    # remove repeated concepts and maintain order
    concepts = list(dict.fromkeys([canonicalize_concept_label(concept) for concept in concepts]))

    # check for filter file
    if filter_file and os.path.exists(filter_file):
        logger.info(f"Filtering concepts using {filter_file}")
        with open(filter_file) as f:
            to_filter_concepts = f.read().split("\n")
        to_filter_concepts = [canonicalize_concept_label(concept) for concept in to_filter_concepts]
        concepts = [concept for concept in concepts if concept not in to_filter_concepts]

    return concepts


def save_concept_count(
    concepts: List[str],
    counts: List[int],
    save_dir: str,
    file_name: str = "concept_counts.txt",
):
    with open(os.path.join(save_dir, file_name), "w") as f:
        if len(concepts) != len(counts):
            raise ValueError("Length of concepts and counts should be the same")
        f.write(f"{concepts[0]} {counts[0]}")
        for concept, count in zip(concepts[1:], counts[1:]):
            f.write(f"\n{concept} {count}")


def load_concept_and_count(
    save_dir: str, file_name: str = "concept_counts.txt", filter_file:Optional[str]=None
) -> Tuple[List[str], List[float]]:
    with open(os.path.join(save_dir, file_name), "r") as f:
        lines = f.readlines()
        concepts = []
        counts = []
        for line in lines:
            concept = line.split(" ")[:-1]
            concept = " ".join(concept)
            count = line.split(" ")[-1]
            concepts.append(canonicalize_concept_label(concept))
            counts.append(float(count))

    if filter_file and os.path.exists(filter_file):
        with open(filter_file) as f:
            logger.info(f"Filtering concepts using {filter_file}")
            to_filter_concepts = f.read().split("\n")
        to_filter_concepts = [canonicalize_concept_label(concept) for concept in to_filter_concepts]
        counts = [count for concept, count in zip(concepts, counts) if concept not in to_filter_concepts]
        concepts = [concept for concept in concepts if concept not in to_filter_concepts]
        assert len(concepts) == len(counts)

    return concepts, counts

def save_filtered_concepts(
    filtered_concepts: List[str],
    save_dir: str,
    file_name: str = "filtered_concepts.txt",
):
    with open(os.path.join(save_dir, file_name), "w") as f:
        if len(filtered_concepts) > 0:
            f.write(filtered_concepts[0])
            for concept in filtered_concepts[1:]:
                f.write("\n" + concept)

def show_box(box, ax, label):
    if plt is None:
        raise ImportError(
            "show_box() requires matplotlib, but it is not installed in this environment."
        )
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2))
    ax.text(x0, y0, label)

def plot_annotations(image_pil: Image.Image, annotations: List[Dict]):
    """
    Plot annotations on image

    Args:
        image_pil (Image.Image): The PIL image
        annotations (List[Dict]): The annotations to plot in the following format:
            - logits: The logits associated with each token of the concept.
            - score: The perplexity of the concept.
            - concept: The concept associated with the bounding box.
            - bbox: The bounding box coordinates.

    Returns:
        plt.Figure: The figure containing the image with annotations.
    """
    if plt is None:
        raise ImportError(
            "plot_annotations() requires matplotlib, but it is not installed in this environment."
        )
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(image_pil)
    for annotation in annotations:
        show_box(annotation["box"], plt.gca(), f"{annotation['label']} : {annotation['logit']:.3f}")
    plt.axis("off")
    return fig
