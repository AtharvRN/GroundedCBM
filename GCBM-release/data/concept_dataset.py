import json
import os
import hashlib
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm

import model.utils as utils
import data.utils as data_utils
from model.cbm import Backbone, ConceptLayer, NormalizationLayer
from data.utils import canonicalize_concept_label, format_concept, get_classes
from glm_saga.elasticnet import IndexedTensorDataset
from data.utils import plot_annotations


def _unwrap_dataset_indices(dataset: Dataset) -> Tuple[Dataset, List[int]]:
    if isinstance(dataset, torch.utils.data.Subset):
        base_dataset, base_indices = _unwrap_dataset_indices(dataset.dataset)
        return base_dataset, [base_indices[idx] for idx in dataset.indices]
    return dataset, list(range(len(dataset)))


def _loader_embedding_cache_path(
    backbone: Backbone,
    loader: DataLoader,
    cache_dir: Optional[str],
    cache_tag: str,
) -> str:
    base_dataset, sample_indices = _unwrap_dataset_indices(loader.dataset)
    dataset_name = getattr(base_dataset, "dataset_name", "unknown")
    split_suffix = getattr(base_dataset, "split_suffix", "unknown")
    concept_hash = hashlib.sha1(
        "\n".join(getattr(base_dataset, "concepts", [])).encode("utf-8")
    ).hexdigest()[:16]
    sample_hash = hashlib.sha1(
        ",".join(map(str, sample_indices)).encode("utf-8")
    ).hexdigest()[:16]
    preprocess_repr = repr(getattr(base_dataset, "preprocess", None))
    preprocess_hash = hashlib.sha1(preprocess_repr.encode("utf-8")).hexdigest()[:16]
    metadata = {
        "backbone_name": getattr(backbone, "backbone_name", backbone.__class__.__name__),
        "feature_layer": getattr(backbone, "feature_layer", ""),
        "dataset_name": dataset_name,
        "split_suffix": split_suffix,
        "confidence_threshold": getattr(base_dataset, "confidence_threshold", None),
        "concept_hash": concept_hash,
        "sample_hash": sample_hash,
        "preprocess_hash": preprocess_hash,
        "cache_tag": cache_tag,
    }
    digest = hashlib.sha1(
        json.dumps(metadata, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    if cache_dir is None:
        default_root = getattr(base_dataset, "cache_dir", os.path.join("outputs", "_cache"))
        cache_dir = os.path.join(default_root, "backbone_embeddings")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(
        cache_dir,
        f"{dataset_name}_{split_suffix}_{cache_tag}_{digest}.pt",
    )


def get_or_create_backbone_embedding_cache(
    backbone: Backbone,
    loader: DataLoader,
    device: str = "cuda",
    cache_dir: Optional[str] = None,
    cache_tag: str = "default",
):
    cache_path = _loader_embedding_cache_path(backbone, loader, cache_dir, cache_tag)
    if os.path.exists(cache_path):
        logger.info("Loading cached backbone embeddings from {}", cache_path)
        return torch.load(cache_path, weights_only=False)

    logger.info("Caching backbone embeddings to {}", cache_path)
    embeddings = []
    concept_one_hots = []
    labels = []
    with torch.no_grad():
        for features, concept_one_hot, batch_labels in tqdm(loader):
            features = features.to(device)
            batch_embeddings = backbone(features).detach().cpu()
            embeddings.append(batch_embeddings)
            concept_one_hots.append(concept_one_hot.cpu())
            labels.append(batch_labels.cpu())

    cached = {
        "embeddings": torch.cat(embeddings, dim=0),
        "concept_one_hot": torch.cat(concept_one_hots, dim=0),
        "labels": torch.cat(labels, dim=0),
    }
    torch.save(cached, cache_path)
    return cached


class ConceptDataset(Dataset):
    def __init__(
        self,
        dataset_name: str,
        torch_dataset: Dataset,
        concepts: List[str] = None,
        split_suffix="train",
        label_dir: str = "outputs",
        confidence_threshold: float = 0.10,
        preprocess=None,
        crop_to_concept_prob: bool = 0.0,
        overlap_iou_threshold: float = 0.5,
        concept_only=False,
        use_annotation_cache: bool = True,
    ):
        self.torch_dataset = torch_dataset
        self.concepts = concepts
        self.dataset_name = dataset_name
        self.split_suffix = split_suffix
        self.dir = f"{label_dir}/{dataset_name}_{split_suffix}"
        self.confidence_threshold = confidence_threshold
        self.preprocess = preprocess
        self.overlap_iou_threshold = overlap_iou_threshold
        self.concept_only = concept_only
        # Return cropped image containing a single concept
        # with probability `crop_to_concept_prob`
        self.crop_to_concept_prob = crop_to_concept_prob
        self.use_annotation_cache = use_annotation_cache
        self.cache_dir = os.path.join(label_dir, "_cache")
        self._annotations = None
        self._concept_matrix = None
        self.cache_build_workers = min(32, max(4, (os.cpu_count() or 8)))

        if self.use_annotation_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
            if self.concepts is not None:
                self._concept_matrix = self._load_or_create_concept_matrix_cache(
                    dataset_name, split_suffix
                )
                if self.crop_to_concept_prob == 0:
                    self._annotations = None
            if self.crop_to_concept_prob > 0:
                self._annotations = self._load_or_create_annotation_cache(
                    dataset_name, split_suffix
                )

    def __len__(self):
        return len(self.torch_dataset)

    def __getitem__(self, idx):
        if self.concept_only:
            return 0, self._get_concept(idx), 0 # 0 is placeholder
        prob = np.random.rand()
        if prob < self.crop_to_concept_prob:
            try:
                return self.__getitem__per_concept(idx)
            except Exception as e:
                logger.warning(f"Failed to get item {idx} per concept: {e}")

        return self.__getitem__all(idx)

    def __getitem__per_concept(self, idx):
        image, target = self.torch_dataset[idx]

        bbxs = self._get_annotations(idx)

        # get mapping of concepts to a random bounding box containing the concept
        concept_bbx_map = []
        for concept_idx, concept in enumerate(self.concepts):
            _, matched_bbxs = self._find_in_list(concept, bbxs)
            if len(matched_bbxs) > 0:
                concept_bbx_map.append((concept_idx, matched_bbxs[np.random.randint(0, len(matched_bbxs))]))

        # get one hot vector of concepts
        concept_one_hot = torch.zeros(len(self.concepts), dtype=torch.float)
        if len(concept_bbx_map) > 0:
            # randomly pick a concept and its bounding box
            random_concept_idx, random_bbx = concept_bbx_map[np.random.randint(0, len(concept_bbx_map))]
            concept_one_hot[random_concept_idx] = 1.0
            image = image.crop(random_bbx["box"])

            # mark concepts with high overlap with the selected concept as 1
            for bbx in bbxs:
                if bbx["label"] == random_bbx["label"]:
                    continue
                else:
                    iou = utils.get_bbox_iou(random_bbx["box"], bbx["box"])
                    try:
                        if iou > self.overlap_iou_threshold:
                            concept_idx = self.concepts.index(bbx["label"])
                            concept_one_hot[concept_idx] = 1.0
                            # logger.debug(f"Marking {bbx['concept']} as 1 due to overlap with {random_bbx['concept']}")
                    except ValueError:
                        continue

        # preprocess image
        if self.preprocess:
            image = self.preprocess(image)

        return image, concept_one_hot, target

    def __getitem__all(self, idx):
        image, target = self.torch_dataset[idx]

        concept_one_hot = self._get_concept(idx)

        # preprocess image
        if self.preprocess:
            image = self.preprocess(image)

        return image, concept_one_hot, target
    
    def _get_concept(self, idx):
        if self._concept_matrix is not None:
            return self._concept_matrix[idx]

        bbxs = self._get_annotations(idx)
        concept_one_hot = [
            1 if self._find_in_list(concept, bbxs)[0] else 0
            for concept in self.concepts
        ]
        return torch.tensor(concept_one_hot, dtype=torch.float)

    def _find_in_list(self, concept: str, bbxs: List[Dict[str, Any]]) -> Tuple[bool, List[Dict[str, Any]]]:
        # randomly pick a bounding box
        matched_bbxs = [bbx for bbx in bbxs if concept == bbx["label"]]
        return len(matched_bbxs) > 0, matched_bbxs

    def _load_raw_data(self, idx):
        data_file = f"{self.dir}/{idx}.json"
        with open(data_file, "r") as f:
            data = json.load(f)
        return data

    def _normalize_annotations(self, data):
        annotations = []
        for bbx in data[1:]:
            if bbx["logit"] <= self.confidence_threshold:
                continue
            normalized = dict(bbx)
            normalized["label"] = canonicalize_concept_label(normalized["label"])
            annotations.append(normalized)
        return annotations

    def _get_annotations(self, idx):
        annotations = self._ensure_annotations_loaded(
            self.dataset_name, self.split_suffix
        )
        if annotations is not None:
            return annotations[idx]
        return self._normalize_annotations(self._load_raw_data(idx))

    def _annotation_cache_path(self, dataset_name: str, split_suffix: str) -> str:
        threshold = str(self.confidence_threshold).replace(".", "p")
        return os.path.join(
            self.cache_dir,
            f"{dataset_name}_{split_suffix}_thr_{threshold}_v1_annotations.pt",
        )

    def _concept_cache_path(self, dataset_name: str, split_suffix: str) -> str:
        threshold = str(self.confidence_threshold).replace(".", "p")
        concept_hash = hashlib.sha1(
            "\n".join(self.concepts).encode("utf-8")
        ).hexdigest()[:16]
        return os.path.join(
            self.cache_dir,
            (
                f"{dataset_name}_{split_suffix}_thr_{threshold}_"
                f"concepts_{concept_hash}_v1_matrix.pt"
            ),
        )

    def _load_or_create_annotation_cache(
        self, dataset_name: str, split_suffix: str
    ) -> List[List[Dict[str, Any]]]:
        cache_path = self._annotation_cache_path(dataset_name, split_suffix)
        if os.path.exists(cache_path):
            logger.info("Loading cached annotations from {}", cache_path)
            return torch.load(cache_path, weights_only=False)

        logger.info("Caching parsed annotations to {}", cache_path)
        annotations = [
            self._normalize_annotations(self._load_raw_data(idx))
            for idx in tqdm(range(len(self.torch_dataset)))
        ]
        torch.save(annotations, cache_path)
        return annotations

    def _load_or_create_concept_matrix_cache(
        self, dataset_name: str, split_suffix: str
    ) -> torch.Tensor:
        cache_path = self._concept_cache_path(dataset_name, split_suffix)
        if os.path.exists(cache_path):
            logger.info("Loading cached concept matrix from {}", cache_path)
            return torch.load(cache_path, weights_only=False)

        logger.info("Caching concept matrix to {}", cache_path)
        concept_to_idx = {concept: idx for idx, concept in enumerate(self.concepts)}
        concept_matrix = torch.zeros(len(self.torch_dataset), len(self.concepts), dtype=torch.bool)

        def build_row(sample_idx: int):
            row = torch.zeros(len(self.concepts), dtype=torch.bool)
            for bbx in self._normalize_annotations(self._load_raw_data(sample_idx)):
                concept_idx = concept_to_idx.get(bbx["label"])
                if concept_idx is not None:
                    row[concept_idx] = True
            return sample_idx, row

        with ThreadPoolExecutor(max_workers=self.cache_build_workers) as executor:
            for sample_idx, row in tqdm(
                executor.map(build_row, range(len(self.torch_dataset))),
                total=len(self.torch_dataset),
            ):
                concept_matrix[sample_idx] = row
        concept_matrix = concept_matrix.float()
        torch.save(concept_matrix, cache_path)
        return concept_matrix

    def _ensure_annotations_loaded(
        self, dataset_name: Optional[str] = None, split_suffix: Optional[str] = None
    ):
        if self._annotations is not None:
            return self._annotations
        if (
            self.use_annotation_cache
            and dataset_name is not None
            and split_suffix is not None
        ):
            self._annotations = self._load_or_create_annotation_cache(
                dataset_name, split_suffix
            )
            return self._annotations
        return None

    def get_annotations(self, idx: int):
        return self._get_annotations(idx)

    def visualize_annotations(self, idx: int):
        image_pil = self.torch_dataset[idx][0]
        annotations = self._get_annotations(idx)
        fig = plot_annotations(image_pil, annotations)
        fig.show()

    def plot_annotations(self, idx: int, annotations: List[Dict[str, Any]]):
        image_pil = self.torch_dataset[idx][0]
        fig = plot_annotations(image_pil, annotations)
        fig.show()

    def get_image_pil(self, idx: int):
        return self.torch_dataset[idx][0]

    def get_target(self, idx):
        _, target = self.torch_dataset[idx]
        return target    

class AllOneConceptDataset(ConceptDataset):
    def __init__(self, classes, *args, **kwargs):
        print(args, kwargs)
        super().__init__(*args, **kwargs)
        self.per_class_concepts = len(self.concepts) // len(classes)
        logger.info(f"Assigning {self.per_class_concepts} concepts to each class")

    def __getitem__(self, idx):
        image, target = self.torch_dataset[idx]
        if self.preprocess:
            image = self.preprocess(image)
        concept_one_hot = torch.zeros((len(self.concepts),), dtype=torch.float)
        concept_one_hot[target * self.per_class_concepts : (target + 1) * self.per_class_concepts] = 1
        return image, concept_one_hot, target


def get_concept_dataloader(
    dataset_name: str,
    split: str,
    concepts: List[str],
    preprocess=None,
    val_split: Optional[float] = 0.1,
    batch_size: int = 256,
    num_workers: int = 4,
    shuffle: bool = False,
    confidence_threshold: float = 0.10,
    crop_to_concept_prob: float = 0.0,
    label_dir="outputs",
    use_allones=False,
    seed: int = 42,
    concept_only=False
):
    dataset = ConceptDataset if not use_allones else partial(AllOneConceptDataset, get_classes(dataset_name))
    if split == "test":
        dataset = dataset(
            dataset_name,
            data_utils.get_data(f"{dataset_name}_val", None),
            concepts,
            split_suffix="val",
            preprocess=preprocess,
            confidence_threshold=confidence_threshold,
            crop_to_concept_prob=crop_to_concept_prob,
            label_dir=label_dir,
            concept_only=concept_only
        )
        logger.info(f"Test dataset size: {len(dataset)}")
    else:
        assert val_split is not None
        dataset = dataset(
            dataset_name,
            data_utils.get_data(f"{dataset_name}_train", None),
            concepts,
            split_suffix="train",
            preprocess=preprocess,
            confidence_threshold=confidence_threshold,
            crop_to_concept_prob=crop_to_concept_prob,
            label_dir=label_dir,
            concept_only=concept_only
        )

        # get split indices
        n_val = int(val_split * len(dataset))
        n_train = len(dataset) - n_val
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed)
        )  # ensure same split in same run

        if split == "train":
            logger.info(f"Train dataset size: {len(train_dataset)}")
            dataset = train_dataset
        else:
            logger.info(f"Val dataset size: {len(val_dataset)}")
            dataset = val_dataset

    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
    return loader


def get_filtered_concepts_and_counts(
    dataset_name,
    raw_concepts,
    preprocess=None,
    val_split: Optional[float] = 0.1,
    batch_size: int = 256,
    num_workers: int = 4,
    confidence_threshold: float = 0.10,
    label_dir="outputs",
    use_allones: bool = False,
    seed: int = 42,
):
    # remove concepts that are not present in the dataset
    dataloader = get_concept_dataloader(
        dataset_name,
        "train",
        raw_concepts,
        preprocess=preprocess,
        val_split=val_split,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        confidence_threshold=confidence_threshold,
        crop_to_concept_prob=0.0,
        label_dir=label_dir,
        use_allones=use_allones,
        seed=seed,
        concept_only=True
    )
    # get concept counts
    raw_concepts_count = torch.zeros(len(raw_concepts))
    for data in tqdm(dataloader):
        raw_concepts_count += data[1].sum(dim=0)

    # remove concepts that are not present in the dataset
    raw_concepts_count = raw_concepts_count.numpy()
    concepts = [concept for concept, count in zip(raw_concepts, raw_concepts_count) if count > 0]
    concept_counts = [count for _, count in zip(raw_concepts, raw_concepts_count) if count > 0]
    filtered_concepts = [concept for concept, count in zip(raw_concepts, raw_concepts_count) if count == 0]
    print(f"Filtered {len(raw_concepts) - len(concepts)} concepts")

    return concepts, concept_counts, filtered_concepts


def get_final_layer_dataset(
    backbone: Backbone,
    cbl: ConceptLayer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    save_dir: str,
    load_dir: str = None,
    batch_size: int = 256,
    device="cuda",
    filter=None,
    use_activation_cache: bool = False,
    activation_cache_dir: Optional[str] = None,
):
    if load_dir is None:
        logger.info("Creating final layer training and validation datasets")
        with torch.no_grad():
            if use_activation_cache:
                train_cached = get_or_create_backbone_embedding_cache(
                    backbone,
                    train_loader,
                    device=device,
                    cache_dir=activation_cache_dir,
                    cache_tag="train",
                )
                val_cached = get_or_create_backbone_embedding_cache(
                    backbone,
                    val_loader,
                    device=device,
                    cache_dir=activation_cache_dir,
                    cache_tag="val",
                )
                logger.info("Creating final layer training dataset from cached embeddings")
                train_concept_features = []
                for start_idx in tqdm(range(0, len(train_cached["embeddings"]), batch_size)):
                    embeddings = train_cached["embeddings"][start_idx:start_idx + batch_size].to(device)
                    train_concept_features.append(cbl(embeddings).detach().cpu())
                train_concept_features = torch.cat(train_concept_features, dim=0)
                train_concept_labels = train_cached["labels"]

                logger.info("Creating final layer validation dataset from cached embeddings")
                val_concept_features = []
                for start_idx in tqdm(range(0, len(val_cached["embeddings"]), batch_size)):
                    embeddings = val_cached["embeddings"][start_idx:start_idx + batch_size].to(device)
                    val_concept_features.append(cbl(embeddings).detach().cpu())
                val_concept_features = torch.cat(val_concept_features, dim=0)
                val_concept_labels = val_cached["labels"]
            else:
                train_concept_features = []
                train_concept_labels = []
                logger.info("Creating final layer training dataset")
                for features, _, labels in tqdm(train_loader):
                    features = features.to(device)
                    concept_logits = cbl(backbone(features))
                    train_concept_features.append(concept_logits.detach().cpu())
                    train_concept_labels.append(labels)
                train_concept_features = torch.cat(train_concept_features, dim=0)
                train_concept_labels = torch.cat(train_concept_labels, dim=0)

                val_concept_features = []
                val_concept_labels = []
                logger.info("Creating final layer validation dataset")
                for features, _, labels in tqdm(val_loader):
                    features = features.to(device)
                    concept_logits = cbl(backbone(features))
                    val_concept_features.append(concept_logits.detach().cpu())
                    val_concept_labels.append(labels)
                val_concept_features = torch.cat(val_concept_features, dim=0)
                val_concept_labels = torch.cat(val_concept_labels, dim=0)

            # normalize concept features
            train_concept_features_mean = train_concept_features.mean(dim=0)
            train_concept_features_std = train_concept_features.std(dim=0)
            train_concept_features = (train_concept_features - train_concept_features_mean) / train_concept_features_std
            val_concept_features = (val_concept_features - train_concept_features_mean) / train_concept_features_std

            # normalization layer
            normalization_layer = NormalizationLayer(train_concept_features_mean, train_concept_features_std, device=device)
    else:
        # load normalized concept features
        logger.info("Loading final layer training dataset")
        train_concept_features = torch.load(os.path.join(load_dir, "train_concept_features.pt"))
        train_concept_labels = torch.load(os.path.join(load_dir, "train_concept_labels.pt"))
        val_concept_features = torch.load(os.path.join(load_dir, "val_concept_features.pt"))
        val_concept_labels = torch.load(os.path.join(load_dir, "val_concept_labels.pt"))
        normalization_layer = NormalizationLayer.from_pretrained(load_dir, device=device)

    # save normalized concept features
    torch.save(train_concept_features, os.path.join(save_dir, "train_concept_features.pt"))
    torch.save(train_concept_labels, os.path.join(save_dir, "train_concept_labels.pt"))
    torch.save(val_concept_features, os.path.join(save_dir, "val_concept_features.pt"))
    torch.save(val_concept_labels, os.path.join(save_dir, "val_concept_labels.pt"))

    # save normalized concept features mean and std
    normalization_layer.save_model(save_dir)
    if filter is not None:
        train_concept_features = train_concept_features[:, filter]
        val_concept_features = val_concept_features[:, filter]
    # Note: glm saga expects y to be on CPU
    train_concept_dataset = IndexedTensorDataset(train_concept_features, train_concept_labels)
    val_concept_dataset = TensorDataset(val_concept_features, val_concept_labels)
    logger.info("Train concept dataset size: {}".format(len(train_concept_dataset)))
    logger.info("Val concept dataset size: {}".format(len(val_concept_dataset)))

    train_concept_loader = DataLoader(train_concept_dataset, batch_size=batch_size, shuffle=True)
    val_concept_loader = DataLoader(val_concept_dataset, batch_size=batch_size, shuffle=False)
    return train_concept_loader, val_concept_loader, normalization_layer
