"""Explicit image identities, one saved split, and cached deterministic targets."""
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sqlite3

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from gcbm.spatial_target_cache import SpatialTargetCache
from gcbm.target_batches import pad_sparse_targets


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def absolute(path, base):
    path = Path(path).expanduser()
    return str((path if path.is_absolute() else base / path).resolve())


def image_reference(value, base):
    if isinstance(value, str):
        return absolute(value, base)
    if not isinstance(value, dict) or set(value) != {'archive', 'offset', 'size'}:
        raise ValueError('Image must be a filename or an archive/offset/size record')
    if type(value['offset']) is not int or type(value['size']) is not int or value['offset'] < 0 or value['size'] < 1:
        raise ValueError('Invalid archive byte range')
    return dict(value, archive=absolute(value['archive'], base))


def detections(row, base):
    if 'detections' in row:
        items = row['detections']
    elif 'annotation' in row:
        payload = json.loads(Path(absolute(row['annotation'], base)).read_text())
        expected = row.get('annotation_image_id', row['id'])
        if isinstance(payload, list) and payload and isinstance(payload[0], dict):
            actual = payload[0].get('file_name', payload[0].get('image_id'))
            items = payload[1:]
        elif isinstance(payload, dict):
            actual, items = payload.get('image_id'), payload.get('detections')
        else:
            raise ValueError('Malformed annotation file')
        if actual != expected:
            raise ValueError(f'Annotation/image identity mismatch: {actual!r} != {expected!r}')
    else:
        return None  # A backbone-only manifest cannot silently train concept negatives.
    if not isinstance(items, list):
        raise ValueError('Detections must be a list; use [] for a verified negative image')
    result = []
    for item in items:
        label = item.get('concept', item.get('label'))
        confidence = float(item.get('confidence', item.get('logit', float('nan'))))
        box = item.get('box')
        coordinates = item.get('coordinates', row.get('box_coordinates'))
        if not isinstance(label, str) or not label.strip() or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError('Invalid concept label or confidence')
        if box is not None:
            if coordinates not in {'normalized_xyxy', 'pixel_xyxy'}:
                raise ValueError('Box coordinates must be explicit; coordinate guessing is not supported')
            if len(box) != 4 or not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError('Invalid xyxy box')
            if min(box) < 0 or (coordinates == 'normalized_xyxy' and max(box) > 1):
                raise ValueError('Box coordinates out of bounds')
        result.append(dict(concept=label.strip(), confidence=confidence, box=box, coordinates=coordinates))
    return result


def prepare(source, output, dataset, concepts=None, split=None, fraction=.05, seed=42):
    source, output = Path(source).resolve(), Path(output).resolve()
    if dataset not in {'cub', 'imagenet', 'places365', 'partimagenetpp'} or not 0 < fraction < 1:
        raise ValueError('Invalid dataset or validation fraction')
    output.mkdir(parents=True, exist_ok=False)
    bank = [] if concepts is None else [s.strip() for s in Path(concepts).read_text().splitlines() if s.strip()]
    if len(bank) != len(set(bank)):
        raise ValueError('Duplicate concept names')
    groups, identities, image_ids = {}, {}, set()
    annotated = 0
    with sqlite3.connect(output/'samples.sqlite') as db, source.open() as stream:
        db.execute('CREATE TABLE samples (position INTEGER PRIMARY KEY, id TEXT UNIQUE, image TEXT UNIQUE, class_id INTEGER, detections TEXT)')
        for i, line in enumerate(stream):
            row = json.loads(line)
            sample_id, category = row['id'], row['class_id']
            if not isinstance(sample_id, str) or not sample_id or type(category) is not int or category < 0:
                raise ValueError('Each sample needs a nonempty string id and nonnegative integer class_id')
            image = json.dumps(image_reference(row['image'], source.parent), sort_keys=True)
            if sample_id in identities or image in image_ids:
                raise ValueError('Duplicate image or sample identity')
            items = detections(row, source.parent)
            annotated += items is not None
            db.execute('INSERT INTO samples VALUES (?,?,?,?,?)', (i, sample_id, image, category, json.dumps(items)))
            identities[sample_id] = i
            image_ids.add(image)
            groups.setdefault(category, []).append(sample_id)
    if not groups or sorted(groups) != list(range(len(groups))):
        raise ValueError('Class ids must be contiguous from zero')
    if split:
        partition = json.loads(Path(split).read_text())
        train, validation = partition['train'], partition['validation']
    else:
        rng = np.random.default_rng(seed)
        train, validation = [], []
        for category in sorted(groups):
            ids = sorted(groups[category])
            if len(ids) < 2:
                raise ValueError('Every class needs at least two images for an internal holdout')
            rng.shuffle(ids)
            count = min(len(ids)-1, max(1, round(len(ids)*fraction)))
            validation.extend(ids[:count]); train.extend(ids[count:])
    if (len(train) != len(set(train)) or len(validation) != len(set(validation))
            or set(train) & set(validation) or set(train) | set(validation) != set(identities)):
        raise ValueError('Split must be disjoint and partition the exact training pool')
    train_ids, val_ids = set(train), set(validation)
    if not train or not validation or any(not set(ids)&train_ids or not set(ids)&val_ids for ids in groups.values()):
        raise ValueError('Every class must occur in both training and validation')
    save_json(output/'split.json', dict(train=train, validation=validation))
    np.savez(output/'indices.npz', train=[identities[x] for x in train], validation=[identities[x] for x in validation])
    (output/'concepts.txt').write_text('\n'.join(bank) + ('\n' if bank else ''))
    files = {n:digest(output/n) for n in ['samples.sqlite','split.json','indices.npz','concepts.txt']}
    protocol = dict(version=1, dataset=dataset, source_sha256=digest(source), files=files,
                    classes=len(groups), images=len(identities), annotated_images=annotated,
                    train=len(train), validation=len(validation), split_seed=seed if not split else None)
    save_json(output/'prepared.json', protocol)
    return protocol


def verify(prepared):
    root = Path(prepared)
    protocol = json.loads((root/'prepared.json').read_text())
    for name, expected in protocol['files'].items():
        if digest(root/name) != expected:
            raise ValueError(f'Prepared data changed: {name}')
    return protocol


def occupancy(box, coordinates, size, resize, crop, grid):
    width, height = size
    xy = np.asarray(box, dtype=np.float64).copy()
    if coordinates == 'normalized_xyxy':
        xy *= [width, height, width, height]
    if xy[2] > width or xy[3] > height:
        raise ValueError('Pixel box exceeds image dimensions')
    rw, rh = ((resize, int(resize*height/width)) if width <= height
              else (int(resize*width/height), resize))
    xy *= [rw/width, rh/height, rw/width, rh/height]
    xy -= [round((rw-crop)/2), round((rh-crop)/2)]*2
    xy = np.clip(xy/crop, 0., 1.)
    starts = np.arange(grid)/grid
    xs = np.maximum(0., np.minimum(starts+1/grid, xy[2])-np.maximum(starts, xy[0]))
    ys = np.maximum(0., np.minimum(starts+1/grid, xy[3])-np.maximum(starts, xy[1]))
    return (ys[:,None]*xs[None,:]*grid*grid).clip(0,1).astype(np.float32)


class Samples(Dataset):
    def __init__(self, cfg, split, protocol, concepts=True):
        self.cfg, self.root, self.concepts = cfg, Path(cfg.data.prepared), concepts
        self.indices = np.load(self.root/'indices.npz')[split]
        self.bank = (self.root/'concepts.txt').read_text().splitlines()
        self.lookup = {c:i for i,c in enumerate(self.bank)}
        if concepts and (not self.bank or protocol['annotated_images'] != protocol['images']):
            raise ValueError('Concept training requires a vocabulary and explicit annotations for every image')
        self.grid = cfg.data.input_size // (8 if cfg.model.stages == 'conv345' else 16)
        self.cache = None if not concepts else SpatialTargetCache(cfg.data.cache, dict(
            version=1, database=protocol['files']['samples.sqlite'], vocabulary=self.bank,
            resize=cfg.data.resize_size, crop=cfg.data.input_size, grid=self.grid,
            confidence=cfg.data.confidence, presence='full_image', union='maximum_occupancy'))
        self.db, self.pid, self.fds = None, None, {}
        interpolation = transforms.InterpolationMode.BILINEAR if concepts else transforms.InterpolationMode.BICUBIC
        spatial = ([transforms.RandomResizedCrop(cfg.data.input_size, interpolation=interpolation), transforms.RandomHorizontalFlip()]
                   if not concepts and split == 'train' else
                   [transforms.Resize(cfg.data.resize_size, interpolation=interpolation), transforms.CenterCrop(cfg.data.input_size)])
        if not concepts and split == 'train' and cfg.training.randaugment_ops:
            spatial.append(transforms.RandAugment(num_ops=cfg.training.randaugment_ops,
                                                 magnitude=cfg.training.randaugment_magnitude))
        self.transform = transforms.Compose(spatial + [transforms.ToTensor(),
            transforms.Normalize((.485,.456,.406),(.229,.224,.225))])
        if not concepts and split == 'train' and cfg.training.random_erasing:
            self.transform.transforms.append(transforms.RandomErasing(p=cfg.training.random_erasing))

    def __len__(self):
        return len(self.indices)

    def __getstate__(self):
        return {**self.__dict__, 'db':None, 'pid':None, 'fds':{}}

    def __getitem__(self, index):
        if self.pid != os.getpid():
            if self.db is not None:
                self.db.close()
            for fd in self.fds.values():
                os.close(fd)
            self.db = sqlite3.connect(f'file:{self.root / "samples.sqlite"}?mode=ro', uri=True)
            self.pid, self.fds = os.getpid(), {}
        position = int(self.indices[index])
        _, reference, category, annotations = self.db.execute(
            'SELECT id,image,class_id,detections FROM samples WHERE position=?', (position,)).fetchone()
        reference = json.loads(reference)
        if isinstance(reference, str):
            source = reference
        else:
            archive = reference['archive']
            if archive not in self.fds:
                self.fds[archive] = os.open(archive, os.O_RDONLY)
            data = os.pread(self.fds[archive], reference['size'], reference['offset'])
            if len(data) != reference['size']:
                raise ValueError('Short archive read')
            source = io.BytesIO(data)
        with Image.open(source) as image:
            if self.concepts:
                targets = self.cache.get(position)
                if targets is None:
                    labels, masks = np.zeros(len(self.bank), dtype=np.float32), {}
                    for item in json.loads(annotations):
                        c = self.lookup.get(item['concept'])
                        if c is None or item['confidence'] <= self.cfg.data.confidence:
                            continue
                        labels[c] = 1.
                        if item['box'] is not None:
                            mask = occupancy(item['box'], item['coordinates'], image.size,
                                self.cfg.data.resize_size, self.cfg.data.input_size, self.grid)
                            if mask.any():
                                masks[c] = np.maximum(masks.get(c, np.zeros_like(mask)), mask)
                    ids = sorted(masks)
                    targets = (torch.from_numpy(labels), torch.tensor(ids,dtype=torch.long),
                               torch.from_numpy(np.stack([masks[k] for k in ids]) if ids else
                                                np.zeros((0,self.grid,self.grid),dtype=np.float32)))
                    self.cache.put(position, targets)
                tensor = self.transform(image.convert('RGB'))
                return tensor, *targets, category
            return self.transform(image.convert('RGB')), category


def collate(samples):
    images, labels, indices, masks, classes = zip(*samples)
    h, w = masks[0].shape[-2:]
    indices, masks, valid = pad_sparse_targets(indices, masks, mask_h=h, mask_w=w)
    return torch.stack(images), torch.stack(labels), indices, masks, valid, torch.tensor(classes)
