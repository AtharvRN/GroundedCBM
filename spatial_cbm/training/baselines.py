"""Baseline-specific heads, supervision and filtering; orchestration stays shared.

Equations and behavior are cross-checked against pinned sources in docs/baselines.md.
The centered-cubic similarity is cosine AFTER cubing, not cubed cosine.
"""
import fcntl
import json
from pathlib import Path
import sqlite3
import tempfile

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import BatchSampler, Dataset
from torchvision.transforms import functional as TF


REGRESSION = {'lf_cbm', 'salf_cbm'}


def concept_key(text):
    text = text.lower().translate(str.maketrans({c:' ' for c in '-,.()'}))
    for article in ('a ', 'an '):
        if text.startswith(article):
            text = text[len(article):]
            break
    return ' '.join(text.split())


def concept_lookup(bank):
    lookup = {concept_key(c):i for i,c in enumerate(bank)}
    if len(lookup) != len(bank):
        raise ValueError('VLG vocabulary contains duplicate normalized concept names')
    return lookup


def cubic_similarity(prediction, target):
    if prediction.shape != target.shape or prediction.ndim not in (2,4) or len(prediction) < 2:
        raise ValueError('Centered-cubic similarity needs matching [N,C,(H,W)] tensors, N>=2')
    def vectors(x):
        x = x.float() if x.dtype in (torch.float16,torch.bfloat16) else x
        centered = x-x.mean(0,keepdim=True)
        # Positive rescaling before cubing preserves cosine and prevents underflow.
        return F.normalize(F.normalize(centered,dim=0,eps=1e-12).pow(3),dim=0,eps=1e-8)
    return (vectors(prediction)*vectors(target)).sum(0)


class BaselineHead(nn.Module):
    def __init__(self, cfg, dims, concepts, indices=None):
        super().__init__()
        indices = torch.arange(concepts) if indices is None else torch.as_tensor(indices,dtype=torch.long).cpu()
        if not len(indices) or indices.ndim != 1 or len(indices.unique()) != len(indices) or (indices<0).any() or (indices>=concepts).any():
            raise ValueError('Invalid retained concept indices')
        self.register_buffer('concept_indices',indices.clone())
        self.method, self.grid = cfg.method, cfg.salf_grid
        self.fusion = 'spatial' if cfg.method=='salf_cbm' else 'global'
        count = len(indices)
        if cfg.method=='salf_cbm':
            self.projection = nn.Conv2d(dims[-1],count,1,bias=False)
            self.map_kind = 'native'
        else:
            self.projection = nn.Linear(dims[-1],count,bias=cfg.method=='vlg_cbm')
            self.hidden = nn.Sequential(*[layer for _ in range(cfg.vlg_hidden if cfg.method=='vlg_cbm' else 0)
                                         for layer in (nn.ReLU(),nn.Linear(count,count))])
            self.map_kind = 'unavailable' if len(self.hidden) else 'linear_cam'

    def forward(self, features):
        spatial = features['conv5']
        if self.method=='salf_cbm':
            resized = F.interpolate(spatial,size=(self.grid,self.grid),mode='bilinear',align_corners=False)
            maps = self.projection(resized)
            flat = maps.flatten(2)
            scores = (flat.softmax(-1)*flat).sum(-1)
        else:
            scores = self.hidden(self.projection(spatial.mean((2,3))))
            maps = (F.conv2d(spatial,self.projection.weight[:,:,None,None],self.projection.bias)
                    if self.map_kind=='linear_cam' else None)
        return dict(final_logits=scores,spatial_maps=maps,map_kind=self.map_kind)


def subset_targets(batch, indices):
    """Remap sparse annotation IDs after concept filtering, retaining padded geometry."""
    if len(batch)<6: return batch
    batch = list(batch)
    lookup = torch.full((batch[1].shape[1],),-1,device=indices.device,dtype=torch.long)
    lookup[indices] = torch.arange(len(indices),device=indices.device)
    mapped = lookup[batch[2].clamp_min(0)]
    batch[1] = batch[1].index_select(1,indices)
    batch[4] = batch[4] & (mapped>=0)
    batch[2] = mapped.clamp_min(0)
    return batch


def vlg_crop(image, annotations, lookup, confidence, probability):
    """Upstream concept-uniform crop selection and IoU>0.5 multi-label update."""
    groups = {}
    for item in annotations:
        key = concept_key(item['concept'])
        if key not in lookup or item['confidence']<=confidence: continue
        groups.setdefault(lookup[key],[])
        if item['box'] is None: continue
        box = np.asarray(item['box'],dtype=float)
        if item['coordinates']=='normalized_xyxy': box *= [image.width,image.height]*2
        groups[lookup[key]].append(box)
    labels = torch.zeros(len(lookup))
    if groups: labels[list(groups)] = 1
    eligible = {c:boxes for c,boxes in groups.items() if boxes}
    if np.random.random() >= probability or not eligible: return image,labels
    boxes = {c:items[np.random.randint(len(items))] for c,items in eligible.items()}
    concept = int(np.random.choice(list(boxes)))
    selected = boxes[concept]
    cropped = image.crop(tuple(selected))
    if min(cropped.size)<1: return image,labels
    labels.zero_(); labels[concept] = 1
    for c, candidates in eligible.items():
        for box in candidates:
            intersection = np.maximum(0,np.minimum(selected[2:],box[2:])-np.maximum(selected[:2],box[:2])).prod()
            union = (selected[2:]-selected[:2]).prod()+(box[2:]-box[:2]).prod()-intersection
            if union>0 and intersection/union>.5: labels[c] = 1; break
    return cropped,labels


class RegressionSamples(Dataset):
    def __init__(self, cfg, split, protocol, metrics=False, teacher_manifest=None, feature_paths=None):
        from .data import Samples
        from .teacher import TeacherTargets
        annotated = protocol['annotated_images']==protocol['images']
        self.base = Samples(cfg,split,protocol,concepts=metrics and annotated,metrics=metrics and annotated,
                            image_augmentation=False)
        self.teacher = TeacherTargets(cfg,protocol,verified_manifest=teacher_manifest)
        self.indices = self.base.indices
        self.feature_paths = feature_paths
        self.cached_features = feature_paths is not None
        if self.cached_features: self._open_features()

    def _open_features(self):
        self.features,self.classes = [np.load(p,mmap_mode='r',allow_pickle=False) for p in self.feature_paths]
        if len(self.features)!=len(self.base) or self.classes.shape!=(len(self.base),):
            raise ValueError('Frozen feature cache cohort mismatch')

    def __getstate__(self):
        return {k:v for k,v in self.__dict__.items() if k not in {'features','classes'}}

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.cached_features: self._open_features()

    def __len__(self): return len(self.base)

    def __getitem__(self, index):
        if self.cached_features:
            feature = torch.from_numpy(np.array(self.features[index],copy=True))
            base = ((feature,int(self.classes[index])) if not self.base.concepts else
                    (feature,*self.base[index][1:]))
        else:
            base = self.base[index]
        return (*base[:-1],self.teacher.get(int(self.indices[index])),base[-1])


def prepare_regression_features(cfg, backbone, protocol):
    """Cache deterministic frozen features; never cache VLG's random crops."""
    import hashlib
    if not cfg.training.cache_regression_features: return None
    identity = dict(version=1,data=protocol['data'],source=protocol['source'],model=cfg.asdict()['model'],
        backbone_sha256=protocol['backbone_sha256'],resize=cfg.data.resize_size,crop=cfg.data.input_size,
        precision='fp32',pool_train=cfg.model.method=='lf_cbm')
    key = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    root = Path(cfg.data.cache)/'regression-features'/key
    root.parent.mkdir(parents=True,exist_ok=True)
    # Keep the lock outside the atomically published directory, on a stable inode.
    with (root.parent/f'{key}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return _prepare_regression_features(cfg,backbone,protocol,identity,root)


def _prepare_regression_features(cfg, backbone, protocol, identity, root):
    from .data import digest, save_json
    marker = root/'complete.json'
    paths = {s:[str(root/f'{s}-{n}.npy') for n in ('features','classes')] for s in ('train','validation')}
    if marker.exists():
        saved = json.loads(marker.read_text())
        if saved['identity']!=identity: raise ValueError('Frozen feature cache identity changed')
        for name,expected in saved['files'].items():
            if digest(root/name)!=expected: raise ValueError('Frozen feature cache content changed')
        return paths
    if root.exists():
        if any(root.iterdir()): raise ValueError('Partial legacy frozen feature cache; use a fresh cache directory')
        root.rmdir()
    with tempfile.TemporaryDirectory(prefix=f'.{root.name}-',dir=root.parent) as temporary:
        staging = Path(temporary)
        staged_paths = {s:[str(staging/Path(p).name) for p in pair] for s,pair in paths.items()}
        _build_regression_features(cfg,backbone,protocol,staged_paths)
        save_json(staging/'complete.json',dict(identity=identity,
            files={Path(p).name:digest(p) for pair in staged_paths.values() for p in pair}))
        staging.replace(root)
    return paths


def _build_regression_features(cfg, backbone, protocol, paths):
    from torch.utils.data import DataLoader
    from .data import Samples
    backbone.eval().requires_grad_(False)
    for split in paths:
        dataset = Samples(cfg,split,protocol['data'],concepts=False,image_augmentation=False)
        loader = DataLoader(dataset,batch_size=cfg.training.feature_batch_size,
                            num_workers=cfg.data.workers,shuffle=False)
        offset,features = 0,None
        classes = np.lib.format.open_memmap(paths[split][1],mode='w+',dtype='int64',shape=(len(dataset),))
        with torch.inference_mode(), torch.autocast(device_type=torch.device(cfg.training.device).type,enabled=False):
            for images,labels in loader:
                images = images.to(cfg.training.device)
                if cfg.training.channels_last: images = images.contiguous(memory_format=torch.channels_last)
                values = backbone(images)['conv5'].float()
                if split=='train' and cfg.model.method=='lf_cbm': values = values.mean((2,3),keepdim=True)
                if not torch.isfinite(values).all(): raise ValueError('Nonfinite cached backbone features')
                if features is None:
                    features = np.lib.format.open_memmap(paths[split][0],mode='w+',dtype='float32',
                        shape=(len(dataset),*values.shape[1:]))
                end = offset+len(images)
                features[offset:end] = values.cpu().numpy()
                classes[offset:end] = labels.numpy()
                offset = end
        if offset!=len(dataset) or features is None: raise ValueError('Incomplete frozen feature cache')
        features.flush(); classes.flush()
        print(json.dumps(dict(stage='frozen_features_cached',split=split,images=offset,shape=list(features.shape))),flush=True)


def regression_collate(samples):
    from .data import collate
    from torch.utils.data import default_collate
    base = [(*sample[:-2],sample[-1]) for sample in samples]
    values = list(collate(base) if len(base[0])>2 else default_collate(base))
    return (*values[:-1],torch.stack([sample[-2] for sample in samples]),values[-1])


class NonSingletonBatches(BatchSampler):
    """Merge tails below three: a centered pair has no useful regression gradient."""
    def __iter__(self):
        previous = None
        for batch in super().__iter__():
            if previous is not None:
                if len(batch)<3: yield previous+batch; return
                yield previous
            previous = batch
        if previous is not None: yield previous

    def __len__(self):
        count = super().__len__()
        return count-int(count>1 and 0<len(self.sampler)%self.batch_size<3)


class RegressionMeter:
    """Full-cohort centering for validation, with disk-backed bounded-memory scores."""
    def __init__(self, rows, cache):
        if rows<2: raise ValueError('Regression validation requires at least two images')
        Path(cache).mkdir(parents=True,exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(prefix='regression-',dir=cache)
        self.rows, self.offset, self.arrays = rows,0,[]

    def update(self, prediction, target):
        if not self.arrays:
            self.arrays = [np.lib.format.open_memmap(Path(self.directory.name)/f'{i}.npy',mode='w+',dtype='float32',
                shape=(self.rows,*prediction.shape[1:])) for i in range(2)]
        end = self.offset+len(prediction)
        for value,array in zip((prediction,target),self.arrays):
            if not torch.isfinite(value).all(): raise ValueError('Nonfinite regression predictions/targets')
            array[self.offset:end] = value.detach().float().cpu().numpy()
        self.offset = end

    def similarities(self):
        if self.offset != self.rows: raise ValueError('Incomplete regression validation cohort')
        shape = self.arrays[0].shape
        chunk = max(1,(32<<20)//(self.rows*int(np.prod(shape[2:]))*8))
        result = []
        for start in range(0,shape[1],chunk):
            values = [torch.from_numpy(np.array(a[:,start:start+chunk],copy=True)).double() for a in self.arrays]
            result.append(cubic_similarity(*values))
        return torch.cat(result)

    def close(self):
        self.arrays.clear(); self.directory.cleanup()


def initial_indices(cfg, protocol, teacher_manifest=None):
    """Training-only vocabulary filters; never use held-out detections or scores."""
    root = Path(cfg.data.prepared)
    bank = (root/'concepts.txt').read_text().splitlines()
    if not bank: raise ValueError('Baseline training requires an explicit concept vocabulary')
    if not cfg.training.filter_concepts or cfg.model.method in {'sg_cbm','salf_cbm'}: return None
    indices = np.load(root/'indices.npz')['train']
    if cfg.model.method=='vlg_cbm':
        positions,lookup,keep = set(indices.tolist()),concept_lookup(bank),torch.zeros(len(bank),dtype=torch.bool)
        with sqlite3.connect(f'file:{root}/samples.sqlite?mode=ro',uri=True) as db:
            for position,raw in db.execute('SELECT position,detections FROM samples'):
                if position not in positions: continue
                items = json.loads(raw)
                if items is None: raise ValueError('VLG requires detector annotations for every training image')
                for item in items:
                    key = concept_key(item['concept'])
                    if key in lookup and item['confidence']>cfg.data.confidence: keep[lookup[key]]=True
    else:
        from .teacher import TeacherTargets
        teacher = TeacherTargets(cfg,protocol,verified_manifest=teacher_manifest)
        best = torch.empty((0,len(bank)))
        for start in range(0,len(indices),256):
            targets = torch.stack([teacher.get(int(i)) for i in indices[start:start+256]])
            combined = torch.cat((best,targets))
            best = combined.topk(min(5,len(combined)),dim=0).values
        keep = best.mean(0)>cfg.training.lf_clip_cutoff
    result = keep.nonzero().flatten()
    if not len(result): raise ValueError('No concepts survive the training-only filter')
    return result


def finalize_lf(cfg, backbone, protocol, teacher_manifest=None, cached_features=None):
    """Apply the upstream interpretability filter to the selected projection only."""
    if not cfg.training.filter_concepts:
        return
    from .data import digest, save_json
    from .engine import atomic_torch, loaders, run_epoch
    from .model import ConceptHead
    root = Path(cfg.training.output)
    receipt = root/'concept_filter.json'
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        if saved['checkpoint_sha256'] != digest(root/'best.pt'): raise ValueError('Changed filtered LF checkpoint')
        return
    state = torch.load(root/'best.pt',map_location='cpu',weights_only=False)
    count = len((Path(cfg.data.prepared)/'concepts.txt').read_text().splitlines())
    head = ConceptHead(cfg.model,backbone.dims,count,state['model']['concept_indices']).to(cfg.training.device)
    head.load_state_dict(state['model'])
    _,val = loaders(cfg,protocol,teacher_manifest=teacher_manifest,cached_features=cached_features)
    meter = RegressionMeter(len(val.dataset),cfg.data.cache)
    try:
        run_epoch(backbone,head,val,cfg,regression_meter=meter)
        similarities = meter.similarities()
    finally:
        meter.close()
    keep = similarities>cfg.training.lf_interpretability_cutoff
    if not keep.any(): raise ValueError('No LF concepts survive validation interpretability filtering')
    old = state['model']['concept_indices']
    state['model']['concept_indices'] = old[keep]
    state['model']['projection.weight'] = state['model']['projection.weight'][keep]
    atomic_torch(root/'best.pt',state)
    save_json(receipt,dict(checkpoint_sha256=digest(root/'best.pt'),cohort='internal validation',
        before=old.tolist(),retained=old[keep].tolist(),similarity=similarities.tolist(),
        cutoff=cfg.training.lf_interpretability_cutoff,comparison='>',
        definition='image-axis centered elementwise-cubic cosine; selected pre-filter validation loss retained'))
