"""FP32 concept features for shared sparse-classifier fitting."""
import json
import fcntl
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import Samples, collate, digest, save_json, verify
from .engine import atomic_torch, provenance
from .model import Backbone, ConceptHead


def export(cfg):
    root = Path(cfg.training.output)
    with (root/'export.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _export(cfg)


def _export(cfg):
    if cfg.training.stage != 'concept':
        raise ValueError('Feature export requires a concept-training configuration')
    root = Path(cfg.training.output)
    if not (root/'complete.json').is_file():
        raise ValueError('Finish training before exporting the selected checkpoint')
    data = verify(cfg.data.prepared)
    selected = torch.load(root/'best.pt', map_location='cpu', weights_only=False)
    if selected['protocol']['config'] != cfg.asdict() or selected['protocol']['data'] != data:
        raise ValueError('Feature extraction protocol differs from selected checkpoint')
    current = provenance(cfg, data)
    for key in ('source', 'backbone_sha256'):
        if selected['protocol'][key] != current[key]:
            raise ValueError(f'Feature extraction provenance changed: {key}')
    identity = dict(checkpoint=digest(root/'best.pt'), data=data, precision='fp32')
    if (root/'features_complete.json').exists():
        previous = json.loads((root/'features_complete.json').read_text())
        if previous['identity'] != identity:
            raise ValueError('Feature cache belongs to another model/data protocol')
        for name, checksum in previous['files'].items():
            if digest(root/name) != checksum:
                raise ValueError('Feature cache changed')
        return previous
    device = cfg.training.device
    backbone = Backbone(cfg.model, data['classes'], data).to(device).eval().requires_grad_(False)
    concepts = (Path(cfg.data.prepared)/'concepts.txt').read_text()
    count = len(concepts.splitlines())
    head = ConceptHead(cfg.model, backbone.dims, count).to(device).eval()
    head.load_state_dict(selected['model'])
    if cfg.training.channels_last:
        backbone.to(memory_format=torch.channels_last); head.to(memory_format=torch.channels_last)
    folder = root/'features'; folder.mkdir(exist_ok=True)
    for split, prefix in [('train','train'), ('validation','val')]:
        dataset = Samples(cfg, split, data)
        loader = DataLoader(dataset, batch_size=cfg.training.batch_size, shuffle=False,
                            num_workers=cfg.data.workers, collate_fn=collate)
        features = np.lib.format.open_memmap(folder/f'{prefix}_features.npy', mode='w+', dtype='float32', shape=(len(dataset),count))
        targets = np.lib.format.open_memmap(folder/f'{prefix}_targets.npy', mode='w+', dtype='int64', shape=(len(dataset),))
        total, squares, offset = np.zeros(count), np.zeros(count), 0
        with torch.inference_mode():
            for batch in loader:
                images = batch[0].to(device)
                if cfg.training.channels_last:
                    images = images.contiguous(memory_format=torch.channels_last)
                scores = head(backbone(images))['final_logits'].float().cpu().numpy()
                if not np.isfinite(scores).all():
                    raise ValueError('Nonfinite classifier features')
                end = offset + len(images)
                features[offset:end] = scores; targets[offset:end] = batch[-1].numpy()
                total += scores.sum(0, dtype=np.float64)
                squares += np.square(scores.astype(np.float64)).sum(0)
                offset = end
        assert offset == len(dataset)
        features.flush(); targets.flush()
        if split == 'train':
            if offset < 2:
                raise ValueError('Normalization needs at least two training images')
            mean = total/offset
            std = np.sqrt(np.maximum((squares-offset*mean*mean)/(offset-1),0)).clip(1e-6)
            atomic_torch(root/'final_layer_normalization.pt', dict(mean=torch.tensor(mean,dtype=torch.float32),
                                                                  std=torch.tensor(std,dtype=torch.float32)))
    (root/'concepts.txt').write_text(concepts)
    files = [str(p.relative_to(root)) for p in folder.glob('*.npy')] + ['final_layer_normalization.pt']
    result = dict(identity=identity, files={n:digest(root/n) for n in files},
                  normalization='training-only sample standard deviation', validation='internal holdout, never test')
    save_json(root/'features_complete.json', result)
    return result
