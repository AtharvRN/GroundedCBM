"""Frozen native-map exports using the release's checkpoint and classifier loaders."""
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np
from PIL import Image
import torch

from .vlm import digest, file_digest, read, save


def backbone_identity(model):
    weights = None
    if model.weights != 'none':
        from torchvision.models import get_model_weights
        preset = get_model_weights(model.backbone)[model.weights]
        weights = dict(name=f'{type(preset).__name__}.{preset.name}', url=preset.url)
    return dict(backbone=model.backbone, weights=weights,
                backbone_sha256=file_digest(model.checkpoint) if model.checkpoint else None)


def cohort(records, count, seed):
    if count < 1 or count > len(records): raise ValueError('Requested image count exceeds the split')
    groups = {}
    for position, name, label in records:
        groups.setdefault(label,[]).append((position,name,label))
    rank = lambda r: hashlib.sha256((str(seed)+':'+r[1]).encode()).hexdigest()
    for rows in groups.values(): rows.sort(key=rank)
    labels = sorted(groups,key=lambda label:hashlib.sha256((str(seed)+':class:'+str(label)).encode()).hexdigest())
    chosen = []
    for depth in range(max(map(len,groups.values()))):
        for label in labels:
            if depth < len(groups[label]): chosen.append(groups[label][depth])
            if len(chosen)==count: return sorted(chosen,key=lambda r:r[1])
    raise ValueError('Insufficient images')


def export(args):
    from spatial_cbm.training.config import load
    from spatial_cbm.training.model import load_selected
    from spatial_cbm.training.data import Samples
    cfg = load(args.config)
    if cfg.dataset not in ('cub','imagenet','places365'): raise ValueError('Unsupported judge dataset')
    if args.top_k < 1 or args.nec < 1: raise ValueError('Invalid concept budget')
    backbone, head, bank, data, identity = load_selected(cfg)
    root = Path(cfg.training.output)
    classifier = read(args.head_dir/'classifier_identity.json')
    normalization = root/'final_layer_normalization.pt'
    if classifier['checkpoint_sha256'] != identity['checkpoint_sha256'] or classifier['normalization_sha256'] != file_digest(normalization):
        raise ValueError('Classifier belongs to another checkpoint/normalization')
    names = [f'W_g@NEC={args.nec}.pt',f'b_g@NEC={args.nec}.pt']
    for name in names:
        if classifier['head_files'][name] != file_digest(args.head_dir/name): raise ValueError('Changed classifier head')
    device = cfg.training.device
    weight, bias = [torch.load(args.head_dir/name,map_location=device,weights_only=True).float() for name in names]
    norm = torch.load(normalization,map_location=device,weights_only=True)
    mean, std = norm['mean'].float().reshape(-1), norm['std'].float().reshape(-1)
    if weight.shape != (data['classes'],len(bank)) or bias.shape != (data['classes'],) or mean.shape != std.shape or len(mean)!=len(bank):
        raise ValueError('Classifier/concept dimensions differ')
    if not all(torch.isfinite(v).all() for v in (weight,bias,mean,std)) or not (std>0).all(): raise ValueError('Invalid classifier tensors')
    dataset = Samples(cfg,args.split,data,concepts=False)
    with sqlite3.connect(f'file:{cfg.data.prepared}/{dataset.database}?mode=ro',uri=True) as db:
        lookup = {position:(name,label) for position,name,label in db.execute('SELECT position,id,class_id FROM samples')}
    records = [(i,*lookup[int(position)]) for i,position in enumerate(dataset.indices)]
    chosen = cohort(records,args.count,args.seed)
    args.output.mkdir(parents=True,exist_ok=False)
    protocol = dict(dataset=cfg.dataset,method=cfg.model.method,checkpoint_sha256=identity['checkpoint_sha256'],
        **backbone_identity(cfg.model),normalization_sha256=file_digest(normalization),
        classifier_files={name:file_digest(args.head_dir/name) for name in names},nec=args.nec,
        concept_bank_sha256=digest(json.dumps(bank).encode()),top_k=args.top_k,
        concept_selection='Top positive standardized-activation times predicted-class weight; no activation-sign or correctness filter',
        preprocessing=dict(resize=cfg.data.resize_size,crop=cfg.data.input_size),source_sha256=file_digest(__file__))
    rows = []; no_concepts = []
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    with torch.inference_mode():
        for start in range(0,len(chosen),16):
            chunk = chosen[start:start+16]
            pixels = torch.stack([dataset[r[0]][0] for r in chunk]).to(device)
            if cfg.training.channels_last: pixels=pixels.contiguous(memory_format=torch.channels_last)
            output = head(backbone(pixels)); raw=output['final_logits'].float(); maps=output['spatial_maps']
            if maps is None: raise ValueError('Model supplies no native maps; use an explicit saliency adapter')
            if not torch.isfinite(raw).all() or not torch.isfinite(maps).all(): raise ValueError('Nonfinite model output')
            z=(raw-mean)/std; logits=torch.nn.functional.linear(z,weight,bias)
            rgb=(pixels.float()*pixels.new_tensor((.229,.224,.225))[None,:,None,None]+
                 pixels.new_tensor((.485,.456,.406))[None,:,None,None]).clamp(0,1)
            for i,(_,name,label) in enumerate(chunk):
                image_id=digest(name.encode())[:24];folder=args.output/'images'/image_id;folder.mkdir(parents=True)
                clean=folder/'clean.png';Image.fromarray(rgb[i].mul(255).round().byte().cpu().numpy().transpose(1,2,0)).save(clean)
                predicted=int(logits[i].argmax());contribution=z[i]*weight[predicted]
                order=torch.argsort(contribution,descending=True,stable=True)
                selected=order[contribution[order]>0][:args.top_k].tolist()
                if not selected:no_concepts.append(name)
                for rank,column in enumerate(selected,1):
                    native=folder/f'concept{column}.npy';np.save(native,maps[i,column].float().cpu().numpy())
                    rows.append(dict(id=image_id+':'+str(column),image_id=name,concept=bank[column],concept_index=column,
                        method=cfg.model.method,rank=rank,predicted_class=predicted,class_id=label,
                        raw=float(raw[i,column]),normalized=float(z[i,column]),
                        weight=float(weight[predicted,column]),contribution=float(contribution[column]),
                        clean=str(clean.relative_to(args.output)),native=str(native.relative_to(args.output)),
                        clean_sha256=file_digest(clean),native_sha256=file_digest(native)))
            print(json.dumps(dict(stage='export',completed=start+len(chunk),total=len(chosen))),flush=True)
    manifest=args.output/'manifest.jsonl';manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    metadata=dict(identity=protocol,split=args.split,images=len(chosen),pairs=len(rows),
        image_ids=[r[1] for r in chosen],images_without_positive_contributions=no_concepts,
        cohort_selection=dict(rule='Hash-ranked class-balanced round robin before inference',seed=args.seed),
        manifest_sha256=file_digest(manifest))
    save(args.output/'export.json',metadata)
    return metadata
