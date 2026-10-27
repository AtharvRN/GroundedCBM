import argparse
import hashlib
import json
from pathlib import Path
import re
import sqlite3

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import functional as TF, InterpolationMode

from spatial_cbm.training.config import load
from spatial_cbm.training.data import Samples, digest
from spatial_cbm.training.metrics import row_ap
from spatial_cbm.training.model import load_selected


GROUPS = {p:[p] for p in ('body','head','neck','beak','tail')}
GROUPS.update(wing=['right_wing','left_wing'],leg=['right_leg','left_leg'],eye=['right_eye','left_eye'])


def canonical(name):
    name = ' '.join(re.sub(r'[-,.()]',' ',name.lower()).split())
    return re.sub(r'^(a|an) ','',name)


def assignments(mapping, bank):
    lookup = {canonical(name):i for i,name in enumerate(bank)}
    if len(lookup) != len(bank): raise ValueError('Ambiguous canonical concept names')
    result = {part:[] for part in GROUPS}
    seen = set()
    for row in mapping['mappings']:
        name,parts = canonical(row['concept']),row['parts']
        if name in seen or not isinstance(parts,list) or len(parts)!=len(set(parts)) or set(parts)-set(GROUPS):
            raise ValueError('Invalid or duplicate concept-to-part mapping')
        seen.add(name)
        if not parts: continue
        if name not in lookup: raise ValueError(f'Mapped concept absent from checkpoint vocabulary: {name}')
        for part in parts: result[part].append(lookup[name])
    if not any(result.values()): raise ValueError('No mapped concepts')
    return result


def select_maps(output, columns, size):
    """Choose within each part using fused scores only, never human overlap."""
    maps,selected = {},{}
    if not torch.isfinite(output['final_logits']).all() or not torch.isfinite(output['spatial_maps']).all():
        raise ValueError('Nonfinite CUB-70 model predictions')
    for part,candidates in columns.items():
        if not candidates: continue
        candidate = torch.tensor(candidates,device=output['final_logits'].device)
        ids = candidate[output['final_logits'][:,candidate].argmax(1)]
        native = output['spatial_maps'][torch.arange(len(ids),device=ids.device),ids,None].float()
        constant = native.flatten(1).amax(1)==native.flatten(1).amin(1)
        score = F.interpolate(native,size=(size,size),mode='bilinear',align_corners=False)[:,0]
        low,high = score.flatten(1).amin(1)[:,None,None],score.flatten(1).amax(1)[:,None,None]
        score = (score-low)/(high-low).clamp_min(1e-12)
        score[constant] = 0
        maps[part],selected[part] = score,ids
    return maps,selected


def evaluate(cfg, mapping_path, output_path, masks_root=None, manifest=None):
    if cfg.dataset != 'cub': raise ValueError('CUB-70 evaluates CUB-trained checkpoints only')
    if (masks_root is None)==(manifest is None): raise ValueError('Provide masks-root or a mask manifest, not both')
    output_path = Path(output_path)
    if output_path.exists(): raise FileExistsError('Use a new CUB-70 result path')
    backbone,head,bank,data,identity = load_selected(cfg)
    calibration_path = Path(cfg.training.output)/'calibration.json'
    calibration = json.loads(calibration_path.read_text())
    if calibration['cohort']!='validation' or calibration['identity']!=identity:
        raise ValueError('CUB-70 requires calibration from the selected checkpoint validation split')
    threshold = calibration['metrics']['localization']['threshold']
    if not 0<=threshold<=1: raise ValueError('Invalid saved localization threshold')
    columns = assignments(json.loads(Path(mapping_path).read_text()),bank)
    with sqlite3.connect(f'file:{Path(cfg.data.prepared)/"test.sqlite"}?mode=ro',uri=True) as db:
        records = {row[1]:dict(position=row[0],id=row[1],image=json.loads(row[2]),class_id=row[3])
                   for row in db.execute('SELECT position,id,image,class_id FROM samples ORDER BY position')}
    roster,missing = [],{part:0 for part in GROUPS}
    if manifest is not None:
        manifest = Path(manifest).resolve()
        seen = set()
        for line in manifest.read_text().splitlines():
            row = json.loads(line)
            if row['id'] in seen or row['id'] not in records or records[row['id']]['class_id']>=70:
                raise ValueError('Mask roster must contain unique held-out CUB-70 image IDs')
            seen.add(row['id'])
            if set(row['masks'])-set(GROUPS): raise ValueError('Unknown human part')
            masks = {}
            for part in GROUPS:
                value = row['masks'].get(part)
                if value is not None and (not isinstance(value,list) or not all(isinstance(p,str) for p in value)):
                    raise ValueError('Each part needs a list of complete component masks, [] for absent, or null for unknown')
                masks[part] = None if value is None else [manifest.parent/Path(p) for p in value]
            roster.append(dict(records[row['id']],masks=masks))
    else:
        root = Path(masks_root)
        for row in records.values():
            if row['class_id']>=70: continue
            masks = {part:[root/str(row['class_id']+1)/f'{Path(row["id"]).stem}_{p}.png' for p in members]
                     for part,members in GROUPS.items()}
            if not any(p.exists() for paths in masks.values() for p in paths): continue
            roster.append(dict(row,masks={part:paths if all(p.exists() for p in paths) else None
                                         for part,paths in masks.items()}))
    if not roster: raise ValueError('No human-annotated held-out images')
    dataset = Samples(cfg,'test',data,concepts=False)
    loader = DataLoader(Subset(dataset,[row['position'] for row in roster]),
        batch_size=cfg.training.batch_size,num_workers=cfg.data.workers,shuffle=False)
    stats = {part:dict(pairs=0,ap=0.,iou=0.,tp=0,fp=0,fn=0,tn=0,absent_after_crop=0) for part in GROUPS}
    selections,offset = [],0
    input_hash = hashlib.sha256()
    with torch.inference_mode():
        for images,_ in loader:
            images = images.to(cfg.training.device)
            if cfg.training.channels_last: images = images.contiguous(memory_format=torch.channels_last)
            maps,chosen = select_maps(head(backbone(images)),columns,cfg.data.input_size)
            for i,row in enumerate(roster[offset:offset+len(images)]):
                if not isinstance(row['image'],str): raise ValueError('CUB-70 requires extracted image files')
                with Image.open(row['image']) as image: original_size = image.size
                input_hash.update(json.dumps([row['id'],digest(row['image'])]).encode())
                for part,paths in row['masks'].items():
                    if paths is None:
                        missing[part]+=1
                        input_hash.update(json.dumps([row['id'],part,None]).encode())
                        continue
                    truth = np.zeros((cfg.data.input_size,cfg.data.input_size),bool)
                    for path in paths:
                        input_hash.update(json.dumps([row['id'],part,str(path),digest(path)]).encode())
                        with Image.open(path) as mask:
                            if mask.size!=original_size: raise ValueError('Human mask and original image geometry differ')
                            mask = TF.center_crop(TF.resize(mask.convert('L'),cfg.data.resize_size,InterpolationMode.NEAREST),
                                                  [cfg.data.input_size,cfg.data.input_size])
                            truth |= np.asarray(mask)>0
                    item = stats[part]
                    if not truth.any(): item['absent_after_crop']+=1; continue
                    score = maps[part][i] if part in maps else torch.zeros(truth.shape,device=images.device)
                    gt = torch.from_numpy(truth).to(images.device)
                    pred = score>=threshold
                    tp,fp,fn,tn = [int(x.sum()) for x in (pred & gt,pred & ~gt,~pred & gt,~pred & ~gt)]
                    ap = float(row_ap(score.flatten()[None],gt.flatten()[None])[0])
                    iou = tp/(tp+fp+fn)
                    item['pairs']+=1; item['ap']+=ap; item['iou']+=iou
                    for key,value in zip(('tp','fp','fn','tn'),(tp,fp,fn,tn)): item[key]+=value
                    selections.append(dict(image=row['id'],part=part,
                        concept=bank[int(chosen[part][i])] if part in chosen else None,pixel_ap=ap,foreground_iou=iou))
            offset+=len(images)
    per_part = {}
    for part,s in stats.items():
        n=s['pairs']; pixels=sum(s[k] for k in ('tp','fp','fn','tn'))
        per_part[part]=dict(pairs=n,pixel_ap=s['ap']/n if n else None,
            mean_foreground_iou=s['iou']/n if n else None,
            pooled_foreground_iou=s['tp']/(s['tp']+s['fp']+s['fn']) if n else None,
            pixel_accuracy=(s['tp']+s['tn'])/pixels if n else None,
            unknown_masks=missing[part],absent_after_crop=s['absent_after_crop'])
    supported = [v for v in per_part.values() if v['pairs']]
    if not supported: raise ValueError('No visible human part masks')
    result = dict(model=identity,mapping_sha256=digest(mapping_path),calibration_sha256=digest(calibration_path),
        evaluator_sha256=digest(__file__),inputs_sha256=input_hash.hexdigest(),threshold=threshold,
        threshold_source='CUB training-pool validation calibration; no human test tuning',
        selection='highest fused score among concepts mapped to each part; ties follow mapping order',
        units='fraction',images=len(roster),heldout_images=len(records),
        heldout_cub70_images=sum(row['class_id']<70 for row in records.values()),
        supported_parts=len(supported),expected_parts=len(GROUPS),per_part=per_part,
        macro_pixel_ap=float(np.mean([v['pixel_ap'] for v in supported])),
        macro_foreground_iou=float(np.mean([v['mean_foreground_iou'] for v in supported])),
        aggregation='equal mean over supported parts, each part averages visible image-part pairs',
        coverage='Unknown/incomplete masks excluded, not treated as background; see per-part coverage',
        warning='Anatomical masks do not verify concept color, texture, or other attribute presence',selections=selections)
    output_path.parent.mkdir(parents=True,exist_ok=True)
    with output_path.open('x') as stream: json.dump(result,stream,indent=2,allow_nan=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--mapping',required=True)
    parser.add_argument('--output',required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--masks-root')
    source.add_argument('--manifest')
    args = parser.parse_args()
    result = evaluate(load(args.config),args.mapping,args.output,args.masks_root,args.manifest)
    print(json.dumps({k:v for k,v in result.items() if k!='selections'},indent=2))


if __name__=='__main__': main()
