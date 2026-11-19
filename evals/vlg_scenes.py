"""Pinned ImageNet/Places365 VLG exports, including nonlinear concept Grad-CAM."""
import argparse
from collections import Counter
import fcntl
import hashlib
import json
from pathlib import Path
import sqlite3
from contextlib import closing

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from . import vlm


def verified(path, expected):
    if vlm.file_digest(path) != expected: raise ValueError('Changed input: '+str(path))


def selected_concepts(raw, mean, std, weight, bias):
    normalized = (raw-mean)/std
    logits = F.linear(normalized,weight,bias)
    if not torch.isfinite(normalized).all() or not torch.isfinite(logits).all():
        raise ValueError('Nonfinite predictions')
    predicted = logits.argmax(1)
    contribution = normalized*weight[predicted]
    selected = []
    for row in contribution:
        order = torch.argsort(row,descending=True,stable=True)
        selected.append(order[row[order]>0][:5].tolist())
    return predicted,normalized,contribution,selected


def gradcam(head, spatial, columns):
    """VJP through the actual pooled concept head, independently for each rank."""
    if len(columns) != len(spatial) or any(len(row)>5 for row in columns):
        raise ValueError('Invalid selected concepts')
    with torch.enable_grad():
        pooled = spatial.detach().mean((-2,-1)).requires_grad_(True)
        scores = head.hidden(head.projection(pooled))
        if not torch.isfinite(scores).all(): raise ValueError('Nonfinite concept output')
        output = [[] for _ in columns]
        for rank in range(max(map(len,columns),default=0)):
            indices = [i for i,row in enumerate(columns) if len(row)>rank]
            ids = [columns[i][rank] for i in indices]
            if any(column<0 or column>=scores.shape[1] for column in ids):
                raise ValueError('Invalid target concept')
            gradient = torch.autograd.grad(scores[indices,ids].sum(),pooled,retain_graph=True)[0]
            weights = gradient[:,:,None,None]/spatial.shape[-2:].numel()
            maps = F.relu((spatial.detach()*weights).sum(1)).detach()
            if not torch.isfinite(maps).all(): raise ValueError('Nonfinite Grad-CAM')
            for i in indices: output[i].append(maps[i])
    return scores.detach(),output


def validate_cohorts(dataset, calibration, test, labels):
    if dataset not in ('imagenet','places365'): raise ValueError('Unsupported dataset')
    for rows,count in ((calibration,200),(test,5000)):
        names = [r['filename'] for r in rows]
        if len(names) != len(set(names)) or len(names) != count:
            raise ValueError('Incomplete or duplicated cohort')
        for row in rows:
            path = Path(row['filename'])
            if path.is_absolute() or len(path.parts)!=1 or labels.get(row['filename']) != row['label']:
                raise ValueError('Filename/class differs from official validation roster')
    if {r['filename'] for r in calibration}&{r['filename'] for r in test}:
        raise ValueError('Calibration/test overlap')
    if len({r['label'] for r in calibration}) != 200:
        raise ValueError('Expected 200 distinct calibration classes')
    counts = Counter(r['label'] for r in test)
    if dataset=='imagenet' and counts != {i:5 for i in range(1000)}:
        raise ValueError('Expected five test images per ImageNet class')
    if dataset=='places365' and (set(counts)!=set(range(365)) or Counter(counts.values())!={14:255,13:110}):
        raise ValueError('Expected original balanced Places365 cohort')


def cohorts(profile):
    database = Path(profile['label_database'])
    verified(database,profile['label_database_sha256'])
    with closing(sqlite3.connect(f'file:{database}?mode=ro',uri=True)) as db:
        records = db.execute('SELECT id,class_id FROM samples').fetchall()
    labels = {Path(name).name:label for name,label in records}
    if len(labels)!=len(records) or len(labels)!={'imagenet':50000,'places365':36500}[profile['dataset']]:
        raise ValueError('Official validation filename roster differs')
    if profile['dataset']=='imagenet':
        for split in ('calibration','test'): verified(profile[split+'_cohort'],profile[split+'_cohort_sha256'])
        calibration = vlm.read(profile['calibration_cohort'])['images']
        test = vlm.read(profile['test_cohort'])['images']
        if vlm.read(profile['calibration_cohort'])['excluded_cohort_sha256'] != profile['test_cohort_sha256']:
            raise ValueError('Calibration excludes another test cohort')
    else:
        verified(profile['cohorts'],profile['cohorts_sha256'])
        splits = vlm.read(profile['cohorts'])['splits'];calibration,test = splits['calibration'],splits['test']
    validate_cohorts(profile['dataset'],calibration,test,labels)
    return dict(validation=calibration,test=test)


def load_model(profile, source, device):
    from spatial_cbm.training.config import Model
    from spatial_cbm.training.model import Backbone, ConceptHead
    root,heads = Path(profile['model_root']),Path(profile['classifier_root'])
    verified(root/'best.pt',profile['checkpoint_sha256'])
    verified(root/'final_layer_normalization.pt',profile['normalization_sha256'])
    verified(profile['training_archive'],profile['training_archive_sha256'])
    verified(profile['backbone'],profile['backbone_sha256'])
    verified(heads/'classification_test.json',profile['classification_sha256'])
    selected = torch.load(root/'best.pt',map_location='cpu',weights_only=False)
    protocol = selected['protocol']
    for name,digest in protocol['source'].items(): verified(source/name,digest)
    if protocol['backbone_sha256'] != profile['backbone_sha256']:
        raise ValueError('Backbone does not belong to the concept checkpoint')
    cfg = Model(**protocol['config']['model'])
    if cfg.method!='vlg_cbm' or cfg.vlg_hidden!=profile['hidden_layers']:
        raise ValueError('Unexpected concept architecture')
    bank = (root/'concepts.txt').read_text().splitlines()
    if len(bank)!=profile['concepts'] or len(set(bank))!=len(bank): raise ValueError('Concept bank differs')
    if not torch.equal(selected['model']['concept_indices'],torch.arange(len(bank))):
        raise ValueError('Scene adapter requires an explicitly unfiltered bank')
    classifier = vlm.read(heads/'classifier_identity.json')
    report = vlm.read(heads/'classification_test.json')
    if (classifier['checkpoint_sha256']!=profile['checkpoint_sha256'] or
            classifier['normalization_sha256']!=profile['normalization_sha256'] or
            classifier['feature_manifest_sha256']!=vlm.file_digest(root/'features_complete.json') or
            report['identity']['model']['checkpoint_sha256']!=profile['checkpoint_sha256'] or
            report['identity']['classifier']!=classifier or
            report['images']!={'imagenet':50000,'places365':36500}[profile['dataset']]):
        raise ValueError('Classifier/reference evaluation identity differs')
    if (heads/'concepts.txt').read_text().splitlines()!=bank: raise ValueError('Classifier concept order differs')
    tensors = []
    for prefix,label in (('W_g','weight'),('b_g','bias')):
        name = prefix+'@NEC=5.pt';path = heads/name
        verified(path,profile[label+'_sha256'])
        if classifier['head_files'][name]!=profile[label+'_sha256']: raise ValueError('Head identity differs')
        tensors.append(torch.load(path,map_location=device,weights_only=True).float())
    weight,bias = tensors
    norm = torch.load(root/'final_layer_normalization.pt',map_location=device,weights_only=True)
    mean,std = norm['mean'].reshape(-1).float(),norm['std'].reshape(-1).float()
    if (weight.shape!=(profile['classes'],len(bank)) or bias.shape!=(profile['classes'],) or
            mean.shape!=(len(bank),) or std.shape!=mean.shape or int(weight.count_nonzero())!=5*profile['classes']):
        raise ValueError('Invalid exact NEC=5 classifier')
    if not all(torch.isfinite(v).all() for v in (weight,bias,mean,std)) or not (std>0).all():
        raise ValueError('Invalid classifier values')
    cfg.checkpoint = profile['backbone']
    backbone = Backbone(cfg,profile['classes'],protocol['data']).to(device).eval().requires_grad_(False)
    head = ConceptHead(cfg,backbone.dims,len(bank),selected['model']['concept_indices']).to(device).eval().requires_grad_(False)
    head.load_state_dict(selected['model'],strict=True)
    if protocol['config']['training']['channels_last']:
        backbone.to(memory_format=torch.channels_last);head.to(memory_format=torch.channels_last)
    if protocol['config']['data']['resize_size']!=256 or protocol['config']['data']['input_size']!=224:
        raise ValueError('Expected original 256/224 preprocessing')
    return backbone,head,bank,(weight,bias,mean,std),protocol,classifier


def publish_completion(output, identity, splits):
    ready = {}
    for split in splits:
        path = output/split/'export.json'
        if not path.exists(): continue
        if vlm.read(path)['identity']!=identity: raise ValueError('Split export identity differs')
        ready[split] = vlm.file_digest(path)
    done = dict(complete=len(ready)==len(splits),smoke=identity['smoke'],
        protocol_sha256=vlm.file_digest(output/'export_protocol.json'),exports=ready,
        cohort_counts={s:len(rows) for s,rows in splits.items()})
    vlm.save(output/'export_status.json',done)
    if done['complete']: vlm.save(output/'complete.json',done)
    return done


def export(args):
    from torchvision import transforms
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    profile = vlm.read(args.profiles)[args.dataset]
    splits = cohorts(profile)
    presence,localization = vlm.prompts(args.dataset)
    original = vlm.read(profile['calibration_prompt_reference'])
    if original['presence_prompt']!=presence or original['localization_prompt']!=localization:
        raise ValueError('Prompts differ from prior dataset evaluation')
    backbone,head,bank,tensors,training,classifier = load_model(profile,args.source,args.device)
    weight,bias,mean,std = tensors
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'export.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        identity = dict(dataset=args.dataset,method='vlg_cbm',nec=5,top_k=5,smoke=args.smoke,
            profile=profile,profiles_sha256=vlm.file_digest(args.profiles),classifier=classifier,
            source_sha256=vlm.file_digest(__file__),device=args.device,
            training_source=training['source'],prompt_reference_sha256=vlm.file_digest(profile['calibration_prompt_reference']),
            maps='Post-hoc raw-concept-logit Grad-CAM at conv5; VJP through actual projection and nonlinear head; ReLU; no class-weight multiplication',
            preprocessing=dict(resize=256,crop=224),native_maps=False,
            concept_selection='Up to five positive standardized-activation times predicted-class-weight contributions; no activation-sign or correctness filter',
            cohort_caveat='Disjoint subsets of official validation, not an untouched official test set or a new independent checkpoint-selection study')
        pin = args.output/'export_protocol.json'
        if pin.exists() and vlm.read(pin)!=identity: raise ValueError('Export identity changed')
        vlm.save(pin,identity)
        crop = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224)])
        tensor = transforms.Compose([transforms.ToTensor(),transforms.Normalize((.485,.456,.406),(.229,.224,.225))])
        requested = tuple(splits) if args.split=='both' else (args.split,)
        for split in requested:
            all_rows = splits[split]
            rows = all_rows[:2] if args.smoke else all_rows
            out = args.output/split;out.mkdir(exist_ok=True)
            if (out/'export.json').exists():
                _,metadata,_ = vlm.load_export(out)
                if metadata['identity']!=identity: raise ValueError('Saved export changed')
                continue
            pairs,predictions,empty = [],[],[]
            for start in range(0,len(rows),args.batch_size):
                batch = rows[start:start+args.batch_size];images=[];input_hashes=[]
                for row in batch:
                    path = Path(profile['images'])/row['filename'];checksum = vlm.file_digest(path)
                    if row.get('sha256',checksum)!=checksum: raise ValueError('Original image changed')
                    input_hashes.append(checksum)
                    with Image.open(path) as image: images.append(crop(image.convert('RGB')))
                pixels = torch.stack([tensor(image) for image in images]).to(args.device)
                if training['config']['training']['channels_last']:
                    pixels = pixels.contiguous(memory_format=torch.channels_last)
                with torch.no_grad(): spatial = backbone(pixels)['conv5']
                raw,maps = gradcam(head,spatial,[[] for _ in batch])
                predicted,z,contributions,chosen = selected_concepts(raw,mean,std,weight,bias)
                checked,maps = gradcam(head,spatial,chosen)
                torch.testing.assert_close(raw,checked)
                if start==0:
                    feature = spatial.detach().requires_grad_(True)
                    direct = head(dict(conv5=feature))['final_logits']
                    torch.testing.assert_close(raw,direct.detach())
                    for i,ids in enumerate(chosen):
                        if not ids: continue
                        gradient = torch.autograd.grad(direct[i,ids[0]],feature)[0].mean((-2,-1),keepdim=True)
                        expected = F.relu((feature.detach()*gradient).sum(1))[i]
                        torch.testing.assert_close(maps[i][0],expected)
                        break
                for i,row in enumerate(batch):
                    name = row['filename'];image_id = hashlib.sha256(name.encode()).hexdigest()[:20]
                    folder = out/image_id;folder.mkdir(exist_ok=True)
                    clean = folder/'clean.png';images[i].save(clean)
                    if not chosen[i]:empty.append(name)
                    for rank,(column,native_map) in enumerate(zip(chosen[i],maps[i]),1):
                        native = folder/f'{rank}.npy';np.save(native,native_map.cpu().numpy())
                        pairs.append(dict(id=f'VLG_{image_id}_{rank}',image_id=name,filename=name,
                            method='VLG',concept=bank[column],concept_index=column,rank=rank,
                            label=row['label'],predicted_class=int(predicted[i]),input_sha256=input_hashes[i],
                            raw=float(raw[i,column]),normalized=float(z[i,column]),
                            weight=float(weight[predicted[i],column]),contribution=float(contributions[i,column]),
                            clean=str(clean.relative_to(out)),native=str(native.relative_to(out)),
                            clean_sha256=vlm.file_digest(clean),native_sha256=vlm.file_digest(native)))
                    predictions.append(dict(filename=name,label=row['label'],predicted=int(predicted[i]),input_sha256=input_hashes[i]))
                progress = dict(split=split,completed=start+len(batch),total=len(rows),smoke=args.smoke)
                vlm.save(args.output/'progress.json',progress);print(json.dumps(progress),flush=True)
            manifest = out/'manifest.jsonl';manifest.write_text(''.join(json.dumps(r)+'\n' for r in pairs))
            vlm.save(out/'predictions.json',predictions)
            vlm.save(out/'export.json',dict(identity=identity,split=split,images=len(rows),pairs=len(pairs),
                image_ids=[r['filename'] for r in rows],images_without_positive_contributions=empty,
                correct=sum(r['label']==r['predicted'] for r in predictions),manifest_sha256=vlm.file_digest(manifest)))
        done = publish_completion(args.output,identity,splits)
        print(json.dumps(done),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profiles',type=Path,required=True)
    parser.add_argument('--dataset',choices=['imagenet','places365'],required=True)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',choices=['cpu','cuda','mps'],default='cpu');parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--split',choices=['both','validation','test'],default='both')
    parser.add_argument('--batch-size',type=int,default=32)
    args = parser.parse_args()
    if args.batch_size<1:parser.error('batch-size must be positive')
    export(args)


if __name__=='__main__':main()
