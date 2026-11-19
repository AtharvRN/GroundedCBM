"""Export release LF-CBM concepts and post-hoc maps on fixed judge cohorts."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from . import vlm
from .vlg_scenes import cohorts, gradcam, publish_completion, selected_concepts, verified


def verify_unfiltered_bank(path, expected):
    value = Path(path).read_bytes()
    # Feature export appends one newline to a prepared bank lacking one.
    if expected not in {hashlib.sha256(v).hexdigest() for v in (value,value.removesuffix(b'\n'))}:
        raise ValueError('Prepared and exported concept banks differ')


def load_model(profile, device):
    from spatial_cbm.training.config import Model
    from spatial_cbm.training.model import Backbone, ConceptHead
    root, heads = Path(profile['model_root']), Path(profile['classifier_root'])
    for name, key in [('best.pt','checkpoint'), ('final_layer_normalization.pt','normalization'),
                      ('features_complete.json','feature_manifest'), ('concepts.txt','concepts_file'),
                      ('history.json','history')]:
        verified(root/name, profile[key+'_sha256'])
    verified(profile['training_archive'], profile['training_archive_sha256'])
    verified(profile['backbone'], profile['backbone_sha256'])
    verified(heads/'classification_test.json', profile['classification_sha256'])
    payload = torch.load(root/'best.pt', map_location='cpu', weights_only=False)
    protocol = payload['protocol']
    for name, checksum in protocol['source'].items(): verified(Path(profile['source'])/name, checksum)
    if protocol['backbone_sha256'] != profile['backbone_sha256']:
        raise ValueError('Checkpoint/backbone identity differs')
    history = vlm.read(root/'history.json')
    if not history or any(not np.isfinite(r['val_loss']) for r in history):
        raise ValueError('Missing finite validation-loss history')
    best = min(history, key=lambda r:(r['val_loss'],r['epoch']))
    if payload['epoch'] != best['epoch'] or payload['val_loss'] != best['val_loss']:
        raise ValueError('Not the minimum-validation-loss checkpoint')
    cfg = Model(**protocol['config']['model'])
    if cfg.method != 'lf_cbm': raise ValueError('Expected an LF-CBM checkpoint')
    bank = (root/'concepts.txt').read_text().splitlines()
    if len(bank) != profile['concepts'] or len(set(bank)) != len(bank):
        raise ValueError('Invalid concept bank')
    indices = payload['model']['concept_indices']
    if not torch.equal(indices, torch.arange(len(bank))):
        raise ValueError('This export profile requires the fixed, unfiltered release bank')
    verify_unfiltered_bank(root/'concepts.txt', protocol['data']['files']['concepts.txt'])
    classifier = vlm.read(heads/'classifier_identity.json')
    report = vlm.read(heads/'classification_test.json')
    if (classifier['checkpoint_sha256'] != profile['checkpoint_sha256'] or
            classifier['normalization_sha256'] != profile['normalization_sha256'] or
            classifier['feature_manifest_sha256'] != profile['feature_manifest_sha256'] or
            report['identity']['model']['checkpoint_sha256'] != profile['checkpoint_sha256'] or
            report['identity']['classifier'] != classifier):
        raise ValueError('Classifier belongs to another checkpoint')
    if (heads/'concepts.txt').read_text().splitlines() != bank:
        raise ValueError('Classifier concept order differs')
    tensors = []
    for name, key in [('W_g@NEC=5.pt','weight'), ('b_g@NEC=5.pt','bias')]:
        verified(heads/name, profile[key+'_sha256'])
        if classifier['head_files'][name] != profile[key+'_sha256']:
            raise ValueError('Head identity differs')
        tensors.append(torch.load(heads/name, map_location=device, weights_only=True).float())
    weight, bias = tensors
    norm = torch.load(root/'final_layer_normalization.pt', map_location=device, weights_only=True)
    mean, std = norm['mean'].reshape(-1).float(), norm['std'].reshape(-1).float()
    if (weight.shape != (profile['classes'],len(bank)) or bias.shape != (profile['classes'],) or
            mean.shape != (len(bank),) or std.shape != mean.shape or
            int((weight.abs()>1e-5).sum()) != 5*profile['classes']):
        raise ValueError('Invalid NEC=5 classifier dimensions/sparsity')
    if not all(torch.isfinite(v).all() for v in (weight,bias,mean,std)) or not (std>0).all():
        raise ValueError('Invalid classifier values')
    cfg.checkpoint = profile['backbone']
    backbone = Backbone(cfg,profile['classes'],protocol['data']).to(device).eval().requires_grad_(False)
    head = ConceptHead(cfg,backbone.dims,len(bank),indices).to(device).eval().requires_grad_(False)
    head.load_state_dict(payload['model'],strict=True)
    if len(head.hidden) or head.projection.bias is not None:
        raise ValueError('Expected the trained bias-free linear LF projection')
    if protocol['config']['training']['channels_last']:
        backbone.to(memory_format=torch.channels_last)
        head.to(memory_format=torch.channels_last)
    data = protocol['config']['data']
    if (data['resize_size'],data['input_size']) != (256,224):
        raise ValueError('Reference crops require original 256/224 preprocessing')
    return backbone,head,bank,(weight,bias,mean,std),protocol,classifier,payload['epoch']


def reference_inputs(profile, training):
    splits, pins = {}, {}
    for split, folder in profile['reference_exports'].items():
        root = Path(folder)
        meta = vlm.read(root/'export.json')
        verified(root/'manifest.jsonl',meta['manifest_sha256'])
        rows = vlm.read(root/'predictions.json')
        names = [r['filename'] for r in rows]
        expected = 200 if split=='validation' else (1000 if profile['dataset']=='cub' else 5000)
        if names != meta['image_ids'] or len(names) != expected or len(set(names)) != expected:
            raise ValueError('Incomplete/changed reference cohort')
        crops = {}
        for pair in map(json.loads,(root/'manifest.jsonl').read_text().splitlines()):
            name, checksum = pair['filename'], pair['clean_sha256']
            if name in crops and crops[name] != checksum: raise ValueError('Conflicting crops')
            crops[name] = checksum
        for row in rows:
            name = row['filename']
            image_id = hashlib.sha256(name.encode()).hexdigest()[:20]
            clean = root/image_id/'clean.png'
            checksum = vlm.file_digest(clean)
            if name in crops and crops[name] != checksum: raise ValueError('Changed reference crop')
            row.update(clean=str(clean),clean_sha256=checksum)
        splits[split] = rows
        pins[split] = dict(metadata_sha256=vlm.file_digest(root/'export.json'),
            predictions_sha256=vlm.file_digest(root/'predictions.json'),manifest_sha256=meta['manifest_sha256'])
    cal = {r['filename'] for r in splits['validation']}
    test = {r['filename'] for r in splits['test']}
    if cal & test: raise ValueError('Calibration/test overlap')
    if profile['dataset']=='cub':
        verified(profile['training_split'],training['data']['files']['split.json'])
        split = vlm.read(profile['training_split'])
        if not cal <= set(split['validation']) or cal & set(split['train']):
            raise ValueError('Calibration overlaps LF fitting data')
        if test & set(split['train']+split['validation']): raise ValueError('Test overlaps LF development data')
        counts = [sum(r['label']==c for r in splits['test']) for c in range(200)]
        if counts != [5]*200: raise ValueError('Expected five CUB test images per class')
    else:
        meta = vlm.read(Path(profile['reference_exports']['test'])/'export.json')
        official = cohorts(meta['identity']['profile'])
        for split, rows in splits.items():
            if [(r['filename'],r['label']) for r in rows] != [(r['filename'],r['label']) for r in official[split]]:
                raise ValueError('Official scene cohort/labels differ')
        if not training['data']['test_disjoint_verified']:
            raise ValueError('Official validation overlaps LF training')
    return splits,pins


def export(args):
    from torchvision import transforms
    torch.set_num_threads(2)
    profile = vlm.read(args.profiles)[args.dataset]
    backbone,head,bank,tensors,training,classifier,epoch = load_model(profile,args.device)
    weight,bias,mean,std = tensors
    splits,reference_pins = reference_inputs(profile,training)
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'export.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        identity = dict(dataset=args.dataset,method='lf_cbm',nec=5,top_k=5,smoke=args.smoke,
            selected_epoch=epoch,profile=profile,classifier=classifier,device=args.device,
            source_sha256=vlm.file_digest(__file__),helper_sha256=vlm.file_digest(Path(__file__).with_name('vlg_scenes.py')),
            profiles_sha256=vlm.file_digest(args.profiles),references=reference_pins,
            training_source=training['source'],native_maps=False,
            maps='Post-hoc raw-concept Grad-CAM at conv5: ReLU(sum(feature * concept_projection_weight)/(H*W)); no bias or class weight; verified against autograd',
            concept_selection='Up to five positive standardized-activation times predicted-class-weight contributions; no activation-sign or correctness filter',
            preprocessing='Reuse verified 224x224 RGB crops after resize256/center-crop224; apply ImageNet normalization only',
            cohort_caveat='Same prior comparison cohorts; scene test cohorts are disjoint subsets of official validation, not a fresh independent checkpoint-selection set')
        pin = args.output/'export_protocol.json'
        if pin.exists() and vlm.read(pin)!=identity: raise ValueError('Export identity changed')
        vlm.save(pin,identity)
        transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((.485,.456,.406),(.229,.224,.225))])
        for split, full_rows in splits.items():
            rows = full_rows[:2] if args.smoke else full_rows
            out = args.output/split
            out.mkdir(exist_ok=True)
            if (out/'export.json').exists():
                _,meta,_ = vlm.load_export(out)
                if meta['identity']!=identity: raise ValueError('Saved export changed')
                continue
            pairs,predictions,empty = [],[],[]
            for start in range(0,len(rows),args.batch_size):
                batch = rows[start:start+args.batch_size]
                pixels = []
                for row in batch:
                    with Image.open(row['clean']) as image:
                        if image.mode!='RGB' or image.size!=(224,224): raise ValueError('Unexpected crop')
                        pixels.append(transform(image))
                pixels = torch.stack(pixels).to(args.device)
                if training['config']['training']['channels_last']:
                    pixels = pixels.contiguous(memory_format=torch.channels_last)
                with torch.no_grad(): spatial = backbone(pixels)['conv5']
                raw,_ = gradcam(head,spatial,[[] for _ in batch])
                predicted,z,contribution,chosen = selected_concepts(raw,mean,std,weight,bias)
                checked,maps = gradcam(head,spatial,chosen)
                torch.testing.assert_close(raw,checked)
                if start==0:
                    features = spatial.detach().requires_grad_(True)
                    direct = head(dict(conv5=features))['final_logits']
                    torch.testing.assert_close(raw,direct.detach())
                    for i,columns in enumerate(chosen):
                        if not columns: continue
                        gradient = torch.autograd.grad(direct[i,columns[0]],features)[0].mean((-2,-1),keepdim=True)
                        expected = F.relu((features.detach()*gradient).sum(1))[i]
                        torch.testing.assert_close(maps[i][0],expected)
                        break
                for i,row in enumerate(batch):
                    name = row['filename']
                    image_id = hashlib.sha256(name.encode()).hexdigest()[:20]
                    folder = out/image_id;folder.mkdir(exist_ok=True)
                    clean = folder/'clean.png'
                    if not clean.exists(): os.link(row['clean'],clean)
                    verified(clean,row['clean_sha256'])
                    if not chosen[i]: empty.append(name)
                    for rank,(column,native_map) in enumerate(zip(chosen[i],maps[i]),1):
                        native = folder/f'{rank}.npy'
                        np.save(native,native_map.cpu().numpy())
                        pairs.append(dict(id=f'LF_{image_id}_{rank}',image_id=name,filename=name,method='LF',
                            concept=bank[column],concept_index=column,rank=rank,label=row['label'],
                            predicted_class=int(predicted[i]),raw=float(raw[i,column]),normalized=float(z[i,column]),
                            weight=float(weight[predicted[i],column]),contribution=float(contribution[i,column]),
                            clean=str(clean.relative_to(out)),native=str(native.relative_to(out)),
                            clean_sha256=row['clean_sha256'],native_sha256=vlm.file_digest(native)))
                    predictions.append(dict(filename=name,label=row['label'],predicted=int(predicted[i]),
                        clean_sha256=row['clean_sha256']))
                progress = dict(dataset=args.dataset,split=split,completed=start+len(batch),total=len(rows),smoke=args.smoke)
                vlm.save(args.output/'progress.json',progress)
                print(json.dumps(progress),flush=True)
            manifest = out/'manifest.jsonl'
            manifest.write_text(''.join(json.dumps(r)+'\n' for r in pairs))
            vlm.save(out/'predictions.json',predictions)
            vlm.save(out/'export.json',dict(identity=identity,split=split,images=len(rows),pairs=len(pairs),
                image_ids=[r['filename'] for r in rows],images_without_positive_contributions=empty,
                correct=sum(r['label']==r['predicted'] for r in predictions),manifest_sha256=vlm.file_digest(manifest)))
        print(json.dumps(publish_completion(args.output,identity,splits)),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profiles',type=Path,required=True)
    parser.add_argument('--dataset',choices=('cub','imagenet','places365'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',choices=('cpu','mps','cuda'),default='cpu')
    parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--smoke',action='store_true')
    args = parser.parse_args()
    if args.batch_size<1: parser.error('Batch size must be positive')
    export(args)


if __name__=='__main__': main()
