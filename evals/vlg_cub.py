"""Export concept-targeted VLG Grad-CAM on existing, pinned CUB judge cohorts."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import tarfile

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from .vlm import file_digest, load_export, read, save


def concept_gradcam(spatial, weight, column):
    """Exact Grad-CAM for a raw concept logit after global average pooling."""
    if spatial.ndim != 4 or weight.ndim != 2 or spatial.shape[1] != weight.shape[1]:
        raise ValueError('Expected compatible feature and linear-concept tensors')
    if not torch.isfinite(spatial).all() or not torch.isfinite(weight).all():
        raise ValueError('Nonfinite Grad-CAM input')
    if column < 0 or column >= len(weight): raise ValueError('Invalid concept index')
    return F.relu((spatial*weight[column][None,:,None,None]).sum(1)/spatial.shape[-2:].numel())


def reference_cohort(root, split, count):
    root = Path(root); receipt = read(root/'complete.json')
    if not receipt['complete']: raise ValueError('Incomplete reference export')
    expected = receipt['splits'][split]; archive = root/(split+'.tar.gz')
    if expected['images'] != count or file_digest(archive) != expected['archive_sha256']:
        raise ValueError('Reference export identity changed')
    with tarfile.open(archive) as stream:
        cohort_bytes = stream.extractfile(split+'/cohort.json').read()
        manifest = stream.extractfile(split+'/manifest.jsonl').read()
    if (hashlib.sha256(cohort_bytes).hexdigest() != expected['cohort_sha256'] or
            hashlib.sha256(manifest).hexdigest() != expected['manifest_sha256']):
        raise ValueError('Reference cohort or manifest changed')
    cohort = json.loads(cohort_bytes)
    if len(cohort['images']) != count or len(set(cohort['images'])) != count:
        raise ValueError('Duplicate or incomplete reference image roster')
    crops = {}
    for row in map(json.loads,manifest.decode().splitlines()):
        name = row['filename']
        if name in crops and crops[name] != row['clean_sha256']:
            raise ValueError('Conflicting reference crop identities')
        crops[name] = row['clean_sha256']
    # Images without selected reference concepts still belong to the fixed cohort.
    missing = set(cohort['images'])-set(crops)
    if missing:
        with tarfile.open(archive) as stream:
            for name in sorted(missing):
                image_id = hashlib.sha256(name.encode()).hexdigest()[:20]
                pixels = stream.extractfile(f'{split}/{image_id}/clean.png').read()
                crops[name] = hashlib.sha256(pixels).hexdigest()
    return cohort, crops, dict(receipt_sha256=file_digest(root/'complete.json'), **expected)


def validate_cohorts(calibration, test, split, labels):
    cal, evaluation = set(calibration['images']), set(test['images'])
    if len(cal) != 200 or len(evaluation) != 1000 or cal & evaluation:
        raise ValueError('Expected disjoint 200/1000 cohorts')
    if not cal <= set(split['validation']) or cal & set(split['train']):
        raise ValueError('Calibration must exclude VLG fitting images')
    if evaluation & set(split['train']+split['validation']):
        raise ValueError('Test overlaps VLG development data')
    counts = {}
    for name in evaluation: counts[labels[name]] = counts.get(labels[name],0)+1
    if len(counts) != 200 or set(counts.values()) != {5}:
        raise ValueError('Expected five test images per class')
    for cohort in (calibration,test):
        if cohort['labels'] != [labels[name] for name in cohort['images']]:
            raise ValueError('Reference class labels changed')


def export(args):
    from torchvision import transforms
    from spatial_cbm.training.config import Model
    from spatial_cbm.training.model import Backbone, ConceptHead
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'export.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        checkpoint = args.model_root/'best.pt'
        if file_digest(checkpoint) != args.checkpoint_sha256:
            raise ValueError('Unexpected VLG checkpoint')
        payload = torch.load(checkpoint,map_location='cpu',weights_only=False)
        protocol = payload['protocol']
        for name, expected in protocol['source'].items():
            if file_digest(args.source/name) != expected: raise ValueError('Training source changed: '+name)
        if file_digest(args.backbone) != protocol['backbone_sha256']:
            raise ValueError('Unexpected VLG backbone')
        cfg = Model(**protocol['config']['model'])
        if cfg.method != 'vlg_cbm' or cfg.vlg_hidden != 0:
            raise ValueError('This Grad-CAM adapter requires a linear VLG concept head')
        prepared = Path(protocol['config']['data']['prepared'])
        if file_digest(prepared/'split.json') != protocol['data']['files']['split.json']:
            raise ValueError('Training split changed')
        bank = (args.model_root/'concepts.txt').read_text().splitlines()
        if len(bank) != len(set(bank)) or not bank: raise ValueError('Invalid concept bank')
        cfg.checkpoint = str(args.backbone)
        backbone = Backbone(cfg,200,protocol['data']).eval().requires_grad_(False)
        full_bank = (prepared/'concepts.txt').read_text().splitlines()
        indices = payload['model']['concept_indices']
        if [full_bank[i] for i in indices.tolist()] != bank:
            raise ValueError('Retained concept names/order differ from training indices')
        head = ConceptHead(cfg,backbone.dims,len(full_bank),indices).eval().requires_grad_(False)
        head.load_state_dict(payload['model'],strict=True)
        classifier = read(args.model_root/'nec/classifier_identity.json')
        norm_path = args.model_root/'final_layer_normalization.pt'
        if (classifier['checkpoint_sha256'] != args.checkpoint_sha256 or
                classifier['normalization_sha256'] != file_digest(norm_path)):
            raise ValueError('Classifier belongs to another model')
        tensors = []
        for name in ('W_g@NEC=5.pt','b_g@NEC=5.pt'):
            path = args.model_root/'nec'/name
            if file_digest(path) != classifier['head_files'][name]: raise ValueError('Classifier changed')
            tensors.append(torch.load(path,map_location='cpu',weights_only=True).float())
        weight,bias = tensors
        norm = torch.load(norm_path,map_location='cpu',weights_only=True)
        mean,std = norm['mean'].reshape(-1).float(),norm['std'].reshape(-1).float()
        if weight.shape != (200,len(bank)) or int((weight.abs()>1e-5).sum()) != 1000:
            raise ValueError('Expected exact NEC=5 classifier')
        if not all(torch.isfinite(v).all() for v in (weight,bias,mean,std)) or not (std>0).all():
            raise ValueError('Invalid classifier normalization')
        calibration,cal_crops,cal_pin = reference_cohort(args.calibration_reference,'calibration',200)
        test,test_crops,test_pin = reference_cohort(args.test_reference,'test',1000)
        names = dict(line.split(maxsplit=1) for line in (args.images.parent/'images.txt').read_text().splitlines())
        labels = {names[i]:int(y)-1 for i,y in map(str.split,(args.images.parent/'image_class_labels.txt').read_text().splitlines())}
        official_test = {names[i] for i,s in map(str.split,(args.images.parent/'train_test_split.txt').read_text().splitlines()) if s=='0'}
        validate_cohorts(calibration,test,read(prepared/'split.json'),labels)
        if not set(test['images']) <= official_test: raise ValueError('Not official test images')
        reference = np.load(args.reference_scores,allow_pickle=False)
        score_lookup = {str(name):i for i,name in enumerate(reference['roster'])}
        if list(reference['concepts']) != bank: raise ValueError('Reference concept order differs')
        identity = dict(dataset='cub',method='vlg_cbm',nec=5,top_k=5,
            checkpoint_sha256=args.checkpoint_sha256,backbone_sha256=file_digest(args.backbone),
            classifier=classifier,concepts_sha256=file_digest(args.model_root/'concepts.txt'),
            source_sha256=file_digest(__file__),reference_scores_sha256=file_digest(args.reference_scores),
            training_source=protocol['source'],training_split_sha256=file_digest(prepared/'split.json'),
            references=dict(calibration=cal_pin,test=test_pin),
            concept_selection='Up to five strictly positive standardized activation * predicted-class weight; no activation-sign or correctness filter',
            maps='Concept-targeted Grad-CAM at conv5: ReLU(sum(channel_gradient_mean * feature)); raw concept logit target, no bias or class weight; analytic result verified against autograd',
            preprocessing=dict(resize=256,crop=224),native_maps=False)
        pin = args.output/'export_protocol.json'
        if pin.exists() and read(pin) != identity: raise ValueError('Export protocol changed')
        save(pin,identity)
        transform = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),
            transforms.Normalize((.485,.456,.406),(.229,.224,.225))])
        for split,cohort,crops in (('validation',calibration,cal_crops),('test',test,test_crops)):
            out = args.output/split;out.mkdir(exist_ok=True)
            if (out/'export.json').exists():
                _,metadata,_ = load_export(out)
                if metadata['identity'] != identity: raise ValueError('Saved export identity differs')
                continue
            rows,predictions,empty = [],[],[]
            for start in range(0,len(cohort['images']),16):
                batch_names = cohort['images'][start:start+16]
                images = []
                for name in batch_names:
                    with Image.open(args.images/name) as image: images.append(transform(image.convert('RGB')))
                pixels = torch.stack(images)
                with torch.no_grad():
                    spatial = backbone(pixels)['conv5']
                    raw = head(dict(conv5=spatial))['final_logits'].float()
                    z = (raw-mean)/std;logits = F.linear(z,weight,bias)
                if split=='test':
                    expected = reference['scores'][[score_lookup[name] for name in batch_names]]
                    np.testing.assert_allclose(raw.numpy(),expected,rtol=2e-4,atol=1e-4)
                for i,name in enumerate(batch_names):
                    predicted = int(logits[i].argmax());contribution = z[i]*weight[predicted]
                    order = torch.argsort(contribution,descending=True,stable=True)
                    chosen = order[contribution[order]>0][:5].tolist()
                    if not chosen: empty.append(name)
                    if i==0 and start==0 and chosen:
                        feature = spatial[i:i+1].detach().requires_grad_(True)
                        score = head(dict(conv5=feature))['final_logits'][0,chosen[0]]
                        gradient = torch.autograd.grad(score,feature)[0].mean((-2,-1),keepdim=True)
                        actual = F.relu((feature*gradient).sum(1)).detach()
                        torch.testing.assert_close(concept_gradcam(spatial[i:i+1],head.projection.weight,chosen[0]),actual)
                    image_id = hashlib.sha256(name.encode()).hexdigest()[:20]
                    folder = out/image_id;folder.mkdir(exist_ok=True)
                    rgb = ((pixels[i]*torch.tensor([.229,.224,.225])[:,None,None]+
                        torch.tensor([.485,.456,.406])[:,None,None]).clamp(0,1)*255).round().byte().permute(1,2,0).numpy()
                    clean = folder/'clean.png';Image.fromarray(rgb).save(clean)
                    if file_digest(clean) != crops[name]: raise ValueError('Model crop differs from prior CUB study')
                    for rank,column in enumerate(chosen,1):
                        native = folder/f'{rank}.npy'
                        np.save(native,concept_gradcam(spatial[i:i+1],head.projection.weight,column)[0].numpy())
                        rows.append(dict(id=f'VLG_{image_id}_{rank}',image_id=name,filename=name,
                            concept=bank[column],concept_index=column,rank=rank,method='VLG',
                            label=labels[name],predicted_class=predicted,raw=float(raw[i,column]),
                            normalized=float(z[i,column]),weight=float(weight[predicted,column]),contribution=float(contribution[column]),
                            clean=str(clean.relative_to(out)),native=str(native.relative_to(out)),
                            clean_sha256=file_digest(clean),native_sha256=file_digest(native)))
                    predictions.append(dict(filename=name,label=labels[name],predicted=predicted))
                progress = dict(stage=split,completed=start+len(batch_names),total=len(cohort['images']))
                save(args.output/'progress.json',progress);print(json.dumps(progress),flush=True)
            manifest = out/'manifest.jsonl';manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            save(out/'predictions.json',predictions)
            save(out/'export.json',dict(identity=identity,split=split,images=len(cohort['images']),pairs=len(rows),
                image_ids=cohort['images'],images_without_positive_contributions=empty,
                manifest_sha256=file_digest(manifest),correct=sum(r['label']==r['predicted'] for r in predictions),
                cohort_selection='Reuse pinned SpG/SALF cohorts before inference; 200 inner-validation and five test images per class'))
        save(args.output/'complete.json',dict(complete=True,protocol_sha256=file_digest(pin),
            exports={s:file_digest(args.output/s/'export.json') for s in ('validation','test')}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model-root','backbone','source','images','calibration-reference','test-reference','reference-scores','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--checkpoint-sha256',required=True)
    export(parser.parse_args())


if __name__=='__main__': main()
