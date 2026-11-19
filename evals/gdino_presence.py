"""Full-vocabulary GroundingDINO concept presence with validation-only F1 calibration."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):value.update(block)
    return value.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def caption_spans(concepts):
    caption='';spans=[]
    for concept in concepts:
        if not concept or concept!=concept.strip():raise ValueError('Invalid concept phrase')
        if caption:caption+=' '
        start=len(caption);caption+=concept
        spans.append([(start,len(caption))]);caption+='.'
    return caption,spans


def prompt_chunks(bank,tokenizer,max_tokens=240,max_concepts=32):
    if len(bank)!=len(set(bank)) or not bank:raise ValueError('Empty or duplicated concept bank')
    if max_tokens<3 or max_concepts<1:raise ValueError('Invalid prompt limits')
    output=[];current=[];start=0
    for concept in bank:
        proposed=current+[concept]
        if len(proposed)>max_concepts or len(tokenizer(caption_spans(proposed)[0])['input_ids'])>max_tokens:
            if not current:raise ValueError('A concept exceeds the text window')
            caption,spans=caption_spans(current)
            output.append(dict(start=start,end=start+len(current),caption=caption,spans=spans))
            start+=len(current);current=[]
        current.append(concept)
        if len(tokenizer(caption_spans(current)[0])['input_ids'])>max_tokens:
            raise ValueError('A concept exceeds the text window')
    if current:
        caption,spans=caption_spans(current)
        output.append(dict(start=start,end=len(bank),caption=caption,spans=spans))
    return output


def phrase_scores(logits,positive_map):
    if logits.ndim!=3 or positive_map.ndim!=2 or logits.shape[2]!=positive_map.shape[1]:
        raise ValueError('Mismatched query/token dimensions')
    if not torch.isfinite(positive_map).all() or (positive_map<0).any() or (positive_map.sum(1)<=0).any():
        raise ValueError('Empty or invalid phrase token map')
    scores=torch.einsum('ct,bqt->bcq',positive_map.float(),logits.float().sigmoid()).amax(-1)
    if not torch.isfinite(scores).all() or (scores<0).any() or (scores>1.000001).any():
        raise ValueError('Invalid untruncated concept confidence')
    return scores


def fit_thresholds(labels,scores):
    from sklearn.metrics import precision_recall_curve
    labels=np.asarray(labels,dtype=bool);scores=np.asarray(scores)
    if labels.shape!=scores.shape or labels.ndim!=2 or not np.isfinite(scores).all():
        raise ValueError('Invalid validation arrays')
    thresholds=[];statuses=[]
    for y,x in zip(labels.T,scores.T):
        if not y.any():threshold,status=np.inf,'no_validation_positives_predict_absent'
        elif y.all():threshold,status=-np.inf,'no_validation_negatives_predict_present'
        else:
            p,r,t=precision_recall_curve(y,x)
            f=2*p[:-1]*r[:-1]/np.maximum(p[:-1]+r[:-1],1e-15)
            threshold=t[np.flatnonzero(f==f.max())[-1]];status='validation_f1_maximum'
        thresholds.append(threshold);statuses.append(status)
    return np.asarray(thresholds,dtype=np.float64),statuses


def presence_metrics(labels,scores,thresholds):
    from sklearn.metrics import average_precision_score,roc_auc_score
    labels=np.asarray(labels,dtype=bool);scores=np.asarray(scores)
    if scores.shape!=labels.shape or scores.ndim!=2 or not np.isfinite(scores).all():
        raise ValueError('Invalid evaluation arrays')
    if np.shape(thresholds)!=(labels.shape[1],) or np.isnan(thresholds).any():raise ValueError('Invalid thresholds')
    eligible=labels.any(0)
    if not eligible.any() or labels.all() or not labels.any():raise ValueError('Ranking requires positives and negatives')
    predicted=scores>=thresholds[None,:]
    tp=(predicted&labels).sum(0);fp=(predicted&~labels).sum(0);fn=(~predicted&labels).sum(0)
    aps=[float(average_precision_score(labels[:,j],scores[:,j])) for j in np.flatnonzero(eligible)]
    return dict(images=len(scores),concepts=scores.shape[1],positive_concepts=int(eligible.sum()),positive_pairs=int(labels.sum()),
        pooled_auroc=float(roc_auc_score(labels.ravel(),scores.ravel())),macro_ap=float(np.mean(aps)),
        macro_f1=float((2*tp/np.maximum(2*tp+fp+fn,1))[eligible].mean()),
        macro_precision=float((tp/np.maximum(tp+fp,1))[eligible].mean()),
        macro_recall=float((tp/np.maximum(tp+fn,1))[eligible].mean()),
        per_concept_ap=dict(zip(map(str,np.flatnonzero(eligible)),aps)),tp=tp.tolist(),fp=fp.tolist(),fn=fn.tolist())


def validate_roster(roster,provenance):
    if len(roster)!=provenance['calibration_images']+provenance['test_images']:raise ValueError('Incomplete roster')
    if len({r['filename'] for r in roster})!=len(roster):raise ValueError('Duplicated or overlapping image cohort')
    for split,offset in (('calibration',0),('test',90000)):
        rows=[r for r in roster if r['split']==split]
        if [r['index'] for r in rows]!=list(range(provenance[split+'_images'])):raise ValueError('Changed split order')
        if len({r['source_index'] for r in rows})!=len(rows):raise ValueError('Duplicated source image index')
        for row in rows:
            path=Path(row['filename'])
            if path.is_absolute() or '..' in path.parts or len(path.parts)!=2:raise ValueError('Unsafe image filename')
            if not offset<=row['source_index']<offset+(90000 if split=='calibration' else 10000):
                raise ValueError('Image belongs to another split')
            if split=='test' and row['source_index']!=offset+row['index']:raise ValueError('Test target order changed')


def load_inputs(root):
    hashes=read(root/'staged_hashes.json')
    for name,expected in hashes.items():
        path=(root/name).resolve()
        if not path.is_relative_to(root.resolve()) or sha(path)!=expected:raise ValueError('Changed staged input: '+name)
    provenance=read(root/'inputs/provenance.json');roster=read(root/'inputs/roster.json')
    validate_roster(roster,provenance)
    expected=read(root/'inputs/split_reference.json')['validation_indices']
    if [r['source_index'] for r in roster if r['split']=='calibration']!=expected:
        raise ValueError('Calibration target order changed')
    bank=(root/'inputs/concepts.txt').read_text().splitlines()
    if len(bank)!=provenance['concepts'] or sha(root/'inputs/concepts.txt')!=provenance['bank_sha256']:
        raise ValueError('Changed concept bank')
    return provenance,roster,bank


def load_detector(root,checkpoint,device):
    sys.path.insert(0,str(root/'detector'))
    from groundingdino.models import build_model
    from groundingdino.util.slconfig import SLConfig
    from groundingdino.util.utils import clean_state_dict
    cfg=SLConfig.fromfile(str(root/'detector/groundingdino/config/GroundingDINO_SwinB_cfg.py'))
    cfg.device=device;model=build_model(cfg)
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)['model']
    missing=model.load_state_dict(clean_state_dict(state),strict=False)
    if missing.missing_keys:raise ValueError('Detector checkpoint has missing parameters: '+str(missing.missing_keys))
    return model.to(device).eval().requires_grad_(False),dict(unexpected_keys=missing.unexpected_keys)


def image_tensor(root,row):
    from torchvision import transforms
    from groundingdino.datasets import transforms as gdino_transforms
    path=root/row['filename']
    with Image.open(path) as image:
        crop=transforms.CenterCrop(224)(transforms.Resize(256)(image.convert('RGB')))
        transform=gdino_transforms.Compose([gdino_transforms.RandomResize([800],max_size=1333),
            gdino_transforms.ToTensor(),gdino_transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
        tensor,_=transform(crop,None)
    return tensor,sha(path),hashlib.sha256(np.asarray(crop).tobytes()).hexdigest()


def infer(args):
    from groundingdino.util.misc import nested_tensor_from_tensor_list
    from groundingdino.util.vl_utils import create_positive_map_from_span
    provenance,roster,bank=load_inputs(args.root)
    checkpoint=args.root/'groundingdino_swinb_cogcoor.pth'
    if sha(checkpoint)!=provenance['detector_weight_sha256']:raise ValueError('Detector weight identity differs')
    for probe in read(args.root/'inputs/image_probes.json'):
        if sha(args.images/probe['filename'])!=probe['sha256']:raise ValueError('Image source differs from CBM input')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    model,load_report=load_detector(args.root,checkpoint,args.device)
    chunks=prompt_chunks(bank,model.tokenizer,min(240,model.max_text_len),args.max_concepts)
    for chunk in chunks:
        tokenized=model.tokenizer(chunk['caption'])
        positive=create_positive_map_from_span(tokenized,token_span=chunk['spans'])
        if (positive.sum(1)<=0).any() or len(tokenized['input_ids'])>model.max_text_len:
            raise ValueError('Concept tokens were truncated or unmapped')
        chunk['positive_map']=positive.to(args.device)
    identity=dict(inputs_sha256=sha(args.root/'staged_hashes.json'),checkpoint_sha256=sha(checkpoint),
        evaluator_sha256=sha(Path(__file__)),images=str(args.images.resolve()),batch_size=args.batch_size,
        shards=args.shards,max_concepts=args.max_concepts,limit=args.limit,device=args.device,precision='FP32',
        prompts=[{k:v for k,v in c.items() if k!='positive_map'} for c in chunks],load=load_report)
    rows=list(enumerate(roster))
    if args.limit:rows=rows[:args.limit]
    rows=rows[args.shard::args.shards]
    if not rows:raise ValueError('Empty inference shard')
    output=args.output/f'shard{args.shard:03d}';output.mkdir(parents=True,exist_ok=True)
    with (output/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        protocol=output/'protocol.json'
        if protocol.exists() and read(protocol)!=identity:raise ValueError('Inference protocol changed')
        save(protocol,identity)
        if (output/'complete.json').exists():return
        cache=output/'batches';cache.mkdir(exist_ok=True)
        verified=False;started=time.monotonic();completed=0
        for start in range(0,len(rows),args.batch_size):
            batch=rows[start:start+args.batch_size];indices=np.asarray([i for i,_ in batch])
            path=cache/f'{start:06d}.npz'
            if path.exists():
                with np.load(path,allow_pickle=False) as saved:
                    if not np.array_equal(saved['indices'],indices) or saved['scores'].shape!=(len(batch),len(bank)):
                        raise ValueError('Changed inference batch')
                    if not np.isfinite(saved['scores']).all():raise ValueError('Nonfinite saved confidence')
                completed+=len(batch);continue
            loaded=[image_tensor(args.images,row) for _,row in batch]
            samples=nested_tensor_from_tensor_list([item[0].to(args.device) for item in loaded])
            with torch.inference_mode():
                model.set_image_tensor(samples);features=model.features;positions=tuple(model.poss)
                columns=[]
                for chunk in chunks:
                    model.set_image_features(features,list(positions))
                    predictions=model(samples,captions=[chunk['caption']]*len(batch),unset_image_tensor=False)
                    columns.append(phrase_scores(predictions['pred_logits'],chunk['positive_map']))
                scores=torch.cat(columns,1)
                model.unset_image_tensor()
                if not verified:
                    for index in sorted(set([0,len(chunks)-1])):
                        chunk=chunks[index]
                        plain=model(samples,captions=[chunk['caption']]*len(batch))
                        expected=phrase_scores(plain['pred_logits'],chunk['positive_map'])
                        torch.testing.assert_close(columns[index],expected,rtol=1e-5,atol=2e-6)
                    verified=True
                    save(output/'cache_equivalence.json',dict(passed=True,chunks=[0,len(chunks)-1],
                        note='Only image-backbone features cached; text-conditioned encoder rerun; positional lists reset each chunk'))
                values=scores.cpu().numpy()
            temporary=path.with_suffix('.tmp')
            with temporary.open('wb') as stream:
                np.savez(stream,indices=indices,scores=values,image_sha256=np.asarray([x[1] for x in loaded]),
                         crop_sha256=np.asarray([x[2] for x in loaded]))
            temporary.replace(path);completed+=len(batch)
            elapsed=time.monotonic()-started
            progress=dict(completed=completed,total=len(rows),seconds=elapsed,images_per_second=completed/max(elapsed,1e-9),
                prompt_chunks=len(chunks),gpu_memory_bytes=torch.cuda.max_memory_allocated() if args.device=='cuda' else None)
            save(output/'progress.json',progress)
            if completed%10==0 or completed==len(rows):print(json.dumps(progress),flush=True)
        save(output/'complete.json',dict(complete=True,images=len(rows),protocol_sha256=sha(protocol),
            batch_hashes={p.name:sha(p) for p in sorted(cache.glob('*.npz'))}))


def evaluate(args):
    provenance,roster,bank=load_inputs(args.root)
    scores=np.full((len(roster),len(bank)),np.nan,dtype=np.float32);seen=set();identity=None
    folders=sorted(args.output.glob('shard[0-9][0-9][0-9]'))
    if not folders:raise ValueError('No inference shards')
    for folder in folders:
        done=read(folder/'complete.json');protocol=read(folder/'protocol.json')
        if (not done['complete'] or done['protocol_sha256']!=sha(folder/'protocol.json') or protocol['limit']):
            raise ValueError('Incomplete, changed, or smoke inference')
        if identity is None:identity=protocol
        if identity!=protocol:raise ValueError('Shard protocols differ')
        if protocol['inputs_sha256']!=sha(args.root/'staged_hashes.json') or protocol['evaluator_sha256']!=sha(Path(__file__)):
            raise ValueError('Inference used different inputs or evaluator')
        for name,expected in done['batch_hashes'].items():
            path=folder/'batches'/name
            if sha(path)!=expected:raise ValueError('Changed confidence batch')
            with np.load(path,allow_pickle=False) as batch:
                indices=batch['indices'];values=batch['scores']
                if values.shape!=(len(indices),len(bank)) or set(map(int,indices))&seen:
                    raise ValueError('Duplicate or invalid score batch')
                if (indices<0).any() or (indices>=len(roster)).any():raise ValueError('Invalid image indices')
                seen.update(map(int,indices));scores[indices]=values
    if len(folders)!=identity['shards'] or seen!=set(range(len(roster))) or not np.isfinite(scores).all():
        raise ValueError('Not every image/concept was evaluated')
    calibration=np.asarray([i for i,r in enumerate(roster) if r['split']=='calibration'])
    test=np.asarray([i for i,r in enumerate(roster) if r['split']=='test'])
    labels=np.load(args.root/'inputs/calibration_labels.npy',allow_pickle=False)
    thresholds,statuses=fit_thresholds(labels,scores[calibration])
    result=args.output/'evaluation';result.mkdir(exist_ok=True)
    np.save(result/'thresholds.npy',thresholds,allow_pickle=False)
    save(result/'calibration.json',dict(images=len(calibration),threshold_sha256=sha(result/'thresholds.npy'),
        statuses=statuses,objective=provenance['calibration'],calibration_labels_sha256=sha(args.root/'inputs/calibration_labels.npy')))
    # Test labels are read only after the validation thresholds have been frozen.
    truth=np.load(args.root/'inputs/test_labels.npy',allow_pickle=False)
    if int(truth.sum())!=provenance['test_positive_pairs'] or int(truth.any(0).sum())!=provenance['test_positive_concepts']:
        raise ValueError('Human test target contract differs')
    metrics=presence_metrics(truth,scores[test],thresholds)
    np.save(result/'calibration_scores.npy',scores[calibration],allow_pickle=False)
    np.save(result/'test_scores.npy',scores[test],allow_pickle=False)
    save(result/'results.json',dict(metrics=metrics,provenance=provenance,identity=identity,
        runs=1,calibration_sha256=sha(result/'calibration.json')))
    save(result/'complete.json',dict(complete=True,results_sha256=sha(result/'results.json')))
    print(json.dumps({k:v for k,v in metrics.items() if k not in ('per_concept_ap','tp','fp','fn')}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    for command in ('infer','evaluate'):
        p=sub.add_parser(command);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
        if command=='infer':
            p.add_argument('--images',type=Path,required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
            p.add_argument('--batch-size',type=int,default=2);p.add_argument('--max-concepts',type=int,default=32)
            p.add_argument('--shards',type=int,default=1);p.add_argument('--shard',type=int,default=0);p.add_argument('--limit',type=int,default=0)
    args=parser.parse_args()
    if args.command=='infer':
        if args.batch_size<1 or args.shards<1 or not 0<=args.shard<args.shards or args.limit<0:parser.error('Invalid inference dimensions')
        sys.path.insert(0,str(args.root/'detector'));infer(args)
    else:evaluate(args)


if __name__=='__main__':main()
