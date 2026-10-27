"""Rank AP monitoring and validation-only threshold calibration."""
from pathlib import Path
import tempfile

import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import average_precision_score


def precision_recall_f1(tp, predicted, positives):
    precision = tp / predicted if predicted else 0.
    recall = tp / positives if positives else 0.
    f1 = 2*tp/(predicted+positives) if predicted+positives else 0.
    return dict(precision=float(precision),recall=float(recall),f1=float(f1),
                true_positives=int(tp),predicted_positives=int(predicted),positives=int(positives))


def row_ap(scores, labels):
    """Exact non-interpolated AP, including tied scores, for each spatial map."""
    values, order = scores.sort(dim=1,descending=True,stable=True)
    truth = labels.gather(1,order).double()
    cumulative = truth.cumsum(1)
    n = scores.shape[1]
    ranks = torch.arange(n,device=scores.device).expand_as(scores)
    ends = torch.cat((values[:,:-1] != values[:,1:],torch.ones_like(values[:,-1:],dtype=torch.bool)),1)
    end_index = torch.where(ends,ranks,n).flip(1).cummin(1).values.flip(1)
    precision = cumulative/(ranks+1)
    return (truth*precision.gather(1,end_index)).sum(1)/truth.sum(1).clamp_min(1)


class Metrics:
    def __init__(self, rows, concepts, cache, calibrate=False, thresholds=None):
        if calibrate and thresholds is not None:
            raise ValueError('Cannot calibrate on an evaluation cohort')
        Path(cache).mkdir(parents=True,exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix='validation-',dir=cache)
        self.scores = np.lib.format.open_memmap(Path(self.temporary.name)/'scores.npy',
            mode='w+',dtype='float32',shape=(rows,concepts))
        self.labels = np.lib.format.open_memmap(Path(self.temporary.name)/'labels.npy',
            mode='w+',dtype='uint8',shape=(rows,concepts))
        self.offset, self.spatial_count, self.spatial_ap = 0, 0, 0.
        self.calibrate, self.thresholds = calibrate, thresholds
        self.concept_grid = (np.r_[np.linspace(0,1,1001),1.000001] if calibrate else
                             np.array([thresholds['concept']]) if thresholds is not None else None)
        self.spatial_grid = (np.linspace(0,1,51) if calibrate else
                             np.array([thresholds['spatial']]) if thresholds is not None else None)
        self.concept_pred = np.zeros(len(self.concept_grid),dtype=np.int64) if self.concept_grid is not None else None
        self.concept_tp = np.zeros_like(self.concept_pred) if self.concept_pred is not None else None
        self.positives = 0
        self.iou = np.zeros(len(self.spatial_grid)) if self.spatial_grid is not None else None

    def update(self, output, batch):
        logits, labels = output['final_logits'].float(), batch[1].bool()
        end = self.offset+len(logits)
        self.scores[self.offset:end] = logits.detach().cpu().numpy()
        self.labels[self.offset:end] = labels.cpu().numpy()
        self.offset = end
        if self.concept_grid is not None:
            grid = torch.tensor(self.concept_grid,device=logits.device,dtype=torch.float64)
            bins = torch.bucketize(logits.sigmoid().double().flatten(),grid,right=True)
            total = torch.bincount(bins,minlength=len(grid)+1)
            positive = torch.bincount(bins[labels.flatten()],minlength=len(grid)+1)
            self.concept_pred += total.flip(0).cumsum(0).flip(0)[1:].cpu().numpy()
            self.concept_tp += positive.flip(0).cumsum(0).flip(0)[1:].cpu().numpy()
            self.positives += int(labels.sum())
        pairs = batch[4].nonzero()
        # Bound the temporary pixel-sort workspace independently of batch size.
        for chunk in pairs.split(16):
            if not len(chunk): continue
            image, slot = chunk.unbind(1)
            concept = batch[2][image,slot]
            maps = output['spatial_maps'][image,concept,None].float()
            maps = F.interpolate(maps,size=batch[5].shape[-2:],mode='bilinear',align_corners=False).flatten(1)
            truth = batch[5][image,slot].flatten(1) > 0
            if not torch.isfinite(maps).all(): raise ValueError('Nonfinite validation maps')
            self.spatial_ap += float(row_ap(maps,truth).sum())
            self.spatial_count += len(maps)
            if self.spatial_grid is not None:
                low, high = maps.amin(1,keepdim=True), maps.amax(1,keepdim=True)
                normalized = (maps-low)/(high-low).clamp_min(1e-12)
                grid = torch.tensor(self.spatial_grid,device=maps.device,dtype=torch.float64)
                bins = torch.bucketize(normalized.double(),grid,right=True)
                counts = torch.zeros((len(maps),len(grid)+1),device=maps.device,dtype=torch.int64)
                positive = torch.zeros_like(counts)
                counts.scatter_add_(1,bins,torch.ones_like(bins))
                positive.scatter_add_(1,bins,truth.long())
                predicted = counts.flip(1).cumsum(1).flip(1)[:,1:]
                tp = positive.flip(1).cumsum(1).flip(1)[:,1:]
                union = predicted+truth.sum(1,keepdim=True)-tp
                self.iou += (tp.double()/union.clamp_min(1)).sum(0).cpu().numpy()

    def result(self):
        if self.offset != len(self.scores): raise ValueError('Incomplete metric cohort')
        aps = []
        for j in range(self.scores.shape[1]):
            truth = np.asarray(self.labels[:,j])
            scores = np.asarray(self.scores[:,j])
            if not np.isfinite(scores).all(): raise ValueError('Nonfinite validation concepts')
            if truth.any(): aps.append(float(average_precision_score(truth,scores)))
        result = dict(concept_ap=float(np.mean(aps)) if aps else None,
            concept_ap_supported=len(aps),concept_ap_total=self.scores.shape[1],
            spatial_ap=self.spatial_ap/self.spatial_count if self.spatial_count else None,
            spatial_pairs=self.spatial_count,
            concept_ap_definition='macro AP over concepts with positive labels; raw-logit ranking',
            spatial_ap_definition='mean pixel AP over annotated image-concept pairs; input-resolution region masks')
        if self.concept_grid is not None:
            f1 = 2*self.concept_tp/np.maximum(self.concept_pred+self.positives,1)
            # Conservative deterministic tie break; no positives selects predict-none.
            i = int(np.flatnonzero(f1==f1.max())[-1]) if self.calibrate else 0
            result['concept'] = dict(precision_recall_f1(self.concept_tp[i],self.concept_pred[i],self.positives),
                threshold=float(self.concept_grid[i]),score='sigmoid(fused_logit)',comparison='>=',averaging='micro')
            mean_iou = self.iou/max(self.spatial_count,1)
            k = int(np.flatnonzero(mean_iou==mean_iou.max())[-1]) if self.calibrate else 0
            result['localization'] = dict(foreground_iou=float(mean_iou[k]) if self.spatial_count else None,
                threshold=float(self.spatial_grid[k]),score='per-map min-max normalized upsampled logit',
                comparison='>=',averaging='mean over detected image-concept pairs')
            if self.calibrate:
                result['search'] = dict(concept_thresholds=self.concept_grid.tolist(),
                    concept_objective='maximum validation micro F1',spatial_thresholds=self.spatial_grid.tolist(),
                    spatial_objective='maximum validation mean foreground IoU',tie_break='highest threshold')
        return result

    def close(self):
        del self.scores, self.labels
        self.temporary.cleanup()


def assess(cfg, test=False):
    """Calibrate selected best.pt on validation, or apply frozen thresholds to test."""
    import fcntl
    import json
    from torch.utils.data import DataLoader
    from .data import Samples, collate, digest, save_json, verify
    from .engine import provenance, run_epoch, minimum_loss
    from .model import Backbone, ConceptHead

    root = Path(cfg.training.output)
    with (root/'assessment.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        if cfg.training.stage != 'concept' or not (root/'complete.json').exists():
            raise ValueError('Finish concept training before calibration/evaluation')
        data = verify(cfg.data.prepared)
        if not data.get('test_disjoint_verified'):
            raise ValueError('Prepare and audit the held-out test manifest first')
        selected = torch.load(root/'best.pt',map_location='cpu',weights_only=False)
        current = provenance(cfg,data)
        for key in ('config','data','source','backbone_sha256'):
            if selected['protocol'][key] != current[key]:
                raise ValueError(f'Assessment provenance mismatch: {key}')
        best = minimum_loss(json.loads((root/'history.json').read_text()))
        if selected['epoch'] != best['epoch'] or selected['val_loss'] != best['val_loss']:
            raise ValueError('Calibration requires the minimum-validation-loss checkpoint')
        identity = dict(checkpoint_sha256=digest(root/'best.pt'),split_sha256=data['files']['split.json'],
                        prepared_files=data['files'],source=current['source'])
        thresholds = None
        if test:
            calibrated = json.loads((root/'calibration.json').read_text())
            if calibrated['identity'] != identity or calibrated['cohort'] != 'validation':
                raise ValueError('Thresholds do not belong to this selected checkpoint and validation split')
            thresholds = dict(concept=calibrated['metrics']['concept']['threshold'],
                              spatial=calibrated['metrics']['localization']['threshold'])
        output_path = root/('test_metrics.json' if test else 'calibration.json')
        if output_path.exists():
            prior = json.loads(output_path.read_text())
            if prior['identity'] != identity: raise ValueError('Existing assessment belongs to another checkpoint')
            if test and prior['thresholds'] != thresholds: raise ValueError('Frozen thresholds changed')
            return prior
        backbone = Backbone(cfg.model,data['classes'],data).to(cfg.training.device).eval().requires_grad_(False)
        count = len((Path(cfg.data.prepared)/'concepts.txt').read_text().splitlines())
        head = ConceptHead(cfg.model,backbone.dims,count).to(cfg.training.device).eval()
        head.load_state_dict(selected['model'])
        if cfg.training.channels_last:
            backbone.to(memory_format=torch.channels_last); head.to(memory_format=torch.channels_last)
        dataset = Samples(cfg,'test' if test else 'validation',data,metrics=True)
        loader = DataLoader(dataset,batch_size=cfg.training.batch_size,num_workers=cfg.data.workers,
                            shuffle=False,collate_fn=collate)
        metrics = Metrics(len(dataset),count,cfg.data.cache,calibrate=not test,thresholds=thresholds)
        try:
            values = run_epoch(backbone,head,loader,cfg,metrics=metrics)
        finally:
            metrics.close()
        if not test and (not values['concept']['positives'] or not values['spatial_pairs']):
            raise ValueError('Calibration requires validation concept positives and visible region targets')
        chosen = dict(concept=values['concept']['threshold'],spatial=values['localization']['threshold'])
        result = dict(identity=identity,cohort='test' if test else 'validation',
            checkpoint_epoch=selected['epoch'],metrics=values,thresholds=chosen,
            supervision=('human part-presence and region annotations; supervised ablation'
                         if cfg.data.supervision=='human' else 'GDINO pseudo-labels and boxes, not independent human annotations'))
        result['metrics']['spatial_ap_definition'] = ('mean pixel AP over annotated image-concept pairs; input-resolution human regions'
            if cfg.data.supervision=='human' else values['spatial_ap_definition'])
        save_json(output_path,result)
        return result
