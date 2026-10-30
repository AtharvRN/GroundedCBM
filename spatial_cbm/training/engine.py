"""One training loop for all datasets, with validation-loss-only selection."""
import copy
import fcntl
import json
import math
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, RandomSampler

from spatial_cbm.losses import weighted_concept_bce, spatial_loss_terms
from .data import Samples, collate, digest, save_json, verify
from .model import Backbone, ConceptHead
from .backbone_recipe import mix_batch, update_ema
from .metrics import Metrics
from .baselines import (REGRESSION, RegressionSamples, RegressionMeter, NonSingletonBatches,
                        regression_collate, cubic_similarity, initial_indices, subset_targets, finalize_lf,
                        prepare_regression_features)


def minimum_loss(history):
    if not history:
        return None
    if any("val_loss" not in row or not math.isfinite(row["val_loss"]) for row in history):
        raise ValueError("Validation-loss history missing or nonfinite; legacy resume requires a new run")
    return min(history, key=lambda row: (row["val_loss"], row["epoch"]))


def stop_for_loss(history, patience, min_epochs):
    if patience < 1 or min_epochs < 1:
        raise ValueError("Validation-loss patience and minimum epochs must be positive")
    best = minimum_loss(history)
    return bool(best and history[-1]["epoch"] >= min_epochs
                and history[-1]["epoch"] - best["epoch"] >= patience)


def atomic_torch(path, value):
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def _apply_checkpoint(output, transaction):
    if transaction['best'] is not None:
        atomic_torch(output/'best.pt', transaction['best'])
    atomic_torch(output/'last.pt', transaction['last'])
    save_json(output/'history.json', transaction['last']['history'])
    (output/'epoch.pending.pt').unlink()


def recover_checkpoint(output):
    """Replay a committed epoch journal under the run's exclusive writer lock."""
    pending = output/'epoch.pending.pt'
    if pending.exists():
        transaction = torch.load(pending, map_location='cpu', weights_only=False)
        _apply_checkpoint(output, transaction)
        return transaction['last']
    if (output/'last.pt').exists():
        state = torch.load(output/'last.pt', map_location='cpu', weights_only=False)
        # Also repairs interrupted legacy writes when best and last agree.
        save_json(output/'history.json', state['history'])
        return state
    if any((output/name).exists() for name in ('best.pt', 'history.json')):
        raise ValueError('Incomplete legacy checkpoint without last.pt; cannot safely resume')
    return None


def publish_checkpoint(output, last, best):
    # The journal is the commit point; every subsequent publication is replayable.
    transaction = dict(last=last, best=best)
    atomic_torch(output/'epoch.pending.pt', transaction)
    _apply_checkpoint(output, transaction)


def objective_terms(output, batch, cfg):
    _, labels, ids, masks, valid = batch[:5]
    bce = weighted_concept_bce(output['final_logits'].float(), labels, cfg.positive_weight)
    kl, patch, images = spatial_loss_terms(output['spatial_maps'].float(), ids, masks, valid)
    sums = torch.stack((bce*labels.numel(), kl, patch))
    counts = torch.stack((images.new_tensor(labels.numel()), images, images)).to(sums)
    return sums, counts


def objective(output, batch, cfg):
    sums, counts = objective_terms(output, batch, cfg)
    return (sums/counts.clamp_min(1)*sums.new_tensor([cfg.fused_bce,cfg.kl,cfg.spatial_bce])).sum()


def loaders(cfg, protocol, epoch=0, teacher_manifest=None, cached_features=None):
    concept = cfg.training.stage == 'concept'
    generator = torch.Generator().manual_seed(cfg.training.seed + epoch)
    kwargs = dict(batch_size=cfg.training.batch_size, num_workers=cfg.data.workers,
                  pin_memory=cfg.training.device.startswith('cuda'),
                  collate_fn=collate if concept else None)
    if concept and cfg.model.method in REGRESSION:
        if protocol['train']<3 or protocol['validation']<2:
            raise ValueError('Regression needs at least three training and two validation images')
        cached_features = cached_features or {}
        train = RegressionSamples(cfg,'train',protocol,teacher_manifest=teacher_manifest,feature_paths=cached_features.get('train'))
        val = RegressionSamples(cfg,'validation',protocol,metrics=True,teacher_manifest=teacher_manifest,feature_paths=cached_features.get('validation'))
        batches = NonSingletonBatches(RandomSampler(train,generator=generator),cfg.training.batch_size,False)
        shared = dict(num_workers=cfg.data.workers,pin_memory=cfg.training.device.startswith('cuda'),collate_fn=regression_collate)
        return (DataLoader(train,batch_sampler=batches,**shared),
                DataLoader(val,batch_size=cfg.training.evaluation_batch_size or cfg.training.batch_size,shuffle=False,**shared))
    return tuple(DataLoader(Samples(cfg, split, protocol, concepts=concept,metrics=concept and split=='validation',
        augment=concept and split=='train' and cfg.model.method=='vlg_cbm'),
        shuffle=split=='train', generator=generator if split=='train' else
        torch.Generator().manual_seed(cfg.training.seed), **kwargs) for split in ('train','validation'))


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def learning_rate(cfg, epoch):
    if epoch <= cfg.warmup_epochs:
        return cfg.lr * epoch / cfg.warmup_epochs
    if cfg.schedule == 'constant':
        return cfg.lr
    fraction = (epoch-cfg.warmup_epochs-1) / max(1, cfg.epochs-cfg.warmup_epochs-1)
    return cfg.min_lr + .5*(cfg.lr-cfg.min_lr)*(1+math.cos(math.pi*fraction))


def run_epoch(backbone, head, loader, cfg, optimizer=None, scaler=None, progress_path=None, ema=None, metrics=None, regression_meter=None):
    train = optimizer is not None
    concept = head is not None
    backbone.train(train and not concept)
    if head is not None:
        head.train(train)
    total, seen, skipped, began = 0., 0, 0, time.monotonic()
    t = cfg.training
    amp = train and t.precision != 'fp32'
    dtype = torch.float16 if t.precision == 'fp16' else torch.bfloat16
    regression = concept and cfg.model.method in REGRESSION
    loss_sums = torch.zeros(3, dtype=torch.float64, device=t.device)
    loss_counts = torch.zeros_like(loss_sums)
    spatial_objective = concept and cfg.model.method == 'sg_cbm'
    teacher_batches = regression and hasattr(loader.dataset,'teacher')
    own_meter = teacher_batches and not train and regression_meter is None
    if own_meter: regression_meter = RegressionMeter(len(loader.dataset),cfg.data.cache)
    with torch.set_grad_enabled(train):
        for batch in loader:
            batch = [x.to(t.device, non_blocking=True) for x in batch]
            target = None
            if teacher_batches:
                target = batch[-2].index_select(1,head.concept_indices)
                batch = [*batch[:-2],batch[-1]]
            if concept and hasattr(head,'concept_indices'):
                batch = subset_targets(batch,head.concept_indices)
            images = batch[0]
            if t.channels_last:
                images = images.contiguous(memory_format=torch.channels_last)
            if train:
                optimizer.zero_grad(set_to_none=True)
            if not concept:
                target_a,target_b,lam = mix_batch(images,batch[1],t) if train else (batch[1],batch[1],1.)
            with torch.autocast(device_type=torch.device(t.device).type, dtype=dtype, enabled=amp):
                if concept:
                    with torch.no_grad():
                        features = ({'conv5':images} if getattr(loader.dataset,'cached_features',False)
                                    else backbone(images))
                    output = head(features)
                else:
                    output = backbone.classify(images)
            # Loss accumulation and validation are FP32 in every dataset.
            if regression:
                prediction = output['spatial_maps'] if cfg.model.method=='salf_cbm' else output['final_logits']
                loss = -cubic_similarity(prediction,target).mean() if train else prediction.sum()*0
                if regression_meter is not None: regression_meter.update(prediction,target)
            elif concept and cfg.model.method=='vlg_cbm':
                loss = weighted_concept_bce(output['final_logits'].float(),batch[1],t.positive_weight)
            elif spatial_objective:
                sums, counts = objective_terms(output, batch, t)
                loss = (sums/counts.clamp_min(1)*sums.new_tensor([t.fused_bce,t.kl,t.spatial_bce])).sum()
                loss_sums += sums.detach().double()
                loss_counts += counts.detach().double()
            else:
                loss = objective(output, batch, t) if concept else (
                lam*F.cross_entropy(output.float(),target_a,label_smoothing=t.label_smoothing)
                +(1-lam)*F.cross_entropy(output.float(),target_b,label_smoothing=t.label_smoothing))
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training/validation loss')
            if metrics is not None:
                if train or not concept: raise ValueError('Concept metrics require validation/evaluation mode')
                metrics.update(output,batch)
            if train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                params = head.parameters() if concept else backbone.parameters()
                finite = all(torch.isfinite(p.grad).all() for p in params if p.grad is not None)
                if not finite and not scaler.is_enabled():
                    raise ValueError('Nonfinite gradients')
                skipped += not finite
                scaler.step(optimizer); scaler.update()
                if ema is not None and finite:
                    ema['updates'] += 1
                    update_ema(ema['model'],backbone,t.ema_decay,ema['updates'])
            total += float(loss.detach())*len(images)
            seen += len(images)
            if progress_path is not None and (seen % (t.batch_size*100) == 0 or seen == len(loader.dataset)):
                progress = dict(stage='training' if train else 'validation', images=seen,
                    total=len(loader.dataset), loss=total/seen, seconds=time.monotonic()-began,
                    images_per_second=seen/(time.monotonic()-began))
                if spatial_objective:
                    progress['loss'] = float((loss_sums/loss_counts.clamp_min(1)*
                        loss_sums.new_tensor([t.fused_bce,t.kl,t.spatial_bce])).sum())
                save_json(progress_path, progress)
                print(json.dumps(progress), flush=True)
    if seen != len(loader.dataset) or not seen:
        raise ValueError('Incomplete epoch')
    result = dict(loss=total/seen, images=seen, seconds=time.monotonic()-began, amp_skipped_steps=skipped)
    if spatial_objective:
        averages = loss_sums/loss_counts.clamp_min(1)
        result['loss'] = float((averages*averages.new_tensor([t.fused_bce,t.kl,t.spatial_bce])).sum())
        result['loss_components'] = dict(zip(('concept_bce','spatial_kl','spatial_bce'),averages.tolist()))
        result['loss_denominators'] = dict(zip(('concept_bce','spatial_kl','spatial_bce'),loss_counts.tolist()))
        result['loss_definition'] = 'cohort concept-element mean BCE plus eligible-image mean spatial KL/BCE'
    if regression_meter is not None:
        try: result['loss'] = -float(regression_meter.similarities().mean())
        finally:
            if own_meter: regression_meter.close()
        result['loss_definition'] = 'full-cohort image-axis centered-cubic cosine, mean over concepts/cells'
    elif regression and not train:
        result['loss'] = None
        result['loss_definition'] = 'Teacher regression loss not computed against annotation labels'
    if metrics is not None: result.update(metrics.result())
    result['seconds'] = time.monotonic()-began
    return result


def provenance(cfg, data):
    root = Path(__file__).resolve().parents[1]
    paths = list((root/'training').glob('*.py')) + [root/x for x in
        ['__init__.py','losses.py','layers.py','targets.py']]
    source = {str(p.relative_to(root.parent)):digest(p) for p in sorted(paths)}
    try:
        commit = subprocess.check_output(['git','rev-parse','HEAD'], cwd=root.parent, text=True).strip()
        dirty = bool(subprocess.check_output(['git','status','--porcelain'], cwd=root.parent, text=True).strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit, dirty = None, None
    teacher = None
    if cfg.training.stage=='concept' and cfg.model.method in REGRESSION:
        from .teacher import verify_teacher
        teacher = verify_teacher(cfg,data)
    return dict(version=1, config=cfg.asdict(), data=data, source=source, git_commit=commit,
                dirty=dirty, torch_version=str(torch.__version__),
                backbone_sha256=digest(cfg.model.checkpoint) if cfg.model.checkpoint else None,
                teacher=teacher,selection='minimum validation loss; earliest tie', validation_precision='fp32')


def fit(cfg, resume=False):
    cfg.validate()
    t = cfg.training
    if not t.device.startswith('cuda') and t.precision == 'fp16':
        raise ValueError('FP16 requires CUDA; use FP32 for CPU tests')
    data = verify(cfg.data.prepared)
    if data['dataset'] != cfg.dataset:
        raise ValueError('Dataset configuration does not match prepared data')
    if data.get('supervision','gdino') != cfg.data.supervision:
        raise ValueError('Configured supervision differs from prepared annotations')
    if not data.get('test_disjoint_verified'):
        raise ValueError('Prepare an explicit disjoint held-out test manifest before training')
    output = Path(t.output); output.mkdir(parents=True, exist_ok=True)
    lock = (output/'writer.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol = provenance(cfg, data)
        if (output/'protocol.json').exists():
            if not resume:
                raise FileExistsError('Existing run; use explicit --resume')
            previous = json.loads((output/'protocol.json').read_text())
            for key in ('config','data','source','backbone_sha256','teacher'):
                if previous[key] != protocol[key]:
                    raise ValueError(f'Resume protocol mismatch: {key}')
        elif resume:
            raise FileNotFoundError('No run to resume')
        else:
            save_json(output/'protocol.json', protocol)
        if resume:
            (output/'complete.json').unlink(missing_ok=True)
        restored = recover_checkpoint(output) if resume else None
        seed_all(t.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        backbone = Backbone(cfg.model, data['classes'], data).to(t.device)
        bank = (Path(cfg.data.prepared)/'concepts.txt').read_text().splitlines()
        indices = initial_indices(cfg,data,protocol['teacher']) if t.stage=='concept' else None
        head = ConceptHead(cfg.model, backbone.dims, len(bank),indices).to(t.device) if t.stage == 'concept' else None
        concept_count = len(head.concept_indices) if head is not None and hasattr(head,'concept_indices') else len(bank)
        if head is not None:
            backbone.eval().requires_grad_(False)
        model = head if head is not None else backbone
        if t.channels_last:
            backbone.to(memory_format=torch.channels_last)
            model.to(memory_format=torch.channels_last)
        cached_features = prepare_regression_features(cfg,backbone,protocol) if head is not None else None
        ema = (dict(model=copy.deepcopy(backbone).eval().requires_grad_(False),updates=0)
               if t.stage == 'backbone' and t.ema_decay else None)
        options = dict(lr=t.lr, weight_decay=t.weight_decay)
        if t.optimizer == 'sgd':
            options['momentum'] = .9
        optimizer = {'adam':torch.optim.Adam,'adamw':torch.optim.AdamW,'sgd':torch.optim.SGD}[t.optimizer](model.parameters(), **options)
        scaler_options = dict(enabled=t.device.startswith('cuda') and t.precision=='fp16', init_scale=128.)
        scaler = (torch.amp.GradScaler('cuda', **scaler_options) if hasattr(torch.amp, 'GradScaler')
                  else torch.cuda.amp.GradScaler(**scaler_options))
        history, first = [], 0
        if restored is not None:
            state = restored
            model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
            scaler.load_state_dict(state['scaler'])
            history, first = state['history'], state['epoch']
            if ema is not None:
                ema['model'].load_state_dict(state['ema_model'])
                ema['updates'] = state['ema_updates']
            minimum_loss(history)
            if not (output/'best.pt').is_file():
                raise FileNotFoundError('Resume requires the retained best checkpoint')
            best_state = torch.load(output/'best.pt', map_location='cpu', weights_only=False)
            best_record = minimum_loss(history)
            if best_state['epoch'] != best_record['epoch'] or best_state['val_loss'] != best_record['val_loss']:
                raise ValueError('Interrupted checkpoint transaction: best and last disagree; audit before resuming')
            del best_state, state, restored
        for epoch in range(first+1, t.epochs+1):
            if stop_for_loss(history, t.patience, t.min_epochs):
                break
            seed_all(t.seed+epoch)
            for group in optimizer.param_groups:
                group['lr'] = learning_rate(t, epoch)
            train, val = loaders(cfg, data, epoch,teacher_manifest=protocol['teacher'],cached_features=cached_features)
            tr = run_epoch(backbone, head, train, cfg, optimizer, scaler, output/'progress.json', ema)
            validation_backbone = ema['model'] if ema is not None else backbone
            annotated = data['annotated_images']==data['images']
            metrics = Metrics(len(val.dataset),concept_count,cfg.data.cache) if head is not None and annotated else None
            try:
                va = run_epoch(validation_backbone, head, val, cfg, progress_path=output/'progress.json',metrics=metrics)
            finally:
                if metrics is not None: metrics.close()
            record = dict(epoch=epoch, train_loss=tr['loss'], val_loss=va['loss'],
                          lr=optimizer.param_groups[0]['lr'], train=tr, validation=va)
            if head is not None and head.fusion == 'convex':
                alpha = head.mixing_alpha()
                record['fusion_alpha'] = alpha.detach().item() if torch.is_tensor(alpha) else alpha
            history.append(record)
            best = minimum_loss(history)
            selected = None
            if best['epoch'] == epoch:
                selected_model = ema['model'] if ema is not None else model
                selected = dict(model=selected_model.state_dict(), epoch=epoch,
                    val_loss=record['val_loss'], protocol=protocol)
            publish_checkpoint(output, dict(model=model.state_dict(), epoch=epoch, history=history,
                optimizer=optimizer.state_dict(), scaler=scaler.state_dict(), protocol=protocol,
                ema_model=ema['model'].state_dict() if ema is not None else None,
                ema_updates=ema['updates'] if ema is not None else 0), selected)
            print(json.dumps(record), flush=True)
        result = dict(complete=True, epochs=len(history), best_epoch=minimum_loss(history)['epoch'],
                      best_val_loss=minimum_loss(history)['val_loss'], stopped_early=len(history)<t.epochs)
        if head is not None and head.fusion == 'convex':
            result['best_alpha'] = minimum_loss(history)['fusion_alpha']
        if head is not None and cfg.model.method=='lf_cbm':
            finalize_lf(cfg,backbone,data,teacher_manifest=protocol['teacher'],cached_features=cached_features)
        save_json(output/'complete.json', result)
        return result
    finally:
        lock.close()
