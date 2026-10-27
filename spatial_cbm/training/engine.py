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
from torch.utils.data import DataLoader

from spatial_cbm.losses import weighted_concept_bce, spatial_losses_vectorized
from .data import Samples, collate, digest, save_json, verify
from .model import Backbone, ConceptHead
from .backbone_recipe import mix_batch, update_ema
from .metrics import Metrics


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


def objective(output, batch, cfg):
    _, labels, ids, masks, valid = batch[:5]
    bce = weighted_concept_bce(output['final_logits'].float(), labels, cfg.positive_weight)
    kl, patch = spatial_losses_vectorized(output['spatial_maps'].float(), ids, masks, valid)
    return cfg.fused_bce*bce + cfg.kl*kl + cfg.spatial_bce*patch


def loaders(cfg, protocol, epoch=0):
    concept = cfg.training.stage == 'concept'
    generator = torch.Generator().manual_seed(cfg.training.seed + epoch)
    kwargs = dict(batch_size=cfg.training.batch_size, num_workers=cfg.data.workers,
                  pin_memory=cfg.training.device.startswith('cuda'),
                  collate_fn=collate if concept else None)
    return tuple(DataLoader(Samples(cfg, split, protocol, concepts=concept,metrics=concept and split=='validation'),
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


def run_epoch(backbone, head, loader, cfg, optimizer=None, scaler=None, progress_path=None, ema=None, metrics=None):
    train = optimizer is not None
    concept = head is not None
    backbone.train(train and not concept)
    if head is not None:
        head.train(train)
    total, seen, skipped, began = 0., 0, 0, time.monotonic()
    t = cfg.training
    amp = train and t.precision != 'fp32'
    dtype = torch.float16 if t.precision == 'fp16' else torch.bfloat16
    with torch.set_grad_enabled(train):
        for batch in loader:
            batch = [x.to(t.device, non_blocking=True) for x in batch]
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
                        features = backbone(images)
                    output = head(features)
                else:
                    output = backbone.classify(images)
            # Loss accumulation and validation are FP32 in every dataset.
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
                save_json(progress_path, progress)
                print(json.dumps(progress), flush=True)
    if seen != len(loader.dataset) or not seen:
        raise ValueError('Incomplete epoch')
    result = dict(loss=total/seen, images=seen, seconds=time.monotonic()-began, amp_skipped_steps=skipped)
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
    return dict(version=1, config=cfg.asdict(), data=data, source=source, git_commit=commit,
                dirty=dirty, torch_version=str(torch.__version__),
                backbone_sha256=digest(cfg.model.checkpoint) if cfg.model.checkpoint else None,
                selection='minimum validation loss; earliest tie', validation_precision='fp32')


def fit(cfg, resume=False):
    cfg.validate()
    t = cfg.training
    if not t.device.startswith('cuda') and t.precision == 'fp16':
        raise ValueError('FP16 requires CUDA; use FP32 for CPU tests')
    data = verify(cfg.data.prepared)
    if data['dataset'] != cfg.dataset:
        raise ValueError('Dataset configuration does not match prepared data')
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
            for key in ('config','data','source','backbone_sha256'):
                if previous[key] != protocol[key]:
                    raise ValueError(f'Resume protocol mismatch: {key}')
        elif resume:
            raise FileNotFoundError('No run to resume')
        else:
            save_json(output/'protocol.json', protocol)
        seed_all(t.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        backbone = Backbone(cfg.model, data['classes'], data).to(t.device)
        bank = (Path(cfg.data.prepared)/'concepts.txt').read_text().splitlines()
        head = ConceptHead(cfg.model, backbone.dims, len(bank)).to(t.device) if t.stage == 'concept' else None
        if head is not None:
            backbone.eval().requires_grad_(False)
        model = head if head is not None else backbone
        if t.channels_last:
            backbone.to(memory_format=torch.channels_last)
            model.to(memory_format=torch.channels_last)
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
        if resume:
            state = torch.load(output/'last.pt', map_location='cpu', weights_only=False)
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
            del best_state
        for epoch in range(first+1, t.epochs+1):
            if stop_for_loss(history, t.patience, t.min_epochs):
                break
            seed_all(t.seed+epoch)
            for group in optimizer.param_groups:
                group['lr'] = learning_rate(t, epoch)
            train, val = loaders(cfg, data, epoch)
            tr = run_epoch(backbone, head, train, cfg, optimizer, scaler, output/'progress.json', ema)
            validation_backbone = ema['model'] if ema is not None else backbone
            metrics = Metrics(len(val.dataset),len(bank),cfg.data.cache) if head is not None else None
            try:
                va = run_epoch(validation_backbone, head, val, cfg, progress_path=output/'progress.json',metrics=metrics)
            finally:
                if metrics is not None: metrics.close()
            record = dict(epoch=epoch, train_loss=tr['loss'], val_loss=va['loss'],
                          lr=optimizer.param_groups[0]['lr'], train=tr, validation=va)
            history.append(record)
            best = minimum_loss(history)
            if best['epoch'] == epoch:
                selected_model = ema['model'] if ema is not None else model
                atomic_torch(output/'best.pt', dict(model=selected_model.state_dict(), epoch=epoch,
                    val_loss=record['val_loss'], protocol=protocol))
            atomic_torch(output/'last.pt', dict(model=model.state_dict(), epoch=epoch, history=history,
                optimizer=optimizer.state_dict(), scaler=scaler.state_dict(), protocol=protocol,
                ema_model=ema['model'].state_dict() if ema is not None else None,
                ema_updates=ema['updates'] if ema is not None else 0))
            save_json(output/'history.json', history)
            print(json.dumps(record), flush=True)
        result = dict(complete=True, epochs=len(history), best_epoch=minimum_loss(history)['epoch'],
                      best_val_loss=minimum_loss(history)['val_loss'], stopped_early=len(history)<t.epochs)
        save_json(output/'complete.json', result)
        return result
    finally:
        lock.close()
