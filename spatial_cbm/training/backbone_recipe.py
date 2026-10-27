"""Mixing and EMA from the archived PartImageNet++ backbone trainer."""
import math
import torch


def mix_batch(images, targets, cfg):
    if (cfg.mixup_alpha <= 0 and cfg.cutmix_alpha <= 0) or torch.rand(()).item() > cfg.mix_prob:
        return targets, targets, 1.
    perm = torch.randperm(images.shape[0], device=images.device)
    cutmix = cfg.cutmix_alpha > 0 and (cfg.mixup_alpha <= 0 or torch.rand(()).item() < cfg.cutmix_switch_prob)
    alpha = cfg.cutmix_alpha if cutmix else cfg.mixup_alpha
    lam = float(torch.distributions.Beta(alpha, alpha).sample())
    if not cutmix:
        paired = images[perm]
        images.mul_(lam).add_(paired, alpha=1-lam)
        return targets, targets[perm], lam
    h,w = images.shape[-2:]
    ratio = math.sqrt(1-lam)
    ch,cw = max(1,int(h*ratio)),max(1,int(w*ratio))
    cy,cx = int(torch.randint(h,())),int(torch.randint(w,()))
    y1,y2 = max(0,cy-ch//2),min(h,cy+ch//2)
    x1,x2 = max(0,cx-cw//2),min(w,cx+cw//2)
    images[:,:,y1:y2,x1:x2] = images[perm,:,y1:y2,x1:x2]
    lam = 1-(y2-y1)*(x2-x1)/(h*w)
    return targets,targets[perm],lam


@torch.no_grad()
def update_ema(ema, model, decay, updates):
    decay = min(decay,(1.+updates)/(10.+updates))
    for average,value in zip(ema.parameters(),model.parameters()):
        average.mul_(decay).add_(value,alpha=1-decay)
    for average,value in zip(ema.buffers(),model.buffers()):
        if torch.is_floating_point(average):
            average.mul_(decay).add_(value,alpha=1-decay)
        else:
            average.copy_(value)
