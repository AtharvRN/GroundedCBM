"""Shared-gather spatial losses preserving the reference per-image reductions."""
import torch
import torch.nn.functional as F


def spatial_losses_vectorized(maps, indices, targets, valid):
    b,c,h,w=maps.shape
    k=indices.shape[1]
    if k==0:
        zero=maps.sum()*0.
        return zero,zero
    safe_indices=torch.where(valid,indices,torch.zeros_like(indices))
    selected=maps.flatten(2).gather(1,safe_indices.unsqueeze(-1).expand(b,k,h*w)).reshape(b,k,h,w)
    if (h,w)!=targets.shape[-2:]:
        selected=F.interpolate(selected,size=targets.shape[-2:],mode='bilinear',align_corners=False)
    pred=selected.flatten(2).float()
    target=torch.where(valid[:,:,None],targets.flatten(2).float(),0.)
    if not bool(torch.isfinite(target).all()) or bool(((target<0)|(target>1)).any()):
        raise ValueError('Soft box BCE requires finite overlap fractions in [0, 1]')
    mass=target.sum(-1,keepdim=True)
    nonempty=(mass.squeeze(-1)>0)&valid
    distribution=target/mass.clamp_min(1e-6)
    kl=F.kl_div(F.log_softmax(pred,dim=-1),distribution,reduction='none').sum(-1)
    bce=F.binary_cross_entropy_with_logits(pred,target,reduction='none').mean(-1)
    counts=valid.sum(1).clamp_min(1)
    images=valid.any(1).sum().clamp_min(1)
    # Empty valid targets retain their denominator slot; images without pairs do not.
    return tuple(((loss*nonempty).sum(1)/counts).sum()/images for loss in (kl,bce))
