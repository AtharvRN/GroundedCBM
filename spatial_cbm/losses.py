"""Concept BCE and vectorized spatial KL/BCE objectives."""
import torch
import torch.nn.functional as F


def weighted_concept_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: float = 1.0,
) -> torch.Tensor:
    """PyTorch mean BCEWithLogitsLoss; unit positive weight is ordinary BCE."""
    return F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=logits.new_tensor(float(pos_weight)), reduction="mean",
    )


def spatial_losses_vectorized(maps, indices, targets, valid):
    kl, bce, images = spatial_loss_terms(maps, indices, targets, valid)
    return kl/images.clamp_min(1), bce/images.clamp_min(1)


def spatial_loss_terms(maps, indices, targets, valid):
    """Per-image spatial loss sums and their actual eligible-image denominator."""
    b,c,h,w=maps.shape
    k=indices.shape[1]
    if k==0:
        zero=maps.sum()*0.
        return zero,zero,maps.new_zeros(())
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
    images=valid.any(1).sum()
    # Empty valid targets retain their denominator slot; images without pairs do not.
    return (*(((loss*nonempty).sum(1)/counts).sum() for loss in (kl,bce)), images)
