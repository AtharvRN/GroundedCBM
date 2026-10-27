"""Shared ResNet features and the established SG-CBM spatial head."""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models

from spatial_cbm.layers import MultiScaleConceptLayer


class Backbone(nn.Module):
    def __init__(self, cfg, classes, protocol):
        super().__init__()
        weights = None if cfg.weights == 'none' else cfg.weights
        self.network = models.get_model(cfg.backbone, weights=weights)
        channels = 512 if cfg.backbone == 'resnet18' else 2048
        self.dims = (channels//4, channels//2, channels)
        self.network.fc = nn.Linear(channels, classes)
        if cfg.checkpoint:
            payload = torch.load(cfg.checkpoint, map_location='cpu', weights_only=False)
            if 'protocol' in payload:
                previous = payload['protocol']
                keys = ['split.json']
                if 'training_images_sha256' in previous['data'] and 'training_images_sha256' in protocol:
                    if previous['data']['training_images_sha256'] != protocol['training_images_sha256']:
                        raise ValueError('Backbone and concept head must use the same data and internal split')
                else:
                    keys.append('samples.sqlite')
                for key in keys:
                    if previous['data']['files'][key] != protocol['files'][key]:
                        raise ValueError('Backbone and concept head must use the same data and internal split')
                if previous['config']['training']['stage'] != 'backbone':
                    raise ValueError('Expected a backbone checkpoint, not a concept-head checkpoint')
            state = payload.get('model', payload.get('state_dict', payload))
            state = {k.removeprefix('module.').removeprefix('network.'):v for k,v in state.items()}
            state = {k:v for k,v in state.items() if not k.startswith('fc.')}
            missing, unexpected = self.network.load_state_dict(state, strict=False)
            if set(missing) != {'fc.weight','fc.bias'} or unexpected:
                raise ValueError(f'Backbone checkpoint mismatch: {missing}, {unexpected}')

    def forward(self, images):
        n = self.network
        x = n.maxpool(n.relu(n.bn1(n.conv1(images))))
        c3 = n.layer2(n.layer1(x)); c4 = n.layer3(c3)
        return dict(conv3=c3, conv4=c4, conv5=n.layer4(c4))

    def classify(self, images):
        return self.network(images)


class ConceptHead(MultiScaleConceptLayer):
    def __init__(self, cfg, dims, concepts):
        super().__init__(nn.Linear(dims[2], concepts), nn.Conv2d(cfg.width, concepts, 1, bias=False),
            conv4_dim=dims[1], conv5_dim=dims[2], fusion_dim=cfg.width,
            global_pool=True, residual_alpha=cfg.alpha, residual_spatial_pooling=cfg.spatial_pooling,
            residual_topk_fraction=cfg.topk_fraction)
        self.stages = cfg.stages
        if cfg.stages == 'conv345':
            self.conv3_proj = nn.Conv2d(dims[0], cfg.width, 1, bias=False)
            self.refine = nn.Conv2d(cfg.width, cfg.width, 3, padding=1, bias=False)
        self.fusion = cfg.fusion
        self.register_parameter('fusion_logit', nn.Parameter(torch.logit(torch.tensor(float(cfg.alpha))))
                                if cfg.learn_alpha else None)

    def mixing_alpha(self):
        return self.residual_alpha if self.fusion_logit is None else self.fusion_logit.sigmoid()

    def forward(self, features):
        output = super().forward(features)
        if self.fusion == 'convex':
            alpha = self.mixing_alpha()
            output['final_logits'] = alpha*output['global_logits'] + (1-alpha)*output['spatial_logits']
        return output

    def forward_spatial(self, features):
        fused = self.fuse_spatial_features(features)
        if self.stages == 'conv345':
            fused = self.conv3_proj(features['conv3']) + F.interpolate(
                fused, size=features['conv3'].shape[-2:], mode='bilinear', align_corners=False)
            fused = F.relu(self.refine(F.relu(fused)), inplace=False)
        return self.spatial_layer(fused)


def load_selected(cfg):
    """Load the finished minimum-loss head with its frozen backbone and identity."""
    import json
    from pathlib import Path
    from .data import verify, digest
    from .engine import provenance, minimum_loss

    root = Path(cfg.training.output)
    if cfg.training.stage != 'concept' or not (root/'complete.json').exists():
        raise ValueError('Finish concept training before evaluation')
    data = verify(cfg.data.prepared)
    selected = torch.load(root/'best.pt',map_location='cpu',weights_only=False)
    current = provenance(cfg,data)
    for key in ('config','data','source','backbone_sha256'):
        if selected['protocol'][key] != current[key]:
            raise ValueError(f'Evaluation provenance mismatch: {key}')
    best = minimum_loss(json.loads((root/'history.json').read_text()))
    if selected['epoch'] != best['epoch'] or selected['val_loss'] != best['val_loss']:
        raise ValueError('Evaluation requires the minimum-validation-loss checkpoint')
    bank = (Path(cfg.data.prepared)/'concepts.txt').read_text().splitlines()
    backbone = Backbone(cfg.model,data['classes'],data).to(cfg.training.device).eval().requires_grad_(False)
    head = ConceptHead(cfg.model,backbone.dims,len(bank)).to(cfg.training.device).eval().requires_grad_(False)
    head.load_state_dict(selected['model'])
    if cfg.training.channels_last:
        backbone.to(memory_format=torch.channels_last); head.to(memory_format=torch.channels_last)
    identity = dict(checkpoint_sha256=digest(root/'best.pt'),split_sha256=data['files']['split.json'],
                    prepared_files=data['files'],source=current['source'])
    return backbone,head,bank,data,identity
