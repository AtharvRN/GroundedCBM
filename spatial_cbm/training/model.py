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
                for key in ('samples.sqlite', 'split.json'):
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
            global_pool=True, residual_alpha=cfg.alpha, residual_spatial_pooling='lse')
        self.stages = cfg.stages
        if cfg.stages == 'conv345':
            self.conv3_proj = nn.Conv2d(dims[0], cfg.width, 1, bias=False)
            self.refine = nn.Conv2d(cfg.width, cfg.width, 3, padding=1, bias=False)

    def forward_spatial(self, features):
        fused = self.fuse_spatial_features(features)
        if self.stages == 'conv345':
            fused = self.conv3_proj(features['conv3']) + F.interpolate(
                fused, size=features['conv3'].shape[-2:], mode='bilinear', align_corners=False)
            fused = F.relu(self.refine(F.relu(fused)), inplace=False)
        return self.spatial_layer(fused)
