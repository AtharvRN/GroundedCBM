"""Frozen image features, method-specific heads, and checkpoint loading."""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models

from spatial_cbm.layers import MultiScaleConceptLayer


class Backbone(nn.Module):
    def __init__(self, cfg, classes, protocol):
        super().__init__()
        self.is_vit = cfg.backbone == 'clip_vit_b16'
        if self.is_vit:
            if not cfg.checkpoint or cfg.weights != 'none':
                raise ValueError('CLIP ViT requires an explicit local checkpoint')
            from open_clip import create_model
            self.network = create_model('ViT-B-16', pretrained=None).visual
            if str(cfg.checkpoint).endswith('.safetensors'):
                from safetensors.torch import load_file
                state = load_file(str(cfg.checkpoint))
            else:
                payload = torch.load(cfg.checkpoint, map_location='cpu', weights_only=True)
                state = payload.get('state_dict', payload)
            state = {k.removeprefix('module.'): v for k, v in state.items()}
            if any(k.startswith('visual.') for k in state):
                state = {k.removeprefix('visual.'): v for k, v in state.items() if k.startswith('visual.')}
            self.network.load_state_dict(state, strict=True)
            self.dims = (self.network.conv1.out_channels,) * 3
            return
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
        if self.is_vit:
            output = n.forward_intermediates(images, indices=[3, 7, 11],
                normalize_intermediates=False, output_fmt='NCHW')
            # Reuse the spatial head's three feature slots for transformer depths 4/8/12.
            c3, c4, c5 = output['image_intermediates']
            return dict(conv3=c3, conv4=c4, conv5=c5, global_features=output['image_features'])
        x = n.maxpool(n.relu(n.bn1(n.conv1(images))))
        c3 = n.layer2(n.layer1(x)); c4 = n.layer3(c3)
        return dict(conv3=c3, conv4=c4, conv5=n.layer4(c4))

    def classify(self, images):
        if self.is_vit:
            raise ValueError('CLIP ViT is a frozen feature extractor, not a dataset classifier')
        return self.network(images)


class SpatialConceptHead(MultiScaleConceptLayer):
    def __init__(self, cfg, dims, concepts):
        super().__init__(nn.Linear(512 if cfg.backbone == 'clip_vit_b16' else dims[2], concepts), nn.Conv2d(cfg.width, concepts, 1, bias=False),
            conv4_dim=dims[1], conv5_dim=dims[2], fusion_dim=cfg.width,
            global_pool=True, residual_alpha=cfg.alpha, residual_spatial_pooling=cfg.spatial_pooling,
            residual_topk_fraction=cfg.topk_fraction)
        self.stages = cfg.stages
        self.stage_ablation = cfg.stage_ablation
        self.spatial_grid = cfg.spatial_grid
        if cfg.stages in {'conv345', 'vit4812'} or self.stage_ablation:
            self.conv3_proj = nn.Conv2d(dims[0], cfg.width, 1, bias=False)
            self.refine = nn.Conv2d(cfg.width, cfg.width, 3, padding=1, bias=False)
        if self.stage_ablation:
            for stage in ('3', '4', '5'):
                if stage not in self.stages:
                    getattr(self, 'conv'+stage+'_proj').requires_grad_(False)
        self.fusion = cfg.fusion
        self.register_parameter('fusion_logit', nn.Parameter(torch.logit(torch.tensor(float(cfg.alpha))))
                                if cfg.learn_alpha else None)
        self.attention_pool = nn.Conv2d(cfg.width, 1, 1, bias=False) if cfg.spatial_pooling == 'attention' else None
        if self.attention_pool is not None:
            nn.init.zeros_(self.attention_pool.weight)

    def mixing_alpha(self):
        return self.residual_alpha if self.fusion_logit is None else self.fusion_logit.sigmoid()

    def forward_global(self, features):
        if self.stages == 'vit4812':
            return self.global_layer(features['global_features'])
        return super().forward_global(features)

    def forward(self, features):
        if self.attention_pool is None:
            output = super().forward(features)
        else:
            spatial_features = self.forward_spatial_features(features)
            maps = self.spatial_layer(spatial_features)
            weights = self.attention_pool(spatial_features).flatten(2).softmax(-1).reshape(
                spatial_features.shape[0], 1, *spatial_features.shape[-2:])
            pooled = (spatial_features * weights).sum((-2, -1))
            spatial = F.linear(pooled, self.spatial_layer.weight.flatten(1), self.spatial_layer.bias)
            global_logits = self.forward_global(features)
            output = dict(global_logits=global_logits, spatial_logits=spatial, spatial_maps=maps,
                          final_logits=global_logits + self.residual_alpha * spatial,
                          attention_weights=weights, pooled_spatial_features=pooled)
        if self.fusion == 'convex':
            alpha = self.mixing_alpha()
            output['final_logits'] = alpha*output['global_logits'] + (1-alpha)*output['spatial_logits']
        return output

    def forward_spatial_features(self, features):
        if self.stage_ablation and self.stages != 'conv345':
            # Preserve the current conv345 interpolation and nonlinearity order.
            coarse = []
            if '4' in self.stages:
                coarse.append(self.conv4_proj(features['conv4']))
            if '5' in self.stages:
                coarse.append(F.interpolate(self.conv5_proj(features['conv5']),
                    size=features['conv4'].shape[-2:], mode='bilinear', align_corners=False))
            fused = None
            if coarse:
                fused = F.interpolate(F.relu(sum(coarse)), size=features['conv3'].shape[-2:],
                                      mode='bilinear', align_corners=False)
            if '3' in self.stages:
                fine = self.conv3_proj(features['conv3'])
                fused = fine if fused is None else fine + fused
            return F.relu(self.refine(F.relu(fused)), inplace=False)
        fused = self.fuse_spatial_features(features)
        if self.stages in {'conv345', 'vit4812'}:
            fine = self.conv3_proj(features['conv3'])
            size = (self.spatial_grid, self.spatial_grid) if self.spatial_grid else fine.shape[-2:]
            if fine.shape[-2:] != size:
                fine = F.interpolate(fine, size=size, mode='bilinear', align_corners=False)
            fused = fine + F.interpolate(fused, size=size, mode='bilinear', align_corners=False)
            fused = F.relu(self.refine(F.relu(fused)), inplace=False)
        return fused

    def forward_spatial(self, features):
        return self.spatial_layer(self.forward_spatial_features(features))


def ConceptHead(cfg, dims, concepts, indices=None):
    if cfg.method == 'sg_cbm':
        if indices is not None: raise ValueError('SG-CBM does not use baseline concept filtering')
        return SpatialConceptHead(cfg,dims,concepts)
    from .baselines import BaselineHead
    return BaselineHead(cfg,dims,concepts,indices)


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
    for key in ('config','data','source','backbone_sha256','teacher'):
        if selected['protocol'].get(key) != current.get(key):
            raise ValueError(f'Evaluation provenance changed: {key}')
    best = minimum_loss(json.loads((root/'history.json').read_text()))
    if selected['epoch'] != best['epoch'] or selected['val_loss'] != best['val_loss']:
        raise ValueError('Evaluation requires the minimum-validation-loss checkpoint')
    bank = (Path(cfg.data.prepared)/'concepts.txt').read_text().splitlines()
    backbone = Backbone(cfg.model,data['classes'],data).to(cfg.training.device).eval().requires_grad_(False)
    indices = selected['model'].get('concept_indices')
    head = ConceptHead(cfg.model,backbone.dims,len(bank),indices).to(cfg.training.device).eval().requires_grad_(False)
    head.load_state_dict(selected['model'])
    if indices is not None: bank = [bank[i] for i in indices.tolist()]
    if cfg.training.channels_last:
        backbone.to(memory_format=torch.channels_last); head.to(memory_format=torch.channels_last)
    from .parallel import parallelize
    backbone, head = parallelize(backbone, head, cfg.training.devices)
    identity = dict(checkpoint_sha256=digest(root/'best.pt'),split_sha256=data['files']['split.json'],
                    prepared_files=data['files'],source=current['source'])
    if current.get('teacher') is not None: identity['teacher'] = current['teacher']
    return backbone,head,bank,data,identity
