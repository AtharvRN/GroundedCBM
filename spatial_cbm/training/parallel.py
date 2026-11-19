"""Batch parallelism with persistent frozen-backbone replicas and native checkpoints."""
import copy

import torch
from torch import nn
from torch.nn.parallel import gather, parallel_apply, scatter


class FrozenFeatures(nn.Module):
    def __init__(self, backbone, devices):
        super().__init__()
        if any(p.requires_grad for p in backbone.parameters()):
            raise ValueError('Persistent replicas require a frozen backbone')
        self.device_ids = list(range(devices))
        self.dims = backbone.dims
        self.replicas = nn.ModuleList([backbone] +
            [copy.deepcopy(backbone).to(f'cuda:{i}') for i in self.device_ids[1:]])

    def forward(self, images):
        chunks = scatter(images, self.device_ids, dim=0)
        outputs = parallel_apply(list(self.replicas[:len(chunks)]),
            [(chunk,) for chunk in chunks], devices=self.device_ids[:len(chunks)])
        return gather(outputs, self.device_ids[0], dim=0)


class ParallelHead(nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__('module'), name)


def parallelize(backbone, head, devices):
    if devices == 1:
        return backbone, head
    if torch.cuda.device_count() < devices:
        raise ValueError(f'Requested {devices} CUDA devices; found {torch.cuda.device_count()}')
    return FrozenFeatures(backbone, devices), ParallelHead(head, device_ids=list(range(devices)))
