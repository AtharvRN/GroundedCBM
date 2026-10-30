"""Strict YAML configuration; no dataset-specific training dispatch."""
from dataclasses import asdict, dataclass, field, fields
import math
from pathlib import Path

import yaml


@dataclass
class Data:
    prepared: str = 'prepared/cub'
    cache: str = '/tmp/sgcbm-target-cache'
    input_size: int = 224
    resize_size: int = 256
    confidence: float = .15
    workers: int = 8
    supervision: str = 'gdino'


@dataclass
class Model:
    backbone: str = 'resnet50'
    checkpoint: str = ''
    weights: str = 'none'
    stages: str = 'conv345'
    width: int = 2048
    alpha: float = .1
    fusion: str = 'residual'
    learn_alpha: bool = False
    spatial_pooling: str = 'lse'
    topk_fraction: float = .2
    method: str = 'sg_cbm'
    salf_grid: int = 7
    vlg_hidden: int = 0


@dataclass
class Training:
    stage: str = 'concept'
    output: str = 'runs/cub'
    seed: int = 42
    device: str = 'cuda'
    epochs: int = 75
    patience: int = 10
    min_epochs: int = 10
    batch_size: int = 32
    optimizer: str = 'adam'
    lr: float = .0001
    weight_decay: float = .00001
    precision: str = 'fp16'
    channels_last: bool = True
    fused_bce: float = 1.
    kl: float = .25
    spatial_bce: float = .25
    positive_weight: float = 1.
    label_smoothing: float = .1
    schedule: str = 'constant'
    warmup_epochs: int = 0
    min_lr: float = .00001
    randaugment_ops: int = 0
    randaugment_magnitude: int = 14
    random_erasing: float = 0.
    mixup_alpha: float = 0.
    cutmix_alpha: float = 0.
    mix_prob: float = 1.
    cutmix_switch_prob: float = .5
    ema_decay: float = 0.
    vlg_crop_probability: float = .5
    filter_concepts: bool = True
    lf_clip_cutoff: float = .25
    lf_interpretability_cutoff: float = .45
    cache_regression_features: bool = False
    feature_batch_size: int = 128
    evaluation_batch_size: int = 0


@dataclass
class Teacher:
    cache: str = ''
    model: str = 'ViT-B/16'
    checkpoint: str = ''
    batch_size: int = 64
    prompt_radius: int = 32
    prompt_grid: str = 'interior'


@dataclass
class Config:
    dataset: str = 'cub'
    data: Data = field(default_factory=Data)
    model: Model = field(default_factory=Model)
    training: Training = field(default_factory=Training)
    teacher: Teacher = field(default_factory=Teacher)

    def validate(self):
        if self.dataset not in {'cub', 'imagenet', 'places365', 'partimagenetpp'}:
            raise ValueError('Unknown dataset')
        d, m, t = self.data, self.model, self.training
        if type(t.filter_concepts) is not bool:
            raise ValueError('filter_concepts must be a boolean')
        if type(t.cache_regression_features) is not bool or type(t.feature_batch_size) is not int or t.feature_batch_size<1:
            raise ValueError('Invalid frozen-feature cache settings')
        if type(t.evaluation_batch_size) is not int or t.evaluation_batch_size<0:
            raise ValueError('Invalid evaluation batch size')
        if t.cache_regression_features and (m.method not in {'lf_cbm','salf_cbm'} or t.stage!='concept' or t.precision!='fp32'):
            raise ValueError('Frozen regression feature caching requires LF/SALF concept training in FP32')
        if m.method not in {'sg_cbm','vlg_cbm','lf_cbm','salf_cbm'}:
            raise ValueError('Unknown concept bottleneck method')
        if type(m.salf_grid) is not int or m.salf_grid < 1 or type(m.vlg_hidden) is not int or m.vlg_hidden < 0:
            raise ValueError('Invalid baseline head dimensions')
        if m.method != 'sg_cbm' and (m.fusion != 'residual' or m.learn_alpha or m.spatial_pooling != 'lse'):
            raise ValueError('SG-CBM fusion/pooling options do not configure baseline heads')
        if not math.isfinite(t.vlg_crop_probability) or not 0 <= t.vlg_crop_probability <= 1:
            raise ValueError('Invalid VLG crop probability')
        if any(not math.isfinite(v) or not -1 <= v <= 1 for v in (t.lf_clip_cutoff,t.lf_interpretability_cutoff)):
            raise ValueError('Invalid LF concept-filter cutoff')
        if m.method in {'lf_cbm','salf_cbm'} and t.stage=='concept':
            if not self.teacher.cache or not self.teacher.checkpoint:
                raise ValueError('LF/SALF require an explicit teacher cache and CLIP checkpoint')
            if t.batch_size < 3:
                raise ValueError('Centered-cubic training requires at least three images per batch')
        if type(self.teacher.batch_size) is not int or self.teacher.batch_size < 1 or type(self.teacher.prompt_radius) is not int or self.teacher.prompt_radius < 1:
            raise ValueError('Invalid teacher batch size or circle radius')
        if self.teacher.prompt_grid not in {'interior','edge'}:
            raise ValueError('Teacher prompt_grid must be interior or edge')
        if m.method == 'salf_cbm' and self.teacher.prompt_grid == 'edge' and m.salf_grid < 2:
            raise ValueError('Edge prompting needs at least two grid positions')
        if m.method == 'vlg_cbm' and d.supervision != 'gdino':
            raise ValueError('VLG-CBM uses detector concept labels')
        if d.supervision not in {'gdino','human'} or (d.supervision=='human' and self.dataset!='partimagenetpp'):
            raise ValueError('Human supervision is supported only for PartImageNet++')
        if m.backbone not in {'resnet18', 'resnet50', 'resnet101'} or m.stages not in {'conv45', 'conv345'}:
            raise ValueError('Unsupported backbone or spatial stages')
        if m.fusion not in {'residual', 'convex'} or type(m.learn_alpha) is not bool:
            raise ValueError('Unsupported concept fusion configuration')
        if m.spatial_pooling not in {'lse','topk'} or not math.isfinite(m.topk_fraction) or not 0 < m.topk_fraction <= 1:
            raise ValueError('Spatial pooling requires lse/topk and a finite top-k fraction in (0, 1]')
        if m.fusion == 'convex' and not 0 <= m.alpha <= 1:
            raise ValueError('Convex fusion alpha must be in [0, 1]')
        if m.learn_alpha and (m.fusion != 'convex' or not 0 < m.alpha < 1):
            raise ValueError('Learned alpha requires convex fusion and initialization inside (0, 1)')
        if t.stage not in {'backbone', 'concept'} or t.optimizer not in {'adam', 'adamw', 'sgd'}:
            raise ValueError('Unsupported training stage or optimizer')
        if t.precision not in {'fp32', 'fp16', 'bf16'}:
            raise ValueError('Unsupported precision')
        if t.schedule not in {'constant','cosine'} or not 0 <= t.warmup_epochs < t.epochs:
            raise ValueError('Invalid schedule or warmup')
        if t.randaugment_ops < 0 or not 0 <= t.randaugment_magnitude <= 30:
            raise ValueError('Invalid RandAugment configuration')
        if not all(0 <= v <= 1 for v in (t.random_erasing,t.mix_prob,t.cutmix_switch_prob)) or not 0 <= t.ema_decay < 1:
            raise ValueError('Invalid augmentation or EMA probability')
        if not all(math.isfinite(v) and v >= 0 for v in (t.mixup_alpha,t.cutmix_alpha)):
            raise ValueError('Invalid mixing concentration')
        if t.stage == 'concept' and any((t.randaugment_ops,t.random_erasing,t.mixup_alpha,t.cutmix_alpha,t.ema_decay)):
            raise ValueError('Backbone augmentations/EMA cannot silently alter concept supervision')
        if not math.isfinite(t.min_lr) or not 0 <= t.min_lr <= t.lr:
            raise ValueError('Invalid minimum learning rate')
        if m.weights not in {'none', 'IMAGENET1K_V1', 'IMAGENET1K_V2'} or (m.checkpoint and m.weights != 'none'):
            raise ValueError('Specify one checkpoint or torchvision weights, not both')
        if t.stage == 'concept' and not m.checkpoint and m.weights == 'none':
            raise ValueError('Concept training requires an explicit frozen backbone')
        for value in (t.epochs, t.patience, t.min_epochs, t.batch_size, m.width, d.input_size, d.resize_size):
            if type(value) is not int or value < 1:
                raise ValueError('Sizes, epochs, and patience must be positive integers')
        if t.min_epochs > t.epochs or d.input_size % 32 or d.resize_size < d.input_size or d.workers < 0:
            raise ValueError('Invalid epoch, image, or worker configuration')
        if not 0 <= d.confidence <= 1 or not 0 <= t.label_smoothing < 1:
            raise ValueError('Invalid confidence or label smoothing')
        for value in (t.lr, t.weight_decay, t.fused_bce, t.kl, t.spatial_bce, t.positive_weight, m.alpha):
            if not math.isfinite(value) or value < 0:
                raise ValueError('Hyperparameters must be finite and nonnegative')
        if t.lr == 0 or t.positive_weight == 0 or t.fused_bce + t.kl + t.spatial_bce == 0:
            raise ValueError('Learning rate, positive weight, and objective must be positive')
        return self

    def asdict(self):
        return asdict(self)


def strict(cls, value):
    if not isinstance(value, dict) or set(value) - {f.name for f in fields(cls)}:
        raise ValueError(f'Unknown fields or malformed {cls.__name__} configuration')
    return cls(**value)


def load(path):
    value = yaml.safe_load(Path(path).read_text())
    if not isinstance(value, dict) or set(value) - {'dataset', 'data', 'model', 'training', 'teacher'}:
        raise ValueError('Unknown top-level configuration fields')
    return Config(dataset=value.get('dataset', 'cub'), data=strict(Data, value.get('data', {})),
                  model=strict(Model, value.get('model', {})),
                  training=strict(Training, value.get('training', {})),
                  teacher=strict(Teacher, value.get('teacher', {}))).validate()
