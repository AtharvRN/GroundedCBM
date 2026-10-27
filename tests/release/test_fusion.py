from pathlib import Path

import pytest
import torch

from spatial_cbm.training.config import Config, Model
from spatial_cbm.training.engine import fit
from spatial_cbm.training.export import export
from spatial_cbm.training.metrics import assess
from spatial_cbm.training.model import ConceptHead, load_selected
from test_training import config, inputs, threads


def features():
    return dict(conv3=torch.randn(2, 2, 8, 8), conv4=torch.randn(2, 4, 4, 4),
                conv5=torch.randn(2, 8, 2, 2))


@pytest.mark.parametrize('alpha', [0., .25, .5, .75, .9, 1.])
def test_convex_formula_gradients_and_unchanged_initialization(alpha):
    torch.manual_seed(6)
    head = ConceptHead(Model(width=8, alpha=alpha, fusion='convex'), (2, 4, 8), 3)
    torch.manual_seed(6)
    baseline = ConceptHead(Model(width=8), (2, 4, 8), 3)
    for key, value in head.state_dict().items():
        torch.testing.assert_close(value, baseline.state_dict()[key], rtol=0, atol=0)
    out = head(features())
    expected = alpha*out['global_logits'] + (1-alpha)*out['spatial_logits']
    torch.testing.assert_close(out['final_logits'], expected, rtol=0, atol=0)
    if alpha in (0., 1.):
        torch.testing.assert_close(out['final_logits'], out['global_logits' if alpha else 'spatial_logits'])
    gradients = torch.autograd.grad(out['final_logits'].sum(), (out['global_logits'], out['spatial_logits']))
    for actual, coefficient in zip(gradients, (alpha, 1-alpha)):
        torch.testing.assert_close(actual, torch.full_like(actual, coefficient))


def test_learned_alpha_gradient_bounds_and_checkpoint_roundtrip(tmp_path):
    cfg = Model(width=8, alpha=.5, fusion='convex', learn_alpha=True)
    torch.manual_seed(6)
    head = ConceptHead(cfg, (2, 4, 8), 3)
    assert head.fusion_logit.numel() == 1
    assert head.mixing_alpha().item() == .5
    feats = features()
    out = head(feats)
    actual = torch.autograd.grad(out['final_logits'].sum(), head.fusion_logit)[0]
    expected = .25*(out['global_logits']-out['spatial_logits']).sum()
    torch.testing.assert_close(actual, expected)
    optimizer = torch.optim.Adam(head.parameters(), lr=.01)
    for _ in range(3):
        optimizer.zero_grad()
        head(feats)['final_logits'].square().mean().backward()
        optimizer.step()
    assert 0 < head.mixing_alpha().item() < 1
    assert head.mixing_alpha().item() != .5
    checkpoint = tmp_path/'head.pt'
    torch.save(head.state_dict(), checkpoint)
    restored = ConceptHead(cfg, (2, 4, 8), 3)
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    torch.testing.assert_close(head(feats)['final_logits'], restored(feats)['final_logits'], rtol=0, atol=0)


@pytest.mark.parametrize('options', [dict(alpha=-.1), dict(alpha=1.1), dict(alpha=float('nan')),
                                    dict(alpha=0., learn_alpha=True), dict(alpha=1., learn_alpha=True),
                                    dict(fusion='residual', learn_alpha=True), dict(fusion='unknown'),
                                    dict(learn_alpha='true')])
def test_invalid_fusion_configuration(options):
    cfg = Config(model=Model(weights='IMAGENET1K_V1', **dict({'fusion':'convex'}, **options)))
    with pytest.raises(ValueError):
        cfg.validate()


@pytest.mark.parametrize('learned', [False, True])
def test_alpha_survives_selection_calibration_export_and_reload(tmp_path, learned):
    prepared, _ = inputs(tmp_path)
    cfg = config(tmp_path, prepared)
    cfg.model.fusion = 'convex'; cfg.model.alpha = .5 if learned else .9; cfg.model.learn_alpha = learned
    result = fit(cfg)
    selected = torch.load(Path(cfg.training.output)/'best.pt', weights_only=False)
    expected = selected['model']['fusion_logit'].sigmoid().item() if learned else cfg.model.alpha
    assert result['best_alpha'] == expected
    assert 0 < result['best_alpha'] < 1
    assert result == fit(cfg, resume=True)
    assert assess(cfg)['checkpoint_epoch'] == result['best_epoch']
    assert assess(cfg, test=True)['checkpoint_epoch'] == result['best_epoch']
    export(cfg)
    _, head, _, _, _ = load_selected(cfg)
    actual = head.mixing_alpha().item() if learned else head.mixing_alpha()
    assert actual == result['best_alpha']
