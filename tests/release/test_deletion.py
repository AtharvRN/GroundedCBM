import json
from pathlib import Path

import pytest
import torch

from evals import deletion
from spatial_cbm.training.data import digest, save_json


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_cub70_compatibility_entry_point():
    from cub70.__main__ import evaluate, select_maps
    from evals import cub70
    assert evaluate is cub70.evaluate
    assert select_maps is cub70.select_maps


def test_selection_requires_positive_activation_and_contribution():
    z = torch.tensor([[-4., 2., 3., 0.], [-1., -1., -1., -1.]])
    weight = torch.tensor([[-10., 2., 1., 100.], [1., 1., 1., 1.]])
    columns, valid = deletion.select_concepts(z, weight, torch.tensor([0, 1]))
    assert columns[0] == 1 and valid.tolist() == [True, False]
    tied, _ = deletion.select_concepts(torch.ones(1, 4), torch.ones(1, 4), torch.tensor([0]))
    assert tied.item() == 0


def test_highest_mean_grid_patch_and_mean_fill():
    maps = torch.zeros(2, 4, 4)
    maps[0, :2, :2] = 3
    maps[0, 3, 3] = 8  # Highest pixel is not the highest-mean window.
    maps[1, 2:, :2] = 4
    patches = deletion.target_patches(maps, 4, 2)
    assert patches.tolist() == [0, 2]
    original = torch.ones(2, 3, 4, 4)
    result = deletion.erase(original, patches, 2)
    assert original.sum() == 96
    assert (result == 0).sum().item() == 24
    assert not result[0, :, :2, :2].any()
    assert not result[1, :, 2:, :2].any()
    with pytest.raises(ValueError, match='divisible'):
        deletion.target_patches(maps, 5, 2)


def test_random_controls_are_paired_and_allow_repeats_and_target_overlap():
    first = deletion.random_patches('image-1', 42, 4, 100)
    assert first == deletion.random_patches('image-1', 42, 4, 100)
    assert first != deletion.random_patches('image-2', 42, 4, 100)
    assert set(first) == {0, 1, 2, 3}
    assert deletion.random_patches('only-window', 42, 1, 3) == [0, 0, 0]


def toy_predict(images):
    values = images.mean((1, 2, 3))
    z = torch.stack((values+2, -values-1), 1)
    maps = images[:, :1].expand(-1, 2, -1, -1)
    return z, z, maps


def test_serial_batch_parity_and_original_inputs_preserved():
    images = torch.arange(96).reshape(2, 3, 4, 4).float()/96
    original = images.clone()
    batched = deletion.deletion_batch(images, ['a', 'b'], toy_predict, torch.eye(2), patch_size=2)
    serial = [deletion.deletion_batch(images[i:i+1], [name], toy_predict, torch.eye(2), patch_size=2)[0]
              for i, name in enumerate(['a', 'b'])]
    assert batched == serial
    assert torch.equal(images, original)
    for row in batched:
        assert row['eligible'] and row['concept_index'] == 0
        assert row['random_class_drop'] == pytest.approx(sum(row['random_class_drops'])/3)
        assert row['targeted_concept_drop'] > 0


def test_keep_clean_class_and_concept_when_prediction_flips():
    def predict(images):
        value = images.mean((1, 2, 3))
        z = torch.stack((value, 1-value), 1)
        return z, z, images[:, :1].expand(-1, 2, -1, -1)
    row = deletion.deletion_batch(torch.ones(1, 3, 4, 4), ['a'], predict,
                                  torch.eye(2), patch_size=4)[0]
    expected = 100*(torch.tensor([1., 0.]).softmax(0)[0]-torch.tensor([0., 1.]).softmax(0)[0])
    assert row['predicted_class'] == 0 and row['concept_index'] == 0
    assert row['targeted_class_drop'] == pytest.approx(float(expected))
    assert row['targeted_concept_drop'] == 1
    assert row['random_concept_drop'] == 1


def test_signed_drops_and_empty_eligibility():
    def predict(images):
        value = images.mean((1, 2, 3))
        z = torch.stack((2-value, torch.zeros_like(value)), 1)
        return z, z, images[:, :1].expand(-1, 2, -1, -1)
    row = deletion.deletion_batch(torch.ones(1, 3, 4, 4), ['a'], predict,
                                  torch.eye(2), patch_size=2)[0]
    assert row['targeted_class_drop'] < 0 and row['targeted_concept_drop'] < 0
    empty = deletion.deletion_batch(torch.ones(1, 3, 4, 4), ['a'], predict,
                                    -torch.eye(2), patch_size=2)
    assert not empty[0]['eligible']
    summary = deletion.summarize(empty)
    assert summary['excluded'] == 1 and summary['class_gap'] is None


@pytest.mark.parametrize('bad', ['missing', 'nonfinite', 'dimensions'])
def test_bad_maps_fail(bad):
    def predict(images):
        z = torch.ones(len(images), 2)
        maps = torch.ones(len(images), 2, 2, 2)
        if bad == 'missing': maps = None
        if bad == 'nonfinite': maps[0, 0, 0, 0] = torch.nan
        if bad == 'dimensions': maps = maps[:, :1]
        return z, z, maps
    with pytest.raises(ValueError):
        deletion.deletion_batch(torch.ones(1, 3, 4, 4), ['a'], predict, torch.eye(2), patch_size=2)


@pytest.fixture
def fitted(tmp_path):
    from test_training import inputs, config
    from spatial_cbm.training.engine import fit
    from spatial_cbm.training.export import export
    prepared, _ = inputs(tmp_path)
    cfg = config(tmp_path, prepared)
    fit(cfg); export(cfg)
    root = Path(cfg.training.output)
    heads = root/'nec'; heads.mkdir()
    files = {}
    for name, value in [('W_g@NEC=1.pt', torch.eye(2)), ('b_g@NEC=1.pt', torch.zeros(2))]:
        torch.save(value, heads/name)
        files[name] = digest(heads/name)
    save_json(heads/'classifier_identity.json', dict(checkpoint_sha256=digest(root/'best.pt'),
        normalization_sha256=digest(root/'final_layer_normalization.pt'),
        feature_manifest_sha256=digest(root/'features_complete.json'), head_files=files))
    return cfg, heads


def test_release_checkpoint_end_to_end_and_batch_parity(fitted, tmp_path):
    cfg, heads = fitted
    prior = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    result = deletion.evaluate(cfg, heads, tmp_path/'batched', nec=1, batch_size=2, patch_size=16)
    serial = deletion.evaluate(cfg, heads, tmp_path/'serial', nec=1, batch_size=1, patch_size=16)
    assert result['primary']['images'] == 2 and serial['primary']['images'] == 2
    assert (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32) == prior
    a = [json.loads(s) for s in (tmp_path/'batched/rows.jsonl').read_text().splitlines()]
    b = [json.loads(s) for s in (tmp_path/'serial/rows.jsonl').read_text().splitlines()]
    for left, right in zip(a, b):
        assert left.keys() == right.keys()
        for key in left:
            if isinstance(left[key], (float, list)):
                assert left[key] == pytest.approx(right[key], abs=2e-4)
            else:
                assert left[key] == right[key]
    protocol = json.loads((tmp_path/'batched/protocol.json').read_text())
    assert protocol['image_ids'] == ['test-0', 'test-1']
    assert protocol['map_kind'] == 'native'
    assert json.loads((tmp_path/'batched/complete.json').read_text())['results_sha256'] == digest(tmp_path/'batched/results.json')
    with pytest.raises(FileExistsError):
        deletion.evaluate(cfg, heads, tmp_path/'batched', nec=1)
    roster = tmp_path/'roster.json'; roster.write_text('["test-1"]')
    subset = deletion.evaluate(cfg, heads, tmp_path/'subset', nec=1, image_ids=roster)
    assert subset['primary']['images'] == 1
    roster.write_text('["0"]')
    with pytest.raises(ValueError, match='non-held-out'):
        deletion.evaluate(cfg, heads, tmp_path/'invalid', nec=1, image_ids=roster)
    assert not (tmp_path/'invalid').exists()


@pytest.mark.parametrize('field', ['checkpoint_sha256', 'feature_manifest_sha256', 'normalization_sha256', 'head_files'])
def test_classifier_provenance_guard(fitted, tmp_path, field):
    cfg, heads = fitted
    metadata = heads/'classifier_identity.json'
    value = json.loads(metadata.read_text())
    if field == 'head_files': value[field]['W_g@NEC=1.pt'] = 'changed'
    else: value[field] = 'changed'
    save_json(metadata, value)
    with pytest.raises(ValueError):
        deletion.evaluate(cfg, heads, tmp_path/'invalid', nec=1)
    assert not (tmp_path/'invalid').exists()


def test_interrupted_evaluation_has_no_completion_marker(fitted, tmp_path, monkeypatch):
    cfg, heads = fitted
    prior = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    def fail(*args, **kwargs): raise RuntimeError('interrupted')
    monkeypatch.setattr(deletion, 'deletion_batch', fail)
    with pytest.raises(RuntimeError, match='interrupted'):
        deletion.evaluate(cfg, heads, tmp_path/'failed', nec=1)
    assert not (tmp_path/'failed/complete.json').exists()
    assert not (tmp_path/'failed/results.json').exists()
    assert (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32) == prior


def test_classifier_accepts_backbone_weight_preset_and_checks_exact_nec(fitted):
    cfg, heads = fitted
    cfg.model.checkpoint = ''
    cfg.model.weights = 'IMAGENET1K_V1'
    root = Path(cfg.training.output)
    identity = dict(checkpoint_sha256=digest(root/'best.pt'))
    values = deletion.classifier(cfg, heads, identity, 2, 2, 1)
    assert '.' not in values[-1]
    path = heads/'W_g@NEC=1.pt'
    torch.save(torch.ones(2, 2), path)
    metadata = heads/'classifier_identity.json'
    recorded = json.loads(metadata.read_text())
    recorded['head_files'][path.name] = digest(path)
    save_json(metadata, recorded)
    with pytest.raises(ValueError, match='Exact NEC'):
        deletion.classifier(cfg, heads, identity, 2, 2, 1)


@pytest.mark.parametrize('method', ['lf_cbm', 'vlg_cbm', 'salf_cbm'])
def test_release_baseline_heads_support_deletion(method):
    from types import SimpleNamespace
    from spatial_cbm.training.baselines import BaselineHead
    head = BaselineHead(SimpleNamespace(method=method, salf_grid=2, vlg_hidden=0), (3, 3, 3), 2)
    torch.nn.init.constant_(head.projection.weight, .5)
    if head.projection.bias is not None:
        torch.nn.init.zeros_(head.projection.bias)
    def predict(images):
        out = head(dict(conv5=images))
        return out['final_logits'], out['final_logits'], out['spatial_maps']
    with torch.inference_mode():
        rows = deletion.deletion_batch(torch.ones(1, 3, 4, 4), ['a'], predict,
                                       torch.eye(2), patch_size=2)
    assert rows[0]['eligible'] and rows[0]['targeted_concept_drop'] > 0
