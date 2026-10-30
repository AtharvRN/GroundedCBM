"""Offline baseline equations and data contracts, independent of training runs."""
import json
from pathlib import Path
import sqlite3

import numpy as np
from PIL import Image
import pytest
import torch
from torch.nn import functional as F

from spatial_cbm.training import baselines, teacher
from spatial_cbm.training.config import Config, Model
from spatial_cbm.training.model import ConceptHead


def reference_similarity(prediction, target):
    left = (prediction - prediction.mean(0, keepdim=True)).pow(3)
    right = (target - target.mean(0, keepdim=True)).pow(3)
    return (left * right).sum(0) / (left.norm(dim=0) * right.norm(dim=0))


@pytest.mark.parametrize('shape', [(7, 3), (7, 3, 2, 4)])
def test_cubic_loss_matches_image_axis_equation_and_gradients(shape):
    generator = torch.Generator().manual_seed(17)
    prediction = torch.randn(shape, generator=generator, dtype=torch.float64, requires_grad=True)
    target = torch.randn(shape, generator=generator, dtype=torch.float64)
    actual = baselines.cubic_similarity(prediction, target)
    expected = reference_similarity(prediction, target)
    assert actual.shape == shape[1:]
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    actual_gradient, = torch.autograd.grad(-actual.mean(), prediction, retain_graph=True)
    expected_gradient, = torch.autograd.grad(-expected.mean(), prediction)
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-10, atol=1e-12)
    assert torch.isfinite(actual_gradient).all() and actual_gradient.norm() > 0

    # One concept/cell must not influence any other concept/cell's similarity.
    changed = prediction.detach().clone()
    changed[:, 0] = changed[:, 0].flip(0)
    torch.testing.assert_close(baselines.cubic_similarity(changed, target)[1:], actual[1:])
    if len(shape) == 4:
        changed = prediction.detach().clone()
        changed[:, 0, 0, 0] = changed[:, 0, 0, 0].flip(0)
        result = baselines.cubic_similarity(changed, target)
        torch.testing.assert_close(result[0, 0, 1:], actual[0, 0, 1:])
        torch.testing.assert_close(result[0, 1], actual[0, 1])


def test_cubic_is_not_cubed_cosine_and_is_shift_scale_invariant():
    prediction = torch.tensor([[0., 2.], [1., -1.], [2., 4.], [8., 3.]], dtype=torch.float64)
    target = torch.tensor([[1., 4.], [4., 3.], [2., -2.], [5., 0.]], dtype=torch.float64)
    actual = baselines.cubic_similarity(prediction, target)
    wrong = F.cosine_similarity(prediction - prediction.mean(0), target - target.mean(0), dim=0).pow(3)
    assert not torch.allclose(actual, wrong, atol=1e-3)
    torch.testing.assert_close(
        baselines.cubic_similarity(prediction * 11 + 19, target * .25 - 7), actual)
    torch.testing.assert_close(baselines.cubic_similarity(-prediction, target), -actual)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
def test_cubic_low_precision_and_zero_variance_are_finite(dtype):
    prediction = torch.tensor([[1., 1.], [2., 1.], [4., 1.], [-3., 1.]], dtype=dtype,
                              requires_grad=True)
    target = torch.tensor([[3., 5.], [-1., 5.], [2., 5.], [4., 5.]], dtype=dtype)
    result = baselines.cubic_similarity(prediction, target)
    assert result.dtype == torch.float32
    torch.testing.assert_close(result[:1], reference_similarity(prediction.float(), target.float())[:1])
    assert result[1] == 0 and torch.isfinite(result).all()
    (-result.mean()).backward()
    assert torch.isfinite(prediction.grad).all()
    torch.testing.assert_close(prediction.grad[:, 1], torch.zeros(4, dtype=dtype))


@pytest.mark.parametrize('left,right', [((1, 3), (1, 3)), ((4, 3), (4, 2)),
                                      ((4, 3, 2), (4, 3, 2))])
def test_cubic_rejects_singletons_mismatched_shapes_and_wrong_rank(left, right):
    with pytest.raises(ValueError, match='Centered-cubic'):
        baselines.cubic_similarity(torch.ones(left), torch.ones(right))


@pytest.mark.parametrize('method', ['lf_cbm', 'vlg_cbm'])
def test_linear_heads_preserve_raw_scores_cam_and_filtered_order(method):
    head = ConceptHead(Model(method=method), (1, 1, 2), 4, torch.tensor([3, 1]))
    with torch.no_grad():
        head.projection.weight.copy_(torch.tensor([[1., -2.], [-3., .5]]))
        if head.projection.bias is not None:
            head.projection.bias.copy_(torch.tensor([-4., 2.]))
    features = torch.arange(12, dtype=torch.float32).reshape(1, 2, 2, 3)
    output = head({'conv5': features})
    expected_maps = torch.einsum('oc,nchw->nohw', head.projection.weight, features)
    if method == 'vlg_cbm':
        expected_maps = expected_maps + head.projection.bias[None, :, None, None]
    else:
        assert head.projection.bias is None
    torch.testing.assert_close(output['spatial_maps'], expected_maps)
    torch.testing.assert_close(output['final_logits'], expected_maps.mean((2, 3)))
    assert output['final_logits'].shape == (1, 2)
    assert (output['final_logits'] < 0).any()
    assert output['map_kind'] == 'linear_cam'
    assert head.state_dict()['concept_indices'].tolist() == [3, 1]
    restored = ConceptHead(Model(method=method), (1, 1, 2), 4, [3, 1])
    restored.load_state_dict(head.state_dict())
    torch.testing.assert_close(restored({'conv5': features})['final_logits'], output['final_logits'])


def test_nonlinear_vlg_is_gap_then_mlp_with_no_spatial_maps():
    head = baselines.BaselineHead(Model(method='vlg_cbm', vlg_hidden=2), (1, 1, 2), 2)
    with torch.no_grad():
        head.projection.weight.copy_(torch.eye(2))
        head.projection.bias.copy_(torch.tensor([-2., 1.]))
        for layer in head.hidden:
            if isinstance(layer, torch.nn.Linear):
                layer.weight.copy_(-torch.eye(2))
                layer.bias.fill_(-3.)
    features = torch.tensor([[[[-8., 10.]], [[-3., 1.]]]])
    expected = F.linear(features.mean((2, 3)), head.projection.weight, head.projection.bias)
    for layer in head.hidden:
        expected = layer(expected)
    output = head({'conv5': features})
    torch.testing.assert_close(output['final_logits'], expected)
    assert (output['final_logits'] < 0).all()
    assert output['spatial_maps'] is None and output['map_kind'] == 'unavailable'
    assert [type(layer) for layer in head.hidden] == [torch.nn.ReLU, torch.nn.Linear] * 2


def test_salf_resizes_then_projects_and_softmax_pools_without_squeezing():
    head = ConceptHead(Model(method='salf_cbm', salf_grid=3), (1, 1, 2), 1)
    with torch.no_grad():
        head.projection.weight.copy_(torch.tensor([[[[.25]], [[-1.]]]]))
    features = torch.arange(12, dtype=torch.float32).reshape(1, 2, 2, 3).requires_grad_()
    output = head({'conv5': features})
    resized = F.interpolate(features, (3, 3), mode='bilinear', align_corners=False)
    maps = .25 * resized[:, :1] - resized[:, 1:]
    flat = maps.flatten(2)
    expected = (flat.softmax(-1) * flat).sum(-1)
    torch.testing.assert_close(output['spatial_maps'], maps)
    torch.testing.assert_close(output['final_logits'], expected)
    assert output['final_logits'].shape == (1, 1)
    assert not torch.allclose(expected, maps.mean((2, 3)))
    assert head.projection.bias is None and output['map_kind'] == 'native'
    output['final_logits'].sum().backward()
    assert torch.isfinite(features.grad).all() and features.grad.norm() > 0


@pytest.mark.parametrize('indices', [[], [0, 0], [-1], [3], [[0, 1]]])
def test_heads_reject_invalid_retained_indices(indices):
    with pytest.raises(ValueError, match='retained concept indices'):
        baselines.BaselineHead(Model(method='lf_cbm'), (1, 1, 2), 3, indices)


@pytest.mark.parametrize('text,expected', [('A Red-wing.', 'red wing'), ('An (orange), beak', 'orange beak'),
                                         ('blue   FOOT', 'blue foot'), ('aqua wing', 'aqua wing')])
def test_vlg_concept_canonicalization(text, expected):
    assert baselines.concept_key(text) == expected


def test_vlg_lookup_rejects_canonical_collisions():
    assert baselines.concept_lookup(['A Red-wing.', 'blue beak']) == {'red wing': 0, 'blue beak': 1}
    with pytest.raises(ValueError, match='duplicate normalized'):
        baselines.concept_lookup(['red wing', 'a red-wing'])


def annotation(concept, box, confidence=1., coordinates='pixel_xyxy'):
    return dict(concept=concept, box=box, confidence=confidence, coordinates=coordinates)


def force_first_crop(monkeypatch):
    monkeypatch.setattr(np.random, 'random', lambda: 0.)
    monkeypatch.setattr(np.random, 'randint', lambda n: 0)
    monkeypatch.setattr(np.random, 'choice', lambda values: values[0])


def test_crop_iou_is_strict_and_checks_all_boxes_with_canonical_labels(monkeypatch):
    force_first_crop(monkeypatch)
    image = Image.new('RGB', (30, 20))
    names = ['red wing', 'half', 'overlap', 'far', 'at cutoff', 'no box']
    items = [annotation('A Red-wing.', [0, 0, 10, 10]), annotation('half', [0, 0, 5, 10]),
             annotation('overlap', [20, 0, 30, 10]), annotation('overlap', [0, 0, 6, 10]),
             annotation('far', [20, 0, 30, 10]), annotation('at cutoff', [0, 0, 10, 10], .5),
             annotation('no box', None), annotation('unknown', [0, 0, 10, 10])]
    cropped, labels = baselines.vlg_crop(image, items, baselines.concept_lookup(names), .5, 1.)
    assert cropped.size == (10, 10)
    assert labels.tolist() == [1., 0., 1., 0., 0., 0.]


def test_crop_selection_is_over_concepts_not_instances(monkeypatch):
    force_first_crop(monkeypatch)
    choices, box_counts = [], []
    monkeypatch.setattr(np.random, 'randint', lambda n: box_counts.append(n) or n - 1)
    monkeypatch.setattr(np.random, 'choice', lambda values: choices.append(list(values)) or values[-1])
    items = [annotation('wing', [0, 0, 10, 10])] * 5 + [annotation('beak', [20, 0, 25, 5])]
    cropped, labels = baselines.vlg_crop(Image.new('RGB', (30, 20)), items,
                                       {'wing': 0, 'beak': 1}, .15, 1.)
    assert choices == [[0, 1]] and box_counts == [5, 1]
    assert cropped.size == (5, 5) and labels.tolist() == [0., 1.]


def test_crop_normalized_xyxy_uses_width_and_height(monkeypatch):
    force_first_crop(monkeypatch)
    image = Image.fromarray(np.arange(20 * 40 * 3, dtype=np.uint8).reshape(20, 40, 3))
    cropped, labels = baselines.vlg_crop(image,
        [annotation('wing', [.25, .25, .75, .75], coordinates='normalized_xyxy')], {'wing': 0}, .15, 1.)
    np.testing.assert_array_equal(np.asarray(cropped), np.asarray(image)[5:15, 10:30])
    assert labels.tolist() == [1.]


def test_crop_probability_boundary_preserves_full_image_presence(monkeypatch):
    monkeypatch.setattr(np.random, 'random', lambda: .5)
    image = Image.new('RGB', (20, 20))
    items = [annotation('A Red-wing', [0, 0, 10, 10], .51), annotation('blue beak', None, .9),
             annotation('cutoff', [0, 0, 10, 10], .5)]
    cropped, labels = baselines.vlg_crop(image, items, {'red wing': 0, 'blue beak': 1, 'cutoff': 2}, .5, .5)
    assert cropped is image and labels.tolist() == [1., 1., 0.]


@pytest.mark.parametrize('items,expected', [([], [0., 0.]), ([annotation('beak', None)], [0., 1.])])
def test_crop_without_eligible_boxes_preserves_image_and_labels(monkeypatch, items, expected):
    force_first_crop(monkeypatch)
    image = Image.new('RGB', (100, 100))
    cropped, labels = baselines.vlg_crop(image, items, {'wing': 0, 'beak': 1}, .15, 1.)
    assert cropped is image and labels.tolist() == expected


def test_zero_size_crop_falls_back_before_overwriting_full_image_labels(monkeypatch):
    force_first_crop(monkeypatch)
    image = Image.new('RGB', (100, 100))
    items = [annotation('wing', [.001, .001, .002, .002], coordinates='normalized_xyxy'),
             annotation('beak', None)]
    cropped, labels = baselines.vlg_crop(image, items, {'wing': 0, 'beak': 1}, .15, 1.)
    assert cropped is image and labels.tolist() == [1., 1.]


@pytest.mark.parametrize('batch_size', [3, 4, 8])
def test_training_batches_merge_small_tails_without_losing_or_reordering_samples(batch_size):
    for rows in range(3, 26):
        sampler = baselines.NonSingletonBatches(range(rows), batch_size, False)
        batches = list(sampler)
        assert len(batches) == len(sampler)
        assert [i for batch in batches for i in batch] == list(range(rows))
        assert all(3 <= len(batch) <= batch_size + 2 for batch in batches)
        assert all(len(batch) == batch_size for batch in batches[:-1])


@pytest.mark.parametrize('rows,expected', [(8, [4, 4]), (9, [4, 5]), (10, [4, 6]), (11, [4, 4, 3])])
def test_training_batch_tail_boundaries(rows, expected):
    assert [len(batch) for batch in baselines.NonSingletonBatches(range(rows), 4, False)] == expected


@pytest.mark.parametrize('shape', [(9, 3), (9, 3, 2, 4)])
@pytest.mark.parametrize('sizes', [[4, 4, 1], [1] * 9, [9]])
def test_regression_meter_centers_whole_cohort_independent_of_batches(tmp_path, shape, sizes):
    generator = torch.Generator().manual_seed(93)
    prediction = torch.randn(shape, generator=generator)
    target = torch.randn(shape, generator=generator)
    expected = reference_similarity(prediction.double(), target.double())
    meter = baselines.RegressionMeter(shape[0], tmp_path)
    directory = Path(meter.directory.name)
    try:
        offset = 0
        for size in sizes:
            meter.update(prediction[offset:offset + size], target[offset:offset + size])
            offset += size
        torch.testing.assert_close(meter.similarities(), expected, rtol=1e-12, atol=1e-12)
        local_mean = (reference_similarity(prediction[:4].double(), target[:4].double()) +
                      reference_similarity(prediction[4:].double(), target[4:].double())) / 2
        assert not torch.allclose(expected, local_mean, atol=1e-3)
    finally:
        meter.close()
    assert not directory.exists()


def test_regression_meter_rejects_incomplete_cohort(tmp_path):
    meter = baselines.RegressionMeter(3, tmp_path)
    try:
        meter.update(torch.ones(2, 1), torch.ones(2, 1))
        with pytest.raises(ValueError, match='Incomplete'):
            meter.similarities()
    finally:
        meter.close()
    with pytest.raises(ValueError, match='at least two'):
        baselines.RegressionMeter(1, tmp_path)


@pytest.mark.parametrize('bad_target', [False, True])
def test_regression_meter_rejects_nonfinite_scores(tmp_path, bad_target):
    meter = baselines.RegressionMeter(3, tmp_path)
    values = [torch.ones(3, 2), torch.ones(3, 2)]
    values[int(bad_target)][0, 0] = float('nan')
    try:
        with pytest.raises(ValueError, match='Nonfinite'):
            meter.update(*values)
    finally:
        meter.close()


def filter_inputs(tmp_path, method, bank, train, validation):
    (tmp_path / 'concepts.txt').write_text('\n'.join(bank) + '\n')
    np.savez(tmp_path / 'indices.npz', train=train, validation=validation)
    cfg = Config()
    cfg.model.method = method
    cfg.data.prepared = str(tmp_path)
    cfg.data.confidence = .5
    cfg.training.lf_clip_cutoff = .25
    return cfg


def test_vlg_filter_uses_only_training_positives_and_strict_canonical_matching(tmp_path):
    bank = ['red wing', 'validation only', 'at cutoff', 'blue beak']
    cfg = filter_inputs(tmp_path, 'vlg_cbm', bank, [2, 0], [1, 3])
    rows = [(0, [annotation('A Red-wing.', None, .9), annotation('at cutoff', None, .5)]),
            (1, [annotation('validation only', None), annotation('at cutoff', None)]),
            (2, [annotation('a blue-beak', None, .75), annotation('not in bank', None)]), (3, None)]
    with sqlite3.connect(tmp_path / 'samples.sqlite') as db:
        db.execute('CREATE TABLE samples (position INTEGER, detections TEXT)')
        db.executemany('INSERT INTO samples VALUES (?, ?)', [(i, json.dumps(items)) for i, items in rows])
    assert baselines.initial_indices(cfg, {}).tolist() == [0, 3]
    with sqlite3.connect(tmp_path / 'samples.sqlite') as db:
        db.execute('UPDATE samples SET detections = ? WHERE position = 2', ('null',))
    with pytest.raises(ValueError, match='annotations for every training image'):
        baselines.initial_indices(cfg, {})


def test_lf_top_five_filter_is_train_only_chunked_strict_and_keeps_literal_names(tmp_path, monkeypatch):
    # The first two literal LF strings collide under VLG normalization intentionally.
    bank = ['a red-wing', 'red wing', 'at cutoff', 'validation only', 'one spike']
    train = np.arange(260)[::-1].copy()
    cfg = filter_inputs(tmp_path, 'lf_cbm', bank, train, [260, 261])
    targets = torch.zeros(262, len(bank))
    targets[[0, 1, 2, 258, 259], 0] = .5
    targets[:260, 1] = .5
    targets[:260, 2] = .25
    targets[260:, 3] = 1.
    targets[0, 4] = 1.
    seen, proof, protocol = [], object(), {'images': 262}

    class FakeTargets:
        def __init__(self, actual_cfg, actual_protocol, verified_manifest=None):
            assert actual_cfg is cfg and actual_protocol is protocol and verified_manifest is proof

        def get(self, position):
            assert position in train
            seen.append(position)
            return targets[position]

    monkeypatch.setattr(teacher, 'TeacherTargets', FakeTargets)
    assert baselines.initial_indices(cfg, protocol, teacher_manifest=proof).tolist() == [0, 1]
    assert seen == train.tolist()


def test_lf_filter_averages_available_images_when_fewer_than_five(tmp_path, monkeypatch):
    cfg = filter_inputs(tmp_path, 'lf_cbm', ['keep', 'equal'], [4, 2, 7], [0])
    targets = {4: torch.tensor([.5, .25]), 2: torch.tensor([.5, .25]), 7: torch.tensor([.5, .25])}

    class FakeTargets:
        def __init__(self, *args, **kwargs):
            pass

        def get(self, position):
            return targets[position]

    monkeypatch.setattr(teacher, 'TeacherTargets', FakeTargets)
    assert baselines.initial_indices(cfg, {}).tolist() == [0]
    cfg.training.lf_clip_cutoff = .5
    with pytest.raises(ValueError, match='No concepts survive'):
        baselines.initial_indices(cfg, {})


@pytest.mark.parametrize('method', ['sg_cbm', 'salf_cbm'])
def test_sg_and_salf_do_not_inherit_lf_or_vlg_filtering(tmp_path, method):
    (tmp_path / 'concepts.txt').write_text('a red-wing\nred wing\n')
    cfg = Config(model=Model(method=method))
    cfg.data.prepared = str(tmp_path)
    # No indices, annotations, or teacher exist: these methods need no initial filter.
    assert baselines.initial_indices(cfg, {}) is None


@pytest.mark.parametrize('method', ['lf_cbm', 'vlg_cbm'])
def test_fixed_vocabulary_bypasses_support_filter_without_reading_targets(tmp_path, method):
    (tmp_path/'concepts.txt').write_text('head\nrare concept\n')
    cfg = Config(model=Model(method=method))
    cfg.data.prepared = str(tmp_path)
    cfg.training.filter_concepts = False
    assert baselines.initial_indices(cfg, {}) is None
    # No checkpoint or validation loader is needed when post-filtering is disabled.
    baselines.finalize_lf(cfg, None, {})


def test_filter_flag_requires_boolean():
    cfg = Config()
    cfg.training.filter_concepts = 'false'
    with pytest.raises(ValueError, match='must be a boolean'):
        cfg.validate()


def test_subset_targets_preserves_sparse_geometry_and_remaps_filtered_order():
    images = torch.zeros(2, 3, 4, 4)
    labels = torch.tensor([[1., 0., 1., 1.], [0., 1., 0., 0.]])
    ids = torch.tensor([[3, 0, 2], [1, -1, -1]])
    masks = torch.arange(24, dtype=torch.float32).reshape(2, 3, 2, 2)
    valid = torch.tensor([[True, True, True], [True, False, False]])
    full = torch.zeros(2, 3, 4, 4)
    classes = torch.tensor([0, 1])
    batch = [images, labels, ids, masks, valid, full, classes]
    result = baselines.subset_targets(batch, torch.tensor([2, 3]))
    torch.testing.assert_close(result[1], labels[:, [2, 3]])
    assert result[2][0].tolist() == [1, 0, 0]
    assert result[4].tolist() == [[True, False, True], [False, False, False]]
    assert result[3] is masks and result[5] is full and result[6] is classes
    assert batch[1] is labels and batch[2] is ids and batch[4] is valid


def test_regression_collate_without_annotations_keeps_teacher_and_classes_separate():
    samples = [(torch.full((3, 2, 2), float(i)), torch.tensor([i, i + 1.]), 4 - i) for i in range(3)]
    images, targets, classes = baselines.regression_collate(samples)
    assert images.shape == (3, 3, 2, 2)
    torch.testing.assert_close(targets, torch.tensor([[0., 1.], [1., 2.], [2., 3.]]))
    assert classes.tolist() == [4, 3, 2]
    assert baselines.subset_targets([images, classes], torch.tensor([1]))[1] is classes
