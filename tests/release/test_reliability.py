"""Crash recovery, batch-invariant validation, and shared cache publication."""
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from spatial_cbm.training import baselines, engine
from spatial_cbm.training.config import Config
from spatial_cbm.training.model import load_selected
from test_training import inputs, config


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize('boundary', ['epoch.pending.pt', 'best.pt', 'last.pt',
                                      'history.json', 'unlink', 'complete.json'])
@pytest.mark.parametrize('when', ['before', 'after'])
def test_epoch_publication_recovers_at_every_boundary(tmp_path, monkeypatch, boundary, when):
    prepared, _ = inputs(tmp_path)
    cfg = config(tmp_path, prepared)
    root = Path(cfg.training.output)
    original_torch, original_json, original_unlink = engine.atomic_torch, engine.save_json, Path.unlink
    fired = False

    def wrapped(call, path, *args, **kwargs):
        nonlocal fired
        name = 'unlink' if call is original_unlink and path.name == 'epoch.pending.pt' else path.name
        active = name == boundary and not fired
        if active:
            fired = True
            if when == 'before':
                raise RuntimeError('simulated interruption')
        result = call(path, *args, **kwargs)
        if active:
            raise RuntimeError('simulated interruption')
        return result

    monkeypatch.setattr(engine, 'atomic_torch', lambda p, v: wrapped(original_torch, p, v))
    monkeypatch.setattr(engine, 'save_json', lambda p, v: wrapped(original_json, p, v))
    monkeypatch.setattr(Path, 'unlink', lambda p, *a, **kw: wrapped(original_unlink, p, *a, **kw))
    with pytest.raises(RuntimeError, match='simulated'):
        engine.fit(cfg)
    assert fired
    result = engine.fit(cfg, resume=True)
    assert result['epochs'] == 2
    history = json.loads((root/'history.json').read_text())
    last = torch.load(root/'last.pt', weights_only=False)
    assert history == last['history']
    assert not (root/'epoch.pending.pt').exists()
    load_selected(cfg)


def test_final_epoch_stale_legacy_history_is_repaired(tmp_path):
    prepared, _ = inputs(tmp_path)
    cfg = config(tmp_path, prepared)
    expected = engine.fit(cfg)
    root = Path(cfg.training.output)
    history = json.loads((root/'history.json').read_text())
    (root/'history.json').write_text(json.dumps(history[:1]))
    assert engine.fit(cfg, resume=True) == expected
    assert json.loads((root/'history.json').read_text()) == history
    load_selected(cfg)


def test_nonimproving_epoch_keeps_previous_best(tmp_path):
    first = dict(epoch=1, val_loss=.2, model={'weight': torch.ones(1)})
    engine.publish_checkpoint(tmp_path, dict(epoch=1, history=[dict(epoch=1, val_loss=.2)]), first)
    last = dict(epoch=2, history=[dict(epoch=1, val_loss=.2), dict(epoch=2, val_loss=.3)])
    engine.atomic_torch(tmp_path/'epoch.pending.pt', dict(last=last, best=None))
    assert engine.recover_checkpoint(tmp_path) == last
    assert torch.load(tmp_path/'best.pt', weights_only=False)['epoch'] == 1


class IdentityHead(nn.Module):
    def forward(self, images):
        return dict(final_logits=images.mean((2, 3)), spatial_maps=images)


@pytest.mark.parametrize('positive_weight', [1., 100.])
@pytest.mark.parametrize('empty', [False, 'no_slots', 'no_valid'])
def test_validation_loss_is_independent_of_batch_partition(positive_weight, empty):
    cfg = Config()
    cfg.training.device = 'cpu'
    cfg.training.channels_last = False
    cfg.training.positive_weight = positive_weight
    torch.manual_seed(6)
    images = torch.randn(5, 2, 3, 3)
    labels = torch.randint(0, 2, (5, 2)).float()
    ids = torch.tensor([[0, 1]]*5)
    targets = torch.rand(5, 2, 3, 3)
    targets[2, 1] = 0  # Empty valid targets still occupy a denominator slot.
    valid = torch.tensor([[True, False], [False, False], [True, True],
                          [False, False], [False, True]])
    if empty == 'no_slots':
        ids, targets, valid = ids[:, :0], targets[:, :0], valid[:, :0]
    elif empty == 'no_valid':
        valid.zero_()
    dataset = TensorDataset(images, labels, ids, targets, valid, torch.zeros(5).long())
    head = IdentityHead()
    expected = engine.objective(head(images), [images, labels, ids, targets, valid], cfg.training).item()
    for size in (1, 2, 3, 5):
        cfg.training.batch_size = size
        result = engine.run_epoch(nn.Identity(), head, DataLoader(dataset, batch_size=size), cfg)
        assert result['loss'] == pytest.approx(expected, rel=1e-6, abs=1e-7)
        assert result['loss_denominators']['concept_bce'] == 10
        assert result['loss_denominators']['spatial_bce'] == (0 if empty else 3)


def cache_config(cache, seed=1):
    cfg = Config()
    cfg.data.cache = str(cache)
    cfg.model.method = 'salf_cbm'
    cfg.training.cache_regression_features = True
    cfg.training.seed = seed
    cfg.training.output = str(cache/f'run-{seed}')
    return cfg


CACHE_PROTOCOL = dict(data={'fixture': True}, source={'fixture': 'v1'}, backbone_sha256='fixture')


def fake_cache_build(cfg, backbone, protocol, paths):
    for pair in paths.values():
        np.save(pair[0], np.ones((2, 3), dtype=np.float32))
        np.save(pair[1], np.zeros(2, dtype=np.int64))


def cache_worker(cache, seed, ready, release, entered, done, queue):
    def build(*args):
        with (cache/'builders.txt').open('a') as stream:
            stream.write(f'{seed}\n')
        ready.set()
        release.wait()
        fake_cache_build(*args)
    baselines._build_regression_features = build
    try:
        entered.set()
        queue.put(baselines.prepare_regression_features(cache_config(cache, seed), None, CACHE_PROTOCOL))
    except Exception as exc:
        queue.put(repr(exc))
    finally:
        done.set()


def test_concurrent_cache_builders_wait_and_share_one_publication(tmp_path):
    context = mp.get_context('spawn')
    ready, release = context.Event(), context.Event()
    entered1, entered2 = context.Event(), context.Event()
    done1, done2, queue = context.Event(), context.Event(), context.Queue()
    first = context.Process(target=cache_worker, args=(tmp_path, 1, ready, release, entered1, done1, queue))
    second = context.Process(target=cache_worker, args=(tmp_path, 2, ready, release, entered2, done2, queue))
    first.start()
    try:
        assert ready.wait(30)
        second.start()
        assert entered2.wait(30)
        assert not done2.wait(2)
        assert not list((tmp_path/'regression-features').glob('*/complete.json'))
        release.set()
        first.join(30)
        second.join(30)
        assert first.exitcode == second.exitcode == 0
        paths1, paths2 = queue.get(timeout=5), queue.get(timeout=5)
        assert isinstance(paths1, dict) and paths1 == paths2
        assert (tmp_path/'builders.txt').read_text().splitlines() == ['1']
    finally:
        release.set()
        for process in (first, second):
            if process.pid is not None:
                if process.is_alive(): process.terminate()
                process.join()


def test_killed_builder_releases_lock_and_does_not_block_retry(tmp_path, monkeypatch):
    context = mp.get_context('spawn')
    ready, release, entered, done = (context.Event() for _ in range(4))
    queue = context.Queue()
    process = context.Process(target=cache_worker,
        args=(tmp_path, 1, ready, release, entered, done, queue))
    process.start()
    try:
        assert ready.wait(30)
        process.kill()
        process.join(30)
        assert process.exitcode != 0
        monkeypatch.setattr(baselines, '_build_regression_features', fake_cache_build)
        paths = baselines.prepare_regression_features(cache_config(tmp_path), None, CACHE_PROTOCOL)
        assert Path(paths['train'][0]).is_file()
    finally:
        if process.is_alive(): process.terminate()
        process.join()
        queue.close()
        queue.join_thread()


def test_failed_cache_builder_does_not_publish_partial_files(tmp_path, monkeypatch):
    cfg = cache_config(tmp_path)
    def fail(cfg, backbone, protocol, paths):
        np.save(paths['train'][0], np.ones(1))
        raise RuntimeError('interrupted cache build')
    monkeypatch.setattr(baselines, '_build_regression_features', fail)
    with pytest.raises(RuntimeError, match='interrupted'):
        baselines.prepare_regression_features(cfg, None, CACHE_PROTOCOL)
    assert not list((tmp_path/'regression-features').glob('*/complete.json'))
    monkeypatch.setattr(baselines, '_build_regression_features', fake_cache_build)
    paths = baselines.prepare_regression_features(cfg, None, CACHE_PROTOCOL)
    assert Path(paths['train'][0]).is_file()
    monkeypatch.setattr(baselines, '_build_regression_features', lambda *_: pytest.fail('Rebuilt completed cache'))
    assert baselines.prepare_regression_features(cfg, None, CACHE_PROTOCOL) == paths
