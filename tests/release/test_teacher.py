"""Offline teacher contract tests; no CLIP installation or downloads needed."""
import json
import os
import pickle
import sqlite3
from copy import deepcopy
from types import SimpleNamespace as NS

import numpy as np
from PIL import Image, ImageDraw
import pytest
import torch
from torchvision import transforms

from spatial_cbm.training import data, teacher


def cpu_autocast_enabled():
    try:
        return torch.is_autocast_enabled('cpu')
    except TypeError:  # PyTorch 2.2 predates the device argument.
        return torch.is_autocast_cpu_enabled()


class _SpawnTargets(torch.utils.data.Dataset):
    def __init__(self, targets):
        self.targets = targets

    def __len__(self):
        return 3

    def __getitem__(self, index):
        mapped = all(isinstance(array, np.memmap) and array.mode == 'r'
                     and array.filename is not None and not array.flags.writeable
                     for array in (self.targets.text, self.targets.images, self.targets.positions))
        return self.targets.get(index), mapped, os.getpid()


@pytest.fixture
def setup(tmp_path, monkeypatch):
    rows = []
    # Non-square grayscale, alpha, and RGB inputs must all reach CLIP as RGB.
    for i, (mode, size) in enumerate([('L', (83, 311)), ('RGBA', (417, 95)), ('RGB', (161, 203))]):
        image = tmp_path / f'{i}.png'
        Image.new(mode, size, 70 if mode == 'L' else (20, 80, 140, 50) if mode == 'RGBA'
                  else (100, 60, 20)).save(image)
        rows.append(dict(id=f'image-{i}', image=str(image), class_id=0))
    source = tmp_path / 'source.jsonl'
    source.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    concepts = tmp_path / 'concepts.txt'
    concepts.write_text('red wing\nblue beak\n')
    split = tmp_path / 'split.json'
    split.write_text(json.dumps(dict(train=['image-2', 'image-0'], validation=['image-1'])))
    heldout = tmp_path / 'heldout.jsonl'
    # The held-out image deliberately does not exist: preparation must never open it.
    heldout.write_text(json.dumps(dict(id='test-only', image=str(tmp_path / 'missing.png'), class_id=0)) + '\n')
    root = tmp_path / 'prepared'
    protocol = data.prepare(source, root, 'cub', concepts, split=split, heldout_source=heldout)
    checkpoint = tmp_path / 'clip.pt'
    checkpoint.write_bytes(b'fake explicit local checkpoint')
    cfg = NS(data=NS(prepared=str(root), resize_size=256, input_size=224),
             model=NS(method='lf_cbm', salf_grid=7), training=NS(device='cpu'),
             teacher=NS(cache=str(tmp_path / 'teacher'), checkpoint=str(checkpoint),
                        model='ViT-B/16', batch_size=5, prompt_radius=32))
    calls = NS(load=[], text=[], rgb=[], images=[], setup=[])

    class Encoder(torch.nn.Module):
        visual = NS(input_resolution=224)

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.eye(3, dtype=torch.float16))

        def float(self):
            calls.setup.append('float')
            return super().float()

        def eval(self):
            calls.setup.append('eval')
            return super().eval()

        def project(self, values):
            assert not self.training and self.weight.dtype == torch.float32
            assert not cpu_autocast_enabled()
            assert not torch.backends.cuda.matmul.allow_tf32
            assert not torch.backends.cudnn.allow_tf32
            result = values @ self.weight
            assert result.dtype == torch.float32
            return result

        def encode_text(self, tokens):
            return self.project(tokens.float())

        def encode_image(self, pixels):
            assert pixels.dtype == torch.float32
            calls.images.append(pixels.clone())
            # Spatial moments distinguish row-major circle locations.
            weights = torch.linspace(0, 1, pixels.shape[-1])[None, None, None, :]
            return self.project(torch.stack([pixels.mean((1, 2, 3)), (pixels * weights).mean((1, 2, 3)),
                                             torch.ones(len(pixels))], dim=1))

    encoder = Encoder()

    def preprocess(image):
        calls.rgb.append((image.mode, image.size))
        return transforms.ToTensor()(image.resize((224, 224)))

    def load(path, device, jit):
        calls.load.append((path, device, jit))
        return encoder, preprocess

    def tokenize(words):
        calls.text.extend(words)
        return torch.tensor([[1., -2., 3.] if word == 'red wing' else [-2., 1., -3.] for word in words])

    original_import = teacher.importlib.import_module
    monkeypatch.setattr(teacher.importlib, 'import_module',
                        lambda name, *args, **kwargs: NS(load=load, tokenize=tokenize) if name == 'clip'
                        else original_import(name, *args, **kwargs))
    # Keep tiny fake-encoder tensor operations inexpensive on many-core hosts.
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield cfg, protocol, calls, encoder, rows
    torch.set_num_threads(threads)


@pytest.mark.parametrize('method,shape', [('lf_cbm', (2,)), ('salf_cbm', (2, 7, 7))])
def test_factorized_targets_rgb_cosines_and_exact_cohort(setup, method, shape):
    cfg, protocol, calls, _, rows = setup
    cfg.model.method = method
    manifest = teacher.prepare_teacher(cfg)
    cache = teacher.Path(cfg.teacher.cache)
    targets = teacher.TeacherTargets(cfg, protocol)
    assert targets.manifest == manifest
    assert calls.load == [(cfg.teacher.checkpoint, 'cpu', False)]
    assert calls.setup == ['float', 'eval']
    assert calls.text == ['red wing', 'blue beak']
    assert protocol['annotated_images'] == 0
    assert manifest['rows_written'] == protocol['train'] + protocol['validation'] == 3
    assert isinstance(targets.images, np.memmap) and isinstance(targets.text, np.memmap)
    assert targets.images.shape == (3, 1 if method == 'lf_cbm' else 49, 3)
    assert targets.images.dtype == targets.text.dtype == np.float32
    roster = [json.loads(line) for line in (cache / 'roster.jsonl').read_text().splitlines()]
    assert [row['id'] for row in roster] == [row['id'] for row in rows]
    assert [row['position'] for row in roster] == [0, 1, 2]
    assert [row['split'] for row in roster] == ['train', 'validation', 'train']
    for position in range(3):
        result = targets.get(position)
        assert result.shape == shape and result.dtype == torch.float32
        expected = np.stack([targets.images[position] @ text for text in targets.text]).reshape(shape)
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-6)
        assert result.abs().max() <= 1.000001
    assert (targets.get(0) < 0).any()  # Cosines, not sigmoid/probability or detector labels.
    for invalid in (-1, 3, '0', True):
        with pytest.raises(KeyError):
            targets.get(invalid)
    assert all(len(batch) <= cfg.teacher.batch_size for batch in calls.images)
    if method == 'lf_cbm':
        assert calls.rgb == [('RGB', (83, 311)), ('RGB', (417, 95)), ('RGB', (161, 203))]
    else:
        assert not calls.rgb  # Never run official crop/resize after adding SALF circles.
        assert 'reconstruction' in manifest['identity']['provenance']
        assert '12x12' in manifest['identity']['provenance']
        assert sum(len(batch) for batch in calls.images) == 147
    assert teacher.prepare_teacher(cfg) == manifest
    assert len(calls.load) == 1  # Existing caches verify without loading CLIP.


def test_salf_geometry_exact_pil_outline_after_backbone_crop(setup):
    cfg, _, calls, _, _ = setup
    cfg.model.method = 'salf_cbm'
    teacher.prepare_teacher(cfg)
    actual = torch.cat(calls.images)
    normalize = transforms.Normalize(teacher._MEAN, teacher._STD)
    # Patterned arbitrary geometry catches crop/resize ordering and interpolation.
    pixels = np.arange(107 * 359 * 3, dtype=np.uint8).reshape(107, 359, 3)
    image = Image.fromarray(pixels)
    base = transforms.CenterCrop(224)(transforms.Resize(256)(image))
    prompts = list(teacher._prompt_images(image, 7, 32))
    for k, prompted in enumerate(prompts):
        x, y = (k % 7 + 1) * 28, (k // 7 + 1) * 28
        expected = base.copy()
        ImageDraw.Draw(expected).ellipse((x-32, y-32, x+32, y+32), outline='red', width=2)
        np.testing.assert_array_equal(prompted, expected)
    assert np.array_equal(np.asarray(prompts[24])[112, 112], np.asarray(base)[112, 112])
    # Encoder sees normalization only, matching the local reconstructed method.
    original = Image.new('RGB', (83, 311), (70, 70, 70))
    expected = torch.stack([normalize(transforms.ToTensor()(p))
                            for p in teacher._prompt_images(original, 7, 32)])
    torch.testing.assert_close(actual[:49], expected)


def test_edge_prompt_geometry_and_cache_identity(setup):
    cfg, _, calls, _, _ = setup
    cfg.model.method = 'salf_cbm'
    cfg.teacher.prompt_grid = 'edge'
    manifest = teacher.prepare_teacher(cfg)
    centers = [0,37,74,111,148,185,222]
    assert teacher.prompt_centers(224,7,'edge') == centers
    assert manifest['identity']['geometry']['centers'] == centers
    image = Image.new('RGB',(224,224),(70,70,70))
    prompts = list(teacher._prompt_images(image,7,32,'edge'))
    assert len(prompts) == 49
    for k,prompt in enumerate(prompts):
        x,y = centers[k%7],centers[k//7]
        expected = image.copy()
        ImageDraw.Draw(expected).ellipse((x-32,y-32,x+32,y+32),outline='red',width=2)
        np.testing.assert_array_equal(prompt,expected)
    normalize = transforms.Normalize(teacher._MEAN,teacher._STD)
    original = Image.new('RGB',(83,311),(70,70,70))
    expected = torch.stack([normalize(transforms.ToTensor()(p))
                           for p in teacher._prompt_images(original,7,32,'edge')])
    torch.testing.assert_close(torch.cat(calls.images)[:49],expected)
    cfg.teacher.prompt_grid = 'interior'
    with pytest.raises(ValueError):
        teacher.verify_teacher(cfg)


def test_edge_prompt_bounds():
    assert teacher.prompt_centers(224,2,'edge') == [0,223]
    with pytest.raises(ValueError):
        teacher.prompt_centers(224,1,'edge')
    with pytest.raises(ValueError):
        teacher.prompt_centers(224,7,'unknown')


def test_archive_offsets_and_source_mutation(setup, tmp_path):
    cfg, _, _, _, rows = setup
    archive = tmp_path / 'images.bin'
    payload = b'header'
    for row in rows:
        content = teacher.Path(row['image']).read_bytes()
        row['image'] = dict(archive=str(archive), offset=len(payload), size=len(content))
        payload += content
    archive.write_bytes(payload)
    source = tmp_path / 'archive.jsonl'
    source.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    root = tmp_path / 'archive-prepared'
    data.prepare(source, root, 'cub', tmp_path / 'concepts.txt', split=tmp_path / 'split.json')
    cfg.data.prepared = str(root)
    teacher.prepare_teacher(cfg)
    assert teacher.verify_teacher(cfg)['rows_written'] == 3
    archive.write_bytes(payload[:-1] + bytes([payload[-1] ^ 1]))
    with pytest.raises(ValueError, match='source image/cohort'):
        teacher.verify_teacher(cfg)
    archive.write_bytes(b'')
    with pytest.raises(ValueError, match='Short archive read'):
        teacher.verify_teacher(cfg)


@pytest.mark.parametrize('artifact', ['text.npy', 'images.npy', 'positions.npy', 'roster.jsonl'])
def test_cache_artifact_hash_mutation(setup, artifact):
    cfg, _, _, _, _ = setup
    teacher.prepare_teacher(cfg)
    path = teacher.Path(cfg.teacher.cache) / artifact
    content = path.read_bytes()
    path.write_bytes(content[:-1] + bytes([content[-1] ^ 1]))
    with pytest.raises(ValueError, match='changed or partial'):
        teacher.verify_teacher(cfg)


@pytest.mark.parametrize('change', ['checkpoint', 'image', 'model', 'batch_size', 'radius', 'method', 'split', 'vocabulary', 'protocol'])
def test_identity_and_input_mutations_rejected(setup, change):
    cfg, protocol, _, _, rows = setup
    teacher.prepare_teacher(cfg)
    if change == 'checkpoint':
        teacher.Path(cfg.teacher.checkpoint).write_bytes(b'new checkpoint')
    elif change == 'image':
        Image.new('RGB', (15, 21)).save(rows[0]['image'])
    elif change == 'model':
        cfg.teacher.model = 'ViT-B/32'
    elif change == 'batch_size':
        cfg.teacher.batch_size += 1
    elif change == 'radius':
        cfg.teacher.prompt_radius = 31
    elif change == 'method':
        cfg.model.method = 'salf_cbm'
    elif change in {'split', 'vocabulary'}:
        name = 'split.json' if change == 'split' else 'concepts.txt'
        with (teacher.Path(cfg.data.prepared) / name).open('a') as stream:
            stream.write('\n')
    else:
        protocol = dict(protocol, images=123)
    with pytest.raises(ValueError):
        teacher.verify_teacher(cfg, protocol)


def test_absolute_noncontiguous_positions_and_roster_validation(setup):
    cfg, protocol, _, _, _ = setup
    root = teacher.Path(cfg.data.prepared)
    with sqlite3.connect(root / 'samples.sqlite') as db:
        db.execute('UPDATE samples SET position=position*10+10')
    with np.load(root / 'indices.npz') as indices:
        changed = {name: indices[name] * 10 + 10 for name in indices.files}
    np.savez(root / 'indices.npz', **changed)
    for name in ('samples.sqlite', 'indices.npz'):
        protocol['files'][name] = data.digest(root / name)
    data.save_json(root / 'prepared.json', protocol)
    teacher.prepare_teacher(cfg)
    targets = teacher.TeacherTargets(cfg, protocol)
    assert targets.get(10).shape == (2,)
    assert targets.get(30).shape == (2,)
    with pytest.raises(KeyError):
        targets.get(0)
    with pytest.raises(KeyError):
        targets.get(11)
    # Rehashing inconsistent prepared indices must not bypass the cohort check.
    cfg.teacher.cache += '-invalid'
    np.savez(root / 'indices.npz', train=[20, 10], validation=[30])
    protocol['files']['indices.npz'] = data.digest(root / 'indices.npz')
    data.save_json(root / 'prepared.json', protocol)
    with pytest.raises(ValueError, match='ID/position mismatch'):
        teacher.prepare_teacher(cfg)
    assert not (teacher.Path(cfg.teacher.cache) / 'complete.json').exists()


def test_failed_preparation_is_explicitly_partial(setup, monkeypatch):
    cfg, _, _, encoder, _ = setup
    monkeypatch.setattr(encoder, 'encode_image', lambda pixels: torch.zeros(len(pixels), 3))
    with pytest.raises(ValueError, match='zero CLIP embedding'):
        teacher.prepare_teacher(cfg)
    assert not (teacher.Path(cfg.teacher.cache) / 'complete.json').exists()
    with pytest.raises(ValueError, match='partial'):
        teacher.prepare_teacher(cfg)
    with pytest.raises(ValueError, match='partial'):
        teacher.verify_teacher(cfg)


def test_written_coverage_checked_even_with_updated_hash(setup):
    cfg, _, _, _, _ = setup
    manifest = teacher.prepare_teacher(cfg)
    cache = teacher.Path(cfg.teacher.cache)
    images = np.load(cache / 'images.npy', mmap_mode='r+')
    images[-1] = 0
    images.flush()
    manifest['files']['images.npy'] = data.digest(cache / 'images.npy')
    data.save_json(cache / 'complete.json', manifest)
    with pytest.raises(ValueError, match='incomplete or not normalized'):
        teacher.verify_teacher(cfg)


def test_marker_coverage_and_grid_are_bound(setup):
    cfg, _, _, _, _ = setup
    cfg.model.method = 'salf_cbm'
    manifest = teacher.prepare_teacher(cfg)
    cfg.model.salf_grid = 12
    with pytest.raises(ValueError, match='identity/settings'):
        teacher.verify_teacher(cfg)
    cfg.model.salf_grid = 7
    manifest['rows_written'] -= 1
    data.save_json(teacher.Path(cfg.teacher.cache) / 'complete.json', manifest)
    with pytest.raises(ValueError, match='incomplete coverage'):
        teacher.verify_teacher(cfg)


def test_roster_reordering_rejected_even_with_updated_hash(setup):
    cfg, _, _, _, _ = setup
    manifest = teacher.prepare_teacher(cfg)
    cache = teacher.Path(cfg.teacher.cache)
    path = cache / 'roster.jsonl'
    lines = path.read_text().splitlines(keepends=True)
    path.write_text(''.join(reversed(lines)))
    manifest['files']['roster.jsonl'] = data.digest(path)
    data.save_json(cache / 'complete.json', manifest)
    with pytest.raises(ValueError, match='cohort roster changed'):
        teacher.verify_teacher(cfg)


def test_no_checkpoint_download_and_no_clip_import_during_verify(setup, monkeypatch):
    cfg, _, calls, _, _ = setup
    teacher.prepare_teacher(cfg)

    def forbidden(*args, **kwargs):
        raise AssertionError('CLIP import must remain lazy')

    monkeypatch.setattr(teacher.importlib, 'import_module', forbidden)
    teacher.verify_teacher(cfg)
    teacher.prepare_teacher(cfg)
    cfg.teacher.cache += '-new'
    cfg.teacher.checkpoint = 'ViT-B/16'
    with pytest.raises(ValueError, match='explicit local'):
        teacher.prepare_teacher(cfg)
    assert len(calls.load) == 1


@pytest.mark.parametrize('setting,value', [('resize_size', 240), ('input_size', 256),
                                          ('prompt_radius', 16), ('salf_grid', 224)])
def test_reject_incompatible_salf_geometry(setup, setting, value):
    cfg, _, calls, _, _ = setup
    cfg.model.method = 'salf_cbm'
    owner = cfg.data if setting.endswith('_size') else cfg.teacher if setting == 'prompt_radius' else cfg.model
    setattr(owner, setting, value)
    with pytest.raises(ValueError):
        teacher.prepare_teacher(cfg)
    assert not calls.load


def test_trusted_proof_skips_all_full_audits(setup, monkeypatch):
    cfg, protocol, _, _, _ = setup
    teacher.prepare_teacher(cfg)
    proof = teacher.verify_teacher(cfg, protocol)
    expected = teacher.TeacherTargets(cfg, protocol, proof).get(0)

    def forbidden(*args, **kwargs):
        raise AssertionError('Trusted proof must not rescan inputs or cache contents')

    for name in ('_identity', '_rows', '_image_bytes', '_verify_cache'):
        monkeypatch.setattr(teacher, name, forbidden)
    monkeypatch.setattr(data, 'verify', forbidden)
    monkeypatch.setattr(data, 'digest', forbidden)
    original_load, opened = np.load, []

    def load(path, **kwargs):
        assert kwargs == dict(mmap_mode='r', allow_pickle=False)
        opened.append(teacher.Path(path).name)
        return original_load(path, **kwargs)

    monkeypatch.setattr(np, 'load', load)
    for _ in range(3):
        targets = teacher.TeacherTargets(cfg, protocol, verified_manifest=proof)
        torch.testing.assert_close(targets.get(0), expected)
    assert opened == ['text.npy', 'images.npy', 'positions.npy'] * 3
    # No implicit global memoization: the default and public audits still scan.
    with pytest.raises(AssertionError, match='must not rescan'):
        teacher.TeacherTargets(cfg, protocol)
    with pytest.raises(AssertionError, match='must not rescan'):
        teacher.verify_teacher(cfg, protocol)
    with pytest.raises(AssertionError, match='must not rescan'):
        teacher.prepare_teacher(cfg)


@pytest.mark.parametrize('change', ['method', 'grid', 'protocol', 'model', 'batch_size',
                                   'radius', 'checkpoint', 'device', 'resize', 'input',
                                   'cache', 'prepared'])
def test_trusted_proof_rejects_wrong_configuration(setup, change):
    cfg, protocol, _, _, _ = setup
    cfg.model.method = 'salf_cbm'
    teacher.prepare_teacher(cfg)
    proof = teacher.verify_teacher(cfg, protocol)
    if change == 'method':
        cfg.model.method = 'lf_cbm'
    elif change == 'grid':
        cfg.model.salf_grid = 12
    elif change == 'protocol':
        protocol = dict(protocol, split_seed=999)
    elif change == 'model':
        cfg.teacher.model = 'ViT-B/32'
    elif change == 'batch_size':
        cfg.teacher.batch_size += 1
    elif change == 'radius':
        cfg.teacher.prompt_radius = 31
    elif change == 'checkpoint':
        cfg.teacher.checkpoint += '-other'
    elif change == 'device':
        cfg.training.device = 'cuda'
    elif change == 'resize':
        cfg.data.resize_size = 240
    elif change == 'input':
        cfg.data.input_size = 256
    elif change == 'cache':
        cfg.teacher.cache += '-other'
    else:
        cfg.data.prepared += '-other'
    with pytest.raises(ValueError):
        teacher.TeacherTargets(cfg, protocol, verified_manifest=proof)


@pytest.mark.parametrize('change', ['method', 'protocol', 'settings', 'pid', 'json'])
def test_trusted_proof_rejects_mutated_or_deserialized_receipt(setup, change):
    cfg, protocol, _, _, _ = setup
    teacher.prepare_teacher(cfg)
    proof = deepcopy(teacher.verify_teacher(cfg, protocol))
    if change == 'method':
        proof['identity']['method'] = 'salf_cbm'
    elif change == 'protocol':
        proof['identity']['protocol']['images'] += 1
    elif change == 'settings':
        proof['identity']['teacher']['batch_size'] += 1
    elif change == 'pid':
        proof._pid = -1
    else:
        proof = json.loads(json.dumps(proof))
    with pytest.raises(ValueError, match='Trusted teacher proof'):
        teacher.TeacherTargets(cfg, protocol, verified_manifest=proof)


def test_pickling_reopens_readonly_memmaps_without_audit(setup, monkeypatch):
    cfg, protocol, _, _, _ = setup
    teacher.prepare_teacher(cfg)
    proof = teacher.verify_teacher(cfg, protocol)
    targets = teacher.TeacherTargets(cfg, protocol, verified_manifest=proof)
    expected = targets.get(1)
    assert not {'text', 'images', 'positions'} & targets.__getstate__().keys()
    payload = pickle.dumps(targets)
    assert len(payload) < len(json.dumps(targets.manifest)) + 1024

    def forbidden(*args, **kwargs):
        raise AssertionError('Workers must not repeat the parent audit')

    monkeypatch.setattr(teacher, 'verify_teacher', forbidden)
    monkeypatch.setattr(data, 'digest', forbidden)
    restored = pickle.loads(payload)
    for name in ('text', 'images', 'positions'):
        array = getattr(restored, name)
        assert isinstance(array, np.memmap) and array.mode == 'r'
        assert not array.flags.writeable
        assert teacher.Path(array.filename) == teacher.Path(getattr(targets, name).filename)
        assert array is not getattr(targets, name)
    torch.testing.assert_close(restored.get(1), expected)


def test_spawn_dataloader_reopens_teacher_memmaps(setup):
    cfg, protocol, _, _, _ = setup
    teacher.prepare_teacher(cfg)
    proof = teacher.verify_teacher(cfg, protocol)
    targets = teacher.TeacherTargets(cfg, protocol, verified_manifest=proof)
    loader = torch.utils.data.DataLoader(_SpawnTargets(targets), batch_size=2, num_workers=1,
                                        multiprocessing_context='spawn', timeout=30)
    batches = list(loader)
    for _, mapped, pids in batches:
        assert mapped.all()
        assert (pids != os.getpid()).all()
    actual = torch.cat([batch[0] for batch in batches])
    torch.testing.assert_close(actual, torch.stack([targets.get(i) for i in range(3)]))


@pytest.mark.parametrize('method', ['lf_cbm', 'salf_cbm'])
@pytest.mark.parametrize('fail', [False, True])
def test_fp32_encoding_overrides_and_restores_caller_precision(setup, monkeypatch, method, fail):
    cfg, _, calls, encoder, _ = setup
    cfg.model.method = method
    assert encoder.weight.dtype == torch.float16
    matmul_tf32, cudnn_tf32 = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    if fail:
        def invalid(pixels):
            encoder.project(torch.ones(len(pixels), 3))
            return torch.zeros(len(pixels), 3)
        monkeypatch.setattr(encoder, 'encode_image', invalid)
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        with torch.autocast('cpu', dtype=torch.bfloat16):
            if fail:
                with pytest.raises(ValueError, match='zero CLIP embedding'):
                    teacher.prepare_teacher(cfg)
            else:
                manifest = teacher.prepare_teacher(cfg)
                assert manifest['identity']['precision'] == dict(
                    encoder='model.float(): float32 weights and image inputs',
                    normalization='float32', storage='float32', autocast=False,
                    cuda_matmul_tf32=False, cudnn_tf32=False)
            assert cpu_autocast_enabled()
        assert calls.setup == ['float', 'eval']
        assert encoder.weight.dtype == torch.float32
        assert torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32


def test_cache_without_compute_precision_provenance_is_rejected(setup):
    cfg, _, _, _, _ = setup
    manifest = teacher.prepare_teacher(cfg)
    del manifest['identity']['precision']
    data.save_json(teacher.Path(cfg.teacher.cache) / 'complete.json', manifest)
    with pytest.raises(ValueError, match='identity/settings mismatch'):
        teacher.verify_teacher(cfg)
