"""Optional, factorized official OpenAI CLIP teachers, never annotation labels.

Only the prepared training pool (train + internal validation) is encoded. SALF
uses the corrected local prompt reconstruction, not an unavailable upstream
training implementation. Verification hashes source image bytes as well as all
cache artifacts; it is deliberately strict and can be I/O intensive.
"""
import hashlib
import importlib
import io
import json
import os
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import sqlite3

import numpy as np
from PIL import Image, ImageDraw
import torch
from torchvision import transforms

from . import data


_ARTIFACTS = ('text.npy', 'images.npy', 'positions.npy', 'roster.jsonl')
_MEAN = (.48145466, .4578275, .40821073)
_STD = (.26862954, .26130258, .27577711)


def _settings_identity(cfg, protocol):
    """Configuration-only identity; no checkpoint, prepared, or image reads."""
    if cfg.model.method not in {'lf_cbm', 'salf_cbm'}:
        raise ValueError('Teacher requires lf_cbm or salf_cbm')
    checkpoint = Path(cfg.teacher.checkpoint).expanduser()
    if not str(cfg.teacher.checkpoint):
        raise ValueError('Teacher requires an explicit local CLIP checkpoint file')
    if not cfg.teacher.model or type(cfg.teacher.batch_size) is not int or cfg.teacher.batch_size < 1:
        raise ValueError('Teacher model and positive integer batch_size are required')
    grid = cfg.model.salf_grid if cfg.model.method == 'salf_cbm' else 1
    layout = getattr(cfg.teacher, 'prompt_grid', 'interior')
    if type(grid) is not int or grid < 1:
        raise ValueError('Invalid SALF grid')
    if cfg.model.method == 'salf_cbm':
        if (cfg.data.resize_size, cfg.data.input_size) != (256, 224):
            raise ValueError('Corrected SALF requires resize256/center224 geometry')
        if cfg.teacher.prompt_radius != 32:
            raise ValueError('Corrected SALF requires radius32')
        prompt_centers(224, grid, layout)
    return dict(version=1, method=cfg.model.method, protocol=protocol,
        teacher=dict(implementation='official OpenAI CLIP', model=cfg.teacher.model,
            checkpoint=str(checkpoint.resolve()),
            batch_size=cfg.teacher.batch_size, device=str(cfg.training.device),
            prompt_radius=cfg.teacher.prompt_radius),
        geometry=dict(resize_size=cfg.data.resize_size, input_size=cfg.data.input_size,
            grid=grid, ordering='image-major, row-major lattice', layout=layout,
            preprocessing=('official CLIP preprocess on original RGB' if grid == 1 and cfg.model.method == 'lf_cbm'
                else 'RGB; bilinear resize256; center224; red outline; CLIP normalization only'),
            circle_width=2, circle_color=[255, 0, 0],
            stride=('input_size // (grid + 1)' if layout=='interior' else 'input_size // (grid - 1)'),
            centers=(prompt_centers(224, grid, layout) if cfg.model.method=='salf_cbm' else None)),
        text_prompt='literal concept strings, no template', dtype='float32',
        precision=dict(encoder='model.float(): float32 weights and image inputs',
            normalization='float32', storage='float32', autocast=False,
            cuda_matmul_tf32=False, cudnn_tf32=False),
        target='L2-normalized image/text dot product; no logit scale',
        cohort='samples.sqlite: train + internal validation only; test excluded',
        supervision='label-free teacher; evaluation requires separately supplied annotations',
        provenance=('LF literal-string CLIP teacher' if cfg.model.method == 'lf_cbm' else
            'Corrected local reconstruction from methods/salf_paper.py and '
            'scripts/train_places365_salf.py; full upstream SALF training unavailable. '
            'Paper classification grid is 7x7; released inference defaults to 12x12.'))


def _identity(cfg, protocol=None):
    root = Path(cfg.data.prepared)
    actual = data.verify(root)
    if protocol is not None and actual != protocol:
        raise ValueError('Teacher prepared protocol mismatch')
    identity = _settings_identity(cfg, actual)
    checkpoint = Path(identity['teacher']['checkpoint'])
    if not checkpoint.is_file():
        raise ValueError('Teacher requires an explicit local CLIP checkpoint file')
    bank = (root / 'concepts.txt').read_text().splitlines()
    if not bank or any(not c.strip() or c != c.strip() for c in bank) or len(set(bank)) != len(bank):
        raise ValueError('Teacher requires a nonempty, unique literal concept vocabulary')
    identity['prepared_sha256'] = data.digest(root / 'prepared.json')
    identity['teacher']['checkpoint_sha256'] = data.digest(checkpoint)
    return identity, bank


class _VerifiedManifest(dict):
    """JSON-compatible manifest with an in-memory, process-local audit receipt."""
    def __init__(self, manifest, cfg):
        super().__init__(deepcopy(manifest))
        self._snapshot = deepcopy(manifest)
        self._pid = os.getpid()
        self._cache = Path(cfg.teacher.cache).resolve()
        self._prepared = Path(cfg.data.prepared).resolve()

    def checked_copy(self, cfg, protocol):
        if (self._pid != os.getpid() or self._cache != Path(cfg.teacher.cache).resolve()
                or self._prepared != Path(cfg.data.prepared).resolve() or self != self._snapshot):
            raise ValueError('Trusted teacher proof changed or belongs to another process/path')
        identity = deepcopy(self['identity'])
        del identity['prepared_sha256']
        del identity['teacher']['checkpoint_sha256']
        if identity != _settings_identity(cfg, protocol):
            raise ValueError('Trusted teacher proof method/grid/protocol/settings mismatch')
        return deepcopy(dict(self))


def _rows(root, protocol):
    """Cross-check the complete sqlite roster against both saved split formats."""
    split = json.loads((root / 'split.json').read_text())
    membership = {}
    with np.load(root / 'indices.npz', allow_pickle=False) as indices:
        for name in ('train', 'validation'):
            positions = indices[name]
            if positions.ndim != 1 or positions.dtype.kind not in 'iu':
                raise ValueError('Invalid teacher cohort indices')
            if len(positions) != len(split[name]) or len(positions) != protocol[name]:
                raise ValueError('Teacher cohort split count mismatch')
            for position, sample_id in zip(positions, split[name]):
                position = int(position)
                if position < 0 or position in membership:
                    raise ValueError('Teacher cohort has overlapping or duplicate positions')
                membership[position] = (sample_id, name)
    count = 0
    with sqlite3.connect(f'{(root / "samples.sqlite").resolve().as_uri()}?mode=ro', uri=True) as db:
        for position, sample_id, reference in db.execute('SELECT position,id,image FROM samples ORDER BY position'):
            member = membership.pop(position, None)
            if member is None or member[0] != sample_id:
                raise ValueError('Teacher cohort ID/position mismatch')
            count += 1
            yield dict(position=position, id=sample_id, image=json.loads(reference), split=member[1])
    if membership or count != protocol['images'] or count == 0:
        raise ValueError('Teacher cohort does not cover the complete prepared pool')


def _image_bytes(reference):
    if isinstance(reference, str):
        return Path(reference).read_bytes()
    if (not isinstance(reference, dict) or set(reference) != {'archive', 'offset', 'size'}
            or type(reference['offset']) is not int or type(reference['size']) is not int
            or reference['offset'] < 0 or reference['size'] < 1):
        raise ValueError('Invalid teacher archive reference')
    with Path(reference['archive']).open('rb') as stream:
        stream.seek(reference['offset'])
        payload = stream.read(reference['size'])
    if len(payload) != reference['size']:
        raise ValueError('Short archive read for teacher image')
    return payload


def _record(row, payload):
    return json.dumps(dict(row, image_sha256=hashlib.sha256(payload).hexdigest()),
                      sort_keys=True, separators=(',', ':')) + '\n'


def prompt_centers(size, grid, layout='interior'):
    if layout not in {'interior','edge'} or grid < (2 if layout=='edge' else 1):
        raise ValueError('Invalid prompt grid layout or size')
    stride = size // (grid + 1 if layout=='interior' else grid - 1)
    if stride < 1:
        raise ValueError('Prompt grid has zero stride')
    indices = range(1,grid+1) if layout=='interior' else range(grid)
    return [min(i*stride,size-1) for i in indices]


def _prompt_images(image, grid, radius, layout='interior'):
    base = transforms.CenterCrop(224)(transforms.Resize(256,
        interpolation=transforms.InterpolationMode.BILINEAR)(image))
    centers = prompt_centers(224, grid, layout)
    for cy in centers:
        for cx in centers:
            prompted = base.copy()
            ImageDraw.Draw(prompted).ellipse((cx-radius, cy-radius, cx+radius, cy+radius),
                                            outline=(255, 0, 0), width=2)
            yield prompted


def _normalized(encoded, count):
    encoded = encoded.detach().float()
    if encoded.ndim != 2 or encoded.shape[0] != count or encoded.shape[1] == 0:
        raise ValueError('Invalid CLIP embedding shape')
    norm = encoded.norm(dim=-1, keepdim=True)
    if not torch.isfinite(encoded).all() or not torch.isfinite(norm).all() or (norm <= 0).any():
        raise ValueError('Invalid nonfinite or zero CLIP embedding')
    return (encoded / norm).cpu().numpy()


@contextmanager
def _fp32_encoding(device):
    """Do not inherit caller AMP/TF32 policy; restore it even on failed writes."""
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with torch.inference_mode(), torch.autocast(device_type=torch.device(device).type, enabled=False):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32


def _check_array(path, shape, dtype, normalized=False):
    array = np.load(path, mmap_mode='r', allow_pickle=False)
    if array.shape != tuple(shape) or array.dtype != np.dtype(dtype):
        raise ValueError(f'Teacher array shape/dtype mismatch: {path.name}')
    if path.stat().st_size != array.offset + array.nbytes:
        raise ValueError(f'Teacher array size mismatch: {path.name}')
    if normalized:
        flat = array.reshape(-1, array.shape[-1])
        for start in range(0, len(flat), 4096):
            block = flat[start:start+4096]
            if not np.isfinite(block).all() or not np.allclose(np.linalg.norm(block, axis=-1), 1., atol=2e-5):
                raise ValueError(f'Teacher embeddings incomplete or not normalized: {path.name}')
    return array


def _verify_cache(cfg, identity, bank, manifest):
    cache, root = Path(cfg.teacher.cache), Path(cfg.data.prepared)
    if manifest.get('identity') != identity:
        raise ValueError('Teacher cache identity/settings mismatch')
    n, c, g = identity['protocol']['images'], len(bank), identity['geometry']['grid']
    d = manifest.get('embedding_dim')
    if type(d) is not int or d < 1 or manifest.get('rows_written') != n:
        raise ValueError('Teacher cache has incomplete coverage')
    if set(manifest.get('files', {})) != set(_ARTIFACTS):
        raise ValueError('Teacher cache artifact roster mismatch')
    for name, expected in manifest['files'].items():
        if not (cache / name).is_file() or data.digest(cache / name) != expected:
            raise ValueError(f'Teacher cache changed or partial: {name}')
    _check_array(cache / 'text.npy', (c, d), 'float32', True)
    _check_array(cache / 'images.npy', (n, g*g, d), 'float32', True)
    positions = _check_array(cache / 'positions.npy', (n,), 'int64')
    with (cache / 'roster.jsonl').open() as roster:
        for index, row in enumerate(_rows(root, identity['protocol'])):
            if index >= n or positions[index] != row['position']:
                raise ValueError('Teacher cache position roster mismatch')
            if roster.readline() != _record(row, _image_bytes(row['image'])):
                raise ValueError('Teacher source image/cohort roster changed')
        if roster.readline():
            raise ValueError('Teacher cache contains extra roster rows')


def verify_teacher(cfg, protocol=None):
    """Verify an existing cache without importing CLIP or generating any targets."""
    marker = Path(cfg.teacher.cache) / 'complete.json'
    if not marker.is_file():
        raise ValueError('Teacher cache missing or partial; prepare in a new/empty directory')
    identity, bank = _identity(cfg, protocol)
    manifest = json.loads(marker.read_text())
    _verify_cache(cfg, identity, bank, manifest)
    return _VerifiedManifest(manifest, cfg)


def prepare_teacher(cfg):
    """Encode the exact prepared pool. Failed/partial caches must be rebuilt.

    The supplied checkpoint is hashed, never downloaded or selected by model
    name. Install official OpenAI CLIP separately; tests can monkeypatch its
    lazy import. No detector annotations or held-out images are consulted.
    Encoding uses model.float(), disabled autocast/TF32, and FP32 normalization
    and storage. This promotes checkpoint weights, not their original precision.
    """
    cache = Path(cfg.teacher.cache)
    if (cache / 'complete.json').exists():
        return verify_teacher(cfg)
    if cache.exists() and any(cache.iterdir()):
        raise ValueError('Teacher cache is partial; use a new/empty directory')
    identity, bank = _identity(cfg)
    try:
        clip = importlib.import_module('clip')
    except ImportError as exc:
        raise ImportError('Teacher preparation requires the optional official OpenAI CLIP package') from exc
    model, preprocess = clip.load(identity['teacher']['checkpoint'], device=cfg.training.device, jit=False)
    model.float()
    model.eval()
    grid = identity['geometry']['grid']
    if cfg.model.method == 'salf_cbm' and int(model.visual.input_resolution) != 224:
        raise ValueError('Corrected SALF requires a CLIP checkpoint with input_resolution=224')
    cache.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents competing preparations from sharing output files.
    with (cache / 'preparing').open('x') as lock:
        lock.write('Incomplete until complete.json is atomically installed.\n')
    batch_size, device = cfg.teacher.batch_size, cfg.training.device
    n, dimension = identity['protocol']['images'], None
    with _fp32_encoding(device):
        for start in range(0, len(bank), batch_size):
            words = bank[start:start+batch_size]
            embeddings = _normalized(model.encode_text(clip.tokenize(words).to(device)), len(words))
            if dimension is None:
                dimension = embeddings.shape[1]
                text = np.lib.format.open_memmap(cache / 'text.npy', mode='w+', dtype='float32',
                                                shape=(len(bank), dimension))
            if embeddings.shape[1] != dimension:
                raise ValueError('CLIP text dimension changed')
            text[start:start+len(words)] = embeddings
        images = np.lib.format.open_memmap(cache / 'images.npy', mode='w+', dtype='float32',
                                           shape=(n, grid*grid, dimension))
        positions = np.lib.format.open_memmap(cache / 'positions.npy', mode='w+', dtype='int64', shape=(n,))
        flat, pending, written, rows_written = images.reshape(-1, dimension), [], 0, 0
        normalize = transforms.Compose([transforms.ToTensor(), transforms.Normalize(_MEAN, _STD)])

        def flush():
            nonlocal written
            if not pending:
                return
            encoded = _normalized(model.encode_image(torch.stack(pending).to(device, dtype=torch.float32)), len(pending))
            if encoded.shape[1] != dimension or written + len(pending) > len(flat):
                raise ValueError('CLIP image dimension/coverage mismatch')
            flat[written:written+len(pending)] = encoded
            written += len(pending)
            pending.clear()

        with (cache / 'roster.jsonl').open('w') as roster:
            for index, row in enumerate(_rows(Path(cfg.data.prepared), identity['protocol'])):
                if index >= n:
                    raise ValueError('Teacher cohort exceeds prepared count')
                payload = _image_bytes(row['image'])
                with Image.open(io.BytesIO(payload)) as original:
                    rgb = original.convert('RGB')
                views = ([preprocess(rgb)] if cfg.model.method == 'lf_cbm' else
                         (normalize(p) for p in _prompt_images(rgb, grid, cfg.teacher.prompt_radius,
                             getattr(cfg.teacher,'prompt_grid','interior'))))
                for tensor in views:
                    pending.append(tensor)
                    if len(pending) == batch_size:
                        flush()
                positions[index] = row['position']
                roster.write(_record(row, payload))
                rows_written += 1
            flush()
        if rows_written != n or written != n * grid * grid:
            raise ValueError('Teacher cache has incomplete written coverage')
        for array in (text, images, positions):
            array.flush()
    for name in _ARTIFACTS:
        with (cache / name).open('rb') as stream:
            os.fsync(stream.fileno())
    manifest = dict(identity=identity, embedding_dim=dimension, rows_written=rows_written,
                    files={name: data.digest(cache / name) for name in _ARTIFACTS})
    # Recheck inputs and every written row before publishing the completion marker.
    current, bank = _identity(cfg)
    _verify_cache(cfg, current, bank, manifest)
    data.save_json(cache / 'complete.json', manifest)
    return manifest


class TeacherTargets:
    """Read-only cosine targets indexed by absolute samples.sqlite position.

    Pass the unmodified result of this process's verify_teacher to reuse its
    audit. The caller must keep the checkpoint, prepared data, source images,
    and cache unchanged for the lifetime of that proof and these readers.
    Without a proof, construction always performs the full audit. Spawned
    workers reopen read-only mappings without repeating the parent's audit.
    """
    def __init__(self, cfg, protocol, verified_manifest=None):
        if verified_manifest is None:
            verified_manifest = verify_teacher(cfg, protocol)
        if not isinstance(verified_manifest, _VerifiedManifest):
            raise ValueError('Trusted teacher proof must come from verify_teacher in this process')
        self.manifest = verified_manifest.checked_copy(cfg, protocol)
        self.cache = Path(cfg.teacher.cache).resolve()
        self._open_arrays()

    def _open_arrays(self):
        self.text = np.load(self.cache / 'text.npy', mmap_mode='r', allow_pickle=False)
        self.images = np.load(self.cache / 'images.npy', mmap_mode='r', allow_pickle=False)
        self.positions = np.load(self.cache / 'positions.npy', mmap_mode='r', allow_pickle=False)

    def __getstate__(self):
        return {key: value for key, value in self.__dict__.items()
                if key not in {'text', 'images', 'positions'}}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._open_arrays()

    def get(self, position):
        if isinstance(position, bool) or not isinstance(position, (int, np.integer)):
            raise KeyError('Teacher position must be an absolute integer samples.sqlite position')
        index = int(np.searchsorted(self.positions, position))
        if index >= len(self.positions) or self.positions[index] != position:
            raise KeyError(f'Position not in prepared teacher cohort: {position}')
        scores = torch.from_numpy(np.asarray(self.text @ self.images[index].T, dtype=np.float32))
        identity = self.manifest['identity']
        if identity['method'] == 'lf_cbm':
            return scores[:, 0].contiguous()
        grid = identity['geometry']['grid']
        return scores.reshape(len(self.text), grid, grid).contiguous()
