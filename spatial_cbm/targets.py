"""Lossless, process-safe target caching on node-local storage.

Only deterministic supervision is cached, never transformed images or model outputs.
Callers must identify annotations, sample ordering, vocabulary, and crop geometry.
SQLite WAL requires a local filesystem, not the shared workspace PVC.
"""
import hashlib
import io
import json
import os
from pathlib import Path
import sqlite3
from typing import Sequence, Tuple

import numpy as np
import torch


class SpatialTargetCache:
    def __init__(self, directory, identity):
        self.identity = json.dumps(identity, sort_keys=True, separators=(',', ':'), allow_nan=False)
        key = hashlib.sha256(self.identity.encode()).hexdigest()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f'{key}.sqlite'
        self.connection, self.pid = None, None
        with sqlite3.connect(self.path, timeout=60) as database:
            database.execute('PRAGMA journal_mode=WAL')
            database.execute('CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
            database.execute('CREATE TABLE IF NOT EXISTS targets (id INTEGER PRIMARY KEY, payload BLOB NOT NULL)')
            database.execute('INSERT OR IGNORE INTO metadata VALUES (?,?)', ('identity', self.identity))
            stored = database.execute('SELECT value FROM metadata WHERE key=?', ('identity',)).fetchone()[0]
            if stored != self.identity:
                raise ValueError('Spatial target cache identity mismatch')

    def _database(self):
        if self.pid != os.getpid():
            self.close()
            self.connection = sqlite3.connect(self.path, timeout=60, isolation_level=None)
            self.connection.execute('PRAGMA synchronous=NORMAL')
            self.pid = os.getpid()
        return self.connection

    def close(self):
        if self.connection is not None:
            self.connection.close()
        self.connection, self.pid = None, None

    def __getstate__(self):
        return {**self.__dict__, 'connection': None, 'pid': None}

    def get(self, index):
        record = self._database().execute('SELECT payload FROM targets WHERE id=?', (int(index),)).fetchone()
        if record is None:
            return None
        with np.load(io.BytesIO(record[0]), allow_pickle=False) as stored:
            return tuple(torch.from_numpy(stored[k].copy()) for k in ('labels', 'ids', 'masks'))

    def put(self, index, value):
        labels, ids, masks = [t.detach().cpu().numpy() for t in value]
        if labels.ndim != 1 or ids.ndim != 1 or masks.ndim != 3 or len(ids) != len(masks):
            raise ValueError('Malformed sparse targets')
        if not np.isfinite(labels).all() or not np.isfinite(masks).all():
            raise ValueError('Nonfinite targets cannot be cached')
        payload = io.BytesIO()
        np.savez_compressed(payload, labels=labels, ids=ids, masks=masks)
        self._database().execute('INSERT OR IGNORE INTO targets VALUES (?,?)',
                                 (int(index), sqlite3.Binary(payload.getvalue())))

    def summary(self):
        count = self._database().execute('SELECT count(*) FROM targets').fetchone()[0]
        return dict(path=str(self.path), records=count, identity=json.loads(self.identity))


def pad_sparse_targets(
    mask_indices: Sequence[torch.Tensor],
    mask_targets: Sequence[torch.Tensor],
    *,
    mask_h: int,
    mask_w: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad sparse per-image concept masks into batched tensors."""
    batch_size = len(mask_indices)
    max_k = max((int(indices.numel()) for indices in mask_indices), default=0)
    if max_k == 0:
        return (
            torch.full((batch_size, 1), -1, dtype=torch.long),
            torch.zeros((batch_size, 1, int(mask_h), int(mask_w)), dtype=torch.float32),
            torch.zeros((batch_size, 1), dtype=torch.bool),
        )

    idx_pad = torch.full((batch_size, max_k), -1, dtype=torch.long)
    mask_pad = torch.zeros((batch_size, max_k, int(mask_h), int(mask_w)), dtype=torch.float32)
    valid = torch.zeros((batch_size, max_k), dtype=torch.bool)
    for batch_index, (indices, masks) in enumerate(zip(mask_indices, mask_targets)):
        count = int(indices.numel())
        if count == 0:
            continue
        idx_pad[batch_index, :count] = indices
        mask_pad[batch_index, :count] = masks
        valid[batch_index, :count] = True
    return idx_pad, mask_pad, valid
