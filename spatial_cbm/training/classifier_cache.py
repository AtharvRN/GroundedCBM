"""Cache standardized concept features without changing DataLoader permutations."""
import mmap
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class RowIndices(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return self.rows

    def __getitem__(self, index):
        return index


class CachedFeatureLoader:
    """Retain the native CPU sampler, worker lifecycle, and RNG consumption."""
    def __init__(self, original, features, targets):
        self.dataset = original.dataset
        self.features, self.targets = features, targets
        self.passes = 0
        kwargs = dict(batch_size=original.batch_size,
                      batch_sampler=original.batch_sampler,
                      num_workers=original.num_workers,
                      pin_memory=original.pin_memory,
                      persistent_workers=original.persistent_workers)
        # batch_sampler already owns batching; the constructor requires defaults.
        kwargs.pop("batch_size")
        if original.num_workers:
            kwargs["prefetch_factor"] = original.prefetch_factor
        self.indices = DataLoader(RowIndices(len(self.dataset)), **kwargs)

    def __len__(self):
        return len(self.indices)

    def __iter__(self):
        self.passes += 1
        started = time.monotonic()
        interval = int(os.environ.get("SALF_CACHE_LOG_EVERY", "0"))
        for batch_index, indices in enumerate(self.indices, 1):
            device_indices = indices.to(self.features.device)
            batch = (self.features[device_indices], self.targets[indices])
            yield (*batch, indices) if self.dataset.include_index else batch
            if interval > 0 and (batch_index == 1 or batch_index % interval == 0 or batch_index == len(self)):
                print(f"[saga:batches] pass={self.passes} batch={batch_index}/{len(self)} seconds={time.monotonic()-started:.1f}", flush=True)


def cache_feature_loader(original, device, table_classes=0, chunk_rows=4096):
    dataset = original.dataset
    rows, columns = dataset.features.shape
    feature_bytes = rows * columns * 4
    if str(device).startswith("cuda"):
        # Release inference workspace retained by PyTorch before capacity checks.
        torch.cuda.empty_cache()
        selected = torch.device(device)
        free, _ = torch.cuda.mem_get_info(selected.index if selected.index is not None else torch.cuda.current_device())
        needed = feature_bytes + rows * table_classes * 4 + 1024**3
        if free < needed:
            raise RuntimeError(f"GPU feature cache needs {needed/1024**3:.2f} GiB, available {free/1024**3:.2f} GiB")
    features = torch.empty((rows, columns), dtype=torch.float32, device=device)
    targets = torch.from_numpy(np.array(dataset.targets, dtype=np.int64, copy=True))
    fd = os.open(dataset.features.filename, os.O_RDONLY)
    started, last_report = time.monotonic(), time.monotonic()
    try:
        for start in range(0, rows, chunk_rows):
            end = min(start + chunk_rows, rows)
            # Exactly the original Dataset's float32 NumPy normalization.
            chunk = np.asarray(dataset.features[start:end], dtype=np.float32)
            if dataset.mean is not None and dataset.std is not None:
                chunk = (chunk - dataset.mean) / dataset.std
            if not np.isfinite(chunk).all():
                raise ValueError("Nonfinite standardized classifier features")
            features[start:end].copy_(torch.from_numpy(np.array(chunk, copy=True)))
            if str(device).startswith("cuda"):
                torch.cuda.synchronize(device)
            # Remove mapped pages before asking the filesystem to discard them.
            begin_byte = int(dataset.features.offset) + start * columns * dataset.features.dtype.itemsize
            end_byte = int(dataset.features.offset) + end * columns * dataset.features.dtype.itemsize
            aligned = begin_byte // mmap.PAGESIZE * mmap.PAGESIZE
            if hasattr(dataset.features._mmap, "madvise"):
                dataset.features._mmap.madvise(mmap.MADV_DONTNEED, aligned, end_byte - aligned)
            if hasattr(os, "posix_fadvise"):
                os.posix_fadvise(fd, aligned, end_byte - aligned, os.POSIX_FADV_DONTNEED)
            now = time.monotonic()
            if now - last_report >= 10 or end == rows:
                print(f"[saga:cache] rows={end}/{rows} device={device} seconds={now-started:.1f}", flush=True)
                last_report = now
    finally:
        os.close(fd)
    return CachedFeatureLoader(original, features, targets)


class MemmapFeatureDataset(Dataset):
    def __init__(
        self,
        feature_path: Path,
        target_path: Path,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
        include_index: bool = False,
    ) -> None:
        self.features = np.load(feature_path, mmap_mode="r")
        self.targets = np.load(target_path, mmap_mode="r")
        self.mean = None if mean is None else np.asarray(mean, dtype=np.float32)
        self.std = None if std is None else np.asarray(std, dtype=np.float32)
        self.include_index = include_index

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int):
        feature = np.asarray(self.features[index], dtype=np.float32)
        if self.mean is not None and self.std is not None:
            feature = (feature - self.mean) / self.std
        tensor = torch.from_numpy(np.array(feature, dtype=np.float32, copy=True))
        target = int(self.targets[index])
        if self.include_index:
            return tensor, target, int(index)
        return tensor, target
