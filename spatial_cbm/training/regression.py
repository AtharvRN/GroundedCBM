"""Full-cohort centered-cubic cosine with bounded working storage."""
import math
import time

import numpy as np
import torch


class StreamingSimilarity:
    """Merge centered moments, never retain rows or center each batch as a loss.

    Pure moments through order six and mixed moments through (3, 3) suffice
    for the normalized cubic cosine. A fixed origin removes large offsets
    before the parallel central-moment merge. Inputs retain the old meter's
    FP32 storage semantics; all accumulation is FP64 on the input device.
    """
    orders = tuple([(i, 0) for i in range(2, 7)] +
                   [(0, i) for i in range(2, 7)] +
                   [(i, j) for i in range(1, 4) for j in range(1, 4)])

    def __init__(self, rows):
        if type(rows) is not int or rows < 2:
            raise ValueError('Regression validation requires at least two images')
        self.rows, self.count = rows, 0
        self.origin = self.means = self.moments = None
        self.closed = False

    @staticmethod
    def _batch(x, y):
        means = (x.mean(0), y.mean(0))
        x, y = x-means[0], y-means[1]
        xp, yp = (x, x*x, x*x*x), (y, y*y, y*y*y)
        moments = {}
        for axis, powers in enumerate((xp, yp)):
            second, third = powers[1:]
            values = {2: second.sum(0), 3: third.sum(0),
                      4: (second*second).sum(0), 5: (second*third).sum(0),
                      6: (third*third).sum(0)}
            for order, value in values.items():
                moments[(order, 0) if axis == 0 else (0, order)] = value
        for i in range(1, 4):
            for j in range(1, 4):
                moments[i, j] = (xp[i-1]*yp[j-1]).sum(0)
        return means, moments

    @classmethod
    def _shift(cls, moments, count, dx, dy):
        powers = []
        for delta in (dx, dy):
            values = [None, delta]
            for _ in range(2, 7):
                values.append(values[-1]*delta)
            powers.append(values)
        result = {}
        for a, b in cls.orders:
            total = torch.zeros_like(dx)
            for i in range(a+1):
                for j in range(b+1):
                    if i+j == 1:
                        continue  # First central moments are zero by definition.
                    value = count if i+j == 0 else moments[i, j]
                    if a > i:
                        value = value*powers[0][a-i]
                    if b > j:
                        value = value*powers[1][b-j]
                    total.add_(value, alpha=math.comb(a, i)*math.comb(b, j))
            result[a, b] = total
        return result

    @torch.no_grad()
    def update(self, prediction, target):
        if self.closed:
            raise ValueError('Regression accumulator is closed')
        if (prediction.shape != target.shape or prediction.ndim not in (2, 4)
                or not all(prediction.shape) or self.count+len(prediction) > self.rows):
            raise ValueError('Invalid regression batch shape or excess cohort rows')
        if prediction.device != target.device:
            raise ValueError('Regression prediction and target devices differ')
        values = [v.detach().float().to(dtype=torch.float64) for v in (prediction, target)]
        if not all(torch.isfinite(v).all() for v in values):
            raise ValueError('Nonfinite regression predictions/targets')
        if self.origin is None:
            self.origin = tuple(v[0].clone() for v in values)
        elif any(v.shape[1:] != o.shape or v.device != o.device
                 for v, o in zip(values, self.origin)):
            raise ValueError('Regression shape or device changed within cohort')
        batch_means, batch_moments = self._batch(*(v-o for v, o in zip(values, self.origin)))
        n, m = self.count, len(prediction)
        if n == 0:
            self.means, self.moments = batch_means, batch_moments
        else:
            delta = tuple(b-a for a, b in zip(self.means, batch_means))
            left = self._shift(self.moments, n, *(-d*(m/(n+m)) for d in delta))
            right = self._shift(batch_moments, m, *(d*(n/(n+m)) for d in delta))
            self.moments = {order: left[order].add_(right[order]) for order in self.orders}
            self.means = tuple(a+d*(m/(n+m)) for a, d in zip(self.means, delta))
        self.count += m

    @torch.no_grad()
    def similarities(self):
        if self.closed or self.count != self.rows:
            raise ValueError('Incomplete or closed regression validation cohort')
        # Algebraically retain BOTH F.normalize epsilons, including tiny vectors.
        denominators = []
        for second, sixth in (((2, 0), (6, 0)), ((0, 2), (0, 6))):
            norm = self.moments[second].clamp_min(0).sqrt().clamp_min(1e-12)
            cube_norm = self.moments[sixth].clamp_min(0).sqrt()
            denominators.append(torch.maximum(cube_norm, norm.pow(3)*1e-8))
        result = self.moments[3, 3]/denominators[0]/denominators[1]
        if not torch.isfinite(result).all():
            raise ValueError('Nonfinite streaming regression result')
        return result.cpu()

    def close(self):
        self.origin = self.means = self.moments = None
        self.closed = True


def sequential_similarity(arrays, block_bytes=32 << 20, progress=None):
    """Reduce image-major arrays in three passes, retaining both normalization epsilons.

    block_bytes bounds one float64 input block; working storage is a small
    multiple of that budget plus per-concept/location float64 accumulators.
    """
    if len(arrays) != 2 or arrays[0].shape != arrays[1].shape:
        raise ValueError('Expected two matching regression arrays')
    shape = arrays[0].shape
    if len(shape) not in (2, 4) or shape[0] < 2 or not all(shape):
        raise ValueError('Expected [N,C,(H,W)] arrays with N>=2')
    if block_bytes < 1:
        raise ValueError('Block budget must be positive')
    rows = max(1, block_bytes // (math.prod(shape[1:]) * 8))
    means = [torch.zeros(shape[1:], dtype=torch.float64) for _ in arrays]
    norms = [torch.zeros_like(x) for x in means]
    cubes = [torch.zeros_like(x) for x in means]
    dot = torch.zeros_like(means[0])
    for stage in range(3):
        began = time.monotonic()
        for start in range(0, shape[0], rows):
            end = min(start + rows, shape[0])
            values = [torch.from_numpy(np.array(a[start:end], dtype=np.float64, copy=True))
                      for a in arrays]
            for i, value in enumerate(values):
                if stage == 0:
                    if not torch.isfinite(value).all():
                        raise ValueError('Nonfinite regression predictions/targets')
                    means[i].add_(value.sum(0))
                else:
                    value.sub_(means[i])
                    if stage == 1:
                        norms[i].add_(value.square_().sum(0))
                    else:
                        value.div_(norms[i]).pow_(3)
                        cubes[i].add_((value * value).sum(0))
            if stage == 2:
                dot.add_((values[0] * values[1]).sum(0))
            if progress is not None:
                progress(stage + 1, end, shape[0], time.monotonic() - began)
        if stage == 0:
            for mean in means:
                mean.div_(shape[0])
        elif stage == 1:
            for norm in norms:
                norm.sqrt_().clamp_min_(1e-12)
    for norm in cubes:
        dot.div_(norm.sqrt_().clamp_min_(1e-8))
    return dot
