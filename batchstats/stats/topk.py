"""Exact, bounded-memory selection of extreme values."""

import operator

import numpy as np

from .._misc import DifferentAxisError, DifferentShapesError, DifferentStatsError, NoValidSamplesError
from ..base.state import StateMixin


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer.")
    value = operator.index(value)
    if value < 1:
        raise ValueError(f"{name} must be positive.")
    return value


def _quantile(q):
    if np.ndim(q) != 0 or not np.isfinite(q) or not 0 <= q <= 1:
        raise ValueError("q must be a finite scalar in [0, 1].")
    return float(q)


def required_k(q, max_samples, largest=True):
    """Capacity needed for an exact linear quantile of at most max_samples values."""
    q = _quantile(q)
    n = _positive_integer(max_samples, "max_samples")
    h = (n - 1) * q
    return n - int(np.floor(h)) if largest else int(np.ceil(h)) + 1


class BatchTopK(StateMixin):
    """Retain k extrema per position; reject NaNs without changing the state.

    Inputs use the package's ``atleast_2d`` convention. Reduced axes are
    flattened; output is a masked array of shape ``(k, *remaining_shape)``.
    Rank 1 is the largest (or smallest) value; missing ranks are masked.
    Integer and floating dtypes are preserved, including infinities and ties.
    """

    _ignore_nan = False

    def __init__(self, k, axis=0, largest=True):
        self.k = _positive_integer(k, "k")
        if axis is not None:
            axes = axis if isinstance(axis, tuple) else (axis,)
            for ax in axes:
                if isinstance(ax, (bool, np.bool_)):
                    raise TypeError("axis must contain integers.")
                operator.index(ax)
        if not isinstance(largest, (bool, np.bool_)):
            raise TypeError("largest must be a boolean.")
        self.axis = (
            tuple(operator.index(ax) for ax in axis)
            if isinstance(axis, tuple)
            else None
            if axis is None
            else operator.index(axis)
        )
        self.largest = bool(largest)
        self.n_samples = None
        self.values = None
        self._ndim = None

    def _reshape(self, batch):
        axes = (
            tuple(range(batch.ndim))
            if self.axis is None
            else (self.axis if isinstance(self.axis, tuple) else (self.axis,))
        )
        if any(ax < -batch.ndim or ax >= batch.ndim for ax in axes):
            raise ValueError("axis out of bounds.")
        axes = tuple(ax % batch.ndim for ax in axes)
        if len(set(axes)) != len(axes):
            raise ValueError("Repeated reduction axis.")
        remaining = tuple(ax for ax in range(batch.ndim) if ax not in axes)
        shape = tuple(batch.shape[ax] for ax in remaining)
        n = int(np.prod([batch.shape[ax] for ax in axes], dtype=np.int64))
        return np.transpose(batch, axes + remaining).reshape((n, *shape))

    def _fill_value(self, dtype):
        if dtype.kind == "f":
            return -np.inf if self.largest else np.inf
        if dtype.kind == "b":
            return not self.largest
        info = np.iinfo(dtype)
        return info.min if self.largest else info.max

    def _select_work(self, work, sort=True):
        """Consume an owned buffer with contiguous ranks on its last axis."""
        n = work.shape[-1]
        if n > self.k:
            cut = n - self.k if self.largest else self.k - 1
            work.partition(cut, axis=-1)
            tail = work[..., -self.k :] if self.largest else work[..., : self.k]
            # Detach the small tail: a view would keep the full batch alive.
            work = tail.copy()
        elif n < self.k:
            padded = np.full((*work.shape[:-1], self.k), self._fill_value(work.dtype), dtype=work.dtype)
            padded[..., :n] = work
            work = padded
        if sort:
            work.sort(axis=-1)
            if self.largest:
                work = work[..., ::-1]
        return np.moveaxis(work, -1, 0)

    def _select(self, values, invalid=None, sort=True):
        if self.k == 1:
            reduce = np.max if self.largest else np.min
            where = True if invalid is None else ~invalid
            return reduce(values, axis=0, keepdims=True, initial=self._fill_value(values.dtype), where=where)
        work = np.moveaxis(values, 0, -1).copy(order="C")
        if invalid is not None:
            np.copyto(work, self._fill_value(work.dtype), where=np.moveaxis(invalid, 0, -1))
        return self._select_work(work, sort=sort)

    def _merge_values(self, values, counts):
        dtype = np.result_type(self.values.dtype, values.dtype)
        work = np.empty((*counts.shape, 2 * self.k), dtype=dtype)
        work[..., : self.k] = np.moveaxis(self.values, 0, -1)
        work[..., self.k :] = np.moveaxis(values, 0, -1)
        # Refill missing ranks after promotion: an integer sentinel need not be
        # an extremum of the promoted dtype. Restored states can have arbitrary
        # values at masked ranks as well.
        ranks = np.arange(self.k)
        fill = self._fill_value(dtype)
        np.copyto(work[..., : self.k], fill, where=ranks >= self.n_samples[..., None])
        np.copyto(work[..., self.k :], fill, where=ranks >= counts[..., None])
        return self._select_work(work)

    def _mask(self):
        ranks = np.arange(self.k).reshape((self.k,) + (1,) * self.n_samples.ndim)
        return ranks >= self.n_samples

    def update_batch(self, batch):
        """Select locally before combining with the retained state."""
        if np.ma.isMaskedArray(batch):
            raise TypeError("Decode masked arrays explicitly before updating.")
        batch = np.atleast_2d(np.asarray(batch))
        if batch.dtype.kind not in "biuf":
            raise TypeError("Top-k requires real numeric arrays.")
        data = self._reshape(batch)
        invalid = np.isnan(data) if data.dtype.kind == "f" else None
        if invalid is not None and not invalid.any():
            invalid = None
        if invalid is not None:
            if not self._ignore_nan:
                raise ValueError("BatchTopK rejects NaNs; use BatchNanTopK.")
            counts = np.asarray(data.shape[0] - np.count_nonzero(invalid, axis=0))
        else:
            counts = np.full(data.shape[1:], data.shape[0], dtype=np.int64)
        if self.values is not None and (counts.shape != self.n_samples.shape or batch.ndim != self._ndim):
            raise DifferentShapesError()
        # Local ranks must be sorted for the merge masks when a position has
        # fewer than k observations; fully populated tails need only one sort.
        sort = self.values is None or np.any(counts < self.k)
        selected = self._select(data, invalid, sort=sort)
        if self.values is not None:
            selected = self._merge_values(selected, counts)
            counts = self.n_samples + counts
        # Keep public state contiguous for rank reads and checkpoint writes;
        # only the temporary selection buffers need contiguous rank vectors.
        self.values, self.n_samples, self._ndim = np.ascontiguousarray(selected), counts, batch.ndim
        return self

    def __call__(self):
        if self.values is None:
            raise NoValidSamplesError()
        return np.ma.array(self.values.copy(), mask=self._mask())

    def rank(self, rank):
        """Read a one-based rank; unavailable observations are masked."""
        rank = _positive_integer(rank, "rank")
        if rank > self.k:
            raise ValueError("Rank exceeds retained capacity.")
        if self.values is None:
            raise NoValidSamplesError()
        return np.ma.array(self.values[rank - 1].copy(), mask=self.n_samples < rank)

    def quantile(self, q, method="linear"):
        """Read an exact scalar quantile; empty positions yield NaN.

        Raise ValueError if any nonempty position requires discarded ranks.
        Integer positions return their value directly; interpolation involving
        infinities follows IEEE arithmetic (and may produce NaN).
        """
        q = _quantile(q)
        if method != "linear":
            raise ValueError("Only method='linear' is supported.")
        if self.values is None:
            raise NoValidSamplesError()
        n = self.n_samples
        h = np.maximum(n - 1, 0) * q
        lo, hi = np.floor(h).astype(np.int64), np.ceil(h).astype(np.int64)
        a, b = (n - 1 - lo, n - 1 - hi) if self.largest else (lo, hi)
        if np.any((n > 0) & ((a >= self.k) | (b >= self.k))):
            raise ValueError("Insufficient retained capacity for an exact quantile.")
        x = np.take_along_axis(self.values, np.maximum(a, 0)[None], axis=0)[0].astype(np.float64)
        y = np.take_along_axis(self.values, np.maximum(b, 0)[None], axis=0)[0].astype(np.float64)
        weight = h - lo
        with np.errstate(invalid="ignore", over="ignore"):
            result = np.where(weight == 0, x, x + weight * (y - x))
        return np.where(n > 0, result, np.nan)

    def __add__(self, other):
        if type(self) is not type(other):
            raise DifferentStatsError()
        if self.axis != other.axis:
            raise DifferentAxisError()
        if self.k != other.k or self.largest != other.largest:
            raise ValueError("Capacity and direction must match.")
        result = type(self)(self.k, axis=self.axis, largest=self.largest)
        if self.values is None or other.values is None:
            source = other if self.values is None else self
            if source.values is not None:
                result.values = source.values.copy()
                result.n_samples = source.n_samples.copy()
                result._ndim = source._ndim
        else:
            if self.n_samples.shape != other.n_samples.shape or self._ndim != other._ndim:
                raise DifferentShapesError()
            result.values = np.ascontiguousarray(self._merge_values(other.values, other.n_samples))
            result.n_samples = self.n_samples + other.n_samples
            result._ndim = self._ndim
        return result
