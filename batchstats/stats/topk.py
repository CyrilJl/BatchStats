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

    def _select(self, values, valid):
        dtype = values.dtype
        if dtype.kind == "f":
            fill = -np.inf if self.largest else np.inf
        elif dtype.kind == "b":
            fill = not self.largest
        else:
            info = np.iinfo(dtype)
            fill = info.min if self.largest else info.max
        work = np.where(valid, values, fill)
        n = work.shape[0]
        if n > self.k:
            cut = n - self.k if self.largest else self.k - 1
            work.partition(cut, axis=0)
            work = work[-self.k :] if self.largest else work[: self.k]
        work = np.sort(work, axis=0)
        if self.largest:
            work = work[::-1]
        result = np.full((self.k, *work.shape[1:]), fill, dtype=dtype)
        result[: min(n, self.k)] = work
        return result

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
        valid = ~np.isnan(data)
        if not self._ignore_nan and not valid.all():
            raise ValueError("BatchTopK rejects NaNs; use BatchNanTopK.")
        counts = np.asarray(np.count_nonzero(valid, axis=0))
        if self.values is not None and (counts.shape != self.n_samples.shape or batch.ndim != self._ndim):
            raise DifferentShapesError()
        selected = self._select(data, valid)
        if self.values is not None:
            local_mask = np.arange(self.k).reshape((self.k,) + (1,) * counts.ndim) < counts
            selected = self._select(
                np.concatenate((self.values, selected)), np.concatenate((~self._mask(), local_mask))
            )
            counts = self.n_samples + counts
        self.values, self.n_samples, self._ndim = selected, counts, batch.ndim
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
        return self()[rank - 1]

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
            result.values = self._select(
                np.concatenate((self.values, other.values)), np.concatenate((~self._mask(), ~other._mask()))
            )
            result.n_samples = self.n_samples + other.n_samples
            result._ndim = self._ndim
        return result
