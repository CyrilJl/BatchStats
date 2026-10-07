"""Optional labelled accumulators. Install with ``pip install 'batchstats[xarray]'``.

These classes accept DataArrays or Datasets, reduce named dimensions and return
the same container type. Each call eagerly materializes only the supplied batch.
The NumPy API in :mod:`batchstats` does not import this module.
"""

from copy import deepcopy
from dataclasses import dataclass

import numpy as np

try:
    import xarray as xr
except ModuleNotFoundError as exc:
    if exc.name != "xarray":
        raise
    raise ImportError("xarray support requires: pip install 'batchstats[xarray]'") from exc

from . import nanstats, stats
from ._misc import NoValidSamplesError

__all__ = [
    "BatchCorr",
    "BatchCov",
    "BatchMax",
    "BatchMean",
    "BatchMin",
    "BatchNanMax",
    "BatchNanMean",
    "BatchNanMin",
    "BatchNanPeakToPeak",
    "BatchNanSum",
    "BatchNanTopK",
    "BatchPeakToPeak",
    "BatchStd",
    "BatchSum",
    "BatchTopK",
    "BatchVar",
    "BatchWeightedMean",
    "BatchWeightedSum",
]


@dataclass
class _Layout:
    reduced: tuple
    remaining: tuple
    shape: tuple
    coords: object
    name: object
    attrs: dict

    @classmethod
    def from_array(cls, array, dim, keep_attrs):
        reduced = tuple(array.dims) if dim is None else tuple(d for d in dim if d in array.dims)
        if dim is not None and not reduced:
            raise ValueError(f"Variable {array.name!r} has no reduction dimension; select the variables to reduce.")
        remaining = tuple(d for d in array.dims if d not in reduced)
        coords = array.coords.to_dataset().drop_vars(
            [name for name, c in array.coords.items() if any(d in reduced for d in c.dims)]
        )
        return cls(
            reduced,
            remaining,
            tuple(array.sizes[d] for d in remaining),
            coords,
            array.name,
            array.attrs if keep_attrs else {},
        )

    def check(self, other):
        if set(self.reduced) != set(other.reduced) or self.remaining != other.remaining or self.shape != other.shape:
            raise ValueError("Batch dimensions and non-reduced sizes must match.")
        if not self.coords.equals(other.coords):
            raise ValueError("Non-reduced coordinates must match exactly (including their order).")

    def values(self, array):
        values = array.transpose(*(self.reduced + self.remaining)).to_numpy()
        if values.dtype.kind not in "biufc":
            raise TypeError(f"Variable {array.name!r} must contain numeric data.")
        n = int(np.prod([array.sizes[d] for d in self.reduced], dtype=np.int64))
        # The NumPy kernels use atleast_2d. Supply an explicit trailing singleton
        # for scalar reductions so that a time series is not treated as one row.
        return values.reshape((n, *(self.shape or (1,))))

    def wrap(self, value, extra=None):
        dims = self.remaining
        shape = self.shape
        coords = self.coords.copy(deep=True).coords
        if extra is not None:
            name, labels = extra
            dims = (name, *dims)
            shape = (len(labels), *shape)
            coords[name] = labels
        # xarray represents missing ranks as NaN and may promote integer data.
        if np.ma.isMaskedArray(value) and not np.ma.getmaskarray(value).any():
            value = value.data
        return xr.DataArray(value.reshape(shape), dims=dims, coords=coords, name=self.name, attrs=deepcopy(self.attrs))


class _Reduction:
    """Shared streaming machinery; the public classes select a NumPy kernel.

    ``dim=None`` reduces all dimensions. A string or sequence of strings selects
    named dimensions. Dataset variables use their intersection with ``dim`` and
    must each have at least one selected dimension. ``keep_attrs=True`` copies
    attributes from the first batch. Coordinates depending on reduced dimensions
    are discarded. Other coordinates must remain identical across batches.
    """

    _kernel = None

    def __init__(self, dim=None, *, keep_attrs=False, **kwargs):
        if isinstance(dim, str):
            dim = (dim,)
        elif dim is not None:
            if not isinstance(dim, (tuple, list)) or not all(isinstance(d, str) for d in dim):
                raise TypeError("dim must be a dimension name, a sequence of names, or None.")
            dim = tuple(dim)
        if dim is not None and (not dim or len(set(dim)) != len(dim)):
            raise ValueError("dim must contain distinct dimension names and cannot be empty.")
        self.dim = dim
        self.keep_attrs = keep_attrs
        self._params = kwargs
        self._new_kernel()  # Validate statistic parameters before receiving data.
        self._layouts = {}
        self._accumulators = {}
        self._kind = None
        self._attrs = {}

    def _new_kernel(self):
        return self._kernel(axis=0, **self._params)

    def _prepare(self, batch, layouts=None):
        if not isinstance(batch, (xr.DataArray, xr.Dataset)):
            raise TypeError("Expected an xarray.DataArray or xarray.Dataset.")
        kind = xr.Dataset if isinstance(batch, xr.Dataset) else xr.DataArray
        if self._kind is not None and self._kind is not kind:
            raise TypeError("Cannot mix DataArray and Dataset batches.")
        if self.dim is not None and set(self.dim) - set(batch.dims):
            raise ValueError(f"Unknown reduction dimensions: {set(self.dim) - set(batch.dims)}")
        arrays = dict(batch.data_vars) if kind is xr.Dataset else {None: batch}
        if not arrays:
            raise ValueError("A Dataset must contain at least one data variable.")
        layouts = self._layouts if layouts is None else layouts
        if layouts and arrays.keys() != layouts.keys():
            raise ValueError("Dataset variables must match across batches.")
        prepared = {}
        for name, array in arrays.items():
            previous = layouts.get(name)
            if previous is not None:
                order = previous.reduced + previous.remaining
                if set(array.dims) != set(order):
                    raise ValueError("Batch dimensions must match.")
                array = array.transpose(*order)
            # A candidate is only a view used for validation. Snapshot metadata
            # once, below, when establishing the first batch's schema.
            layout = _Layout.from_array(array, self.dim, self.keep_attrs and previous is None)
            if previous is not None:
                previous.check(layout)
                layout = previous
            prepared[name] = (array, layout)
        if not layouts:
            retained = {c for _, layout in prepared.values() for c in layout.coords.variables}
            coords = batch.coords.to_dataset().drop_vars([c for c in batch.coords if c not in retained])
            # Deep copying lazy backend arrays alone does not detach them from
            # their file. Load only retained coordinates, then own their buffers.
            # The shallow copy prevents load() from changing the caller's cache.
            snapshot = coords.copy(deep=False).load().copy(deep=True)
            for _, layout in prepared.values():
                layout.coords = snapshot.drop_vars([c for c in snapshot.coords if c not in layout.coords])
                layout.attrs = deepcopy(layout.attrs)
        return kind, prepared

    def _update(self, batch, options):
        kind, prepared = self._prepare(batch)
        updated = {}
        # Stage changes so that a rejected Dataset cannot update some variables
        # while leaving others behind. Only bounded accumulator state is copied.
        for name, (array, layout) in prepared.items():
            accumulator = deepcopy(self._accumulators[name]) if name in self._accumulators else self._new_kernel()
            accumulator.update_batch(layout.values(array), **options(name, array, layout))
            updated[name] = accumulator
        self._commit(batch, kind, prepared, updated)
        return self

    def _commit(self, batch, kind, prepared, updated):
        if self._kind is None:
            self._attrs = deepcopy(batch.attrs) if self.keep_attrs and kind is xr.Dataset else {}
        self._kind = kind
        self._layouts = {name: layout for name, (_, layout) in prepared.items()}
        self._accumulators = updated

    def update_batch(self, batch, assume_valid=False):
        """Accumulate a labelled batch; return this accumulator."""
        return self._update(batch, lambda *_: {"assume_valid": assume_valid})

    def _collect(self, values):
        if self._kind is xr.Dataset:
            return xr.Dataset(values, attrs=deepcopy(self._attrs))
        return values[None]

    def _result(self, operation=None, *args, extra=None, **kwargs):
        if self._kind is None:
            raise NoValidSamplesError()
        values = {}
        for name, accumulator in self._accumulators.items():
            value = accumulator() if operation is None else getattr(accumulator, operation)(*args, **kwargs)
            values[name] = self._layouts[name].wrap(value, extra=extra)
        return self._collect(values)

    def __call__(self):
        """Return the statistic as a DataArray or Dataset."""
        return self._result()

    def to_state(self):
        """Export an independent, versioned checkpoint (JSON metadata + arrays)."""
        from ._xarray_state import to_state

        return to_state(self, copy=True)

    @classmethod
    def from_state(cls, state):
        """Restore and validate a labelled checkpoint, copying its arrays."""
        from ._xarray_state import from_state

        return from_state(cls, state)

    def save(self, path):
        """Atomically save an NPZ checkpoint to the exact path, without pickle."""
        from ._xarray_state import save

        save(self, path)

    @classmethod
    def load(cls, path):
        """Load an NPZ checkpoint and resume with update_batch or merging."""
        from ._xarray_state import load

        return load(cls, path)

    @property
    def n_samples(self):
        """Labelled counts (per variable for Datasets); None for weighted stats."""
        if self._kind is None:
            return None
        values = {}
        for name, accumulator in self._accumulators.items():
            if isinstance(accumulator, nanstats.BatchNanPeakToPeak):
                count = accumulator.batchnanmin.n_samples
            else:
                count = accumulator.n_samples
            if count is None:
                return None
            values[name] = (
                xr.DataArray(count) if np.ndim(count) == 0 else self._layouts[name].wrap(np.asarray(count).copy())
            )
            values[name].attrs = {}
        result = self._collect(values)
        result.attrs = {}
        return result

    def __add__(self, other):
        """Merge compatible accumulators without sharing mutable state."""
        if type(self) is not type(other):
            raise TypeError("Cannot merge different statistic types.")
        if self.dim != other.dim or self.keep_attrs != other.keep_attrs or self._params != other._params:
            raise ValueError("Reduction dimensions and statistic parameters must match.")
        if self._kind is None or other._kind is None:
            return deepcopy(other if self._kind is None else self)
        if self._kind is not other._kind or self._layouts.keys() != other._layouts.keys():
            raise ValueError("Container types and variables must match.")
        for name, layout in self._layouts.items():
            layout.check(other._layouts[name])
        # Copy metadata together (preserving internal coordinate sharing), but
        # do not copy kernels that will immediately be replaced by merged ones.
        result = object.__new__(type(self))
        result.__dict__.update(deepcopy({k: v for k, v in vars(self).items() if k != "_accumulators"}))
        result._accumulators = {
            name: _own_merged_kernel(accumulator + other._accumulators[name], accumulator, other._accumulators[name])
            for name, accumulator in self._accumulators.items()
        }
        return result


def _own_merged_kernel(merged, left, right):
    """Detach empty-operand aliases, including children of composite kernels.

    The NumPy kernels allocate new numeric buffers when combining populated
    states, but may return an existing kernel when the other operand is empty.
    """
    if merged is left or merged is right:
        return deepcopy(merged)
    for name, child in vars(merged).items():
        if hasattr(child, "update_batch"):
            setattr(merged, name, _own_merged_kernel(child, getattr(left, name), getattr(right, name)))
    return merged


class BatchSum(_Reduction):
    """Streaming sum over named dimensions."""

    _kernel = stats.BatchSum


class BatchMean(_Reduction):
    """Streaming mean over named dimensions."""

    _kernel = stats.BatchMean


class BatchMin(_Reduction):
    """Streaming minimum over named dimensions."""

    _kernel = stats.BatchMin


class BatchMax(_Reduction):
    """Streaming maximum over named dimensions."""

    _kernel = stats.BatchMax


class BatchPeakToPeak(_Reduction):
    """Streaming range over named dimensions."""

    _kernel = stats.BatchPeakToPeak


class BatchVar(_Reduction):
    """Streaming variance over named dimensions, with optional ddof."""

    _kernel = stats.BatchVar

    def __init__(self, dim=None, ddof=0, *, keep_attrs=False):
        super().__init__(dim, keep_attrs=keep_attrs, ddof=ddof)


class BatchStd(BatchVar):
    """Streaming standard deviation over named dimensions."""

    _kernel = stats.BatchStd


class _NanReduction(_Reduction):
    def update_batch(self, batch):
        """Accumulate a labelled batch, ignoring NaNs independently per cell."""
        return self._update(batch, lambda *_: {})


class BatchNanSum(_NanReduction):
    """Streaming sum ignoring NaNs per cell."""

    _kernel = nanstats.BatchNanSum


class BatchNanMean(_NanReduction):
    """Streaming mean ignoring NaNs per cell."""

    _kernel = nanstats.BatchNanMean


class BatchNanMin(_NanReduction):
    """Streaming minimum ignoring NaNs per cell."""

    _kernel = nanstats.BatchNanMin


class BatchNanMax(_NanReduction):
    """Streaming maximum ignoring NaNs per cell."""

    _kernel = nanstats.BatchNanMax


class BatchNanPeakToPeak(_NanReduction):
    """Streaming range ignoring NaNs per cell."""

    _kernel = nanstats.BatchNanPeakToPeak


class BatchWeightedSum(_Reduction):
    """Streaming weighted sum; weights broadcast by dimension name."""

    _kernel = stats.BatchWeightedSum

    def update_batch(self, batch, weights):
        """Use scalar, DataArray, or matching Dataset weights."""
        if isinstance(weights, xr.Dataset):
            if not isinstance(batch, xr.Dataset) or set(weights.data_vars) != set(batch.data_vars):
                raise ValueError("Weight Dataset variables must match the batch Dataset.")
        elif not isinstance(weights, xr.DataArray) and not np.isscalar(weights):
            raise TypeError("weights must be a scalar, DataArray, or Dataset.")

        def options(name, array, layout):
            weight = weights[name] if isinstance(weights, xr.Dataset) else weights
            if not isinstance(weight, xr.DataArray):
                weight = xr.DataArray(weight)
            if set(weight.dims) - set(array.dims):
                raise ValueError("Weight dimensions must be a subset of the variable dimensions.")
            xr.align(array, weight, join="exact", copy=False)
            for coord in set(weight.coords) & set(array.coords):
                expected = array.coords[coord].variable
                actual = weight.coords[coord].variable
                if set(actual.dims) != set(expected.dims) or not actual.transpose(*expected.dims).equals(expected):
                    raise ValueError("Weight coordinates must match the batch exactly.")
            weight = weight.broadcast_like(array)
            return {"weights": layout.values(weight)}

        return self._update(batch, options)


class BatchWeightedMean(BatchWeightedSum):
    """Streaming weighted mean; weights broadcast by dimension name."""

    _kernel = stats.BatchWeightedMean


class BatchTopK(_NanReduction):
    """Exact extreme ranks with a leading rank dimension; NaNs are rejected."""

    _kernel = stats.BatchTopK

    def __init__(self, k, dim=None, largest=True, *, keep_attrs=False, rank_dim="rank"):
        if not isinstance(rank_dim, str) or not rank_dim:
            raise TypeError("rank_dim must be a nonempty string.")
        self.rank_dim = rank_dim
        super().__init__(dim, keep_attrs=keep_attrs, k=k, largest=largest)

    def _prepare(self, batch, layouts=None):
        kind, prepared = super()._prepare(batch, layouts)
        if (
            self.rank_dim in batch.dims
            or self.rank_dim in batch.coords
            or (kind is xr.Dataset and self.rank_dim in batch.data_vars)
        ):
            raise ValueError("rank_dim conflicts with the batch; choose another rank_dim.")
        return kind, prepared

    def __call__(self):
        return self._result(extra=(self.rank_dim, np.arange(1, self._params["k"] + 1)))

    def rank(self, rank):
        """Read a one-based rank; unavailable values become NaN."""
        return self._result("rank", rank)

    def quantile(self, q, method="linear"):
        """Read an exact tail quantile when the retained capacity is sufficient."""
        return self._result("quantile", q, method=method)

    def __add__(self, other):
        if isinstance(other, BatchTopK) and self.rank_dim != other.rank_dim:
            raise ValueError("Rank dimension names must match.")
        return super().__add__(other)


class BatchNanTopK(BatchTopK):
    """Exact extreme ranks, ignoring NaNs independently per cell."""

    _kernel = nanstats.BatchNanTopK


class BatchCov(_Reduction):
    """Streaming covariance matrices for each variable.

    Non-reduced dimensions describe features. Output dimensions from the second
    input (or a second copy of the first input) receive a ``_2`` suffix.
    """

    _kernel = stats.BatchCov

    def __init__(self, dim=None, ddof=0, *, keep_attrs=False):
        self._right_layouts = {}
        self._paired = None
        super().__init__(dim, keep_attrs=keep_attrs, ddof=ddof)

    def _new_kernel(self):
        return self._kernel(**self._params)

    def update_batch(self, batch, batch2=None, assume_valid=False):
        paired = batch2 is not None
        if self._paired is not None and self._paired != paired:
            raise ValueError("Cannot mix updates with and without batch2.")
        kind, left = self._prepare(batch)
        right_kind, right = (kind, left) if batch2 is None else self._prepare(batch2, self._right_layouts)
        if kind is not right_kind or left.keys() != right.keys():
            raise ValueError("Paired batches must have matching container types and variables.")
        generated_names = {
            f"{coord}_2"
            for _, layout in right.values()
            for coord in set(layout.coords.variables) | set(layout.remaining)
        }
        existing_names = set(batch.dims) | set(batch.coords)
        if kind is xr.Dataset:
            existing_names |= set(batch.data_vars)
        if generated_names & existing_names:
            raise ValueError("Covariance output names collide with the '_2' suffix; rename the input coordinates.")
        updated = {}
        for name, (array, layout) in left.items():
            array2, layout2 = right[name]
            if layout.reduced != layout2.reduced:
                raise ValueError("Paired reduction dimensions must match.")
            for dim in layout.reduced:
                if array.sizes[dim] != array2.sizes[dim]:
                    raise ValueError("Paired sample sizes must match.")

            def sample_coords(a, dims):
                return a.coords.to_dataset().drop_vars(
                    [c for c, v in a.coords.items() if not set(v.dims).issubset(dims)]
                )

            if not sample_coords(array, layout.reduced).equals(sample_coords(array2, layout2.reduced)):
                raise ValueError("Paired sample coordinates must match exactly.")
            self._matrix_layout(layout, layout2)  # Validate generated names before updating.
            values = layout.values(array)
            values = values.reshape((values.shape[0], int(np.prod(layout.shape))))
            values2 = None
            if paired:
                values2 = layout2.values(array2)
                values2 = values2.reshape((values2.shape[0], int(np.prod(layout2.shape))))
            accumulator = deepcopy(self._accumulators[name]) if name in self._accumulators else self._new_kernel()
            accumulator.update_batch(values, values2, assume_valid=assume_valid)
            updated[name] = accumulator
        self._commit(batch, kind, left, updated)
        self._right_layouts = {name: layout for name, (_, layout) in right.items()}
        self._paired = paired
        return self

    @staticmethod
    def _matrix_layout(left, right):
        rename = {name: f"{name}_2" for name in set(right.coords.variables) | set(right.remaining)}
        if set(rename.values()) & (set(left.coords.variables) | set(left.remaining)):
            raise ValueError("Covariance output names collide with the '_2' suffix; rename the input coordinates.")
        coordinate_rename = {
            name: target
            for name, target in rename.items()
            if name in right.coords.variables or name in right.coords.dims
        }
        coords = xr.merge([left.coords, right.coords.rename(coordinate_rename)], join="exact")
        return _Layout(
            (),
            left.remaining + tuple(rename[d] for d in right.remaining),
            left.shape + right.shape,
            coords,
            left.name,
            left.attrs,
        )

    def __call__(self):
        if self._kind is None:
            raise NoValidSamplesError()
        return self._collect(
            {
                name: self._matrix_layout(self._layouts[name], self._right_layouts[name]).wrap(accumulator())
                for name, accumulator in self._accumulators.items()
            }
        )

    def __add__(self, other):
        if type(self) is type(other) and self._kind is not None and other._kind is not None:
            if self._paired != other._paired:
                raise ValueError("Paired batch modes must match.")
            if self._right_layouts.keys() != other._right_layouts.keys():
                raise ValueError("Dataset variables must match.")
            for name, layout in self._right_layouts.items():
                layout.check(other._right_layouts[name])
        return super().__add__(other)


class BatchCorr(BatchCov):
    """Streaming correlation matrices with labelled feature dimensions."""

    _kernel = stats.BatchCorr
