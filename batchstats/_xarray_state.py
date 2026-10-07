"""Version 1 labelled checkpoints: explicit schemas, JSON and non-object arrays.

Imported only by the optional xarray checkpoint API. No Python class is imported
or instantiated based on names in a checkpoint; kernels come from fresh public
accumulators and their existing child kernels.
"""

import base64
import datetime as dt
import json
import os
import tempfile
from pathlib import Path
from zipfile import BadZipFile

import numpy as np
import pandas as pd
import xarray as xr
from xarray.core.indexes import PandasIndex, PandasMultiIndex

from . import nanstats, stats
from .base import BatchNanStat


class _Codec:
    """Encode metadata without coercing tuples, NumPy dtypes or object labels."""

    def __init__(self, arrays=None, copy=True):
        self.arrays = {} if arrays is None else arrays
        self.used = set()
        self.copy = copy

    def pack(self, value):
        if isinstance(value, np.dtype) and value.kind in "biufcUSmMO":
            return {"tag": "dtype", "value": value.str}
        if isinstance(value, (np.ndarray, np.generic)):
            scalar = isinstance(value, np.generic)
            array = np.asarray(value)
            if array.dtype.kind == "O":
                return {"tag": "object", "shape": list(array.shape), "items": [self.pack(v) for v in array.flat]}
            if array.dtype.kind not in "biufcUSmM":
                raise TypeError(f"Unsupported checkpoint array dtype: {array.dtype}")
            key = f"a{len(self.arrays)}"
            self.arrays[key] = array.copy() if self.copy else array
            return {"tag": "array", "key": key, "dtype": array.dtype.str, "shape": list(array.shape), "scalar": scalar}
        if value is None or type(value) in (bool, int, str):
            return value
        if type(value) is float:
            return {"tag": "float", "value": value.hex()}
        if type(value) is complex:
            return {"tag": "complex", "real": value.real.hex(), "imag": value.imag.hex()}
        if type(value) is bytes:
            return {"tag": "bytes", "value": base64.b64encode(value).decode("ascii")}
        if value is pd.NaT:
            return {"tag": "nat"}
        if value is pd.NA:
            return {"tag": "na"}
        if isinstance(value, pd.Timestamp):
            return {"tag": "timestamp", "value": value.isoformat()}
        if isinstance(value, pd.Timedelta):
            return {"tag": "pd_timedelta", "value": value.isoformat()}
        if type(value) in (dt.datetime, dt.date):
            return {"tag": type(value).__name__, "value": value.isoformat()}
        if type(value) is dt.timedelta:
            return {
                "tag": "timedelta",
                "days": value.days,
                "seconds": value.seconds,
                "microseconds": value.microseconds,
            }
        if type(value) in (list, tuple):
            return {"tag": type(value).__name__, "items": [self.pack(v) for v in value]}
        if type(value) is dict:
            return {"tag": "dict", "items": [[self.pack(k), self.pack(v)] for k, v in value.items()]}
        raise TypeError(f"Unsupported checkpoint metadata type: {type(value).__name__}. Convert it to standard values.")

    def unpack(self, value):
        if value is None or type(value) in (bool, int, str):
            return value
        tag = value["tag"]
        if tag == "array":
            array = self.arrays[value["key"]]
            if not isinstance(array, np.ndarray) or array.dtype.kind not in "biufcUSmM":
                raise ValueError("Invalid checkpoint array dtype.")
            if list(array.shape) != value["shape"] or array.dtype.str != value["dtype"]:
                raise ValueError("Checkpoint array shape or dtype mismatch.")
            self.used.add(value["key"])
            if type(value["scalar"]) is not bool or (value["scalar"] and array.ndim != 0):
                raise ValueError("Invalid scalar array.")
            return array[()] if value["scalar"] else array.copy()
        if tag == "object":
            shape = _shape(value["shape"])
            if int(np.prod(shape, dtype=object)) != len(value["items"]):
                raise ValueError("Invalid object array shape.")
            array = np.empty(len(value["items"]), dtype=object)
            for i, item in enumerate(value["items"]):
                array[i] = self.unpack(item)
            return array.reshape(shape)
        if tag == "float":
            return float.fromhex(value["value"])
        if tag == "complex":
            return complex(float.fromhex(value["real"]), float.fromhex(value["imag"]))
        if tag == "bytes":
            return base64.b64decode(value["value"], validate=True)
        if tag == "dtype":
            dtype = np.dtype(value["value"])
            if dtype.kind not in "biufcUSmMO":
                raise ValueError("Unsupported metadata dtype.")
            return dtype
        if tag == "nat":
            return pd.NaT
        if tag == "na":
            return pd.NA
        if tag in ("timestamp", "pd_timedelta", "datetime", "date"):
            constructor = {
                "timestamp": pd.Timestamp,
                "pd_timedelta": pd.Timedelta,
                "datetime": dt.datetime.fromisoformat,
                "date": dt.date.fromisoformat,
            }[tag]
            return constructor(value["value"])
        if tag == "timedelta":
            return dt.timedelta(days=value["days"], seconds=value["seconds"], microseconds=value["microseconds"])
        if tag in ("list", "tuple"):
            items = [self.unpack(v) for v in value["items"]]
            return tuple(items) if tag == "tuple" else items
        if tag == "dict":
            result = {}
            for key, item in value["items"]:
                key = self.unpack(key)
                if key in result:
                    raise ValueError("Duplicate metadata key.")
                result[key] = self.unpack(item)
            return result
        raise ValueError(f"Unknown checkpoint metadata tag: {tag!r}")


def _shape(value):
    if not isinstance(value, list) or any(type(n) is not int or n < 0 for n in value):
        raise ValueError("Invalid checkpoint shape.")
    return tuple(value)


def _pack_coords(coords, codec):
    indexes, seen = [], set()
    for name, index in coords.xindexes.items():
        if id(index) in seen:
            continue
        seen.add(id(index))
        if type(index) not in (PandasIndex, PandasMultiIndex):
            raise TypeError(f"Unsupported checkpoint coordinate index: {type(index).__name__}")
        pandas_index = index.to_pandas_index()
        for candidate in [*pandas_index.levels] if isinstance(pandas_index, pd.MultiIndex) else [pandas_index]:
            if isinstance(candidate, (pd.CategoricalIndex, pd.PeriodIndex, pd.IntervalIndex)) or (
                isinstance(candidate, pd.DatetimeIndex) and candidate.tz is not None
            ):
                raise TypeError(
                    "Checkpoint indexes support numeric/string labels, timezone-naive dates and MultiIndexes."
                )
        if type(index) is PandasMultiIndex:
            indexes.append(
                {
                    "kind": "multi",
                    "dim": codec.pack(index.dim),
                    "names": codec.pack(tuple(pandas_index.names)),
                    "levels": [codec.pack(level.to_numpy()) for level in pandas_index.levels],
                    "codes": [codec.pack(np.asarray(code)) for code in pandas_index.codes],
                }
            )
        else:
            indexes.append({"kind": "single", "name": codec.pack(name)})
    variables = [
        {
            "name": codec.pack(name),
            "dims": codec.pack(tuple(var.dims)),
            "data": codec.pack(var.to_numpy()),
            "attrs": codec.pack(var.attrs),
            "encoding": codec.pack(var.encoding),
        }
        for name, var in coords.coords.items()
    ]
    return {"variables": variables, "indexes": indexes}


def _unpack_coords(value, codec):
    variables = {}
    for item in value["variables"]:
        name = codec.unpack(item["name"])
        if name in variables:
            raise ValueError("Duplicate coordinate name.")
        variable = xr.Variable(
            codec.unpack(item["dims"]), codec.unpack(item["data"]), attrs=codec.unpack(item["attrs"])
        )
        variable.encoding = codec.unpack(item["encoding"])
        variables[name] = variable
    # Construct without indexes first, including coordinates deliberately dropped
    # from the input's indexes; reinstall exactly the serialized index groups.
    multi_names = set()
    for index in value["indexes"]:
        if index["kind"] == "multi":
            multi_names.add(codec.unpack(index["dim"]))
            multi_names.update(codec.unpack(index["names"]))
    result = xr.Dataset(coords={name: var for name, var in variables.items() if name not in multi_names})
    if result.xindexes:
        result = result.drop_indexes(list(result.xindexes))
    for index in value["indexes"]:
        if index["kind"] == "single":
            result = result.set_xindex(codec.unpack(index["name"]))
        elif index["kind"] == "multi":
            dim = codec.unpack(index["dim"])
            names = codec.unpack(index["names"])
            pandas_index = pd.MultiIndex(
                levels=[codec.unpack(v) for v in index["levels"]],
                codes=[codec.unpack(v) for v in index["codes"]],
                names=names,
            )
            restored_index = PandasMultiIndex(
                pandas_index, dim, level_coords_dtype={name: variables[name].dtype for name in names}
            )
            index_variables = restored_index.create_variables(variables)
            for name in (dim, *names):
                matches = (
                    pd.MultiIndex.from_tuples(variables[name].values, names=names).equals(pandas_index)
                    if name == dim
                    else variables[name].equals(index_variables[name])
                )
                if not matches:
                    raise ValueError("MultiIndex levels disagree with coordinate values.")
            indexes = dict.fromkeys(index_variables, restored_index)
            if hasattr(xr, "Coordinates"):
                result = result.assign_coords(xr.Coordinates(index_variables, indexes=indexes))
            else:  # xarray 2023.1 predates the public Coordinates constructor.
                result = result._replace(
                    variables={**result.variables, **index_variables},
                    coord_names=set(result.coords) | set(index_variables),
                    dims={**result.sizes, dim: len(pandas_index)},
                    indexes={**result.xindexes, **indexes},
                )
        else:
            raise ValueError("Unknown checkpoint index kind.")
    if set(result.coords) != set(variables):
        raise ValueError("Checkpoint indexes and coordinates disagree.")
    return result


def _pack_layout(layout, codec):
    return {
        "reduced": codec.pack(layout.reduced),
        "remaining": codec.pack(layout.remaining),
        "shape": list(layout.shape),
        "coords": _pack_coords(layout.coords, codec),
        "name": codec.pack(layout.name),
        "attrs": codec.pack(layout.attrs),
    }


def _unpack_layout(value, codec, result):
    from .xarray import _Layout

    reduced, remaining = codec.unpack(value["reduced"]), codec.unpack(value["remaining"])
    shape = _shape(value["shape"])
    if not isinstance(reduced, tuple) or not isinstance(remaining, tuple):
        raise ValueError("Invalid checkpoint dimensions.")
    if len(set(reduced + remaining)) != len(reduced + remaining) or len(shape) != len(remaining):
        raise ValueError("Checkpoint dimensions are duplicated or inconsistent.")
    if result.dim is None:
        if remaining:
            raise ValueError("All-dimension reductions cannot have remaining dimensions.")
    elif not reduced or reduced != tuple(d for d in result.dim if d in reduced + remaining):
        raise ValueError("Checkpoint reduction dimensions disagree with the parameters.")
    coords = _unpack_coords(value["coords"], codec)
    sizes = dict(zip(remaining, shape, strict=True))
    if any(d not in sizes or n != sizes[d] for d, n in coords.sizes.items()):
        raise ValueError("Checkpoint coordinates disagree with the output shape.")
    attrs = codec.unpack(value["attrs"])
    if not isinstance(attrs, dict) or (not result.keep_attrs and attrs):
        raise ValueError("Invalid checkpoint attributes.")
    return _Layout(reduced, remaining, shape, coords, codec.unpack(value["name"]), attrs)


def _pack_kernel(kernel, codec):
    fields = {}
    for name, value in vars(kernel).items():
        if hasattr(value, "update_batch"):
            fields[name] = _pack_kernel(value, codec)
        else:
            fields[name] = codec.pack(value)
    return {"type": type(kernel).__name__, "fields": fields}


def _restore_kernel(value, kernel, codec, shape, right_shape=None):
    if value["type"] != type(kernel).__name__ or set(value["fields"]) != set(vars(kernel)):
        raise ValueError("Checkpoint kernel type or fields do not match.")
    for name, default in list(vars(kernel).items()):
        field = value["fields"][name]
        if hasattr(default, "update_batch"):
            child_shape = right_shape if name in ("mean2", "var2") else shape
            _restore_kernel(field, default, codec, child_shape, right_shape if name == "cov" else None)
        else:
            restored = codec.unpack(field)
            if name in ("axis", "ddof", "k", "largest", "_is_list_mode"):
                if type(restored) is not type(default) or restored != default:
                    raise ValueError(f"Checkpoint kernel parameter {name!r} does not match.")
            setattr(kernel, name, restored)
    _validate_kernel(kernel, shape, right_shape)
    return kernel


def _validate_kernel(kernel, shape, right_shape):
    count = kernel.n_samples
    weighted = isinstance(kernel, (stats.BatchWeightedSum, stats.BatchWeightedMean))
    per_cell = isinstance(kernel, (BatchNanStat, stats.BatchTopK))
    if (
        not weighted
        and not per_cell
        and (not isinstance(count, (int, np.integer)) or isinstance(count, (bool, np.bool_)))
    ):
        raise ValueError("Ordinary checkpoint sample counts must be integers.")
    if per_cell and count is not None and not isinstance(count, np.ndarray):
        raise ValueError("Per-cell checkpoint sample counts must be arrays.")
    if count is not None:
        counts = np.asarray(count)
        if counts.dtype.kind not in "iu" or np.any(counts < 0):
            raise ValueError("Invalid checkpoint sample counts.")
        expected_count_shape = shape if isinstance(kernel, (BatchNanStat, stats.BatchTopK)) else ()
        if counts.shape != expected_count_shape:
            raise ValueError("Checkpoint sample counts have the wrong shape.")
    for name, value in vars(kernel).items():
        if not isinstance(value, np.ndarray) or name == "n_samples":
            continue
        expected = (1, *shape)
        if isinstance(kernel, nanstats.BatchNanSum):
            expected = shape
        elif isinstance(kernel, stats.BatchTopK):
            expected = (kernel.k, *shape)
        elif isinstance(kernel, stats.BatchCov):
            expected = (shape[0], right_shape[0])
        if value.dtype.kind not in "biufc" or value.shape != expected:
            raise ValueError(f"Invalid checkpoint kernel array {name!r}.")
    if isinstance(kernel, stats.BatchTopK):
        type(kernel).from_state(kernel.to_state())  # Includes rank ordering and count validation.
        if kernel.values is not None and kernel._ndim != len(shape) + 1:
            raise ValueError("Invalid top-k checkpoint dimensions.")
    if isinstance(kernel, stats.BatchWeightedSum):
        if kernel._weights_pattern not in (None, shape) or count is not None:
            raise ValueError("Invalid checkpoint weight pattern.")
    elif isinstance(kernel, stats.BatchWeightedMean):
        if count is not None:
            raise ValueError("Weighted checkpoints cannot have sample counts.")
    elif isinstance(kernel, stats.BatchCorr):
        if kernel._is_1d is not None and type(kernel._is_1d) is not bool:
            raise ValueError("Invalid correlation mode.")
    # Validate count relationships for composite kernels. NanPeakToPeak keeps
    # counts only in its two children; weighted kernels do not use counts.
    children = [v for v in vars(kernel).values() if hasattr(v, "update_batch")]
    for child in children:
        if isinstance(kernel, stats.BatchCorr) and kernel._is_1d and child in (kernel.var1, kernel.var2):
            continue
        if isinstance(kernel, nanstats.BatchNanPeakToPeak):
            reference = kernel.batchnanmin.n_samples
        else:
            reference = count
        if not np.array_equal(reference, child.n_samples):
            raise ValueError("Inconsistent checkpoint sample counts in child kernels.")
    for field in ("sum", "mean", "min", "max", "var", "cov", "values"):
        if field not in vars(kernel) or hasattr(getattr(kernel, field), "update_batch"):
            continue
        array = getattr(kernel, field)
        if array is not None and not isinstance(array, np.ndarray):
            raise ValueError("Checkpoint statistic values must be arrays.")
        if array is not None and count is None and not weighted:
            raise ValueError("Checkpoint statistic values have no sample counts.")
        if array is None and count is not None and np.any(np.asarray(count) > 0):
            raise ValueError("Checkpoint has sample counts but no statistic values.")


def to_state(accumulator, copy=True):
    from .xarray import BatchCov, BatchTopK

    codec = _Codec(copy=copy)
    variables = []
    for name, layout in accumulator._layouts.items():
        variables.append(
            {
                "name": codec.pack(name),
                "layout": _pack_layout(layout, codec),
                "right_layout": _pack_layout(accumulator._right_layouts[name], codec)
                if isinstance(accumulator, BatchCov)
                else None,
                "kernel": _pack_kernel(accumulator._accumulators[name], codec),
            }
        )
    metadata = {
        "format": "batchstats.xarray",
        "version": 1,
        "type": type(accumulator).__name__,
        "dim": codec.pack(accumulator.dim),
        "keep_attrs": accumulator.keep_attrs,
        "params": codec.pack(accumulator._params),
        "rank_dim": accumulator.rank_dim if isinstance(accumulator, BatchTopK) else None,
        "paired": accumulator._paired if isinstance(accumulator, BatchCov) else None,
        "kind": None if accumulator._kind is None else accumulator._kind.__name__,
        "attrs": codec.pack(accumulator._attrs),
        "variables": variables,
    }
    return {"metadata": metadata, "arrays": codec.arrays}


def from_state(cls, state):
    from . import xarray as bx

    try:
        meta, arrays = state["metadata"], state["arrays"]
        if (
            cls.__name__ not in bx.__all__
            or getattr(bx, cls.__name__) is not cls
            or meta["format"] != "batchstats.xarray"
            or type(meta["version"]) is not int
            or meta["version"] != 1
            or meta["type"] != cls.__name__
        ):
            raise ValueError("Unknown checkpoint format, version or statistic type.")
        if type(meta["keep_attrs"]) is not bool or meta["kind"] not in (None, "DataArray", "Dataset"):
            raise ValueError("Invalid checkpoint container or attribute policy.")
        codec = _Codec(arrays)
        params = codec.unpack(meta["params"])
        kwargs = {"rank_dim": meta["rank_dim"]} if issubclass(cls, bx.BatchTopK) else {}
        result = cls(dim=codec.unpack(meta["dim"]), keep_attrs=meta["keep_attrs"], **params, **kwargs)
        if result._params != params or (not isinstance(result, bx.BatchTopK) and meta["rank_dim"] is not None):
            raise ValueError("Invalid checkpoint parameters.")
        result._attrs = codec.unpack(meta["attrs"])
        if not isinstance(result._attrs, dict) or (result._attrs and not result.keep_attrs):
            raise ValueError("Invalid checkpoint Dataset attributes.")
        result._kind = {None: None, "DataArray": xr.DataArray, "Dataset": xr.Dataset}[meta["kind"]]
        if (result._kind is None) != (not meta["variables"]):
            raise ValueError("Checkpoint container and variables disagree.")
        if isinstance(result, bx.BatchCov):
            if (meta["paired"] is None) != (result._kind is None) or (
                meta["paired"] is not None and type(meta["paired"]) is not bool
            ):
                raise ValueError("Invalid checkpoint paired mode.")
            result._paired = meta["paired"]
        elif meta["paired"] is not None:
            raise ValueError("Unexpected paired mode.")
        for variable in meta["variables"]:
            name = codec.unpack(variable["name"])
            if name in result._layouts:
                raise ValueError("Duplicate checkpoint variable.")
            layout = _unpack_layout(variable["layout"], codec, result)
            if result._kind is xr.Dataset and layout.name != name:
                raise ValueError("Dataset variable and layout names disagree.")
            result._layouts[name] = layout
            shape, right_shape = layout.shape or (1,), None
            if isinstance(result, bx.BatchCov):
                right = _unpack_layout(variable["right_layout"], codec, result)
                if layout.reduced != right.reduced:
                    raise ValueError("Paired checkpoint dimensions disagree.")
                if not result._paired:
                    layout.check(right)
                result._matrix_layout(layout, right)
                result._right_layouts[name] = right
                shape, right_shape = (int(np.prod(layout.shape)),), (int(np.prod(right.shape)),)
            elif variable["right_layout"] is not None:
                raise ValueError("Unexpected paired layout.")
            result._accumulators[name] = _restore_kernel(
                variable["kernel"], result._new_kernel(), codec, shape, right_shape
            )
            if isinstance(result, bx.BatchCorr) and result._accumulators[name]._is_1d is not (not result._paired):
                raise ValueError("Correlation checkpoint mode disagrees with the paired layout.")
        if result._kind is xr.DataArray and set(result._layouts) != {None}:
            raise ValueError("A DataArray checkpoint must contain one unnamed slot.")
        if codec.used != set(arrays):
            raise ValueError("Unused checkpoint arrays.")
        sizes = {}
        for layout in result._layouts.values():
            for dim, size in zip(layout.remaining, layout.shape, strict=True):
                if dim in sizes and sizes[dim] != size:
                    raise ValueError("Dataset variables have inconsistent dimension sizes.")
                sizes[dim] = size
            if isinstance(result, bx.BatchTopK) and result.rank_dim in (
                set(layout.reduced + layout.remaining) | set(layout.coords.variables) | set(result._layouts)
            ):
                raise ValueError("Checkpoint rank dimension conflicts with input names.")
        if result._layouts:
            xr.merge([layout.coords for layout in result._layouts.values()], join="exact", compat="equals")
        return result
    except (KeyError, TypeError, AttributeError, IndexError, OverflowError, RecursionError) as exc:
        raise ValueError("Malformed xarray checkpoint.") from exc


def save(accumulator, path):
    """Prepare before opening any file; replace destination only after success."""
    state = to_state(accumulator, copy=False)
    metadata = np.asarray(json.dumps(state["metadata"], allow_nan=False))
    destination = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            np.savez(stream, metadata=metadata, **state["arrays"])
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load(cls, path):
    try:
        with np.load(path, allow_pickle=False) as archive:
            if len(archive.files) != len(set(archive.files)):
                raise ValueError("Duplicate checkpoint archive members.")
            metadata = archive["metadata"]
            if metadata.ndim != 0 or metadata.dtype.kind != "U":
                raise ValueError("Invalid checkpoint JSON metadata.")
            state = {
                "metadata": json.loads(str(metadata)),
                "arrays": {name: archive[name] for name in archive.files if name != "metadata"},
            }
        return from_state(cls, state)
    except (BadZipFile, EOFError, KeyError, TypeError, AttributeError) as exc:
        raise ValueError("Malformed xarray checkpoint file.") from exc
