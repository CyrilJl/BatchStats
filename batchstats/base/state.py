"""Versioned metadata and NumPy arrays, without pickle."""

import json

import numpy as np


class StateMixin:
    """State/checkpoint API for NaN sums/means and top-k accumulators."""

    def to_state(self):
        """Return independent arrays plus JSON-compatible metadata."""
        from ..nanstats.nan_mean import BatchNanMean
        from ..stats.topk import BatchTopK

        target = self.sum if isinstance(self, BatchNanMean) else self
        field = "values" if isinstance(self, BatchTopK) else "sum"
        value = getattr(target, field)
        axis = self.axis
        params = {"axis": [int(ax) for ax in axis] if isinstance(axis, tuple) else None if axis is None else int(axis)}
        if isinstance(self, BatchTopK):
            params.update(k=self.k, largest=self.largest)
        arrays = {} if value is None else {field: value.copy(), "n_samples": target.n_samples.copy()}
        metadata = {
            "version": 1,
            "type": type(self).__name__,
            "params": params,
            "arrays": {key: {"shape": list(a.shape), "dtype": a.dtype.str} for key, a in arrays.items()},
        }
        if isinstance(self, BatchTopK):
            metadata["ndim"] = self._ndim
        return {"metadata": metadata, "arrays": arrays}

    @classmethod
    def from_state(cls, state):
        """Restore a validated state and copy all supplied arrays."""
        from ..nanstats.nan_mean import BatchNanMean
        from ..stats.topk import BatchTopK

        try:
            meta, arrays = state["metadata"], state["arrays"]
            if type(meta["version"]) is not int or meta["version"] != 1 or meta["type"] != cls.__name__:
                raise ValueError("Unknown state version or statistic type.")
            params = dict(meta["params"])
            if isinstance(params["axis"], list):
                params["axis"] = tuple(params["axis"])
            axis = params["axis"]
            if axis is not None:
                axes = axis if isinstance(axis, tuple) else (axis,)
                if any(type(ax) is not int for ax in axes) or len(set(axes)) != len(axes):
                    raise ValueError("Invalid state axes.")
            result = cls(**params)
            field = "values" if isinstance(result, BatchTopK) else "sum"
            if set(arrays) != set(meta["arrays"]) or (arrays and set(arrays) != {field, "n_samples"}):
                raise ValueError("Invalid state arrays.")
            for key, a in arrays.items():
                spec = meta["arrays"][key]
                if not isinstance(a, np.ndarray) or a.dtype.kind not in "biufc":
                    raise ValueError("Invalid numeric state array.")
                if list(a.shape) != spec["shape"] or a.dtype.str != spec["dtype"]:
                    raise ValueError("State shape or dtype mismatch.")
            if arrays:
                count, value = arrays["n_samples"], arrays[field]
                if count.dtype.kind not in "iu" or np.any(count < 0):
                    raise ValueError("Invalid sample counts.")
                expected = (result.k, *count.shape) if isinstance(result, BatchTopK) else count.shape
                if value.shape != expected:
                    raise ValueError("Inconsistent state shapes.")
                if isinstance(result, BatchTopK):
                    ndim = meta["ndim"]
                    if type(ndim) is not int or ndim < 2 or value.dtype.kind not in "biuf":
                        raise ValueError("Invalid top-k dimensions or dtype.")
                    result._ndim = ndim
                    dummy = np.empty((0,) * ndim)
                    remaining = result._reshape(dummy).ndim - 1
                    if remaining != count.ndim:
                        raise ValueError("State axes and shape disagree.")
                    mask = np.arange(result.k).reshape((result.k,) + (1,) * count.ndim) < count
                    if np.any(np.isnan(value) & mask):
                        raise ValueError("NaN in retained values.")
                    ordered = value[:-1] >= value[1:] if result.largest else value[:-1] <= value[1:]
                    if np.any(~ordered & mask[1:]):
                        raise ValueError("Retained values are not ordered.")
                target = result.sum if isinstance(result, BatchNanMean) else result
                setattr(target, field, value.copy())
                target.n_samples = count.copy()
                result.n_samples = target.n_samples
            elif isinstance(result, BatchTopK) and meta["ndim"] is not None:
                raise ValueError("Uninitialized state has dimensions.")
            return result
        except (KeyError, TypeError, AttributeError, OverflowError) as exc:
            raise ValueError("Malformed accumulator state.") from exc

    def save(self, path):
        """Write metadata as JSON inside an NPZ archive (not atomic)."""
        state = self.to_state()
        np.savez(path, metadata=np.asarray(json.dumps(state["metadata"])), **state["arrays"])

    @classmethod
    def load(cls, path):
        """Read an NPZ checkpoint with allow_pickle=False."""
        with np.load(path, allow_pickle=False) as archive:
            meta = json.loads(str(archive["metadata"]))
            arrays = {key: archive[key] for key in archive.files if key != "metadata"}
        return cls.from_state({"metadata": meta, "arrays": arrays})
