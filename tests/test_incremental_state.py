import copy

import numpy as np
import pytest

from batchstats import BatchNanMean, BatchNanSum, BatchNanTopK, BatchTopK, NoValidSamplesError


@pytest.mark.parametrize("cls", [BatchNanMean, BatchNanSum])
@pytest.mark.parametrize("axis", [0, -3, (0, 2), None])
def test_nan_merge(cls, axis):
    data = np.random.default_rng(5).normal(size=(29, 3, 4))
    data[::3, :, 0] = np.nan
    data[:, 1] = np.nan
    a = cls(axis=axis).update_batch(data[:4])
    b = cls(axis=axis).update_batch(data[4:])
    merged = a + b
    sequential = cls(axis=axis).update_batch(data[:4]).update_batch(data[4:])
    np.testing.assert_allclose(merged(), sequential(), rtol=1e-14, atol=1e-14)
    reference = np.nansum(data, axis=axis)
    counts = np.count_nonzero(~np.isnan(data), axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        reference = reference / counts if cls is BatchNanMean else np.where(counts, reference, np.nan)
    np.testing.assert_allclose(merged(), reference, rtol=1e-14, atol=1e-14)
    np.testing.assert_array_equal(merged.n_samples, counts)
    for result in [a + cls(axis=axis), cls(axis=axis) + a]:
        result.update_batch(data[:4])
        np.testing.assert_allclose(a(), cls(axis=axis).update_batch(data[:4])())
    incompatible = [cls(axis=1)]
    if axis is not None:
        incompatible.append(cls(axis=axis).update_batch(np.zeros((1, 8, 7))))
    for other in incompatible:
        with pytest.raises(ValueError):
            a + other
    with pytest.raises(ValueError):
        a + BatchTopK(2)


def test_sum_dtype_and_invalid_update():
    a = BatchNanSum().update_batch(np.array([[2, 3]], dtype=np.int16))
    b = BatchNanSum().update_batch(np.array([[0.5, 0.25]], dtype=np.float32))
    assert (a + b).sum.dtype == np.result_type(a.sum.dtype, b.sum.dtype)
    a.update_batch(np.array([[0.5, 0.25]], dtype=np.float32))
    np.testing.assert_array_equal(a(), [2.5, 3.25])
    with pytest.raises(ValueError):
        a.update_batch([[1, 2, 3]])
    np.testing.assert_array_equal(a.n_samples, [2, 2])
    with pytest.raises(NoValidSamplesError):
        (BatchNanSum() + BatchNanSum())()


@pytest.mark.parametrize("cls", [BatchNanSum, BatchNanMean, BatchTopK, BatchNanTopK])
@pytest.mark.parametrize("initialized", [False, True])
@pytest.mark.parametrize("axis", [0, (0, 1), None])
def test_state_roundtrip(cls, initialized, axis, tmp_path):
    kwargs = {"k": 5} if cls in [BatchTopK, BatchNanTopK] else {}
    stat = cls(axis=axis, **kwargs)
    data = np.arange(24, dtype=np.float32).reshape(4, 3, 2)
    if initialized:
        stat.update_batch(data)
    state = stat.to_state()
    restored = cls.from_state(state)
    path = tmp_path / "checkpoint.npz"
    stat.save(path)
    loaded = cls.load(path)
    for candidate in [restored, loaded]:
        candidate.update_batch(data)
        expected = cls(axis=axis, **kwargs)
        if initialized:
            expected.update_batch(data)
        expected.update_batch(data)
        np.testing.assert_array_equal(candidate(), expected())
        merged = candidate + expected
        expected.update_batch(data)
        if initialized:
            expected.update_batch(data)
        np.testing.assert_array_equal(merged(), expected())
    if initialized:
        state["arrays"]["n_samples"][...] = 0
        assert np.all(stat.n_samples > 0)
        assert np.all(restored.n_samples > 0)


@pytest.mark.parametrize("cls", [BatchNanSum, BatchNanMean, BatchTopK, BatchNanTopK])
def test_corrupted_states(cls):
    kwargs = {"k": 3} if cls in [BatchTopK, BatchNanTopK] else {}
    state = cls(**kwargs).update_batch([[1, 2], [3, 4]]).to_state()
    for change in ["version", "type", "shape", "count", "dtype", "axis", "missing"]:
        bad = copy.deepcopy(state)
        if change in ["version", "type"]:
            bad["metadata"][change] = "unknown"
        elif change == "shape":
            bad["metadata"]["arrays"]["n_samples"]["shape"] = [999]
        elif change == "count":
            bad["arrays"]["n_samples"][0] = -1
        elif change == "dtype":
            bad["metadata"]["arrays"]["n_samples"]["dtype"] = "float32"
        elif change == "axis":
            bad["metadata"]["params"]["axis"] = "invalid"
        else:
            del bad["arrays"]["n_samples"]
        with pytest.raises(ValueError):
            cls.from_state(bad)


def test_topk_state_order():
    state = BatchTopK(3).update_batch([[1], [2], [3]]).to_state()
    state["arrays"]["values"][:] = [[1], [2], [3]]
    with pytest.raises(ValueError, match="ordered"):
        BatchTopK.from_state(state)
