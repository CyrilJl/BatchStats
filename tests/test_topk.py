import numpy as np
import pytest

from batchstats import BatchNanTopK, BatchTopK, NoValidSamplesError, required_k


@pytest.mark.parametrize("axis", [0, -3, (0, 2), (2, 0), None, ()])
@pytest.mark.parametrize("largest", [True, False])
@pytest.mark.parametrize("k", [1, 5, 80])
def test_selection(axis, largest, k):
    data = np.random.default_rng(42).integers(-4, 5, (17, 3, 2))
    original = data.copy()
    stat = BatchTopK(k, axis=axis, largest=largest).update_batch(data)
    axes = tuple(range(3)) if axis is None else axis if isinstance(axis, tuple) else (axis,)
    axes = tuple(ax % 3 for ax in axes)
    remaining = tuple(ax for ax in range(3) if ax not in axes)
    shape = tuple(data.shape[ax] for ax in remaining)
    flat = data.transpose(axes + remaining).reshape((-1, *shape))
    expected = np.sort(flat, axis=0)
    if largest:
        expected = expected[::-1]
    count = min(k, len(expected))
    np.testing.assert_array_equal(stat()[:count], expected[:count])
    assert stat()[count:].mask.all()
    assert stat().dtype == data.dtype
    np.testing.assert_array_equal(original, data)


@pytest.mark.parametrize("largest", [True, False])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_nan_selection_and_merge(largest, dtype):
    data = np.random.default_rng(2).normal(size=(53, 4, 3)).astype(dtype)
    data[::4, 0] = np.nan
    data[:, 1, 0] = np.nan
    data[0, 2, 1], data[1, 2, 1] = np.inf, -np.inf
    data = data[:, :, ::-1]  # noncontiguous
    parts = [BatchNanTopK(7, largest=largest).update_batch(x) for x in np.array_split(data, [1, 15, 50])]
    sequential = BatchNanTopK(7, largest=largest)
    for x in np.array_split(data, [1, 15, 50]):
        sequential.update_batch(x)
    merged = (parts[0] + parts[1]) + (parts[2] + parts[3])
    reversed_tree = parts[3] + (parts[2] + (parts[1] + parts[0]))
    whole = BatchNanTopK(7, largest=largest).update_batch(data)
    for stat in [whole, sequential, merged, reversed_tree]:
        assert stat().dtype == dtype
        np.testing.assert_array_equal(stat.n_samples, np.count_nonzero(~np.isnan(data), axis=0))
        for index in np.ndindex(data.shape[1:]):
            column = data[(slice(None), *index)]
            expected = np.sort(column[~np.isnan(column)])
            if largest:
                expected = expected[::-1]
            actual = stat()[(slice(None), *index)].compressed()
            np.testing.assert_array_equal(actual, expected[:7])
    np.testing.assert_array_equal(whole().mask, merged().mask)
    assert not np.shares_memory(parts[0].values, merged.values)


@pytest.mark.parametrize("n,q,k", [(365, 0.99, 5), (366, 0.99, 5), (8760, 0.998, 19), (8784, 0.998, 19)])
@pytest.mark.parametrize("largest", [True, False])
def test_annual_quantiles(n, q, k, largest):
    if not largest:
        q = 1 - q
    assert required_k(q, n, largest) == k
    data = np.random.default_rng(8).normal(size=(n, 5))
    data[::11, 2] = np.nan
    stat = BatchNanTopK(k, largest=largest)
    for block in np.array_split(data, 37):
        stat.update_batch(block)
    np.testing.assert_allclose(stat.quantile(q), np.nanquantile(data, q, axis=0), rtol=1e-13)


def test_rank_quantile_empty_and_infinite():
    stat = BatchNanTopK(5).update_batch([[1, np.nan, np.inf], [3, np.nan, np.inf]])
    np.testing.assert_array_equal(stat.rank(1).mask, [False, True, False])
    np.testing.assert_allclose(stat.quantile(0), [1, np.nan, np.inf])
    np.testing.assert_allclose(stat.quantile(1), [3, np.nan, np.inf])
    np.testing.assert_allclose(stat.quantile(0.5)[:2], [2, np.nan])
    single = BatchTopK(1).update_batch([[np.inf]])
    assert single.quantile(0.3)[0] == np.inf
    with pytest.raises(ValueError, match="capacity"):
        BatchTopK(2).update_batch(np.arange(10)[:, None]).quantile(0.5)
    with pytest.raises(ValueError):
        stat.rank(6)
    with pytest.raises(NoValidSamplesError):
        BatchTopK(2)()
    empty = BatchTopK(2).update_batch(np.empty((0, 3)))
    assert empty().mask.all()
    assert np.isnan(empty.quantile(0.5)).all()


@pytest.mark.parametrize("kwargs", [{"k": 0}, {"k": -1}, {"k": True}, {"k": 1.5}, {"k": 1, "largest": 1}])
def test_invalid_parameters(kwargs):
    with pytest.raises((ValueError, TypeError)):
        BatchTopK(**kwargs)


def test_errors_are_transactional():
    stat = BatchTopK(2).update_batch([[1, 2]])
    with pytest.raises(TypeError, match="masked"):
        stat.update_batch(np.ma.array([[1, 2]], mask=[[True, False]]))
    for data in [[[np.nan, 3]], [[1, 2, 3]], np.zeros((1, 1, 2)), [[1j, 2j]]]:
        with pytest.raises((ValueError, TypeError)):
            stat.update_batch(data)
        np.testing.assert_array_equal(stat.n_samples, [1, 1])
    for other in [
        BatchTopK(3),
        BatchTopK(2, largest=False),
        BatchTopK(2, axis=1),
        BatchNanTopK(2),
        BatchTopK(2).update_batch([[1, 2, 3]]),
    ]:
        with pytest.raises(ValueError):
            stat + other
    for q in [-0.1, 1.1, np.nan, [0.5]]:
        with pytest.raises(ValueError):
            stat.quantile(q)
    with pytest.raises(ValueError):
        stat.quantile(0.5, method="nearest")
    for axis in [(0, 0), 3, -4]:
        with pytest.raises(ValueError):
            BatchTopK(1, axis=axis).update_batch([[1]])


def test_integer_precision_and_empty_merge():
    data = np.array([[2**63 - 2], [2**63 - 1]], dtype=np.int64)
    stat = BatchTopK(3).update_batch(data)
    np.testing.assert_array_equal(stat().compressed(), data[::-1, 0])
    for result in [stat + BatchTopK(3), BatchTopK(3) + stat]:
        result.values[0] = 0
        assert stat.values[0, 0] == 2**63 - 1
