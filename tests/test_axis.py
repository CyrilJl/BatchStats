import numpy as np
import pytest

from batchstats import BatchStd, BatchVar


@pytest.fixture
def data_3d():
    shape = (100, 10, 20)
    return np.random.randn(*shape)


@pytest.fixture
def n_batches():
    return 13


def test_var_axis_tuple(data_3d, n_batches):
    axis = (0, 1)
    true_stat = np.var(data_3d, axis=axis)

    batchvar = BatchVar(axis=axis)
    for batch_data in np.array_split(data_3d, n_batches):
        batchvar.update_batch(batch=batch_data)
    batch_stat = batchvar()
    assert np.allclose(true_stat, batch_stat)


def test_var_axis_all(data_3d, n_batches):
    axis = (0, 1, 2)
    true_stat = np.var(data_3d, axis=axis)

    batchvar = BatchVar(axis=axis)
    for batch_data in np.array_split(data_3d, n_batches):
        batchvar.update_batch(batch=batch_data)
    batch_stat = batchvar()
    assert np.allclose(true_stat, batch_stat)


def test_std_axis_tuple(data_3d, n_batches):
    axis = (0, 1)
    true_stat = np.std(data_3d, axis=axis)

    batchstd = BatchStd(axis=axis)
    for batch_data in np.array_split(data_3d, n_batches):
        batchstd.update_batch(batch=batch_data)
    batch_stat = batchstd()
    assert np.allclose(true_stat, batch_stat)


def test_var_axis_tuple_unordered(data_3d, n_batches):
    axis = (1, 0)
    true_stat = np.var(data_3d, axis=axis)

    batchvar = BatchVar(axis=axis)
    for batch_data in np.array_split(data_3d, n_batches):
        batchvar.update_batch(batch=batch_data)
    batch_stat = batchvar()
    assert np.allclose(true_stat, batch_stat)


def test_var_axis_tuple_mixed(data_3d, n_batches):
    axis = (0, 2)
    true_stat = np.var(data_3d, axis=axis)

    batchvar = BatchVar(axis=axis)
    for batch_data in np.array_split(data_3d, n_batches):
        batchvar.update_batch(batch=batch_data)
    batch_stat = batchvar()
    assert np.allclose(true_stat, batch_stat)


def test_std_axis_all(data_3d, n_batches):
    axis = (0, 1, 2)
    true_stat = np.std(data_3d, axis=axis)

    batchstd = BatchStd(axis=axis)
    for batch_data in np.array_split(data_3d, n_batches):
        batchstd.update_batch(batch=batch_data)
    batch_stat = batchstd()
    assert np.allclose(true_stat, batch_stat)
