import numpy as np
import pytest

from batchstats import BatchNanMax, BatchNanMean, BatchNanMin, BatchNanPeakToPeak, BatchNanSum


@pytest.fixture
def data():
    m, n = 1_000_000, 50
    nan_ratio = 0.05
    data = np.random.randn(m, n)
    num_nans = int(m * n * nan_ratio)
    nan_indices = np.random.choice(range(m * n), num_nans, replace=False)
    data.ravel()[nan_indices] = np.nan
    return data


@pytest.fixture
def n_batches():
    return 31


def test_nansum(data, n_batches):
    true_stat = np.nansum(data, axis=0)

    batchsum = BatchNanSum()
    for batch_data in np.array_split(data, n_batches):
        batchsum.update_batch(batch=batch_data)
    batch_stat = batchsum()
    assert np.allclose(true_stat, batch_stat)


def test_nanmean(data, n_batches):
    true_stat = np.nanmean(data, axis=0)

    batchmean = BatchNanMean()
    for batch_data in np.array_split(data, n_batches):
        batchmean.update_batch(batch=batch_data)
    batch_stat = batchmean()
    assert np.allclose(true_stat, batch_stat)


def test_nanmin(data, n_batches):
    true_stat = np.nanmin(data, axis=0)

    batchmin = BatchNanMin()
    for batch_data in np.array_split(data, n_batches):
        batchmin.update_batch(batch=batch_data)
    batch_stat = batchmin()
    assert np.allclose(true_stat, batch_stat)


def test_nanmax(data, n_batches):
    true_stat = np.nanmax(data, axis=0)

    batchmax = BatchNanMax()
    for batch_data in np.array_split(data, n_batches):
        batchmax.update_batch(batch=batch_data)
    batch_stat = batchmax()
    assert np.allclose(true_stat, batch_stat)


def test_nanptp(data, n_batches):
    true_stat = np.nanmax(data, axis=0) - np.nanmin(data, axis=0)

    batchptp = BatchNanPeakToPeak()
    for batch_data in np.array_split(data, n_batches):
        batchptp.update_batch(batch=batch_data)
    batch_stat = batchptp()
    assert np.allclose(true_stat, batch_stat)
