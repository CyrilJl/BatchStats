import numpy as np
import pytest

from batchstats import BatchNanMean, BatchNanSum


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
