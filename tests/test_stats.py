import numpy as np
import pytest

from batchstats import BatchCov, BatchMax, BatchMean, BatchMin, BatchPeakToPeak, BatchStd, BatchSum, BatchVar


@pytest.fixture
def data():
    m, n = 1_000_000, 50
    return 1e1*np.random.randn(m, n) + 1e3


@pytest.fixture
def data_2d_features():
    m, n, o = 100_000, 50, 60
    return 1e1*np.random.randn(m, n, o) + 1e3


@pytest.fixture
def n_batches():
    return 31


def test_min(data, n_batches):
    true_stat = np.min(data, axis=0)

    batchmin = BatchMin()
    for batch_data in np.array_split(data, n_batches):
        batchmin.update_batch(batch=batch_data)
    batch_stat = batchmin()
    assert np.allclose(true_stat, batch_stat)


def test_max(data, n_batches):
    true_stat = np.max(data, axis=0)

    batchmax = BatchMax()
    for batch_data in np.array_split(data, n_batches):
        batchmax.update_batch(batch=batch_data)
    batch_stat = batchmax()
    assert np.allclose(true_stat, batch_stat)


def test_mean(data, n_batches):
    true_stat = np.mean(data, axis=0)

    batchmean = BatchMean()
    for batch_data in np.array_split(data, n_batches):
        batchmean.update_batch(batch=batch_data)
    batch_stat = batchmean()
    assert np.allclose(true_stat, batch_stat)


def test_ptp(data, n_batches):
    true_stat = np.ptp(data, axis=0)

    batchptp = BatchPeakToPeak()
    for batch_data in np.array_split(data, n_batches):
        batchptp.update_batch(batch=batch_data)
    batch_stat = batchptp()
    assert np.allclose(true_stat, batch_stat)


def test_sum(data, n_batches):
    true_stat = np.sum(data, axis=0)

    batchsum = BatchSum()
    for batch_data in np.array_split(data, n_batches):
        batchsum.update_batch(batch=batch_data)
    batch_stat = batchsum()
    assert np.allclose(true_stat, batch_stat)


def test_std(data, n_batches):
    true_stat = np.std(data, axis=0)

    batchstd = BatchStd()
    for batch_data in np.array_split(data, n_batches):
        batchstd.update_batch(batch=batch_data)
    batch_stat = batchstd()
    assert np.allclose(true_stat, batch_stat)


def test_var(data, n_batches):
    true_stat = np.var(data, axis=0)

    batchvar = BatchVar()
    for batch_data in np.array_split(data, n_batches):
        batchvar.update_batch(batch=batch_data)
    batch_stat = batchvar()
    assert np.allclose(true_stat, batch_stat)


def test_var_ddof(data, n_batches):
    ddof = 2
    true_stat = np.var(data, axis=0, ddof=ddof)

    batchvar = BatchVar(ddof=ddof)
    for batch_data in np.array_split(data, n_batches):
        batchvar.update_batch(batch=batch_data)
    batch_stat = batchvar()
    assert np.allclose(true_stat, batch_stat)


def test_cov_1(data, n_batches):
    true_cov = np.cov(data.T, ddof=0)
    true_var = np.var(data, axis=0)

    batchvar = BatchVar()
    batchcov = BatchCov()
    for batch_data in np.array_split(data, n_batches):
        batchvar.update_batch(batch_data)
        batchcov.update_batch(batch_data)

    cov = batchcov()
    var = batchvar()

    assert np.allclose(cov, true_cov)
    assert np.allclose(var, true_var)
    assert np.allclose(var, np.diag(cov))


def test_cov_2(data, n_batches):
    true_cov = np.cov(data.T, ddof=0)
    index = np.arange(25)

    batchcov = BatchCov()
    for batch_data in np.array_split(data, n_batches):
        batchcov.update_batch(batch_data, batch_data[:, index])

    cov = batchcov()

    assert np.allclose(cov, true_cov[:, index])


def test_mean_2d_features(data_2d_features, n_batches):
    true_stat = np.mean(data_2d_features, axis=0)

    batchvar = BatchMean()
    for batch_data in np.array_split(data_2d_features, n_batches):
        batchvar.update_batch(batch=batch_data)
    batch_stat = batchvar()
    assert np.allclose(true_stat, batch_stat)


def test_var_2d_features(data_2d_features, n_batches):
    true_stat = np.var(data_2d_features, axis=0)

    batchvar = BatchVar()
    for batch_data in np.array_split(data_2d_features, n_batches):
        batchvar.update_batch(batch=batch_data)
    batch_stat = batchvar()
    assert np.allclose(true_stat, batch_stat)
