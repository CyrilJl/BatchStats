import numpy as np
import pytest

from batchstats import BatchCov, BatchMax, BatchMean, BatchMin, BatchPeakToPeak, BatchStd, BatchSum, BatchVar


@pytest.fixture
def data():
    m, n = 25_000, 50
    return np.random.randn(m, n), np.random.randn(2 * m, n)


def test_merge(data):
    data1, data2 = data
    data0 = np.concatenate([data1, data2])
    for stat in (BatchCov, BatchMax, BatchMean, BatchMin, BatchPeakToPeak, BatchStd, BatchSum, BatchVar):
        s0 = stat().update_batch(data0)
        s1 = stat().update_batch(data1)
        s2 = stat().update_batch(data2)
        s3 = s1 + s2
        assert np.allclose(s0(), s3())


@pytest.fixture
def data_3d():
    shape = (100, 10, 20)
    return np.random.randn(*shape)


@pytest.mark.parametrize("axis", [0, (0, 1), (0, 2), (0, 1, 2)])
def test_merge_axis(data_3d, axis):
    data1, data2 = np.array_split(data_3d, 2)
    data0 = np.concatenate([data1, data2])

    for stat_class in (BatchMax, BatchMean, BatchMin, BatchPeakToPeak, BatchStd, BatchSum, BatchVar):
        s0 = stat_class(axis=axis).update_batch(data0)
        s1 = stat_class(axis=axis).update_batch(data1)
        s2 = stat_class(axis=axis).update_batch(data2)
        s3 = s1 + s2
        assert np.allclose(s0(), s3())
