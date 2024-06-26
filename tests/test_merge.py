import numpy as np
import pytest

from batchstats import BatchCov, BatchMax, BatchMean, BatchMin, BatchPeakToPeak, BatchStd, BatchSum, BatchVar


@pytest.fixture
def data():
    m, n = 25_000, 50
    return np.random.randn(m, n), np.random.randn(2*m, n)


def test_merge(data):
    data1, data2 = data
    data0 = np.concatenate([data1, data2])
    for stat in (BatchCov, BatchMax, BatchMean, BatchMin, BatchPeakToPeak, BatchStd, BatchSum, BatchVar):
        s0 = stat().update_batch(data0)
        s1 = stat().update_batch(data1)
        s2 = stat().update_batch(data2)
        s3 = s1 + s2
        assert np.allclose(s0(), s3())
