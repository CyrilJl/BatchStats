import numpy as np
import pytest

from batchstats import BatchCov, BatchMax, BatchMean, BatchMin, BatchStd, BatchSum, BatchVar


@pytest.fixture
def data():
    m, n = 25_000, 50
    return np.random.randn(m, n), np.random.randn(m, n)


def test_merge(data):
    data1, data2 = data
    for stat in (BatchCov, BatchMax, BatchMean, BatchMin, BatchStd, BatchSum, BatchVar):
        print(stat)
        s0 = stat().update_batch(np.concatenate([data1, data2]))
        s1 = stat().update_batch(data1)
        s2 = stat().update_batch(data2)
        s3 = s1 + s2
        assert np.allclose(s0(), s3())
