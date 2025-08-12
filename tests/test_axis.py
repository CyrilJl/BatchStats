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


@pytest.mark.parametrize(
    "stat_class, ref_func, axis",
    [
        (BatchVar, np.var, (0, 1)),
        (BatchVar, np.var, (0, 1, 2)),
        (BatchStd, np.std, (0, 1)),
        (BatchVar, np.var, (1, 0)),
        (BatchVar, np.var, (0, 2)),
        (BatchStd, np.std, (0, 1, 2)),
    ],
)
def test_batch_stats(data_3d, n_batches, stat_class, ref_func, axis):
    """Test batch statistics calculation for various operations and axes."""
    true_stat = ref_func(data_3d, axis=axis)

    batch_stat_processor = stat_class(axis=axis)
    for batch_data in np.array_split(data_3d, n_batches):
        batch_stat_processor.update_batch(batch=batch_data)
    batch_stat = batch_stat_processor()

    assert np.allclose(true_stat, batch_stat)
