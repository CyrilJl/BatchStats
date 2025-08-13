import numpy as np
import pytest

from batchstats.stats.weighted_mean import BatchWeightedMean
from batchstats.stats.weighted_sum import BatchWeightedSum


@pytest.fixture
def data_3d():
    m, n, p = 100, 50, 10
    return 1e1 * np.random.randn(m, n, p) + 1e3


axis_weight_scenarios = [
    (0, "full"),
    (0, "broadcast_row"),
    (1, "full"),
    (1, "broadcast_col"),
    (2, "full"),
    ((1, 2), "full"),
    ((1, 2), "broadcast_plane"),
]


@pytest.fixture(params=axis_weight_scenarios)
def scenario(request):
    return request.param


@pytest.fixture
def axis(scenario):
    return scenario[0]


@pytest.fixture
def weights_type(scenario):
    return scenario[1]


@pytest.fixture
def weights_3d(data_3d, axis, weights_type):
    shape = data_3d.shape

    if weights_type == "full":
        return np.random.rand(*shape)

    w_shape = list(shape)
    # This logic is a bit naive, but covers the test cases
    if weights_type == "broadcast_row": # axis 0
        w_shape = [shape[0], 1, 1]
    elif weights_type == "broadcast_col": # axis 1
        w_shape = [1, shape[1], 1]
    elif weights_type == "broadcast_plane": # axis (1,2)
        w_shape = [1, shape[1], shape[2]]

    return np.random.rand(*w_shape)


@pytest.fixture
def n_batches():
    return 13


@pytest.mark.parametrize("klass", [BatchWeightedSum, BatchWeightedMean])
def test_weighted_stats_3d(data_3d, n_batches, axis, weights_3d, klass):
    if klass == BatchWeightedSum:
        true_stat = np.sum(data_3d * weights_3d, axis=axis)
    else:
        broadcasted_weights = np.broadcast_to(weights_3d, data_3d.shape)
        true_stat = np.sum(data_3d * weights_3d, axis=axis) / np.sum(broadcasted_weights, axis=axis)

    batch_op = klass(axis=axis)

    data_batches = np.array_split(data_3d, n_batches, axis=0)

    if weights_3d.shape[0] > 1:
        weights_batches = np.array_split(weights_3d, n_batches, axis=0)
    else:
        weights_batches = [weights_3d] * n_batches

    for batch_data, batch_weights in zip(data_batches, weights_batches):
        batch_op.update_batch(batch=batch_data, weights=batch_weights)

    batch_stat = batch_op()
    assert np.allclose(true_stat, batch_stat)


@pytest.mark.parametrize("klass", [BatchWeightedSum, BatchWeightedMean])
def test_weighted_merge_3d(data_3d, axis, weights_3d, klass):
    if klass == BatchWeightedSum:
        true_stat = np.sum(data_3d * weights_3d, axis=axis)
    else:
        broadcasted_weights = np.broadcast_to(weights_3d, data_3d.shape)
        true_stat = np.sum(data_3d * weights_3d, axis=axis) / np.sum(broadcasted_weights, axis=axis)

    # Split data and weights into two halves
    d1, d2 = np.array_split(data_3d, 2, axis=0)
    if weights_3d.shape[0] > 1:
        w1, w2 = np.array_split(weights_3d, 2, axis=0)
    else:
        w1, w2 = weights_3d, weights_3d

    # Create and update two separate objects
    op1 = klass(axis=axis)
    op1.update_batch(d1, w1)

    op2 = klass(axis=axis)
    op2.update_batch(d2, w2)

    # Merge them
    merged_op = op1 + op2

    merged_stat = merged_op()
    assert np.allclose(true_stat, merged_stat)


def test_inconsistent_weights_shape_raises_error(data_3d):
    bws = BatchWeightedSum(axis=1) # Sum over a non-batch axis

    # First batch with per-column weights
    batch1 = data_3d[:10]
    weights1 = np.random.rand(1, data_3d.shape[1], 1)
    bws.update_batch(batch1, weights1)

    # Second batch with per-plane weights
    batch2 = data_3d[10:20]
    weights2 = np.random.rand(1, data_3d.shape[1], data_3d.shape[2])
    with pytest.raises(ValueError, match="Inconsistent weights shape pattern"):
        bws.update_batch(batch2, weights2)
