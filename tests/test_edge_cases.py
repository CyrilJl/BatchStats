import importlib
import warnings

import numpy as np
import pytest

import batchstats._misc as misc
from batchstats import (
    BatchCorr,
    BatchMax,
    BatchMean,
    BatchMin,
    BatchNanMax,
    BatchNanMean,
    BatchNanMin,
    BatchNanPeakToPeak,
    BatchNanSum,
    BatchPeakToPeak,
    BatchStd,
    BatchSum,
    BatchVar,
    DifferentAxisError,
    DifferentShapesError,
    DifferentStatsError,
    NoValidSamplesError,
    UnequalSamplesNumber,
)


@pytest.mark.parametrize("stat_class", [BatchSum, BatchMean])
def test_standard_stats_preserve_infinities(stat_class):
    data = np.array([[np.inf, -np.inf], [1.0, 2.0]])

    result = stat_class().update_batch(data)()

    np.testing.assert_array_equal(result, [np.inf, -np.inf])


@pytest.mark.parametrize("stat_class", [BatchNanSum, BatchNanMean])
def test_nan_stats_count_infinities_as_valid(stat_class):
    data = np.array([[np.inf, -np.inf], [np.nan, np.nan]])
    stat = stat_class().update_batch(data)

    np.testing.assert_array_equal(stat(), [np.inf, -np.inf])
    np.testing.assert_array_equal(stat.sum.n_samples if isinstance(stat, BatchNanMean) else stat.n_samples, [1, 1])


@pytest.mark.parametrize("stat_class, reducer", [(BatchNanMin, np.nanmin), (BatchNanMax, np.nanmax)])
def test_nan_extrema_merge_recovers_all_nan_slices(stat_class, reducer):
    left_data = np.array([[np.nan, 4.0], [np.nan, 2.0]])
    right_data = np.array([[3.0, np.nan], [1.0, np.nan]])

    merged = stat_class().update_batch(left_data) + stat_class().update_batch(right_data)

    np.testing.assert_array_equal(merged(), reducer(np.concatenate([left_data, right_data]), axis=0))
    np.testing.assert_array_equal(merged.n_samples, [2, 2])


def test_empty_accumulator_has_public_error_type():
    with pytest.raises(NoValidSamplesError, match="No valid samples"):
        BatchMean()()


def test_correlation_rejects_unequal_batch_lengths():
    with pytest.raises(UnequalSamplesNumber, match="same lengths"):
        BatchCorr().update_batch(np.ones((3, 2)), np.ones((2, 2)))


@pytest.mark.parametrize("stat_class", [BatchSum, BatchMean, BatchMin, BatchMax, BatchPeakToPeak, BatchVar, BatchStd])
def test_empty_accumulator_is_merge_identity(stat_class):
    data = np.arange(12.0).reshape(6, 2)
    populated = stat_class().update_batch(data)

    np.testing.assert_allclose((stat_class() + populated)(), populated())
    np.testing.assert_allclose((populated + stat_class())(), populated())


@pytest.mark.parametrize("stat_class", [BatchNanMin, BatchNanMax, BatchNanPeakToPeak])
def test_empty_nan_accumulator_is_merge_identity(stat_class):
    populated = stat_class().update_batch([[1.0, np.nan], [2.0, 3.0]])

    np.testing.assert_allclose((stat_class() + populated)(), populated(), equal_nan=True)
    np.testing.assert_allclose((populated + stat_class())(), populated(), equal_nan=True)


def test_merge_rejects_incompatible_accumulators():
    with pytest.raises(DifferentStatsError):
        BatchSum() + BatchMean()
    with pytest.raises(DifferentAxisError):
        BatchSum(axis=0) + BatchSum(axis=1)
    with pytest.raises(DifferentShapesError):
        BatchSum().update_batch(np.ones((2, 2))) + BatchSum().update_batch(np.ones((2, 3)))


def test_import_does_not_replace_process_warning_formatter(monkeypatch):
    def formatter(message, *args, **kwargs):
        return str(message)

    monkeypatch.setattr(warnings, "formatwarning", formatter)

    importlib.reload(misc)

    assert warnings.formatwarning is formatter
