from importlib.metadata import version

from ._misc import (
    DifferentAxisError,
    DifferentShapesError,
    DifferentStatsError,
    NoValidSamplesError,
    UnequalSamplesNumber,
)
from .base import BatchNanStat, BatchStat
from .nanstats import BatchNanMax, BatchNanMean, BatchNanMin, BatchNanPeakToPeak, BatchNanSum
from .stats import (
    BatchCorr,
    BatchCov,
    BatchMax,
    BatchMean,
    BatchMin,
    BatchPeakToPeak,
    BatchStd,
    BatchSum,
    BatchVar,
    BatchWeightedMean,
    BatchWeightedSum,
)

__all__ = [
    "BatchCorr",
    "BatchCov",
    "BatchMax",
    "BatchMean",
    "BatchMin",
    "BatchNanMax",
    "BatchNanMean",
    "BatchNanMin",
    "BatchNanPeakToPeak",
    "BatchNanStat",
    "BatchNanSum",
    "BatchPeakToPeak",
    "BatchStat",
    "BatchStd",
    "BatchSum",
    "BatchVar",
    "BatchWeightedMean",
    "BatchWeightedSum",
    "DifferentAxisError",
    "DifferentShapesError",
    "DifferentStatsError",
    "NoValidSamplesError",
    "UnequalSamplesNumber",
]

__version__ = version("batchstats")
