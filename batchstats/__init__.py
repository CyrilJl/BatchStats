from importlib.metadata import version

from ._misc import (
    DifferentAxisError,
    DifferentShapesError,
    DifferentStatsError,
    NoValidSamplesError,
    UnequalSamplesNumber,
)
from .base import BatchNanStat, BatchStat
from .nanstats import BatchNanMax, BatchNanMean, BatchNanMin, BatchNanPeakToPeak, BatchNanSum, BatchNanTopK
from .stats import (
    BatchCorr,
    BatchCov,
    BatchMax,
    BatchMean,
    BatchMin,
    BatchPeakToPeak,
    BatchStd,
    BatchSum,
    BatchTopK,
    BatchVar,
    BatchWeightedMean,
    BatchWeightedSum,
    required_k,
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
    "BatchNanTopK",
    "BatchPeakToPeak",
    "BatchStat",
    "BatchStd",
    "BatchSum",
    "BatchTopK",
    "BatchVar",
    "BatchWeightedMean",
    "BatchWeightedSum",
    "DifferentAxisError",
    "DifferentShapesError",
    "DifferentStatsError",
    "NoValidSamplesError",
    "UnequalSamplesNumber",
    "required_k",
]

__version__ = version("batchstats")
