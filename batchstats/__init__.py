from importlib.metadata import version

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
    "BatchPeakToPeak",
    "BatchStat",
    "BatchStd",
    "BatchSum",
    "BatchVar",
    "BatchWeightedMean",
    "BatchWeightedSum",
    "BatchNanMax",
    "BatchNanMean",
    "BatchNanMin",
    "BatchNanPeakToPeak",
    "BatchNanStat",
    "BatchNanSum",
]

__version__ = version("batchstats")
