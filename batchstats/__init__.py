from importlib.metadata import version

from .base import BatchNanStat, BatchStat
from .nanstats import BatchNanMax, BatchNanMean, BatchNanMin, BatchNanPeakToPeak, BatchNanSum
from .stats import (
    BatchCov,
    BatchMax,
    BatchMean,
    BatchMin,
    BatchPeakToPeak,
    BatchStd,
    BatchSum,
    BatchVar,
)

__all__ = [
    "BatchCov",
    "BatchMax",
    "BatchMean",
    "BatchMin",
    "BatchPeakToPeak",
    "BatchStat",
    "BatchStd",
    "BatchSum",
    "BatchVar",
    "BatchNanMax",
    "BatchNanMean",
    "BatchNanMin",
    "BatchNanPeakToPeak",
    "BatchNanStat",
    "BatchNanSum",
]

__version__ = version("batchstats")
