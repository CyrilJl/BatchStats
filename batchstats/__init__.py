from .nanstats import BatchNanMean, BatchNanStat, BatchNanSum
from .stats import (BatchCov, BatchMax, BatchMean, BatchMin, BatchPeakToPeak,
                    BatchStat, BatchSum, BatchVar)

__all__ = ['BatchCov', 'BatchMax', 'BatchMean', 'BatchMin', 'BatchPeakToPeak', 'BatchStat', 'BatchSum', 'BatchVar',
           'BatchNanMean', 'BatchNanStat', 'BatchNanSum']

__version__ = '0.3.2'
