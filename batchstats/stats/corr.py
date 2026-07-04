import numpy as np

from .._misc import (
    DifferentAxisError,
    DifferentShapesError,
    DifferentStatsError,
    NoValidSamplesError,
    UnequalSamplesNumber,
    any_nan,
)
from ..base import BatchStat
from .cov import BatchCov
from .var import BatchVar


class BatchCorr(BatchStat):
    """
    Class for calculating the correlation of batches of data.

    The algorithm is an implementation of an online correlation calculation.
    It is numerically stable and avoids a two-pass approach.

    Args:
        ddof (int, optional): Means Delta Degrees of Freedom. The divisor used in calculations is N - ddof, where N represents the number of elements. By default ddof is zero.

    .. code:: python

        import numpy as np
        from batchstats import BatchCorr

        # create some data
        data1 = np.array([[1, 2], [3, 4]])
        data2 = np.array([[5, 6], [7, 8]])

        # create a BatchCorr object
        bc = BatchCorr()

        # update with the first batch
        bc.update_batch(data1)

        # update with the second batch
        bc.update_batch(data2)

        # get the correlation
        total_corr = bc()

    """

    def __init__(self, ddof=0):
        super().__init__()
        self.cov = BatchCov(ddof=ddof)
        self.var1 = BatchVar(ddof=ddof)
        self.var2 = BatchVar(ddof=ddof)
        self.ddof = ddof
        self._is_1d = None

    def update_batch(self, batch1, batch2=None, assume_valid=False):
        """
        Update the correlation with new batches of data.

        Args:
            batch1 (numpy.ndarray): Input batch 1.
            batch2 (numpy.ndarray, optional): Input batch 2. Default is None.
            assume_valid (bool, optional): If True, assumes all elements in the batches are valid. Default is False.

        Returns:
            BatchCorr: Updated BatchCorr object.

        """
        is_1d = batch2 is None
        if self._is_1d is None:
            self._is_1d = is_1d
        elif self._is_1d != is_1d:
            raise ValueError("Inconsistent use of BatchCorr. Cannot mix calls with and without batch2.")

        _batch1 = batch1
        _batch2 = batch2
        if self._is_1d:
            _batch2 = _batch1

        if not assume_valid:
            if _batch2 is None:
                _batch1 = _batch2 = np.atleast_2d(np.asarray(_batch1))
            else:
                _batch1, _batch2 = np.atleast_2d(np.asarray(_batch1)), np.atleast_2d(np.asarray(_batch2))

            if len(_batch1) != len(_batch2):
                raise UnequalSamplesNumber("batch1 and batch2 don't have the same lengths.")

            # NaN can only occur in inexact dtypes; identical batches need a single scan
            mask = None
            if np.issubdtype(_batch1.dtype, np.inexact):
                mask = ~any_nan(_batch1, axis=1)
            if _batch2 is not _batch1 and np.issubdtype(_batch2.dtype, np.inexact):
                mask2 = ~any_nan(_batch2, axis=1)
                mask = mask2 if mask is None else mask & mask2

            if mask is not None and not mask.all():
                if _batch2 is _batch1:
                    _batch1 = _batch2 = _batch1[mask]
                else:
                    _batch1, _batch2 = _batch1[mask], _batch2[mask]
            assume_valid = True

        if len(_batch1) > 0:
            self.cov.update_batch(_batch1, _batch2, assume_valid=assume_valid)
            if not self._is_1d:
                # in 1d mode the variances are the diagonal of the covariance:
                # no need to maintain separate BatchVar objects
                self.var1.update_batch(_batch1, assume_valid=assume_valid)
                self.var2.update_batch(_batch2, assume_valid=assume_valid)
            self.n_samples = self.cov.n_samples

        return self

    def __call__(self) -> np.ndarray:
        """
        Calculate the correlation.

        Returns:
            numpy.ndarray: Correlation of the batches.

        Raises:
            NoValidSamplesError: If no valid samples are available.
        """
        if self.cov.cov is None:
            raise NoValidSamplesError("No valid samples for calculating correlation.")

        cov = self.cov()
        if self._is_1d:
            std1 = std2 = np.sqrt(np.diag(cov))
        else:
            std1 = np.sqrt(self.var1())
            std2 = np.sqrt(self.var2())

        with np.errstate(divide="ignore", invalid="ignore"):
            corr = cov / np.outer(std1, std2)

        return corr

    def __add__(self, other):
        """
        Merge two BatchCorr objects.
        """
        if type(self) != type(other):
            raise DifferentStatsError()
        if self.axis != other.axis:
            raise DifferentAxisError()

        if self.cov.cov is not None and other.cov.cov is not None:
            if self.cov.cov.shape != other.cov.cov.shape:
                raise DifferentShapesError()

        if self.n_samples == 0:
            return other
        elif other.n_samples == 0:
            return self

        if self._is_1d != other._is_1d:
            raise ValueError("Cannot merge BatchCorr objects with different setups (1d vs 2d).")

        ret = BatchCorr(ddof=self.ddof)
        ret.cov = self.cov + other.cov

        # in 1d mode var1/var2 are never populated (variances come from diag(cov))
        if not self._is_1d:
            ret.var1 = self.var1 + other.var1
            ret.var2 = self.var2 + other.var2

        ret.n_samples = ret.cov.n_samples
        ret._is_1d = self._is_1d
        return ret
