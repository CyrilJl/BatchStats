import numpy as np

from .._misc import NoValidSamplesError, UnequalSamplesNumber, any_nan, check_params
from ..base import BatchStat
from .mean import BatchMean


class BatchCov(BatchStat):
    """
    Class for calculating the covariance of batches of data.

    The algorithm is an implementation of an online covariance calculation.
    It is numerically stable and avoids a two-pass approach.

    Args:
        ddof (int, optional): Means Delta Degrees of Freedom. The divisor used in calculations is N - ddof, where N represents the number of elements. By default ddof is zero.

    .. code:: python

        import numpy as np
        from batchstats import BatchCov

        # create some data
        data1 = np.array([[1, 2], [3, 4]])
        data2 = np.array([[5, 6], [7, 8]])

        # create a BatchCov object
        bc = BatchCov()

        # update with the first batch
        bc.update_batch(data1)

        # update with the second batch
        bc.update_batch(data2)

        # get the covariance
        total_cov = bc()

        # verify the result
        expected_cov = np.array([[5., 5.], [5., 5.]])
        np.testing.assert_allclose(total_cov, expected_cov)

    """

    def __init__(self, ddof=0):
        super().__init__()
        self.mean1 = BatchMean()
        self.mean2 = BatchMean()
        self.cov = None
        self.ddof = check_params(param=ddof, types=int)

    def _process_batch(self, batch1, batch2=None, assume_valid=False):
        """
        Process the input batches, handling NaN values if necessary.

        Args:
            batch1 (numpy.ndarray): Input batch 1.
            batch2 (numpy.ndarray, optional): Input batch 2. Default is None.
            assume_valid (bool, optional): If True, assumes all elements in the batches are valid. Default is False.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray]: Processed batches 1 and 2.

        Raises:
            UnequalSamplesNumber: If the batches have unequal lengths.

        """
        if batch2 is None:
            batch1 = batch2 = np.atleast_2d(np.asarray(batch1))
        else:
            batch1, batch2 = np.atleast_2d(np.asarray(batch1)), np.atleast_2d(np.asarray(batch2))
        if assume_valid:
            self.n_samples += len(batch1)
            return batch1, batch2
        else:
            if len(batch1) != len(batch2):
                raise UnequalSamplesNumber("batch1 and batch2 don't have the same lengths.")
            mask = ~any_nan(batch1, axis=1) & ~any_nan(batch2, axis=1)
            n = mask.sum()
            self.n_samples += n
            if np.all(mask):
                return batch1, batch2
            else:
                return batch1[mask], batch2[mask]

    def update_batch(self, batch1, batch2=None, assume_valid=False):
        """
        Update the covariance with new batches of data.

        Args:
            batch1 (numpy.ndarray): Input batch 1.
            batch2 (numpy.ndarray, optional): Input batch 2. Default is None.
            assume_valid (bool, optional): If True, assumes all elements in the batches are valid. Default is False.

        Returns:
            BatchCov: Updated BatchCov object.

        """
        batch1, batch2 = self._process_batch(batch1, batch2, assume_valid=assume_valid)
        n = len(batch1)
        if n > 0:
            if self.cov is None:
                n1, n2 = batch1.shape[1], batch2.shape[1]
                self.mean1.update_batch(batch1, assume_valid=True)
                self.mean2.update_batch(batch2, assume_valid=True)
                if n == 1:
                    self.cov = np.zeros((n1, n2))
                else:
                    self.cov = (batch1 - self.mean1()).T @ ((batch2 - self.mean2()) / n)
            else:
                m1 = self.mean1()
                self.mean2.update_batch(batch2, assume_valid=True)
                m2 = self.mean2()
                self.cov += (batch1 - m1).T @ ((batch2 - m2) / self.mean1.n_samples)
                self.cov *= self.mean1.n_samples / self.mean2.n_samples
                self.mean1.update_batch(batch1, assume_valid=True)
        return self

    def __call__(self) -> np.ndarray:
        """
        Calculate the covariance.

        Returns:
            numpy.ndarray: Covariance of the batches.

        Raises:
            NoValidSamplesError: If no valid samples are available.

        """
        if self.cov is None:
            raise NoValidSamplesError("No valid samples for calculating covariance.")
        return self.n_samples / (self.n_samples - self.ddof) * self.cov

    def __add__(self, other):
        self.merge_test(other, field="cov")
        if self.n_samples == 0:
            return other
        elif other.n_samples == 0:
            return self
        else:
            ret = BatchCov(ddof=self.ddof)
            ret.n_samples = self.n_samples + other.n_samples
            ret.mean1 = self.mean1 + other.mean1
            ret.mean2 = self.mean2 + other.mean2
            ret.cov = self.n_samples * self.cov + other.n_samples * other.cov
            ret.cov += self.n_samples * (self.mean1() - ret.mean1())[:, None] * (self.mean2() - ret.mean2())
            ret.cov += other.n_samples * (other.mean1() - ret.mean1())[:, None] * (other.mean2() - ret.mean2())
            ret.cov /= ret.n_samples
            return ret
