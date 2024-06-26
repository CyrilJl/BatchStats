import string

import numpy as np

from ._misc import NoValidSamplesError, UnequalSamplesNumber, any_nan, check_params
from .core import BatchStat


class BatchSum(BatchStat):
    """
    Class for calculating the sum of batches of data.

    """

    def __init__(self, axis=0):
        super().__init__(axis=axis)
        self.sum = None

    def update_batch(self, batch, assume_valid=False):
        """
        Update the sum with a new batch of data.

        Args:
            batch (numpy.ndarray): Input batch.
            assume_valid (bool, optional): If True, assumes all elements in the batch are valid. Default is False.

        Returns:
            BatchSum: Updated BatchSum object.

        """
        valid_batch = self._process_batch(batch=batch, assume_valid=assume_valid)
        n = len(valid_batch)
        if n > 0:
            if self.sum is None:
                self.sum = np.sum(a=valid_batch, axis=self.axis)
            else:
                self.sum += np.sum(a=valid_batch, axis=self.axis)
        return self

    def __call__(self) -> np.ndarray:
        """
        Calculate the sum.

        Returns:
            numpy.ndarray: Sum of the batches.

        Raises:
            NoValidSamplesError: If no valid samples are available.

        """
        if self.sum is None:
            raise NoValidSamplesError("No valid samples for calculating sum.")
        else:
            return self.sum.copy()

    def __add__(self, other):
        self.merge_test(other, field='sum')
        if self.n_samples == 0:
            return other
        elif other.n_samples == 0:
            return self
        else:
            ret = BatchSum(axis=self.axis)
            ret.n_samples = self.n_samples + other.n_samples
            ret.sum = self.sum + other.sum
            return ret


class BatchMax(BatchStat):
    """
    Class for calculating the maximum of batches of data.

    """

    def __init__(self, axis=0):
        super().__init__(axis=axis)
        self.max = None

    def update_batch(self, batch, assume_valid=False):
        """
        Update the maximum with a new batch of data.

        Args:
            batch (numpy.ndarray): Input batch.
            assume_valid (bool, optional): If True, assumes all elements in the batch are valid. Default is False.

        Returns:
            BatchMax: Updated BatchMax object.

        """
        valid_batch = self._process_batch(batch=batch, assume_valid=assume_valid)
        n = len(valid_batch)
        if n > 0:
            if self.max is None:
                self.max = np.max(valid_batch, axis=self.axis)
            else:
                np.maximum(self.max, np.max(valid_batch, axis=self.axis), out=self.max)
        return self

    def __call__(self) -> np.ndarray:
        """
        Calculate the maximum.

        Returns:
            numpy.ndarray: Maximum of the batches.

        Raises:
            NoValidSamplesError: If no valid samples are available.

        """
        if self.max is None:
            raise NoValidSamplesError("No valid samples for calculating max.")
        else:
            return self.max.copy()

    def __add__(self, other):
        self.merge_test(other, field='max')
        if self.n_samples == 0:
            return other
        elif other.n_samples == 0:
            return self
        else:
            ret = BatchMax(axis=self.axis)
            ret.n_samples = self.n_samples + other.n_samples
            ret.max = np.maximum(self.max, other.max)
            return ret


class BatchMin(BatchStat):
    """
    Class for calculating the minimum of batches of data.

    """

    def __init__(self, axis=0):
        super().__init__(axis=axis)
        self.min = None

    def update_batch(self, batch, assume_valid=False):
        """
        Update the minimum with a new batch of data.

        Args:
            batch (numpy.ndarray): Input batch.
            assume_valid (bool, optional): If True, assumes all elements in the batch are valid. Default is False.

        Returns:
            BatchMin: Updated BatchMin object.

        """
        valid_batch = self._process_batch(batch=batch, assume_valid=assume_valid)
        n = len(valid_batch)
        if n > 0:
            if self.min is None:
                self.min = np.min(valid_batch, axis=self.axis)
            else:
                np.minimum(self.min, np.min(valid_batch, axis=self.axis), out=self.min)
        return self

    def __call__(self) -> np.ndarray:
        """
        Calculate the minimum.

        Returns:
            numpy.ndarray: Minimum of the batches.

        Raises:
            NoValidSamplesError: If no valid samples are available.

        """
        if self.min is None:
            raise NoValidSamplesError("No valid samples for calculating min.")
        else:
            return self.min.copy()

    def __add__(self, other):
        self.merge_test(other, field='min')
        if self.n_samples == 0:
            return other
        elif other.n_samples == 0:
            return self
        else:
            ret = BatchMin(axis=self.axis)
            ret.n_samples = self.n_samples + other.n_samples
            ret.min = np.minimum(self.min, other.min)
            return ret


class BatchMean(BatchStat):
    """
    Class for calculating the mean of batches of data.

    """

    def __init__(self, axis=0):
        super().__init__(axis=axis)
        self.mean = None

    def update_batch(self, batch, assume_valid=False):
        """
        Update the mean with a new batch of data.

        Args:
            batch (numpy.ndarray): Input batch.
            assume_valid (bool, optional): If True, assumes all elements in the batch are valid. Default is False.

        Returns:
            BatchMean: Updated BatchMean object.

        """
        valid_batch = self._process_batch(batch=batch, assume_valid=assume_valid)
        n = len(valid_batch)
        if n > 0:
            if self.mean is None:
                self.mean = np.mean(valid_batch, axis=self.axis)
            else:
                mean_batch = np.mean(valid_batch, axis=self.axis)
                self.mean = ((self.n_samples - n) * self.mean + np.sum(valid_batch-mean_batch, axis=self.axis) + n*mean_batch) / self.n_samples
        return self

    def __call__(self) -> np.ndarray:
        """
        Calculate the mean.

        Returns:
            numpy.ndarray: Mean of the batches.

        Raises:
            NoValidSamplesError: If no valid samples are available.

        """
        if self.mean is None:
            raise NoValidSamplesError("No valid samples for calculating mean.")
        else:
            return self.mean.copy()

    def __add__(self, other):
        self.merge_test(other, field='mean')
        if self.n_samples == 0:
            return other
        elif other.n_samples == 0:
            return self
        else:
            ret = BatchMean(axis=self.axis)
            ret.n_samples = self.n_samples + other.n_samples
            ret.mean = self.n_samples*self.mean + other.n_samples*other.mean
            ret.mean /= ret.n_samples
            return ret


class BatchPeakToPeak(BatchStat):
    """
    Class for calculating the peak-to-peak (max - min) of batches of data.
    """

    def __init__(self, axis=0):
        super().__init__(axis=axis)
        self.batchmax = BatchMax(axis=axis)
        self.batchmin = BatchMin(axis=axis)

    def update_batch(self, batch, assume_valid=False):
        """
        Update the peak-to-peak with a new batch of data.

        Args:
            batch (numpy.ndarray): Input batch.
            assume_valid (bool, optional): If True, assumes all elements in the batch are valid. Default is False.

        Returns:
            BatchPeakToPeak: Updated BatchPeakToPeak object.

        """
        valid_batch = self._process_batch(batch=batch, assume_valid=assume_valid)
        self.batchmax.update_batch(batch, assume_valid=True)
        self.batchmin.update_batch(batch, assume_valid=True)
        return self

    def __call__(self) -> np.ndarray:
        """
        Calculate the peak-to-peak.

        Returns:
            numpy.ndarray: Peak-to-peak of the batches.

        Raises:
            NoValidSamplesError: If no valid samples are available.

        """
        return self.batchmax() - self.batchmin()

    def __add__(self, other):
        self.batchmax.merge_test(other.batchmax, field='max')
        self.batchmin.merge_test(other.batchmin, field='min')
        if self.n_samples == 0:
            return other
        elif other.n_samples == 0:
            return self
        else:
            ret = BatchPeakToPeak(axis=self.axis)
            ret.n_samples = self.n_samples + other.n_samples
            ret.batchmax = self.batchmax + other.batchmax
            ret.batchmin = self.batchmin + other.batchmin
            return ret


class BatchVar(BatchMean):
    """
    Class for calculating the variance of batches of data.

    Args:
        ddof (int, optional): Means Delta Degrees of Freedom. The divisor used in calculations is N - ddof, where N represents the number of elements. By default ddof is zero.
    """

    def __init__(self, axis=0, ddof=0):
        super().__init__(axis=axis)
        self.mean = BatchMean(axis=axis)
        self.var = None
        self.ddof = check_params(param=ddof, types=int)

    @classmethod
    def init_var(cls, v, vm):
        """
        Initialize variance.

        Args:
            v (numpy.ndarray): Input data.
            vm (numpy.ndarray): Mean of the input data.

        Returns:
            numpy.ndarray: Initialized variance.

        """
        ret = cls.compute_incremental_variance(v, vm, vm)
        ret /= len(v)
        return ret

    @staticmethod
    def compute_incremental_variance(v, p, u):
        """
        Compute incremental variance.
        For v 2D and p/u 1D, equivalent to ``((v-p).T@(v-u)).sum(axis=0)`` or
        ``np.einsum('ji,ji->i', v - p, v - u)``. Faster and less memory consumer because
        no intermediate 2D array are created.

        Args:
            v (numpy.ndarray): Input data.
            p (numpy.ndarray): Previous mean.
            u (numpy.ndarray): Updated mean.

        Returns:
            numpy.ndarray: Incremental variance.

        """
        alphabet = string.ascii_lowercase
        ndim = v.ndim
        assert p.ndim == u.ndim == ndim-1
        ij, j = alphabet[:ndim], alphabet[1:ndim]

        ret = np.einsum(f'{ij},{ij}->{j}', v, v)
        ret -= np.einsum(f'{j},{ij}->{j}', p + u, v)
        ret += len(v)*p*u
        return ret

    def update_batch(self, batch, assume_valid=False):
        """
        Update the variance with a new batch of data.

        Args:
            batch (numpy.ndarray): Input batch.
            assume_valid (bool, optional): If True, assumes all elements in the batch are valid. Default is False.

        Returns:
            BatchVar: Updated BatchVar object.

        """
        valid_batch = self._process_batch(batch, assume_valid=assume_valid)
        n = len(valid_batch)
        if n > 0:
            if self.var is None:
                self.mean.update_batch(valid_batch, assume_valid=True)
                self.var = self.init_var(valid_batch, self.mean())
            else:
                previous_mean = self.mean()
                self.mean.update_batch(valid_batch, assume_valid=True)
                updated_mean = self.mean()
                incremental_variance = self.compute_incremental_variance(valid_batch, previous_mean, updated_mean)
                self.var = ((self.n_samples - n) * self.var + incremental_variance) / self.n_samples
        return self

    def __call__(self) -> np.ndarray:
        """
        Calculate the variance.

        Returns:
            numpy.ndarray: Variance of the batches.

        Raises:
            NoValidSamplesError: If no valid samples are available.

        """
        if self.var is None:
            raise NoValidSamplesError("No valid samples for calculating variance.")
        return (self.n_samples / (self.n_samples - self.ddof)) * self.var

    def __add__(self, other):
        self.merge_test(other, field='var')
        if self.n_samples == 0:
            return other
        elif other.n_samples == 0:
            return self
        else:
            ret = BatchVar(axis=self.axis, ddof=self.ddof)
            ret.n_samples = self.n_samples + other.n_samples
            ret.mean = self.mean + other.mean
            ret.var = self.n_samples*self.var + other.n_samples*other.var

            ret.var += self.n_samples*(self.mean()-ret.mean())**2
            ret.var += other.n_samples*(other.mean()-ret.mean())**2

            ret.var /= ret.n_samples
            return ret


class BatchStd(BatchStat):
    """Class for calculating the standard deviation of batches of data.

    Args:
        ddof (int, optional): Means Delta Degrees of Freedom. The divisor used in calculations is N - ddof, where N represents the number of elements. By default ddof is zero.
    """

    def __init__(self, axis=0, ddof=0):
        super().__init__(axis=axis)
        self.ddof = ddof
        self.var = BatchVar(axis=axis, ddof=ddof)

    def update_batch(self, batch, assume_valid=False):
        """Update the standard deviation with a new batch of data.

        Args:
            batch (numpy.ndarray): Input batch.
            assume_valid (bool, optional): If True, assumes all elements in the batch are valid. Default is False.

        Returns:
            BatchStd: Updated BatchStd object.
        """
        batch = self._process_batch(batch=batch, assume_valid=assume_valid)
        self.var.update_batch(batch=batch, assume_valid=True)
        return self

    def __call__(self) -> np.ndarray:
        """Calculate the standard deviation.

        Returns:
            numpy.ndarray: Standard deviation of the batches.

        Raises:
            NoValidSamplesError: If no valid samples are available.
        """
        return np.sqrt(self.var())

    def __add__(self, other):
        self.var.merge_test(other.var, field='var')
        if self.n_samples == 0:
            return other
        elif other.n_samples == 0:
            return self
        else:
            ret = BatchStd(axis=self.axis, ddof=self.ddof)
            ret.n_samples = self.n_samples + other.n_samples
            ret.var = self.var + other.var
            return ret


class BatchCov(BatchStat):
    """
    Class for calculating the covariance of batches of data.

    Args:
        ddof (int, optional): Means Delta Degrees of Freedom. The divisor used in calculations is N - ddof, where N represents the number of elements. By default ddof is zero.
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
                    self.cov = (batch1-self.mean1()).T@((batch2-self.mean2())/n)
            else:
                m1 = self.mean1()
                self.mean2.update_batch(batch2, assume_valid=True)
                m2 = self.mean2()
                self.cov += (batch1-m1).T@((batch2-m2)/self.mean1.n_samples)
                self.cov *= (self.mean1.n_samples/self.mean2.n_samples)
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
        return self.n_samples/(self.n_samples - self.ddof)*self.cov

    def __add__(self, other):
        self.merge_test(other, field='cov')
        if self.n_samples == 0:
            return other
        elif other.n_samples == 0:
            return self
        else:
            ret = BatchCov(ddof=self.ddof)
            ret.n_samples = self.n_samples + other.n_samples
            ret.mean1 = self.mean1 + other.mean1
            ret.mean2 = self.mean2 + other.mean2
            ret.cov = self.n_samples*self.cov + other.n_samples*other.cov
            ret.cov += self.n_samples*(self.mean1()-ret.mean1())[:, None]*(self.mean2()-ret.mean2())
            ret.cov += other.n_samples*(other.mean1()-ret.mean1())[:, None]*(other.mean2()-ret.mean2())
            ret.cov /= ret.n_samples
            return ret
