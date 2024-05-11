import numpy as np


class NoValidSamplesError(ValueError):
    """
    Error raised when there are no valid samples for calculation.
    """
    pass


class UnequalSamplesNumber(ValueError):
    """
    Error raised when two batches have unequal lengths.
    """
    pass


def any_nan(x, axis=None):
    """
    Check if there are any NaN values in the input array.

    Args:
        x (numpy.ndarray): Input array.
        axis (int or tuple of ints, optional): Axis or axes along which to operate. Default is None.

    Returns:
        numpy.ndarray: Boolean array indicating NaN presence.

    """
    return np.isnan(np.add.reduce(array=x, axis=axis))


class BatchStat:
    """
    Base class for calculating statistics over batches of data.

    Attributes:
        n_samples (int): Total number of samples processed.

    """

    def __init__(self):
        self.n_samples = 0

    def _process_batch(self, batch, assume_valid=False):
        """
        Process the input batch, handling NaN values if necessary.

        Args:
            batch (numpy.ndarray): Input batch.
            assume_valid (bool, optional): If True, assumes all elements in the batch are valid. Default is False.

        Returns:
            numpy.ndarray: Processed batch.

        """
        batch = np.atleast_2d(np.asarray(batch))
        if assume_valid:
            self.n_samples += len(batch)
            return batch
        else:
            axis = tuple(range(1, batch.ndim))
            nan_mask = any_nan(batch, axis=axis)
            if nan_mask.any():
                valid_batch = batch[~nan_mask]
            else:
                valid_batch = batch
            self.n_samples += len(valid_batch)
            return valid_batch

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class BatchSum(BatchStat):
    """
    Class for calculating the sum of batches of data.

    """

    def __init__(self):
        super().__init__()
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
                self.sum = np.sum(a=valid_batch, axis=0)
            else:
                self.sum += np.sum(a=valid_batch, axis=0)
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


class BatchMax(BatchStat):
    """
    Class for calculating the maximum of batches of data.

    """

    def __init__(self):
        super().__init__()
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
                self.max = np.max(valid_batch, axis=0)
            else:
                np.maximum(self.max, np.max(valid_batch, axis=0), out=self.max)
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


class BatchMin(BatchStat):
    """
    Class for calculating the minimum of batches of data.

    """

    def __init__(self):
        super().__init__()
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
                self.min = np.min(valid_batch, axis=0)
            else:
                np.minimum(self.min, np.min(valid_batch, axis=0), out=self.min)
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


class BatchMean(BatchStat):
    """
    Class for calculating the mean of batches of data.

    """

    def __init__(self):
        super().__init__()
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
                self.mean = np.mean(valid_batch, axis=0)
            else:
                self.mean = ((self.n_samples - n) * self.mean + np.sum(valid_batch, axis=0)) / self.n_samples
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


class BatchVar(BatchMean):
    """
    Class for calculating the variance of batches of data.

    """

    def __init__(self, ddof=0):
        super().__init__()
        self.mean = BatchMean()
        self.var = None
        self.ddof = ddof

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

        Args:
            v (numpy.ndarray): Input data.
            p (numpy.ndarray): Previous mean.
            u (numpy.ndarray): Updated mean.

        Returns:
            numpy.ndarray: Incremental variance.

        """
        ret = np.einsum('ij,ij->j', v, v)
        ret -= np.einsum('j,ij->j', p + u, v)
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


class BatchCov(BatchStat):
    """
    Class for calculating the covariance of batches of data.

    """

    def __init__(self, ddof=0):
        super().__init__()
        self.mean1 = BatchMean()
        self.mean2 = BatchMean()
        self.cov = None
        self.ddof = ddof

    def _process_batch(self, batch1, batch2, assume_valid=False):
        """
        Process the input batches, handling NaN values if necessary.

        Args:
            batch1 (numpy.ndarray): Input batch 1.
            batch2 (numpy.ndarray): Input batch 2.
            assume_valid (bool, optional): If True, assumes all elements in the batches are valid. Default is False.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray]: Processed batches 1 and 2.

        Raises:
            UnequalSamplesNumber: If the batches have unequal lengths.

        """
        batch1, batch2 = np.atleast_2d(np.asarray(batch1)), np.atleast_2d(np.asarray(batch2))
        if assume_valid:
            self.n_samples += len(batch1)
            return batch1, batch2
        else:
            if len(batch1) != len(batch2):
                raise UnequalSamplesNumber("batch1 and batch2 don't have the same lengths.")
            mask = ~any_nan(batch1) & ~any_nan(batch2)
            n = mask.sum()
            self.n_samples += n
            if np.all(mask):
                return batch1, batch2
            else:
                return batch1[mask], batch2[mask]

    def update_batch(self, batch1, batch2, assume_valid=False):
        """
        Update the covariance with new batches of data.

        Args:
            batch1 (numpy.ndarray): Input batch 1.
            batch2 (numpy.ndarray): Input batch 2.
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
