import string

import numpy as np

from ._misc import NoValidSamplesError, UnequalSamplesNumber, any_nan, check_params
from .core import BatchStat


class BatchSum(BatchStat):
    """
    Class for calculating the sum of batches of data.

    The algorithm used is a simple cumulative sum. Each time a new batch is added,
    the sum of the new batch is added to the existing sum.

    .. code:: python

        import numpy as np
        from batchstats import BatchSum

        # create some data
        data1 = np.array([[1, 2], [3, 4]])
        data2 = np.array([[5, 6], [7, 8]])

        # create a BatchSum object
        bs = BatchSum()

        # update with the first batch
        bs.update_batch(data1)

        # update with the second batch
        bs.update_batch(data2)

        # get the sum
        total_sum = bs()

        # verify the result
        expected_sum = np.array([16, 20])
        np.testing.assert_allclose(total_sum, expected_sum)


    .. admonition:: Example with multiple axes and data > 2 dimensions

        .. code:: python

            # create some 3d data
            data1 = np.arange(24).reshape(2, 3, 4)
            data2 = np.arange(24, 48).reshape(2, 3, 4)

            # create a BatchSum object to sum over the last two axes
            bs = BatchSum(axis=(1, 2))

            # update with the first batch
            bs.update_batch(data1)

            # update with the second batch
            bs.update_batch(data2)

            # get the sum
            total_sum = bs()

            # verify the result
            expected_sum = data1.sum(axis=(1,2)) + data2.sum(axis=(1,2))
            np.testing.assert_allclose(total_sum, expected_sum)

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
            batch_sum = np.sum(a=valid_batch, axis=self.axis, keepdims=True)
            if self.sum is None:
                self.sum = batch_sum
            else:
                self.sum += batch_sum
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
            sum_ = self.sum.copy()
            if self.axis is not None:
                return sum_.squeeze(axis=self.axis)
            return sum_

    def __add__(self, other):
        self.merge_test(other, field="sum")
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

    The algorithm keeps track of the element-wise maximum. When a new batch is added,
    the element-wise maximum between the current maximum and the new batch's maximum is computed.

    .. code:: python

        import numpy as np
        from batchstats import BatchMax

        # create some data
        data1 = np.array([[1, 8], [3, 4]])
        data2 = np.array([[5, 6], [7, 2]])

        # create a BatchMax object
        bm = BatchMax()

        # update with the first batch
        bm.update_batch(data1)

        # update with the second batch
        bm.update_batch(data2)

        # get the maximum
        total_max = bm()

        # verify the result
        expected_max = np.array([7, 8])
        np.testing.assert_allclose(total_max, expected_max)


    .. admonition:: Example with multiple axes and data > 2 dimensions

        .. code:: python

            # create some 3d data
            data1 = np.arange(24).reshape(2, 3, 4)
            data2 = np.arange(24, 48).reshape(2, 3, 4)

            # create a BatchMax object to get the max over the last two axes
            bm = BatchMax(axis=(1, 2))

            # update with the first batch
            bm.update_batch(data1)

            # update with the second batch
            bm.update_batch(data2)

            # get the max
            total_max = bm()

            # verify the result
            expected_max = np.maximum(data1.max(axis=(1,2)), data2.max(axis=(1,2)))
            np.testing.assert_allclose(total_max, expected_max)

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
            batch_max = np.max(valid_batch, axis=self.axis, keepdims=True)
            if self.max is None:
                self.max = batch_max
            else:
                self.max = np.maximum(self.max, batch_max)
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
            max_ = self.max.copy()
            if self.axis is not None:
                return max_.squeeze(axis=self.axis)
            return max_

    def __add__(self, other):
        self.merge_test(other, field="max")
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

    The algorithm keeps track of the element-wise minimum. When a new batch is added,
    the element-wise minimum between the current minimum and the new batch's minimum is computed.

    .. code:: python

        import numpy as np
        from batchstats import BatchMin

        # create some data
        data1 = np.array([[1, 8], [3, 4]])
        data2 = np.array([[5, 6], [7, 2]])

        # create a BatchMin object
        bm = BatchMin()

        # update with the first batch
        bm.update_batch(data1)

        # update with the second batch
        bm.update_batch(data2)

        # get the minimum
        total_min = bm()

        # verify the result
        expected_min = np.array([1, 2])
        np.testing.assert_allclose(total_min, expected_min)


    .. admonition:: Example with multiple axes and data > 2 dimensions

        .. code:: python

            # create some 3d data
            data1 = np.arange(24).reshape(2, 3, 4)
            data2 = np.arange(24, 48).reshape(2, 3, 4)

            # create a BatchMin object to get the min over the last two axes
            bm = BatchMin(axis=(1, 2))

            # update with the first batch
            bm.update_batch(data1)

            # update with the second batch
            bm.update_batch(data2)

            # get the min
            total_min = bm()

            # verify the result
            expected_min = np.minimum(data1.min(axis=(1,2)), data2.min(axis=(1,2)))
            np.testing.assert_allclose(total_min, expected_min)

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
            batch_min = np.min(valid_batch, axis=self.axis, keepdims=True)
            if self.min is None:
                self.min = batch_min
            else:
                self.min = np.minimum(self.min, batch_min)
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
            min_ = self.min.copy()
            if self.axis is not None:
                return min_.squeeze(axis=self.axis)
            return min_

    def __add__(self, other):
        self.merge_test(other, field="min")
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

    The algorithm uses an incremental mean calculation. The new mean is computed from
    the previous mean, the new data, and the number of samples.

    .. code:: python

        import numpy as np
        from batchstats import BatchMean

        # create some data
        data1 = np.array([[1, 2], [3, 4]])
        data2 = np.array([[5, 6], [7, 8]])

        # create a BatchMean object
        bm = BatchMean()

        # update with the first batch
        bm.update_batch(data1)

        # update with the second batch
        bm.update_batch(data2)

        # get the mean
        total_mean = bm()

        # verify the result
        expected_mean = np.array([4., 5.])
        np.testing.assert_allclose(total_mean, expected_mean)


    .. admonition:: Example with multiple axes and data > 2 dimensions

        .. code:: python

            # create some 3d data
            data1 = np.arange(24).reshape(2, 3, 4)
            data2 = np.arange(24, 48).reshape(2, 3, 4)

            # create a BatchMean object to get the mean over the last two axes
            bm = BatchMean(axis=(1, 2))

            # update with the first batch
            bm.update_batch(data1)

            # update with the second batch
            bm.update_batch(data2)

            # get the mean
            total_mean = bm()

            # verify the result
            d = np.concatenate((data1, data2))
            expected_mean = d.mean(axis=(1,2))
            np.testing.assert_allclose(total_mean, expected_mean)

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
        n = self._get_n_samples_in_batch(valid_batch)
        if n > 0:
            if self.mean is None:
                self.mean = np.mean(valid_batch, axis=self.axis, keepdims=True)
            else:
                self.mean = (
                    (self.n_samples - n) * self.mean + np.sum(valid_batch, axis=self.axis, keepdims=True)
                ) / self.n_samples
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
            mean_ = self.mean.copy()
            if self.axis is not None:
                return mean_.squeeze(axis=self.axis)
            return mean_

    def __add__(self, other):
        self.merge_test(other, field="mean")
        if self.n_samples == 0:
            return other
        elif other.n_samples == 0:
            return self
        else:
            ret = BatchMean(axis=self.axis)
            ret.n_samples = self.n_samples + other.n_samples
            ret.mean = self.n_samples * self.mean + other.n_samples * other.mean
            ret.mean /= ret.n_samples
            return ret


class BatchPeakToPeak(BatchStat):
    """
    Class for calculating the peak-to-peak (max - min) of batches of data.

    This class uses `BatchMax` and `BatchMin` internally to keep track of the
    element-wise maximum and minimum values. The peak-to-peak value is the
    difference between the maximum and minimum.

    .. code:: python

        import numpy as np
        from batchstats import BatchPeakToPeak

        # create some data
        data1 = np.array([[1, 8], [3, 4]])
        data2 = np.array([[5, 6], [7, 2]])

        # create a BatchPeakToPeak object
        bpp = BatchPeakToPeak()

        # update with the first batch
        bpp.update_batch(data1)

        # update with the second batch
        bpp.update_batch(data2)

        # get the peak-to-peak
        total_ptp = bpp()

        # verify the result
        expected_ptp = np.array([6, 6])
        np.testing.assert_allclose(total_ptp, expected_ptp)


    .. admonition:: Example with multiple axes and data > 2 dimensions

        .. code:: python

            # create some 3d data
            data1 = np.arange(24).reshape(2, 3, 4)
            data2 = np.arange(24, 48).reshape(2, 3, 4)

            # create a BatchPeakToPeak object to get the ptp over the last two axes
            bpp = BatchPeakToPeak(axis=(1, 2))

            # update with the first batch
            bpp.update_batch(data1)

            # update with the second batch
            bpp.update_batch(data2)

            # get the ptp
            total_ptp = bpp()

            # verify the result
            d = np.concatenate((data1, data2))
            expected_ptp = d.max(axis=(1,2)) - d.min(axis=(1,2))
            np.testing.assert_allclose(total_ptp, expected_ptp)

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
        self.batchmax.update_batch(valid_batch, assume_valid=True)
        self.batchmin.update_batch(valid_batch, assume_valid=True)
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
        self.batchmax.merge_test(other.batchmax, field="max")
        self.batchmin.merge_test(other.batchmin, field="min")
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


class BatchVar(BatchStat):
    """
    Class for calculating the variance of batches of data.

    The algorithm is an implementation of Welford's online algorithm for computing variance.
    It is numerically stable and avoids a two-pass approach.

    Args:
        ddof (int, optional): Means Delta Degrees of Freedom. The divisor used in calculations is N - ddof, where N represents the number of elements. By default ddof is zero.

    .. code:: python

        import numpy as np
        from batchstats import BatchVar

        # create some data
        data1 = np.array([[1, 2], [3, 4]])
        data2 = np.array([[5, 6], [7, 8]])

        # create a BatchVar object
        bv = BatchVar()

        # update with the first batch
        bv.update_batch(data1)

        # update with the second batch
        bv.update_batch(data2)

        # get the variance
        total_var = bv()

        # verify the result
        expected_var = np.array([5., 5.])
        np.testing.assert_allclose(total_var, expected_var)


    .. admonition:: Example with multiple axes and data > 2 dimensions

        .. code:: python

            # create some 3d data
            data1 = np.arange(24).reshape(2, 3, 4)
            data2 = np.arange(24, 48).reshape(2, 3, 4)

            # create a BatchVar object to get the var over the last two axes
            bv = BatchVar(axis=(1, 2))

            # update with the first batch
            bv.update_batch(data1)

            # update with the second batch
            bv.update_batch(data2)

            # get the var
            total_var = bv()

            # verify the result
            d = np.concatenate((data1, data2))
            expected_var = d.var(axis=(1,2))
            np.testing.assert_allclose(total_var, expected_var)

    """

    def __init__(self, axis=0, ddof=0):
        super().__init__(axis=axis)
        self.mean = BatchMean(axis=axis)
        self.var = None
        self.ddof = check_params(param=ddof, types=int)

    def compute_incremental_variance(self, v, p, u):
        """
        Compute incremental variance.
        For v 2D and p/u 1D, equivalent to ``((v-p).T@(v-u)).sum(axis=0)``

        Args:
            v (numpy.ndarray): Input data.
            p (numpy.ndarray): Previous mean.
            u (numpy.ndarray): Updated mean.

        Returns:
            numpy.ndarray: Incremental variance.

        """
        return np.sum((v - p) * (v - u), axis=self.axis, keepdims=True)

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
        n = self._get_n_samples_in_batch(valid_batch)
        if n > 0:
            if self.var is None:
                self.mean.update_batch(valid_batch, assume_valid=True)
                self.var = np.var(valid_batch, axis=self.axis, keepdims=True)
            else:
                previous_mean = self.mean.mean
                self.mean.update_batch(valid_batch, assume_valid=True)
                updated_mean = self.mean.mean
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
        var_ = (self.n_samples / (self.n_samples - self.ddof)) * self.var
        if self.axis is not None:
            return var_.squeeze(axis=self.axis)
        return var_

    def __add__(self, other):
        self.merge_test(other, field="var")
        if self.n_samples == 0:
            return other
        elif other.n_samples == 0:
            return self
        else:
            ret = BatchVar(axis=self.axis, ddof=self.ddof)
            ret.n_samples = self.n_samples + other.n_samples
            ret.mean = self.mean + other.mean
            ret.var = self.n_samples * self.var + other.n_samples * other.var

            ret.var += self.n_samples * (self.mean.mean - ret.mean.mean) ** 2
            ret.var += other.n_samples * (other.mean.mean - ret.mean.mean) ** 2

            ret.var /= ret.n_samples
            return ret


class BatchStd(BatchStat):
    """
    Class for calculating the standard deviation of batches of data.

    This class uses `BatchVar` internally and takes the square root of the result.

    Args:
        ddof (int, optional): Means Delta Degrees of Freedom. The divisor used in calculations is N - ddof, where N represents the number of elements. By default ddof is zero.

    .. code:: python

        import numpy as np
        from batchstats import BatchStd

        # create some data
        data1 = np.array([[1, 2], [3, 4]])
        data2 = np.array([[5, 6], [7, 8]])

        # create a BatchStd object
        bs = BatchStd()

        # update with the first batch
        bs.update_batch(data1)

        # update with the second batch
        bs.update_batch(data2)

        # get the standard deviation
        total_std = bs()

        # verify the result
        expected_std = np.array([2.23606798, 2.23606798])
        np.testing.assert_allclose(total_std, expected_std)


    .. admonition:: Example with multiple axes and data > 2 dimensions

        .. code:: python

            # create some 3d data
            data1 = np.arange(24).reshape(2, 3, 4)
            data2 = np.arange(24, 48).reshape(2, 3, 4)

            # create a BatchStd object to get the std over the last two axes
            bs = BatchStd(axis=(1, 2))

            # update with the first batch
            bs.update_batch(data1)

            # update with the second batch
            bs.update_batch(data2)

            # get the std
            total_std = bs()

            # verify the result
            d = np.concatenate((data1, data2))
            expected_std = d.std(axis=(1,2))
            np.testing.assert_allclose(total_std, expected_std)

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
        self.var.merge_test(other.var, field="var")
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
