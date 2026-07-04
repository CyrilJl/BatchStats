import string

import numpy as np

from .._misc import NoValidSamplesError, check_params
from ..base import BatchStat
from .mean import BatchMean


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

    @staticmethod
    def _get_axis_size(arr, axis):
        """Computes the product of dimensions for a given axis tuple."""
        axis_ = np.atleast_1d(axis)
        size = 1
        for ax in axis_:
            size *= arr.shape[ax]
        return size

    @classmethod
    def init_var(cls, v, vm, axis, v_sum=None):
        ret = cls.compute_incremental_variance(v, vm, vm, axis=axis, v_sum=v_sum)
        size = cls._get_axis_size(v, axis)
        ret /= size
        return ret

    @classmethod
    def compute_incremental_variance(cls, v, p, u, axis, v_sum=None):
        axis_ = tuple(np.atleast_1d(axis))
        v_ndim = v.ndim
        v_indices = string.ascii_lowercase[:v_ndim]
        pu_indices = "".join([v_indices[k] for k in range(v_ndim) if k not in axis_])
        ret = np.einsum(f"{v_indices},{v_indices}->{pu_indices}", v, v)
        # sum_i (p+u)*v_i factors into (p+u)*sum_i(v_i): reuse the batch sum already
        # computed for the mean update instead of a second O(batch) pass
        if v_sum is None:
            v_sum = np.add.reduce(v, axis=axis_)
        else:
            v_sum = v_sum.squeeze(axis=axis_) if v_sum.ndim == v_ndim else v_sum
        ret -= (p + u) * v_sum
        size = cls._get_axis_size(v, axis)
        ret += size * p * u
        if ret.ndim < v_ndim:
            ret = np.expand_dims(ret, axis=tuple(axis_))
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
        n = self._get_n_samples_in_batch(valid_batch)
        if n > 0:
            # single pass for the batch sum, shared by the mean and variance updates
            batch_sum = np.sum(valid_batch, axis=self.axis, keepdims=True)
            if self.var is None:
                self.mean._update_from_sum(batch_sum, n, count_samples=True)
                self.var = self.init_var(v=valid_batch, vm=self.mean._value(), axis=self.axis, v_sum=batch_sum)
            else:
                previous_mean = self.mean._value()
                self.mean._update_from_sum(batch_sum, n, count_samples=True)
                updated_mean = self.mean._value()
                incremental_variance = self.compute_incremental_variance(
                    valid_batch,
                    previous_mean,
                    updated_mean,
                    axis=self.axis,
                    v_sum=batch_sum,
                )
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
