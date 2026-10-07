import numpy as np

from .._misc import NoValidSamplesError
from ..base import BatchNanStat
from ..base.state import StateMixin


class BatchNanSum(StateMixin, BatchNanStat):
    """
    Class for calculating the sum of batches of data that can contain NaN values.

    The algorithm is a simple cumulative sum, ignoring NaN values.

    .. code:: python

        import numpy as np
        from batchstats import BatchNanSum

        # create some data with NaNs
        data1 = np.array([[1, 2], [3, np.nan]])
        data2 = np.array([[5, 6], [np.nan, 8]])

        # create a BatchNanSum object
        bns = BatchNanSum()

        # update with the first batch
        bns.update_batch(data1)

        # update with the second batch
        bns.update_batch(data2)

        # get the sum
        total_sum = bns()

        # verify the result
        expected_sum = np.array([9., 16.])
        np.testing.assert_allclose(total_sum, expected_sum)


    .. admonition:: Example with multiple axes and data > 2 dimensions

        .. code:: python

            # create some 3d data with NaNs
            data1 = np.arange(24).reshape(2, 3, 4).astype(float)
            data1[0, 1, 1] = np.nan
            data2 = np.arange(24, 48).reshape(2, 3, 4).astype(float)
            data2[1, 2, 0] = np.nan


            # create a BatchNanSum object to sum over the last two axes
            bns = BatchNanSum(axis=(1, 2))

            # update with the first batch
            bns.update_batch(data1)

            # update with the second batch
            bns.update_batch(data2)

            # get the sum
            total_sum = bns()

            # verify the result
            d = np.concatenate((data1, data2))
            expected_sum = np.nansum(d, axis=(1,2))
            np.testing.assert_allclose(total_sum, expected_sum)

    """

    def __init__(self, axis=0):
        """
        Initialize the BatchNanSum object.
        """
        super().__init__(axis=axis)
        self.sum = None

    def update_batch(self, batch):
        """
        Update the sum with a new batch of data that can contain NaN values.

        Args:
            batch (numpy.ndarray): Input batch.

        Returns:
            BatchNanSum: Updated BatchNanSum object.

        """
        batch = np.atleast_2d(np.asarray(batch))
        invalid = np.isnan(batch) if batch.dtype.kind not in "biu" else None
        if invalid is not None and invalid.any():
            # Reuse the mask for both reductions without a batch-sized numeric
            # copy (np.nansum replaces NaNs in a temporary array).
            np.logical_not(invalid, out=invalid)
            batch_sum = np.asarray(np.sum(batch, axis=self.axis, where=invalid))
            n_valid = np.asarray(np.count_nonzero(invalid, axis=self.axis))
        else:
            batch_sum = np.asarray(np.sum(batch, axis=self.axis))
            n = batch.size // batch_sum.size if batch_sum.size else 0
            n_valid = np.full(batch_sum.shape, n, dtype=np.int64)
        if self.sum is not None and self.sum.shape != batch_sum.shape:
            raise ValueError("Non-reduced dimensions must have the same shape.")
        self.sum = batch_sum if self.sum is None else self.sum + batch_sum
        self._add_valid_count(n_valid)
        return self

    def __add__(self, other):
        from ..base import BatchStat

        BatchStat.merge_test(self, other, field="sum")
        result = type(self)(axis=self.axis)
        if self.sum is None or other.sum is None:
            source = other if self.sum is None else self
            if source.sum is not None:
                result.sum = source.sum.copy()
                result.n_samples = source.n_samples.copy()
        else:
            result.sum = self.sum + other.sum
            result.n_samples = self.n_samples + other.n_samples
        return result

    def __call__(self):
        """
        Calculate the sum of the batches that can contain NaN values.

        Returns:
            numpy.ndarray: Sum of the batches.

        Raises:
            NoValidSamplesError: If no valid samples are available.

        """
        if self.sum is None:
            raise NoValidSamplesError()
        else:
            return np.where(self.n_samples > 0, self.sum, np.nan)
