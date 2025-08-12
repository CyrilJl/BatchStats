import numpy as np

from .._misc import NoValidSamplesError
from ..base import BatchNanStat


class BatchNanSum(BatchNanStat):
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
        batch = self._process_batch(batch)
        if self.sum is None:
            self.sum = np.nansum(batch, axis=self.axis)
        else:
            self.sum += np.nansum(batch, axis=self.axis)
        return self

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
