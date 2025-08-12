from ..base import BatchNanStat
from .nan_sum import BatchNanSum


class BatchNanMean(BatchNanStat):
    """
    Class for calculating the mean of batches of data that can contain NaN values.

    The algorithm uses `BatchNanSum` to compute the sum and the number of valid samples,
    then divides the sum by the number of samples to get the mean.

    .. code:: python

        import numpy as np
        from batchstats import BatchNanMean

        # create some data with NaNs
        data1 = np.array([[1, 2], [3, np.nan]])
        data2 = np.array([[5, 6], [np.nan, 8]])

        # create a BatchNanMean object
        bnm = BatchNanMean()

        # update with the first batch
        bnm.update_batch(data1)

        # update with the second batch
        bnm.update_batch(data2)

        # get the mean
        total_mean = bnm()

        # verify the result
        expected_mean = np.array([3., 5.33333333])
        np.testing.assert_allclose(total_mean, expected_mean)


    .. admonition:: Example with multiple axes and data > 2 dimensions

        .. code:: python

            # create some 3d data with NaNs
            data1 = np.arange(24).reshape(2, 3, 4).astype(float)
            data1[0, 1, 1] = np.nan
            data2 = np.arange(24, 48).reshape(2, 3, 4).astype(float)
            data2[1, 2, 0] = np.nan

            # create a BatchNanMean object to get the mean over the last two axes
            bnm = BatchNanMean(axis=(1, 2))

            # update with the first batch
            bnm.update_batch(data1)

            # update with the second batch
            bnm.update_batch(data2)

            # get the mean
            total_mean = bnm()

            # verify the result
            d = np.concatenate((data1, data2))
            expected_mean = np.nanmean(d, axis=(1,2))
            np.testing.assert_allclose(total_mean, expected_mean)

    """

    def __init__(self, axis=0):
        """
        Initialize the BatchNanMean object.
        """
        super().__init__(axis=axis)
        self.sum = BatchNanSum(axis=axis)

    def update_batch(self, batch):
        """
        Update the mean with a new batch of data that can contain NaN values.

        Args:
            batch (numpy.ndarray): Input batch.

        Returns:
            BatchNanMean: Updated BatchNanMean object.

        """
        self.sum.update_batch(batch)
        return self

    def __call__(self):
        """
        Calculate the mean of the batches that can contain NaN values.

        Returns:
            numpy.ndarray: Mean of the batches.

        """
        return self.sum() / self.sum.n_samples
