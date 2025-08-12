import numpy as np

from .._misc import NoValidSamplesError
from ..base import BatchStat


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
