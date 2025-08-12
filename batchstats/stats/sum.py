import numpy as np

from .._misc import NoValidSamplesError
from ..base import BatchStat


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
