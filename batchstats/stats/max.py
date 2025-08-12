import numpy as np

from .._misc import NoValidSamplesError
from ..base import BatchStat


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
