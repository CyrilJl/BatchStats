import numpy as np

from .._misc import NoValidSamplesError
from ..base import BatchStat


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
