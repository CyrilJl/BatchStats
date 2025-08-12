import numpy as np

from ..base import BatchStat
from .var import BatchVar


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
