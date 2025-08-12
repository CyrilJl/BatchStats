import numpy as np

from ..base import BatchStat
from .max import BatchMax
from .min import BatchMin


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
