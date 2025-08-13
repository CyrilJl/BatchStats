import numpy as np

from ..base import BatchNanStat
from .nan_max import BatchNanMax
from .nan_min import BatchNanMin


class BatchNanPeakToPeak(BatchNanStat):
    """
    Class for calculating the peak-to-peak (max - min) of batches of data that can contain NaN values.

    This class uses `BatchNanMax` and `BatchNanMin` internally to keep track of the
    element-wise maximum and minimum values. The peak-to-peak value is the
    difference between the maximum and minimum.

    .. code:: python

        import numpy as np
        from batchstats import BatchNanPeakToPeak

        # create some data with NaNs
        data1 = np.array([[1, 2], [3, np.nan]])
        data2 = np.array([[5, 6], [np.nan, 8]])

        # create a BatchNanPeakToPeak object
        bnpp = BatchNanPeakToPeak()

        # update with the first batch
        bnpp.update_batch(data1)

        # update with the second batch
        bnpp.update_batch(data2)

        # get the peak-to-peak
        total_ptp = bnpp()

        # verify the result
        expected_ptp = np.array([4., 6.])
        np.testing.assert_allclose(total_ptp, expected_ptp)


    .. admonition:: Example with multiple axes and data > 2 dimensions

        .. code:: python

            # create some 3d data with NaNs
            data1 = np.arange(24).reshape(2, 3, 4).astype(float)
            data1[0, 1, 1] = np.nan
            data2 = np.arange(24, 48).reshape(2, 3, 4).astype(float)
            data2[1, 2, 0] = np.nan

            # create a BatchNanPeakToPeak object to get the ptp over the last two axes
            bnpp = BatchNanPeakToPeak(axis=(1, 2))

            # update with the first batch
            bnpp.update_batch(data1)

            # update with the second batch
            bnpp.update_batch(data2)

            # get the ptp
            total_ptp = bnpp()

            # verify the result
            d = np.concatenate((data1, data2))
            expected_ptp = np.nanmax(d, axis=(1,2)) - np.nanmin(d, axis=(1,2))
            np.testing.assert_allclose(total_ptp, expected_ptp)

    """

    def __init__(self, axis=0):
        super().__init__(axis=axis)
        self.batchnanmax = BatchNanMax(axis=axis)
        self.batchnanmin = BatchNanMin(axis=axis)

    def update_batch(self, batch):
        """
        Update the peak-to-peak with a new batch of data that can contain NaN values.

        Args:
            batch (numpy.ndarray): Input batch.

        Returns:
            BatchNanPeakToPeak: Updated BatchNanPeakToPeak object.

        """
        self.batchnanmax.update_batch(batch)
        self.batchnanmin.update_batch(batch)
        return self

    def __call__(self) -> np.ndarray:
        """
        Calculate the peak-to-peak.

        Returns:
            numpy.ndarray: Peak-to-peak of the batches.

        Raises:
            NoValidSamplesError: If no valid samples are available.

        """
        return self.batchnanmax() - self.batchnanmin()

    def __add__(self, other):
        self.batchnanmax.merge_test(other.batchnanmax, field="max")
        self.batchnanmin.merge_test(other.batchnanmin, field="min")
        if self.batchnanmin.min is None:
            return other
        elif other.batchnanmin.min is None:
            return self
        else:
            ret = BatchNanPeakToPeak(axis=self.axis)
            ret.batchnanmax = self.batchnanmax + other.batchnanmax
            ret.batchnanmin = self.batchnanmin + other.batchnanmin
            return ret
