import warnings

import numpy as np

from .._misc import NoValidSamplesError
from ..base import BatchNanStat


class BatchNanMin(BatchNanStat):
    """
    Class for calculating the minimum of batches of data that can contain NaN values.

    The algorithm keeps track of the element-wise minimum. When a new batch is added,
    the element-wise minimum between the current minimum and the new batch's minimum is computed, ignoring NaNs.

    .. code:: python

        import numpy as np
        from batchstats import BatchNanMin

        # create some data with NaNs
        data1 = np.array([[1, 2], [3, np.nan]])
        data2 = np.array([[5, 6], [np.nan, 8]])

        # create a BatchNanMin object
        bnm = BatchNanMin()

        # update with the first batch
        bnm.update_batch(data1)

        # update with the second batch
        bnm.update_batch(data2)

        # get the minimum
        total_min = bnm()

        # verify the result
        expected_min = np.array([1., 2.])
        np.testing.assert_allclose(total_min, expected_min)


    .. admonition:: Example with multiple axes and data > 2 dimensions

        .. code:: python

            # create some 3d data with NaNs
            data1 = np.arange(24).reshape(2, 3, 4).astype(float)
            data1[0, 1, 1] = np.nan
            data2 = np.arange(24, 48).reshape(2, 3, 4).astype(float)
            data2[1, 2, 0] = np.nan

            # create a BatchNanMin object to get the min over the last two axes
            bnm = BatchNanMin(axis=(1, 2))

            # update with the first batch
            bnm.update_batch(data1)

            # update with the second batch
            bnm.update_batch(data2)

            # get the min
            total_min = bnm()

            # verify the result
            d = np.concatenate((data1, data2))
            expected_min = np.nanmin(d, axis=(1,2))
            np.testing.assert_allclose(total_min, expected_min)

    """

    def __init__(self, axis=0):
        super().__init__(axis=axis)
        self.min = None

    def update_batch(self, batch):
        """
        Update the minimum with a new batch of data that can contain NaN values.

        Args:
            batch (numpy.ndarray): Input batch.

        Returns:
            BatchNanMin: Updated BatchNanMin object.

        """
        processed_batch = self._process_batch(batch)

        # get the number of valid samples along the given axis
        n_valid_samples = np.isfinite(processed_batch).sum(axis=self.axis)

        # only compute the batch_min if there are valid samples
        if np.any(n_valid_samples > 0):
            with warnings.catch_warnings():
                # suppress warning when all-nan slice is encountered
                warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
                batch_min = np.nanmin(processed_batch, axis=self.axis, keepdims=True)

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
                squeezed_min = min_.squeeze(axis=self.axis)
                # if all values along an axis were nan, the result for this axis is nan
                if squeezed_min.ndim > 0:
                    squeezed_min[self.n_samples == 0] = np.nan
                elif self.n_samples == 0:
                    return np.nan
                return squeezed_min
            return min_

    def __add__(self, other):
        self.merge_test(other, field="min")
        if self.min is None:
            return other
        elif other.min is None:
            return self
        else:
            ret = BatchNanMin(axis=self.axis)
            ret.n_samples = self.n_samples + other.n_samples
            ret.min = np.minimum(self.min, other.min)
            return ret

    def merge_test(self, other, field=None):
        if not isinstance(other, self.__class__):
            raise TypeError(f"Cannot merge {self.__class__} with {other.__class__}")
        if self.axis != other.axis:
            raise ValueError("Cannot merge objects with different axis values.")
