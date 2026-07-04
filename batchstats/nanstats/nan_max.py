import numpy as np

from .._misc import NoValidSamplesError
from ..base import BatchNanStat


class BatchNanMax(BatchNanStat):
    """
    Class for calculating the maximum of batches of data that can contain NaN values.

    The algorithm keeps track of the element-wise maximum. When a new batch is added,
    the element-wise maximum between the current maximum and the new batch's maximum is computed, ignoring NaNs.

    .. code:: python

        import numpy as np
        from batchstats import BatchNanMax

        # create some data with NaNs
        data1 = np.array([[1, 2], [3, np.nan]])
        data2 = np.array([[5, 6], [np.nan, 8]])

        # create a BatchNanMax object
        bnm = BatchNanMax()

        # update with the first batch
        bnm.update_batch(data1)

        # update with the second batch
        bnm.update_batch(data2)

        # get the maximum
        total_max = bnm()

        # verify the result
        expected_max = np.array([5., 8.])
        np.testing.assert_allclose(total_max, expected_max)


    .. admonition:: Example with multiple axes and data > 2 dimensions

        .. code:: python

            # create some 3d data with NaNs
            data1 = np.arange(24).reshape(2, 3, 4).astype(float)
            data1[0, 1, 1] = np.nan
            data2 = np.arange(24, 48).reshape(2, 3, 4).astype(float)
            data2[1, 2, 0] = np.nan

            # create a BatchNanMax object to get the max over the last two axes
            bnm = BatchNanMax(axis=(1, 2))

            # update with the first batch
            bnm.update_batch(data1)

            # update with the second batch
            bnm.update_batch(data2)

            # get the max
            total_max = bnm()

            # verify the result
            d = np.concatenate((data1, data2))
            expected_max = np.nanmax(d, axis=(1,2))
            np.testing.assert_allclose(total_max, expected_max)

    """

    def __init__(self, axis=0):
        super().__init__(axis=axis)
        self.max = None

    def update_batch(self, batch):
        """
        Update the maximum with a new batch of data that can contain NaN values.

        Args:
            batch (numpy.ndarray): Input batch.

        Returns:
            BatchNanMax: Updated BatchNanMax object.

        """
        processed_batch, n_valid_samples = self._process_batch(batch)
        self._update_extremum(processed_batch, n_valid_samples)
        return self

    def _update_extremum(self, batch, n_valid_samples):
        # only compute the batch_max if there are valid samples
        if np.any(n_valid_samples > 0):
            # np.fmax ignores NaNs: no internal copy of the batch (unlike np.nanmax),
            # no all-NaN warning, and all-NaN slices don't poison the running max
            batch_max = np.fmax.reduce(batch, axis=self.axis, keepdims=True)
            if self.max is None:
                self.max = batch_max
            else:
                self.max = np.fmax(self.max, batch_max)

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
                squeezed_max = max_.squeeze(axis=self.axis)
                # if all values along an axis were nan, the result for this axis is nan
                if squeezed_max.ndim > 0:
                    no_valid = self.n_samples == 0
                    if np.any(no_valid):
                        if not np.issubdtype(squeezed_max.dtype, np.inexact):
                            squeezed_max = squeezed_max.astype(float)
                        squeezed_max[no_valid] = np.nan
                elif self.n_samples == 0:
                    return np.nan
                return squeezed_max
            return max_

    def __add__(self, other):
        self.merge_test(other, field="max")
        if self.max is None:
            return other
        elif other.max is None:
            return self
        else:
            ret = BatchNanMax(axis=self.axis)
            ret.n_samples = self.n_samples + other.n_samples
            ret.max = np.maximum(self.max, other.max)
            return ret

    def merge_test(self, other, field=None):
        if not isinstance(other, self.__class__):
            raise TypeError(f"Cannot merge {self.__class__} with {other.__class__}")
        if self.axis != other.axis:
            raise ValueError("Cannot merge objects with different axis values.")
